from .DeepSIC_Block import DeepSIC_Block_double_proj, DeepSIC_Block_single_proj,DeepSIC_Block_No_proj
from src.Pulse.projection_fn import define_projection_matrix_and_bias,define_F_projection
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Detector.base_model import base_model_generator
from scipy.io import loadmat

class DeepSIC_proj():

    """Bayesian DeepSIC where each block is an independent Bayesian neural network.

        Args:
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antennas (int): Number of receive antennas.
            num_layers (int): Number of soft interference cancellation (SIC) layers.
            hidden_dim (int): Size of the hidden layer of each block.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.
            init_cov_scale (float, optional): Initial parameter covariance scale. Defaults to 0.1.
            Pulse (bool, optional): Whether to use projection matrices. Defaults to False.
            OU (bool, optional): Whether to use Ornstein-Uhlenbeck process for online training. Defaults to True.
            F (bool, optional): Whether to use state transition matrix F in online training. Defaults to False.
        """

    def __init__(
            self,
            symbol_bits: int,
            num_users: int,
            num_antennas: int,
            num_layers: int,
            hidden_dim: int,
            cov_type: str,
            init_cov_scale: float = 0.1,
            obs_cov_scale: float = 0.1,
            dlr_rank: int = 10,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'double_proj',
            learning_rate:float = 0.1,
            learning_proj :bool = False,
            learned_mat : bool = False,
            learned_params : bool = False,
            learned_A_path : str = "none",
            learned_phi_path: str = "none",
            learned_params_path: str = "none",
            param_wanted : int = 10,
            block_update: str ="single_block",
            pred_path: str = "none",
            F_style: str = "full"
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * (num_users) *2
        self.Pulse = Pulse
        self.cov_type = cov_type
        self.init_cov_scale = init_cov_scale
        self.obs_cov_scale = obs_cov_scale
        self.dlr_rank = dlr_rank
        self.learning_rate = learning_rate
        self.OU = OU
        self.F = F
        self.block_method = block_method.lower()
        self.proj_learn =learning_proj
        self.learned_mat = learned_mat
        self.learned_param = learned_params
        self.num_wanted = param_wanted
        self.block_update = block_update
        self.F_style = F_style


        """if self.learned_mat ==True and self.proj_learn == True:
            raise "cant learn matrices and used learned ones. problem in DeepSIC initialization"""

        if self.F == True and self.OU == True:
            raise "cant use OU and F together. problem in DeepSIC initialization"

        if self.F == True:
            if self.cov_type == "full":
                self.pred_type = "F"
            else:
                raise "cant use F prediction with diag or dlr covariance. problem in DeepSIC initialization"
        else:
            self.pred_type = "OU"

        if learned_mat == True:
            self.A_matrices,self.phi_vectors,self.params,self.F_matrices,self.Q_matrices,self.beta_vectors = self.upload_matrices(learned_A_path,learned_phi_path,learned_params_path,pred_path)
            self.num_wanted = self.A_matrices.shape[0]

        self.blocks = self.build_blocks()

        self.block_list = []
        self.block_sizes = []
        self.block_index = []  # (layer, user) לכל chunk

        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                b = self.blocks[layer_idx][user]
                self.block_list.append(b)
                self.block_sizes.append(b.z_layers.numel())
                self.block_index.append((layer_idx, user))

        self.cached_inputs = torch.ones((1, self.block_input_size), requires_grad=False)
        self.cached_next_input = torch.ones((1, self.block_input_size), requires_grad=False)
        self.cached_R = torch.eye(self.symbol_bits * 2) * 0.01



    def upload_matrices(self,A_path,phi_path,params_path,pred_path):
        A_data = loadmat(A_path)
        phi_data = loadmat(phi_path)

        A_np = A_data["A_matrices"]
        phi_np = phi_data["Phi_vectors"]

        if self.pred_type == "F":
            pred_data = loadmat(pred_path)
            F_np = torch.tensor(pred_data["F_matrices"], dtype=torch.float64)
            Q_np = torch.tensor(pred_data["Q_matrices"], dtype=torch.float64)
            b_np = pred_data["beta_vectors"].squeeze()
        else:
            F_np = None
            Q_np = None
            b_np = None

        if self.learned_param == True:
            param_data = loadmat(params_path)
            param_np = param_data["Params_matrices"]
            return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),torch.tensor(param_np,dtype=torch.float64)

        return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),None,F_np,Q_np,b_np

    def build_blocks(self):

        blocks = []
        base_model = base_model_generator(
            input_size=self.block_input_size,
            hidden_dim=self.hidden_dim,
            output_size=self.symbol_bits*2,
        )

        if self.block_method == 'double_proj':
            last_params = list(base_model.fc3.parameters())
            n_last = sum(p.numel() for p in last_params)
            base_params = list(base_model.fc1.parameters()) + list(base_model.fc2.parameters())
            n_hidden = sum(p.numel() for p in base_params)
            n_total = n_hidden+n_last
            num_wanted_hidden = 150
            num_wanted_last = 30
        else:
            layers_params = list(base_model.parameters())
            n_total = sum(p.numel() for p in layers_params)
            num_wanted_single = self.num_wanted

        for layer_idx in range(self.num_layers):
            layer_block = []
            for user in range(self.num_users):
                if self.block_method == 'double_proj':
                    # Define projection matrices and activation functions
                    projection_mat_hidden, phi_hidden = define_projection_matrix_and_bias(n_hidden,num_wanted_hidden,learnable=self.proj_learn)
                    projection_mat_last, phi_last = define_projection_matrix_and_bias(n_last,num_wanted_last,learnable=self.proj_learn)

                    block = DeepSIC_Block_double_proj(
                        base_model=base_model,
                        symbol_bits=self.symbol_bits,
                        num_users=self.num_users,
                        num_antenas=self.num_antennas,
                        hidden_dim=self.hidden_dim,
                        projection_mat_hidden=projection_mat_hidden,
                        projection_mat_last=projection_mat_last,
                        phi_hidden=phi_hidden,
                        phi_last=phi_last,
                        cov_type= self.cov_type,
                        init_cov_scale=self.init_cov_scale,
                        dlr_rank=self.dlr_rank
                    )
                    layer_block.append(block)

                elif self.block_method == 'single_proj':
                    F_blck, Q_block, beta_block = None, None, None
                    # Define projection matrices
                    if self.learned_mat == False:
                        projection_mat_hidden, phi_hidden = define_projection_matrix_and_bias(n_total, num_wanted_single,learnable=self.proj_learn)
                        if self.pred_type == "F":
                            F_blck, Q_block, beta_block = define_F_projection(num_wanted_single, F_style=getattr(self, "F_style", "full"))
                    else:
                        idx = layer_idx*self.num_users+user
                        projection_mat_hidden = self.A_matrices[:,:,idx]
                        phi_hidden = self.phi_vectors[:,idx]
                        if self.F_matrices is not None:
                            if getattr(self, "F_style", "full") == "diag":
                                F_blck = self.F_matrices[:, idx]
                            else:
                                F_blck = self.F_matrices[:, :, idx]
                            Q_block = self.Q_matrices[:, :, idx]
                            beta_block = self.beta_vectors[:,idx]



                    block = DeepSIC_Block_single_proj(
                        base_model=base_model,
                        symbol_bits=self.symbol_bits,
                        num_users=self.num_users,
                        num_antenas=self.num_antennas,
                        hidden_dim=self.hidden_dim,
                        projection_mat_hidden=projection_mat_hidden,
                        phi_hidden=phi_hidden,
                        cov_type=self.cov_type,
                        Rank=self.dlr_rank,
                        pred_type = self.pred_type,
                        F = F_blck,
                        Q = Q_block,
                        beta = beta_block,
                        init_cov_scale=self.init_cov_scale,
                        dlr_rank=self.dlr_rank
                    )
                    layer_block.append(block)
                else:
                    raise "problem with config: projection mode "

            blocks.append(layer_block)

        if self.learned_param == True:
            for layer_idx in range(self.num_layers):
                for user in range(self.num_users):
                    block = blocks [layer_idx][user]
                    block.z_layers = nn.Parameter(self.params[:,layer_idx*self.num_users+user])


        return blocks


    def soft_decode_batch (self,rx):
        concat_prediction = None
        flag = True
        for train_rx in rx:
            prediction = self.soft_decode(train_rx,flag)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction,prediction],dim = 0)
            flag = False
        return concat_prediction


    def soft_decode(self, rx,flag):
        """Soft decode the received signal using the DeepSIC architecture."""
        # generate inputs for first block
        inputs = self.cached_inputs
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = self.cached_next_input
        next_input.copy_(inputs)
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs.copy_(next_input)
                block = self.blocks[layer_idx][user]
                #generate inputs
                block_input,start,end = self.generate_input(user,self.rx_size,inputs)
                with torch.no_grad():
                    next_input[0,self.rx_size+start:self.rx_size+end] = block.forward_soft(inputs,flag)

            inputs.copy_(next_input)
        return next_input[:, self.rx_size:]




    def train_batch (self,train_fn,rx,symbols ,method ,gamma = 0.999):
        for train_rx , labels in zip(rx, symbols):
            self.online_training(train_fn,train_rx,labels,method,gamma)


    def online_training(self,train_fn,rx,symbols,method,gamma):
        """Perform online training of the DeepSIC model each block independently."""
        ## symbols : list of true symbols for each user (2d tensor)

        #generate inputs for first block
        if self.block_update == "single_block":
            inputs = self.cached_inputs
            inputs[0, :self.rx_size] = rx
            inputs[0, self.rx_size:] = 0.5
            next_input = self.cached_next_input
            next_input.copy_(inputs)
            R = self.cached_R
            # Iterate over layers
            if method.lower()=='bong':
                if self.cov_type != 'dlr':
                    for layer_idx in range(self.num_layers):
                        for user in range(self.num_users):
                            block = self.blocks[layer_idx][user]
                            # generate input
                            start = user * self.symbol_bits * 2
                            end = start + self.symbol_bits * 2

                            #train layers
                            if self.pred_type == "OU":
                                mean_upd, cov_upd = train_fn(block.z_layers,block.cov_layers,block , inputs , symbols[user] ,
                                         R,gamma,
                                         block.initial_cov_layers,
                                         block.initial_mean_layers,
                                         self.learning_rate)
                            elif self.pred_type == "F":
                                mean_upd, cov_upd = train_fn(block.z_layers,block.cov_layers,block , inputs , symbols[user] ,
                                         R,
                                         F=block.F,
                                         beta = block.beta,
                                         Q=block.Q)
                            else:
                                raise "problem with config: prediction type in DeepSIC training"
                            with torch.no_grad():
                                block.z_layers.copy_(mean_upd)
                                block.cov_layers.copy_(cov_upd)
                            #train last layer
                            """if self.block_method == 'double_proj':
                                mean_upd, cov_upd = train_fn(block.z_last, block.cov_last, block, inputs, symbols[user],
                                         R_last,gamma,
                                         block.initial_cov_last,
                                         block.initial_mean_last,
                                         self.learning_rate)
                                with torch.no_grad():
                                    block.z_last.copy_(mean_upd)
                                    block.cov_last.copy_(cov_upd)
                            next_input[0, self.rx_size+start:self.rx_size+end] = block.forward(inputs)"""
                        #print(next_input)
                        inputs.copy_(next_input)

                else:
                    for layer_idx in range(self.num_layers):
                        for user in range(self.num_users):
                            block = self.blocks[layer_idx][user]
                            start = user * self.symbol_bits * 2
                            end = start + self.symbol_bits * 2
                            mean_upd, prec_diag_upd, prec_low_rank_upd =train_fn(block.z_layers,block.diag_layers,block.lr_cov_layers ,block , inputs ,
                                     symbols[user] ,
                                     R,0.999,
                                     block.initial_diag_layers,
                                     block.initial_lr_cov_layers,
                                     block.initial_mean_layers,
                                     self.learning_rate)
                            with torch.no_grad():
                                block.z_layers.copy_(mean_upd)
                                block.diag_layers.copy_(prec_diag_upd)
                                block.lr_cov_layers.copy_(prec_low_rank_upd)

                            """if self.block_method == 'double_proj':
                                mean_upd, prec_diag_upd, prec_low_rank_upd = train_fn(block.z_last, block.diag_last, block.lr_cov_last, block, inputs,
                                         symbols[user],
                                         R_last,0.999,
                                         block.initial_diag_last,
                                         block.initial_lr_cov_last,
                                         block.initial_mean_last,
                                         self.learning_rate)
                                with torch.no_grad():
                                    block.z_last.copy_(mean_upd)
                                    block.diag_last.copy_(prec_diag_upd)
                                    block.lr_cov_last.copy_(prec_low_rank_upd)
                                #print(a)
                                next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)"""


                        #print(next_input)
                        inputs.copy_(next_input)
            """elif method.lower() == 'sgd':
                for layer in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer][user]
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                        # Define loss function
                        train_fn(block, inputs, symbols, self.learning_rate)
    
                        next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)
                    inputs = next_input"""
        elif self.block_update == "entire":
            size = self.symbol_bits * 2
            R = torch.eye(size) * self.obs_cov_scale
            train_fn(self,rx,symbols,R)
            return



    def generate_input(self,user,rx_size,inputs):
        rx_part = inputs[:, :rx_size]
        soft_bits = inputs[:, rx_size:]

        start = user * self.symbol_bits*2
        end = start + self.symbol_bits*2
        other_users = torch.cat([soft_bits[:, :start], soft_bits[:, end:]], dim=-1)
        block_input = torch.cat([rx_part, other_users], dim=-1)
        return block_input,start,end


    def forward_offline(self,layer,user,train_1,label_1,train_2,label_2,train_fn):
        """
        Function to menege offline learning process for single block
        ARGS:
        layer (int) : layer index
        user (int) : user  index
        train_1 (torch.tensor ) : recieved signals for online update
        label_1 (torch.tensor) : true label for online update
        train_2 (torch.tensor) : recieved singnals for offline prediction

        Return:
            prediction (torch.tensor) : prediction for the block after process
        """
        block = self.blocks[layer][user]
        prediction = []
        true_label = []
        size = self.symbol_bits*2
        R = torch.eye(size) * self.obs_cov_scale
        for rx,label in zip(train_1,label_1):
            if layer == 0:
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = rx
                inputs[0, self.rx_size:] = 0.5
            else:
                inputs = self.process_throw_layers(rx,layer)

            """block_input, start, end = self.generate_input(user, self.rx_size, inputs)"""

            if self.cov_type == 'diag' or self.cov_type == 'full':
                if self.pred_type == "OU":
                    mean_upd, cov_upd = train_fn(block.z_layers, block.cov_layers, block, inputs, label[user],
                                                 R, 0.999,
                                                 block.initial_cov_layers,
                                                 block.initial_mean_layers,
                                                 self.learning_rate)
                elif self.pred_type == "F":
                    mean_upd, cov_upd = train_fn(block.z_layers, block.cov_layers, block, inputs, label[user],
                                                 R,
                                                 F=block.F,
                                                 beta=block.beta,
                                                 Q=block.Q)
                else:
                    raise "problem with config: prediction type in DeepSIC training"

                block.z_layers=mean_upd
                block.cov_layers=cov_upd

            else:
                mean_upd, prec_diag_upd, prec_low_rank_upd = train_fn(block.z_layers, block.diag_layers,
                                                                      block.lr_cov_layers, block, inputs,
                                                                      label[user],
                                                                      R, 0.999,
                                                                      block.initial_diag_layers,
                                                                      block.initial_lr_cov_layers,
                                                                      block.initial_mean_layers,
                                                                      self.learning_rate)

                block.z_layers=mean_upd
                block.diag_layers=prec_diag_upd
                block.lr_cov_layers=prec_low_rank_upd
            for i in range(10):
                rx_to_pred = next(train_2)
                y_true = next(label_2)[user]
                if layer == 0:
                    inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                    inputs[0, :self.rx_size] = rx_to_pred
                    inputs[0, self.rx_size:] = 0.5
                else:
                    inputs = self.process_throw_layers(rx_to_pred, layer)

                logits = self.blocks[layer][user].forward_logits(inputs)
                prediction.append(logits)

                true_label.append(y_true)
        prediction = torch.stack(prediction,dim =0)
        true_label = torch.stack(true_label,dim =0)

        return prediction,true_label


    def pack_latents(self):
        return torch.cat([b.z_layers.reshape(-1) for b in self.block_list], dim=0)


    def forward_bong_entire(self, rx, z_big, method='layers', return_intermediate=True):
        """
            Functional DeepSIC forward for BONG: uses external latent vector z_big.
            rx: Tensor shape (rx_size,) or (1, rx_size)
            z_big: 1D tensor concatenating all blocks' latents
            return_intermediate: if True, returns also per-(layer,user) outputs
            """
        if rx.dim() == 1:
            rx_in = rx.unsqueeze(0)
        else:
            rx_in = rx

        z_chunks = torch.split(z_big, self.block_sizes)

        inputs = torch.ones((1, self.block_input_size), device=rx_in.device, dtype=rx_in.dtype)
        inputs[0, :self.rx_size] = rx_in[0]
        inputs[0, self.rx_size:] = 0.5

        next_input = inputs.clone()

        inter = []

        z_idx = 0
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs = next_input

                block = self.blocks[layer_idx][user]
                z_block = z_chunks[z_idx]

                block_input, start, end = self.generate_input(user, self.rx_size, inputs)

                out = block.forward_bong(inputs, z_block, method=method)

                new_soft = inputs[:, self.rx_size:].clone()
                new_soft[0, start:end] = out
                next_input = torch.cat([inputs[:, :self.rx_size], new_soft], dim=1)

                if return_intermediate:
                    inter.append((layer_idx, user, out))

                z_idx += 1

            inputs = next_input
        y_pred = inputs[:, self.rx_size:]


        self.y_pred_entire = y_pred
        self.y_pred_inter = inter
        return y_pred,inter




    def process_throw_layers(self,train_rx,layer):
        """
        Args:
        train_rx (torch.tensor) : received vector
        layer (int) : layer number to proces throw

        Returns:
            prediction of the first layers
        """
        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size),requires_grad=False)
        inputs[0, :self.rx_size] = train_rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone()
        for layer_idx in range(layer):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs = next_input
                block = self.blocks[layer_idx][user]
                # generate inputs
                block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                with torch.no_grad():
                    next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)
        inputs = next_input
        return inputs



    def update_initial_params(self):
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                block = self.blocks[layer_idx][user]
                block.initial_mean_layers = block.z_layers.clone().detach()

class DeepSIC():
    """Bayesian DeepSIC where each block is an independent Bayesian neural network.

            Args:
                symbol_bits (int): Number of bits per symbol.
                num_users (int): Number of users.
                num_antennas (int): Number of receive antennas.
                num_layers (int): Number of soft interference cancellation (SIC) layers.
                hidden_dim (int): Size of the hidden layer of each block.
                cov_type (CovarianceType, optional): Type of covariance for the parameters.
                init_cov_scale (float, optional): Initial parameter covariance scale. Defaults to 0.1.
                Pulse (bool, optional): Whether to use projection matrices. Defaults to False.
                OU (bool, optional): Whether to use Ornstein-Uhlenbeck process for online training. Defaults to True.
                F (bool, optional): Whether to use state transition matrix F in online training. Defaults to False.
            """


    def __init__(
            self,
            symbol_bits: int,
            num_users: int,
            num_antennas: int,
            num_layers: int,
            hidden_dim: int,
            cov_type: str,
            init_cov_scale: float = 0.1,
            obs_cov_scale: float = 0.1,
            dlr_rank: int = 10,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'double_proj',
            learning_rate: float = 0.1
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * (num_users)*2
        self.cov_type = cov_type
        self.learning_rate = learning_rate
        self.init_cov_scale = init_cov_scale
        self.obs_cov_scale = obs_cov_scale
        self.dlr_rank = dlr_rank

        self.OU = OU
        self.F = F

        self.blocks = self.build_blocks()


    def build_blocks(self):
        blocks = []
        base_model = base_model_generator(
            input_size=self.block_input_size,
            hidden_dim=self.hidden_dim,
            output_size=self.symbol_bits*2,
        )
        for layer_idx in range(self.num_layers):
            layer_block = []
            for user in range(self.num_users):
                block = DeepSIC_Block_No_proj(
                    base_model=base_model,
                    symbol_bits=self.symbol_bits,
                    num_users=self.num_users,
                    num_antenas=self.num_antennas,
                    hidden_dim=self.hidden_dim,
                    cov_type= self.cov_type,
                    init_cov_scale=self.init_cov_scale,
                    dlr_rank=self.dlr_rank
                )
                layer_block.append(block)

            blocks.append(layer_block)
        return blocks


    def soft_decode_batch(self, rx):
        concat_prediction = None
        for train_rx in rx:
            prediction = self.soft_decode(train_rx)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction, prediction], dim=0)

        return concat_prediction

    def soft_decode(self, rx):
        """Soft decode the received signal using the DeepSIC architecture."""
        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size ))
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone()
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs = next_input
                block = self.blocks[layer_idx][user]
                # generate inputs
                block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                with torch.no_grad():
                    next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)

            inputs = next_input
        return inputs[:, self.rx_size:]


    def train_batch(self, train_fn, rx, symbols, method ,gamma=0.999):
        for train_rx, labels in zip(rx, symbols):
            self.online_training(train_fn, train_rx, labels, method, gamma)

    def online_training(self, train_fn, rx, symbols,method,gamma):
        """Perform online training of the DeepSIC model each block independently."""
        ## symbols : list of true symbols for each user (2d tensor)

        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size), requires_grad=False)
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone().detach()
        size = self.symbol_bits*2
        R = torch.eye(size) * self.obs_cov_scale
        size = self.symbol_bits
        R_last = torch.eye(size) * self.obs_cov_scale
        # Iterate over layers
        if method.lower() == 'bong':
            if self.cov_type != 'dlr':
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        # generate input
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                        # train layers
                        mean_upd, cov_upd = train_fn(block.z_layers, block.cov_layers, block, inputs,
                                                     symbols[user],
                                                     R, gamma,
                                                     block.initial_cov_layers,
                                                     block.initial_mean_layers,
                                                     self.learning_rate)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.cov_layers.copy_(cov_upd)
                        # train last layer
                        next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)
                    # print(next_input)
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                        mean_upd, prec_diag_upd, prec_low_rank_upd = train_fn(block.z_layers, block.diag_layers,
                                                                              block.lr_cov_layers, block,
                                                                              inputs,
                                                                              symbols[user],
                                                                              R, gamma,
                                                                              block.initial_diag_layers,
                                                                              block.initial_lr_cov_layers,
                                                                              block.initial_mean_layers,
                                                                              self.learning_rate)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.diag_layers.copy_(prec_diag_upd)
                            block.lr_cov_layers.copy_(prec_low_rank_upd)


                            next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)

                    # print(next_input)
                    inputs = next_input

        elif method.lower() == 'sgd':
            for layer in range(self.num_layers):
                for user in range(self.num_users):
                    block = self.blocks[layer][user]
                    block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                    # Define loss function
                    train_fn(block, inputs, symbols[user], lr=self.learning_rate,num_steps=10)
                    with torch.no_grad():
                        next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(inputs)
                inputs = next_input


    def generate_input(self, user, rx_size, inputs):
        rx_part = inputs[:, :rx_size]
        soft_bits = inputs[:, rx_size:]

        start = user * self.symbol_bits*2
        end = start + self.symbol_bits*2
        other_users = torch.cat([soft_bits[:, :start], soft_bits[:, end:]], dim=-1)
        block_input = torch.cat([rx_part, other_users], dim=-1)
        return block_input, start, end

    def update_initial_params(self):
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                block = self.blocks[layer_idx][user]
                block.initial_mean_layers = block.z_layers.clone().detach()
                if self.cov_type != 'dlr':
                    block.initial_cov_layers = block.cov_layers.clone().detach()
                else:
                    block.initial_diag_layers = block.diag_layers.clone()
                    block.initial_lr_cov_layers = block.lr_cov_layers.clone()