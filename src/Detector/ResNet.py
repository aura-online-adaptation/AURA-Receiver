from importlib.util import set_package

from torch import while_loop

from .DeepSIC_Block import DeepSIC_Block_double_proj, DeepSIC_Block_single_proj,DeepSIC_Block_No_proj
from src.Pulse.projection_fn import define_projection_matrix_and_bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Detector.base_model import base_model_generator_ResNet
from scipy.io import loadmat
from src.Pulse.projection_fn import define_projection_matrix_and_bias, define_F_projection
from src.Pulse.DNN_projection import DNNProjection

class ResNet_proj(nn.Module):
    """ResNet detector where each block is an independent DNN neural network that presict all blocks.

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
            num_layers: int ,
            hidden_dim: int,
            cov_type: str,
            init_cov_scale: float = 0.1,
            obs_cov_scale: float = 0.1,
            dlr_rank: int = 10,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'single_proj',
            learning_rate:float = 0.1,
            learning_proj :bool = False,
            learned_mat : bool = False,
            learned_params : bool = False,
            learned_A_path : str = "none",
            learned_phi_path: str = "none",
            learned_params_path: str = "none",
            pred_type : str = 'OU',
            proj_type : str = 'dnn',
            learned_DNN_path : str = "none",
            pred_path : str = "none",
            update_type : str = 'each block',
            F_style : str = "full"
    ):
        super().__init__()
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = int(num_layers//2)
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size
        self.block_output_size = symbol_bits * (num_users) *2
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
        self.proj_type = proj_type.lower()
        self.pred_path = pred_path
        self.update_type = update_type.lower()
        self.F_style = F_style

        self.pred_type = 'OU' if self.OU else 'F' if self.F else None
        if self.OU == True and self.F == True:
            raise "cant have both OU and F true in ResNet initialization"


        self.base_model=base_model_generator_ResNet(self.block_input_size,12,self.block_output_size,self.num_layers)
        if self.learned_mat ==True and self.proj_learn == True:
            raise "cant learn matrices and used learned ones. problem in ResNet initialization"

        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)

        if self.update_type == "each_block":
            self.BONG_sizes = ([sum(self.sizes[:2])] +
                    [sum(self.sizes[i:i + 4]) for i in range(2, len(self.sizes) - 2, 4)] +
                    [sum(self.sizes[-2:])])
            Proj_sizes = [100] * (self.num_layers + 2)
        elif self.update_type == "entire":
            self.BONG_sizes = [self.total_params]
            Proj_sizes = [300]
        self.generate_projection(learned_mat,learned_path=learned_DNN_path,Proj_sizes=Proj_sizes,num_wanted=30)

        self.norms = nn.ModuleList([nn.LayerNorm(12) for _ in range(self.num_layers)])

        self.compile = None
        self.generate_cov_matrix(cov_type, Rank=self.dlr_rank)

        if self.pred_type == 'OU':
            self.generate_initial_cov_matrix(self.cov_type)
            self.initial_mean_layers_arr = [z.clone() for z in self.z_layers_arr]

        if self.pred_type == 'F':
            self.F_data, self.Q, self.beta = self.generate_F_projection(self.pred_path)

    def generate_projection(self, learned_mat, learned_path=None, Proj_sizes=None, num_wanted=10):
        if learned_mat:
            self.load_projection_checkpoint(learned_path)
            self.Proj_sizes = []
            self.proj_models = nn.ModuleList()
            for state_dict in self.loaded_state_dicts:
                w1 = state_dict["layer1.weight"]
                w2 = state_dict["layer2.weight"]
                d = w1.shape[1]
                out_size = w2.shape[0]
                #out_size = w1.shape[0]
                self.Proj_sizes.append(d)
                model = DNNProjection(input_size=d, hidden_size=d*10, output_size=out_size)
                model.load_state_dict(state_dict)
                self.proj_models.append(model)
                self.num_wanted = d
        else:
            assert Proj_sizes is not None
            self.Proj_sizes = Proj_sizes
            self.num_wanted = Proj_sizes[0]
            self.proj_models = nn.ModuleList(
                [DNNProjection(input_size=d, hidden_size=d * 10, output_size=size) for d, size in
                 zip(self.Proj_sizes, self.BONG_sizes)])
        self.z_layers, self.z_layers_arr = self.generate_mean_layers()

        assert sum(self.Proj_sizes) == self.z_layers.numel();
        assert len(self.Proj_sizes) == len(self.proj_models)
        return

    def generate_F_projection(self,pred_path):
        if self.learned_mat == True:
            pred_data =loadmat(pred_path)
            if getattr(self, "F_style", "full") == "diag":
                F_data = torch.tensor(pred_data["F_matrices"], dtype=torch.float32).permute(1, 0)
                Q = torch.tensor(pred_data["Q_matrices"], dtype=torch.float32).permute(1, 0)
            else:
                F_data = torch.tensor(pred_data["F_matrices"], dtype=torch.float32).permute(2, 0, 1)
                Q = torch.tensor(pred_data["Q_matrices"], dtype=torch.float32).permute(2, 0, 1)
            beta = torch.tensor(pred_data["beta_vectors"], dtype=torch.float32).permute(1, 0)
            if self.update_type == "each_block":
                beta = beta.permute(1, 0)
        else:
            F_data = nn.ParameterList()
            Q = nn.ParameterList()
            beta = nn.ParameterList()
            for z_block in self.z_layers_arr:
                num_wanted_param = z_block.numel()
                F_i, Q_i, beta_i = define_F_projection(num_wanted_param, F_style=getattr(self, "F_style", "full"))
                F_data.append(nn.Parameter(F_i))
                Q.append(nn.Parameter(Q_i))
                beta.append(nn.Parameter(beta_i))

        return F_data,Q,beta





    def load_projection_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu")

        self.loaded_proj_type = checkpoint["proj_type"]
        self.loaded_BONG_sizes = checkpoint["BONG_sizes"]
        self.loaded_Proj_sizes = checkpoint["Proj_sizes"]
        self.loaded_state_dicts = checkpoint["state_dicts"]

        return



    def generate_mean_layers(self):
        self.z_layers = []
        self.z_layers_arr = []
        z_blocks = [0.01 * torch.randn(d) for d in self.Proj_sizes]
        return torch.cat(z_blocks, dim=0) ,z_blocks


    def generate_cov_matrix(self, cov_type, Rank=None):
        self.cov_layers_arr = []

        for d in self.Proj_sizes:
            if cov_type == 'full':
                P = self.init_cov_scale * torch.eye(d)
            elif cov_type == 'diag':
                P = self.init_cov_scale * torch.ones(d)
            elif cov_type == 'dlr':
                diag = self.init_cov_scale * torch.ones(d)
                lr = self.init_cov_scale * torch.randn(d, Rank)
                P = (diag, lr)
            else:
                raise ValueError(cov_type)

            self.cov_layers_arr.append(P)

        return self.cov_layers_arr

    def generate_initial_cov_matrix(self, cov_type):
        if cov_type in ['full', 'diag']:
            self.initial_cov_layers_arr = [P.clone() for P in self.cov_layers_arr]
        else:
            self.initial_diag_layers_arr = [d.clone() for d in self.diag_layers_arr]
            self.initial_lr_cov_layers_arr = [lr.clone() for lr in self.lr_cov_layers_arr]
        return


    def upload_matrices(self,A_path,phi_path,params_path):
        A_data = loadmat(A_path)
        phi_data = loadmat(phi_path)

        print("A keys:", A_data.keys())
        print("phi keys:", phi_data.keys())

        A_np = A_data["A_matrices"]
        phi_np = phi_data["Phi_vectors"]
        if self.learned_param == True:
            param_data = loadmat(params_path)
            param_np = param_data["Params_matrices"]
            return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),torch.tensor(param_np,dtype=torch.float64)

        return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),None


    def expend(self):
        """Expand parameters from vector to model parameters"""
        if self.proj_type == 'affine':
            raise "NO affine projection in ResNet"
        elif self.proj_type == 'dnn':
            theta_blocks = [proj(z) for proj, z in zip(self.proj_models, self.z_layers_arr)]
            return torch.cat(theta_blocks, dim=0)
        else:
            raise "Projection type not recognized in expend function"

    def forward(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size

        W_in, b_in = params[0], params[1]
        x = F.relu(F.linear(x, W_in, b_in))
        index = 2
        for i in range(self.num_layers):
            """x_norm = self.norms[i](x)"""
            x_norm = x
            residual = x
            out = F.leaky_relu(F.linear(x_norm, params[index], params[index + 1]),0.1)
            out = F.linear(out, params[index + 2], params[index + 3])
            x = F.leaky_relu(out + residual,0.1)
            index += 4

        W_out, b_out = params[-2], params[-1]
        x = F.linear(x, W_out, b_out)

        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x


    def forward_logits(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size

        W_in, b_in = params[0], params[1]
        x = F.relu(F.linear(x, W_in, b_in))
        index = 2
        for i in range(self.num_layers):
            """x_norm = self.norms[i](x)"""
            x_norm = x
            residual = x
            out = F.leaky_relu(F.linear(x_norm, params[index], params[index + 1]), 0.1)
            out = F.linear(out, params[index + 2], params[index + 3])
            x = F.leaky_relu(out + residual, 0.1)
            index += 4

        W_out, b_out = params[-2], params[-1]
        x = F.linear(x, W_out, b_out)

        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = x.view(-1)
        return x

    def forward_bong(self, x,z,method='layers'):
        theta = self.expend_bong(z,method)
        # Unravel theta into model parameters
        """offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size"""
        chunks = torch.split(theta, self.sizes)
        params = [chunk.view(shape) for chunk, shape in zip(chunks, self.shapes)]
        W_in, b_in = params[0], params[1]
        x = F.relu(F.linear(x, W_in, b_in))
        index=2
        for i in range(self.num_layers):
            x_norm = self.norms[i](x)
            residual = x
            out = F.leaky_relu(F.linear(x_norm, params[index], params[index + 1]), 0.1)
            out = F.linear(out, params[index + 2], params[index + 3])
            x = F.leaky_relu(out + residual, 0.1)
            index += 4


        W_out, b_out = params[-2], params[-1]
        x = F.linear(x, W_out, b_out)

        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x


    def expend_bong(self,z,method='layers'):
        """Expand parameters from vector to model parameters"""
        if self.proj_type == 'affine':
            raise "NO affine projection in ResNet"

        elif self.proj_type == 'dnn':
            theta = torch.cat([proj(zb) for proj, zb in zip(self.proj_models, torch.split(z, self.Proj_sizes))], dim=0)
        else:
            raise "Projection type not recognized in expend function"
        return theta

    def train_batch(self, train_fn, rx, symbols,method, gamma=0.999):
        if method.lower() == 'bong':
            size = self.symbol_bits * self.num_users*2
            R = torch.eye(size)*self.obs_cov_scale
            for train_rx, labels in zip(rx, symbols):
                labels = labels.flatten()
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = train_rx
                #inputs[0, self.rx_size:] = 0.5
                if self.OU == True:
                    mean_upd, mean_upd_arr , cov_upd = train_fn(self.z_layers , self.z_layers_arr, self.cov_layers_arr, self, inputs, labels,
                                                 R, gamma,
                                                 self.initial_cov_layers_arr,
                                                 self.initial_mean_layers_arr,
                                                 self.Proj_sizes)
                    with torch.no_grad():
                        self.z_layers.copy_(mean_upd)
                        self.z_layers_arr = mean_upd_arr
                        self.cov_layers_arr= cov_upd

                if self.F == True:
                    mean_upd, mean_upd_arr , cov_upd = train_fn(self.z_layers , self.z_layers_arr, self.cov_layers_arr, self, inputs, labels,
                                                 R,self.F_data,self.Q,self.beta,self.Proj_sizes)
                    with torch.no_grad():
                        self.z_layers.copy_(mean_upd)
                        self.z_layers_arr = mean_upd_arr
                        self.cov_layers_arr= cov_upd

        elif method.lower() == 'sgd':
            for train_rx, labels in zip(rx, symbols):
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = train_rx
                inputs[0, self.rx_size:] = 0.5
                train_fn(self, inputs, labels, lr=self.learning_rate, num_steps=1)



    def soft_decode_batch(self,rx):
        concat_prediction = None
        for train_rx in rx:
            inputs = torch.ones((1, self.block_input_size))
            inputs[0, :self.rx_size] = train_rx
            #inputs[0, self.rx_size:] = 0.5
            prediction = self.forward(inputs)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction, prediction], dim=0)

        return concat_prediction

    def forward_offline(self,train_1,label_1,train_2,label_2,train_fn):
        if self.z_layers is not None:
            self.z_layers = self.z_layers.detach()

        if self.cov_layers_arr is not None and self.z_layers_arr is not None:
            self.z_layers_arr = [z.detach().clone() for z in self.z_layers_arr]
            self.cov_layers_arr = [c.detach().clone() for c in self.cov_layers_arr]

        prediction = []
        true_label = []
        size = self.symbol_bits * self.num_users*2
        R = torch.eye(size) * self.obs_cov_scale
        for train_rx, labels in zip(train_1, label_1):

            inputs = torch.ones((1, self.block_input_size), requires_grad=False)
            inputs[0, :self.rx_size] = train_rx
            #inputs[0, self.rx_size:] = 0.5
            labels = labels.flatten()
            if self.cov_type !='dlr':
                if self.OU == True:
                    mean_upd, mean_upd_arr , cov_upd = train_fn(self.z_layers , self.z_layers_arr, self.cov_layers_arr, self, inputs, labels,
                                                 R, 0.999,
                                                 self.initial_cov_layers_arr,
                                                 self.initial_mean_layers_arr,
                                                 self.Proj_sizes)

                elif self.F == True:
                    mean_upd, mean_upd_arr , cov_upd = train_fn(self.z_layers , self.z_layers_arr, self.cov_layers_arr, self, inputs, labels,
                                                 R,self.F_data,self.Q,self.beta,self.Proj_sizes)

                self.z_layers = mean_upd
                self.z_layers_arr = mean_upd_arr
                self.cov_layers_arr = cov_upd

            else :
                raise "No DLR in ResNet"
            for i in range(5):
                rx_to_pred = next(train_2)
                y_true = next(label_2)
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = rx_to_pred
                #inputs[0, self.rx_size:] = 0.5

                logits = self.forward_logits(inputs)
                prediction.append(logits)
                y_true = y_true.flatten()
                true_label.append(y_true)

        prediction = torch.stack(prediction, dim=0)
        true_label = torch.stack(true_label, dim=0)

        return prediction, true_label







class ResNet_no_proj(nn.Module):
    """ResNet detector where each block is an independent DNN neural network that presict all blocks.

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
            num_layers: int ,
            hidden_dim: int,
            cov_type: str,
            init_cov_scale: float = 0.1,
            obs_cov_scale: float = 0.1,
            dlr_rank: int = 10,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'single_proj',
            learning_rate:float = 0.1,
            learning_proj :bool = False,
            learned_mat : bool = False,
            learned_params : bool = False,
            learned_A_path : str = "none",
            learned_phi_path: str = "none",
            learned_params_path: str = "none",
            pred_type : str = 'OU',
    ):
        super().__init__()
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = int(num_layers)
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size
        self.block_output_size = symbol_bits * (num_users) *2
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

        self.base_model=base_model_generator_ResNet(self.block_input_size,12,self.block_output_size,self.num_layers)
        if self.learned_mat ==True and self.proj_learn == True:
            raise "cant learn matrices and used learned ones. problem in ResNet initialization"

        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)

        self.norms = nn.ModuleList([nn.LayerNorm(12) for _ in range(self.num_layers)])





        self.z_layers = nn.Parameter(0.01*torch.randn(self.total_params))

        self.generate_cov_matrix(cov_type, Rank=self.dlr_rank)

        if pred_type == 'OU':
            self.generate_initial_cov_matrix(self.cov_type)
            self.initial_mean_layers = self.z_layers.clone()


    def generate_cov_matrix(self,cov_type,Rank):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'full':
            self.cov_layers = torch.eye(self.total_params)*self.init_cov_scale

        elif cov_type == 'diag':
            self.cov_layers = torch.ones(self.total_params)*self.init_cov_scale

        else:
            self.diag_layers = torch.ones(self.total_params)*self.init_cov_scale
            self.lr_cov_layers = torch.randn(self.total_params,Rank)*self.init_cov_scale
        return


    def generate_initial_cov_matrix(self,cov_type):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'full':
            self.initial_cov_layers = self.cov_layers.clone()
        elif cov_type == 'diag':
            self.initial_cov_layers= self.cov_layers.clone()
        else:
            self.initial_diag_layers = self.diag_layers.clone()
            self.initial_lr_cov_layers = self.lr_cov_layers.clone()

        return

    def upload_matrices(self,A_path,phi_path,params_path):
        A_data = loadmat(A_path)
        phi_data = loadmat(phi_path)

        print("A keys:", A_data.keys())
        print("phi keys:", phi_data.keys())

        A_np = A_data["A_matrices"]
        phi_np = phi_data["Phi_vectors"]
        if self.learned_param == True:
            param_data = loadmat(params_path)
            param_np = param_data["Params_matrices"]
            return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),torch.tensor(param_np,dtype=torch.float64)

        return torch.tensor(A_np, dtype=torch.float64),torch.tensor(phi_np, dtype=torch.float64),None




    def forward(self, x):
        theta = self.z_layers
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size

        W_in, b_in = params[0], params[1]
        x = F.relu(F.linear(x, W_in, b_in))
        index = 2
        for i in range(self.num_layers):
            x_norm = self.norms[i](x)
            residual = x
            out = F.gelu(F.linear(x_norm, params[index], params[index + 1]))
            out = F.linear(out, params[index + 2], params[index + 3])
            x = F.gelu(out + residual)
            index += 4

        W_out, b_out = params[-2], params[-1]
        x = F.linear(x, W_out, b_out)

        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x




    def forward_bong(self, x,z,method='layers'):
        theta = z
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size

        W_in, b_in = params[0], params[1]
        x = F.relu(F.linear(x, W_in, b_in))
        index=2
        for i in range(self.num_layers):
            x_norm = self.norms[i](x)
            residual = x
            out = F.gelu(F.linear(x_norm, params[index], params[index + 1]))
            out = F.linear(out, params[index + 2], params[index + 3])
            x = F.gelu(out + residual)
            index += 4

        W_out, b_out = params[-2], params[-1]
        x = F.linear(x, W_out, b_out)

        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x



    def train_batch(self, train_fn, rx, symbols,method, gamma=0.999):
        if method.lower() == 'bong':
            size = self.symbol_bits * self.num_users*2
            R = torch.eye(size) * self.obs_cov_scale
            for train_rx, labels in zip(rx, symbols):
                labels = labels.flatten()
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = train_rx
                #inputs[0, self.rx_size:] = 0.5
                if self.OU == True:
                    mean_upd, cov_upd = train_fn(self.z_layers, self.cov_layers, self, inputs, labels,
                                                 R, gamma,
                                                 self.initial_cov_layers,
                                                 self.initial_mean_layers,
                                                 self.learning_rate)
                    with torch.no_grad():
                        self.z_layers.copy_(mean_upd)
                        self.cov_layers.copy_(cov_upd)
        elif method.lower() == 'sgd':
            for train_rx, labels in zip(rx, symbols):
                inputs = torch.ones((1, self.block_input_size), requires_grad=False)
                inputs[0, :self.rx_size] = train_rx
                inputs[0, self.rx_size:] = 0.5
                initial = self.z_layers.clone()
                if rx.shape[0] == 6:
                    train_fn(self,self.z_layers, inputs, labels,lr=self.learning_rate,num_steps=10)
                else:
                    train_fn(self,self.z_layers, inputs, labels,lr=self.learning_rate,num_steps=10)
                #print(torch.norm(self.z_layers-initial))



    def soft_decode_batch(self,rx):
        concat_prediction = None
        for train_rx in rx:
            inputs = torch.ones((1, self.block_input_size))
            inputs[0, :self.rx_size] = train_rx
            #inputs[0, self.rx_size:] = 0.5
            prediction = self.forward(inputs)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction, prediction], dim=0)

        return concat_prediction



