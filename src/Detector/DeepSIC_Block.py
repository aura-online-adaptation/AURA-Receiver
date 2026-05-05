import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype

from .base_model import base_model_generator
torch.set_default_dtype(torch.float64)
class DeepSIC_Block_double_proj (nn.Module):
    def __init__(self,base_model, symbol_bits, num_users, num_antenas,hidden_dim ,projection_mat_hidden ,projection_mat_last,
                 phi_hidden,phi_last,cov_type='full',Rank=10,pred_type='OU', init_cov_scale=0.1, dlr_rank=10):

        """
        Args:
            base_model (nn.Module): Base neural network model to be used in the block.
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antenas (int): Number of receive antennas.
            hidden_dim (int): Size of the hidden layer of the block.
            projection_mat_last (torch.Tensor): Projection matrix for hidden layer.
            projection_mat_hidden (torch.Tensor): Projection matrix for last layer.
            phi_hidden (torch.tensor): Activation function for hidden layer.
            phi_last (torch.tensor): Activation function for last layer.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.

        """
        super().__init__()
        self.base_model = base_model
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antenas = num_antenas
        self.hidden_dim = hidden_dim
        self.cov_type = cov_type

        #projection matrices and biases

        self.A_hidden = projection_mat_hidden
        self.A_last = projection_mat_last
        self.phi_hidden = phi_hidden
        self.phi_last = phi_last
        # Latent variables
        self.z_layers = nn.Parameter(0.01*torch.randn(self.A_hidden.shape[0]))
        self.z_last = nn.Parameter(0.01*torch.randn(self.A_last.shape[0]))



        self.init_cov_scale = init_cov_scale
        self.dlr_rank = dlr_rank

        #generate covariance matrices
        self.generate_cov_matrix(cov_type,self.dlr_rank)

        if pred_type == 'OU':
            self.generate_initial_cov_matrix(self.cov_type)
            self.initial_mean_layers = self.z_layers.clone()
            self.initial_mean_last = self.z_last.clone()

        # base model parameter
        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)

        self.offsets = []
        offset_val = 0
        for s in self.sizes:
            self.offsets.append((offset_val, offset_val + s))
            offset_val += s

        self.activation = self.extract_activations(self.base_model)


    def extract_activations(self,base_model):
        activations = []
        layers = list(base_model.children())

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    activations.append(layers[i + 1])
                else:
                    activations.append(None)

        return nn.ModuleList(activations)


    def forward(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        idx = 0
        for i in range(len(self.shapes)//2):
            w_start, w_end = self.offsets[idx]
            weight = theta[:, w_start:w_end].view(self.shapes[idx])
            idx += 1
            
            b_start, b_end = self.offsets[idx]
            bias = theta[:, b_start:b_end].view(self.shapes[idx])
            idx += 1
            
            x = F.linear(x, weight, bias=bias)
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        return x.squeeze(0)



    def expend(self):
        """Expand parameters from vector to model parameters"""
        theta_hidden =  self.z_layers @ self.A_hidden  + self.phi_hidden
        theta_last = self.z_last @ self.A_last + self.phi_last
        theta = torch.cat([theta_hidden, theta_last], dim=1)
        return theta




    def generate_cov_matrix(self,cov_type,Rank):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'full':
            self.cov_layers = nn.Parameter(torch.eye(self.A_hidden.shape[0])*self.init_cov_scale)
            self.cov_last = nn.Parameter(torch.eye(self.A_last.shape[0])*self.init_cov_scale)
        elif cov_type == 'diag':
            self.cov_layers = nn.Parameter(torch.ones(self.A_hidden.shape[0])*self.init_cov_scale)
            self.cov_last = nn.Parameter(torch.ones(self.A_last.shape[0])*self.init_cov_scale)
        else:
            self.diag_layers = nn.Parameter(torch.ones(self.A_hidden.shape[0])*self.init_cov_scale)
            self.diag_last = nn.Parameter(torch.ones(self.A_last.shape[0])*self.init_cov_scale)
            self.lr_cov_layers = nn.Parameter(torch.randn(self.A_hidden.shape[0],Rank)*self.init_cov_scale)
            self.lr_cov_last = nn.Parameter(torch.randn(self.A_last.shape[0],15)*self.init_cov_scale)
        return

    def generate_initial_cov_matrix(self,cov_type):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'full':
            self.initial_cov_layers = self.cov_layers.clone()
            self.initial_cov_last =self.cov_last.clone()
        elif cov_type == 'diag':
            self.initial_cov_layers= self.cov_layers.clone()
            self.initial_cov_last = self.cov_last.clone()
        else:
            self.initial_diag_layers = self.diag_layers.clone()
            self.initial_diag_last = self.diag_last.clone()
            self.initial_lr_cov_layers = self.lr_cov_layers.clone()
            self.initial_lr_cov_last = self.lr_cov_last.clone()
        return

    def forward_bong(self, x,z,method = 'layers'):
        theta = self.expend_bong(z,method)
        # Unravel theta into model parameters
        idx = 0
        for i in range(len(self.shapes)//2):
            w_start, w_end = self.offsets[idx]
            weight = theta[:, w_start:w_end].view(self.shapes[idx])
            idx += 1
            
            b_start, b_end = self.offsets[idx]
            bias = theta[:, b_start:b_end].view(self.shapes[idx])
            idx += 1
            
            x = F.linear(x, weight, bias=bias)
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        return x.squeeze(0)

    def expend_bong(self,z,method='layers'):
        """Expand parameters from vector to model parameters"""
        if method == 'layers':
            theta_hidden = z @ self.A_hidden + self.phi_hidden
            theta_last = self.z_last @ self.A_last + self.phi_last
        elif method == 'last':
            theta_hidden = self.z_layers @ self.A_hidden + self.phi_hidden
            theta_last = z @ self.A_last + self.phi_last

        theta = torch.cat([theta_hidden, theta_last], dim=1)
        return theta




class DeepSIC_Block_single_proj (nn.Module):
    def __init__(self,base_model, symbol_bits, num_users, num_antenas,hidden_dim ,projection_mat_hidden ,
                 phi_hidden,cov_type='full',Rank=10,pred_type='OU',F=None,Q=None ,beta=None, init_cov_scale=0.1, dlr_rank=10):

        """
        Args:
            base_model (nn.Module): Base neural network model to be used in the block.
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antenas (int): Number of receive antennas.
            hidden_dim (int): Size of the hidden layer of the block.
            projection_mat_hidden (torch.Tensor): Projection matrix for last layer.
            phi_hidden (torch.tensor): Activation function for hidden layer.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.

        """
        super().__init__()
        self.base_model = base_model
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antenas = num_antenas
        self.hidden_dim = hidden_dim
        self.cov_type = cov_type

        #projection matrices and biases
        self.A_hidden = nn.Parameter(projection_mat_hidden.to(dtype=torch.float32))
        self.phi_hidden = nn.Parameter(phi_hidden.to(dtype=torch.float32))

        # SSM parameters
        if F is not None and Q is not None and beta is not None:
            self.F = nn.Parameter(F.to(dtype=torch.float32))
            self.Q = nn.Parameter(Q.to(dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))


        # Latent variables
        self.z_layers =  torch.randn(self.A_hidden.shape[0],dtype=torch.float32,)*0.1

        self.init_cov_scale = init_cov_scale
        self.dlr_rank = dlr_rank

        #generate covariance matrices
        self.generate_cov_matrix(cov_type,self.dlr_rank)

        if pred_type == 'OU':
            self.generate_initial_cov_matrix(self.cov_type)
            self.initial_mean_layers = self.z_layers.clone()

        # base model parameter
        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)
        
        self.offsets = []
        offset_val = 0
        for s in self.sizes:
            self.offsets.append((offset_val, offset_val + s))
            offset_val += s
            
        self.activation = self.extract_activations(self.base_model)

        self.large_param = None


    def extract_activations(self,base_model):
        activations = []
        layers = list(base_model.children())

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    activations.append(layers[i + 1])
                else:
                    activations.append(None)

        return nn.ModuleList(activations)


    def forward(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        idx = 0
        for i in range(len(self.shapes)//2):
            w_start, w_end = self.offsets[idx]
            weight = theta[w_start:w_end].view(self.shapes[idx])
            idx += 1
            
            b_start, b_end = self.offsets[idx]
            bias = theta[b_start:b_end].view(self.shapes[idx])
            idx += 1
            
            x = F.linear(x, weight, bias=bias)
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x


    def expend(self):
        """Expand parameters from vector to model parameters"""
        z = self.z_layers
        theta =  z @ self.A_hidden  + self.phi_hidden
        return theta

    def forward_soft(self, x,flag=False):
        # Unravel theta into model parameters
        if flag == True:
            z = self.z_layers.detach()
            theta = z @ self.A_hidden + self.phi_hidden
            offset = 0
            params = []
            for shape, size in zip(self.shapes, self.sizes):
                params.append(theta[offset:offset + size].view(shape))
                offset += size
            self.large_param = params
        for i in range(len(self.large_param)//2):
            x = F.linear(x, self.large_param[2*i], bias=self.large_param[2*i+1])
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x,dim=-1)
        x = x.view(-1)
        return x


    def generate_cov_matrix(self,cov_type,Rank):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'full':
            self.cov_layers = torch.eye(self.A_hidden.shape[0])*self.init_cov_scale

        elif cov_type == 'diag':
            self.cov_layers = torch.ones(self.A_hidden.shape[0])*self.init_cov_scale

        else:
            self.diag_layers = torch.ones(self.A_hidden.shape[0])*self.init_cov_scale
            self.lr_cov_layers = torch.randn(self.A_hidden.shape[0],Rank)*self.init_cov_scale
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

    def forward_bong(self, x,z,method = 'layers'):
        theta = self.expend_bong(z,method)
        # Unravel theta into model parameters
        idx = 0
        for i in range(len(self.shapes)//2):
            w_start, w_end = self.offsets[idx]
            weight = theta[w_start:w_end].view(self.shapes[idx])
            idx += 1
            
            b_start, b_end = self.offsets[idx]
            bias = theta[b_start:b_end].view(self.shapes[idx])
            idx += 1
            
            x = F.linear(x, weight, bias=bias)
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1,2)
        x = torch.softmax(x,dim=-1)
        x= x.view(-1)
        return x

    def forward_logits(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        """if theta.requires_grad:
            g = torch.autograd.grad(theta.sum(), self.z_layers, allow_unused=True, retain_graph=True)[0]
            print("d(theta)/d(z_layers) is None?", g is None)"""
        idx = 0
        for i in range(len(self.shapes) // 2):
            w_start, w_end = self.offsets[idx]
            weight = theta[w_start:w_end].view(self.shapes[idx])
            idx += 1
            
            b_start, b_end = self.offsets[idx]
            bias = theta[b_start:b_end].view(self.shapes[idx])
            idx += 1
            
            x = F.linear(x, weight, bias=bias)
            activation = self.activation[i]
            if activation != None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = x.view(-1)
        return x

    def expend_bong(self,z,method='layers'):
        """Expand parameters from vector to model parameters"""
        if method == 'layers':
            theta = z @ self.A_hidden + self.phi_hidden
        else:
            raise "Implemented for 2 layers while method is for one"

        return theta


class DeepSIC_Block_No_proj(nn.Module):
    def __init__(self, base_model, symbol_bits, num_users, num_antenas, hidden_dim,
                 cov_type='full', Rank=10, pred_type='OU', init_cov_scale=0.1, dlr_rank=10):

        """
        Args:
            base_model (nn.Module): Base neural network model to be used in the block.
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antenas (int): Number of receive antennas.
            hidden_dim (int): Size of the hidden layer of the block.
            projection_mat_hidden (torch.Tensor): Projection matrix for last layer.
            phi_hidden (torch.tensor): Activation function for hidden layer.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.

        """
        super().__init__()
        self.base_model = base_model
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antenas = num_antenas
        self.hidden_dim = hidden_dim
        self.cov_type = cov_type

        # base model parameter
        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)
        self.activation = self.extract_activations(self.base_model)


        # Latent variables
        self.z_layers = nn.Parameter(generate_mean(self.num_antenas,self.symbol_bits,self.num_users,self.hidden_dim))



        self.init_cov_scale = init_cov_scale
        self.dlr_rank = dlr_rank

        # generate covariance matrices
        self.generate_cov_matrix(cov_type, self.dlr_rank)

        if pred_type == 'OU':
            self.generate_initial_cov_matrix(self.cov_type)
            self.initial_mean_layers = self.z_layers.clone()



    def extract_activations(self, base_model):
        activations = []
        layers = list(base_model.children())

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    activations.append(layers[i + 1])
                else:
                    activations.append(None)

        return nn.ModuleList(activations)

    def forward(self, x):
        theta = self.z_layers
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size
        for i in range(len(params) // 2):
            x = F.linear(x, params[2 * i], bias=params[2 * i + 1])
            activation = self.activation[i]
            if activation != None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x, dim=-1)
        x = x.view(-1)
        return x


    def generate_cov_matrix(self, cov_type, Rank):
        """Generate covariance matrix based on the specified type."""
        if cov_type == 'full':
            self.cov_layers = nn.Parameter(torch.eye(self.total_params)*self.init_cov_scale)

        elif cov_type == 'diag':
            self.cov_layers = nn.Parameter(torch.ones(self.total_params)*self.init_cov_scale)

        else:
            self.diag_layers = nn.Parameter(torch.ones(self.total_params) * self.init_cov_scale)
            self.lr_cov_layers = nn.Parameter(torch.eye(self.total_params, Rank) * self.init_cov_scale)
        return

    def generate_initial_cov_matrix(self, cov_type):
        """Generate covariance matrix based on the specified type."""
        if cov_type == 'full':
            self.initial_cov_layers = self.cov_layers.clone()
        elif cov_type == 'diag':
            self.initial_cov_layers = self.cov_layers.clone()
        else:
            self.initial_diag_layers = self.diag_layers.clone()
            self.initial_lr_cov_layers = self.lr_cov_layers.clone()

        return

    def forward_bong(self, x, z, method='layers'):
        theta = z
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size
        for i in range(len(params) // 2):
            x = F.linear(x, params[2 * i], bias=params[2 * i + 1])
            activation = self.activation[i]
            if activation != None:
                x = activation(x)
        x = x.squeeze(0)

        x = x.view(-1, 2)
        x = torch.softmax(x, dim=-1)
        x = x.view(-1)
        return x


def generate_mean(num_antenas,symbol_bits,num_users,hidden_dim):
    model = base_model_generator(num_antenas*2+symbol_bits*(num_users)*2,hidden_dim,symbol_bits*2)
    params = [p.detach().view(-1) for p in model.parameters()]
    return torch.cat(params)

