import torch
torch.set_default_dtype(torch.float32)
def define_projection_matrix_and_bias(total_parameter,num_wanted_params,method='xavier',learnable=False ):

    """Define projection matrix and bias function
    Args :
        total_parameter (int): Total number of parameters in the model part (hidden,last).
        num_wanted_params (int): Number of wanted parameters after projection.
        method (str, optional): Method for initializing the projection matrix. Defaults to 'xavier'.
        learnable (bool, optional): Whether the projection matrices is learnable. Defaults to False

    Returns:
        projection_mat (torch.Tensor): Projection matrix of shape (total_parameter, num_wanted_params).
        phi (torch.tensor): Bias function of shape (num_wanted_params,1).
    """
    A = torch.zeros((num_wanted_params, total_parameter))
    phi = torch.zeros((total_parameter,2))

    if method == 'xavier':
        torch.nn.init.xavier_uniform_(A)
        torch.nn.init.xavier_uniform_(phi)
    elif method == 'normal':
        torch.nn.init.normal_(A, mean=0.0, std=1.0)
        torch.nn.init.normal_(phi, mean=0.0, std=1.0)
    else:
        raise ValueError("Unsupported initialization method")

    if learnable:
        A = torch.nn.Parameter(A)
        phi = phi[:,0]
        phi = torch.nn.Parameter(phi)
    else:
        phi = phi[:,0]
    return A,phi


def define_F_projection(num_wanted_params, F_style='full'):
    """Define projection matrix and bias function for F projection
    Args :
        total_parameter (int): Total number of parameters in the model part (hidden,last).
        num_wanted_params (int): Number of wanted parameters after projection.
        method (str, optional): Method for initializing the projection matrix. Defaults to 'xavier'.
        learnable (bool, optional): Whether the projection matrices is learnable. Defaults to False

    Returns:
        projection_mat (torch.Tensor): Projection matrix of shape (total_parameter, num_wanted_params).
        phi (torch.tensor): Bias function of shape (num_wanted_params,1).
    """
    if F_style == 'diag':
        F = torch.ones(num_wanted_params) * 0.99
    else:
        F = torch.eye(num_wanted_params)*0.99
    
    Q = 1e-3 * torch.eye(num_wanted_params) #+ torch.ones(num_wanted_params,num_wanted_params)*1e-6

    beta = torch.normal(0, 0.1, size=(num_wanted_params,))*0.01

    return F,Q,beta


