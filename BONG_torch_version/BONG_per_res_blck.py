import numpy as np
import  torch
from functorch import jacrev


from .SSM_predict import predict_full_OU_mul_blocks, predict_full_F_mul_blocks

def Update_BONG_lin_full_OU_ResNet(mean, mean_arr , Cov, model, obs,y, R, gamma, initial_cov_arr, init_mean_arr , ResNet_sizes=None):
    """
    fn to compute BONG update  with linearization and full covariance matrix
    prediction takes action according to Ornstein–Uhlenbeck proccess

    Arg:
    Mean (torch.Tensor): Mean vector of the prior distribution
    Mean_arr (list of torch.Tensor): List of mean vectors at each residual block
    Cov (torch.Tensor): Array that contain Covariance matrices of the prior distribution for each residual block
    model (Callable or nn.Module): supply forward function of the model
    obs (torch.Tensor): Observation vector
    y (torch.Tensor): Actual observation vector
    R (torch.Tensor): Observation noise covariance matrix
    gamma (float): OU process parameter
    initial_cov (torch.Tensor): Initial covariance matrix for OU process
    init_mean (torch.Tensor): Initial mean vector for OU process

    Returns:
    Mean_upd (torch.Tensor): Mean vector of the posterior distribution
    Cov_upd (torch.Tensor): Covariance matrix of the posterior distribution
    """

    predict_mean,predict_cov = predict_full_OU_mul_blocks(mean_arr, Cov, gamma, initial_cov_arr, init_mean_arr)


    y_pred = model(obs)
    z = mean.requires_grad_(True)
    H = torch.func.jacrev(lambda z_: model.forward_bong(obs, z_, 'layers'))(z)

    mean_upd = None
    mean_upd_arr = []
    cov_upd = []
    Hs = torch.split(H, ResNet_sizes, dim=1)
    inovation = y - y_pred
    for Hb, mean_b, Cov_b in zip(Hs, predict_mean, predict_cov):

        C = Cov_b @ Hb.T
        S = R + Hb @ C

        K = torch.linalg.lstsq(S, C.T).solution.T
        #print(torch.norm(y-y_pred))
        mean_upd_tag = mean_b + K @ inovation
        cov_upd_tag = Cov_b- K @ S @ K.T

        if mean_upd is None:
            mean_upd = mean_upd_tag
            mean_upd_arr.append(mean_upd_tag)
            cov_upd.append(cov_upd_tag)

        else:
            mean_upd_arr.append(mean_upd_tag)
            cov_upd.append(cov_upd_tag)

    mean_upd = torch.concat((mean_upd_arr), dim=0)
    return mean_upd, mean_upd_arr , cov_upd


def Update_BONG_lin_full_F_ResNet(mean, mean_arr , Cov, model, obs,y, R, F,Q,beta,ResNet_sizes=None):
    """
    fn to compute BONG update  with linearization and full covariance matrix
    prediction takes action according to Ornstein–Uhlenbeck proccess

    Arg:
    Mean (torch.Tensor): Mean vector of the prior distribution
    Mean_arr (list of torch.Tensor): List of mean vectors at each residual block
    Cov (torch.Tensor): Array that contain Covariance matrices of the prior distribution for each residual block
    model (Callable or nn.Module): supply forward function of the model
    obs (torch.Tensor): Observation vector
    y (torch.Tensor): Actual observation vector
    R (torch.Tensor): Observation noise covariance matrix
    F (torch.Tensor): State transition matrix for F process
    Q (torch.Tensor): Process noise covariance matrix for F process
    beta (torch.Tensor): Bias term for F process

    Returns:
    Mean_upd (torch.Tensor): Mean vector of the posterior distribution
    Cov_upd (torch.Tensor): Covariance matrix of the posterior distribution
    """

    predict_mean,predict_cov = predict_full_F_mul_blocks(mean_arr, Cov, F , Q , beta)


    y_pred = model(obs)
    z = mean.requires_grad_(True)
    H = torch.func.jacrev(lambda z_: model.forward_bong(obs, z_, 'layers'))(z)

    mean_upd = None
    mean_upd_arr = []
    cov_upd = []
    Hs = torch.split(H, ResNet_sizes, dim=1)
    inovation = y - y_pred
    for Hb, mean_b, Cov_b in zip(Hs, predict_mean, predict_cov):

        C = Cov_b @ Hb.T
        S = R + Hb @ C

        K = torch.linalg.lstsq(S, C.T).solution.T
        #print(torch.norm(y-y_pred))
        mean_upd_tag = mean_b + K @ inovation
        cov_upd_tag = Cov_b- K @ S @ K.T

        if mean_upd is None:
            mean_upd = mean_upd_tag
            mean_upd_arr.append(mean_upd_tag)
            cov_upd.append(cov_upd_tag)

        else:
            mean_upd_arr.append(mean_upd_tag)
            cov_upd.append(cov_upd_tag)

    mean_upd = torch.concat((mean_upd_arr), dim=0)
    return mean_upd, mean_upd_arr , cov_upd


def Update_BONG_lin_full_OU_DeepSIC(detector, rx, y_true, R):

    z = detector.pack_latents().requires_grad_(True)

    y_pred,inter = detector.forward_bong_entire(rx, z, return_intermediate=True)

    H = torch.func.jacrev(lambda z_: detector.forward_bong_entire(rx, z_, return_intermediate=False))(z)
    H = H[0]
    H = H.squeeze(0)
    y_pred = y_pred.squeeze(0)

    y_true = torch.flatten(y_true)

    out_dim_user = detector.symbol_bits * 2
    mean_upd_arr, cov_upd_arr = [], []

    col_ptr = 0
    for (layer, user, out), b in zip(inter, detector.block_list):
        z_dim = b.z_layers.numel()
        col_start = col_ptr
        col_end = col_ptr + z_dim
        col_ptr += z_dim

        row_start = user * out_dim_user
        row_end = row_start + out_dim_user

        Hb = H[row_start:row_end, col_start:col_end]

        innovation = y_true[row_start:row_end] - out

        C = b.cov_layers @ Hb.T
        S = R + Hb @ C
        K = torch.linalg.lstsq(S, C.T).solution.T

        mean_upd_arr.append(b.z_layers + K @ innovation)
        cov_upd_arr.append(b.cov_layers - K @ S @ K.T)

    for b, m, c in zip(detector.block_list, mean_upd_arr, cov_upd_arr):
        with torch.no_grad():
            b.z_layers.copy_(m)
            b.cov_layers.copy_(c)
