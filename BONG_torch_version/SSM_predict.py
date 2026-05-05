import torch







def predict_full_F(mean, Cov, F, beta, Q):
    """fn to predict next state based on matrix (F) with full covariance matrix

    ARGS:
        mean (torch.Tensor): mean vector
        Cov (torch.Tensor): covariance matrix
        F (torch.Tensor): state transition matrix or vector
        beta (torch.Tensor): bias term
        Q (torch.Tensor): process noise covariance matrix
    RETURNS:
        predict_mean (torch.Tensor): predicted mean vector
        predict_Cov (torch.Tensor): predicted covariance matrix

    """
    
    if F.ndim == 1:
        # F is a diagonal matrix represented as a vector
        predict_mean = F * mean + beta
        predict_Cov = (F.unsqueeze(1) * Cov) * F.unsqueeze(0) + Q
    else:
        # F is a full matrix
        predict_mean = F @ mean + beta
        predict_Cov = F @ Cov @ F.T + Q
        

    return predict_mean, predict_Cov



def predict_full_OU(mean, Cov,  gamma, initial_Cov, init_mean):
    """fn to predict next state based on OU process with full covariance matrix

    ARGS:
        mean (torch.Tensor): mean vector
        Cov (torch.Tensor): covariance matrix
        gamma (float): OU process parameter
        initial_Cov (torch.Tensor): initial covariance matrix
        init_mean (torch.Tensor): initial mean vector
    RETURNS:
        predict_mean (torch.Tensor): predicted mean vector
        predict_Cov (torch.Tensor): predicted covariance matrix

    """

    predict_mean = mean * gamma + (1 - gamma) * init_mean
    predict_Cov = (gamma ** 2) * Cov + (1 - gamma ** 2) * initial_Cov

    return predict_mean, predict_Cov


def predict_full_OU_mul_blocks(mean_arr, Cov_arr, gamma, initial_Cov_arr, init_mean_arr):
    """fn to predict next state based on OU process with full covariance matrix for multiple blocks

    ARGS:
        mean_arr (list of torch.Tensor): list of mean vectors for each block
        Cov_arr (list of torch.Tensor): list of covariance matrices for each block
        gamma (float): OU process parameter
        initial_Cov_arr (list of torch.Tensor): list of initial covariance matrices for each block
        init_mean_arr (list of torch.Tensor): list of initial mean vectors for each block
    RETURNS:
        predict_mean_arr (list of torch.Tensor): list of predicted mean vectors for each block
        predict_Cov_arr (list of torch.Tensor): list of predicted covariance matrices for each block

    """

    predict_mean_arr = []
    predict_Cov_arr = []

    for mean, Cov, initial_Cov, init_mean in zip(mean_arr, Cov_arr, initial_Cov_arr, init_mean_arr):
        predict_mean, predict_Cov = predict_full_OU(mean, Cov, gamma, initial_Cov, init_mean)
        predict_mean_arr.append(predict_mean)
        predict_Cov_arr.append(predict_Cov)

    return predict_mean_arr, predict_Cov_arr


def predict_full_F_mul_blocks(mean_arr, Cov_arr, F,Q,beta):
    """fn to predict next state based on OU process with full covariance matrix for multiple blocks

    ARGS:
        mean_arr (list of torch.Tensor): list of mean vectors for each block
        Cov_arr (list of torch.Tensor): list of covariance matrices for each block
        F (list of torch.Tensor): state transition matrix
        Q (list of torch.Tensor): process noise covariance matrix
        beta (list of torch.Tensor): bias term
    RETURNS:
        predict_mean_arr (list of torch.Tensor): list of predicted mean vectors for each block
        predict_Cov_arr (list of torch.Tensor): list of predicted covariance matrices for each block

    """

    predict_mean_arr = []
    predict_Cov_arr = []

    for mean, Cov, f,q,b in zip(mean_arr, Cov_arr, F,Q,beta):
        predict_mean, predict_Cov = predict_full_F(mean, Cov, f,b,q)
        predict_mean_arr.append(predict_mean)
        predict_Cov_arr.append(predict_Cov)

    return predict_mean_arr, predict_Cov_arr

def predict_diag_F(mean, diag_Cov, F, beta, Q):
    """fn to predict next state based on matrix (F) with diagonal covariance matrix

    ARGS:
        mean (torch.Tensor): mean vector
        diag_Cov (torch.Tensor): diagonal component of the covariance matrix
        F (torch.Tensor): state transition matrix
        beta (torch.Tensor): bias term
        Q (torch.Tensor): process noise covariance matrix
    RETURNS:
        predict_mean (torch.Tensor): predicted mean vector
        predict_diag_Cov (torch.Tensor): predicted diagonal component of the covariance matrix

    """

    predict_mean = F @ mean + beta
    predict_cov = F @ torch.diag(diag_Cov.squeeze()) @ F.T + Q
    predict_diag_Cov = torch.diag(predict_cov)

    return predict_mean, predict_diag_Cov


def predict_diag_OU(mean, diag_Cov, gamma, initial_diag_Cov, init_mean):
    """fn to predict next state based on OU process with diagonal covariance matrix

    ARGS:
        mean (torch.Tensor): mean vector
        diag_Cov (torch.Tensor): diagonal component of the covariance matrix
        gamma (float): OU process parameter
        initial_diag_Cov (torch.Tensor): initial diagonal component of the covariance matrix
        init_mean (torch.Tensor): initial mean vector
    RETURNS:
        predict_mean (torch.Tensor): predicted mean vector
        predict_diag_Cov (torch.Tensor): predicted diagonal component of the covariance matrix

    """

    predict_mean = mean * gamma + (1 - gamma) * init_mean
    predict_diag_Cov = (gamma ** 2) * diag_Cov + (1 - gamma ** 2) * initial_diag_Cov

    return predict_mean, predict_diag_Cov


def predict_DLR_F(mean, prec_diag, prec_low_rank, F, beta, Q):
    """fn to predict next state based on matrix (F) with diagonal low rank covariance matrix

    ARGS:
        mean (torch.Tensor): mean vector
        prec_diag (torch.Tensor): diagonal component of the precision matrix
        prec_low_rank (torch.Tensor): low rank component of the precision matrix
        F (torch.Tensor): state transition matrix
        beta (torch.Tensor): bias term
        Q (torch.Tensor): process noise covariance matrix
    RETURNS:
        predict_mean (torch.Tensor): predicted mean vector
        predict_prec_diag (torch.Tensor): predicted diagonal component of the precision matrix
        predict_prec_low_rank (torch.Tensor): predicted low rank component of the precision matrix

    """

    predict_mean = F @ mean + beta
    predict_cov = F @ (torch.diag(prec_diag.squeeze()) + prec_low_rank @ (prec_low_rank.T)) @ F.T + Q
    P, L = prec_low_rank.shape
    U, S, _ = torch.linalg.svd(predict_cov, full_matrices=False)
    U_new, S_new = U[:, :L], S[:L]
    predict_prec_low_rank = U_new * torch.sqrt(S_new)
    extra_U, extra_S = U[:, L:], S[L:]
    add_diag = torch.einsum("ij,ij->i", extra_U * torch.sqrt(extra_S), extra_U * torch.sqrt(extra_S)).unsqueeze(1)
    predict_prec_diag = torch.diag(predict_cov) + add_diag

    return predict_mean, predict_prec_diag, predict_prec_low_rank

def predict_DLR_OU(mean, prec_diag, prec_low_rank, gamma, initial_prec_diag, initial_prec_low_rank, init_mean):
    """fn to predict next state based on OU process with diagonal low rank covariance matrix

        ARGS:
            mean (torch.Tensor): mean vector
            prec_diag (torch.Tensor): diagonal component of the precision matrix
            prec_low_rank (torch.Tensor): low rank component of the precision matrix
            F (torch.Tensor): state transition matrix
            beta (torch.Tensor): bias term
            Q (torch.Tensor): process noise covariance matrix
        RETURNS:
            predict_mean (torch.Tensor): predicted mean vector
            predict_prec_diag (torch.Tensor): predicted diagonal component of the precision matrix
            predict_prec_low_rank (torch.Tensor): predicted low rank component of the precision matrix

        """

    predict_mean = mean * gamma + (1 - gamma) * init_mean
    predict_prec_diag = (gamma ** 2) * prec_diag + (1 - gamma ** 2) * initial_prec_diag
    predict_prec_low_rank = (gamma ** 2) * prec_low_rank + (1 - gamma ** 2) * initial_prec_low_rank

    return predict_mean, predict_prec_diag, predict_prec_low_rank