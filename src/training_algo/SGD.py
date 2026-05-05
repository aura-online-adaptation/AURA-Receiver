from importlib.metadata import requires

import  torch
from torch import nn, optim

def ResNet_SGD(model , z_layers, rx, y_true, lr=0.01,num_steps=10):
    """
    Perform a single step of Stochastic Gradient Descent (SGD) on the given model.

    Parameters:
    - model: The neural network model to be trained.
    - z_layers: List of layer indices to consider for the update.
    - x: Input data tensor.
    - y: Target data tensor.
    - lr: Learning rate for the SGD update.

    Returns:
    - Updated model after one SGD step.
    """
    # define model parameters to be updated
    z_layers = model.z_layers.clone().detach()
    model.z_layers.requires_grad = True
    if model.cov_type == 'dlr':
        model.diag_layers.requires_grad = False
        model.lr_cov_layers.requires_grad = False
    else:
        model.cov_layers.requires_grad = False
    # Define update function
    optimizer = optim.SGD([model.z_layers], lr=lr)
    eps = 1e-8
    # Zero the gradients
    for step in range(num_steps):
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(rx)
        preds = outputs.view(-1, 2)
        targets = y_true.view(-1,2)

        preds = preds.clamp(eps, 1 - eps)
        loss = -(targets * torch.log(preds)).sum(dim=1).mean()

        # Backward pass
        loss.backward()
        #print("loss:", loss.item())
        #print("grad norm:", model.z_layers.grad.norm())
        optimizer.step()

    #print(torch.norm(z_layers-model.z_layers))

    return model


def DeepSIC_SGD(model , rx, y_true, lr=0.01,num_steps=10):
    """
    Perform a single step of Stochastic Gradient Descent (SGD) on the given model.

    Parameters:
    - model: The neural network model to be trained.
    - z_layers: List of layer indices to consider for the update.
    - x: Input data tensor.
    - y: Target data tensor.
    - lr: Learning rate for the SGD update.

    Returns:
    - Updated model after one SGD step.
    """
    # define model parameters to be updated
    model.z_layers.requires_grad = True
    if model.cov_type == 'dlr':
        model.diag_layers.requires_grad = False
        model.lr_cov_layers.requires_grad = False
    else:
        model.cov_layers.requires_grad = False
    # Define update function
    optimizer = optim.SGD([model.z_layers], lr=lr)
    eps = 1e-8
    # Zero the gradients
    for step in range(num_steps):
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(rx)
        preds = outputs.view(-1, 2)
        targets = y_true.view(-1,2)

        preds = preds.clamp(eps, 1 - eps)
        loss = -(targets * torch.log(preds)).sum(dim=1).mean()

        # Backward pass
        loss.backward()
        #print("loss:", loss.item())
        #print("grad norm:", model.z_layers.grad.norm())
        optimizer.step()

    #print(torch.norm(z_layers-model.z_layers))

    return model