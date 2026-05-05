import torch
import time
import torch.nn as nn
import argparse
import copy
import itertools
from networkx.algorithms.isomorphism.isomorph import fast_graph_could_be_isomorphic

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float64)

import gc
import sys
from scipy.io import savemat
import os
import numpy as np
from tqdm import tqdm
from experiments.framework.read_config import load_config,validate_config,clean_config
from src.channel.Uplink_MIMO_Channel import UplinkMimoChannel
from src.Detector.DeepSIC_Det import DeepSIC_proj,DeepSIC
from src.channel.Modulation import MODULATIONS
from BONG_torch_version import BONG,BOG
from src.Utils.utils import prepare_experiment_data
from experiments.framework.single_run import create_online_train_fn, create_model,create_channel





def train_projection(config):
    device = "cpu"
    #create channel
    channel =create_channel(config)
    #create model
    detector =  create_model(config)
    #create online function
    train_fn = create_online_train_fn(config)

    # data preperation
    sync_frames = config['experiment']['sync_frames']

    sync_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['symbols_per_frame'],
        num_frames=sync_frames,
        snr=config['channel']['snr'],
    )

    train_1_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples= config['projection']['symbols per frame'],
        num_frames= config['projection']['frames'],
        snr = config['channel']['snr'],
        start_frame=sync_frames
    )

    train_2_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['projection']['symbols per frame']*20,
        num_frames=config['projection']['frames'],
        snr=config['channel']['snr'],
        start_frame=sync_frames
    )

    test_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples = config['projection']['test symbols'],
        num_frames=  config['projection']['frames'],
        snr= config['channel']['snr'],
        start_frame=sync_frames
    )

    #pre-train sync phase
    """for train_rx,train_label in  tqdm(sync_dataloader, total=sync_frames, leave=False, desc='Sync frames'):
        detector.train_batch(train_fn,train_rx,train_label)"""




    #somehow to run over users and layers
    #for every user and layer:
    #run over batches
        #forward
        #loss definition
        #loss backwards
        #optimizeer step
        #generate inputs for next layer
    def get_lr(key, default):
        return config.get('projection', {}).get('optimizer', {}).get(key, default)

    if config['model']['type'].lower() == 'deepsic':
        loss_log_blocks = [[[] for u in range(detector.num_users)] for l in range(detector.num_layers)]
        acc_log_blocks = [[[] for u in range(detector.num_users)] for l in range(detector.num_layers)]
        step = 0
        
        lr_proj = get_lr('projection_lr', 1e-3)
        lr_proj_F = get_lr('projection_lr', 1e-5)
        lr_F = get_lr('F_lr', 1e-7)
        lr_Q = get_lr('Q_lr', 1e-7)
        lr_beta = get_lr('beta_lr', 1e-7)

        if config['model']['OU'] == True and config['model']['F'] == False:
            optimizers = [[torch.optim.Adam(
                        [detector.blocks[l][u].A_hidden, detector.blocks[l][u].phi_hidden],
                        lr=lr_proj) for u in range(detector.num_users)] for l in range(detector.num_layers)]
            detector_param = block_named_param(detector,names = ['A_hidden','phi_hidden'])
        elif config['model']['F'] == True and config['model']['OU'] == False:
            optimizers = [[torch.optim.Adam([
            {"params": [detector.blocks[l][u].A_hidden, detector.blocks[l][u].phi_hidden], "lr": lr_proj_F},
            {"params": [detector.blocks[l][u].F], "lr": lr_F},
            {"params": [detector.blocks[l][u].Q], "lr": lr_Q},
            {"params": [detector.blocks[l][u].beta], "lr": lr_beta},]) for u in range(detector.num_users)] for l in range(detector.num_layers)]
            detector_param = block_named_param(detector, names=['A_hidden', 'phi_hidden','F','Q','beta'])
        else:
            raise ValueError("problem with config, OU and F should not be both true or both false")
        criterion = nn.BCELoss()

    elif config['model']['type'].lower() == 'resnet':
        loss_log = []
        acc_log = []
        step = 0
        proj_params = []
        if config['projection']['proj_type'] == 'affine':
            proj_params = [detector.A_matrix, detector.phi_vector]

        elif config['projection']['proj_type'].lower() == 'dnn':
            for blk in detector.proj_models:
                proj_params += list(blk.parameters())

        is_dnn = config['projection']['proj_type'].lower() == 'dnn'
        default_proj = 1e-3 if is_dnn else 1e-4
        key_proj = 'projection_lr_dnn' if is_dnn else 'projection_lr_affine'
        
        lr_proj = get_lr(key_proj, default_proj)
        lr_F = get_lr('F_lr', 1e-7)
        lr_Q = get_lr('Q_lr', 1e-7)
        lr_beta = get_lr('beta_lr', 1e-7)

        if config['model']['OU'] == True and config['model']['F'] == False:
            optimizer = torch.optim.Adam([
                {"params": proj_params,
                    "lr": lr_proj}])

        elif config['model']['F'] == True and config['model']['OU'] == False:

            optimizer = torch.optim.Adam([
                {"params": proj_params, "lr": lr_proj},
                {"params": detector.F_data, "lr": lr_F},
                {"params": detector.Q, "lr": lr_Q},
                {"params": detector.beta, "lr": lr_beta},])

        else:
            raise ValueError("problem with config, OU and F should not be both true or both false")

        criterion = nn.BCELoss()

    best_BER = 0.1

    for (train_batch, label_batch), (train_2_batch, label_2_batch) in tqdm(
            zip(train_1_dataloader, train_2_dataloader),
            total=len(train_1_dataloader),
            desc="Training batches",
            file=sys.stdout,
            leave=True):

        if config['model']['type'].lower() == 'deepsic':
            for layer in range(config['model']['num_layers']):
                for user in range(config['channel']['num_users']):
                    train_2_batch_iter = iter(train_2_batch)
                    label_2_batch_iter = iter(label_2_batch)
                    block = detector.blocks[layer][user]
                    initial_A = block.A_hidden.detach().clone()
                    optimizer = optimizers[layer][user]
                    y_pred,y_true= detector.forward_offline (layer,user,train_batch,label_batch,train_2_batch_iter,label_2_batch_iter,train_fn)
                    if step %10 == 0:
                        with torch.no_grad():
                            yp = y_pred.view(-1, 2)  # logits
                            yt = y_true.view(-1, 2).argmax(dim=-1)  # 0/1
                            pred = yp.argmax(dim=-1)
                            errors = (pred != yt).float().sum().item()
                            BER = errors / yt.numel()
                            del pred

                    y_pred = y_pred.reshape(config['projection']['symbols per frame'], -1, 2).float()
                    y_true = y_true.reshape(config['projection']['symbols per frame'], -1, 2).float()
                    y_pred = y_pred.reshape(-1, 2)
                    targets = y_true.reshape(-1, 2).argmax(dim=-1)

                    loss = torch.nn.CrossEntropyLoss()(y_pred, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step %20 == 0:
                        loss_log_blocks[layer][user].append(loss.item())
                        acc_log_blocks[layer][user].append(BER)
                    if step %20 == 0:
                        print(
                            f"batch {step}/{config['projection']['frames']} block number: {layer * 3 + user} , BER= {BER}, change in A :{torch.norm(initial_A - block.A_hidden)} loss: {loss.item()}")

                    del y_pred, y_true, loss
                    del train_2_batch_iter, label_2_batch_iter,initial_A
            detach_everything_saved_DeepSIC(detector)

        elif config['model']['type'].lower() == 'resnet':
            train_2_batch_iter = iter(train_2_batch)
            label_2_batch_iter = iter(label_2_batch)
            y_pred, y_true = detector.forward_offline( train_batch, label_batch, train_2_batch_iter,
                                                      label_2_batch_iter, train_fn)
            if step % 3 == 0:
                with torch.no_grad():
                    y_pred_pairs = y_pred.view(-1, 2)
                    y_true_pairs = y_true.view(-1, 2)
                    predicted_bits = torch.argmax(y_pred_pairs, dim=-1)
                    true_bits = torch.argmax(y_true_pairs, dim=-1)
                    errors = (predicted_bits != true_bits).float().sum().item()
                    total_bits = true_bits.numel()
                    BER = errors / total_bits
                    del predicted_bits, true_bits, y_pred_pairs, y_true_pairs

            y_pred = y_pred.reshape(config['projection']['symbols per frame'], -1, 2).float()
            y_true = y_true.reshape(config['projection']['symbols per frame'], -1, 2).float()
            y_pred = y_pred.reshape(-1, 2)
            targets = y_true.reshape(-1, 2).argmax(dim=-1)

            loss = torch.nn.CrossEntropyLoss()(y_pred, targets)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            detach_everything_saved_in_resnet(detector)

            if step % 3 == 0:
                loss_log.append(loss.item())
                acc_log.append(BER)
            if step % 5 == 0:
                print(f"batch {step}/{config['projection']['frames']}  , BER= {BER}, loss: {loss.item()}")

            if BER <= best_BER and step > 150:
                best_BER = BER
                save_matrices_ResNet(config, detector)

                print(f"New best BER: {best_BER} at step {step}")
            del y_pred, y_true, loss
            del train_2_batch_iter, label_2_batch_iter

        gc.collect()

        step += 1




    #pul out the matrices
    if config['model']['type'].lower() == 'deepsic':
        run_dir = save_matrices_DeepsIC(config,detector)
    elif config['model']['type'].lower() == 'resnet':
        run_dir = save_matrices_ResNet(config,detector)

    #save matrices
    return run_dir



def block_named_param(detector,names = None):
    for l in range(detector.num_layers):
        for u in range(detector.num_users):
            b = detector.blocks[l][u]
            for n in names:
                if not hasattr(b, n):
                    continue
                p = getattr(b, n)
                if p is None:
                    continue
                if isinstance(p, nn.Parameter) or (torch.is_tensor(p) and p.requires_grad):
                    yield f"blocks[{l}][{u}].{n}", p

def detach_everything_saved_DeepSIC(detector):
    for l in range(detector.num_layers):
        for u in range(detector.num_users):
            b = detector.blocks[l][u]
            for k, v in list(b.__dict__.items()):
                if isinstance(v, nn.Parameter):
                    continue
                if torch.is_tensor(v) and v.grad_fn is not None:
                    b.__dict__[k] = v.detach()
                elif isinstance(v, (list, tuple)):
                    new = []
                    changed = False
                    for item in v:
                        if torch.is_tensor(item) and item.grad_fn is not None:
                            new.append(item.detach())
                            changed = True
                        else:
                            new.append(item)
                    if changed:
                        b.__dict__[k] = type(v)(new)
                elif isinstance(v, dict):
                    new = {}
                    changed = False
                    for dk, dv in v.items():
                        if torch.is_tensor(dv) and dv.grad_fn is not None:
                            new[dk] = dv.detach()
                            changed = True
                        else:
                            new[dk] = dv
                    if changed:
                        b.__dict__[k] = new



def detach_everything_saved_in_resnet(detector):
    def detach_in_obj(obj):
        for k, v in list(obj.__dict__.items()):
            if isinstance(v, nn.Parameter):
                continue
            if torch.is_tensor(v) and v.grad_fn is not None:
                obj.__dict__[k] = v.detach()
            elif isinstance(v, (list, tuple)):
                new = []
                changed = False
                for item in v:
                    if torch.is_tensor(item) and item.grad_fn is not None:
                        new.append(item.detach())
                        changed = True
                    else:
                        new.append(item)
                if changed:
                    obj.__dict__[k] = type(v)(new)
            elif isinstance(v, dict):
                new = {}
                changed = False
                for dk, dv in v.items():
                    if torch.is_tensor(dv) and dv.grad_fn is not None:
                        new[dk] = dv.detach()
                        changed = True
                    else:
                        new[dk] = dv
                if changed:
                    obj.__dict__[k] = new

    detach_in_obj(detector)

    for m in detector.modules():
        if m is detector:
            continue
        detach_in_obj(m)



def save_matrices_DeepsIC(config,detector):
    """Pull out learning matrices and saves them"""
    num_layers = detector.num_layers
    num_users = detector.num_users

    M,N = detector.blocks[0][0].A_hidden.shape
    T = detector.blocks[0][0].phi_hidden.shape[0]

    A_matrices = np.zeros((M, N, num_layers* num_users))
    Phi_vectors = np.zeros((T, num_layers* num_users))

    if detector.pred_type =="F":
        if getattr(detector, "F_style", "full") == "diag":
            Kf1 = detector.blocks[0][0].F.shape[0]
            F_matrices = np.zeros((Kf1, num_layers * num_users))
        else:
            Kf1, Kf2 = detector.blocks[0][0].F.shape
            F_matrices = np.zeros((Kf1, Kf2, num_layers * num_users))
        
        Kq1, Kq2 = detector.blocks[0][0].Q.shape
        Q_matrices = np.zeros((Kq1, Kq2, num_layers * num_users))
        beta_len = detector.blocks[0][0].beta.numel()
        beta_vectors = np.zeros((beta_len, num_layers * num_users))

    for layer in range(num_layers):
        for user in range(num_users):
            block = detector.blocks[layer][user]
            idx = layer * num_users + user
            A_matrices[:, :, idx] = block.A_hidden.detach().cpu().numpy()
            Phi_vectors[:,idx] = block.phi_hidden.detach().cpu().numpy()

            if detector.pred_type == "F":
                if getattr(detector, "F_style", "full") == "diag":
                    F_matrices[:, idx] = block.F.detach().cpu().numpy()
                else:
                    F_matrices[:, :, idx] = block.F.detach().cpu().numpy()
                Q_matrices[:, :, idx] = block.Q.detach().cpu().numpy()
                beta_vectors[:, idx] = block.beta.detach().reshape(-1).cpu().numpy()


    save_dir = config['projection'].get('save_dir', 'experiments/data/Learned_matrices')
    frames = config['projection']['frames']
    symbols = config['projection']['symbols per frame']
    covtype = config['algorithm']['covariance_type']
    snr = config['channel']['snr']
    pred_type = "F" if config['model']['F'] else "OU"
    if pred_type == "F" and getattr(detector, "F_style", "full") == "diag":
        pred_type = "F_diag"
    
    linear_str = "linear" if config['channel'].get('linear_channel', True) else "nonlinear"
    
    run_name = f"deepsic_{snr}snr_affine_{frames}x{symbols}_wanted{config['projection']['num_wanted']}_{pred_type}_cov{covtype}_{linear_str}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    filename_A = os.path.join(run_dir, "A.mat")
    filename_phi = os.path.join(run_dir, "Phi.mat")
    savemat(filename_A, {"A_matrices": A_matrices})
    savemat(filename_phi, {"Phi_vectors": Phi_vectors})

    if detector.pred_type == "F":
        filename_pred = os.path.join(run_dir, "pred.mat")
        savemat(filename_pred, {"F_matrices": F_matrices, "Q_matrices": Q_matrices, "beta_vectors": beta_vectors})
        print("Saving prediction mats to:", filename_pred)

    import json
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("Saving matrices to:", filename_A)
    print("Saving matrices to:", filename_phi)

    return run_dir



def save_matrices_ResNet(config,detector):
    """Pull out learning matrices and saves them"""

    if config['projection']['proj_type'].lower() == 'affine':
        M,N = detector.A_matrix.shape
        T = detector.phi_vector.shape[0]

        A_matrices = np.zeros((M, N))
        Phi_vectors = np.zeros(T)


        A_matrices[:, :] = detector.A_matrix.detach().cpu().numpy()
        Phi_vectors[:] = detector.phi_vector.detach().cpu().numpy()


    save_dir = config['projection'].get('save_dir', 'experiments/data/Learned_matrices')
    frames = config['projection']['frames']
    symbols = config['projection']['symbols per frame']
    covtype = config['algorithm']['covariance_type']
    snr = config['channel']['snr']
    pred_type = "F" if config['model']['F'] else "OU"
    if pred_type == "F" and getattr(detector, "F_style", "full") == "diag":
        pred_type = "F_diag"
    proj_type = config['projection']['proj_type'].lower()

    linear_str = "linear" if config['channel'].get('linear_channel', True) else "nonlinear"

    run_name = f"resnet_{snr}snr_{proj_type}_{frames}x{symbols}_wanted{config['projection']['num_wanted']}_{pred_type}_cov{covtype}_{linear_str}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if config['projection']['proj_type'].lower() == 'affine':
        filename_A = os.path.join(run_dir, "A.mat")
        filename_phi = os.path.join(run_dir, "Phi.mat")
        savemat(filename_A, {"A_matrices": A_matrices})
        savemat(filename_phi, {"Phi_vectors": Phi_vectors})
        print("Saving matrices to:", filename_A)
        print("Saving matrices to:", filename_phi)

    if config['projection']['proj_type'].lower() == 'dnn':
        checkpoint = {
            "proj_type": "dnn",
            "Proj_sizes": tuple(int(p) for p in detector.Proj_sizes),
            "BONG_sizes": tuple(int(b) for b in detector.BONG_sizes),
            "state_dicts": [blk.state_dict() for blk in detector.proj_models],
        }
        filename = os.path.join(run_dir, "DNN.pt")
        torch.save(checkpoint, filename)
        print("Saving matrices to:", filename)

    if detector.pred_type == "F":
        num_blocks = len(detector.F_data)
        beta_len = detector.beta[0].numel()
        if getattr(detector, "F_style", "full") == "diag":
            Kf1 = detector.F_data[0].shape[0]
            F_matrices = np.zeros((Kf1, num_blocks))
            Q_matrices = np.zeros((Kf1, num_blocks))
        else:
            Kf1, Kf2 = detector.F_data[0].shape
            Kq1, Kq2 = detector.Q[0].shape
            F_matrices = np.zeros((Kf1, Kf2, num_blocks))
            Q_matrices = np.zeros((Kq1, Kq2, num_blocks))
        beta_vectors = np.zeros((beta_len, num_blocks))

        for idx in range(num_blocks):
            if getattr(detector, "F_style", "full") == "diag":
                F_matrices[:, idx] = detector.F_data[idx].detach().cpu().numpy()
                Q_matrices[:, idx] = detector.Q[idx].detach().cpu().numpy()
            else:
                F_matrices[:, :, idx] = detector.F_data[idx].detach().cpu().numpy()
                Q_matrices[:, :, idx] = detector.Q[idx].detach().cpu().numpy()
            beta_vectors[:, idx] = detector.beta[idx].detach().reshape(-1).cpu().numpy()

        filename_pred = os.path.join(run_dir, "pred.mat")
        savemat(filename_pred, {"F_matrices": F_matrices, "Q_matrices": Q_matrices, "beta_vectors": beta_vectors})
        print("Saving prediction mats to:", filename_pred)

    import json
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return run_dir



def set_nested(cfg, key_path, value):
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value




def main():
    """main function to run offline projection learning
     """


    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    torch.set_default_device(device)

    import argparse

    parser = argparse.ArgumentParser(description='Run experiment from JSON config')
    parser.add_argument('--config_path', type=str, help='Path to experiment config JSON file',
                        default="single_config.json")
    parser.add_argument('--output_dir', type=str, help='Base output directory for results',
                        default=r"C:\Users\owner\OneDrive\Desktop\Msc\Codes\adaptive-deep-receiver-Torch\experiments\data")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)

    base = config.get("base_config", config)
    sweep = config.get("sweep", {})
    if not sweep:
        train_projection(base)
        return

    keys = list(sweep.keys())
    values = list(sweep.values())
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base)

        for k, v in zip(keys, combo):
            set_nested(cfg, k, v)

        train_projection(cfg)
    #save results
    return



if __name__ =='__main__':
    torch.set_default_dtype(torch.float64)
    main()