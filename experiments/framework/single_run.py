import os
"""os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
"""
import torch
"""torch.set_num_threads(1)"""

import time
import json
import math
import gc

torch.set_default_dtype(torch.float32)

import os
from tqdm import tqdm
from experiments.framework.read_config import load_config,validate_config,clean_config
from src.channel.Uplink_MIMO_Channel import UplinkMimoChannel
from src.Detector.DeepSIC_Det import DeepSIC_proj,DeepSIC
from src.channel.Modulation import MODULATIONS
from BONG_torch_version import BONG,BOG,BONG_per_res_blck
from src.Utils.utils import prepare_experiment_data
from src.Detector.ResNet import ResNet_proj,ResNet_no_proj
from src.training_algo import SGD



def create_model(config):
    """create DeepSIC Detector base on the configuration"""
    model_config = config['model']
    channel_config = config['channel']
    algo_config = config['algorithm']
    model_type = model_config['type'].lower()

    if config['model']['learned matrix'] == True:
        load_dir = config['projection'].get('load_dir')
        if not load_dir:
            raise ValueError("load_dir is required in projection config when learned matrix is True.")
            
        offline_config_path = os.path.join(load_dir, "config.json")
        if os.path.exists(offline_config_path):
            with open(offline_config_path, "r") as f:
                offline_cfg = json.load(f)
            off_lin = offline_cfg.get('channel', {}).get('linear_channel', True)
            on_lin = channel_config.get('linear_channel', True)
            if off_lin != on_lin:
                if not config.get('experiment', {}).get('allow_mismatch', False):
                    raise ValueError(f"Mismatch! Offline matrix was learned on linear_channel={off_lin}, "
                                     f"but online is linear_channel={on_lin}. Use experiment.allow_mismatch=True to override.")
                                     
        A_path = os.path.join(load_dir, "A.mat")
        phi_path = os.path.join(load_dir, "Phi.mat")
        pred_path = os.path.join(load_dir, "pred.mat")
        dnn_path = os.path.join(load_dir, "DNN.pt")
        param_path = "none"
    else:
        A_path = "none"
        phi_path = "none"
        pred_path = "none"
        dnn_path = "none"
        param_path = "none"

    if model_type == 'deepsic':
        if model_config['Pulse'] == True:
            detector = DeepSIC_proj(
                symbol_bits = int(torch.log2(torch.tensor(len(MODULATIONS[channel_config['modulation']])))),
                num_users=channel_config['num_users'],
                num_antennas=channel_config['num_antennas'],
                num_layers=model_config['num_layers'],
                hidden_dim=model_config['hidden_dim'],
                cov_type= config['algorithm']['covariance_type'],
                init_cov_scale= model_config['init_param_cov'],
                obs_cov_scale= algo_config.get('obs_cov_scale', 0.1),
                dlr_rank= algo_config.get('dlr_rank', 10),
                Pulse=model_config["Pulse"],
                OU= model_config["OU"],
                F = model_config["F"],
                block_method =model_config['projection_mode'],
                learning_rate= algo_config['learning_rate'],
                learning_proj= config['projection']['learning proj'],
                learned_mat= config['model']['learned matrix'],
                learned_params=config['model']['learned param'],
                learned_A_path=A_path,
                learned_phi_path=phi_path,
                learned_params_path=param_path,
                param_wanted= config['projection']['num_wanted'],
                block_update=config['model']['block_update'],
                pred_path= pred_path,
                F_style= model_config.get('F_style', 'full')
            )
        else:
            detector=DeepSIC(
                symbol_bits = int(torch.log2(torch.tensor(len(MODULATIONS[channel_config['modulation']])))),
                num_users=channel_config['num_users'],
                num_antennas=channel_config['num_antennas'],
                num_layers=model_config['num_layers'],
                hidden_dim=model_config['hidden_dim'],
                cov_type= config['algorithm']['covariance_type'],
                init_cov_scale= model_config['init_param_cov'],
                obs_cov_scale= algo_config.get('obs_cov_scale', 0.1),
                dlr_rank= algo_config.get('dlr_rank', 10),
                Pulse=model_config["Pulse"],
                OU= model_config["OU"],
                F = model_config["F"],
                block_method =model_config['projection_mode'],
                learning_rate = algo_config['learning_rate']
            )

    elif model_type == 'resnet':
        if model_config['Pulse'] == True:
            detector = ResNet_proj(
                symbol_bits = int(torch.log2(torch.tensor(len(MODULATIONS[channel_config['modulation']])))),
                num_users=channel_config['num_users'],
                num_antennas=channel_config['num_antennas'],
                num_layers=model_config['num_layers'],
                hidden_dim=model_config['hidden_dim'],
                cov_type= config['algorithm']['covariance_type'],
                init_cov_scale= model_config['init_param_cov'],
                obs_cov_scale= algo_config.get('obs_cov_scale', 0.1),
                dlr_rank= algo_config.get('dlr_rank', 10),
                Pulse=model_config["Pulse"],
                OU= model_config["OU"],
                F = model_config["F"],
                block_method =model_config['projection_mode'],
                learning_rate= algo_config['learning_rate'],
                learning_proj= config['projection']['learning proj'],
                learned_mat= config['model']['learned matrix'],
                learned_params=config['model']['learned param'],
                learned_A_path=A_path,
                learned_phi_path=phi_path,
                learned_params_path=param_path,
                pred_type= 'OU',
                proj_type= config['projection']['proj_type'],
                learned_DNN_path= dnn_path,
                pred_path=pred_path,
                update_type= config['model']['block_update'],
                F_style= model_config.get('F_style', 'full')
            )

        else:
            detector = ResNet_no_proj(
                symbol_bits = int(torch.log2(torch.tensor(len(MODULATIONS[channel_config['modulation']])))),
                num_users=channel_config['num_users'],
                num_antennas=channel_config['num_antennas'],
                num_layers=model_config['num_layers'],
                hidden_dim=model_config['hidden_dim'],
                cov_type= config['algorithm']['covariance_type'],
                init_cov_scale= model_config['init_param_cov'],
                obs_cov_scale= algo_config.get('obs_cov_scale', 0.1),
                dlr_rank= algo_config.get('dlr_rank', 10),
                Pulse=model_config["Pulse"],
                OU= model_config["OU"],
                F = model_config["F"],
                block_method =model_config['projection_mode'],
                learning_rate= algo_config['learning_rate'],
                learning_proj= config['projection']['learning proj'],
                learned_mat= config['model']['learned matrix'],
                learned_params=config['model']['learned param'],
                learned_A_path=A_path,
                learned_phi_path=phi_path,
                learned_params_path=param_path,
                pred_type= 'OU'
            )


    return detector


def create_channel(config):
    """create up-link Mimo channel"""
    channel_config = config['channel']
    channel = UplinkMimoChannel(
        path =channel_config['channel_path'],
        modulation_type=channel_config['modulation'],
        num_users=channel_config['num_users'],
        num_antennas=channel_config['num_antennas'],
        apply_non_linearity= not channel_config['linear_channel']
    )

    return channel

def create_online_train_fn(config):
    """create fn for online learning"""
    if config['algorithm']['method'].lower() == 'sgd':
        if config['model']['type'].lower() == 'resnet':
            online_fn = getattr(SGD, 'ResNet_SGD')
        else:
            online_fn = getattr(SGD, 'DeepSIC_SGD')
        return online_fn


    algo_config = config['algorithm']
    fn_name = 'Update'

    if algo_config['method'].lower() == 'bong':
        fn_name += '_BONG_lin'
    else:
        fn_name += '_BOG_lin'

    if algo_config['covariance_type'].lower() == 'dlr':
        fn_name += '_DLR'
    elif algo_config['covariance_type'].lower() == 'full':
        fn_name += '_full'
    else:
        fn_name += '_diag'

    if config['model']['OU'] == True or config['model']['Pulse'] == False:
        fn_name += '_OU'
    else:
        fn_name += '_F'

    if algo_config['method'].lower() == 'bog' and  algo_config['reparameterized'] == True:
        fn_name += '_reparm'

    if config['model']['Pulse'] == True and  algo_config['covariance_type'].lower() == "diag":
        fn_name += '_proj'

    if algo_config['method'].lower() == 'bong':
        if config['model']['type'].lower() == 'resnet' and config['model']['Pulse'] == True :
            fn_name += '_ResNet'
            online_fn =getattr(BONG_per_res_blck, fn_name)
        else:
            online_fn = getattr(BONG, fn_name)
    else:
        online_fn = getattr(BOG,fn_name)

    return online_fn

def evaluate_model(model, test_rx: torch.Tensor, test_labels: torch.Tensor) -> float:
    """
    Test model and return Bit Error Rate (BER).

    Args:
        model: DeepSIC-like model with soft_decode_batch method.
        test_rx (torch.Tensor): Received signals, shape (num_samples, rx_dim)
        test_labels (torch.Tensor): True bits, shape (num_samples, num_users, symbol_bits)

    Returns:
        float: Bit Error Rate (BER)
    """
    predictions = model.soft_decode_batch(test_rx)

    num_users = model.num_users
    symbol_bits = model.symbol_bits

    batch = predictions.shape[0]
    predictions = predictions.reshape(test_labels.shape)

    """num_users = model.num_users
    symbol_bits = model.symbol_bits
    predictions = predictions.reshape(-1, num_users, symbol_bits)"""

    predicted_bits = (predictions > 0.5).float()

    total_bits = test_labels.numel()
    bit_errors = (predicted_bits != test_labels).sum().item()
    ber = bit_errors / total_bits

    return ber

def save_results(config,results, base_dir,i=1):
    if config['model']["F"] == True:
        if config['model'].get('F_style', 'full') == "diag":
            type = "F_diag"
        else:
            type = "F"
    else:
        type = "OU"
    wanted = results.get('num_params', config['projection'].get('num_wanted', 0))
    run_name = (
        f"{config['model']['type']}_"
        f"{config['algorithm']['method']}_"
        f"cov-{config['algorithm']['covariance_type']}__"
        f"pulse-{config['model']['Pulse']}_"
        f"lm-{config['model']['learned matrix']}__"
        f"snr-{config['channel']['snr']}--"
        f"sqz-{wanted}__"
        f"run-{i}__"
        f"{type}"
    )

    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)


    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({
            "config": config,
            "results": results
        }, f, indent=4)

    print(f"[+] Results saved to {run_dir}")
    return run_dir


def measure_runtime(config,detector,step_fn,rx,y):
    """measure runtime of the model"""
    method = config['algorithm']['method'].lower()
    gc.collect()
    gc.disable()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        detector.train_batch(step_fn,rx,y,method,gamma=0.999)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
    finally:gc.enable()
    train_time = end_time-start_time

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        detector.soft_decode_batch(rx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
    finally:
        gc.enable()
    inference_time = end_time-start_time

    return train_time/config['experiment']['pilot_per_frame'],inference_time/config['experiment']['pilot_per_frame']


def run_experiment(config):
    """perform one experiment due to configuration"""
    # create channel
    channel = create_channel(config)
    # create model
    detector = create_model(config)
    # import online learning function
    step_fn = create_online_train_fn(config)

    method = config['algorithm']['method'].lower()

    # make dataset
    sync_frames = config['experiment']['sync_frames']
    track_frames = config['experiment']['track_frames']
    total_frames = sync_frames + track_frames

    sync_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['symbols_per_frame'],
        num_frames=sync_frames,
        snr=config['channel']['snr'],
        start_frame=500

    )

    track_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['pilot_per_frame'],
        num_frames=track_frames,
        snr=config['channel']['snr'],
        start_frame=sync_frames+500
    )

    test_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['test_dim'],
        num_frames=total_frames+3,
        snr=config['channel']['snr'],
        start_frame=500
    )

    test_dataloader_iter = iter(test_dataloader)

    #measure runtime


    # run sync frames+test each 3 frames.

    sync_ber = []
    for train_rx,train_label in  tqdm(sync_dataloader, total=sync_frames, leave=False, desc='Sync frames'):
        detector.train_batch(step_fn,train_rx,train_label,method,gamma=0.999)
        test_rx, test_labels = next(test_dataloader_iter)
        ber = evaluate_model(detector,test_rx,test_labels)
        sync_ber.append(ber)

    # update initial parameters for tracking phase
    #detector.update_initial_params()

    """for layer in detector.blocks:
        for user in layer:
            print (user.z_layers)"""
    # run train frames +test each frame.
    track_ber = []
    step=0
    for train_rx, train_label in tqdm(track_dataloader, total=track_frames, leave=False, desc='track frames'):
        step +=1
        # measure runtime
        if step == 15:
            training_time, inference_time = measure_runtime(config,detector, step_fn, train_rx, train_label)
            test_rx, test_label = next(test_dataloader_iter)
            ber = evaluate_model(detector, test_rx, test_label)
            track_ber.append(ber)
            continue
        detector.train_batch(step_fn, train_rx, train_label,method,gamma= 0.999)
        test_rx, test_label = next(test_dataloader_iter)
        ber = evaluate_model(detector,test_rx, test_label)
        track_ber.append(ber)

    # generate results

    track_mean_BER = torch.mean(torch.tensor(track_ber))
    sync_ber_mean_BER = torch.mean(torch.tensor(sync_ber))
    
    if config['model']['Pulse'] == False:
        if config['model']['type'].lower() == 'deepsic':
             num_params = sum(p.numel() for p in detector.blocks[0][0].base_model.parameters())
        else:
             num_params = sum(p.numel() for p in detector.base_model.parameters())
    else:
        num_params = config['projection'].get('num_wanted', 0)

    results = {
        "training_time": training_time,
        "inference_time": inference_time,
        "sync_ber_list": sync_ber,
        "track_ber_list": track_ber,
        "track_mean_ber": float(track_mean_BER),
        "sync_mean_ber": float(sync_ber_mean_BER),
        "num_params": num_params,
    }
    print(training_time)
    print(track_mean_BER,sync_ber_mean_BER)
    return results



def main():
    """main function to run experiment """
    torch.set_num_threads(1)
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
    validate_config(config)
    #config = clean_config(config)
    """config_hash = generate_config_hash(config)"""


    results = run_experiment(config)
    save_dir = "experiments/data/results"
    save = save_results(config=config,results=results,base_dir=save_dir)

    return



if __name__ =='__main__':
    main()





