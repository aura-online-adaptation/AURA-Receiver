import json
from copy import deepcopy
import os

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def validate_config(config):
    """Validate that the config contains all required sections and parameters."""
    required_sections = ['model', 'channel', 'experiment', 'algorithm']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    algo_required = ['method', 'dynamics_decay']
    for param in algo_required:
        if param not in config['algorithm']:
            raise ValueError(f"Missing required algorithm parameter: {param}")

    model_required = ['type', 'num_layers', 'hidden_dim']
    for param in model_required:
        if param not in config['model']:
            raise ValueError(f"Missing required model parameter: {param}")

    channel_required = ['modulation', 'snr', 'num_users', 'num_antennas']
    for param in channel_required:
        if param not in config['channel']:
            raise ValueError(f"Missing required channel parameter: {param}")

    exp_required = ['sync_frames', 'track_frames', 'symbols_per_frame', 'pilot_per_frame', 'test_dim', 'seed']
    for param in exp_required:
        if param not in config['experiment']:
            raise ValueError(f"Missing required experiment parameter: {param}")
    return

def clean_config(config):
    """Remove unused parameters based on the method type."""
    method = config['algorithm']['method'].lower()
    model_type = config['model']['type'].lower()
    cleaned_config = deepcopy(config)

    # Method-specific parameter requirements
    method_params = {
        'bog': {
            'required': {'covariance_type', 'learning_rate'},
            'unused': {'batch_size'}
        },
        'bong': {
            'required': {'covariance_type'},
            'unused': {'learning_rate', 'num_iter', 'batch_size'}
        },
    }
    method_spec = method_params.get(method, {})
    required_params = method_spec.get('required', set())
    unused_params = method_spec.get('unused', set())

    # Handle special cases
    if cleaned_config['algorithm'].get('linplugin', False) and cleaned_config['algorithm'].get('num_samples', 1) != 1:
        # For linplugin, num_samples must be 1
        print(f"Warning: linplugin does not require sampling, setting num_samples to 1")
        cleaned_config['algorithm']['num_samples'] = 1


    if model_type == 'resnet' and cleaned_config['algorithm'].get('covariance_type', None) == 'full':
        # Full covariance methods will run out of memory for ResNet model
        print(
            f"Warning: Full covariance methods will run out of memory for ResNet model, setting covariance_type to diag")
        cleaned_config['algorithm']['covariance_type'] = 'diag'

    if not (cleaned_config['algorithm'].get('linplugin', True) or cleaned_config['algorithm'].get('empirical_fisher',
                                                                                                  True)):
        # We don't allow MC-based methods gradients without empirical fisher due to computational complexity
        print(f"Warning: Running without linplugin requires empirical fisher, setting empirical_fisher to true")
        cleaned_config['algorithm']['empirical_fisher'] = True

    # Remove all unused parameters
    for param in unused_params:
        if param in cleaned_config['algorithm']:
            cleaned_config['algorithm'].pop(param)

    # Check that all required parameters are present
    for param in required_params:
        if param not in cleaned_config['algorithm']:
            raise ValueError(f"Missing required parameter: {param}")

    return cleaned_config
    return

