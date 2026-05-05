import torch
import torch.nn as nn
from src.channel.Uplink_MIMO_Channel import UplinkMimoChannel
import json



def prepare_single_batch(channel: UplinkMimoChannel, num_samples: int, frame_idx: int, snr: int) -> tuple[torch.Tensor,torch.Tensor]:
    """Prepares online training experiment data for a single time frame.

    Args:
        channel (UplinkMimoChannel): Channel used to generate data.
        num_samples (int): Number of samples per time frame.
        frame_idx (int): Index of the time frame.
        snr (int): Signal-to-noise ratio.

    Return:
        rx (torch.tensor) : received input
        label_bits (torch.tensor) : real origin data transmit
    """
    num_users = channel.num_users
    bps = int(torch.log2(torch.tensor(channel.constellation_points.shape[0])))
    label_bits = torch.randint(
        low=0, high=2,
        size=(num_samples, num_users, bps),
        dtype=torch.int32
    )
    labels = label_bits[..., 0]
    for i in range(1, label_bits.shape[-1]):
        labels = labels * 2 + label_bits[..., i]
    rx = channel.transmit(s=labels, snr=snr, frame_idx=frame_idx)
    if channel.modulation_type != 'BPSK':
        rx = torch.stack([torch.real(rx), torch.imag(rx)], dim=-1)
    rx = rx.reshape(num_samples, -1)
    label_bits_f = label_bits.float()

    b0 = label_bits_f[..., 0]
    b1 = label_bits_f[..., 1]

    pair0 = torch.stack([1 - b0, b0], dim=-1)  # shape: (N, U, 2)
    pair1 = torch.stack([1 - b1, b1], dim=-1)

    labels_transformed = torch.cat([pair0, pair1], dim=-1)

    return rx, labels_transformed



def prepare_experiment_data(channel: UplinkMimoChannel, num_samples: int, num_frames: int, snr: int, start_frame: int = 0) -> torch.utils.data.DataLoader:
    """Prepares data for online training experiments.

    Args:
        channel (UplinkMimoChannel): Channel used to generate data.
        num_samples (int): Number of samples per time frame.
        num_frames (int): Number of time frames.
        snr (int): Signal-to-noise ratio.
        start_frame (int): Starting frame index for data generation. Defaults is 0.

    Returns:
        torch.utils.data.DataLoader: DataLoader containing the prepared data.
    """
    label_blocks = torch.zeros((0, 0))
    receive_blocks = torch.zeros((0, 0))
    for frame_idx in range(start_frame, start_frame + num_frames):
        rx, labels = prepare_single_batch(channel, num_samples, frame_idx, snr)
        label_blocks = labels if frame_idx == start_frame else torch.cat([label_blocks, labels], dim=0)
        receive_blocks = rx if frame_idx == start_frame else torch.cat([receive_blocks, rx], dim=0)

    dataset = torch.utils.data.TensorDataset(receive_blocks, label_blocks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)
    return dataloader