from .Modulation  import MODULATIONS
import os
import torch
import scipy.io


class UplinkMimoChannel:
    """Memory-less time-varying uplink MIMO channel.

    Args:
        path (str): Path to .mat file containing the channel matrices.
        modulation_type (str): Modulation type, e.g., "BPSK", "QPSK", "16-QAM".
        num_users (int): Number of single-antenna users.
        num_antennas (int): Number of receive antennas.
        apply_non_linearity (bool, optional): Whether to apply non-linearity to the received signal. Defaults to False.
    """

    def __init__(
            self,
            path: str,
            modulation_type: str,
            num_users: int,
            num_antennas: int,
            apply_non_linearity: bool = False,
        ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        self.path = path
        if modulation_type not in MODULATIONS.keys():
            raise ValueError(f"Modulation type must be one of {MODULATIONS.keys()}.")
        self.modulation_type = modulation_type
        self.constellation_points = MODULATIONS[self.modulation_type]
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.apply_non_linearity = apply_non_linearity
        self.load_mat_file()

    def load_mat_file(self):
        h_np = scipy.io.loadmat(self.path)['H']
        self.h = torch.from_numpy(h_np).to(torch.complex64)
        self.h = self.h[:self.num_users, :self.num_antennas, :].permute(2, 0, 1)
        self.num_frames = self.h.shape[0]

    @staticmethod
    def _compute_channel_signal_convolution(h: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """Compute the convolution of a channel matrix and an tensor of constellation points."""
        conv = tx @ h
        return conv

    def transmit(self, s: torch.Tensor, snr: float, frame_idx: int = 0) -> torch.Tensor:
        """Simulate transmission of symbols.

        Args:
            key (Array): Random key for noise generation.
            s (Array): Symbols to be transmitted.
            snr (float): Signal-to-noise ratio.
            frame_idx (int, optional): Time frame index. Defaults to 0.
        """
        h = self.h[frame_idx]
        tx = self.constellation_points[s].reshape(-1, self.num_users)
        conv = self._compute_channel_signal_convolution(h, tx)
        var = 10 ** (-0.1 * snr)
        random_noise = torch.sqrt(torch.tensor(var)) / 2 * torch.randn((2, tx.shape[0], self.num_antennas))
        w = random_noise[0] + 1j * random_noise[1]
        y = conv + w

        if self.apply_non_linearity:
            y = torch.tanh(0.5 * y.real) + 1j * torch.tanh(0.5 * y.imag)
            #ratio = torch.mean(torch.abs(torch.tanh(0.5 * y.real) + 1j * torch.tanh(0.5 * y.imag))) / torch.mean(torch.abs(conv))
            #print(ratio.item())
            #ratio = torch.mean(torch.abs(alpha * conv * torch.abs(conv) ** 2)) / torch.mean(torch.abs(conv))
            #print(ratio.item())
            #alpha = 0.5
            #y = y + alpha * y * torch.abs(y)**2
        return y


