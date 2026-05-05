import torch

MODULATIONS = {
    "BPSK": torch.tensor([-1, 1], dtype=torch.float32),
    "QPSK": torch.tensor([complex(x, y) for y in [1, -1] for x in [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0)),
    "16-QAM": torch.tensor([complex(x, y) for x in [3, 1, -3, -1] for y in [3, 1, -3, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(10))
}