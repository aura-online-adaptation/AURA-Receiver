import torch
import torch.nn as nn

class base_model_generator(nn.Module):
    """ Define generic Base model class for neural network architectures."""
    def __init__(self,input_size,hidden_dim,output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size,32,bias = True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,output_size,bias=True)


    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class base_model_generator_ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim*2))  # fc1
            self.blocks.append(nn.Linear(hidden_dim*2, hidden_dim))  # fc2

        self.fc_out = nn.Linear(hidden_dim, output_dim)