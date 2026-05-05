import torch
import torch.nn as nn
import torch.nn.functional as F

"""class DNNProjection(torch.nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(DNNProjection, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size,bias=True)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

        nn.init.zeros_(self.layer2.bias)
    def forward(self, x):
        x = self.layer1(x)
        x = F.leaky_relu(x,0.1,True)
        x=self.layer2(x)

        return x"""

class DNNProjection(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNNProjection, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, output_size, bias=True)
        self.layer2  = torch.nn.Linear(input_size, output_size)
        torch.nn.init.normal_(self.layer2.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        linear_base = self.layer1(x)
        non_lin = self.layer2(x)
        non_lin = F.leaky_relu(non_lin,0.1,True)

        return linear_base + non_lin
