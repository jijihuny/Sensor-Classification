import torch.nn as nn
import torch
from torch.nn import Module, Sequential

class Conv1dClassfier(Module):
    def __init__(self, filter_size: int = 8, frame_length: int = 8):
        super(Conv1dClassfier, self).__init__()

        self.layer_1 = Sequential(*[
            nn.Conv1d(5, filter_size, 3, padding=1),
            nn.BatchNorm1d(filter_size),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ])

        self.layer_2 = Sequential(*[
            nn.Conv1d(filter_size, 2 * filter_size, 3, padding=1),
            nn.BatchNorm1d(2 * filter_size),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ])

        self.fc = Sequential(*[
            nn.Flatten(),
            nn.Linear(int((frame_length * filter_size) / 2), 7),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        ])

        self.drop = nn.Dropout()

    def forward(self, input: torch.FloatTensor):
        out_1 = self.layer_1(input.transpose(1, 2))
        out_2 = self.layer_2(
            self.drop(out_1) if self.training else out_1
        )
        return self.fc(out_2)