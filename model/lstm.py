import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseNet

# Hyperparameters
hidden_dim = 64
number_layers = 1
linear_pre_dim = 32
linear_post_dim = 32


input_dim = 32  # 16*2
output_pos_dim = 32  # 16*2
output_move_dim = 16


# LSTM class
class LSTMNet(BaseNet):

    def __init__(self, apply_bias, predict_polar, filter_moving):

        super(LSTMNet, self).__init__(apply_bias=apply_bias, predict_polar=predict_polar, filter_moving=filter_moving)

        self.lin1 = nn.Linear(input_dim, linear_pre_dim)

        self.lstm = nn.LSTM(linear_pre_dim, hidden_dim, num_layers=number_layers, bidirectional=False, batch_first=True)

        self.lin2 = nn.Linear(hidden_dim, linear_post_dim)

        self.lin3 = nn.Linear(linear_post_dim, output_pos_dim)

        self.lin4 = nn.Linear(linear_post_dim, output_move_dim)

        self.criterion = nn.MSELoss()

        self.params = {'batch_size': 32,
                       'shuffle': False,
                       'num_workers': 0}

    def fw(self, x):
        # [1x32] -> [1x32]
        x = self.lin1(x)
        x = F.relu(x)
        # [1x32] -> [1x64]
        x, _ = self.lstm(x)
        # [1x64] -> [1x32]
        x = self.lin2(x)
        x = F.relu(x)
        # [1x32] -> [1x32]
        x_pos = self.lin3(x)  # TODO try tanh
        # [1x32] -> [1x16]
        x_exist = self.lin4(x)
        x_exist = torch.sigmoid(x_exist)

        return x_pos, x_exist
