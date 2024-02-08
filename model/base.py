import numpy as np
import torch
import torch.nn as nn
from utils import constants


class BaseNet(nn.Module):

    def __init__(self, apply_bias=True, predict_polar=True, filter_moving=True):

        super(BaseNet, self).__init__()

        self.criterion = nn.MSELoss()

        self.params = {"batch_size": 32, "shuffle": False, "num_workers": 0}

        self.apply_bias = apply_bias
        self.predict_polar = predict_polar
        self.filter_moving = filter_moving

    @staticmethod
    def calculate_x_r(x):
        x_delta = x[:, 1:, :] - x[:, :-1, :]
        x_mask = (x[:, :-1, :] > 0.0).float() * (x[:, 1:, :] > 0.0).float()
        x_delta = x_delta * x_mask
        x_delta = torch.cat([x_delta[:, :1, :], x_delta], dim=1).to(x_delta.device)
        x_r = (x_delta[:, :, ::2] ** 2 + x_delta[:, :, 1::2] ** 2) ** 0.5
        # x_angle = torch.atan2(x[:,:,::2], x[:,:,1::2])
        return x_r

    @staticmethod
    def polar_to_cart(x_polar):
        # x_polar_r = torch.relu(x_polar[:, :, :16])
        # x_polar_angle = torch.tanh(x_polar[:, :, 16:]) * np.pi
        x_cart = torch.zeros(x_polar.shape).to(x_polar.device)
        # x_cart[:,:,::2] = x_polar_r * torch.cos(x_polar_angle)
        # x_cart[:,:,1::2] = x_polar_r * torch.sin(x_polar_angle)
        x_cart[:, :, ::2] = x_polar[:, :, :16] * torch.cos(x_polar[:, :, 16:])
        x_cart[:, :, 1::2] = x_polar[:, :, :16] * torch.sin(x_polar[:, :, 16:])
        return x_cart

    @staticmethod
    def bias(x):
        x_bias = constants.TABLE_LENGTH / 2.0
        y_bias = constants.TABLE_WIDTH / 2.0
        x_norm = torch.zeros(x.shape).to(x.device)
        x_norm[:, :, ::2] = x[:, :, ::2] - x_bias
        x_norm[:, :, 1::2] = x[:, :, 1::2] - y_bias
        return x_norm

    @staticmethod
    # def filter(x, x_move):
    #     # x_move_bin = (x_move > 0.5).float()
    #     x_filtered = torch.zeros(x.shape).to(x.device)
    #     x_filtered[:, :, ::2] = x[:, :, ::2]
    #     x_filtered[:, :, 1::2] = x[:, :, 1::2]
    #     return x_filtered
    def filter(x, x_move):
        x_move_bin = (x_move > 0.5).float()
        x_filtered = torch.zeros(x.shape).to(x.device)
        x_filtered[:, :, ::2] = x[:, :, ::2] * x_move_bin
        x_filtered[:, :, 1::2] = x[:, :, 1::2] * x_move_bin
        return x_filtered

    def fw(self, x):
        raise NotImplementedError(
            "Abstract method 'fw' must be implemented by inheriting class"
        )

    # def forward(self, x, tgt):
    #     print("x shape ", x.shape)
    #     print("tgt shape ", tgt.shape)
    #     x_bias = self.bias(x) if self.apply_bias else x
    #     x_delta, x_move = self.fw(src=x_bias, tgt=tgt)
    #     # print("x delta", x_delta)
    #     # assert not torch.isnan(x_delta).any() and not torch.isnan(x_move).any()
    #     x_delta_cart = self.polar_to_cart(x_delta) if self.predict_polar else x_delta
    #     x_delta_filtered = self.filter(x_delta_cart, x_move) if self.filter_moving else x_delta_cart
    #     # print(x[:,-1:,:].shape, x_delta_filtered.shape)
    #     # print(x[:, -1:, :], x_delta_filtered)
    #     x_out = x[:,-1:,:] + x_delta_filtered

    #     return x_out

    def forward(self, x):
        x_bias = self.bias(x) if self.apply_bias else x
        x_delta, x_move = self.fw(x_bias)
        assert not torch.isnan(x_delta).any() and not torch.isnan(x_move).any()
        x_delta_cart = self.polar_to_cart(x_delta) if self.predict_polar else x_delta
        x_delta_filtered = (
            self.filter(x_delta_cart, x_move) if self.filter_moving else x_delta_cart
        )
        x_out = x + x_delta_filtered

        return x_out, x_move
