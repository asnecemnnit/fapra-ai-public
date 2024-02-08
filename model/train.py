import argparse
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from model.lstm import LSTMNet
from model.dataset import SequenceDataset, SyntheticDataset

# run on cpu or gpu?
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--gpu', action='store_true', help='Train on GPU')
args = parser.parse_args()
if args.gpu:
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)


# Hyperparameters
learning_rate = 0.0001

epochs = 100
batch_size = 32
input_dim = 32  # 16*2
output_dim = 32  # 16*2
num_frames_in = 15   # 300 frames = 5 seconds
num_frames_out = 15
use_gt = False
steps_gt = 5
time_sampling = 2
data_sampling = 1
latest_start = 30
apply_bias = True
predict_polar = True
filter_moving = True

model_name = 'lstm_rel'

path_model = 'model/savedModels/lstm_model_' + model_name

# path_dataset must lead to a folder with csv-files in it:
path_dataset = 'dataset/dataset_final_train'



##########################################################################

# read dataset
csv_files = []
for file in os.listdir(path_dataset):
    if file.endswith('.csv'):
        csv_files.append(os.path.join(path_dataset, file))
plays = []
for csv_file in csv_files:
    play = pd.read_csv(csv_file)
    play = np.array(play)
    plays.append(play)
random.seed(42)
random.shuffle(plays)

lstm = LSTMNet(apply_bias=apply_bias, predict_polar=predict_polar, filter_moving=filter_moving).to(device)
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)


def loss_mse(y, y_hat, y_move=None, y_hat_move=None, endpoint=False):
    # Assert format [B, 32]
    assert y_hat.shape == y.shape == (y.shape[0], y.shape[1], 32)
    if endpoint:
        y_hat = y_hat[:, -1, :]
        y = y[:, -1, :]
    loss = 0.0
    y_exist = (y >= 0).float()
    loss_pos_unreduced = F.mse_loss(y_hat, y, reduction='none')
    loss_pos_exist = loss_pos_unreduced * y_exist
    loss_pos = torch.mean(loss_pos_exist)
    loss += loss_pos
    if filter_moving and y_move is not None and y_hat_move is not None:
        loss_move = F.binary_cross_entropy(y_hat_move[:, -1, :], y_move[:, -1, :])
        loss += 0.001 * loss_move
    loss_distance = (loss_pos * 2) ** 0.5

    return loss, loss_distance


def calc_y_move(y):
    move_threshold = 0.05
    y_delta = y[:, :1] - y[:, -1:]
    y_delta = y_delta[:, :, :16] ** 2 + y_delta[:, :, 16:] ** 2
    y_move = (y_delta >= move_threshold ** 2).float()
    #ratio_move = y_move.sum() / torch.numel(y_move)
    y_move = y_move.repeat(1, y.shape[1], 1)
    # y_delta_all = np.concatenate((y_delta_all, y_delta.detach().cpu().numpy().flatten()))

    return y_move


train_val_ratio = 0.75
synthetic = False
size_synthetic_dataset = 1000
debug = True and not args.gpu

if synthetic:
    train = SyntheticDataset(num_frames_in, num_frames_out, relative=False, length=int(size_synthetic_dataset * train_val_ratio))
    val = SyntheticDataset(num_frames_in, num_frames_out, relative=False, length=int(size_synthetic_dataset * (1 - train_val_ratio)))
else:
    train = SequenceDataset(plays[:int(train_val_ratio * len(plays))], num_frames_in, num_frames_out,
                            time_sampling=time_sampling, data_sampling=data_sampling, latest_start=latest_start)
    val = SequenceDataset(plays[int(train_val_ratio * len(plays)):], num_frames_in, num_frames_out,
                          time_sampling=time_sampling, data_sampling=data_sampling, latest_start=latest_start)

train_losses = []
val_losses = []
loss_naive = 0.0
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    epoch_loss_train = 0.0
    epoch_loss_validation = 0.0

    data_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in data_loader:
        lstm.zero_grad()
        x = batch['x'].float().to(device)
        y = batch['y'].float().to(device)
        y_move = calc_y_move(y)

        frame_stack = torch.zeros((batch_size, num_frames_in + num_frames_out, 32), dtype=torch.float).to(device)
        y_hat_move = torch.zeros((batch_size, num_frames_out, 16), dtype=torch.float).to(device)
        for j in range(num_frames_in):
            frame_stack[:, j, :] = x[:, j, :]

        for i in range(num_frames_in, num_frames_in + num_frames_out):
            y_hat_seq, y_hat_move_seq = lstm.forward(frame_stack[:, i - num_frames_in:i, :])
            frame_stack[:, i, :] = y_hat_seq[:, num_frames_in - 1, :]
            y_hat_move[:, i - num_frames_in, :] = y_hat_move_seq[:, num_frames_in - 1, :]
            if use_gt and i - num_frames_in % steps_gt == 0:
                frame_stack[:, i, :] = y[:, i - num_frames_in, :]
        y_hat = frame_stack[:, num_frames_in:num_frames_in + num_frames_out, :]
        #if debug:
        #    batch_0 = np.concatenate((x[0].detach().numpy(), y_hat_0[0:1].detach().numpy()))
        if epoch == 0:  # loss for using the input as output
            _, loss_distance_naive = loss_mse(y, x[:, -1:, :].repeat(1, y.shape[1], 1))
            loss_naive += loss_distance_naive.item()
        loss, loss_distance = loss_mse(y, y_hat, y_move, y_hat_move)
        epoch_loss_train += loss_distance.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
        optimizer.step()
    epoch_loss_train /= len(data_loader)
    if epoch == 0:
        loss_naive /= len(data_loader)

    with torch.no_grad():
        data_loader_val = data.DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        for batch in data_loader_val:
            x = batch['x'].float().to(device)
            y = batch['y'].float().to(device)
            y_move = calc_y_move(y)

            frame_stack = torch.zeros((batch_size, num_frames_in + num_frames_out, 32), dtype=torch.float).to(device)
            y_hat_move = torch.zeros((batch_size, num_frames_out, 16), dtype=torch.float).to(device)
            for j in range(num_frames_in):
                frame_stack[:, j, :] = x[:, j, :]

            for i in range(num_frames_in, num_frames_in + num_frames_out):
                y_hat_seq, y_hat_move_seq = lstm.forward(frame_stack[:, i - num_frames_in:i, :])
                frame_stack[:, i, :] = y_hat_seq[:, num_frames_in - 1, :]
                y_hat_move[:, i - num_frames_in, :] = y_hat_move_seq[:, num_frames_in - 1, :]
                if i - num_frames_in % steps_gt == 0:
                    frame_stack[:, i, :] = y[:, i - num_frames_in, :]

            y_hat = frame_stack[:, num_frames_in:num_frames_in + num_frames_out, :]

            loss, loss_distance = loss_mse(y, y_hat, y_move, y_hat_move)
            epoch_loss_validation += loss_distance.item()

        epoch_loss_validation /= len(data_loader_val)
    print(f'loss_naive: {loss_naive}')
    print(f'epoch_loss_train: {epoch_loss_train}')
    print(f'epoch_loss_validation: {epoch_loss_validation}')
    print()

    train_losses.append(epoch_loss_train)
    val_losses.append(epoch_loss_validation)

    torch.save(lstm, f"{path_model}_in{num_frames_in}_out{num_frames_out}_ts{time_sampling}_ds{data_sampling}"
                     f"_gt{steps_gt*use_gt}_ls{latest_start}_bias{apply_bias}_pol{predict_polar}_mov{filter_moving}"
                     f"_lr{learning_rate}_e{epoch}_n{loss_naive:.4f}_t{epoch_loss_train:.4f}_v{epoch_loss_validation:.4f}.pt")

t = range(epochs)

plt.figure(1)
plt.plot(t, train_losses)
plt.savefig('losses_training_lstm.png')
plt.figure(2)
plt.plot(t, val_losses)

plt.savefig('losses_validation_lstm.png')
