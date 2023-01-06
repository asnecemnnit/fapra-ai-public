import torch.utils.data as data
import numpy as np
from utils.ball import Balls


class SequenceDataset(data.Dataset):
    def __init__(self, plays: list, num_frames_in: int, num_frames_out: int, relative: bool = False,
                 data_sampling: int = 1, time_sampling: int = 1, latest_start: int = 10000):
        """
        :param plays: List of nx32 numpy arrays providing time series of length n with 32 positions each.
        :param num_frames_in: Number of frames added to the x label.
        :param num_frames_out: Number of frames added to the y label.
        :param relative: Defines whether positions are provided absolute or relative to previous timepoint.
        :param data_sampling: Defines the number of frames between two datapoints in the dataset.
                              Reduces amount of similar data.
        :param time_sampling: Defines the number of frames between two timepoints in the time series of a datapoint.
                              Increases positions deltas, therefore the signal/noise ratio and the loss.
        :param latest_start: Defines the latest time step from which a sequence is sampled.
                             Avoids considering the end of sequences where only small motion happens.
        """
        super().__init__()
        assert all([val >= 1 for val in (num_frames_in, num_frames_out, data_sampling, time_sampling)])
        self.num_frames_in = num_frames_in
        self.num_frames_out = num_frames_out
        self.relative = relative
        self.data_sampling = data_sampling
        self.time_sampling = time_sampling
        self.frames_per_datapoint = (num_frames_in + num_frames_out) * time_sampling
        self.plays = []
        # List of datapoints as tuples (play_idx, frame_idx) pointing to the start frame of the datapoint
        self.x = []
        for play_idx, play in enumerate(plays):
            self.plays.append(play)
            num_samples = play.shape[0] - self.frames_per_datapoint
            # Add datapoint per each data_sampling number of frames
            for frame_idx in range(relative*time_sampling, min(num_samples+time_sampling, latest_start), data_sampling):
                self.x.append((play_idx, frame_idx))

    def __getitem__(self, idx):
        play_idx, start_frame_idx = self.x[idx]
        play = self.plays[play_idx]
        x = []
        y = []
        for frame_idx in range(start_frame_idx, start_frame_idx+self.frames_per_datapoint, self.time_sampling):
            if self.relative:
                positions = play[frame_idx] - play[frame_idx-self.time_sampling]
            else:
                positions = play[frame_idx]
            if frame_idx < start_frame_idx+self.num_frames_in*self.time_sampling:
                x.append(positions)
            else:
                y.append(positions)
        return {'x': np.array(x), 'y': np.array(y)}

    def __len__(self):
        return len(self.x)


class SyntheticDataset(data.Dataset):
    def __init__(self, num_frames_in: int, num_frames_out: int, relative: bool = True, time_sampling: int = 1, length=100, start_token=False):
        """
        :param num_frames_in: Number of frames added to the x label.
        :param num_frames_out: Number of frames added to the y label.
        :param relative: Defines whether positions are provided absolute or relative to previous timepoint.
        :param time_sampling: Defines the number of frames between two timepoints in the time series of a datapoint.
                              Increases positions deltas, therefore the signal/noise ratio and the loss.
        """
        super().__init__()
        assert all([val >= 1 for val in (num_frames_in, num_frames_out, time_sampling)])
        self.num_frames_in = num_frames_in
        self.num_frames_out = num_frames_out
        self.relative = relative
        self.time_sampling = time_sampling
        self.length = length
        self.frames_per_datapoint = (num_frames_in + num_frames_out) * time_sampling
        self.start_token = start_token

    def __getitem__(self, idx):
        if idx >= self.length:
            raise StopIteration

        sampling_offset = self.relative * self.time_sampling
        balls = Balls()
        balls.init_random(p_exist=0.8, p_motion=0.5, max_velocity=0.1)
        play = []
        for i in range(sampling_offset + self.frames_per_datapoint):
            balls.step()
            positions = balls.to_csv()[0]
            play.append(np.array(positions))
        x = []
        y = []
        if self.start_token:
            y.append(np.zeros(play[0].shape[0]))
        for frame_idx in range(sampling_offset, sampling_offset + self.frames_per_datapoint, self.time_sampling):
            if self.relative:
                positions = play[frame_idx] - play[frame_idx-self.time_sampling]
            else:
                positions = play[frame_idx]
            if frame_idx < sampling_offset+self.num_frames_in*self.time_sampling:
                x.append(positions)
            else:
                y.append(positions)
        return {'x': np.array(x), 'y': np.array(y)}

    def __len__(self):
        return self.length
