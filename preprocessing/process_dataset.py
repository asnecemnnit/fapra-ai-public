# pip install pyyaml
import yaml
import csv
import os
import re
import argparse
from main import preprocess, track

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--60fps",
                    help="process video as if it contains 60 frames per second",
                    action="store_true", default=False)
args = vars(ap.parse_args())

# clip is distorted
if not args.get("60fps", False):
    clip_is_60fps = False
# otherwise not distortion
else:
    clip_is_60fps = True

data_root = os.path.abspath("../data/clips60fps")
dataset_root = os.path.abspath("../dataset/")
with open(os.path.join(dataset_root, "dataset.yaml")) as f:
    dataset = yaml.safe_load(f)
    for video, metadata in dataset.items():
        for strike_i, strike_data in metadata.items():
            is_valid = strike_data['valid']
            start_frame = strike_data['start_frame']
            if is_valid:
                print(f"Process video {video} strike {strike_i}")
                video_path = os.path.join(data_root, video)
                balls_lst, frame_lst = preprocess(video_path, start_frame)
                csv_lst, csv_lst_horizontal, csv_lst_vertical, csv_lst_horizontal_vertical = track(balls_lst, frame_lst)

                # Write to CSV
                csv_path = os.path.join(dataset_root, f"dataset_{video.split('.')[0]}_strike_{strike_i}.csv")
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    if clip_is_60fps:
                        row_is_even = True
                        for csv_row in csv_lst:
                            if row_is_even:
                                w.writerow(csv_row)
                            row_is_even = not row_is_even
                    else:
                        for csv_row in csv_lst:
                            w.writerow(csv_row)

                # Write to CSV (horizontal mirrored dataset)
                csv_horizontal_path = re.sub(r'\.csv$', '', csv_path) + "_horizontal.csv"
                with open(csv_horizontal_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    if clip_is_60fps:
                        row_is_even = True
                        for csv_row in csv_lst_horizontal:
                            if row_is_even:
                                w.writerow(csv_row)
                            row_is_even = not row_is_even
                    else:
                        for csv_row in csv_lst_horizontal:
                            w.writerow(csv_row)

                # Write to CSV (vertical mirrored dataset)
                csv_vertical_path = re.sub(r'\.csv$', '', csv_path) + "_vertical.csv"
                with open(csv_vertical_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    if clip_is_60fps:
                        row_is_even = True
                        for csv_row in csv_lst_vertical:
                            if row_is_even:
                                w.writerow(csv_row)
                            row_is_even = not row_is_even
                    else:
                        for csv_row in csv_lst_vertical:
                            w.writerow(csv_row)

                # Write to CSV (horizontal & vertical mirrored dataset)
                csv_horizontal_vertical_path = re.sub(r'\.csv$', '', csv_path) + "_vertical_horizontal.csv"
                with open(csv_horizontal_vertical_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    if clip_is_60fps:
                        row_is_even = True
                        for csv_row in csv_lst_horizontal_vertical:
                            if row_is_even:
                                w.writerow(csv_row)
                            row_is_even = not row_is_even
                    else:
                        for csv_row in csv_lst_horizontal_vertical:
                            w.writerow(csv_row)