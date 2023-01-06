import argparse
import pandas as pd
from numpy import *
import torch
from rendering.renderingMain import *

################################ Parameters ####################################
################################################################################

model_name = 'transformer_model_transformer_rel_in15_out1_ts2_ds1_gt0_ls30_biasTrue_polTrue_movTrue_lr0.0001_e11_n0.0104_t0.0167_v0.0199'
number_input_frames = 15  # must be <= frames_without_prediction
frames_without_prediction = number_input_frames
frames_to_decoder = 5
predict_entire_clip = True  # predict entire clip with multistep (like with prediction_steps = frames in clip):
prediction_steps = 1  # number of steps in multistep prediction (1 for singlestep)

################################################################################
################################################################################

# run on cpu or gpu?
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--gpu', action='store_true', help='run on GPU')
args = parser.parse_args()
if args.gpu:
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)

# load model for prediction:
path_model = 'savedModels/' + model_name + '.pt'
transF = torch.load(path_model).to(device)
transF.eval()

for name, para in transF.named_parameters():
    print('{}: {}'.format(name, para.shape))

# select dataset which shall be used for prediction:
path_dataset_folder = '../dataset/dataset_final_test/'
name_dataset = 'dataset_test_clip_11_strike_1.csv'

# this one is in the training set !!!!!!! :
#path_dataset = '../dataset/dataset_final_train/dataset_clip1_28_strike_1_vertical.csv'
################################

path_dataset = path_dataset_folder + name_dataset

# read csv
data = pd.read_csv(path_dataset)
data = np.array(data)
path_videos = 'videos/'

if predict_entire_clip:
    OUTPUT_FILE = path_videos + model_name + '_' + 'entire_clip' + '_' + name_dataset +'.mp4' #MP4
else:
    OUTPUT_FILE = path_videos + model_name + '_' + str(prediction_steps) + '_' + name_dataset + '.mp4' #MP4

path_frames = 'frames_temp/'
filename_frame = path_frames + 'frame'
width = 1280 #MP4
height = 480 #MP4
fourcc = cv2.VideoWriter_fourcc('M','P','4','V') #MP4
writer = cv2.VideoWriter(OUTPUT_FILE,
                         fourcc,
                         30, # fps
                         (width, height)) # resolution #MP4

################################################################################
################################################################################
with torch.no_grad():
    print(shape(data)[0])
    # prediction loops:
    if predict_entire_clip:  # predict entire clip with multistep (prediction_steps = frames in clip):
        current_frames = np.zeros((number_input_frames, 32))
        for j in range(number_input_frames):
            current_frames[j] = data[frames_without_prediction-number_input_frames+j]
        print(shape(data)[0] - number_input_frames - 1)
        # for i in range(shape(data)[0] - number_input_frames - 1):
        for i in range(shape(data)[0] - 1):
            renderMP4(data[i], filename_frame, i, 0, 1)
            real_window = cv2.imread(filename_frame + '_0_' + str(i) + '.png')
            if i < frames_without_prediction:
                # print("render dummy")
                renderMP4(data[i], filename_frame, i, 3, 1)
            else:
                dimensions = shape(current_frames)
                dimensions = list(dimensions)
                print(dimensions)
                tensor_in_model = torch.zeros(dimensions)
                # loop for number of first number_input_frames frames
                for m in range(int(shape(current_frames)[0])):
                    # loop for 32 coordinates
                    for n in range(int(shape(current_frames)[1])):
                        tensor_in_model[m][n] = current_frames[m][n]
                print("input sequence range ", i - number_input_frames, i - frames_to_decoder)
                print("decode input sequence range ", i - frames_to_decoder - 1, i - 1)
                # pred_pos = transF.forward(tensor_in_model[None, i - number_input_frames : i - frames_to_decoder, :].to(device),
                #             tensor_in_model[None,  i - frames_to_decoder - 1 : i - 1, :].to(device))
                pred_pos = transF.forward(tensor_in_model[None, 0 : number_input_frames - frames_to_decoder, :].to(device),
                            tensor_in_model[None,  number_input_frames - frames_to_decoder - 1 : number_input_frames, :].to(device))
                # print(pred_pos[0])
                # predicted_frame = pred_pos.detach().cpu().numpy()
                predicted_frame = pred_pos[:, -1, :].cpu().numpy()
                # print(pred_pos)
                current_frames[0:current_frames.shape[0]-1] = current_frames[1:current_frames.shape[0]]
                # current_frames[current_frames.shape[0]-1] = predicted_frame[:, number_input_frames-1]
                current_frames[current_frames.shape[0]-1] = predicted_frame
                renderMP4(current_frames[number_input_frames-1], filename_frame, i, 3, 1)
            predicted_window = cv2.imread(filename_frame + '_3_' + str(i) + '.png')
            vis = np.concatenate((real_window, predicted_window), axis=1)
            cv2.imwrite(filename_frame + str(i) + '.png', vis)
            display_window = cv2.imread(filename_frame + str(i) + '.png')
            cv2.imshow("COMPARE", display_window)
            cv2.waitKey(1)  # MP4
            writer.write(display_window)  # MP4
        writer.release()  # MP4
    else:  # multistep with 'prediction_steps' steps:
        current_frames = np.zeros((number_input_frames, 32))
        temp_frames = np.zeros((number_input_frames, 32))
        for i in range(shape(data)[0] - number_input_frames - 1):
            renderMP4(data[i], filename_frame, i, 0, 1)
            real_window = cv2.imread(filename_frame + '_0_' + str(i) + '.png')
            if i < frames_without_prediction:
                # print("render dummy")
                renderMP4(data[i], filename_frame, i, 3, 1)
            else:
                for j in range(number_input_frames):
                    current_frames[j] = data[i - 1 - number_input_frames - prediction_steps + j]
                dimensions = shape(current_frames)
                dimensions = list(dimensions)
                tensor_in_model = torch.zeros(dimensions)
                for k in range(prediction_steps):
                    for m in range(int(shape(current_frames)[0])):
                        for n in range(int(shape(current_frames)[1])):
                            tensor_in_model[m][n] = current_frames[m][n]
                    # pred_pos, pred_exist = transF.forward(tensor_in_model[None, :, :].to(device))
                    pred_pos = transF.forward(
                        tensor_in_model[None, 0: number_input_frames - frames_to_decoder, :].to(device),
                        tensor_in_model[None, number_input_frames - frames_to_decoder - 1: number_input_frames,
                        :].to(device))
                    # print(pred_pos[0])
                    # predicted_frame = pred_pos.detach().cpu().numpy()
                    predicted_frame = pred_pos[:,-1,:].cpu().numpy()
                    current_frames[0:current_frames.shape[0] - 1] = current_frames[1:current_frames.shape[0]]
                    # current_frames[current_frames.shape[0] - 1] = predicted_frame[:, number_input_frames - 1]
                    current_frames[current_frames.shape[0] - 1] = predicted_frame
                renderMP4(current_frames[number_input_frames - 1], filename_frame, i, 3, 1)
            predicted_window = cv2.imread(filename_frame + '_3_' + str(i) + '.png')
            vis = np.concatenate((real_window, predicted_window), axis=1)
            cv2.imwrite(filename_frame + str(i) + '.png', vis)
            display_window = cv2.imread(filename_frame + str(i) + '.png')
            cv2.imshow("COMPARE", display_window)
            cv2.waitKey(1)  # MP4
            writer.write(display_window)  # MP4
        writer.release()  # MP4