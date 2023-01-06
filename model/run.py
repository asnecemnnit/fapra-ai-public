import argparse
import pandas as pd
from numpy import *
import torch
from rendering.renderingMain import *

################################ Parameters ####################################
################################################################################

model_name = 'lstm_model_lstm_rel_in15_out15_ts2_ds1_gt0_ls30_biasTrue_polTrue_movTrue_lr0.0001_e8_n0.0703_t0.0633_v0.0635'
number_input_frames = 15  # must be <= frames_without_prediction
frames_without_prediction = number_input_frames
predict_entire_clip = True  # predict entire clip with multistep (like with prediction_steps = frames in clip):
prediction_steps = 10  # number of steps in multistep prediction (1 for singlestep)

################################################################################
################################################################################

# run on cpu or gpu?
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--gpu', action='store_true', help='run on GPU')
args = parser.parse_args()
if args.gpu:
    device_name = 'gpu'
else:
    device_name = 'cpu'
device = torch.device(device_name)

# load model for prediction:
path_model = 'savedModels/' + model_name + '.pt'
lstm = torch.load(path_model).to(device)
lstm.eval()

# select dataset which shall be used for prediction:
path_dataset_folder = '../dataset/dataset_final_test/'
name_dataset = 'dataset_slow_30_strike_1.csv'

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

# prediction loops:
if predict_entire_clip:  # predict entire clip with multistep (prediction_steps = frames in clip):
    current_frames = np.zeros((number_input_frames, 32))
    for j in range(number_input_frames):
        current_frames[j] = data[frames_without_prediction-number_input_frames+j]
    for i in range(shape(data)[0] - number_input_frames - 1):
        renderMP4(data[i], filename_frame, i, 0, 1)
        real_window = cv2.imread(filename_frame + '_0_' + str(i) + '.png')
        if i < frames_without_prediction:
            renderMP4(data[i], filename_frame, i, 1, 1)
        else:
            dimensions = shape(current_frames)
            dimensions = list(dimensions)
            tensor_in_model = torch.zeros(dimensions)
            for m in range(int(shape(current_frames)[0])):
                for n in range(int(shape(current_frames)[1])):
                    tensor_in_model[m][n] = current_frames[m][n]
            pred_pos, pred_exist = lstm.forward(tensor_in_model[None, :, :].to(device))
            predicted_frame = pred_pos.detach().numpy()
            current_frames[0:current_frames.shape[0]-1] = current_frames[1:current_frames.shape[0]]
            current_frames[current_frames.shape[0]-1] = predicted_frame[:, number_input_frames-1]
            renderMP4(current_frames[number_input_frames-1], filename_frame, i, 1, 1)
        predicted_window = cv2.imread(filename_frame + '_1_' + str(i) + '.png')
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
            renderMP4(data[i], filename_frame, i, 1, 1)
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
                pred_pos, pred_exist = lstm.forward(tensor_in_model[None, :, :].to(device))
                predicted_frame = pred_pos.detach().numpy()
                current_frames[0:current_frames.shape[0] - 1] = current_frames[1:current_frames.shape[0]]
                current_frames[current_frames.shape[0] - 1] = predicted_frame[:, number_input_frames - 1]
            renderMP4(current_frames[number_input_frames - 1], filename_frame, i, 1, 1)
        predicted_window = cv2.imread(filename_frame + '_1_' + str(i) + '.png')
        vis = np.concatenate((real_window, predicted_window), axis=1)
        cv2.imwrite(filename_frame + str(i) + '.png', vis)
        display_window = cv2.imread(filename_frame + str(i) + '.png')
        cv2.imshow("COMPARE", display_window)
        cv2.waitKey(1)  # MP4
        writer.write(display_window)  # MP4
    writer.release()  # MP4
    """
    current_yhat = 0
    for p in range(shape(data)[0]):
        if p < frames_without_prediction:
            renderMP4(data[p], filename_frame, p)
            #yhat = lstm.forward(data[0:i][None, :, :])
            #current_yhat = yhat
        else:
            in_model = data[p-number_input_frames:p][None, :, :]
            dimensions = shape(in_model)
            dimensions = list(dimensions)
            #print(dimensions)
            tensor_in_model = torch.zeros(dimensions)
            for m in range(int(shape(in_model)[0])):
                for n in range(int(shape(in_model)[1])):
                    counter = 0
                    for i in range(int(shape(in_model)[2])):
                        tensor_in_model[m][n][i] = in_model[m][n][i]/100
            #print(tensor_in_model)
            yhat = lstm.forward(tensor_in_model)
            #print(yhat)
            #yhat = lstm.forward(data[p-number_prediction_frames:p][None, :, :])

            current_yhat = yhat
            for_rendering_in_cm = current_yhat *100
            renderMP4(for_rendering_in_cm[0,number_input_frames-1].detach().numpy(), filename_frame, p)
        myfig = cv2.imread(filename_frame + str(p) + '.png')  # MP4
        cv2.waitKey(1)  # MP4
        writer.write(myfig)  # MP4
    writer.release()  # MP4
    """
    # old loop
    """
    # render first 30 frames:
    frames_without_prediction = 30
    current_frames = np.zeros((number_prediction_frames, 32))
    print(current_frames)
    #print(current_frames.shape)
    for j in range(number_prediction_frames):
        current_frames[j] = data[frames_without_prediction-1-number_prediction_frames+j]
    print(current_frames)

    for i in range(shape(data)[0] - number_prediction_frames - 1):
    #for i in range(35):
        if(i < frames_without_prediction):
            #render(data[i])
            renderMP4(data[i], filename_frame, i)
        else:
            predicted_frame = lstm.forward(current_frames[None,:,:])[0].detach().numpy()
            print(predicted_frame)
            current_frames[0:current_frames.shape[0]-1] = current_frames[1:current_frames.shape[0]]
            current_frames[current_frames.shape[0]-1] = predicted_frame[number_prediction_frames-1]
            #print(current_frames)
            #render(current_frames[number_prediction_frames-1])
            renderMP4(current_frames[number_prediction_frames-1], filename_frame, i)
        myfig = cv2.imread(filename_frame + str(i) + '.png')  # MP4
        cv2.waitKey(1)  # MP4
        writer.write(myfig)  # MP4
    """

"""
else:
    for i in range(data.shape[0]):
        renderMP4(data[i], filename_frame, i)
        myfig = cv2.imread(filename_frame + str(i) + '.png')  # MP4
        cv2.waitKey(1)  # MP4
        writer.write(myfig)  # MP4
"""







# old auxiliary functions:
"""
def forward(x):
    dimensions = shape(x)
    augmented_input = torch.zeros((1, dimensions[0], dimensions[1]))

    output = lstm.forward(augmented_input)

    # dummy forward function:
    #output = np.array(x[4])
    #for i in range(shape(output)[0]):
    #    output[i]= 2 * output[i]
    #output = list(output)

    return output

def predict(input_frames):
    current_frames = []
    current_frames = input_frames
    for i in range(number_prediction_frames - 1):
        temp_frames = current_frames
        temp_frame = forward(temp_frames)
        current_frames = current_frames[1:len(current_frames)]
        current_frames.append(temp_frame)
        print('current_frames:')
        print(current_frames)
    predicted_frame = forward(current_frames)
    print(predicted_frame)
    #showFrame(predicted_frame)
"""
"""
frame= [1,1,1,1,1]
frames = [frame,frame,frame,frame,frame]
print('frames:')
print(frames)

predict(frames)
"""
