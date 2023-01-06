import pandas as pd
from numpy import *
from rendering.renderingMain import *

# showTable(table)
# plt.pause(100)
# plt.cla()


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

for i in range(shape(data)[0]):
    renderMP4(data[i], "xyz", i, 0, 0)