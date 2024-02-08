import os
import shutil
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation as animation
import time
import cv2
from utils.ball import *

matplotlib.use("tkagg")

#####     CLASSES FOR RENDERING     #####


class TableRendering:
    tableLength = 1.94
    tableWidth = 0.97
    tableBorderWidth = 0.06
    holeRadius = 0.04
    tableColour = "lightseagreen"
    tableBorderColour = "teal"
    holeColour = "darkslategrey"

    def __init__(self):
        tableLength = 1.94
        tableWidth = 0.97
        tableBorderWidth = 0.05
        holeRadius = 0.01
        tableColour = "lightseagreen"
        tableBorderColour = "teal"
        holeColour = "darkslategrey"


class BallRendering(Ball):
    ballRadius = 0.0285
    ballDiameter = 2 * ballRadius
    ballPosition = (0.0885, 0.0885)
    ballColour = "w"
    ballExist = 1

    def __init__(self, ball_type=None):
        self.ball_type = ball_type

    def __str__(self):
        return f"{self.ball_type.name}|x={self.x}, y={self.y}"

    def setColour(self, colour):
        self.ballColour = colour

    def setPosition(self, x, y):
        self.ballPosition = (float(x) + 0.0885, float(y) + 0.0885)

        self.x = float(x)
        self.y = float(y)
        if ((self.x < 0) and (self.x > 1.94)) or ((self.y < 0) and (self.y > 0.97)):
            self.ballExist = 0
        else:
            self.ballExist = 1


#####     RENDEIRNG SHOW FUNCTIONS  (Pass Table and Ball objects)    #####


def showTable(Table):
    boundary = plt.Rectangle(
        (0, 0),
        Table.tableLength + (2 * Table.tableBorderWidth),
        Table.tableWidth + (2 * Table.tableBorderWidth),
        fc=Table.tableBorderColour,
    )
    rectangle = plt.Rectangle(
        (Table.tableBorderWidth, Table.tableBorderWidth),
        Table.tableLength,
        Table.tableWidth,
        fc=Table.tableColour,
    )
    hole1 = plt.Circle(
        (Table.holeRadius, Table.holeRadius), Table.holeRadius, fc=Table.holeColour
    )
    hole2 = plt.Circle(
        (
            Table.holeRadius,
            Table.tableWidth + (2 * Table.tableBorderWidth) - Table.holeRadius,
        ),
        Table.holeRadius,
        fc=Table.holeColour,
    )
    hole3 = plt.Circle(
        (
            (Table.tableLength + (2 * Table.tableBorderWidth)) / 2,
            Table.tableWidth + (2 * Table.tableBorderWidth) - Table.holeRadius,
        ),
        Table.holeRadius,
        fc=Table.holeColour,
    )
    hole4 = plt.Circle(
        (
            Table.tableLength + (2 * Table.tableBorderWidth) - Table.holeRadius,
            Table.tableWidth + (2 * Table.tableBorderWidth) - Table.holeRadius,
        ),
        Table.holeRadius,
        fc=Table.holeColour,
    )
    hole5 = plt.Circle(
        (
            Table.tableLength + (2 * Table.tableBorderWidth) - Table.holeRadius,
            Table.holeRadius,
        ),
        Table.holeRadius,
        fc=Table.holeColour,
    )
    hole6 = plt.Circle(
        ((Table.tableLength + (2 * Table.tableBorderWidth)) / 2, Table.holeRadius),
        Table.holeRadius,
        fc=Table.holeColour,
    )
    plt.gca().add_patch(boundary)
    plt.gca().add_patch(rectangle)
    plt.gca().add_patch(hole1)
    plt.gca().add_patch(hole2)
    plt.gca().add_patch(hole3)
    plt.gca().add_patch(hole4)
    plt.gca().add_patch(hole5)
    plt.gca().add_patch(hole6)
    # print("showTable")


def showSolidBall(Ball):
    ball = plt.Circle(Ball.ballPosition, Ball.ballRadius, fc=Ball.ballColour)
    if Ball.ballExist == 1:
        plt.gca().add_patch(ball)
    # print("showSolidBall")


def showStripedBall(Ball):
    ball = plt.Circle(Ball.ballPosition, Ball.ballRadius, fc=Ball.ballColour)
    stripe = plt.Circle(
        Ball.ballPosition, 0.01, fc="w"
    )  # white centre for striped ball
    if Ball.ballExist == 1:
        plt.gca().add_patch(ball)
        plt.gca().add_patch(stripe)
    # print("showStripedBall")


def showFrame(ax):
    showTable(table)
    showSolidBall(ballCB)
    showSolidBall(ballSY)
    showSolidBall(ballSB)
    showSolidBall(ballSR)
    showSolidBall(ballSV)
    showSolidBall(ballSO)
    showSolidBall(ballSG)
    showSolidBall(ballSM)
    showSolidBall(ballSBlk)
    showStripedBall(ballYS)
    showStripedBall(ballBS)
    showStripedBall(ballRS)
    showStripedBall(ballVS)
    showStripedBall(ballOS)
    showStripedBall(ballGS)
    showStripedBall(ballMS)


##### Render function (Pass Ball Position, just call in final code)


def renderMP4(ballPositions, file_name, frame_no, axes, window, saveToDisk):
    ballPosition = [float(x) for x in ballPositions]

    # Set the positions of the balls for the specified subplot
    balls = [
        ballCB,
        ballSY,
        ballSB,
        ballSR,
        ballSV,
        ballSO,
        ballSG,
        ballSM,
        ballSBlk,
        ballYS,
        ballBS,
        ballRS,
        ballVS,
        ballOS,
        ballGS,
        ballMS,
    ]

    for ball, (x, y) in zip(balls, zip(ballPosition[::2], ballPosition[1::2])):
        ball.setPosition(x, y)

    # Show the frame with the balls in the appropriate subplot
    showFrame(axes)

    # Convert the plot to a numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    plot_img_np = np.array(fig.canvas.renderer._renderer)
    # Convert the numpy array to BGR format
    plt_cv = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)

    # If saving as MP4, save the figure
    if saveToDisk == 1:
        # Save image using OpenCV
        cv2.imwrite(
            file_name + "_" + str(window) + "_" + str(frame_no) + ".png", plt_cv
        )

    return plt_cv


# --------------------------------------------------------------------------------------------------
#####                                   RENDERING MAIN                                        #####
# --------------------------------------------------------------------------------------------------

# Create Table and Ball objects first
table = TableRendering()
ballCB = BallRendering(BallType.white_solid)
ballSY = BallRendering(BallType.yellow_solid)
ballSB = BallRendering(BallType.blue_solid)
ballSR = BallRendering(BallType.red_solid)
ballSV = BallRendering(BallType.purple_solid)
ballSO = BallRendering(BallType.orange_solid)
ballSG = BallRendering(BallType.green_solid)
ballSM = BallRendering(BallType.brown_solid)
ballSBlk = BallRendering(BallType.black_solid)
ballYS = BallRendering(BallType.yellow_stripe)
ballBS = BallRendering(BallType.blue_stripe)
ballRS = BallRendering(BallType.red_stripe)
ballVS = BallRendering(BallType.purple_stripe)
ballOS = BallRendering(BallType.orange_stripe)
ballGS = BallRendering(BallType.green_stripe)
ballMS = BallRendering(BallType.brown_stripe)

# set ball colours

ballColours = [
    "w",
    "gold",
    "midnightblue",
    "red",
    "indigo",
    "darkorange",
    "g",
    "maroon",
    "k",
    "gold",
    "midnightblue",
    "red",
    "indigo",
    "darkorange",
    "g",
    "maroon",
]

ballCB.setColour(ballColours[0])
ballSY.setColour(ballColours[1])
ballSB.setColour(ballColours[2])
ballSR.setColour(ballColours[3])
ballSV.setColour(ballColours[4])
ballSO.setColour(ballColours[5])
ballSG.setColour(ballColours[6])
ballSM.setColour(ballColours[7])
ballSBlk.setColour(ballColours[8])
ballYS.setColour(ballColours[9])
ballBS.setColour(ballColours[10])
ballRS.setColour(ballColours[11])
ballVS.setColour(ballColours[12])
ballOS.setColour(ballColours[13])
ballGS.setColour(ballColours[14])
ballMS.setColour(ballColours[15])


# Final Rendering

width = 1280  # MP4 #640 each
height = 480  # MP4
temp_folder_name = "rendering/temp/"
# Check if the folder exists
if os.path.exists(temp_folder_name):
    # If it exists, remove it and its contents
    shutil.rmtree(temp_folder_name)
    print(f"Folder '{temp_folder_name}' and its contents removed.")

# Create the folder
os.makedirs(temp_folder_name)
print(f"Folder '{temp_folder_name}' created.")
file_name = temp_folder_name + "rendering"  # MP4

video_folder_name = "rendering/video/"

# Check if the folder exists
if os.path.exists(video_folder_name):
    # If it exists, remove it and its contents
    shutil.rmtree(video_folder_name)
    print(f"Folder '{video_folder_name}' and its contents removed.")

# Create the folder
os.makedirs(video_folder_name)
print(f"Folder '{video_folder_name}' created.")
OUTPUT_FILE = video_folder_name + "rendering.mp4"  # MP4
fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "V")  # MP4
writer = cv2.VideoWriter(
    OUTPUT_FILE, fourcc, 30, (width, height)  # fps
)  # resolution #MP4

file_input_path = (
    "dataset/dataset_final_test/dataset_slow_30_strike_1_vertical_horizontal.csv"
)


with open(file_input_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    frame_no = 0  # MP4
    for row in csv_reader:
        # Create a figure with a subplot
        fig1, axes1 = plt.subplots(1, 1)
        axes1.set_xlim(0, 2.06)
        axes1.set_ylim(0, 1.09)
        axes1.set_aspect("equal")
        axes1.set_title("Real Data")

        # Render real data
        real_window = renderMP4(
            row, file_name, frame_no, axes1, window=0, saveToDisk=0
        )  # Call this function to render frame, pass row of 32 values
        plt.close(fig1)

        # real_window = cv2.imread(file_name + "_0_" + str(frame_no) + ".png")  # MP4

        # Create a figure with a subplot
        fig2, axes2 = plt.subplots(1, 1)
        axes2.set_xlim(0, 2.06)
        axes2.set_ylim(0, 1.09)
        axes2.set_aspect("equal")
        axes2.set_title("Predicted output (LSTM)")

        # Render predicted data (LSTM)
        predicted_window = renderMP4(
            row, file_name, frame_no, axes2, window=1, saveToDisk=0
        )  # Call this function to render frame, pass row of 32 values
        plt.close(fig2)

        # predicted_window = cv2.imread(file_name + "_1_" + str(frame_no) + ".png")  # MP4

        # print("Real window size:", real_window.shape)
        # print("Predicted window size:", predicted_window.shape)

        # Concatenate two windows
        vis = np.concatenate((real_window, predicted_window), axis=1)
        display_window = vis
        # cv2.imwrite(file_name + str(frame_no) + ".png", vis)
        # display_window = cv2.imread(file_name + str(frame_no) + ".png")

        writer.write(display_window)  # MP4

        if not ((frame_no + 1) % 10):
            print(f"frame {frame_no+1} processed")

        frame_no += 1  # MP4

print("all frames processed")
writer.release()  # MP4


# ---------------------------------------------------------------------------------------------------
#####                                         END                                              #####
# ---------------------------------------------------------------------------------------------------
