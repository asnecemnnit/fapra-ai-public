import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation as animation
import time
import cv2
from utils.ball import *

matplotlib.use('tkagg')

#####     CLASSES FOR RENDERING     #####

class TableRendering:
    tableLength=1.94
    tableWidth=0.97
    tableBorderWidth=0.06
    holeRadius=0.04
    tableColour='lightseagreen'
    tableBorderColour='teal'
    holeColour='darkslategrey'
    
    def __init__(self):
        tableLength=1.94
        tableWidth=0.97
        tableBorderWidth=0.05
        holeRadius=0.01
        tableColour='lightseagreen'
        tableBorderColour='teal'
        holeColour='darkslategrey'

class BallRendering(Ball):
    ballRadius=0.0285
    ballDiameter=2*ballRadius
    ballPosition=(0.0885,0.0885)
    ballColour='w'
    ballExist=1
         
    def __init__(self,ball_type=None):
        self.ball_type=ball_type
        
    def __str__(self):
        return f"{self.ball_type.name}|x={self.x}, y={self.y}"

    def setColour(self,colour):
        self.ballColour=colour

    def setPosition(self,x,y):
        self.ballPosition=(x+0.0885,y+0.0885)
        self.x=x
        self.y=y
        if ((self.x<0) and (self.x>1.94)) or ((self.y<0) and (self.y>0.97)):
            self.ballExist=0
        else:
            self.ballExist=1


#####     RENDEIRNG SHOW FUNCTIONS  (Pass Table and Ball objects)    #####


def showTable(Table):
    boundary = plt.Rectangle((0, 0), Table.tableLength+(2*Table.tableBorderWidth), Table.tableWidth+(2*Table.tableBorderWidth), fc=Table.tableBorderColour)
    rectangle = plt.Rectangle((Table.tableBorderWidth, Table.tableBorderWidth), Table.tableLength, Table.tableWidth, fc=Table.tableColour)
    hole1 = plt.Circle((Table.holeRadius, Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    hole2 = plt.Circle((Table.holeRadius, Table.tableWidth+(2*Table.tableBorderWidth)-Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    hole3 = plt.Circle(((Table.tableLength+(2*Table.tableBorderWidth))/2, Table.tableWidth+(2*Table.tableBorderWidth)-Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    hole4 = plt.Circle((Table.tableLength+(2*Table.tableBorderWidth)-Table.holeRadius, Table.tableWidth+(2*Table.tableBorderWidth)-Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    hole5 = plt.Circle((Table.tableLength+(2*Table.tableBorderWidth)-Table.holeRadius,Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    hole6 = plt.Circle(((Table.tableLength+(2*Table.tableBorderWidth))/2, Table.holeRadius), Table.holeRadius, fc=Table.holeColour)
    plt.gca().add_patch(boundary)
    plt.gca().add_patch(rectangle)
    plt.gca().add_patch(hole1)
    plt.gca().add_patch(hole2)
    plt.gca().add_patch(hole3)
    plt.gca().add_patch(hole4)
    plt.gca().add_patch(hole5)
    plt.gca().add_patch(hole6)
    plt.xlim(0, Table.tableLength+(2*Table.tableBorderWidth))
    plt.ylim(0, Table.tableWidth + (2 * Table.tableBorderWidth))
    #print("showTable")

def showSolidBall(Ball):
    ball=plt.Circle(Ball.ballPosition, Ball.ballRadius, fc=Ball.ballColour)
    if(Ball.ballExist==1):
        plt.gca().add_patch(ball)
    #print("showSolidBall")

def showStripedBall(Ball):
    ball=plt.Circle(Ball.ballPosition, Ball.ballRadius, fc=Ball.ballColour)
    stripe = plt.Circle(Ball.ballPosition, 0.01, fc='w') #white centre for striped ball
    if(Ball.ballExist==1):
        plt.gca().add_patch(ball)
        plt.gca().add_patch(stripe)
    #print("showStripedBall")

def showFrame():
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

def renderMP4(ballPositions,file_name,frame_no,window,doMP4):

    ballPosition = [float(x) for x in ballPositions]

    if window==0:
        plt.title("Real Data")
    elif window==1:
        plt.title("Predicted output (LSTM)")
    elif window==2:
        plt.title("Predicted output (Linear)")
    elif window==3:
        plt.title("Predicted output (Transformer)")

    plt.axis([0,2.06,0,1.09])
    plt.axes().set_aspect('auto')
          
    #print(ballPosition)

    ballCB.setPosition(ballPosition[0],ballPosition[1])
    ballSY.setPosition(ballPosition[2],ballPosition[3])
    ballSB.setPosition(ballPosition[4],ballPosition[5])
    ballSR.setPosition(ballPosition[6],ballPosition[7])
    ballSV.setPosition(ballPosition[8],ballPosition[9])
    ballSO.setPosition(ballPosition[10],ballPosition[11])
    ballSG.setPosition(ballPosition[12],ballPosition[13])
    ballSM.setPosition(ballPosition[14],ballPosition[15])
    ballSBlk.setPosition(ballPosition[16],ballPosition[17])
    ballYS.setPosition(ballPosition[18],ballPosition[19])
    ballBS.setPosition(ballPosition[20],ballPosition[21])
    ballRS.setPosition(ballPosition[22],ballPosition[23])
    ballVS.setPosition(ballPosition[24],ballPosition[25])
    ballOS.setPosition(ballPosition[26],ballPosition[27])
    ballGS.setPosition(ballPosition[28],ballPosition[29])
    ballMS.setPosition(ballPosition[30],ballPosition[31])

    showFrame()

    if doMP4==1:
        plt.savefig(file_name+'_'+str(window)+'_'+str(frame_no)+'.png')
    else:
        plt.pause(0.01)
        plt.cla()


#--------------------------------------------------------------------------------------------------
#####                                   RENDERING MAIN                                        #####
#--------------------------------------------------------------------------------------------------

#Create Table and Ball objects first
table=TableRendering()
ballCB=BallRendering(BallType.white_solid)
ballSY=BallRendering(BallType.yellow_solid)
ballSB=BallRendering(BallType.blue_solid)
ballSR=BallRendering(BallType.red_solid)
ballSV=BallRendering(BallType.purple_solid)
ballSO=BallRendering(BallType.orange_solid)
ballSG=BallRendering(BallType.green_solid)
ballSM=BallRendering(BallType.brown_solid)
ballSBlk=BallRendering(BallType.black_solid)
ballYS=BallRendering(BallType.yellow_stripe)
ballBS=BallRendering(BallType.blue_stripe)
ballRS=BallRendering(BallType.red_stripe)
ballVS=BallRendering(BallType.purple_stripe)
ballOS=BallRendering(BallType.orange_stripe)
ballGS=BallRendering(BallType.green_stripe)
ballMS=BallRendering(BallType.brown_stripe)

#set ball colours

ballColours=['w','gold','midnightblue','red','indigo','darkorange','g','maroon','k','gold','midnightblue','red','indigo','darkorange','g','maroon']

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



#Final Rendering

"""

width = 1280 #MP4 #640 each
height = 480 #MP4
file_name='rendering' #MP4
OUTPUT_FILE = file_name+'.mp4' #MP4
fourcc = cv2.VideoWriter_fourcc('M','P','4','V') #MP4
writer = cv2.VideoWriter(OUTPUT_FILE, 
                         fourcc,
                         30, # fps
                         (width, height)) # resolution #MP4

file_input_path='dataset_1.csv'

with open(file_input_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    frame_no = 0 #MP4
    for row in csv_reader:

        #Render real data
        renderMP4(row,file_name,frame_no,0) #Call this function to render frame, pass row of 32 values
        real_window = cv2.imread(file_name+'_0_'+str(frame_no)+'.png') #MP4

        #Render predicted data (LSTM)
        renderMP4(row,file_name,frame_no,1) #Call this function to render frame, pass row of 32 values
        predicted_window = cv2.imread(file_name+'_1_'+str(frame_no)+'.png') #MP4

        #Render predicted data (Linear)
        #renderMP4(row,file_name,frame_no,2) #Call this function to render frame, pass row of 32 values
        #predicted_window = cv2.imread(file_name+'_2_'+str(frame_no)+'.png') #MP4

        #Concatenate two windows
        vis = np.concatenate((real_window, predicted_window), axis=1)
        cv2.imwrite(file_name+str(frame_no)+'.png', vis)
        display_window = cv2.imread(file_name+str(frame_no)+'.png')

        #cv2.imshow("COMPARE", display_window)
        cv2.waitKey(1) #MP4
        writer.write(display_window) #MP4

        frame_no = frame_no + 1 #MP4

    writer.release() #MP4


"""

#---------------------------------------------------------------------------------------------------
#####                                         END                                              #####
#---------------------------------------------------------------------------------------------------
