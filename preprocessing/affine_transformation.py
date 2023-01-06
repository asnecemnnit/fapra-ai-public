import numpy as np
import cv2
import utils.constants as constants


def affine_transform(frame, corners):
    rows, cols, ch = frame.shape
    # print(corners)
    # pts1 = np.float32(corners)
    # pts1 = np.float32([[61, 51], [912, 53], [57, 433], [915, 434]])
    pts2 = np.float32([[0, 0], [int(constants.TABLE_LENGTH * 1000), 0], [0, int(constants.TABLE_WIDTH * 1000)],
                       [int(constants.TABLE_LENGTH * 1000), int(constants.TABLE_WIDTH * 1000)]])
    pts1 = np.float32([[80, 74], [1198, 76], [76, 642], [1207, 645]])
    # print(pts1)
    # pts2 = np.float32([[0, 0], [1500, 0], [0, 1500]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (int(constants.TABLE_LENGTH * 1000), int(constants.TABLE_WIDTH * 1000)))

    # M = cv2.getAffineTransform(pts1, pts2)

    # dst = cv2.warpAffine(frame, M, (1500, 1500))
    # cv2.imshow("affine", dst)
    # return dst
    return dst

def ball_in_the_hole(x, y, r):
    # left upper hole
    if x < r and y > constants.TABLE_WIDTH - r:
        return True
    # middle upper hole
    elif constants.TABLE_LENGTH/2 - r < x < constants.TABLE_LENGTH/2 + r and y > constants.TABLE_WIDTH - r:
        return True
    # right upper hole
    elif x > constants.TABLE_LENGTH - r and y > constants.TABLE_WIDTH - r:
        return True
    # left lower hole
    elif x < r and y < r:
        return True
    # middle lower hole
    elif constants.TABLE_LENGTH/2 - r < x < constants.TABLE_LENGTH/2 + r and y < r:
        return True
    # right lower hole
    elif x > constants.TABLE_LENGTH - r and y < r:
        return True
    # not in the hole
    else:
        return False

def getTableCoordinates(x, y, r):
    x = float(x / 1000)
    y = float(y / 1000)
    y = constants.TABLE_WIDTH - y
    r = float(r / 1000)
    # Store upto two decimal places
    x = float("{:.3f}".format(x))
    y = float("{:.3f}".format(y))
    r = float("{:.3f}".format(r))

    if ball_in_the_hole(x, y, r):
        x = float("{:.3f}".format(-0.1))
        y = float("{:.3f}".format(-0.1))

    return x, y, r


def convert_to_table_coordinates(balls):
    for ball in balls:
        ball.x, ball.y, ball.radius = getTableCoordinates(ball.x, ball.y, ball.radius)
    return balls
