from libraries import *

def detect_corners_and_reshape(frame):
    operatedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)

    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 3, 0.04)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)
    # print(dest)

    # print (dest > 0.01 * dest.max())

    # Reverting back to the original image,
    # with optimal threshold value
    frame[dest > 0.01 * dest.max()] = [0, 0, 255]

    #TODO determine four corner coordinates
    coordinates = np.argwhere(dest > 0.01 * dest.max())
    # print(coordinates)
    corners = detect_four_corners(frame, coordinates)
    # print(corners)
    # print("hi")
    # corners = [[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]]
    # affine_transform(frame, corners)

def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def detect_four_corners(frame, coor):
    thresh = 250
    # print(coor)
    coor_list = [l.tolist() for l in list(coor)]

    coor_tuples = [tuple(l) for l in coor_list]
    coor_tuples_copy = coor_tuples

    i = 1
    for pt1 in coor_tuples:

        # print(' I :', i)
        for pt2 in coor_tuples[i::1]:
            # print(pt1, pt2)
            # print('Distance :', distance(pt1, pt2))
            if (distance(pt1, pt2) < thresh):
                coor_tuples_copy.remove(pt2)
        i += 1

    return coor_tuples


