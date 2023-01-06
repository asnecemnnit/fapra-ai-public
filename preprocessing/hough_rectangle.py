from libraries import *

gX=0
gY=0
gW=0
gH=0

def hough_rectangle(img, debug=False, color = "blue"):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "green":
        # Define range of green color in HSV
        lower = np.array([35, 50, 10])
        upper = np.array([80, 255, 255])
    else:
        # Define range of blue color in HSV
        lower = np.array([80, 50, 50])
        upper = np.array([130, 255, 255])

    # Threshold the HSV image to get mask with defined color range
    mask = cv2.inRange(hsv, lower, upper)

    # Apply mask to original image
    hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    hsv_value = hsv[:, :, 2]
    edges = cv2.Canny(hsv_value, 150, 200)
    hough_img = np.zeros(shape=edges.shape, dtype=np.uint8)
    # Results are marked through the dilated corners
    hough_img = cv2.dilate(hough_img, None)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        cv2.line(hough_img, (x1, y1), (x2, y2), 255, 1)

    #lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
    #for line in lines:
    #    x1, y1, x2, y2 = line[0]
    #    cv2.line(hough_img, (x1, y1), (x2, y2), 255, 2)

    contours, hierarchy = cv2.findContours(hough_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # print(f"{len(contours)} contours found")

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    min_area = 1e10
    min_rectangle = None
    for contour in contours:
        # Straight bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Skip rectangles that are too small
        if w < 600 or h < 300:
            continue
        ratio = w/h
        # print(ratio)
        # print(w,h)
        if ratio < 1.5 or ratio > 2.5:
            continue
        # Keep the smallest rectangle
        # print(min_area)
        # print(w*h)
        # print(ratio)

        if w*h < min_area:
            min_area = w*h
            min_rectangle = x, y, w, h
            # print(x,y,w,h)

    if debug:
        cv2.imshow("Canny", edges)
        cv2.imshow("Hough edges", hough_img)
        cv2.imshow("HSV Value", hsv_value)

    if min_rectangle != None:
        x, y, w, h = min_rectangle
        global gX, gY, gW, gH
        gX, gY, gW, gH = x, y, w, h
    # print (gX,gY,gW,gH)
    cv2.rectangle(img, (gX, gY), (gX + gW, gY + gH), (0, 0, 255), 4, 16)

    corners = [[gX, gY], [gX+gW, gY], [gX, gY+gH], [gX+gW, gY+gH]]
    return corners


# if __name__ == "__main__":
#     img = cv2.imread('../data/table_2.png')
#     x, y, w, h = hough_rectangle(img, debug=True)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4, 16)
#     cv2.imshow("Image", img)
#     cv2.waitKey()
