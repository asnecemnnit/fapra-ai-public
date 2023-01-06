from libraries import *
from shadow_removal import shadow_remove
from utils.ball import Ball, Balls

def strategy_hue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]
    hue_median = np.median(hue)
    hue = np.where(np.abs(hue-hue_median) < 10, hue, 255)
    #hue = cv2.erode(hue, (5,5), iterations=1)
    hue = cv2.morphologyEx(hue, cv2.MORPH_CLOSE, (3,3), iterations=2)
    hue = cv2.Canny(hue, 100, 200)
    hue = cv2.dilate(hue, (3,3), iterations=3)

    return hue

def strategy_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = hsv[:,:,0] * 0.5 + hsv[:,:,2] * 0.5
    gray = gray.astype(np.uint8)
    return gray

def strategy_rgb(frame):
    # Convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.erode(gray, (3,3), iterations=2)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, (3,3), iterations=3)
    gray = cv2.Canny(gray, 85, 170)
    gray = cv2.dilate(gray, (3,3), iterations=5)

    return gray

def strategy_shadow(frame):
    shadow = shadow_remove(frame)
    # Convert to grayscale.
    gray = cv2.cvtColor(shadow, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, (3,3), iterations=1)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (5,5), iterations=1)
    #gray = cv2.Canny(gray, 150, 20)

    #gray = cv2.erode(gray, (3,3), iterations=1)
    #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, (5,5), iterations=1)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    #gray = 255-gray
    #_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray

def detect_circles(frame):
    gradient_hue = strategy_hue(frame)
    gradient_shadow = strategy_shadow(frame)
    gradient_rgb = strategy_rgb(frame)
    gray = np.maximum(np.maximum(gradient_shadow, gradient_hue*0.01), gradient_rgb*0)
    gray = gray.astype(np.uint8)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    cv2.imshow("Circle", gray)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 50, param1=25,
                                        param2=22, minRadius=20, maxRadius=36)
    balls = Balls()
    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # rmax = 0
        ravg = 0
        rsum = 0
        for pt in detected_circles[0]:
            a, b, r = pt[0], pt[1], pt[2]
            # print(r)

            # if r > rmax:
            #     rmax = r

            rsum = rsum + r

            #
            # # print(rmax)
            #
            # # Draw the circumference of the circle.
            # cv2.circle(frame, (a, b), ravg, (0, 255, 0), 2)
            # cv2.circle(frame, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            # cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
            # cv2.imshow("Detected Circle", frame)
        ravg = np.uint16(rsum / len(detected_circles[0]))
        # print(ravg)
        for pt in detected_circles[0]:
            a, b, r = pt[0], pt[1], pt[2]
            balls.add(Ball(x=a, y=b, radius=ravg))
            # # Draw the circumference of the circle.
            cv2.circle(frame, (a, b), ravg, (0, 255, 0), 2)
        balls.radius_avg = ravg
    return balls