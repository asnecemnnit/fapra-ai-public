import cv2
from shadow_removal import shadow_remove

window_title = 'Parametrize'

global dilate_val, blur_val, erode_val, open_val, frame_raw, initialized
dilate_val = 4
blur_val = 7
erode_val = 0
open_val = 0
initialized = False

def parametrize(frame):
    global initialized
    if not initialized:
        parametrize_init()
        initialized = True
    global frame_raw
    frame_raw = frame

def process_img():
    global frame_raw
    processed = shadow_remove(frame_raw, dilate_size=dilate_val, blur_size=blur_val)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    # processed = cv2.Canny(processed, 100, 200)
    processed = cv2.erode(processed, (3,3), iterations=erode_val)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, (5,5), iterations=open_val)
    cv2.imshow(window_title, processed)

def on_trackbar_dilate(val):
    global dilate_val
    dilate_val = int(val)
    process_img()

def on_trackbar_blur(val):
    global blur_val
    blur_val = int(val)
    process_img()

def on_trackbar_erode(val):
    global erode_val
    erode_val = int(val)
    process_img()

def on_trackbar_open(val):
    global open_val
    open_val = int(val)
    process_img()

def parametrize_init():
    cv2.namedWindow(window_title)
    cv2.createTrackbar("Dilate size", window_title, dilate_val, 10, on_trackbar_dilate)
    cv2.createTrackbar("Blur size", window_title, blur_val, 20, on_trackbar_blur)
    cv2.createTrackbar("Erode size", window_title, erode_val, 20, on_trackbar_erode)
    cv2.createTrackbar("Open size", window_title, open_val, 20, on_trackbar_open)
