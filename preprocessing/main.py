from affine_transformation import *
from circle_detection import *
from color_detection import *
from hough_rectangle import *
from image_enhancement import *
from parametrize import *
from tracking import *
from undistort import *

def preprocess(video_path, frame_number=0, is_distorted=True):
    # CAUTION: Requires ~150MB RAM per 1s video
    # List to store ball positions per frame
    balls_lst = []
    frame_lst = []

    vs = cv2.VideoCapture(video_path)
    frames_total = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_number >= frames_total:
        raise Exception("Start frame exceeds total number of frames")
    vs.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # keep looping
    while True:
        # grab the current frame
        res, frame = vs.read()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if not res or frame is None:
            break

        # Compensate lense distortion
        if is_distorted:
            frame = undistort(frame)

        parametrize(frame.copy())

        # Detect table's innermost rectangle and return its corners
        # Note: provide table color as "blue"/"green"
        corners = hough_rectangle(frame, False, "blue")
        # corners = 0

        # Affine transform to real table dimensions
        frame = affine_transform(frame, corners)

        # intensity matrix
        im = np.ones(frame.shape, dtype="uint8") * 50
        # Brighten the image
        frame = cv2.add(frame, im)
        # Darken the image
        frame = cv2.subtract(frame, im)

        # TODO gamma correction
        frame = adjust_gamma(frame, 0.8)
        # frame = adjust_gamma(frame, 2)

        # TODO investigate the role of shadows on pre-processing

        # detects all the circles in the frame with their coordinate info
        # TODO circle detection is not accurate yet. gamma-correction and shadows
        balls = detect_circles(frame)

        # detect colors of all the detected balls
        balls = detect_colors(frame, balls)

        balls_lst.append(balls)
        frame_lst.append(frame)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    vs.release()
    # close all windows
    cv2.destroyAllWindows()

    return balls_lst, frame_lst


def track(balls_lst, frame_lst):
    csv_lst = None
    max_i = -1
    max_balls = -1
    # Find most suitable frame to start tracking
    for i, balls in enumerate(balls_lst):
        # balls.balls = balls.balls[:5]  # TODO remove after improving color detection
        if balls.is_unique() and len(balls) > max_balls:
            max_i = i
            max_balls = len(balls)
    # print(max_i, max_balls)
    if max_i >= 0 and max_balls >= 0:
        csv_lst = [None] * len(balls_lst)
        csv_lst_horizontal = [None] * len(balls_lst)
        csv_lst_vertical = [None] * len(balls_lst)
        csv_lst_horizontal_vertical = [None] * len(balls_lst)
        backward_tracker = BallsTracker()
        # Track backward in time from max_i to 0
        for i, balls_frame in enumerate(zip(balls_lst[max_i::-1], frame_lst[max_i::-1])):
            balls, frame = balls_frame
            balls = track_balls(balls, frame, balls.radius_avg, backward_tracker)
            # show the frame to our screen
            cv2.imshow("Frames in backward direction", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("b"):
                break
            balls = convert_to_table_coordinates(balls)
            csv_lst[max_i - i], csv_lst_horizontal[max_i - i], csv_lst_vertical[max_i - i], \
            csv_lst_horizontal_vertical[max_i - i] = balls.to_csv()
        del backward_tracker

        forward_tracker = BallsTracker()
        # Track forward in time from max_i to end
        for i, balls_frame in enumerate(zip(balls_lst[max_i::], frame_lst[max_i::])):
            balls, frame = balls_frame
            balls = track_balls(balls, frame, balls.radius_avg, forward_tracker)
            # show the frame to oufr screen
            cv2.imshow("Frames in forward direction", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("f"):
                break
            balls = convert_to_table_coordinates(balls)
            csv_lst[max_i + i], csv_lst_horizontal[max_i + i], csv_lst_vertical[max_i + i], \
            csv_lst_horizontal_vertical[max_i + i] = balls.to_csv()
        del forward_tracker

    return csv_lst, csv_lst_horizontal, csv_lst_vertical, csv_lst_horizontal_vertical


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("--distorted",
                    help="process video as if clip is distorted (default : clip is undistorted)",
                    action="store_true", default=False)
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())

    # clip is distorted
    if not args.get("distorted", False):
        clip_is_distorted = False
    # otherwise not distortion
    else:
        clip_is_distorted = True

    video_path = args["video"]

    balls_lst, frame_lst = preprocess(video_path, is_distorted=clip_is_distorted)
    csv_lst, csv_lst_horizontal, csv_lst_vertical, csv_lst_horizontal_vertical = track(balls_lst, frame_lst)
