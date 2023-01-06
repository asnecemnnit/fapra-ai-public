from libraries import *
from utils.ball import Ball, Balls, BallType

class BallsTracker:

    def __init__(self, init_once = False, tracker_list = [], tracker = None):
        self.init_once = False
        self.tracker_list = []
        self.tracker = None

    def tracker_init(self):
        self.tracker = cv2.MultiTracker_create()
        self.init_once = True

    def add_tracker(self, frame, bbox):
        self.tracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    def add_to_tracker_list(self, ball_type, x, y, ravg):
        self.tracker_list.append([ball_type, ((x - ravg) if x > ravg else 0,
                                              (y - ravg) if y > ravg else 0, 2 * ravg, 2 * ravg)])

    def update_tracker(self, frame):
        return self.tracker.update(frame)

    def get_tracker_list(self):
        return self.tracker_list


def track_balls(balls, frame, ravg, balls_tracker):

    # add bboxs for tracking
    if not balls_tracker.init_once:
        balls_tracker.tracker_init()
        for ball in balls:
            balls_tracker.add_to_tracker_list(ball.ball_type, ball.x, ball.y, ravg)
        for label, bbox in balls_tracker.tracker_list:
            balls_tracker.add_tracker(frame, bbox)

    # update tracker
    ret, boxes = balls_tracker.update_tracker(frame)
    print("success", ret)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    try:
        # print(tracker_list)
        tracked_balls = Balls()
        for idx, item in enumerate(balls_tracker.get_tracker_list()):
            item[1] = list(boxes[idx])
            tracked_balls.add(Ball(x=int(item[1][0] + item[1][2] / 2), y=int(item[1][1] + item[1][2] / 2),
                                   radius=int(item[1][2] / 2), ball_type=item[0]))

    except:
        print('tracker_list problem')

    # print(len(tracked_balls))

    for ball in tracked_balls:
        cv2.putText(frame, str(ball.ball_type), (ball.x - 20, ball.y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    return tracked_balls
