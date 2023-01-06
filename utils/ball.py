from enum import Enum, IntEnum
import random
import math
import utils.constants as constants


class BallType(IntEnum):
    white_solid = 0
    yellow_solid = 1
    blue_solid = 2
    red_solid = 3
    purple_solid = 4
    orange_solid = 5
    green_solid = 6
    brown_solid = 7
    black_solid = 8
    yellow_stripe = 9
    blue_stripe = 10
    red_stripe = 11
    purple_stripe = 12
    orange_stripe = 13
    green_stripe = 14
    brown_stripe = 15


class Ball:
    def __init__(self, x=None, y=None, radius=None, ball_type=None):
        self.ball_type = ball_type
        self.x = x
        self.y = y
        self.radius = radius

    def __str__(self):
        return f"Ball type={self.ball_type}, x={self.x}, y={self.y}, r={self.radius}"

    def set_position(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def set_color(self, ball_type: BallType):
        self.ball_type = ball_type

    def set_motion(self, velocity, angle):
        self.velocity = velocity
        self.angle = angle

    def step(self):
        x = self.x + self.velocity * math.cos(self.angle)
        y = self.y + self.velocity * math.sin(self.angle)
        self.x = x
        self.y = y


class Balls:
    def __init__(self):
        self.balls = []
        self.radius_avg = None

    def __len__(self):
        return len(self.balls)

    def __iter__(self):
        return self.balls.__iter__()

    def add(self, ball: Ball):
        self.balls.append(ball)

    def init_random(self, p_exist=1.0, p_motion=0.3, max_velocity=0.1):
        for ball_type in BallType:
            if not random.random() < p_exist:
                continue
            rand_x = random.uniform(0.0, constants.TABLE_LENGTH)
            rand_y = random.uniform(0.0, constants.TABLE_WIDTH)
            ball = Ball(x=rand_x, y=rand_y, radius=constants.BALL_RADIUS, ball_type=ball_type.name)
            is_moving = random.random() < p_motion
            rand_velocity = random.uniform(max_velocity / 5, max_velocity) if is_moving else 0.0
            rand_angle = random.uniform(0.0, 2 * math.pi) if is_moving else 0.0

            ball.set_motion(velocity=rand_velocity, angle=rand_angle)
            self.add(ball)

    def step(self):
        for ball in self.balls:
            ball.step()

    def is_unique(self):
        x = [x.ball_type for x in self.balls]
        return len(x) == len(set(x))

    def to_dict(self):
        if not self.is_unique():
            raise Exception("List of balls is not unique")
        ball_dict = {}
        for ball in self.balls:
            ball_dict[ball.ball_type] = ball
        return ball_dict

    def to_csv(self):
        # Column 2n : x ordinate of nth ball
        # Column 2n+1 : y ordinate of nth ball
        csv_row = [-0.1] * 32
        csv_row_horizontal = [-0.1] * 32
        csv_row_vertical = [-0.1] * 32
        csv_row_horizontal_vertical = [-0.1] * 32

        ball_dict = self.to_dict()
        for i, ball_type in enumerate(BallType):
            if ball_type.name in ball_dict.keys():
                csv_row[2 * i] = float("{:.3f}".format(ball_dict[ball_type.name].x))
                csv_row[2 * i + 1] = float("{:.3f}".format(ball_dict[ball_type.name].y))

                # data augmentation (horizontal mirroring)
                csv_row_horizontal[2 * i] = float("{:.3f}".format(ball_dict[ball_type.name].x))
                if ball_dict[ball_type.name].y != -0.1:
                    csv_row_horizontal[2 * i + 1] = float("{:.3f}".format(constants.TABLE_WIDTH -
                                                                          ball_dict[ball_type.name].y))

                # data augmentation (vertical mirroring)
                if ball_dict[ball_type.name].x != -0.1:
                    csv_row_vertical[2 * i] = float("{:.3f}".format(constants.TABLE_LENGTH -
                                                                    ball_dict[ball_type.name].x))
                csv_row_vertical[2 * i + 1] = float("{:.3f}".format(ball_dict[ball_type.name].y))

                # data augmentation (both horizontal & vertical mirroring)
                if ball_dict[ball_type.name].x != -0.1:
                    csv_row_horizontal_vertical[2 * i] = float("{:.3f}".format(constants.TABLE_LENGTH -
                                                                               ball_dict[ball_type.name].x))
                if ball_dict[ball_type.name].y != -0.1:
                    csv_row_horizontal_vertical[2 * i + 1] = float("{:.3f}".format(constants.TABLE_WIDTH -
                                                                                   ball_dict[ball_type.name].y))
        return csv_row, csv_row_horizontal, csv_row_vertical, csv_row_horizontal_vertical
