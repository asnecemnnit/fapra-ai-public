from libraries import *
from utils.ball import Ball, Balls, BallType

COLOR_NAMES = ["white_solid", "yellow_solid", "blue_solid", "red_solid", "purple_solid", "orange_solid",
               "green_solid", "brown_solid", "black_solid"]
COLOR_NAMES_STRIPE = ["dummy_white_stripe", "yellow_stripe", "blue_stripe", "red_stripe", "purple_stripe",
                      "orange_stripe", "green_stripe", "brown_stripe", "dummy_black_stripe"]

COLOR_RANGES_HSV = {
    "white_solid": [(0, 0, 150), (180, 65, 255)],  # (0, 0, 60)->(360, 25, 100)
    "yellow_solid": [(20, 80, 150), (30, 255, 255)],  # (40, 30, 60)->(60, 100, 100)
    "blue_solid": [(100, 130, 75), (128, 255, 255)],  # (200, 50, 30)->(256, 100, 100)
    "red_solid": [(0, 130, 75), (2, 255, 255)],  # new #(0, 50, 30)->(5, 100, 100)
    "purple_solid": [(120, 130, 50), (145, 255, 255)],  # (240, 50, 20)->(290, 100, 100)
    "orange_solid": [(5, 100, 150), (16, 255, 255)],  # (10, 54, 73)->(32, 100, 100)
    "green_solid": [(40, 130, 75), (95, 255, 100)],  # (80, 50, 30)->(190, 100, 40)
    "brown_solid": [(5, 165, 40), (15, 255, 128)],  # (10, 65, 15)->(30, 100, 50)
    "black_solid": [(0, 0, 1), (180, 255, 40)],  # (0, 0, 1)->(360, 100, 15)
    "red_solid_2": [(175, 130, 130), (180, 240, 210)],  # new #(350, 50, 50)->(360, 94, 82)
    "brown_solid_2": [(150, 50, 20), (180, 255, 100)],  # (300, 20, 8)->(360, 100, 40)
}

perfect_frame_found = False


def filter_idx(primaryIndex, pixels_per_color):
    newIndex = primaryIndex

    if primaryIndex == BallType.orange_solid:
        if pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid

    if primaryIndex == BallType.yellow_solid:
        if pixels_per_color[BallType.orange_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.orange_solid] < 20:
                if pixels_per_color[BallType.red_solid] != 0:
                    if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                        newIndex = BallType.red_solid
                    else:
                        newIndex = BallType.orange_solid
                else:
                    newIndex = BallType.orange_solid

    elif primaryIndex == BallType.blue_solid:
        if pixels_per_color[BallType.purple_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.purple_solid] < 20:
                if pixels_per_color[BallType.orange_solid] != 0:
                    if pixels_per_color[primaryIndex] / pixels_per_color[BallType.orange_solid] < 20:
                        if pixels_per_color[BallType.red_solid] != 0:
                            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                                newIndex = BallType.red_solid
                            else:
                                newIndex = BallType.orange_solid
                        else:
                            newIndex = BallType.orange_solid
                else:
                    newIndex = BallType.purple_solid
        elif pixels_per_color[BallType.orange_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.orange_solid] < 20:
                if pixels_per_color[BallType.red_solid] != 0:
                    if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                        newIndex = BallType.red_solid
                    else:
                        newIndex = BallType.orange_solid
                else:
                    newIndex = BallType.orange_solid
        elif pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid


    elif primaryIndex == BallType.purple_solid:
        if pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid

    elif primaryIndex == BallType.brown_solid:
        if pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid

    elif primaryIndex == 0:
        if pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid

    elif primaryIndex == BallType.green_solid:
        if pixels_per_color[BallType.yellow_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.yellow_solid] < 20:
                if pixels_per_color[BallType.orange_solid] != 0:
                    if pixels_per_color[primaryIndex] / pixels_per_color[BallType.orange_solid] < 20:
                        if pixels_per_color[BallType.red_solid] != 0:
                            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                                newIndex = BallType.red_solid
                            else:
                                newIndex = BallType.orange_solid
                        else:
                            newIndex = BallType.orange_solid
                else:
                    newIndex = BallType.yellow_solid
        elif pixels_per_color[BallType.orange_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.orange_solid] < 20:
                if pixels_per_color[BallType.red_solid] != 0:
                    if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                        newIndex = BallType.red_solid
                    else:
                        newIndex = BallType.orange_solid
                else:
                    newIndex = BallType.orange_solid
        elif pixels_per_color[BallType.red_solid] != 0:
            if pixels_per_color[primaryIndex] / pixels_per_color[BallType.red_solid] < 20:
                newIndex = BallType.red_solid

    return newIndex


def filter_color(pixels_per_color):
    # TODO Do not use hardcoded values
    # "No white component" or "black dominant" or "white dominant" => only solid colors
    if pixels_per_color[BallType.white_solid] == 0 or pixels_per_color[BallType.black_solid] == max(
            pixels_per_color):
        primaryIndex = filter_idx(pixels_per_color.index(max(pixels_per_color)), pixels_per_color)

        return COLOR_NAMES[primaryIndex], 1e9

    # "Corner case: "ratio of dominant color to white is very high" => only solid
    elif max(pixels_per_color) / pixels_per_color[BallType.white_solid] >= 10 or \
            (pixels_per_color[BallType.white_solid] < 250 and
             max(pixels_per_color) / pixels_per_color[BallType.white_solid] >= 3):
        primaryIndex = filter_idx(pixels_per_color.index(max(pixels_per_color)), pixels_per_color)
        return COLOR_NAMES[primaryIndex], pixels_per_color[primaryIndex] / pixels_per_color[BallType.white_solid]
    # Otherwise => stripe
    else:
        if pixels_per_color.index(max(pixels_per_color)) == BallType.white_solid:
            pixels_per_color = np.array(pixels_per_color)
            secondHighestIndex = np.argsort(pixels_per_color)[-2]
            if pixels_per_color[BallType.white_solid] < 1800:
                if secondHighestIndex == BallType.black_solid:
                    thirdHighestIndex = np.argsort(pixels_per_color)[-3]
                    primaryIndex = filter_idx(thirdHighestIndex, pixels_per_color)
                    if pixels_per_color[BallType.white_solid] / pixels_per_color[primaryIndex] > 120:
                        primaryIndex = thirdHighestIndex
                else:
                    primaryIndex = filter_idx(secondHighestIndex, pixels_per_color)
                    if pixels_per_color[BallType.white_solid] / pixels_per_color[primaryIndex] > 120:
                        primaryIndex = secondHighestIndex
            else:
                primaryIndex = BallType.white_solid

            pixels_per_color = pixels_per_color.tolist()

            if primaryIndex == BallType.white_solid:
                return COLOR_NAMES[BallType.white_solid], \
                       pixels_per_color[BallType.white_solid] / pixels_per_color[BallType.white_solid]
            elif pixels_per_color[BallType.white_solid] > 250:
                return COLOR_NAMES_STRIPE[primaryIndex], \
                       pixels_per_color[primaryIndex] / pixels_per_color[BallType.white_solid]
            elif pixels_per_color[primaryIndex] == 0 or \
                    pixels_per_color[BallType.white_solid] / pixels_per_color[primaryIndex] >= 10:
                return COLOR_NAMES[BallType.white_solid], \
                       pixels_per_color[BallType.white_solid] / pixels_per_color[BallType.white_solid]
            else:
                return COLOR_NAMES_STRIPE[primaryIndex], \
                       pixels_per_color[primaryIndex] / pixels_per_color[BallType.white_solid]
        else:
            primaryIndex = filter_idx(pixels_per_color.index(max(pixels_per_color)), pixels_per_color)
            return COLOR_NAMES_STRIPE[primaryIndex], \
                   pixels_per_color[primaryIndex] / pixels_per_color[BallType.white_solid]


def count_color(frame, color_range):
    lower = np.array(color_range[0], dtype=np.uint8)
    upper = np.array(color_range[1], dtype=np.uint8)

    frame_filtered = cv2.inRange(frame, lower, upper)
    count = cv2.countNonZero(frame_filtered)

    return count


def get_dominant_color(frame, ball):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_blurr = cv2.GaussianBlur(frame_hsv, (3, 3), 0)

    mask = np.zeros(frame[:, :, 0].shape, np.uint8)

    # cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
    cv2.circle(mask, (ball.x, ball.y), ball.radius, 255, -1)

    values = frame_blurr[np.where(mask == 255)]
    # print(np.median(values, axis=0))
    frame_values = np.expand_dims(values, axis=0)

    pixels_per_color = []
    for color, color_range in COLOR_RANGES_HSV.items():
        count = count_color(frame_values, color_range)
        if color == "red_solid_2":
            pixels_per_color[int(BallType.red_solid)] += count
        elif color == "brown_solid_2":
            pixels_per_color[int(BallType.brown_solid)] += count
        else:
            pixels_per_color.append(count)

    return filter_color(pixels_per_color)


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def classify_two_solids(balls, cwr_list, balltype_list):
    idx = duplicates(balltype_list, str(BallType.yellow_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.yellow_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.yellow_solid])

    idx = duplicates(balltype_list, str(BallType.purple_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.purple_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.purple_solid])

    idx = duplicates(balltype_list, str(BallType.orange_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.orange_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.orange_solid])

    idx = duplicates(balltype_list, str(BallType.brown_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.brown_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.brown_solid])

    idx = duplicates(balltype_list, str(BallType.blue_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.blue_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.blue_solid])

    idx = duplicates(balltype_list, str(BallType.red_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.red_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.red_solid])

    idx = duplicates(balltype_list, str(BallType.green_solid.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] < cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES_STRIPE[BallType.green_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES_STRIPE[BallType.green_solid])

    return balls


def classify_two_stripes(balls, cwr_list, balltype_list):
    idx = duplicates(balltype_list, str(BallType.yellow_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.yellow_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.yellow_solid])

    idx = duplicates(balltype_list, str(BallType.purple_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.purple_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.purple_solid])

    idx = duplicates(balltype_list, str(BallType.orange_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.orange_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.orange_solid])

    idx = duplicates(balltype_list, str(BallType.brown_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.brown_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.brown_solid])

    idx = duplicates(balltype_list, str(BallType.blue_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.blue_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.blue_solid])

    idx = duplicates(balltype_list, str(BallType.red_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.red_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.red_solid])

    idx = duplicates(balltype_list, str(BallType.green_stripe.name))
    if len(idx) == 2:
        if cwr_list[idx[0]] > cwr_list[idx[1]]:
            balls.balls[idx[0]].set_color(COLOR_NAMES[BallType.green_solid])
        else:
            balls.balls[idx[1]].set_color(COLOR_NAMES[BallType.green_solid])

    return balls


def filter_data(balls, cwr_list):
    if balls.is_unique():
        # print("ha")
        return balls
    else:
        y = (x.ball_type for x in balls)
        balltype_list = list(y)
        # print(balltype_list)

        balls = classify_two_solids(balls, cwr_list, balltype_list)

        balls = classify_two_stripes(balls, cwr_list, balltype_list)

        return balls


def detect_colors(frame, balls):
    if balls is not None:
        cwr_list = []
        for ball in balls:
            color, color_to_white_ratio = get_dominant_color(frame, ball)
            # print(color)
            cwr_list.append(color_to_white_ratio)
            ball.set_color(color)

        # print(data)
        balls = filter_data(balls, cwr_list)
        # balls.__str__()

        return balls
