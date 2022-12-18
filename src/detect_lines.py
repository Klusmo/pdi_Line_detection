import cv2
import numpy as np


def pre_processing(img):
    # reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny algorithm for edge detection
    edges = cv2.Canny(img, 100, 200)

    # morphological closing to fill in gaps
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, None)
    return edges


def __rect_lines(lines):
    arr_lines = []
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 - 1000 * np.sin(theta))
        y1 = int(y0 + 1000 * np.cos(theta))
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 + 1000 * np.sin(theta))
        y2 = int(y0 - 1000 * np.cos(theta))
        arr_lines.append((x1, y1, x2, y2))

    return [arr_lines]


def get_lines(img):
    edges = pre_processing(img)
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1.3, np.pi / 180, 200)

    # if no lines are found, return an empty list
    # otherwise, return the lines or the lines converted to a rectangle format
    return [] if lines is None else __rect_lines(lines)


def get_hough_p(img):
    edges = pre_processing(img)

    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(
        edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap
    )

    return lines


def draw_lines(img, lines, ang_op=30):
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            deg = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            # if the line is too horizontal, skip it
            if not ang_op < abs(deg) < 180 - ang_op:
                continue

            pts = np.array([[x1, y1], [x2, y2]], np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
