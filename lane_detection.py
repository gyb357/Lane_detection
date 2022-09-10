import numpy as np
import cv2
from PIL import ImageGrab


def screen(x, y):
    grab = np.array(ImageGrab.grab(bbox=(0, 0, x, y)))
    return cv2.cvtColor(grab, cv2.COLOR_BGR2RGB)


def canny(image, low, high, kernel):
    Gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Gaus  = cv2.GaussianBlur(Gray, (kernel, kernel), 0)
    Canny = cv2.Canny(Gaus, low, high)
    return Canny


def roi(image, ver):
    zeros = np.zeros_like(image)
    cv2.fillPoly(zeros, ver, 255)
    return cv2.bitwise_and(image, zeros)


def hough_lines(image, minLength, maxGap):
    radian = np.pi/180
    # line   = cv2.HoughLinesP(image, 2, radian, 100, np.array([]), minLength, maxGap)
    # line2  = np.zeros((*image.shape, 3), dtype=np.uint8)
    return cv2.HoughLinesP(image, 2, radian, 100, np.array([]), minLength, maxGap)


def draw_lines(image, lines, width):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), width)
    return line_image


def screen_line(image, init_image):
    return cv2.addWeighted(init_image, 0.8, image, 1, 0)


vertices = np.array([[10, 500], [10, 300], [300, 50], [500, 50], [800, 300], [800, 500]])


while (True):
    Screen       = screen(800, 600)                # x, y
    Screen_canny = canny(Screen, 100, 150, 5)      # image, low, high, kernel
    Screen_roi   = roi(Screen_canny, [vertices])   # image, ver

    lines        = hough_lines(Screen_roi, 50, 5)  # image, minLength, maxGap
    line_image   = draw_lines(Screen, lines, 2)    # image, lines
    Screen_line  = screen_line(Screen, line_image) # image, init_image


    cv2.imshow('1', Screen_line)
    cv2.imshow('2', Screen_roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

