
from collections import deque
import numpy as np
import argparse
import cv2

# Set up Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# Define HSV values for color (Red in this case)
redLower = (0,75,75)
redUpper = (3,255,255)
pts = deque(maxlen=args["buffer"])

#Get webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

while True:
    # Current Frame
    (grabbed, frame) = camera.read()

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Mask red with small cleanups
    mask = cv2.inRange(hsv, redLower, redUpper)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.erode(mask, element, iterations=1)
    mask = cv2.dilate(mask, element, iterations=1)

    # Display views
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('Mask',mask)
    cv2.imshow('Res',res)
    key = cv2.waitKey(1) & 0xFF

    #Close all winows on q
    if key == ord("q"):
        break

#Close camera
camera.release()
cv2.destroyAllWindows()
