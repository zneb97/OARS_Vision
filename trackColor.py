
from shapeDetector import ShapeDetector
from collections import deque
import numpy as np
import imutils
import argparse
import cv2

def detects(image):

    #Resize images
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    # Find contours, begin ShapeDetector
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    # Loop over all contours
    for c in cnts:
    	# compute the center of the contour, then detect the name of the
    	# shape using only the contour
    	shape = sd.detect(c)

        print 'Detected shape: %s' % shape
    	# multiply the contour (x, y)-coordinates by the resize ratio,
    	# then draw the contours and the name of the shape on the image
    	c = c.astype("float")
    	c *= ratio
    	c = c.astype("int")
    	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

# Set up Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# Define HSV values for color (Red in this case)
redLower1 = (0,70,70)
redUpper1 = (10,255,255)
redLower2 = (170,70,50)
redUpper2 = (180,255,255)
pts = deque(maxlen=args["buffer"])

#Get webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

while True:
    # Current Frame
    (grabbed, frame) = camera.read()

    # Resize, covert frame to HSV
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Mask all colors but red in frame
    mask1 = cv2.inRange(hsv, redLower1, redUpper1)
    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    mask = mask1+mask2

    #Clean up masks
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Display views
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('Mask',mask)
    cv2.imshow('Res',res)

    #Clean up mask for contour detection
    mask = mask*255

    #Pass mask into shape detection
    detects(mask)

    #Escape sequence
    key = cv2.waitKey(1) & 0xFF
    #Close all winows on q
    if key == ord("q"):
        break

#Close camera
camera.release()
cv2.destroyAllWindows()
