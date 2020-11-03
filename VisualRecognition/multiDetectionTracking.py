from __future__ import print_function
import sys
import cv2
from random import randint
import math



# initialising video capture object to read frames from
cap = cv2.VideoCapture(0)
print('Starting program ...')

# INITIALISE VARIABLES -------------------------------------------------------
bboxes = []             # Selected ROIs and ROI colours
colors = []
manualROIselection = 1  # Manual ROI selection activated
trackingStage = 0       # Tracking stage deactivated

# AUTOMATED ROI DETECTION ----------------------------------------------------
print('Automated ROI Detection Stage ... ')
while True:
    key = cv2.waitKey(25) & 0xFF
    success, frame = cap.read()
    if not success:
        print('Lost connection with video stream at automated ROI Detection stage ...')
        sys.exit()


    # ------------------------------------------------- #
    # detection stage goes here ----------------------- #
    # ------------------------------------------------- #


    # enter manual ROI selection
    if key == ord('m'):
        manualROIselection = 1
        break
    # enter tracking with detected ROIs
    elif key == ord('t'):
        manualROIselection = 0
        break


    cv2.imshow('Video Stream', frame)
    if key == ord('q'):
        print('Program Shutdown ...')
        cv2.destroyAllWindows()
        sys.exit()
        break




# MANUAL ROI SELECTION -------------------------------------------------------
# while loop is not necessarily needed here, but kept for future flexibility
# if further functionality is needed during this stage
while True:
    key = cv2.waitKey(1) & 0xFF
    success, frame = cap.read()
    if not success:
        print('Lost connection with video stream at manual ROI Selection stage ...')
        sys.exit(1)

    # detected ROIs are to be used, pass through this stage
    if manualROIselection == 0:
        break

    # ROIs are to be manually selected
    elif manualROIselection == 1:
        print('Manual ROI Selection Stage ...')
        while True:
            bbox = cv2.selectROI('Select Region of Interest', frame)
            bboxes.append(bbox)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            k = cv2.waitKey(0) & 0xFF
            if k == 113:
                cv2.destroyWindow('Select Region of Interest')
                print('Selected bounding boxes {}'.format(bboxes))
                break
    break



# TRACKING STAGE ---------------------------------------------------------
print('Tracking Stage ...')
while True:
    key = cv2.waitKey(1) & 0xFF
    success, frame = cap.read()
    if not success:
        print('Lost connection with video stream at Tracking stage ...')
        sys.exit()

    multiTracker = cv2.MultiTracker_create()

    for bbox in bboxes:
        multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        success, boxes = multiTracker.update(frame)

        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print('exitting program ...')
            exitProgram = 1
            break

    if exitProgram == 1:
        break

cv2.destroyAllWindows()
