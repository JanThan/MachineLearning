# Program to track multiple 'pucks'
# All various trackertypes have been defined for testing purposes
# CSRT was found to have the best results for our purpose
from __future__ import print_function
import sys
import cv2
from random import randint
import math

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker



trackerType = "CSRT"

  # Create a video capture object to read videos
cap = cv2.VideoCapture(0)

while True:

  # Read first frame
  success, frame = cap.read()
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  bboxes = []
  colors = []

  # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  # So we will call this function in a loop till we are done selecting all objects
  while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
      break

  print('Selected bounding boxes {}'.format(bboxes))

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  # Specify tracker type

  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()

  # Initialize MultiTracker
  for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)



  # Process video and track objects
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    #print(boxes)
    x1, y1, w1, h1 = boxes[0]
    x2, y2, w2, h2 = boxes[1]
    centreAx = x1+(w1/2)
    centreAy = y1+(h1/2)
    centreBx = x2+(w2/2)
    centreBy = y2+(h2/2)
    # print(centreA)
    # print(centreB)
    distX = centreBx - centreAx
    distY = centreBy - centreAy
    angleRad = math.atan(distY/distX)
    angleDeg = math.degrees(angleRad)

    #print(angleDeg)
    #print(distX, ' ', distY)

    # Top right quadrant
    if distX >= 0 and distY < 0:
      print(math.degrees(angleRad))
    # Top left quadrant
    elif distX < 0 and distY < 0:
      angleRad = math.pi + angleRad
      print(math.degrees(angleRad))
    # Bottom left quadrant
    elif distX < 0 and distY >= 0:
      angleRad = math.pi + angleRad
      print(math.degrees(angleRad))
    # Bottom right quadrant
    elif distX >=0 and distY >= 0:
      angleRad = 3*(math.pi)/2 - angleRad
      print(math.degrees(angleRad))


    #print(math.degrees(angleRad))
    #targetFile = open('/home/team-g/catkin_ws/src/simple_navigation_goals/src/TriTrackTheta.txt','w')
    #targetFile.write(str(angleRad))
    #targetFile.close()

    # show frame
    cv2.imshow('MultiTracker', frame)


    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == ord('l'):  # Esc pressed
      break
  break
cv2.destroyAllWindows()
