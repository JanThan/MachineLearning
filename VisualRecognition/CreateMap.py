import cv2, numpy as np
# setting up camera
cap = cv2.VideoCapture(0)
import random as rng

while(True):
  # Capture frame
  rng.seed(12345)
  ret, frame = cap.read()
  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  key = cv2.waitKey(1) & 0xFF
  
  # filter out yellow and green and black here using colourspace
  
  kernel = np.ones((3,3),np.uint8)
  # canny edge detect
  edges = cv2.Canny(frame,100,200)
  # dilate the edges a bit to help fill contours
  edges = cv2.dilate(edges,kernel)
  # find all available contours
  contours,hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # when you click c, fill all of the found contours in white
  if key == ord('c'):
    contourLines = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255,255,255)
        cv2.drawContours(contourLines, contours, i, color, -1)#2, cv2.LINE_8, hierarchy, 0)
        
  # export to Shehroz's mapping 
    # Show in a window
    cv2.imshow('Contours', contourLines)
    cv2.imshow('Filled In',filledIn)
  
  
  # Display Video Feed
  cv2.imshow('Camera Feed',frame)

  # Option to exit Video Capture
  if key == ord('q'):
    break
  
cap.release()

cv2.destroyAllWindows()
