# Below program was used to identify the below error codes from a flashing led on a moving 'puck' sized robot. 
# General idea was to create a black mask around the detected object (to ignore all other sources of light from the image) 
# Consequently the sequence of light that was registered would be reported to result in the respective error code.

# Error Codes:
# normal    : red, blue, green
# V001      : red, red, blue, green
# V002      : red, red, green, blue
# 500ms, 50% duty Cycle pulses

# look at auto white balance
# HSL as a possible method to get around the lightness issue

import cv2
import numpy as np

def ColourSpace(image):
    # convert input image to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # applying filter to remove 'salt and pepper' effect
    hsv = cv2.medianBlur(hsv, 5)# (3,3), 0)
    # colour ranges in HSV hard-coded here --> histogram showed 119 average for these numbers. Check in different lighting
    lower_green = np.array([57,46,155])
    upper_green = np.array([90,255,255])

    lower_blue = np.array([100,67,221])
    upper_blue = np.array([120,255,255])

    lower_red = np.array([130,67,0])
    upper_red = np.array([255,255,255])

    # threshold the hsv image
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # bitwise-AND the mask and frame
    res_blue = cv2.bitwise_and(image,image,mask = blue_mask)
    res_green = cv2.bitwise_and(image,image,mask = green_mask)
    res_red = cv2.bitwise_and(image,image,mask = red_mask)

    # add the red, green and blue together to see each  on the same output feed
    res = np.add(res_blue,res_green)
    res = np.add(res,res_red)

    return (res, res_red, res_green, res_blue)



cap = cv2.VideoCapture(0)
# initialising counter and startSequence
i = 0
startSequence = 0
error = 0
# initialising error codes
Normal = ['R','Z','B','Z','G']
V001 = ['R', 'Z', 'R', 'Z', 'B', 'Z', 'G']
V002 = ['R', 'Z', 'R', 'Z', 'G', 'Z', 'B']

Normal = ['R','Z','B','Z','G',      'Z','R','Z','B','Z','G']
V001 = ['R', 'Z', 'R', 'Z', 'B', 'Z', 'G',      'Z','R', 'Z', 'R', 'Z', 'B', 'Z', 'G']
V001alt = ['R', 'Z', 'B', 'Z', 'G',      'Z','R', 'Z', 'R', 'Z', 'B', 'Z', 'G']
V002 = ['R', 'Z', 'R', 'Z', 'G', 'Z', 'B',      'Z','R', 'Z', 'R', 'Z', 'G', 'Z', 'B', ]
V002alt = ['R', 'Z', 'G', 'Z', 'B',      'Z','R', 'Z', 'R', 'Z', 'G', 'Z', 'B', ]


# initialising raw and final arrays
lightsRaw =[]
sequence =[]
xmin, ymin, width, height = 0, 0, 480, 640

while True:
    k = cv2.waitKey(1) & 0xFF
    # resetting due to error in sequence detection
    if error == 99:
        i = 0
        startSequence = 0
        error =0
        lightsRaw =[]
        sequence = []

    # read current frame
    ret, image_np = cap.read()
    image_np = cv2.medianBlur(image_np, 9)# (3,3), 0)

    # ------------ create mask ----------------#
    # Create the basic black image
    mask = np.zeros(image_np.shape, dtype = "uint8")
    if k == ord('w'):
        initBB = cv2.selectROI('Output Feed',image_np,fromCenter=False,showCrosshair=True)
        xmin, ymin, width, height = initBB

    cv2.rectangle(mask, (xmin, ymin), (xmin+width,ymin+height ), (255, 255, 255), -1)
    image_np = cv2.bitwise_and(image_np,mask)

    # running ColourSpace program, returns combined colours in an array for each frame of video
    (outputFeed, redArray, greenArray, blueArray) = ColourSpace(image_np)
    redArray = cv2.medianBlur(redArray, 7)
    greenArray = cv2.medianBlur(greenArray, 3)
    blueArray = cv2.medianBlur(blueArray, 5)

    # find the max value of each array, find if it is red, green or blue as other arrays will be 0
    maxRed = np.amax(redArray)
    maxGreen = np.amax(greenArray)
    maxBlue = np.amax(blueArray)

    # make sure the sequence starts at Red for ease
    # if red, sets StartSequence to 1 for the rest of this loop to run
    if maxRed >= 170:
        startSequence = 1
        if i == 0:
            print('starting visual error code detection')

    # loop to store colours once red value has been found
    if startSequence == 1:
        # check if red, green or blue (or black),
        # append this to the end of the raw array
        if maxRed >= 170:
            #print('Red')
            lightsRaw.append('R')

        elif maxGreen >= 170:
            #print('Green')
            lightsRaw.append('G')

        elif maxBlue >= 170:
            #print('Blue')
            lightsRaw.append('B')

        else:
            #print('Black')
            lightsRaw.append('Z')

        # check if the current colour is different from the previous
        if i != 0:
            if lightsRaw[i] != lightsRaw[i-1]:
                # if the colour is different add it to the sequence
                sequence.append(lightsRaw[i])
                print('sequence:',sequence)

                # first check if the sequence is less than the maximum length of the defined error codes
                if len(sequence) > max(len(Normal),len(V001),len(V002)):
                    error = 99
                    print('incorrect Sequence detected, starting again')

                # and then check if this sequence is equal to any of the known error code sequences
                elif sequence == Normal:
                    # RBG
                    print('sequence:',sequence)
                    print('error code: Normal Operation')
                    break

                elif sequence == V001:
                    # RRBG
                    print('sequence:',sequence)
                    print('error code: V001')
                    break

                elif sequence == V001alt:
                    # RRBG
                    print('sequence:',sequence)
                    print('error code: V001')
                    break

                elif sequence == V002:
                    # RRGB
                    print('sequence:',sequence)
                    print('error code: V002')
                    break

                elif sequence == V002alt:
                    # RRGB
                    print('sequence:',sequence)
                    print('error code: V002')
                    break

        # Things to do just for the first loop iteration
        elif i == 0:
            # append the first colour
            sequence.append(lightsRaw[i])

        i = i + 1
        # output the resultant processed image

    cv2.imshow('original image', image_np)
    cv2.imshow('green', greenArray)
    cv2.imshow('blue', blueArray)
    cv2.imshow('red', redArray)

    if k == ord('q'):
        break
    if k == ord('s'):
        while(1):
            if (cv2.waitKey(1) & 0xFF) == ord('s'):
                break
cv2.destroyAllWindows()
