import cv2 as cv
import time
import numpy as np
import PoseModule as pm

#cap = cv.VideoCapture("Data_Collection/shoulder_guy.MP4")
#cap = cv.VideoCapture("Data_Collection/shoulderpressb.MP4")
#cap = cv.VideoCapture("Data_Collection/shoulder_press.MP4")
#cap = cv.VideoCapture("Data_Collection/shoulderpressh.MP4")
#cap = cv.VideoCapture("Data_Collection/shoulderpressi.MP4")
#cap = cv.VideoCapture("Data_Collection/shoulderpressj.MP4")

#cap = cv.VideoCapture("Data_Collection/shoulderpressc.MP4") #side angle limitation

cap = cv.VideoCapture("Data_Collection/shoulderpressd.MP4") # detect the wrong person
#limitation when more than one people in the video too close to each other

#cap = cv.VideoCapture("Data_Collection/shoulderpressa.MP4") #the counting start before the exercise
#need the indicator for when to start the exercise
#cap = cv.VideoCapture("Data_Collection/shoulderpresse.MP4") #the counting start before the exercise
#need the indicator for when to start the exercise

#cap = cv.VideoCapture("Data_Collection/shoulderpressf.MP4") #couldnt detect cause to many object around the target
#limitation

#cap = cv.VideoCapture("Data_Collection/shoulderpressg.MP4")
#SOME ANGLE CONFUSE THE MODEL, HENCE THE VIDEO ANGLE CONSISTENCY IS REQUIRED


detector = pm.poseDetector()
count = 0
dir = 0  # direction



pTime = 0

while True:
    success, img = cap.read()
    #img = cv.imread("Data_Collection/body.jpg") # read image
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        righta=detector.findAngle(img, 11, 13, 15)
        lefta= detector.findAngle(img, 11, 13, 15)
        #angle = detector.findAngle(img, 11, 13, 15)
        if lefta <180:
          per = np.interp(lefta, (75, 155), (0, 100))
        elif lefta > 200:
          per = np.interp(lefta, (205, 305), (0, 100))

        bar = np.interp(lefta, (200, 288), (650,100))
        print(lefta, per)

        # counting reps
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5

                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        #cv.rectangle(img, (100,100), (400, 1000), color, 3)
        #cv.rectangle(img, (1000, int(bar)), (800, 650), color,cv.FILLED)
        cv.putText(img, f'{int(per)} %', (400, 75), cv.FONT_HERSHEY_PLAIN,4, color,4)

        # Draw Curl Count
        cv.rectangle(img,(0,450), (250,720), (0,255,0),cv.FILLED)
        cv.putText(img, str(int(count)), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 0, 5), 25)
        #cv.putText(img, str(int(count)), (50, 100), cv.FONT_HERSHEY_PLAIN,3, (255, 0, 0, 5),5)




    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
