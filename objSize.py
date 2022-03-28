from typing import final
import cv2
import numpy as np
import utils
webcam = False
path = 'C:\\Users\\siddh\\OneDrive\\Desktop\\Siddharth\\cv\\cvProjects\\ObjSize\\static\\Test3.jpeg'
vid = cv2.VideoCapture(0)
vid.set(10, 160)
vid.set(3, 1920)
vid.set(4, 1080)
scaleFactor = 2
# img = cv2.imread(path)
# cv2.imshow("Test", img)
wP = 210*scaleFactor
hP = 297*scaleFactor


while True:
    if webcam:  
        success, img = vid.read()
    else:
        img = cv2.imread(path)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgCnt, finalCnt = utils.getCounters(img, minArea=10000, filter=4, draw=True)
    # cv2.imshow("test", imgCnt)
    # cv2.imshow("Img",img)
    if len(finalCnt)!=0:
        biggest = finalCnt[0][2]
        imgWarp = utils.warpImg(img, biggest, wP, hP)
        # cv2.imshow("img", imgWarp)
        imgCnt2, finalCnt2 = utils.getCounters(imgWarp, filter=4)
        # cv2.imshow("Warp", imgCnt2)
        if len(finalCnt2) !=0:
            for obj in finalCnt2:
                cv2.polylines(imgCnt2, [obj[2]], True, (0,255,0), 3)
                nPoints = utils.reorder(obj[2])

                nw = round(utils.findDistance(nPoints[0][0]//scaleFactor, nPoints[1][0]//scaleFactor)/10, 1)
                nh = round(utils.findDistance(nPoints[0][0]//scaleFactor, nPoints[2][0]//scaleFactor)/10, 1)

                finalArea = "Area = "+str(nw*nh)
                imgFinal = cv2.putText(imgCnt2, finalArea, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 135, 255), 4)
        cv2.imshow("Something", imgCnt2)
    # cv2.imshow("Output", img)
    cv2.waitKey(1)