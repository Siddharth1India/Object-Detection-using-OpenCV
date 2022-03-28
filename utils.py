import cv2
import numpy as np 

def getCounters(img, cannyTh=[200, 200], showCanny = False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cannyTh[0], cannyTh[1])
    kernel = np.ones((5,5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThreshold = cv2.erode(imgDialation, kernel, iterations=3)
    if showCanny:
        cv2.imshow("Canny", imgThreshold)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    finalCon = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter>0:
                if len(approx)==filter:
                    finalCon.append([len(approx), area, approx, bbox, cnt])
            else:
                finalCon.append([len(approx), area, approx, bbox, cnt])
    finalCon = sorted(finalCon, key =  lambda x:x[1], reverse=True)
    if draw:
        for cnt in finalCon:
            cv2.drawContours(img, cnt[4], -1, (0,0,255), 5)
    return img, finalCon

def reorder(pts):
    myPts = np.zeros_like(pts)
    pts = pts.reshape((4,2))
    add = pts.sum(1)
    myPts[0] = pts[np.argmin(add)] 
    myPts[3] = pts[np.argmax(add)]
    diff = np.diff(pts, axis = 1)
    myPts[1] = pts[np.argmin(diff)]
    myPts[2] = pts[np.argmax(diff)]

    return myPts

def warpImg(img, pts, w, h, pad=20):
    # print(pts)
    pts = reorder(pts)
    pts1 = np.float32(pts)
    pts2 = np.float32([[0,0],[w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    # imgWarp = imgWarp[pad: imgWarp.shape[0]-pad, pad: imgWarp.shape[1]-pad]
    return imgWarp

def findDistance(x, y):
    return ((y[0]-x[0])**2 + (y[1]-x[1])**2)**0.5
