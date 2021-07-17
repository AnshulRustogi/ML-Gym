import cv2 as cv2
import numpy as np

class Squats:
    def __init__(self, MPIINet, scalingFactor, camera, threshold, displaySkeleton):
        
        self.net = MPIINet
        
        self.scalingFactor = scalingFactor
        
        self.humanPoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.pairs = [[2,3], [3,4], [5,6], [6,7], [2,5], [8,11], [2,8], [8,9], [9,10], [5,11], [11,12], [12,13]]
        
        self.camera = camera
        hasFrame, frame = self.camera.read()
        if not hasFrame:
            print("Error reading from webcam")
            exit(-1)
        print("Input Video Dimension: ",frame.shape)
        
        self.inHeight = frame.shape[0]
        self.inWidth = frame.shape[1]
        
        self.displaySkeleton = displaySkeleton
        
        self.threshold = threshold
        
        self.activityCount = 0
        self.activityState = 1
        
    def start(self):
        while(True):
            
            hasFrame, frame = self.camera.read()
            if not hasFrame:
                break
            
            inpBlobImage = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight), (0,0,0), swapRB=False, crop=False)
            self.net.setInput(inpBlobImage)
            output = self.net.forward()
            H = output.shape[2]
            W = output.shape[3]
            
            xyPoints = []
            
            for i in self.humanPoints:
                probMap = output[0, i, :, :]
                _, prob, _, point = cv2.minMaxLoc(probMap)
                
                xloc = (self.inWidth * point[0]) / W
                yloc = (self.inHeight * point[1]) / H
                
                xloc = ((self.inWidth*self.scalingFactor)/2)-((self.inWidth/2)-xloc)
                yloc = (self.inHeight*self.scalingFactor)-(self.inHeight-yloc)  
                
                if prob > self.threshold:
                    xyPoints.append((int(xloc),int(yloc)))
                else:
                    xyPoints.append(None)
            
            self.countSquats(xyPoints)
            frame = cv2.putText(frame, 'Total Squats: {}'.format(self.activityCount), (self.scalingFactor*int(3.5*self.inWidth)//8,self.scalingFactor*(7*self.inHeight)//8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
    
            if self.displaySkeleton:
                if xyPoints[0] and xyPoints[1]:
                    xface = int((xyPoints[0][0] + xyPoints[1][0])/2)
                    yface = int((xyPoints[0][1] + xyPoints[1][1])/2)
                    cv2.circle(frame, (xface, yface), 18, (255,255,255), thickness=-1, lineType=cv2.FILLED)
                    
                for pair in self.pairs:
                    if xyPoints[pair[0]] and xyPoints[pair[1]]:
                        cv2.line(frame, xyPoints[pair[0]], xyPoints[pair[1]], (255,255,255), 4, lineType=cv2.LINE_AA)
            
            cv2.imshow('Output', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    def countSquats(self, xyPoints):

        '''
        squashState=1 -> The person is in upright position, knee angle>=160
        squashState=0 -> The person is in down position, knee angle in b/w (90,75)
        squashState goes from 1->0->1 is a pushup

        '''

        if xyPoints[11] == None or xyPoints[12] == None or xyPoints[13] == None:
            leftKneeAngle = 150
        else:
            leftKneeAngle = self.getAngleBetweenPoints(xyPoints[11], xyPoints[12], xyPoints[13])
        
        if leftKneeAngle is None:
            return
        
        if leftKneeAngle >= 160 and self.activityState == 0:
            self.activityState = 1
            self.activityCount += 1
        elif leftKneeAngle < 100 and leftKneeAngle >= 75 and self.activityState == 1:
            self.activityState = 0
              
    def getAngleBetweenPoints(self, point1, pointMid, point3):
        point1, pointMid, point3  = np.array(point1), np.array(pointMid), np.array(point3)
        
        if point1.size == 1 or pointMid.size == 1 or point3.size == 1:
            return None 
        
        vector1 = point1 - pointMid
        vector2 = point3 - pointMid

        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        
        return np.arccos(np.dot(vector1, vector2))*(180/np.pi)
    
            