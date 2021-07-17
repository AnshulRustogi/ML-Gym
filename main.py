#Using MPII Model

import cv2 as cv2
import os
import argparse
from pushup import PushUp
from squat import Squats


class MLGym:
    def __init__(self, args, modelfiles):
        
        protoFile, weightsFile = modelfiles
        if os.path.isfile(protoFile) and os.path.isfile(weightsFile):
            self.MPIINet = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        else:
            print("Error reading model files")
            exit(-1)
        
        allowedActivity = ["pushup", "squats"]

        if args.device == 'cpu':
            self.MPIINet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print('Using CPU Device')
        elif args.device == 'gpu':
            self.MPIINet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.MPIINet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU Device")
        else:
            print("Error finding compute device")
            exit(-1)
            
        if args.videoInput == "cam":
            self.input = 0
        else:
            if os.path.isfile(args.videoInput):
                self.input = args.videoInput
            else:
                print("Error finding input video file")
                exit(-1)
                
        self.activity = "None"
        if args.activity in allowedActivity:
            self.activity = args.activity
        else:
            while self.activity not in allowedActivity:
                self.activity = input("Please enter valid activity: ")
                self.activity = self.activity.lower()
            
        if args.displaySkeleton == '1':
            self.displaySkeleton = 1
        else:
            self.displaySkeleton = 0

        self.scalingFactor = 1
        
        self.camera = cv2.VideoCapture(self.input)
        
        self.threshold = 0
        
    def start(self):
        
        if self.activity == "pushup":
            pushup = PushUp(self.MPIINet, self.scalingFactor, self.camera, self.threshold, self.displaySkeleton)
            pushup.start()
        
        elif self.activity == "squats":
            squats = Squats(self.MPIINet, self.scalingFactor, self.camera, self.threshold, self.displaySkeleton)
            squats.start()
            
            
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ML Gym ')
    parser.add_argument('--device', default='gpu', help='Device to run the interface on: cpu/gpu')
    parser.add_argument('--videoInput', default='cam', help='Source of video input: cam/<source>')
    parser.add_argument('--activity', default='None', help='Activity you wish to perform: squats/pushup')
    parser.add_argument('--displaySkeleton', default='1', help='Display Skeleton: 0/1')
    args = parser.parse_args()
    
    modelfiles = ("models/MPII/pose_deploy_linevec.prototxt", "models/MPII/pose_iter_160000.caffemodel")
    
    mlgym = MLGym(args, modelfiles)
    mlgym.start()