# function for extraxting data point for model training
import math
import pandas as pd
import numpy as np

# load ml model 
import joblib
WaveModel = joblib.load('ML/WaveDetectionModels/wave_model_random_forest.sav')



dataBufferVal = 0
dataDict = {'upperRightShoulder':[], 'upperLeftShoulder':[], 'TargetWave':[]}
dataBuffer = {'upperRightShoulder':[], 'upperLeftShoulder':[]}
waveDetect = 0 #initalise prediction variable

def extract_data(keyPoints,wave):
    """
        keypoint -> full body keypoint
        wave -> while training wave send true others false
    """
    global dataDict
    global dataBufferVal
    global dataBuffer
    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]

    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    
    upperRightShoulder = int(360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) )
    upperLeftShoulder = int(calculate_angle(right_shoulder,left_shoulder,left_wrist))


    # data collection
    if dataBufferVal >= 60:
        dataDict['upperRightShoulder'].append(dataBuffer['upperRightShoulder']) # add buffer to main data dict
        dataDict['upperLeftShoulder'].append(dataBuffer['upperLeftShoulder'])
        if wave:
            dataDict['TargetWave'].append(1) # For wave 
        else:
            dataDict['TargetWave'].append(0) # No wave

        dataBufferVal = 0 # reset data buffer values
        dataBuffer = {'upperRightShoulder':[], 'upperLeftShoulder':[]}#  empty the data buffer
    else:
        dataBuffer['upperRightShoulder'].append(upperRightShoulder)
        dataBuffer['upperLeftShoulder'].append(upperLeftShoulder)
        dataBufferVal += 1

    return dataDict


def infer(keyPoints):
    global dataBufferVal
    global dataBuffer
    global waveDetect

    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]

    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    
    upperRightShoulder = int(360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) )
    upperLeftShoulder = int(calculate_angle(right_shoulder,left_shoulder,left_wrist))

    # data collection
    if dataBufferVal >= 60:
        # ML infer
       detectData = dataBuffer["upperRightShoulder"] + dataBuffer['upperLeftShoulder']
       waveDetect = WaveModel.predict([detectData])


       dataBufferVal = 59 # reset data buffer values
       dataBuffer['upperRightShoulder'].pop(0) # remove first element
       dataBuffer['upperLeftShoulder'].pop(0) # remove first element


    else:
        dataBuffer['upperRightShoulder'].append(upperRightShoulder)
        dataBuffer['upperLeftShoulder'].append(upperLeftShoulder)
        dataBufferVal += 1

    return waveDetect
    


def save_to_csv(data):
    daraFrame = pd.DataFrame(data)
    daraFrame.to_csv("train_wave.csv")

def save_to_numpy(data):
    daraFrame = pd.DataFrame(data)
    df = daraFrame.to_numpy()
    np.save("train_wave_npy.npy",df)
    np.savez("train_wave_savez.npz",df)

def calculate_angle(pt1,pt2,pt3):
    """Calculates the angle between pt1 - pt2 - pt3 | pt = [y,x]"""
    dx1 = pt1[1] - pt2[1] # dist x1 -x2
    dy1 = pt1[0] - pt2[0] # dist y1 -y2
    dx2 = pt3[1] - pt2[1] # dist x1 -x2
    dy2 = pt3[0] - pt2[0] # dist y3 -y2
    
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    angle = angle1 - angle2
    if angle < 0:
        angle += 2 * math.pi
    return math.degrees(angle) if angle >= 0 else math.degrees(angle + 2 * math.pi)
