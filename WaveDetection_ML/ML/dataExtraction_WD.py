# function for extraxting data point for model training
import math
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras

WaveModel = tf.keras.models.load_model('ML/WaveDetectionModels/model_nn_40pt_v2.h5')
dataBufferVal = 0
dataDict = {'upperRightShoulder':[], 'upperLeftShoulder':[], 
            'rightShoulderElbow':[], 'leftShoulderElbow':[],
            'rightElbowWrist':[], 'leftElbowWrist':[],
            'TargetWave':[]}
dataBuffer = {'upperRightShoulder':[], 'upperLeftShoulder':[],
              'rightShoulderElbow':[], 'leftShoulderElbow':[],
              'rightElbowWrist':[], 'leftElbowWrist':[]}
waveDetect = 0 #initalise prediction variable
inference_window = 0

def extract_data(keyPoints,wave):
    """
        keypoint -> full body keypoint
        wave -> while training wave send true others false
    """
    global dataDict
    global dataBufferVal
    global dataBuffer
    
    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_elbow,left_elbow = keyPoints[3], keyPoints[6]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]

    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    
    upperRightShoulder = int(360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) )
    upperLeftShoulder = int(calculate_angle(right_shoulder,left_shoulder,left_wrist))

    rightShoulderElbow = int(360 - calculate_angle(left_shoulder,right_shoulder,right_elbow) )
    leftShoulderElbow = int(calculate_angle(right_shoulder,left_shoulder,left_elbow))

    rightElbowWrist = int(360 - calculate_angle(right_shoulder,right_elbow,right_wrist) )
    leftElbowWrist = int(calculate_angle(left_shoulder,left_elbow,left_wrist))
    
    # data collection
    if dataBufferVal >= 100:
        dataDict['upperRightShoulder'].append(dataBuffer['upperRightShoulder']) # add buffer to main data dict
        dataDict['upperLeftShoulder'].append(dataBuffer['upperLeftShoulder'])
        dataDict['rightShoulderElbow'].append(dataBuffer['rightShoulderElbow'])
        dataDict['leftShoulderElbow'].append(dataBuffer['leftShoulderElbow'])
        dataDict['rightElbowWrist'].append(dataBuffer['rightElbowWrist'])
        dataDict['leftElbowWrist'].append(dataBuffer['leftElbowWrist'])
        if wave:
            dataDict['TargetWave'].append(1) # For wave 
        else:
            dataDict['TargetWave'].append(0) # No wave

        dataBufferVal = 0 # reset data buffer values
        dataBuffer = {'upperRightShoulder':[], 'upperLeftShoulder':[],
                      'rightShoulderElbow':[], 'leftShoulderElbow':[],
                      'rightElbowWrist':[], 'leftElbowWrist':[]}
        #  empty the data buffer
    else:
        dataBuffer['upperRightShoulder'].append(upperRightShoulder)
        dataBuffer['upperLeftShoulder'].append(upperLeftShoulder)
        dataBuffer['rightShoulderElbow'].append(rightShoulderElbow)
        dataBuffer['leftShoulderElbow'].append(leftShoulderElbow)
        dataBuffer['rightElbowWrist'].append(rightElbowWrist)
        dataBuffer['leftElbowWrist'].append(leftElbowWrist)       
        dataBufferVal += 1

    return dataDict

def infer(keyPoints):
    global dataBufferVal
    global dataBuffer
    global waveDetect
    global inference_window

    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_elbow,left_elbow = keyPoints[3], keyPoints[6]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]

    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    
    upperRightShoulder = int(360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) )
    upperLeftShoulder = int(calculate_angle(right_shoulder,left_shoulder,left_wrist))

    rightShoulderElbow = int(360 - calculate_angle(left_shoulder,right_shoulder,right_elbow) )
    leftShoulderElbow = int(calculate_angle(right_shoulder,left_shoulder,left_elbow))

    rightElbowWrist = int(360 - calculate_angle(right_shoulder,right_elbow,right_wrist) )
    leftElbowWrist = int(calculate_angle(left_shoulder,left_elbow,left_wrist))



    # data collection
    if dataBufferVal >= 40:
        # ML infer
       detectData = []
       temp = []
       temp.append(dataBuffer["upperRightShoulder"])
       temp.append(dataBuffer['upperLeftShoulder'])

       temp.append(dataBuffer['rightShoulderElbow'])
       temp.append(dataBuffer['leftShoulderElbow'])

       temp.append(dataBuffer['rightElbowWrist'])
       temp.append(dataBuffer['leftElbowWrist'])

       detectData.append([temp])
       detectData = np.array(detectData)

       if inference_window >= 5:
        #waveDetect = 1 if WaveModel.predict([detectData],verbose = 0) >= 0.5 else 0
        waveDetect = np.round(WaveModel.predict([detectData],verbose = 0),2) * 100
        inference_window = 0

       dataBufferVal -=1 # reset data buffer values
       dataBuffer['upperRightShoulder'].pop(0) # remove first element
       dataBuffer['upperLeftShoulder'].pop(0) # remove first element
       dataBuffer['rightShoulderElbow'].pop(0)
       dataBuffer['leftShoulderElbow'].pop(0)
       dataBuffer['rightElbowWrist'].pop(0)
       dataBuffer['leftElbowWrist'].pop(0)

    else:
        dataBuffer['upperRightShoulder'].append(upperRightShoulder)
        dataBuffer['upperLeftShoulder'].append(upperLeftShoulder)
        dataBuffer['rightShoulderElbow'].append(rightShoulderElbow)
        dataBuffer['leftShoulderElbow'].append(leftShoulderElbow)
        dataBuffer['rightElbowWrist'].append(rightElbowWrist)
        dataBuffer['leftElbowWrist'].append(leftElbowWrist)  
        
        dataBufferVal += 1

    inference_window +=1
    return waveDetect
    
def save_to_csv(data):
    daraFrame = pd.DataFrame(data)
    daraFrame.to_csv("train_wave.csv")

def save_to_numpy(data):
    daraFrame = pd.DataFrame(data)
    df = daraFrame.to_numpy()
    np.save("train_wave_npy7.npy",df)
    #np.savez("train_wave_savez.npz",df)

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

def wave_detection(keyPoints,waveCounter,initialState):
    wave = False
    toggleAngle = 115 # state switching angle
    waveSpaceAngles = [250,30] # wave detection lower threshold and hand cross threshold
    handWaved = 4 # time the hand is waved
    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]


    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    angleA = 360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) 
    angleB = calculate_angle(right_shoulder,left_shoulder,left_wrist)
    angleAvg = int((angleA + angleB ) / 2 )

    #print(angleAvg)
    if angleA < waveSpaceAngles[0]  and angleB < waveSpaceAngles[0] and angleA > waveSpaceAngles[1] and angleB > waveSpaceAngles[1]: # wave detection space
        if (angleA >= (angleB - 20) and angleA <= (angleB + 20)) and( angleA >= (angleB - 20) and angleA <= (angleB + 20)): # halleluya X
            if angleAvg >= toggleAngle: # initial sate -> open
                currentState = 0
            elif angleAvg <= toggleAngle: # initial sate -> close
                currentState = 1
            else:
                currentState = None
                wave = False
                #waveCounter = 0 # resets in check in every frame
            
            if currentState == 1 and initialState == 0: # open to close
                waveCounter += 1
            if currentState == 0 and initialState == 1: # open to close
                waveCounter += 1
        else:
            currentState = None
    else:
        currentState = None # anything but open or close
        wave = False
        waveCounter = 0
        
    if waveCounter >= handWaved:
        wave = True

    initialState = currentState
    # print(f"count:{waveCounter} state:{initialState}\n")
    return waveCounter, initialState, wave