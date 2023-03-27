# function for extraxting data point for model training
import math
import pandas as pd

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

dataDict = {'upperRightShoulder':[], 'upperLeftShoulder':[], 'handRaise':[]}


def extract_data(keyPoints,handraise):
    global dataDict
    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]

    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    
    upperRightShoulder = int(360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) )
    upperLeftShoulder = int(calculate_angle(right_shoulder,left_shoulder,left_wrist))

    dataDict['upperRightShoulder'].append(upperRightShoulder)
    dataDict['upperLeftShoulder'].append(upperLeftShoulder)

    if handraise:
        dataDict['handRaise'].append(1) # if handraise is truee append target as 1 -> true
    else:
        dataDict['handRaise'].append(0)

    return dataDict

def save_to_csv(daraFrame):
    daraFrame = pd.DataFrame(dataDict)
    daraFrame.to_csv("train1.csv")
