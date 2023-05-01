# Wave detection helper functions

import math

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