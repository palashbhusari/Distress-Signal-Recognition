import argparse
import time
import math
import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def calculate_angle(pt1,pt2,pt3):
    """Calculates the angle between three points pt1(x1, y1),pt2 (x2, y2), and pt3(x3, y3)"""
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

def wave_detection(keyPoints,waveCounter,initial_state):
    wave = False
    right_wrist,left_wrist = keyPoints[4], keyPoints[7]
    right_shoulder,left_shoulder = keyPoints[2],keyPoints[5]
    if right_wrist[0] == None or left_wrist[0] == None: # check if we have wrist coordinates
        return
    angleA = 360 - calculate_angle(left_shoulder,right_shoulder,right_wrist) # angle between pt1 - pt2 - pt3 | pt = [y,x]
    angleB = calculate_angle(right_shoulder,left_shoulder,left_wrist)
    angleAvg = int((angleA + angleB ) / 2 )
    print(angleAvg)
    if angleA < 200  and angleB < 200 and angleA > 30 and angleB > 30: # above shoulder
        if angleAvg > 105: # initial sate -> open
            currentState = 0
        elif angleAvg < 110: # initial sate -> close
            currentState = 1
        else:
            currentState = 99
            wave = False
            waveCounter = 0
        
        if currentState == 1 and initial_state == 0: # open to close
            waveCounter += 1
        if currentState == 0 and initial_state == 1: # open to close
            waveCounter += 1
    else:
        currentState = 99 # anything but open or close
        wave = False
        waveCounter = 0
        
    
    if waveCounter >= 3:
        wave = True

    initial_state = currentState
    print(f"count:{waveCounter} state:{initial_state}")
    return waveCounter, initial_state, wave



def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    new_frame_time = 0 
    prev_frame_time = 0 

    #initializing starting global wave variables
    wavecounter = 0
    state = 99 # not open or close - > 0 0r 1
    

    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)


        font = cv2.FONT_HERSHEY_SIMPLEX   # font
        fontScale = 1 #fontScale
        color = (255, 255, 255)
        thickness = 2

        new_frame_time = time.time()
 # Calculating the fps
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        fps = 1/(new_frame_time-prev_frame_time)  # we will be subtracting it to get more accurate result
        prev_frame_time = new_frame_time
        fps = int(fps) # converting the fps into integer
        # Using cv2.putText() method
        cv2.putText(img, 'FPS: '+ str(fps), (50,50), font, 
                        fontScale, (255,0,0), thickness, cv2.LINE_AA)
        
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        # for pose in current_poses:  # drawing pose
        #     pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    
        for pose in current_poses:
    # order of the [y,x] pose.keypoints => [0'nose', 1'neck', 2'r_sho', 3'r_elb', 4'r_wri', 5'l_sho', 6'l_elb',7'l_wri', 8'r_hip', 
    #                            9'r_knee', 10'r_ank', 11'l_hip', 12'l_knee', 13'l_ank', 14'r_eye', 15'l_eye', 16'r_ear', 17'l_ear']
            # coordinateY = pose.keypoints[0][0] # sample - getting nose key points y coordinate
            # coordinateX = pose.keypoints[0][1] # sample - getting nose key points x coordinate 
            right_shoulderY,right_shoulderX =  pose.keypoints[2][0], pose.keypoints[2][1]
            left_shoulderY, left_shoulderX =  pose.keypoints[5][0], pose.keypoints[5][1]

            right_elbowY, right_elbowX = pose.keypoints[3][0], pose.keypoints[3][1]
            left_elbowY, left_elbowX = pose.keypoints[6][0], pose.keypoints[6][1]

            #detect right hand raise
            handraise = False
            facing_front = right_shoulderY <  left_shoulderY # check face direction
            if right_elbowX < right_shoulderX and left_elbowX < left_shoulderX and facing_front: # NOTE: change hand raise from above shoulder to above chest
                handraise = True
                cv2.putText(img, "Hand raise", (300, 50), font, 2,(0,0,255), thickness, cv2.LINE_4)

            wavecounter, state, wave = wave_detection(pose.keypoints,wavecounter,state) # hand wave detection algorithm
            if wave:
                cv2.putText(img, "Wave Detected", (300, 100), font, 2,(0,0,255), thickness, cv2.LINE_4)

############ just for refrence
            right_wrist,left_wrist = pose.keypoints[4], pose.keypoints[7]
            cv2.line(img, (right_shoulderY,right_shoulderX), (left_shoulderY, left_shoulderX), color, thickness)
            cv2.line(img, (right_shoulderY,right_shoulderX), right_wrist, color, thickness) 
            cv2.line(img, (left_shoulderY, left_shoulderX), left_wrist, color, thickness) 
################         

    # show keypoints on display
            #draw coordinates on frame
            cv2.putText(img, "R", (right_shoulderY,right_shoulderX ), font, fontScale, # Righ side inidcator
                        color, thickness, cv2.LINE_4)
            
            # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
            #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)


        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
