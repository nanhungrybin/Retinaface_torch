import os
import cv2
from detector import Retinaface_Detector
import numpy as np
import pandas as pd
import logging

# 로그 설정
log_filename = 'faceerror_log.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


vid_error = []



# default parameters ()
params = {'mobnet_weights_file':'/workspace/retinaface-pytorch-inference/mnet.25.pth',
          'threshold': 0.4,
          'pixel_means': [0,0,0],
          'pixel_stds': [1.0,1.0,1.0],
          'pixel_scale': 1.0,
          'target_size': 1024 ,#240, #1024, # size to resize smaller dimension of input piture
          'max_size': 1980 #270 #1980 # size to further resize bigger dimension it its too big
    }

detector = Retinaface_Detector(params)

# 디렉토리 경로 설정
directory_path = '/workspace/retinaface-pytorch-inference/mosei_down_videos'

# 디렉토리 내의 파일 목록 읽기
#video_files = [f for f in os.listdir(directory_path)]
video_files = "bUFAN2TgPaU_4.mp4"


video_file_path = os.path.join(directory_path, video_files)

# Initialize variables for each video file
no_face_ranges = []  # List to store the ranges of frames without faces
start_frame = None  # Initialize the start frame variable
read_error = []
#no_face = []

# VideoCapture를 사용하여 동영상 파일 열기
cap = cv2.VideoCapture(video_file_path)

# 비디오 저장을 위한 설정
output_path = '/workspace/retinaface-pytorch-inference/test_reface_result/' + video_files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 비디오의 프레임 속도 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 각 frame별 동영상 처리 코드 추가
while True:
    ret, frame = cap.read()

    if not ret:
        read_error.append(video_file_path)
        break

    # detect face
    results = detector.detect(frame, threshold=None) # if None, default threshold from params is used

    if results: #face detect
        if start_frame is not None:
            # If a face is detected after a sequence of non-face frames, record the end frame of the non-face range
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            no_face_ranges.append((start_frame, end_frame))
            start_frame = None  # Reset the start frame



        for result in results:
            face = result[0]


            #시각화
            color = (0, 0, 255)
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), color, 2)
    
            
            

    else:  # No face detected in the frame
        if start_frame is None:
            # Set the start frame of the non-face range
            start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            #no_face.append(video_file_path)


    # 비디오 프레임을 저장
    out.write(frame)



# ends with a non-face range, record it
if start_frame is not None:
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    no_face_ranges.append((start_frame, end_frame))

# percentage of frames without faces
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frames_without_faces = sum(end - start + 1 for start, end in no_face_ranges)

if total_frames > 0:
    percentage_without_faces = (frames_without_faces / total_frames) * 100

    if percentage_without_faces > 0:
        print(f"No Face: {video_file_path} & No Face Percentage: {percentage_without_faces:.2f}%")
        logging.error(f"No Face: {video_file_path} & No Face Percentage: {percentage_without_faces:.2f}%")

        # Print the ranges of frames without faces
        for start, end in no_face_ranges:
            print(f"& No Face Frame Range: {start}-{end}")
            logging.error(f"& No Face Frame Range: {start}-{end}")



# 사용한 자원 해제
cap.release()
out.release()

# 로그 파일 닫기
logging.shutdown()


print(len(read_error))

# 리스트를 DataFrame으로 변환
df = pd.DataFrame({'Video Name': vid_error})

# DataFrame을 CSV 파일로 저장
df.to_csv('face_error_range.csv', index=False)
