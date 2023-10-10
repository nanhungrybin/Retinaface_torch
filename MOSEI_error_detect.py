import os
import cv2
from detector import Retinaface_Detector
import numpy as np
import re
import pandas as pd

# 수집된 데이터 중
# 1. 사람 얼굴이 없는 동영상
# 2. 10프레임 중에 반복되는 영상 찾기
# 3. 비디오 리더기 에러 나는 동영상 검출 => moov atom not found 오류 발생: 동영상 파일의 형식 문제 또는 손상된 파일
# 목표 : 실현 가능한 데이터 리스트와 개수 찾기


vid_error = []
# 프레임 간의 최대 차이를 설정
max_frame_diff = 10

# default parameters ()
params = {'mobnet_weights_file':'/workspace/retinaface-pytorch-inference/mnet.25.pth',
          'threshold': 0.8,
          'pixel_means': [0,0,0],
          'pixel_stds': [1.0,1.0,1.0],
          'pixel_scale': 1.0,
          'target_size': 1024, # size to resize smaller dimension of input piture
          'max_size': 1980 # size to further resize bigger dimension it its too big
    }

detector = Retinaface_Detector(params)

# 디렉토리 경로 설정
directory_path = '/workspace/retinaface-pytorch-inference/mosei_down_videos'


# 디렉토리 내의 파일 목록 읽기
video_files = [f for f in os.listdir(directory_path)]

for video_file in video_files:
    video_file_path = os.path.join(directory_path, video_file)


    # VideoCapture를 사용하여 동영상 파일 열기
    cap = cv2.VideoCapture(video_file_path)

    prev_frame = None
    frame_diff_count = 0
    
    # 각 frame별 동영상 처리 코드 추가
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error reading video: {video_file_path}")

            #현재 동영상 파일의 어느 프레임에서 오류
            #print(f"Error message: {cap.get(cv2.CAP_PROP_POS_FRAME)}")

            vid_error.append(video_file_path)

            break
        
        # detect face
        results = detector.detect(frame, threshold=None) # if None, default threshold from params is used

        # no face detect
        if results == []:
            vid_error.append(video_file)
            print(f"No Face videoname : {video_file}")
            break

        # 10프레임 중에 반복되는 영상 찾기
        if prev_frame is not None: ############
            frame_diff = cv2.absdiff(prev_frame, frame)
            frame_diff_count = cv2.countNonZero(cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY))

            if frame_diff_count > max_frame_diff:
                print(f"Frame freeze detected in video: {video_file_path}")
                vid_error.append(video_file)
                break

        prev_frame = frame.copy()    
        
    # 사용한 자원 해제
    cap.release()


print(vid_error)


# 리스트를 DataFrame으로 변환
df = pd.DataFrame(vid_error)

# DataFrame 출력
print(df)
