import os
import cv2
from detector import Retinaface_Detector
import numpy as np
import re
import pandas as pd
import logging
from queue import Queue

# 로그 설정
log_filename = '10-13_log.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 수집된 데이터 중
# 1. 사람 얼굴이 없는 동영상
# 2. 10프레임 반복되는 영상 찾기
# 3. 비디오 리더기 에러 나는 동영상 검출 => moov atom not found 오류 발생: 동영상 파일의 형식 문제 또는 손상된 파일
# 목표 : 실현 가능한 데이터 리스트와 개수 찾기

vid_error = []

# 프레임 간의 최대 차이를 설정
# max_frame_diff 개의 이전 프레임만 비교하도록
max_frame_diff = 24 #15


# default parameters ()
params = {'mobnet_weights_file':'/workspace/retinaface-pytorch-inference/mnet.25.pth',
          'threshold': 0.8,
          'pixel_means': [0,0,0],
          'pixel_stds': [1.0,1.0,1.0],
          'pixel_scale': 1.0,
          'target_size': 240, #1024, # size to resize smaller dimension of input piture
          'max_size': 270 #1980 # size to further resize bigger dimension it its too big
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

    ######## 이전 max_frame_diff개의 프레임을 저장 위한 QUEUE #######
    prev_frames = Queue()


    no_face_ranges = []  # List to store the ranges of frames without faces
    start_frame = None  # Initialize the start frame variable
    
    # 각 frame별 동영상 처리 코드 추가
    while True:
        ret, frame = cap.read()
        
        # video reading error
        read_error = []

        if not ret: #False
            read_error.append(video_file_path)

            if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0: #for division error

                if read_error and len(read_error) / cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0.5: #because one of video after 148 frame error


                # 예외처리: "mmco: unref short failure" 에러가 발생할 때만 해당 동영상 파일을 vid_error 리스트에 추가
                    if "mmco: unref short failure" in str(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                        # 터미널 출력 내용을 로그 파일에도 기록
                        logging.error(f"Error reading video: {video_file_path}")
                        print(f"Error reading video: {video_file_path}")
                        vid_error.append(video_file_path)
                    else:
                        # 터미널 출력 내용을 로그 파일에도 기록
                        logging.error(f"Error reading video: {video_file_path}")
                        print(f"Error reading video: {video_file_path}")
                        vid_error.append(video_file_path)

            elif cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:

                if "mmco: unref short failure" in str(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                        # 터미널 출력 내용을 로그 파일에도 기록
                        logging.error(f"Error reading video: {video_file_path}")
                        print(f"Error reading video: {video_file_path}")
                        vid_error.append(video_file_path)
                else:
                    # 터미널 출력 내용을 로그 파일에도 기록
                    logging.error(f"Error reading video: {video_file_path}")
                    print(f"Error reading video: {video_file_path}")
                    vid_error.append(video_file_path)


            break


        # 10프레임 사이 반복되는 영상 찾기

        # 이전 프레임을 큐에 추가
        # prev_frames.append(frame.copy())
        # 현재 프레임 추가
        prev_frames.put(frame)


        # 큐의 길이가 max_frame_diff를 초과하면 가장 오래된 프레임을 제거
        if prev_frames.qsize() > max_frame_diff:
            prev_frames.get()

        if prev_frames.qsize() == max_frame_diff:
            frames_equal = all(np.array_equal(prev_frames.queue[0], prev_frames.queue[i]) for i in range(1, max_frame_diff))
            
            if frames_equal:
                # 10 프레임이 모두 동일하면 반복된 것으로 간주
                print(f"Frame freeze detected in video: {video_file_path}")
                vid_error.append(video_file_path)
                #read_error.append(video_file)
                logging.error(f"Frame freeze detected in video: {video_file_path}")
                break

        
        no_face = []
        # 전체 프레임 중 중간에 얼굴이 나올경우

        # detect face
        results = detector.detect(frame, threshold=None) # if None, default threshold from params is used

        if results: #face detect

            if start_frame is not None:
                # If a face is detected after a sequence of non-face frames, record the end frame of the non-face range
                end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                no_face_ranges.append((start_frame, end_frame))
                start_frame = None  # Reset the start frame

        else:  # No face detected in the frame

            no_face.append(video_file_path)

            if start_frame is None:
                # Set the start frame of the non-face range
                start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            #no_face_detected = True

            # if no_face_detected:
            #     no_face.append(video_file)


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

# 로그 파일 닫기
logging.shutdown()


print(len(read_error))
print(len(no_face))
print(len(vid_error))



# 리스트를 DataFrame으로 변환
df = pd.DataFrame({'Video Name': vid_error})

# DataFrame을 CSV 파일로 저장
df.to_csv('vid_error.csv', index=False)
