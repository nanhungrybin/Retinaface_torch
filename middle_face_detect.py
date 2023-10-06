import os
import cv2
from detector import Retinaface_Detector
import numpy as np


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



#detector = Retinaface_Detector()
# can be used to update some params on runtime, even model can be reloaded, if different file is provided
# detector.set_params(params)

test_images = os.listdir('/workspace/retinaface-pytorch-inference/test_images')

cnt = 0
for image in test_images:
    imgpath = os.path.join('/workspace/retinaface-pytorch-inference/test_images', image)

    img = cv2.imread(imgpath)
    results = detector.detect(img, threshold=None) # if None, default threshold from params is used

    #print(results)
    

    if results:
        # 중앙 좌표 계산
        middle_x = img.shape[1] // 2
        middle_y = img.shape[0] // 2
        min_distance = float('inf') #처음에는 어떤 거리 값보다 큰 값을 가지게 
        closest_face = None

        for result in results:
            face = result[0]
            face_x = (face[0] + face[2]) // 2
            face_y = (face[1] + face[3]) // 2
            distance = ((middle_x - face_x) ** 2 + (middle_y - face_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_face = face

              
        # 중앙에 가장 가까운 얼굴 바운딩 박스만 표시
        if closest_face is not None:
            color = (0, 0, 255)
            cv2.rectangle(img, (closest_face[0], closest_face[1]), (closest_face[2], closest_face[3]), color, 2)
        


        
    cv2.imwrite('/workspace/retinaface-pytorch-inference/middle_detect_results/' + image, img)
