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

    print(results)
    


    face_default = 0
    max_width = None

    for result in results:
        
        face = result[0]
        max_face_idx = np.argmax((face[2] - face[0])* (face[3] - face[1]))

        max_face = face[max_face_idx]

        if max_face > face_default:
            face_default = max_face
            max_width = face

    print(max_width)


    #시각화

    color = (0, 0, 255)
    cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), color, 2)


        
    cv2.imwrite('/workspace/retinaface-pytorch-inference/one_detect_results/' + image, img)
