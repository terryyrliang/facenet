import dlib
import cv2
import numpy as np
from configuration import configuration_service as cs

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cs.shape_68_dat)

def validate_face(image_data):
    img_gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    # dets is face numbers
    dets = detector(img_gray, 0)
    if len(dets) != 0:
        # detect faces
        for i in range(len(dets)):
            for idx, point in enumerate(dets):
                landmarks = np.matrix([[p.x, p.y] for p in predictor(image_data, dets[i]).parts()])
                lenOfLandmarks = len(landmarks)
                if (lenOfLandmarks >= 68):
                    return True
    return False

def validate_face_with_path(image_path):
    image_data = cv2.imread(image_path)
    return validate_face(image_data)