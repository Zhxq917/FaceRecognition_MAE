"""
@author: JiXuan Xu, Jun Wang
@date: 20201016
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')

import yaml
import cv2
import numpy as np

from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.safe_load(f)
    
if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]

    print('Start to load the face recognition model...')
    # load model
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
    except Exception as e:
        print('Failed to parse model configuration file!')
        print(e)
        sys.exit(-1)
    else:
        print('Successfully parsed the model configuration file model_meta.json!')
        
    try:
        model, cfg = faceRecModelLoader.load_model()
    except Exception as e:
        print('Model loading failed!')
        print(e)
        sys.exit(-1)
    else:
        print('Successfully loaded the face recognition model!')

    # read image
    image_path = 'api_usage/test_images/test1_cropped.jpg'
    image = cv2.imread(image_path)
    faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)

    try:
        feature = faceRecModelHandler.inference_on_image(image)
    except Exception as e:
        print('Failed to extract facial features!')
        print(e)
        sys.exit(-1)
    else:
        print('Successfully extracted facial features!')

    np.save('api_usage/temp/test1_feature.npy', feature)
