"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')

import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.safe_load(f)

def run(image, detect_res, scene):
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    print('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        print('Failed to parse model configuration file!')
        print(e)
        sys.exit(-1)
    else:
        print('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        print('Model loading failed!')
        print(e)
        sys.exit(-1)
    else:
        print('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    # read image
    # image_path = 'api_usage/test_images/test1.jpg'
    # image_det_txt_path = 'api_usage/test_images/test1_detect_res.txt'
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if type(image) is not np.ndarray:
      image = np.array(image)
    else:
      img_float32 = np.float32(image)
      image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
    image = cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)
    # with open(image_det_txt_path, 'r') as f:
        # lines = f.readlines()
    detect_res_lines = detect_res
    lines = []
    try:
        for i, line in enumerate(detect_res_lines):
            line = line.strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            # save_path_img = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.jpg'
            # save_path_txt = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.txt'
            image_show = image.copy()
            # with open(save_path_txt, "w") as fd:
            for (x, y) in landmarks.astype(np.int32):
                cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                line = str(x) + ' ' + str(y) + ' '
                lines.append(line)
            # cv2.imwrite(save_path_img, image_show)
    except Exception as e:
        print('Face landmark failed!')
        print(e)
        sys.exit(-1)
    else:
        print('Successful face landmark!')
        return lines
