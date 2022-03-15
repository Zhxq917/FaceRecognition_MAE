from api_usage import face_alignment, face_detect
from fma_3d import add_mask_one
from PIL import Image
import numpy as np


def run(image):
    # detect
    detect_res = face_detect.run(image, 'non-mask')

    # alignment
    landmarks = face_alignment.run(image, detect_res, 'non-mask')
    landmarks_str = ''.join(landmarks[:106]).strip()

    # add mask
    new_image = add_mask_one.run(image, landmarks_str)
    new_image = Image.fromarray((new_image * 255).astype(np.uint8))

    # detect again
    detect_res_2 = face_detect.run(new_image, 'mask')

    # alignment again
    landmarks_2 = face_alignment.run(new_image, detect_res_2, 'mask')
    landmarks_str_2 = ''.join(landmarks_2[:106]).strip()   

    print(landmarks_str_2)

    return new_image