from http.client import IM_USED
from api_usage import face_alignment, face_detect
from fma_3d import add_mask_one
from PIL import Image


def run(image_path):
    image = Image.open(image_path)

    # detect
    detect_res = face_detect.run(image)

    # alignment
    landmarks = face_alignment.run(image, detect_res)
    landmarks_str = ''.join(landmarks[:106]).strip()

    # add mask
    new_image = add_mask_one.run(image, landmarks_str)

    return new_image