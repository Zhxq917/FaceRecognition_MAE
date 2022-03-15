from api_usage import face_alignment, face_detect
from fma_3d import add_mask_one
from PIL import Image


def run(image):

    # detect
    detect_res = face_detect.run(image)

    # alignment
    landmarks = face_alignment.run(image, detect_res)
    landmarks_str = ''.join(landmarks[:106]).strip()

    # add mask
    new_image = add_mask_one.run(image, landmarks_str)

    # detect again
    detect_res_2 = face_detect.run(new_image)

    # alignment again
    landmarks_2 = face_alignment.run(new_image, detect_res_2)
    landmarks_str_2 = ''.join(landmarks_2[:106]).strip()   

    print(landmarks_str_2)

    return new_image