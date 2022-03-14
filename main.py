from api_usage import face_alignment, face_detect
from fma_3d import add_mask_one


def run(image_path):
    # detect
    detect_res = face_detect.run(image_path)

    # alignment
    landmarks = face_alignment.run(image_path, detect_res)

    # add mask
    new_image = add_mask_one.run(image_path, landmarks[0])


    return landmarks