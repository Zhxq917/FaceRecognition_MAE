"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

from fma_3d.face_masker import FaceMasker

def run(image, face_lms_str):
    is_aug = False
    # image_path = 'Data/test-data/test1.jpg'
    # face_lms_file = 'Data/test-data/test1_landmark.txt'
    template_name = '7.png'
    masked_face_path = ''
    face_lms_str = face_lms_str.strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    new_image = face_masker.add_mask_one(image, face_lms, template_name, masked_face_path)
    return new_image
