from api_usage import face_alignment, face_detect
print(123)
def run_detect_and_alignment(image_path):
    # detect
    detect_res = face_detect.run(image_path)

    # alignment
    landmarks = face_alignment.run(image_path, detect_res)

    print(landmarks)