import cv2

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    ret, frame = camera.read()
    if ret:
        print("Frame captured successfully.")
    else:
        print("Error: Could not read frame.")

camera.release()
