import cv2
import time
from picamera2 import Picamera2, Preview

# Initialize the camera
#camera = cv2.VideoCapture(0)  # Use the appropriate index for your camera
picam2 = Picamera2()

# Testing with picam instead of cv2
def setup_camera_picam():
    cam_config = picam2.create_preview_configuration()
    picam2.configure(cam_config)
    picam2.start_preview(Preview.NULL) # use (Preview.QTGL) when have display to see preview
    picam2.start()
    print("Camera setup complete.")
    time.sleep(2)
    picam2.capture_file("test.jpg")


#def setup_camera():
  #  """Set up the camera."""
  #  if not camera.isOpened():

        # Debug print statement
  #      print("Error: Camera could not be opened.")
        #lower resolution if unopened 
  #      camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  #      camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  #      camera.open(0)  # Reopen the camera if it was closed
  #  print("Camera setup complete.")

def cleanup_camera():
    """Release the camera resource."""
    picam2.stop()
    picam2.close()
    print("Camera cleanup complete.")

def generate_frames():
    """Generate frames from the camera and encode them as JPEG."""
    while True:
        success, frame = picam2.capture_array()
        if frame is None:
            break
        else:
            # Resize the frame if needed (using OpenCV)
            # frame = cv2.resize(frame, (640, 480))  # Example resize to 640x480

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
