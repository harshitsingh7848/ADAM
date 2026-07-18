from flask import Flask, render_template, Response
from picamera2 import Picamera2, Preview
import atexit
import sys
import threading
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware import led_control, buzzer_control, servo_control, camera_control

app = Flask(__name__)

# Initialize hardware components
led_control.setup()
buzzer_control.setup()
servo_pwm = servo_control.setup()
camera_control.setup_camera_picam()

# Register cleanup functions to be called on exit
atexit.register(led_control.cleanup)
atexit.register(buzzer_control.cleanup)
atexit.register(servo_control.cleanup)
atexit.register(camera_control.cleanup_camera)


@app.route('/')
def index():
    return render_template('index.html')  # This will be the main HTML page

@app.route('/video_feed')
def video_feed():
    return Response(camera_control.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activate-deterrent', methods=['GET'])
def activate_deterrent():
    # COUNT DOWN 3 SECONDS
    # Create threads to run the LED and buzzer functions simultaneously
    led_thread = threading.Thread(target=led_control.synchronized_flash)
    buzzer_thread = threading.Thread(target=buzzer_control.synchronized_buzz)

    # Start the threads
    led_thread.start()
    buzzer_thread.start()

    # Wait for both threads to finish
    led_thread.join()
    buzzer_thread.join()

    # INITIATE
    # Create threads to run the LED and buzzer functions simultaneously
    led_thread = threading.Thread(target=led_control.hold_led)
    buzzer_thread = threading.Thread(target=buzzer_control.buzz)
    # servo_thread = threading.Thread(target=servo_control.move_servo)

    # Start the threads
    led_thread.start()
    buzzer_thread.start()
    # servo_thread.start()

    '''DETERENT ACTIVATED NOTIFY?'''

    # Wait for both threads to finish (will happen after 2 seconds or unknown servo timing)
    led_thread.join()
    buzzer_thread.join()
    # servo_thread.join()

    return "Deterrent activated!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5092, debug=False)
