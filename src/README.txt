Running the code:
'''cd running_code'''
'''python web_server/app.py'''

1. Core Components and Structure
Main Program: Acts as the entry point and orchestrates the different parts of your project.
Model Module: Handles loading the PyTorch model, image processing, and inference.
Hardware Control Module: Manages the LED, buzzer, and servo control.
Web Server Module: Hosts the web interface and handles real-time data updates and user interactions.
Camera Module: Captures video frames from the camera and serves the live feed to the web server.
Database/State Manager: Keeps track of the model's outputs and the number of detections for decision-making (like a simple in-memory store).

-------------------------------------------------------------------------------------------------------------------------------------------------
2. Technology Stack & Implementation Guide
Web Server:

Framework: Use Flask or FastAPI to build the web server. These frameworks are lightweight and work well on a Raspberry Pi.
Live Camera Feed: Use OpenCV to capture the feed and stream it via Flask. You can use an HTTP multipart stream for efficiency.
Graph Updates: Use WebSockets (via Flask-SocketIO) to update the graph in real time on the web page.
Model Inference:

Load your PyTorch model at startup.
Use a loop to periodically capture frames from the camera and run them through the model. Keep track of the results using a simple list or data structure.
Trigger the deterrent if there are 3 out of 5 hits.
Hardware Control:

Use the RPi.GPIO library or pigpio for precise control of the LED, buzzer, and servo.
LED Flash: Implement using a function that sets the GPIO pin high and low with appropriate delays.
Buzzer Sound: Use a similar approach as the LED, but you may need to adjust for the specific sound pattern you want.
Servo Movement: Use the pigpio library for accurate PWM control if you need precise servo positioning.
Manual Activation:

Create an endpoint in your web server to handle the manual activation request.
Use the hardware control module to trigger the deterrent actions.
Graph and Data Visualization:

Use a JavaScript library like Chart.js or Plotly for the graph on the web page.
Update the graph in real time using WebSockets to send data from the server.

------------------------------------------------------------------------------------------------------------------------------------------------------------
3. Folder Structure
/running_code
│
├── main.py                   # Entry point for the program
├── web_server/
│   ├── app.py                # Flask or FastAPI app
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, and image files
│
├── hardware/
│   ├── led_control.py        # Functions to control the LED
│   ├── buzzer_control.py     # Functions to control the buzzer
│   └── servo_control.py      # Functions to control the servo
│
├── model/
│   ├── inference.py          # Code to load the model and run inference
│   └── model.pth             # Pre-trained PyTorch model file
│
├── camera/
│   └── camera_feed.py        # Code to capture and stream video
│
├── utils/
│   └── state_manager.py      # Keeps track of model hits and other state data
└── requirements.txt          # Dependencies for the project
