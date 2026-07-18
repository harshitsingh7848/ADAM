import gpiod
import time

CHIP = 'gpiochip0'
SERVO_PIN = 26

chip = gpiod.Chip(CHIP)
line = chip.get_line(SERVO_PIN)

# print(f"servo_control: Initialized line for SERVO_PIN {SERVO_PIN}: {line}")

def setup():
    line.request(consumer="servo_control", type=gpiod.LINE_REQ_DIR_OUT)

def move_servo(angle=90):
    return
    # # Convert angle to duty cycle range (e.g., 0.5ms - 2.5ms pulse width)
    # duty_cycle = (angle / 180.0) * 2.0 + 0.5
    # pulse_width = duty_cycle / 1000.0

    # # Generate PWM signal for a short period to move the servo
    # for _ in range(50):  # Generate 50 pulses
    #     line.set_value(1)
    #     time.sleep(pulse_width)
    #     line.set_value(0)
    #     time.sleep(0.02 - pulse_width)

def cleanup():
    line.release()  # Release the line
