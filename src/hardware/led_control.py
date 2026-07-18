import gpiod
import time

CHIP = 'gpiochip0'
LED_PIN = 17

chip = gpiod.Chip(CHIP)
line = chip.get_line(LED_PIN)

def setup():
    """Request the line as an output to initialize the LED."""
    line.request(consumer="led_control", type=gpiod.LINE_REQ_DIR_OUT)
    print("LED setup complete.")

def synchronized_flash(total_duration=3):
    """
    Flash the LED in sync with the buzzer: 4 times slow, 4 times faster, 4 times even faster,
    4 times even faster
    
    :param total_duration: Total duration of the event in seconds
    """
    # Divide the total duration into four equal parts, minus 2 seconds for the solid hold
    part_duration = (total_duration) / 4

    # Calculate the pulse durations
    slow_pulse_duration = part_duration / 4
    medium_pulse_duration = slow_pulse_duration / 2
    fast_pulse_duration = medium_pulse_duration / 2
    fastest_pulse_duration = fast_pulse_duration / 2

    # Perform 4 slow flashes
    for _ in range(4):
        line.set_value(1)
        time.sleep(slow_pulse_duration)
        line.set_value(0)
        time.sleep(slow_pulse_duration)

    # Perform 4 medium flashes
    for _ in range(4):
        line.set_value(1)
        time.sleep(medium_pulse_duration)
        line.set_value(0)
        time.sleep(medium_pulse_duration)

    # Perform 4 fast flashes
    for _ in range(4):
        line.set_value(1)
        time.sleep(fast_pulse_duration)
        line.set_value(0)
        time.sleep(fast_pulse_duration)

    # Perform 4 fastest flashes
    for _ in range(4):
        line.set_value(1)
        time.sleep(fastest_pulse_duration)
        line.set_value(0)
        time.sleep(fastest_pulse_duration)

def hold_led(time_arg=2):
    # Hold the LED solid for default 2 seconds
    time.sleep(0.3)
    line.set_value(1)
    time.sleep(time_arg)
    line.set_value(0)

def cleanup():
    """Release the line to free the GPIO resources."""
    line.release()
    # print("LED cleanup complete.")
