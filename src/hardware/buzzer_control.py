import gpiod
import time

CHIP = 'gpiochip0'
BUZZER_PIN = 18

chip = gpiod.Chip(CHIP)
line = chip.get_line(BUZZER_PIN)

def setup():
    """Request the line as an output to initialize the buzzer."""
    line.request(consumer="buzzer_control", type=gpiod.LINE_REQ_DIR_OUT)
    print("Buzzer setup complete.")

def synchronized_buzz(total_duration=3):
    """
    Buzz in sync with the LED: 4 times slow, 4 times faster, 4 times even faster,
    4 times even faster, all at 1 kHz.
    
    :param total_duration: Total duration of the event in seconds
    """
    # Divide the total duration into four equal parts, minus 2 seconds for the solid hold
    part_duration = (total_duration) / 4

    # Calculate the pulse durations
    slow_pulse_duration = part_duration / 4
    medium_pulse_duration = slow_pulse_duration / 2
    fast_pulse_duration = medium_pulse_duration / 2
    fastest_pulse_duration = fast_pulse_duration / 2

    # Define the PWM frequency and period
    frequency = 1000  # 1 kHz
    period = 1.0 / frequency
    half_period = period / 2

    def play_pwm_for_duration(duration):
        end_time = time.time() + duration
        while time.time() < end_time:
            line.set_value(1)  # Turn the buzzer on
            time.sleep(half_period)
            line.set_value(0)
            time.sleep(half_period)

    # Perform 4 slow buzzes
    for _ in range(4):
        play_pwm_for_duration(slow_pulse_duration)
        time.sleep(slow_pulse_duration)

    # Perform 4 medium buzzes
    for _ in range(4):
        play_pwm_for_duration(medium_pulse_duration)
        time.sleep(medium_pulse_duration)

    # Perform 4 fast buzzes
    for _ in range(4):
        play_pwm_for_duration(fast_pulse_duration)
        time.sleep(fast_pulse_duration)

    # Perform 4 fastest buzzes
    for _ in range(4):
        play_pwm_for_duration(fastest_pulse_duration)
        time.sleep(fastest_pulse_duration)

def buzz(time_arg=2):
    time.sleep(0.3)

    # Define the PWM frequency and period
    frequency = 1000  # 1 kHz
    period = 1.0 / frequency
    half_period = period / 2

    # Hold the buzzer solid for default 2 seconds
    end_time = time.time() + time_arg
    while time.time() < end_time:
        line.set_value(1)  # Turn the buzzer on
        time.sleep(half_period)
        line.set_value(0)
        time.sleep(half_period)


def cleanup():
    """Release the line to free the GPIO resources."""
    line.release()
    # print("Buzzer cleanup complete.")
