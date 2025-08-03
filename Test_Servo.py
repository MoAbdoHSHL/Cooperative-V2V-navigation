import RPi.GPIO as GPIO
import time

SERVO_PIN = 12  # GPIO 12 (pin 32 on the Pi)

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(0)

def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.2)  # allow servo to reach position
    pwm.ChangeDutyCycle(0)  # stop PWM to prevent drift

try:
    while True:
        set_angle(10)    # Full Left
        time.sleep(0.1)

        set_angle(170)   # Full Right
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
