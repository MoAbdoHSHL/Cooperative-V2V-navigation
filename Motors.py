import RPi.GPIO as GPIO
import time

# Pin definitions
IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 23
ENA = 18
ENB = 13

GPIO.setmode(GPIO.BCM)

# Setup pins as output
for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)

# Setup PWM for speed control at 1000 Hz
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(0)
pwmB.start(0)

def motorA_forward(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)

def motorA_backward(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed)

def motorA_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)

def motorB_forward(speed=50):
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmB.ChangeDutyCycle(speed)

def motorB_backward(speed=50):
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmB.ChangeDutyCycle(speed)

def motorB_stop():
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmB.ChangeDutyCycle(0)

try:
    print("Motor A forward")
    motorA_forward(70)
    print("Motor B forward")
    motorB_forward(70)
    time.sleep(3)

    print("Motor A backward")
    motorA_backward(70)
    print("Motor B backward")
    motorB_backward(70)
    time.sleep(3)

    print("Stopping motors")
    motorA_stop()
    motorB_stop()
    time.sleep(1)

finally:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
