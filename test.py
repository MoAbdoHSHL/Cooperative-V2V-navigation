import RPi.GPIO as GPIO
import time

# === Pin setup (BCM) ===
IN1, IN2 = 17, 27   # Left motor direction
IN3, IN4 = 22, 23   # Right motor direction
ENA, ENB = 18, 13   # Enable pins (PWM)

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 1000)  # Left motor
pwmB = GPIO.PWM(ENB, 1000)  # Right motor
pwmA.start(0)
pwmB.start(0)

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    print("STOP motors")

def left_motor_forward(speed=40):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed)
    print(f"LEFT motor FORWARD @ {speed}%")

def left_motor_backward(speed=40):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)
    print(f"LEFT motor BACKWARD @ {speed}%")

def right_motor_forward(speed=40):
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmB.ChangeDutyCycle(speed)
    print(f"RIGHT motor FORWARD @ {speed}%")

def right_motor_backward(speed=40):
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmB.ChangeDutyCycle(speed)
    print(f"RIGHT motor BACKWARD @ {speed}%")

try:
    print("=== MOTOR TEST START ===")

    left_motor_forward(40)
    time.sleep(1)
    motor_stop()
    time.sleep(1)

    left_motor_backward(40)
    time.sleep(1)
    motor_stop()
    time.sleep(1)

    right_motor_forward(40)
    time.sleep(1)
    motor_stop()
    time.sleep(1)

    right_motor_backward(40)
    time.sleep(1)
    motor_stop()

    print("=== TEST COMPLETE ===")

except KeyboardInterrupt:
    pass
finally:
    motor_stop()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup done")
