import RPi.GPIO as GPIO
import time
import serial
import threading

# ----------------- SERVO SETUP -------------------
SERVO_PIN = 12  # GPIO 12 (Pin 32)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(0)

def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.2)
    pwm.ChangeDutyCycle(0)

def servo_loop():
    while True:
        set_angle(10)    # Full Left
        time.sleep(0.1)
        set_angle(170)   # Full Right
        time.sleep(0.1)

# ----------------- LIDAR SETUP -------------------
ser = serial.Serial("/dev/serial0", 115200, timeout=1)
DISTANCE_THRESHOLD = 100  # cm

def lidar_loop():
    while True:
        if ser.read() == b'\x59':  # Header 1
            if ser.read() == b'\x59':  # Header 2
                data = ser.read(7)
                if len(data) != 7:
                    print("❌ Incomplete LIDAR frame received. Skipping.")
                    continue

                distance = data[0] + (data[1] << 8)
                strength = data[2] + (data[3] << 8)
                received_checksum = data[6]

                checksum = (0x59 + 0x59 + sum(data[:6])) & 0xFF
                if checksum != received_checksum:
                    print("❌ LIDAR checksum mismatch. Skipping.")
                    continue

                # Print the detection result
                if 0 < distance < DISTANCE_THRESHOLD:
                    print(f"🔴 Object Detected! Distance: {distance} cm | Strength: {strength}")
                else:
                    print(f"✅ Clear. Distance: {distance} cm")

                time.sleep(0.1)

# ----------------- MAIN -------------------
try:
    # Start both threads
    servo_thread = threading.Thread(target=servo_loop, daemon=True)
    lidar_thread = threading.Thread(target=lidar_loop, daemon=True)

    servo_thread.start()
    lidar_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Stopping program...")

finally:
    pwm.stop()
    GPIO.cleanup()
    ser.close()
