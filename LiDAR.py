import serial
import time

ser = serial.Serial("/dev/serial0", 115200, timeout=1)

# Set your detection threshold here (in cm)
DISTANCE_THRESHOLD = 100  

def read_lidar():
    while True:
        if ser.read() == b'\x59':      # start byte 1
            if ser.read() == b'\x59':  # start byte 2
                data = ser.read(7)
                if len(data) < 7:
                    continue  # incomplete frame, discard
                
                frame = b'\x59\x59' + data
                checksum = sum(frame[0:8]) & 0xFF
                if checksum != frame[8]:
                    print("Checksum mismatch, discarding frame")
                    continue
                
                distance = frame[2] + (frame[3] << 8)
                strength = frame[4] + (frame[5] << 8)
                
                if distance > 0 and distance < DISTANCE_THRESHOLD:
                    print(f"Object detected! Distance: {distance} cm | Strength: {strength}")
                else:
                    print(f"No object within {DISTANCE_THRESHOLD} cm. Distance: {distance} cm")

                time.sleep(0.1)

try:
    read_lidar()
except KeyboardInterrupt:
    print("Stopping...")
finally:
    ser.close()
