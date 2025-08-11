import os
import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
from ultralytics import YOLO
import RPi.GPIO as GPIO
import serial

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# Constants and configurations
class_names = ['pathholde', 'Manholde', 'Duck', 'Snow', 'Stop']

# ------------------ MOTOR SETUP ------------------
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 13

# ----------------- SERVO SETUP -------------------
SERVO_PIN = 12  # GPIO 12 (Pin 32)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
pwm.start(0)

current_angle = 10  # Track current angle for visualization
scan_data = {}
data_lock = threading.Lock()
target_angle = 10  # Track where servo is moving to

# ----------------- LIDAR SETUP -------------------
ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=0.1)
HEADER = 0x59

# Initialize GPIO for motors
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Initialize motor PWMs
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

# Global control variables
object_detected = False
stop_requested = False
running = True

# ----------------- MOTOR FUNCTIONS ------------------
def motor_forward(speed_left=2, speed_right=2):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed_left)
    pwmB.ChangeDutyCycle(speed_right)

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def motor_turn_left(speed_left=2, speed_right=5):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed_left)   # Left wheel slower
    pwmB.ChangeDutyCycle(speed_right)  # Right wheel faster

def motor_turn_right(speed_left=5, speed_right=2):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed_left)   # Left wheel faster
    pwmB.ChangeDutyCycle(speed_right)  # Right wheel slower

# ----------------- SERVO FUNCTIONS ------------------
def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.2)
    pwm.ChangeDutyCycle(0)

def servo_loop():
    global target_angle, running
    while running:
        target_angle = 10
        set_angle(10)    # Full Left
        time.sleep(0.1)
        
        target_angle = 170
        set_angle(170)   # Full Right
        time.sleep(0.1)

# ----------------- LIDAR FUNCTIONS ------------------
def read_frame():
    while True:
        b = ser.read(1)
        if len(b) == 0:
            continue
        if b[0] == HEADER:
            b2 = ser.read(1)
            if len(b2) == 0:
                continue
            if b2[0] == HEADER:
                frame = ser.read(7)
                if len(frame) != 7:
                    continue
                uart = [HEADER, HEADER] + list(frame)
                checksum = sum(uart[:8]) & 0xFF
                if checksum == uart[8]:
                    dist = uart[2] + (uart[3] << 8)
                    strength = uart[4] + (uart[5] << 8)
                    return dist, strength

def lidar_loop():
    global scan_data, current_angle, stop_requested, running
    
    # Simulation of servo movement for angle tracking
    last_target = target_angle
    movement_start_time = time.time()
    start_angle = current_angle
    
    while running:
        try:
            dist, strength = read_frame()
            
            # Simulate servo movement for angle estimation
            if target_angle != last_target:
                movement_start_time = time.time()
                start_angle = current_angle
                last_target = target_angle
            
            # Estimate current angle based on movement time
            movement_time = time.time() - movement_start_time
            if movement_time < 0.2:  # During the 0.2s movement period
                progress = movement_time / 0.2
                current_angle = start_angle + (target_angle - start_angle) * progress
            else:
                current_angle = target_angle
            
            # Only store valid readings (changed to >= 5 to include 5cm)
            if dist >= 5 and dist <= 400:
                with data_lock:
                    # Store only the latest reading for each angle
                    scan_data[current_angle] = {
                        'distance': min(dist, 200),
                        'strength': strength,
                        'timestamp': time.time()
                    }
            
            # Emergency stop if obstacle too close
            if 0 < dist <= 30:
                stop_requested = True
            else:
                stop_requested = False  # Clear stop if no close object
            
            time.sleep(0.002)
            
        except Exception as e:
            print(f"LIDAR error: {e}")
            time.sleep(0.01)

# ----------------- CAMERA FUNCTIONS -----------------
def initialize_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        buffer_count=3,
        transform=Transform(hflip=1, vflip=1)
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({"ScalerCrop": (0, 0, 4608, 2592)})
    return picam2

def initialize_model():
    model = YOLO("/home/mohamed/Desktop/Code/runs/detect/train2/weights/best.pt").to("cpu")
    model.fuse()
    model.amp = True
    _ = model(np.zeros((160, 160, 3), dtype=np.uint8), imgsz=160, verbose=False)
    return model

class FrameCaptureThread(threading.Thread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            frame = self.camera.capture_array()
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False

# ----------------- VISION PROCESSING -----------------
def process_frame(frame, model):
    global object_detected, stop_requested
    
    results = model(frame, imgsz=160, conf=0.5, half=True, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    
    # Check if any of our target classes are detected
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:  # Only consider confident detections
            object_detected = True
            stop_requested = True
            label = f"{class_names[cls_id]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_frame

def detect_lanes(frame):
    """Process frame to detect lanes and return steering angle"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to isolate white lanes
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
    # Apply region of interest mask (focus on lower half of image)
    height, width = thresh.shape
    mask = np.zeros_like(thresh)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2),
        (0, height // 2),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(thresh, mask)
    
    # Detect edges
    edges = cv2.Canny(masked, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=100)
    
    # Process lines to find lanes
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:  # Skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:  # Skip near-horizontal lines
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
    
    # Calculate average lines
    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None
    
    # Calculate steering angle
    steering_angle = 90  # Neutral position
    
    if left_avg is not None and right_avg is not None:
        # Both lanes detected - drive between them
        left_x = (left_avg[0] + left_avg[2]) // 2
        right_x = (right_avg[0] + right_avg[2]) // 2
        center = (left_x + right_x) // 2
        steering_angle = 90 + (center - width // 2) // 10  # Scale the offset
        
    elif left_avg is not None:
        # Only left lane detected - follow it with offset
        left_x = (left_avg[0] + left_avg[2]) // 2
        steering_angle = 90 + (left_x + 100 - width // 2) // 10  # Offset to right
        
    elif right_avg is not None:
        # Only right lane detected - follow it with offset
        right_x = (right_avg[0] + right_avg[2]) // 2
        steering_angle = 90 + (right_x - 100 - width // 2) // 10  # Offset to left
    
    # Constrain steering angle
    steering_angle = np.clip(steering_angle, 60, 120)
    
    return steering_angle, edges

# ----------------- MAIN CONTROL LOOP -----------------
def main():
    global running, stop_requested, object_detected
    
    # Initialize hardware
    camera = initialize_camera()
    model = initialize_model()
    
    # Start threads
    frame_thread = FrameCaptureThread(camera)
    frame_thread.start()
    
    servo_thread = threading.Thread(target=servo_loop)
    servo_thread.daemon = True
    servo_thread.start()
    
    lidar_thread = threading.Thread(target=lidar_loop)
    lidar_thread.daemon = True
    lidar_thread.start()
    
    try:
        frame_count = 0
        base_speed = 12  # Base motor speed
        
        while True:
            frame = frame_thread.get_frame()
            if frame is None:
                continue
                
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            # Process frame for object detection
            start_time = time.time()
            annotated = process_frame(frame, model)
            print(f"Inference time: {time.time() - start_time:.2f} sec")
            
            # Lane detection and steering
            if not stop_requested:
                steering_angle, lane_debug = detect_lanes(frame)
                
                # Adjust motor based on steering angle
                if steering_angle < 85:  # Sharp left turn
                    motor_turn_left(base_speed)
                elif steering_angle > 95:  # Sharp right turn
                    motor_turn_right(base_speed)
                else:  # Go straight
                    motor_forward(base_speed, base_speed)
            else:
                # Emergency stop
                motor_stop()
                if not object_detected:  # If stopped by LIDAR but no object detected
                    stop_requested = False  # Resume after a delay
                    time.sleep(1)
                else:  # If stopped by object detection
                    print("Object detected - stopped!")
                    time.sleep(3)  # Wait before resuming
                    object_detected = False
                    stop_requested = False
            
            # Display results
            cv2.imshow("Road Hazard Detection", annotated)
            cv2.imshow("Lane Detection", lane_debug)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    finally:
        # Cleanup
        running = False
        frame_thread.stop()
        frame_thread.join()
        motor_stop()
        pwm.stop()
        pwmA.stop()
        pwmB.stop()
        GPIO.cleanup()
        camera.stop()
        cv2.destroyAllWindows()
        ser.close()

if __name__ == "__main__":
    main()