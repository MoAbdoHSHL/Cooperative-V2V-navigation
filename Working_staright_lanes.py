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
import paho.mqtt.client as mqtt


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'


class_names = ['pathholde', 'Manholde', 'Duck', 'Snow', 'Stop']
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 13
SERVO_PIN = 12
RIGHT_MOTOR_FACTOR = 1.5


BROKER_IP = "172.20.10.13"  # Pi_2 (broker)
TOPIC = "servo/control"
# Lane memory
last_center = None
last_lane_width = None
last_confidence = 0.0


GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)
current_angle = 10


GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)


scan_data = {}
data_lock = threading.Lock()
target_angle = 10
ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=0.1)
HEADER = 0x59
object_detected = False
stop_requested = False
running = True


mqtt_lock = threading.Lock()
last_detected_class = None
new_detection = False


def motor_forward(speed_left=18, speed_right=18):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)   # Left motor forward
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)   # Right motor forward
    pwmA.ChangeDutyCycle(speed_left)
    pwmB.ChangeDutyCycle(min(speed_right * RIGHT_MOTOR_FACTOR, 100))

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    time.sleep(0.5)
    
def motor_turn_right(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)     # Left motor OFF
    pwmA.ChangeDutyCycle(0)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)    # Right motor forward
    pwmB.ChangeDutyCycle(min(speed * RIGHT_MOTOR_FACTOR, 100))

def motor_turn_left(speed=75):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)    # Left motor forward
    pwmA.ChangeDutyCycle(speed)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)     # Right motor OFF
    pwmB.ChangeDutyCycle(0)


def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.2)
    pwm.ChangeDutyCycle(0)

def servo_loop():
    global target_angle, running
    while running:
        target_angle = 10
        set_angle(10)
        time.sleep(0.1)
        target_angle = 170
        set_angle(170)
        time.sleep(0.1)


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
    last_target = target_angle
    movement_start_time = time.time()
    start_angle = current_angle
    while running:
        try:
            dist, strength = read_frame()
            if target_angle != last_target:
                movement_start_time = time.time()
                start_angle = current_angle
                last_target = target_angle
            movement_time = time.time() - movement_start_time
            if movement_time < 0.2:
                progress = movement_time / 0.2
                current_angle = start_angle + (target_angle - start_angle) * progress
            else:
                current_angle = target_angle
            if dist >= 5 and dist <= 400:
                with data_lock:
                    scan_data[current_angle] = {
                        'distance': min(dist, 200),
                        'strength': strength,
                        'timestamp': time.time()
                    }
            if 0 < dist <= 30:
                stop_requested = True
            else:
                stop_requested = False
            time.sleep(0.002)
        except Exception as e:
            print(f"LIDAR error: {e}")
            time.sleep(0.01)


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


def setup_mqtt():
    client = mqtt.Client()
    client.connect(BROKER_IP, 1883, 60)
    return client

def mqtt_publisher_thread():
    global last_detected_class, new_detection, running
    client = setup_mqtt()
    while running:
        with mqtt_lock:
            if new_detection and last_detected_class:
                client.publish(TOPIC, last_detected_class)
                print(f"MQTT sent: {last_detected_class}")
                new_detection = False
        time.sleep(0.1)
    client.disconnect()

def process_frame(frame, model):
    global object_detected, stop_requested, last_detected_class, new_detection
    
    results = model(frame, imgsz=160, conf=0.5, half=True, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    current_detections = set()
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            class_name = class_names[cls_id]
            current_detections.add(class_name)
            
            label = f"{class_name} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
 
    with mqtt_lock:
        if "pathholde" in current_detections:
            last_detected_class = "pathholde"
            new_detection = True
        elif current_detections:  # Other objects
            last_detected_class = list(current_detections)[0]  # Send first detected class
            new_detection = True
        else:  # No detections
            last_detected_class = "clear"
            new_detection = True
    
    return annotated_frame

############ here ######
def preprocess_frame(frame):
    """Convert frame to HSV and threshold for white lane lines."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def region_of_interest(mask):
    """Apply ROI mask to focus on road region."""
    height, width = mask.shape[:2]
    roi_mask = np.zeros_like(mask)

    polygon = np.array([[
        (int(0.00 * width), height),
        (int(0.30 * width), int(0.25 * height)),
        (int(0.70 * width), int(0.25 * height)),
        (int(1.00 * width), height)
    ]], np.int32)

    cv2.fillPoly(roi_mask, polygon, 255)
    masked_img = cv2.bitwise_and(mask, roi_mask)
    return masked_img, polygon


def clean_mask(masked_img):
    """Reduce noise using morphology."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, kernel)
    return cleaned_mask


def detect_lines(cleaned_mask):
    """Run Hough Transform to detect lane lines."""
    lines = cv2.HoughLinesP(
        cleaned_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=90,
        maxLineGap=8
    )
    return lines


def separate_lanes(lines, debug_img):
    """Classify lines into left and right lanes by slope."""
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if slope < -0.3:  # left lane
                    left_lines.append((slope, intercept))
                    cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                elif slope > 0.3:  # right lane
                    right_lines.append((slope, intercept))
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return left_lines, right_lines


def calculate_steering(left_lane, right_lane, width, height):
    """Compute steering angle based on lane positions."""
    steering_angle = 90
    both_lanes_detected = left_lane is not None and right_lane is not None
    left_lane_detected = left_lane is not None
    right_lane_detected = right_lane is not None

    if both_lanes_detected:
        slope_left, intercept_left = left_lane
        slope_right, intercept_right = right_lane
        y = height
        x_left = (y - intercept_left) / slope_left
        x_right = (y - intercept_right) / slope_right
        center = (x_left + x_right) / 2
        deviation = (center - width / 2) / (width / 2)
        steering_angle = 90 + deviation * 30
    elif left_lane_detected:
        steering_angle = 105
    elif right_lane_detected:
        steering_angle = 75

    return steering_angle, both_lanes_detected, left_lane_detected, right_lane_detected


    
def detect_lanes(frame):
    """Main lane detection pipeline (uses sub-functions)."""
    mask = preprocess_frame(frame)
    masked_img, polygon = region_of_interest(mask)
    cleaned_mask = clean_mask(masked_img)

    debug_img = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    cv2.polylines(debug_img, polygon, isClosed=True, color=(0, 255, 0), thickness=2)

    lines = detect_lines(cleaned_mask)
    left_lines, right_lines = separate_lanes(lines, debug_img)

    left_lane = np.average(left_lines, axis=0) if left_lines else None
    right_lane = np.average(right_lines, axis=0) if right_lines else None

    height, width = cleaned_mask.shape[:2]
    steering_angle, both, left, right = calculate_steering(left_lane, right_lane, width, height)

    return steering_angle, debug_img, both, left, right

##########################





def main():
    global running, stop_requested, object_detected
    
   
    mqtt_thread = threading.Thread(target=mqtt_publisher_thread)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    camera = initialize_camera()
    model = initialize_model()
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
        base_speed = 15
        steer_speed = 18
        
        while True:
            frame = frame_thread.get_frame()
            if frame is None:
                continue
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            start_time = time.time()
            annotated = process_frame(frame, model)
            print(f"Inference time: {time.time() - start_time:.2f} sec")
            
            # Get lane detection results
            steering_angle, lane_debug, both_lanes, left_lane, right_lane = detect_lanes(frame)
            
            if not stop_requested:
                # Drive forward if both lanes detected
                if both_lanes:
                    motor_forward(base_speed, base_speed)
                # Turn left if only left lane detected
                elif left_lane:
                    motor_turn_right(steer_speed)
                    time.sleep(0.05)
                # Turn right if only right lane detected
                elif right_lane:
                    motor_turn_left(steer_speed)
                    time.sleep(0.05)
                # Default behavior if no lanes detected
                else:
                    if steering_angle < 85:
                        motor_turn_right(steer_speed)
                        time.sleep(0.05)
                    elif steering_angle > 95:
                        motor_turn_left(steer_speed)
                        time.sleep(0.05)
                    else:
                        motor_forward(base_speed, base_speed)
            else:
                motor_stop()
                if not object_detected:
                    stop_requested = False
                    time.sleep(1)
                else:
                    print("Object detected - stopped!")
                    time.sleep(3)
                    object_detected = False
                    stop_requested = False
            cv2.imshow("Road Hazard Detection", annotated)
            cv2.imshow("Lane Detection", lane_debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
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
