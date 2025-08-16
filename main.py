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
class_names = ['pathholde', 'Manholde', 'Duck', 'Snow', 'Stop']
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 13
SERVO_PIN = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)
current_angle = 10
scan_data = {}
data_lock = threading.Lock()
target_angle = 10
ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=0.1)
HEADER = 0x59
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
object_detected = False
stop_requested = False
running = True
RIGHT_MOTOR_FACTOR = 1.5  # Adjust until both motors move equally

def motor_forward(speed_left=18, speed_right=18):
    # Both motors forward
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)   # Left motor forward
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)   # Right motor forward
    pwmA.ChangeDutyCycle(speed_left)
    pwmB.ChangeDutyCycle(min(speed_right * RIGHT_MOTOR_FACTOR, 100))

def motor_stop():
    # Both motors stop
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def motor_turn_right(speed=75):
    # Stop left, right keeps forward 
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)     # Left motor OFF
    pwmA.ChangeDutyCycle(0)

    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)    # Right motor forward
    pwmB.ChangeDutyCycle(min(speed * RIGHT_MOTOR_FACTOR, 100))

def motor_turn_left(speed=75):
    # Left keeps forward, stop right
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
def process_frame(frame, model):
    global object_detected, stop_requested
    results = model(frame, imgsz=160, conf=0.5, half=True, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            object_detected = True
            stop_requested = True
            label = f"{class_names[cls_id]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated_frame
def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    height, width = mask.shape[:2]
    roi_mask = np.zeros_like(mask)
    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.3 * width), int(0.6 * height)),
        (int(0.7 * width), int(0.6 * height)),
        (int(0.95 * width), height)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    masked_img = cv2.bitwise_and(mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(
        cleaned_mask,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=20,
        maxLineGap=20
    )
    debug_img = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if slope < -0.3:
                    left_lines.append((slope, intercept))
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif slope > 0.3:
                    right_lines.append((slope, intercept))
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    left_lane = np.average(left_lines, axis=0) if left_lines else None
    right_lane = np.average(right_lines, axis=0) if right_lines else None
    steering_angle = 90
    if left_lane is not None and right_lane is not None:
        slope_left, intercept_left = left_lane
        slope_right, intercept_right = right_lane
        y = height
        x_left = (y - intercept_left) / slope_left
        x_right = (y - intercept_right) / slope_right
        center = (x_left + x_right) / 2
        deviation = (center - width/2) / (width/2)
        steering_angle = 90 + deviation * 30
    elif left_lane is not None:
        steering_angle = 105
    elif right_lane is not None:
        steering_angle = 75
    return steering_angle, debug_img
def main():
    global running, stop_requested, object_detected
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
        base_speed = 18
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
            if not stop_requested:
                steering_angle, lane_debug = detect_lanes(frame)
                if steering_angle < 85:
                    motor_turn_left(base_speed)
                elif steering_angle > 95:
                    motor_turn_right(base_speed)
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


