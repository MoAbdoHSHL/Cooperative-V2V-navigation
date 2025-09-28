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
import math

# ---- Environment settings ----
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# ---- Class names and pins ----
class_names = ['pathholde', 'Manholde', 'Duck', 'Snow', 'Stop']
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 13
SERVO_PIN = 12
RIGHT_MOTOR_FACTOR = 0.75  # boost right motor speed


# ---- MQTT Subscriber settings ---lidar_loop-
BROKER_IP = "172.20.10.13"  # Broker Pi
TOPIC_SUB = "servo/control"  # Topic for commands

# Lane memory
last_center = None
last_lane_width = None
last_confidence = 0.0

# ---- GPIO setup ----

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

# ---- LIDAR / serial ----
scan_data = {}
data_lock = threading.Lock()
target_angle = 10
ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=0.1)
HEADER = 0x59

# ---- Control flags ----
object_detected = False
stop_requested = False
running = True
lidar_stop = False  
mqtt_stop = False   
local_stop = False  

# ---- MQTT lock ----
mqtt_lock = threading.Lock()
last_detected_class = None
new_detection = False

# ---- Motor functions ----
def motor_forward(speed_left=20, speed_right=20):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed_left)
    pwmB.ChangeDutyCycle(speed_right * RIGHT_MOTOR_FACTOR)

is_stopped = False

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    
def smooth_ramp(start, end, steps, i):
    return start + (end - start) * (1 - math.cos(math.pi * i / steps)) / 2

# Add these globals somewhere in your code
lane_left_active = False
lane_right_active = False

def motor_turn_right(speed=40, ramp_time=0.05, step_delay=0.01, start_speed=25):
    global lane_left_active
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    steps = int(ramp_time / step_delay)
    for i in range(1, steps + 1):
        # interrupt ramp if lane is gone
        if not lane_left_active:
            break
        left_speed = smooth_ramp(speed, speed * 0.5, steps, i)
        right_speed = smooth_ramp(start_speed, speed, steps, i)
        pwmA.ChangeDutyCycle(left_speed)
        pwmB.ChangeDutyCycle(min(right_speed * RIGHT_MOTOR_FACTOR, 100))
        time.sleep(step_delay)
    pwmA.ChangeDutyCycle(speed * 0.3)
    pwmB.ChangeDutyCycle(min(speed * RIGHT_MOTOR_FACTOR, 100))


def motor_turn_left(speed=70, ramp_time=0.05, step_delay=0.01, start_speed=5):
    global lane_right_active
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

    steps = int(ramp_time / step_delay)
    for i in range(1, steps + 1):
        if not lane_right_active:
            break
        # Ramp left and right motors gradually
        left_speed  = smooth_ramp(start_speed, speed, steps, i)
        right_speed = smooth_ramp(start_speed * 0.5, speed * 0.3, steps, i)
        pwmA.ChangeDutyCycle(left_speed)
        pwmB.ChangeDutyCycle(right_speed)
        time.sleep(step_delay)

    # final slow-down at end
    pwmA.ChangeDutyCycle(speed * 0.8)
    pwmB.ChangeDutyCycle(speed * 0.3)



# ---- Servo functions ----
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

# ---- LIDAR functions ----
def read_frame():
    while True:
        b = ser.read(1)
        if len(b) == 0: continue
        if b[0] == HEADER:
            b2 = ser.read(1)
            if len(b2) == 0: continue
            if b2[0] == HEADER:
                frame = ser.read(7)
                if len(frame) != 7: continue
                uart = [HEADER, HEADER] + list(frame)
                checksum = sum(uart[:8]) & 0xFF
                if checksum == uart[8]:
                    dist = uart[2] + (uart[3] << 8)
                    strength = uart[4] + (uart[5] << 8)
                    return dist, strength
import random  # ⭐⭐ ADD THIS IMPORT ⭐⭐

def lidar_loop():
    global scan_data, running, lidar_stop
    SAFE_DISTANCE_CM = 30
    # ⭐⭐ ADD FILTERING VARIABLES ⭐⭐
    distance_buffer = []
    BUFFER_SIZE = 3
    consecutive_clear_readings = 0
    REQUIRED_CLEAR_READINGS = 2
    
    while running:
        try:
            dist, strength = read_frame()

            # ⭐⭐ FILTER OUT INVALID READINGS ⭐⭐
            if dist == 0 or dist > 12000:  # Filter obviously wrong readings
                continue
                
            if 5 <= dist <= 400:
                with data_lock:
                    scan_data[0] = {'distance': min(dist, 200), 'strength': strength, 'timestamp': time.time()}

            # ⭐⭐ USE MOVING AVERAGE FOR MORE STABLE READINGS ⭐⭐
            distance_buffer.append(dist)
            if len(distance_buffer) > BUFFER_SIZE:
                distance_buffer.pop(0)
            
            avg_distance = sum(distance_buffer) / len(distance_buffer)
            
            # ⭐⭐ ADD STRENGTH VALIDATION ⭐⭐
            MIN_STRENGTH = 100  # Minimum signal strength for reliable reading
            
            # DEBUG: Print distance occasionally
            if random.random() < 0.05:  # Print ~5% of readings for better debugging
                print(f"[LIDAR] Dist: {dist}cm, Avg: {avg_distance:.1f}cm, Strength: {strength}, Stop: {lidar_stop}")

            # ⭐⭐ IMPROVED STOP LOGIC WITH HYSTERESIS ⭐⭐
            if 0 < avg_distance <= SAFE_DISTANCE_CM and strength >= MIN_STRENGTH:
                if not lidar_stop:
                    consecutive_clear_readings = 0
                    lidar_stop = True
                    print(f"[LIDAR] 🛑 Object at {avg_distance:.1f}cm (strength: {strength}) -> STOP")
            else:
                if lidar_stop:
                    consecutive_clear_readings += 1
                    if consecutive_clear_readings >= REQUIRED_CLEAR_READINGS:
                        lidar_stop = False
                        print(f"[LIDAR] ✅ Path clear ({avg_distance:.1f}cm) -> RESUME")
                        consecutive_clear_readings = 0
                else:
                    consecutive_clear_readings = 0

            time.sleep(0.01)
        except Exception as e:
            print(f"LIDAR error: {e}")
            time.sleep(0.01)

# ---- Camera & model ----
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
    model = YOLO("/home/mohamed/Cooperative-V2V-navigation/runs/detect/train2/weights/best.pt").to("cpu")
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

# ---- Lane Detection Functions ----
def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def region_of_interest(mask):
    height, width = mask.shape[:2]
    roi_mask = np.zeros_like(mask)
    polygon = np.array([[(int(0.0*width),height),(int(0.4*width),int(0.15*height)),(int(0.6*width),int(0.15*height)),(int(1.0*width),height)]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    masked_img = cv2.bitwise_and(mask, roi_mask)
    return masked_img, polygon

def clean_mask(masked_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, kernel)

def detect_lines(cleaned_mask):
    lines = cv2.HoughLinesP(cleaned_mask, rho=1, theta=np.pi/180, threshold=20, minLineLength=80, maxLineGap=30)
    return lines

def separate_lanes(lines, debug_img):
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if x2 != x1:
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                if slope<-0.3:
                    left_lines.append((slope,intercept))
                    cv2.line(debug_img,(x1,y1),(x2,y2),(255,0,0),2)
                elif slope>0.3:
                    right_lines.append((slope,intercept))
                    cv2.line(debug_img,(x1,y1),(x2,y2),(0,0,255),2)
    return left_lines, right_lines

def calculate_steering(left_lane,right_lane,width,height):
    steering_angle = 90
    both_lanes_detected = left_lane is not None and right_lane is not None
    left_lane_detected = left_lane is not None
    right_lane_detected = right_lane is not None
    if both_lanes_detected:
        slope_left,intercept_left = left_lane
        slope_right,intercept_right = right_lane
        y = height
        x_left = (y - intercept_left)/slope_left
        x_right = (y - intercept_right)/slope_right
        center = (x_left + x_right)/2
        deviation = (center - width/2)/(width/2)
        steering_angle = 90 + deviation*30
    elif left_lane_detected:
        steering_angle = 105
    elif right_lane_detected:
        steering_angle = 75
    return steering_angle, both_lanes_detected, left_lane_detected, right_lane_detected

def detect_lanes(frame):
    mask = preprocess_frame(frame)
    masked_img, polygon = region_of_interest(mask)
    cleaned_mask = clean_mask(masked_img)
    debug_img = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    cv2.polylines(debug_img, polygon, isClosed=True, color=(0,255,0), thickness=2)
    lines = detect_lines(cleaned_mask)
    left_lines, right_lines = separate_lanes(lines, debug_img)
    left_lane = np.average(left_lines, axis=0) if left_lines else None
    right_lane = np.average(right_lines, axis=0) if right_lines else None
    height, width = cleaned_mask.shape[:2]
    steering_angle, both, left, right = calculate_steering(left_lane,right_lane,width,height)
    return steering_angle, debug_img, both, left, right
# Add these with other global variables in SUB code
mqtt_lidar_stop = False      # LiDAR stop from other car
mqtt_class_stop = False      # Class detection from other car
def on_message(client, userdata, msg):
    global mqtt_stop, mqtt_lidar_stop, mqtt_class_stop
    
    try:
        payload = msg.payload.decode()
        print(f"MQTT Received: {payload}")
        
        # Parse the combined message
        if "lidar:" in payload and "class:" in payload:
            lidar_part = payload.split(",")[0]  # "lidar:1" or "lidar:0"
            class_part = payload.split(",")[1]  # "class:pathholde" or "class:clear"
            
            lidar_stop_received = int(lidar_part.split(":")[1]) == 1
            class_received = class_part.split(":")[1]
            
            # Update stop flags based on received data
            mqtt_lidar_stop = lidar_stop_received
            mqtt_class_stop = class_received != "clear"
            
            print(f"📡 MQTT Parsed - LiDAR: {'STOP' if mqtt_lidar_stop else 'GO'}, Class: {class_received}")
            
            # Stop if either LiDAR OR class detection from other car
            if mqtt_lidar_stop or mqtt_class_stop:
                mqtt_stop = True
                print(">>> MQTT STOP ACTIVATED (LiDAR or Class from other car)")
            else:
                mqtt_stop = False
                print(">>> MQTT CLEAR (both clear from other car)")
                
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")





def mqtt_subscriber_thread():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER_IP, 1883, 60)
    client.subscribe(TOPIC_SUB)
    client.loop_forever()

# ---- Object detection ----
# Global timer
stop_until = 0

def process_frame(frame, model):
    global last_detected_class, local_stop
    results = model(frame, imgsz=160, conf=0.5, half=True, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    detected_class = None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            detected_class = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{detected_class} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break

    # Update local stop flag
    if detected_class:
        last_detected_class = detected_class
        local_stop = True
    else:
        last_detected_class = "clear"
        local_stop = False

    return annotated_frame


# ---- Main ----
def main():
    global running

    # Start MQTT subscriber thread
    mqtt_thread = threading.Thread(target=mqtt_subscriber_thread)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    # Initialize camera and model
    camera = initialize_camera()
    model = initialize_model()
    frame_thread = FrameCaptureThread(camera)
    frame_thread.start()

    # Servo and LIDAR threads
    servo_thread = threading.Thread(target=servo_loop)
    servo_thread.daemon = True
    servo_thread.start()

    lidar_thread = threading.Thread(target=lidar_loop)
    lidar_thread.daemon = True
    lidar_thread.start()

    try:
        frame_count = 0
        base_speed = 20
        steer_speed = 25
        is_stopped = False
        stop_start_time = 0
        MIN_STOP_TIME = 1.0  # Minimum stop time in seconds

        while True:
            frame = frame_thread.get_frame()
            if frame is None:
                continue

            annotated = process_frame(frame, model)
            steering_angle, lane_debug, both_lanes, left_lane, right_lane = detect_lanes(frame)

            # ⭐⭐ IMPROVED STOP LOGIC WITH TIMING ⭐⭐
            should_stop = lidar_stop or mqtt_stop or local_stop
            
            current_time = time.time()
            
            # If we need to stop but aren't currently stopped
            if should_stop and not is_stopped:
                motor_stop()
                is_stopped = True
                stop_start_time = current_time
                print(f"🛑 STOPPED - LIDAR:{lidar_stop}, MQTT:{mqtt_stop}, Local:{local_stop}")
            
            # If we are stopped
            elif is_stopped:
                # Check if we've been stopped for minimum time and path is clear
                time_stopped = current_time - stop_start_time
                if time_stopped >= MIN_STOP_TIME and not should_stop:
                    print("✅ Path clear! Resuming navigation...")
                    is_stopped = False
                else:
                    if time_stopped < MIN_STOP_TIME:
                        print(f"⏳ Minimum stop time: {MIN_STOP_TIME - time_stopped:.1f}s remaining")
                    else:
                        print(f"⏳ Waiting... LIDAR:{lidar_stop}, MQTT:{mqtt_stop}, Local:{local_stop}")
                    motor_stop()
                    time.sleep(0.1)
                    continue
            
            # If not stopped, drive normally
            if not is_stopped:
                # ⭐⭐ ADD SAFETY CHECK - DOUBLE CHECK LIDAR BEFORE MOVING ⭐⭐
                if lidar_stop:
                    print("⚠️  Safety override: LiDAR detected obstacle during movement!")
                    motor_stop()
                    is_stopped = True
                    stop_start_time = current_time
                    continue
                    
                if both_lanes:
                    motor_forward(base_speed, base_speed)
                elif left_lane:
                    motor_turn_right(steer_speed)
                elif right_lane:
                    motor_turn_left(steer_speed)
                else:
                    if steering_angle < 85:
                        motor_turn_right(steer_speed)
                    elif steering_angle > 95:
                        motor_turn_left(steer_speed)
                    else:
                        motor_forward(base_speed, base_speed)

            # Display frames with status information
            status_text = f"LIDAR: {'STOP' if lidar_stop else 'GO'}, State: {'STOPPED' if is_stopped else 'MOVING'}"
            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
