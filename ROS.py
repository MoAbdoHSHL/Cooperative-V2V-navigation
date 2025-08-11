"""
Modified OpenCV-based LIDAR Visualizer
- Shows only real-time live objects as green dots (one per scan)
- Reduced GUI to only show the analyzing half (left side)
- Now detects objects as close as 5cm
"""

import cv2
import numpy as np
import threading
import time
import serial
import RPi.GPIO as GPIO
import math
from collections import deque

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

def set_angle(angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.2)
    pwm.ChangeDutyCycle(0)

def servo_loop():
    global target_angle
    while True:
        target_angle = 10
        set_angle(10)    # Full Left
        time.sleep(0.1)
        
        target_angle = 170
        set_angle(170)   # Full Right
        time.sleep(0.1)

# ----------------- LIDAR SETUP -------------------
ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=0.1)
HEADER = 0x59

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
    global scan_data, current_angle
    
    # Simulation of servo movement for angle tracking
    last_target = target_angle
    movement_start_time = time.time()
    start_angle = current_angle
    
    while True:
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
            
            time.sleep(0.002)
            
        except Exception as e:
            print(f"LIDAR error: {e}")
            time.sleep(0.01)

# ----------------- OPENCV VISUALIZATION -------------------
class LIDARVisualizer:
    def __init__(self, width=700, height=900):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2 + 100
        self.max_range = 200
        self.scale = min(width * 0.35, height * 0.35) / self.max_range
        
        # Colors (BGR format)
        self.bg_color = (15, 15, 15)
        self.grid_color = (80, 80, 80)
        self.beam_color = (0, 255, 255)
        self.point_color = (0, 255, 0)  # Green for live objects
        self.text_color = (255, 255, 255)
        self.current_beam_color = (0, 100, 255)
        
        # Create window
        cv2.namedWindow('LIDAR Visualizer', cv2.WINDOW_AUTOSIZE)
    
    def draw_grid(self, img):
        """Draw only the left half of the radar view (0°-180°)"""
        # Draw range circles (added 5cm and 10cm circles)
        for r in [5, 10, 25, 50, 75, 100, 125, 150, 175, 200]:
            radius = int(r * self.scale)
            cv2.circle(img, (self.center_x, self.center_y), radius, self.grid_color, 1 if r <= 25 else 2)
            
            # Range labels at top (only show labels for 25cm and above to avoid clutter)
            if r >= 25:
                label = f"{r}cm"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(img, label, 
                           (self.center_x - label_size[0]//2, self.center_y - radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Draw angle lines only for left half (0°-180°)
        for angle in range(0, 181, 15):
            angle_rad = math.radians(angle)
            end_x = self.center_x + int(self.max_range * self.scale * math.cos(angle_rad))
            end_y = self.center_y + int(self.max_range * self.scale * math.sin(angle_rad))
            
            line_color = (120, 120, 120)
            line_width = 2
                
            cv2.line(img, (self.center_x, self.center_y), (end_x, end_y), line_color, line_width)
            
            # Angle labels
            if angle % 30 == 0:
                label_x = self.center_x + int((self.max_range + 20) * self.scale * math.cos(angle_rad))
                label_y = self.center_y + int((self.max_range + 20) * self.scale * math.sin(angle_rad))
                cv2.putText(img, f"{angle}°", (label_x - 15, label_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)
    
    def draw_beam(self, img, angle, distance):
        """Draw current laser beam"""
        angle_rad = math.radians(angle)
        
        # Beam end point
        beam_end_x = self.center_x + int(distance * self.scale * math.cos(angle_rad))
        beam_end_y = self.center_y + int(distance * self.scale * math.sin(angle_rad))
        
        # Full beam (faded)
        max_beam_x = self.center_x + int(self.max_range * self.scale * math.cos(angle_rad))
        max_beam_y = self.center_y + int(self.max_range * self.scale * math.sin(angle_rad))
        cv2.line(img, (self.center_x, self.center_y), (max_beam_x, max_beam_y), 
                (0, 50, 100), 2)
        
        # Active beam (bright)
        cv2.line(img, (self.center_x, self.center_y), (beam_end_x, beam_end_y), 
                self.current_beam_color, 4)
        
        # Draw beam origin
        cv2.circle(img, (self.center_x, self.center_y), 8, (0, 255, 255), -1)
        cv2.circle(img, (self.center_x, self.center_y), 6, (0, 0, 0), -1)
        
        return beam_end_x, beam_end_y
    
    def draw_points(self, img):
        """Draw only the latest live objects as green dots"""
        current_time = time.time()
        
        with data_lock:
            data_copy = dict(scan_data)
        
        # Remove old data (older than 0.5 seconds)
        angles_to_remove = []
        for angle, data in data_copy.items():
            if current_time - data['timestamp'] > 0.5:  # Only show very recent points
                angles_to_remove.append(angle)
        
        if angles_to_remove:
            with data_lock:
                for angle in angles_to_remove:
                    if angle in scan_data:
                        del scan_data[angle]
        
        # Draw only the latest points (changed to >= 5 to include 5cm)
        for angle, data in data_copy.items():
            distance = data['distance']
            if distance >= 5 and distance <= self.max_range:
                angle_rad = math.radians(angle)
                point_x = self.center_x + int(distance * self.scale * math.cos(angle_rad))
                point_y = self.center_y + int(distance * self.scale * math.sin(angle_rad))
                
                # Draw green dot (size based on strength)
                strength = min(max(data['strength'], 100), 1000)
                size = max(3, min(8, strength // 150))
                cv2.circle(img, (point_x, point_y), size, self.point_color, -1)
    
    def draw_info_panel(self, img):
        """Draw simplified information panel"""
        panel_height = 80
        cv2.rectangle(img, (0, 0), (self.width, panel_height), (40, 40, 40), -1)
        cv2.rectangle(img, (0, 0), (self.width, panel_height), (100, 100, 100), 2)
        
        with data_lock:
            data_count = len(scan_data)
            latest_data = None
            if scan_data:
                if current_angle in scan_data:
                    latest_data = scan_data[current_angle]
                else:
                    closest_angle = min(scan_data.keys(), key=lambda x: abs(x - current_angle))
                    latest_data = scan_data[closest_angle]
        
        # Main info line
        if latest_data:
            info_text = [
                f"Angle: {current_angle:6.1f}°",
                f"Dist: {latest_data['distance']:3d}cm", 
                f"Str: {latest_data['strength']:4d}",
                f"Objects: {data_count}"
            ]
        else:
            info_text = [
                f"Angle: {current_angle:6.1f}°",
                "Dist: ---cm", 
                "Str: ----",
                f"Objects: {data_count}"
            ]
        
        for i, text in enumerate(info_text):
            cv2.putText(img, text, (20 + i * 160, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        
        # Status
        cv2.putText(img, "LIVE OBJECTS (5-200cm)", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run(self):
        """Main visualization loop"""
        print("LIDAR Visualizer - Live Objects Only (5-200cm range)")
        print("Press 'q' to quit")
        
        while True:
            # Create blank image
            img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            
            # Draw components
            self.draw_grid(img)
            self.draw_points(img)
            
            # Draw current beam if we have current data
            with data_lock:
                current_data = scan_data.get(current_angle)
                if current_data:
                    self.draw_beam(img, current_angle, current_data['distance'])
            
            # Draw info panel
            self.draw_info_panel(img)
            
            # Show image
            cv2.imshow('LIDAR Visualizer', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

# ----------------- MAIN -------------------
def main():
    try:
        print("Starting LIDAR Visualizer - Live Objects Mode (5-200cm range)")
        
        # Start background threads
        servo_thread = threading.Thread(target=servo_loop, daemon=True)
        lidar_thread = threading.Thread(target=lidar_loop, daemon=True)
        
        servo_thread.start()
        lidar_thread.start()
        
        # Give threads time to start
        time.sleep(2)
        
        # Start visualization
        visualizer = LIDARVisualizer()
        visualizer.run()
        
    except KeyboardInterrupt:
        print("Stopping program...")
    finally:
        pwm.stop()
        GPIO.cleanup()
        ser.close()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    main()