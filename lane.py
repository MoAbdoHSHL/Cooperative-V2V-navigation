import cv2
import os
import numpy as np

# Path to lane image folder
image_folder = r'D:\Project_Work\MyDataSet\lane'

# Load image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height), (img.shape[1], height), (img.shape[1], int(height*0.55)), (0, int(height*0.55))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.5:  # Skip near-horizontal lines
            continue
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)

    height = img.shape[0]
    final_img = img.copy()

    def average_slope_intercept(lines):
        x = []
        y = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x.extend([x1, x2])
            y.extend([y1, y2])
        if len(x) == 0: return None
        fit = np.polyfit(y, x, 1)  # x = my + b
        y1 = height
        y2 = int(height * 0.55)
        x1 = int(fit[0]*y1 + fit[1])
        x2 = int(fit[0]*y2 + fit[1])
        return ((x1, y1), (x2, y2))

    left_lane = average_slope_intercept(left_lines)
    right_lane = average_slope_intercept(right_lines)

    if left_lane:
        cv2.line(final_img, left_lane[0], left_lane[1], (255, 0, 0), 5)
    if right_lane:
        cv2.line(final_img, right_lane[0], right_lane[1], (0, 255, 0), 5)

    if left_lane and right_lane:
        mid_bottom = ((left_lane[0][0] + right_lane[0][0]) // 2, height)
        mid_top = ((left_lane[1][0] + right_lane[1][0]) // 2, int(height * 0.55))
        cv2.line(final_img, mid_bottom, mid_top, (0, 0, 255), 3)  # Center line in red

    return final_img

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        return
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)

    if lines is not None:
        final = draw_lines(img, lines)
    else:
        print("No lines detected")
        final = img

    cv2.imshow(os.path.basename(img_path), final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process each image
for f in image_files:
    print(f"Processing {f}")
    process_image(os.path.join(image_folder, f))
