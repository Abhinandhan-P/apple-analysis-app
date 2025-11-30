import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Paths
img_path = r"C:/Users/ABHINANDHAN/.gemini/antigravity/brain/ae1291e6-4f34-4b80-b350-faba821a97c6/uploaded_image_1764498069086.png"
color_model_path = 'runs/detect/train/weights/best.pt'
ripeness_model_path = 'best.pt'

# Helper function for IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Helper function to analyze color ratios
def get_color_ratios(image_crop):
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]
    
    if total_pixels == 0:
        return 0.0, 0.0, 0.0

    # 1. Green Range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = cv2.countNonZero(green_mask) / total_pixels

    # 2. Red Range (wraps around 0/180)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = cv2.countNonZero(red_mask) / total_pixels

    # 3. Maroon (Dark Red) Range
    lower_maroon1 = np.array([0, 20, 20])
    upper_maroon1 = np.array([10, 255, 100])
    lower_maroon2 = np.array([170, 20, 20])
    upper_maroon2 = np.array([180, 255, 100])
    maroon_mask = cv2.inRange(hsv, lower_maroon1, upper_maroon1) + cv2.inRange(hsv, lower_maroon2, upper_maroon2)
    maroon_ratio = cv2.countNonZero(maroon_mask) / total_pixels

    return green_ratio, red_ratio, maroon_ratio

def main():
    print(f"Loading models...")
    try:
        color_model = YOLO(color_model_path)
        # ripeness_model not needed
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print(f"Processing image: {img_path}")
    if not os.path.exists(img_path):
        print("Image file not found!")
        return

    image = Image.open(img_path).convert('RGB')
    img_array = np.array(image)
    
    # Inference
    color_results = color_model(img_array, conf=0.45)
    color_result = color_results[0]
    
    print(f"\n--- Analysis Results ---")
    print(f"Detected {len(color_result.boxes)} apples.")
    
    for i, c_box in enumerate(color_result.boxes):
        c_cls_id = int(c_box.cls[0])
        c_conf = float(c_box.conf[0])
        
        c_xyxy = c_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, c_xyxy)
        apple_crop = img_array[y1:y2, x1:x2]
        
        # Analyze Colors
        g_ratio, r_ratio, m_ratio = get_color_ratios(apple_crop)
        
        # Apply Rules
        status = "Unknown"
        if m_ratio > 0.05:
            status = "Overripe"
        elif g_ratio > 0.10 and r_ratio > 0.10:
            status = "Semi-Ripe"
        elif r_ratio > g_ratio:
            status = "Ripe"
        else:
            status = "Unripe"
            
        print(f"\nApple #{i+1}:")
        print(f"  Ratios -> Green: {g_ratio:.3f}, Red: {r_ratio:.3f}, Maroon: {m_ratio:.3f}")
        print(f"  Final Status: {status}")

if __name__ == "__main__":
    main()
