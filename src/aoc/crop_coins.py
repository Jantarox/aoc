import cv2
import numpy as np
from PIL import Image
import os
import uuid

def detect_and_extract_coins(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    image_height, image_width = image.shape[:2]
    total_area = image_width * image_height
    min_area = total_area * 0.01
    
    #DEBUG KONTURY
    debug_image = image.copy()
    for contour in contours:
        cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
    
    debug_contours_path = os.path.join(
        output_dir, 
        "debug_contours_" + os.path.basename(image_path)
    )
    cv2.imwrite(debug_contours_path, debug_image)
    
    coin_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        else:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        if w == 0 or h == 0:
            continue
        
        coin_roi = image[y:y+h, x:x+w]

        # --------------------------------------------------
        # A) circularity >= 0.6 -> wycinamy TYLKO sam kontur
        # --------------------------------------------------
        if circularity >= 0.7:
            mask_full = np.zeros_like(cleaned)[y:y+h, x:x+w]
            cv2.drawContours(mask_full, [contour], -1, 255, -1, offset=(-x, -y))
            
            coin_masked = cv2.bitwise_and(coin_roi, coin_roi, mask=mask_full)
            coin_rgba = cv2.cvtColor(coin_masked, cv2.COLOR_BGR2BGRA)
            coin_rgba[:, :, 3] = mask_full

            contour_output_path = os.path.join(output_dir, f"contour_{uuid.uuid4()}.png")
            cv2.imwrite(contour_output_path, coin_rgba)
            coin_count += 1
        
        # --------------------------------------------------
        # B) circularity < 0.7 -> uruchamiamy HoughCircles
        # --------------------------------------------------
        else:
            gray_sub = cv2.cvtColor(coin_roi, cv2.COLOR_BGR2GRAY)
            gray_sub = cv2.medianBlur(gray_sub, 5)
            
            rows = gray_sub.shape[0]
            circles = cv2.HoughCircles(
                gray_sub,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=rows / 8,
                param1=200,  
                param2=100,  
                minRadius=0,
                maxRadius=0
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for c in circles[0]:
                    center_x = int(c[0])
                    center_y = int(c[1])
                    radius   = int(c[2])
                    
                    x1 = max(center_x - radius, 0)
                    y1 = max(center_y - radius, 0)
                    x2 = min(center_x + radius, coin_roi.shape[1])
                    y2 = min(center_y + radius, coin_roi.shape[0])
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    circle_mask = np.zeros_like(gray_sub, dtype=np.uint8)
                    cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
                    
                    circle_cropped = cv2.bitwise_and(coin_roi, coin_roi, mask=circle_mask)
                    circle_cropped = circle_cropped[y1:y2, x1:x2]
                    
                    circle_rgba = cv2.cvtColor(circle_cropped, cv2.COLOR_BGR2BGRA)
                    circle_mask_roi = circle_mask[y1:y2, x1:x2]
                    circle_rgba[:, :, 3] = circle_mask_roi
                    
                    circle_output_path = os.path.join(
                        output_dir, f"circle_{uuid.uuid4()}.png"
                    )
                    cv2.imwrite(circle_output_path, circle_rgba)
                    coin_count += 1
    
    return coin_count


def process_multiple_images(input_dir, output_dir):
    supported_formats = ('.jpg')
    total_coins = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(input_dir, filename)
            try:
                coins_found = detect_and_extract_coins(image_path, output_dir)
                total_coins += coins_found
                print(f"Processed {filename}: found {coins_found} coins")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return total_coins


if __name__ == "__main__":
    input_dir = "new"
    output_dir = "cropped_coins"
    
    total_coins = process_multiple_images(input_dir, output_dir)
    print(f"\nTotal coins extracted: {total_coins}")
