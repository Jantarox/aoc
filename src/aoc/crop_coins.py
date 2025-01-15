import cv2
import numpy as np
import os
import uuid

def detect_and_extract_coins(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    coin_count = 0
    processed_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Podziel kontury na dwie grupy
    contours_by_circularity = []
    contours_by_hough = []
    
    # Analizuj wszystkie kontury i przydziel je do odpowiednich grup
    mask = np.zeros((image_height, image_width), dtype=np.uint8)  # Reużywalna maska
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
            
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Jeżeli kontur jest wystarczająco "okrągły"
        if circularity >= 0.7:
            contours_by_circularity.append(contour)
        else:
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.33 <= aspect_ratio <= 3:
                roi = image[y:y+h, x:x+w]
                mask[:] = 0
                cv2.drawContours(mask[y:y+h, x:x+w], 
                                 [contour - np.array([x, y])], 
                                 -1, 255, -1)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                nonzero_pixels = roi_gray[mask[y:y+h, x:x+w] > 0]
                
                if len(nonzero_pixels) > 0:
                    std_dev = np.std(nonzero_pixels)
                    if std_dev >= 10:
                        contours_by_hough.append(contour)
    
    # Przetwórz kontury o wysokiej okrągłości
    for contour in contours_by_circularity:
        x, y, w, h = cv2.boundingRect(contour)
        mask[:] = 0
        cv2.drawContours(mask[y:y+h, x:x+w], 
                         [contour - np.array([x, y])], 
                         -1, 255, -1)
        
        cv2.drawContours(processed_mask, [contour], -1, 255, -1)
        
        coin_roi = image[y:y+h, x:x+w]
        coin_masked = cv2.bitwise_and(coin_roi, coin_roi, mask=mask[y:y+h, x:x+w])
        coin_rgba = cv2.cvtColor(coin_masked, cv2.COLOR_BGR2BGRA)
        coin_rgba[:, :, 3] = mask[y:y+h, x:x+w]
        
        contour_output_path = os.path.join(output_dir, f"contour_{uuid.uuid4()}.png")
        cv2.imwrite(contour_output_path, coin_rgba)
        coin_count += 1
    
    # Przetwórz kontury dla HoughCircles
    for contour in contours_by_hough:
        x, y, w, h = cv2.boundingRect(contour)
        
        coin_roi = image[y:y+h, x:x+w]
        mask[:] = 0
        cv2.drawContours(mask[y:y+h, x:x+w], 
                         [contour - np.array([x, y])], 
                         -1, 255, -1)
        
        gray_sub = cv2.cvtColor(coin_roi, cv2.COLOR_BGR2GRAY)
        gray_sub = cv2.medianBlur(gray_sub, 5)
        
        min_radius = min(gray_sub.shape) // 4
        max_radius = min(gray_sub.shape) // 2
        
        circles = cv2.HoughCircles(
            gray_sub,
            cv2.HOUGH_GRADIENT,
            dp=1.8,  
            minDist=min_radius,
            param1=90,
            param2=50,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            # W HoughCircles zwracane wartości są zwykle float,
            # po zaokrągleniu do uint16 może dojść do przepełnienia przy odejmowaniu.
            # Dlatego konwertujemy je na int (Python) poniżej.
            circles = np.uint16(np.around(circles[0]))
            
            # Przygotuj maskę konturu
            contour_mask = mask[y:y+h, x:x+w]
            contour_area = np.count_nonzero(contour_mask)
            
            for cx, cy, r in circles:
                # Konwersja na zwykłe int (rozwiązuje problem przepełnienia)
                center_x = int(cx)
                center_y = int(cy)
                radius   = int(r)
                
                circle_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
                
                # Oblicz pokrycie kontur <-> okrąg
                intersection = cv2.bitwise_and(contour_mask, circle_mask)
                intersection_area = np.count_nonzero(intersection)

                coverage_contour_percent = (intersection_area / contour_area) * 100
                circle_area = np.count_nonzero(circle_mask)
                coverage_circle_percent = (intersection_area / circle_area) * 100

                # print(f"coverage_contour_percent={coverage_contour_percent:.2f}%, "
                #       f"coverage_circle_percent={coverage_circle_percent:.2f}%")
                
                # Jeżeli okrąg w więcej niż 10% wystaje poza kontur,
                # to coverage_circle_percent < 90 → pomijamy.
                if coverage_circle_percent < 90:
                    # print("Skipping circle - it goes >10% outside the contour")
                    continue

                # Tworzenie (opcjonalnego) debug_image
                # debug_image = np.zeros((h, w, 3), dtype=np.uint8)
                # debug_image[contour_mask > 0] = [0, 0, 255]    # Kontur - czerwony
                # debug_image[circle_mask > 0] = [0, 255, 0]     # Okrąg - zielony
                # debug_image[intersection > 0] = [255, 255, 0]  # Część wspólna - żółty
                # coverage_output_path = os.path.join(output_dir, f"coverage_{uuid.uuid4()}.png")
                # cv2.imwrite(coverage_output_path, debug_image)
                
                # print(f"Circle accepted. coverage_circle_percent = {coverage_circle_percent:.2f}%")

                # Sprawdzamy nakładanie z already processed
                global_mask = np.zeros_like(processed_mask)
                cv2.circle(global_mask, (x + center_x, y + center_y), radius, 255, -1)
                
                overlap = cv2.bitwise_and(processed_mask, global_mask)
                overlap_area = np.count_nonzero(overlap)
                circle_area_global = np.count_nonzero(global_mask)
                overlap_percent = (overlap_area / circle_area_global) * 100

                if overlap_percent > 30:
                    # print(f"Circle overlaps {overlap_percent:.2f}% with processed regions - skipping")
                    continue
                
                processed_mask = cv2.bitwise_or(processed_mask, global_mask)
                
                # Obliczenie współrzędnych wycinka w coin_roi
                x1 = center_x - radius
                y1 = center_y - radius
                x2 = center_x + radius
                y2 = center_y + radius

                # Upewniamy się, że x1 < x2, y1 < y2 i nie wychodzimy poza ROI
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, w)
                y2 = min(y2, h)
                
                # Jeśli x2 <= x1 albo y2 <= y1, obszar pusty → pomijamy
                if x2 <= x1 or y2 <= y1:
                    # print("Circle bounding box invalid - skipping")
                    continue
                
                # Wycinanie i zapisywanie okręgu
                circle_cropped = cv2.bitwise_and(coin_roi, coin_roi, mask=circle_mask)
                circle_cropped = circle_cropped[y1:y2, x1:x2]
                if circle_cropped.size == 0:
                    # print("circle_cropped is empty - skipping")
                    continue

                circle_rgba = cv2.cvtColor(circle_cropped, cv2.COLOR_BGR2BGRA)
                circle_mask_roi = circle_mask[y1:y2, x1:x2]
                circle_rgba[:, :, 3] = circle_mask_roi
                
                circle_output_path = os.path.join(output_dir, f"circle_{uuid.uuid4()}.png")
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
