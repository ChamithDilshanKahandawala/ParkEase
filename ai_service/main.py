import os
# --- CRITICAL FIX FOR OMP ERROR #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --------------------------------------

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time

# --- SETUP FOLDERS ---
SAVE_FOLDER = "detected_plates"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# --- CONFIGURATION ---
MODEL_PATH = 'my_model.pt'
print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False) 

# --- AUTO-DETECT CAMERA ---
print("\n--- Searching for cameras ---")
target_camera_index = None

for i in range(3):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        ret, _ = temp_cap.read()
        if ret:
            print(f"✅ Camera found at index {i}!")
            target_camera_index = i
            temp_cap.release()
            break
    temp_cap.release()

if target_camera_index is None:
    print("\n[ERROR] No working camera found. Check iVCam.")
    exit()

print(f"Connecting to camera {target_camera_index}...")
cap = cv2.VideoCapture(target_camera_index)


# --- ADVANCED PREPROCESSING FUNCTION ---
def preprocess_plate_advanced(img):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This balances lighting (removes shadows/glare)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # 3. Upscale (3x) for better OCR reading
    scale_factor = 3
    resized = cv2.resize(contrast_enhanced, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 4. Bilateral Filter (Removes noise but keeps edges sharp!)
    # Better than Gaussian Blur for text
    filtered = cv2.bilateralFilter(resized, 11, 17, 17)
    
    # 5. Adaptive Thresholding (Strict Black & White)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 6. Morphological Opening (Removes small white noise specks)
    kernel = np.ones((1,1), np.uint8) # Small kernel to just clean pixels
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return clean

# --- STATE VARIABLES ---
last_capture_time = 0
COOLDOWN_SECONDS = 3.0
MIN_CONFIDENCE = 0.3  # Ignore results with less than 30% confidence

print("\n" + "="*40)
print("  ANPR SYSTEM STARTED (HIGH ACCURACY)")
print(f"  Minimum Confidence Required: {MIN_CONFIDENCE*100}%")
print("  Waiting for vehicle...")
print("="*40 + "\n")

# --- MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5, verbose=False)

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            h, w, _ = frame.shape
            y1_crop = max(0, y1-5)
            y2_crop = min(h, y2+5)
            x1_crop = max(0, x1-5)
            x2_crop = min(w, x2+5)
            
            plate_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            try:
                # Use Advanced Preprocessing
                processed_plate = preprocess_plate_advanced(plate_img)

                # Get DETAILED results (allows access to confidence score)
                # detail=1 returns [ [box, text, confidence], ... ]
                ocr_result = reader.readtext(
                    processed_plate, 
                    detail=1, 
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                )

                # Combine all found text parts
                plate_text = ""
                total_confidence = 0
                count = 0

                for res in ocr_result:
                    text = res[1]
                    confidence = res[2]
                    
                    # Only add if confidence is high enough
                    if confidence > MIN_CONFIDENCE:
                        plate_text += text
                        total_confidence += confidence
                        count += 1
                
                # Final cleanup of the text
                plate_text = "".join(filter(str.isalnum, plate_text)).upper()

                if len(plate_text) > 3 and count > 0:
                    avg_conf = total_confidence / count
                    
                    # VISUALS
                    # Green Box for Plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Background for text (Black)
                    cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (0, 0, 0), -1)
                    
                    # Text: Plate Number + Confidence %
                    display_text = f"{plate_text} ({int(avg_conf*100)}%)"
                    cv2.putText(frame, display_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # DEBUG WINDOW: Show what the AI sees (The B&W image)
                    cv2.imshow("AI Vision (Debug)", processed_plate)

                    # --- CAPTURE ---
                    current_time = time.time()
                    if current_time - last_capture_time > COOLDOWN_SECONDS:
                        print(f"🚗 DETECTED: {plate_text} | Confidence: {int(avg_conf*100)}%")
                        
                        timestamp = int(current_time)
                        filename = f"{SAVE_FOLDER}/plate_{plate_text}_{timestamp}.jpg"
                        cv2.imwrite(filename, plate_img)
                        print(f"📸 Saved: {filename}")
                        
                        cv2.imshow("LAST CAPTURE", plate_img)
                        last_capture_time = current_time

            except Exception as e:
                pass

    cv2.imshow("Live Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()