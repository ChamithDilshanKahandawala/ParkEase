import uvicorn
import cv2
import numpy as np
import io
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# --- Configuration & Model Loading ---

# 1. Initialize FastAPI App
app = FastAPI(
    title="ParkEase AI Service",
    description="Provides vehicle and license plate detection.",
    version="1.0.0"
)

# 2. Load AI Models (This happens only once at startup)
try:
    print("Loading AI models...")
    
    # Load YOLOv8 Model (The "Detector")
    # Make sure 'best.pt' is in the same folder
    yolo_model = YOLO("best.pt")
    print("✅ YOLOv8 model loaded successfully.")

    # Load EasyOCR Model (The "Reader")
    # This will download models on first run
    ocr_reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a compatible GPU
    print("✅ EasyOCR model loaded successfully.")
    
    print("\n--- AI Service is Ready ---")
    
except Exception as e:
    print(f"❌ Critical Error: Could not load AI models: {e}")
    # In a real-world app, you might want to exit if models fail to load
    # raise e 

# --- Helper Functions ---

def clean_plate_text(text: str) -> str:
    """Utility to clean the raw text from EasyOCR."""
    if not text:
        return ""
    
    # Keep only alphanumeric characters, convert to uppercase
    # This is tuned for Sri Lankan plates (e.g., "CBA 1234" -> "CBA1234")
    return "".join(filter(str.isalnum, text)).upper()

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint to check if the service is running."""
    return {"status": "ParkEase AI Service is running."}


@app.post("/api/ai/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    Receives an image, finds a license plate (YOLO), and reads it (EasyOCR).
    """
    try:
        # 1. Read image from upload
        image_data = await file.read()
        
        # Convert image data to an OpenCV (numpy) array
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2. Run YOLO Detection (Find the Plate)
        # We run detection on the full image
        results = yolo_model(img)

        plate_detections = []
        for result in results:
            for box in result.boxes:
                # Check if the detected object is a license plate
                # We assume your custom model's class 0 is "license_plate"
                if int(box.cls[0]) == 0: 
                    plate_detections.append({
                        'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        'conf': box.conf[0].cpu().numpy()
                    })

        # 3. Check if any plates were found
        if not plate_detections:
            print("ℹ️ No license plate detected.")
            return JSONResponse(content={"plate_number": None, "confidence": 0.0})

        # 4. Select the best plate (highest confidence)
        best_plate = max(plate_detections, key=lambda p: p['conf'])
        x1, y1, x2, y2 = best_plate['bbox'].astype(int)
        detection_confidence = float(best_plate['conf'])
        
        print(f"🎯 Plate detected with {detection_confidence*100:.1f}% confidence.")

        # 5. Crop the image to *only* the license plate
        # Add a small pixel padding to ensure we get the whole plate
        padding = 5 
        cropped_plate_img = img[
            max(y1 - padding, 0):min(y2 + padding, img.shape[0]),
            max(x1 - padding, 0):min(x2 + padding, img.shape[1])
        ]
        
        # 6. Run EasyOCR (Read the Plate)
        # We run the reader on the *small cropped image*
        ocr_result = ocr_reader.readtext(cropped_plate_img)

        if not ocr_result:
            print("⚠️ Plate detected, but OCR could not read any text.")
            return JSONResponse(content={"plate_number": None, "confidence": 0.0})
        
        # 7. Clean and return the text
        # We assume the first result is the most likely plate number
        raw_text = ocr_result[0][1]
        ocr_confidence = ocr_result[0][2]
        
        cleaned_text = clean_plate_text(raw_text)
        
        print(f"📖 OCR Result: '{raw_text}' -> Cleaned: '{cleaned_text}' (Conf: {ocr_confidence:.2f})")

        return JSONResponse(content={
            "plate_number": cleaned_text,
            "confidence": ocr_confidence
        })

    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# --- Main execution ---

if __name__ == "__main__":
    """
    Run the FastAPI server using uvicorn.
    'reload=True' will auto-restart the server when you save changes.
    """
    print("--- Starting ParkEase AI Service ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)