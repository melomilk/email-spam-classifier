from fastapi import FastAPI, HTTPException
from app.schemas import EmailRequest, PredictionResponse
from app.model import predict
import time
import logging
from app.model import predict, extract_text_from_image
from fastapi import FastAPI, HTTPException, UploadFile, File

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Spam Classifier API")

@app.post("/predict", response_model=PredictionResponse)
def classify_email(email: EmailRequest):
    start = time.time()
    try:
        text = f"{email.subject or ''} {email.body or ''}"
        label, confidence = predict(text)
        latency = round((time.time() - start) * 1000, 2)
        logging.info(f"Prediction: {label}, confidence: {confidence}")
        return PredictionResponse(
            label=label,
            confidence=confidence,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-image")
async def classify_email_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        text = extract_text_from_image(image_bytes)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in image")
        
        label, confidence = predict(text)
        
        return {
            "extracted_text": text,
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def health():
    return {"status": "ok"}