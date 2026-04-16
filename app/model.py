import pickle
import easyocr

reader = easyocr.Reader(['en', 'ru'])

with open("ml/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ml/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict(text: str):
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0])
    if confidence < 0.6:
        label_str = "uncertain"
    else:
        label_str = "spam" if str(label) == "1" else "ham"
        return label_str, round(float(confidence), 4)

def extract_text_from_image(image_bytes: bytes) -> str:
    import numpy as np
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = reader.readtext(img)
    text = ' '.join([item[1] for item in result])
    return text