pet project: email spam classifier api

a machine learning api that classifies emails as **spam** or **ham** - screening both raw text and image screenshots via OCR.

built as a pet project combining nlp + computer vision.

features:

text-based classification (TF-IDF + logistic regression)
image-based classification via EasyOCR (English, Russian, Kazakh)
confidence threshold — returns `uncertain` if confidence < 0.6
REST API with FastAPI + swagger UI
dockerized for easy deployment!

model performance

| type of model | accuracy | F1 Score |
|---|---|---|
| logistic regression | 97.8% | 0.97 |
| multinomial NB | 96.1% | 0.96 |
| bernoulli NB | 95.4% | 0.95 |

evaluated on 20% holdout test set from UCI SMS Spam Collection dataset.

tech stack

| layer | tools |
| ML | scikit-learn, TF-IDF, logistic regression |
| OCR | easyOCR, openCV |
| API | fastAPI, uvicorn |
| Deploy | docker |

API endpoints

| Method | Endpoint | Description |
| POST | `/predict` | classify email text |
| POST | `/predict-image` | classify email screenshot via OCR |
| GET | `/` | health check |

run locally:

```bash
git clone https://github.com/yourusername/email-spam-classifier
cd email-spam-classifier
pip install -r requirements.txt
uvicorn app.main:app --reload
```

run with docker!

```bash
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```

Then open: http://localhost:8000/docs

---

## 📁 Project Structure
email-classifier/ 
├── app/ │ 
├── main.py # FastAPI routes │ 
├── model.py # ML + OCR inference 
│ └── schemas.py # pydantic models 
├── ml/ │ 
├── train.py # model training │ 
├── model.pkl # trained model 
│ └── vectorizer.pkl # TF-IDF vectorizer 
├── Dockerfile 
└── requirements.txt
