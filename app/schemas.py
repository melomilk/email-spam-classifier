from pydantic import BaseModel

class EmailRequest(BaseModel):
    subject: str | None = None
    body: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float