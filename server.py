from typing import Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from gold_predictor import get_model

app = FastAPI()


class GoldPredictorRequest(BaseModel):
    review: str

class GoldPredictorResponse(BaseModel):
    prediction: str
    gold_prob: float
    fake_prob: float
    review_words: list
    attention_scores: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict", response_model=GoldPredictorResponse)
def predict(request: GoldPredictorRequest, model=Depends(get_model)):
    prediction, normalized_score, review_words, attention_scores = model.predict(request.review)    
    return GoldPredictorResponse(
        prediction=prediction,
        gold_prob=normalized_score[0],
        fake_prob=normalized_score[1],
        review_words=review_words,
        attention_scores=attention_scores
    )