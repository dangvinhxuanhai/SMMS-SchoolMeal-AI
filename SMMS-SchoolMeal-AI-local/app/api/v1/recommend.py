# app/api/v1/recommend.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.schemas import RecommendRequest, RecommendResponse
from app.core.db import get_db
from app.services.recommender import MenuRecommender

router = APIRouter()
recommender = MenuRecommender.load()

@router.post("/recommend", response_model=RecommendResponse)
def recommend_menu(req: RecommendRequest, db: Session = Depends(get_db)):
    return recommender.recommend(db, req)
