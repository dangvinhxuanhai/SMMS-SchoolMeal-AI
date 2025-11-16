# app/models/schemas.py
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

class RecommendRequest(BaseModel):
    user_id: UUID
    school_id: UUID
    
    main_ingredient_ids: List[int] = []
    side_ingredient_ids: List[int] = []

    avoid_allergen_ids: List[int] = []
    max_main_kcal: Optional[float] = 600.0
    max_side_kcal: Optional[float] = 250.0

    top_k_main: int = 5
    top_k_side: int = 5

class DishDto(BaseModel):
    food_id: int
    food_name: str
    is_main_dish: bool
    total_kcal: Optional[float] = None
    score: float

class RecommendResponse(BaseModel):
    session_id: int         # ID log trong DB
    recommended_main: List[DishDto]
    recommended_side: List[DishDto]
