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

    # mới: mức phạt tối đa theo tỉ lệ dị ứng (0–1)
    allergen_penalty_weight: float = 0.5
    """
    0   => bỏ qua tỉ lệ dị ứng, không phạt
    0.5 => nếu 100% học sinh bị dị ứng -> score món chứa allergen đó * (1 - 0.5) = 0.5
    1   => nếu 100% học sinh bị dị ứng -> score ~ 0
    """

class DishDto(BaseModel):
    food_id: int
    food_name: str
    is_main_dish: bool
    total_kcal: Optional[float] = None
    score: float

class RecommendResponse(BaseModel):
    recommended_main: List[DishDto]
    recommended_side: List[DishDto]
    message: Optional[str] = None
