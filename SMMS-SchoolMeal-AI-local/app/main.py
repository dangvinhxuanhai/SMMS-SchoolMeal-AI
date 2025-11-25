# app/main.py
from fastapi import FastAPI
from app.api.v1.recommend import router as recommend_router
from app.api.v1.admin import router as admin_router

app = FastAPI(title="EduMeal AI Menu Service")

app.include_router(recommend_router, prefix="/api/v1/menu", tags=["menu"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"]) 