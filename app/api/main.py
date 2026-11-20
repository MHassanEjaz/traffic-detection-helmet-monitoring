
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Traffic Detection and Helmet Monitoring API",
    description="Backend API for traffic detection, counting and helmet monitoring.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Traffic Detection API is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is working properly."
    }


app.include_router(router, prefix="/api", tags=["Video Processing"])
