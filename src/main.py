from fastapi import FastAPI
from src.api.conversation_api import router

app = FastAPI(title="XYZ Analytics API")

app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}
