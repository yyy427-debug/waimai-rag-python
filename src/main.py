# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import data_receiver, recommendation
from src.config import API_TITLE, API_VERSION, API_DESCRIPTION

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# 添加 CORS 中间件，允许 Java 后端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中，应指定具体的 Java 后端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(data_receiver.router)
app.include_router(recommendation.router)

@app.get("/")
async def root():
    return {"message": "欢迎使用智能外卖推荐 RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)