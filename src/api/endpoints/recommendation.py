from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.rag.rag_engine import generate_rag_response

router = APIRouter()

# 扩展请求模型：接收 Java 传递的四维度信息
class RecommendationRequest(BaseModel):
    user_query: str = ""  # 用户提问（允许为空，支持主动推荐）
    user_purchase_history: str = None  # Java 传递的用户历史购买信息（如“近30天常买奶茶”）
    user_action: str = None  # Java 传递的用户操作信息（如“浏览火锅分类”）
    weather_info: str = None  # Java 传递的天气信息（如“深圳晴，35℃”）
    sessionId: str = None  # 会话ID（用于上下文关联，Java 传递）

class RecommendationResponse(BaseModel):
    response: str  # 返回给 Java 的推荐结果

@router.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """接收 Java 传递的四维度信息，返回 RAG 推荐结果"""
    try:
        # 校验：用户查询和操作信息不能同时为空（至少一个用于推荐）
        if not request.user_query.strip() and not request.user_action:
            raise HTTPException(status_code=400, detail="用户查询和操作信息不能同时为空")

        # 调用 RAG 引擎（传递四维度信息）
        recommendation = generate_rag_response(
            user_query=request.user_query,
            user_purchase_history=request.user_purchase_history,
            user_action=request.user_action,
            weather_info=request.weather_info
        )

        return {"response": recommendation}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成推荐失败: {str(e)}")