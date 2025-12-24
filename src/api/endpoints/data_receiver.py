# src/api/endpoints/data_receiver.py
from fastapi import APIRouter, HTTPException
import os
from pydantic import BaseModel

router = APIRouter()


# 定义接收数据的模型
class MerchantInfo(BaseModel):
    merchant_id: str
    name: str
    category: str
    rating: float
    description: str
    # 可以根据需要添加更多字段，如地址、营业时间、推荐菜品等


class UserInfo(BaseModel):
    user_id: str
    preferences: list[str]  # 如 ["川菜", "不辣", "性价比高"]
    # 可以添加历史订单等信息


@router.post("/api/receive/merchant")
async def receive_merchant_info(merchant: MerchantInfo):
    """接收商户信息并保存到 TXT 文件"""
    try:
        # 将商户信息格式化为一行文本
        # 示例格式: merchant_id|name|category|rating|description
        line = f"{merchant.merchant_id}|{merchant.name}|{merchant.category}|{merchant.rating}|{merchant.description}\n"

        # 确保知识库目录存在
        os.makedirs("knowledge_base", exist_ok=True)

        # 追加到 TXT 文件
        with open("knowledge_base/merchants.txt", "a", encoding="utf-8") as f:
            f.write(line)

        return {"status": "success", "message": "商户信息已保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@router.post("/api/receive/user")
async def receive_user_info(user: UserInfo):
    """接收用户信息（可选，用户信息也可以在查询时直接传递）"""
    # 类似地，你可以将用户信息保存起来，用于后续的个性化推荐
    # 这里简化处理，只返回成功
    return {"status": "success", "message": "用户信息已收到"}