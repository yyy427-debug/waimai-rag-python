# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# FastAPI 配置
API_TITLE = "智能外卖推荐 RAG API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "一个基于 RAG 和 QWen 模型的智能外卖推荐服务"

# 知识库配置
KNOWLEDGE_BASE_TXT_PATH = os.getenv("KNOWLEDGE_BASE_TXT_PATH", "knowledge_base/merchants.txt")