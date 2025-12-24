import chromadb
import os
from typing import List, Dict, Optional
from src.rag.langchain_utils import get_embeddings  # 新增LangChain嵌入导入

current_script_dir = os.path.dirname(os.path.abspath(__file__))
MERCHANT_FILE_PATH = os.path.join(current_script_dir, "..", "knowledge_base", "merchants.txt")

print(f"📌 实际商户文件路径：{MERCHANT_FILE_PATH}")

class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.EMBED_MODEL = "nomic-embed-text:latest"
        self.COLLECTION_NAME = "merchant_db"
        self.PERSIST_DIR = persist_dir

        self.client = chromadb.PersistentClient(path=self.PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Chroma 集合 '{self.COLLECTION_NAME}' 初始化完成")

    def _extract_tags(self, description: str) -> Dict[str, str]:
        tags = {
            "taste": "", "scene": "", "price": "",
            "招牌": "", "配送": "", "优惠": ""
        }
        if "【口味：" in description:
            start = description.find("【口味：") + 5
            end = description.find("】", start)
            tags["taste"] = description[start:end].strip() if end != -1 else ""
        if "【场景：" in description:
            start = description.find("【场景：") + 5
            end = description.find("】", start)
            tags["scene"] = description[start:end].strip() if end != -1 else ""
        price_keywords = ["10元", "15元", "20元", "平价", "高性价比", "起送价"]
        for kw in price_keywords:
            if kw in description:
                tags["price"] += kw + "|"
        tags["price"] = tags["price"].rstrip("|")

        if "【招牌：" in description:
            start = description.find("【招牌：") + 5
            end = description.find("】", start)
            tags["招牌"] = description[start:end].strip() if end != -1 else ""
        if "【配送：" in description:
            start = description.find("【配送：") + 5
            end = description.find("】", start)
            tags["配送"] = description[start:end].strip() if end != -1 else ""
        if "【优惠：" in description:
            start = description.find("【优惠：") + 5
            end = description.find("】", start)
            tags["优惠"] = description[start:end].strip() if end != -1 else ""

        return tags

    def _extract_item_tags(self, item_part: str) -> str:
        if item_part.startswith("【") and item_part.endswith("】"):
            return item_part[1:-1].strip()
        return ""

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """替换为LangChain的Ollama嵌入生成（兼容原有重试逻辑）"""
        max_retry = 2
        for retry in range(max_retry):
            try:
                # 使用LangChain封装的嵌入模型
                embedding = get_embeddings().embed_query(text)
                if len(embedding) >= 100:
                    return embedding
                print(f"⚠️  第 {retry+1} 次生成嵌入失败：向量无效")
            except Exception as e:
                print(f"⚠️  第 {retry+1} 次生成嵌入报错：{str(e)}")
        return None

    def load_data(self, file_path: str = MERCHANT_FILE_PATH):
        if not os.path.exists(file_path):
            print(f"❌ 知识库文件不存在：{file_path}")
            return

        print(f"📥 正在加载数据：{file_path}")
        documents, metadatas, embeddings_list, ids = [], [], [], []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f.readlines(), 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 5:
                    print(f"❌ 跳过格式错误行（{line_num}）：{line}")
                    continue

                merchant_id = parts[0].strip()
                name = parts[1].strip()
                category = parts[2].strip()
                rating = parts[3].strip()
                item_part = parts[4].strip()
                description = "|".join(parts[5:]).strip()

                item_tags = self._extract_item_tags(item_part)
                tags = self._extract_tags(description)

                embed_text = (
                    f"实物关键词：{item_tags} | 实物关键词：{item_tags} | 实物关键词：{item_tags} | "
                    f"商户名称：{name} | 口味：{tags['taste']} | 场景：{tags['scene']} | 评分：{rating}"
                )

                embedding = self._get_embedding(embed_text)
                if not embedding:
                    print(f"❌ 跳过商户（{name}）：嵌入生成失败")
                    continue

                metadatas.append({
                    "merchant_id": merchant_id,
                    "name": name,
                    "category": category,
                    "rating": float(rating) if rating.replace(".", "").isdigit() else 0,
                    "item_tags": item_tags,
                    "taste": tags["taste"],
                    "scene": tags["scene"],
                    "price": tags["price"],
                    "招牌": tags["招牌"],
                    "配送": tags["配送"],
                    "优惠": tags["优惠"],
                    "raw": line
                })
                print(f"📥 存储商户：{name} | 招牌：{tags['招牌']} | 配送：{tags['配送']} | 优惠：{tags['优惠']}")
                documents.append(embed_text)
                embeddings_list.append(embedding)
                ids.append(f"merchant_{merchant_id}")

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list,
                ids=ids
            )
            print(f"✅ 成功加载 {len(documents)} 条商户数据（含实物标签+招牌+配送+优惠）")
        else:
            print(f"❌ 无有效商户数据加载")

    def search(self, query_text: str, top_k: int = 15) -> Dict[str, List]:
        """原有检索方法（保留，兼容旧逻辑）"""
        print(f"🔍 向量检索：查询='{query_text}'")
        query_embedding = self._get_embedding(query_text)
        if not query_embedding:
            print("❌ 检索失败：查询嵌入生成失败")
            return {"documents": [], "metadatas": [], "distances": []}
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=None,
                include=["documents", "metadatas", "distances"]
            )
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
        except Exception as e:
            print(f"❌ 检索执行失败：{str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}

    def get_langchain_retriever(self, top_k: int = 15):
        """新增：获取LangChain兼容的检索器"""
        from langchain_chroma import Chroma
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding_function=get_embeddings()
        )
        return langchain_chroma.as_retriever(search_kwargs={"k": top_k})

    def search_with_retriever(self, query_text: str, top_k: int = 15) -> Dict[str, List]:
        """新增：通过LangChain检索器检索（兼容原有返回格式）"""
        try:
            retriever = self.get_langchain_retriever(top_k=top_k)
            docs = retriever.invoke(query_text)
            metadatas = [doc.metadata for doc in docs]
            documents = [doc.page_content for doc in docs]
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": [0.0]*len(docs)  # 兜底兼容
            }
        except Exception as e:
            print(f"❌ LangChain检索失败：{str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}

# 初始化并加载数据
kb = KnowledgeBase()
kb.load_data()