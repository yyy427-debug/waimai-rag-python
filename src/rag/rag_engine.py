from typing import Optional, Dict, List, Union
import os
import re
from src.rag.knowledge_base import kb
# 替换原有ollama调用，导入LangChain工具
from src.rag.langchain_utils import (
    call_llm, call_llm_with_retry,
    extract_core_items_prompt, passive_recommend_prompt, active_recommend_prompt
)

# 基础配置
SMALL_MODEL = "qwen3:1.7b"
LARGE_MODEL = "qwen3:1.7b"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
MERCHANT_FILE_PATH = os.path.join(current_script_dir, "..", "knowledge_base", "merchants.txt")
print(f"📌 实际商户文件路径：{MERCHANT_FILE_PATH}")

# item语义扩展字典
ITEM_EXPAND = {
    "炸鸡": ["炸鸡腿", "香辣鸡", "炸物"],
    "面条": ["拉面", "汤面", "挂面"],
    "奶茶": ["奶绿", "果茶", "茶饮"],
    "烤串": ["烤鸡翅", "烤肉类"],
    "麻辣烫": ["麻辣香锅", "冒菜"],
    "饺子": ["水饺", "蒸饺"],
    "汉堡": ["汉堡包", "鸡肉堡"],
    "咖啡": ["美式", "拿铁", "卡布奇诺"],
    "甜品": ["蛋糕", "甜点", "马卡龙"],
    "水煮鱼": ["酸菜鱼", "麻辣鱼"],
    "剁椒鱼头": ["鱼头", "辣鱼头"],
    "小龙虾": ["香辣虾", "蒜蓉虾"]
}

# 主动推荐权重配置
WEIGHT_CONFIG = {
    "purchase_core": 5,
    "purchase_extend": 3,
    "browse_core": 3,
    "browse_extend": 1,
    "frequency_per_10": 1,
    "weather_fit": 2
}


# ========== 商户字段解析工具函数 ==========
def _parse_merchant_field(part: str, prefix: str) -> str:
    cleaned = part.replace("【", "").replace("】", "").replace("[", "").replace("]", "").strip()
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):].strip()
    return cleaned


# ========== 实物标签提取 ==========
def _get_item_reference_list() -> List[str]:
    item_tags = []
    abs_path = os.path.abspath(MERCHANT_FILE_PATH)
    print(f"🔍 读取商户文件：{abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ 商户文件不存在！")
        return ["炸鸡", "汉堡", "面条", "奶茶", "烤串"]

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"✅ 读取{len(lines)}行数据，逐行提取标签...")

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                line = line.replace("｜", "|")
                parts = [p.strip() for p in line.split("|") if p.strip()]

                if len(parts) < 11:
                    print(f"⚠️  第{line_num}行字段不足11个，跳过")
                    continue

                item_part = parts[4]
                item_tags_str = _parse_merchant_field(item_part, prefix="")
                item_tags_str = item_tags_str.replace("，", ",").replace("、", ",")
                items = [i.strip() for i in item_tags_str.split(",") if i.strip()]

                if items:
                    item_tags.extend(items)
                    print(f"ℹ️  第{line_num}行（{parts[1]}）提取标签：{items}")

    except Exception as e:
        print(f"❌ 读取文件异常，使用兜底标签：{str(e)}")
        item_tags = ["炸鸡", "汉堡", "面条", "奶茶", "烤串", "麻辣烫", "饺子"]

    reference = list(set(item_tags))
    print(f"📋 最终实物参考范围（共{len(reference)}个）：{reference}")
    return reference


# ========== 修复：新增商户商品标签提取函数 ==========
def _get_merchant_item_tags(meta: Dict) -> str:
    item_tags = meta.get("item_tags", "").strip()
    if item_tags:
        return item_tags.lower()

    raw_line = meta.get("raw", "").strip()
    if raw_line:
        parts = raw_line.split("|")
        if len(parts) >= 5:
            item_part = parts[4].strip()
            item_part = _parse_merchant_field(item_part, prefix="")
            item_part = item_part.replace("，", ",").replace("、", ",")
            return item_part.lower()

    return ""


# ========== 小模型提取3个商品标签（替换为LangChain调用） ==========
def _extract_core_item(user_query: str) -> List[str]:
    item_reference = _get_item_reference_list()
    if not item_reference:
        print(f"⚠️  参考范围为空，返回默认标签")
        return ["水煮鱼", "剁椒鱼头", "麻辣烫"]

    try:
        # 使用LangChain标准化Prompt模板
        prompt = extract_core_items_prompt()
        # 调用LangChain封装的小模型
        core_items_str = call_llm(
            prompt=prompt,
            input_data={
                "user_query": user_query,
                "item_reference": item_reference
            },
            llm_type="small"
        )
        core_items = core_items_str.strip().split(",")
        core_items = [item.strip() for item in core_items if item.strip() in item_reference]
        print(f"ℹ️  小模型提取结果（清洗后）：{core_items}")

        if len(core_items) < 3:
            for item in item_reference:
                if item not in core_items and any(kw in item for kw in user_query.split()):
                    core_items.append(item)
                if len(core_items) >= 3:
                    break
        return core_items[:3]

    except Exception as e:
        print(f"❌ 小模型调用失败，手动匹配：{str(e)}")
        matched = [item for item in item_reference if any(kw in item for kw in user_query.split())][:3]
        if len(matched) < 3:
            default_spicy = ["水煮鱼", "剁椒鱼头", "麻辣烫"]
            for item in default_spicy:
                if item not in matched:
                    matched.append(item)
                if len(matched) >= 3:
                    break
        return matched


# ========== 原有配置 ==========
SCENE_MAPPING = {
    "午餐": ["午餐", "工作餐", "单人餐", "简餐", "上班族午餐"],
    "晚餐": ["晚餐", "家庭餐", "多人聚餐", "居家聚餐"],
    "深夜": ["深夜", "22:00-04:00", "深夜简餐"],
    "下午茶": ["下午茶", "打卡", "甜品", "饮品"],
    "单人餐": ["单人餐", "工作餐", "简餐"],
    "多人": ["多人聚餐", "家庭餐", "朋友小聚"]
}

DEMAND_KEYWORDS = {
    "core_type": {
        "吃": "餐食",
        "喝": "饮品",
        "外卖": "外卖",
        "甜品": "甜品",
        "咖啡": "饮品",
        "奶茶": "饮品"
    },
    "taste": {
        "清淡": "清淡",
        "低卡": "低卡",
        "辣": "辣",
        "麻辣": "麻辣",
        "香辣": "香辣",
        "偏辣": "偏辣",
        "酸辣": "酸辣",
        "甜": "甜",
        "咸香": "咸香"
    },
    "scene": {
        "午餐": "午餐",
        "工作餐": "午餐",
        "晚餐": "晚餐",
        "深夜": "深夜",
        "下午茶": "下午茶",
        "单人餐": "单人餐",
        "多人": "多人"
    },
    "price": {
        "便宜": "平价",
        "平价": "平价",
        "高性价比": "高性价比",
        "10元": "10元",
        "15元": "15元"
    }
}


# ========== 需求解析 ==========
def _parse_demand(user_query: str,
                  user_action: Optional[str] = None,
                  user_purchase_history: Optional[str] = None) -> Dict[str, Union[List[str], str]]:
    all_context = f"{user_query} {user_action or ''} {user_purchase_history or ''}".lower()
    demand = {
        "core_type": "外卖",
        "item": [item.lower() for item in _extract_core_item(user_query)],
        "taste": "",
        "scene": "",
        "price": ""
    }

    for kw, core_type in DEMAND_KEYWORDS["core_type"].items():
        if kw in all_context:
            demand["core_type"] = core_type
            break
    for kw, taste in DEMAND_KEYWORDS["taste"].items():
        if kw in all_context:
            demand["taste"] = taste
            break
    for kw, scene in DEMAND_KEYWORDS["scene"].items():
        if kw in all_context:
            demand["scene"] = scene
            break
    for kw, price in DEMAND_KEYWORDS["price"].items():
        if kw in all_context:
            demand["price"] = price
            break

    print(f"📊 解析需求结果：{demand}")
    return demand


# ========== 商户信息提取 ==========
def _get_merchant_meta(meta: Dict) -> Dict[str, str]:
    raw_line = meta.get("raw", "").strip()
    parts = raw_line.split("|") if raw_line else []

    delivery = "无"
    if len(parts) > 9:
        delivery = _parse_merchant_field(parts[9], prefix="配送：")

    discount = "无"
    if len(parts) > 10:
        discount = _parse_merchant_field(parts[10], prefix="优惠：")

    signature = meta.get("招牌", "")
    if not signature and len(parts) > 7:
        signature = _parse_merchant_field(parts[7], prefix="招牌：")

    rating = meta.get("rating", "0")
    if len(parts) > 3:
        rating = parts[3].strip()

    return {
        "name": meta.get("name", "未知"),
        "rating": rating,
        "signature": signature,
        "taste": meta.get("taste", "无"),
        "delivery": delivery,
        "discount": discount
    }


# ========== 大模型Prompt（替换为LangChain模板） ==========
def _build_prompt(demand: Dict[str, Union[List[str], str]], retrieved_metadatas: List[Dict]) -> str:
    merchant_info = []
    for idx, meta in enumerate(retrieved_metadatas, 1):
        info = _get_merchant_meta(meta)
        merchant_line = (
            f"{idx}. 商户名称：{info['name']} | 评分：{info['rating']} | "
            f"招牌商品：{info['signature']} | 口味：{info['taste']} | "
            f"配送（原始数据）：{info['delivery']} | 优惠（原始数据）：{info['discount']}"
        )
        merchant_info.append(merchant_line)
    merchant_text = "\n".join(merchant_info)

    demand_items = demand["item"] if isinstance(demand["item"], list) else [demand["item"]]

    # 使用LangChain标准化Prompt模板
    prompt = passive_recommend_prompt()
    input_data = {
        "demand_items": demand_items,
        "merchant_text": merchant_text,
        "demand_desc": f"{demand['core_type']}（想吃：{','.join(demand_items) if demand_items else '无'}，口味：{demand['taste'] or '无'}）"
    }
    return prompt.format(**input_data)


# ========== 主动推荐辅助函数 ==========
def _extract_action_tags(action_text: Optional[str], is_purchase: bool = False) -> Dict[str, List[str]]:
    if not action_text:
        print(f"ℹ️  行为文本为空，返回空标签")
        return {"core": [], "extend": []}

    item_reference = _get_item_reference_list()
    core_tags = []
    for tag in item_reference:
        if tag in action_text:
            core_tags.append(tag)
    core_tags = list(set(core_tags))
    print(f"ℹ️  提取核心标签：{core_tags}（{'购买' if is_purchase else '浏览'}行为）")

    extend_tags = []
    for tag in core_tags:
        extend_tags.extend(ITEM_EXPAND.get(tag, []))
    extend_tags = list(set([t for t in extend_tags if t in item_reference]))
    print(f"ℹ️  提取扩展标签：{extend_tags}（{'购买' if is_purchase else '浏览'}行为）")

    return {"core": core_tags, "extend": extend_tags}


def _is_tag_fit_weather(tag: str, weather_info: Optional[str]) -> bool:
    if not weather_info:
        print(f"ℹ️  天气信息为空，默认不适配")
        return False

    is_sunny = "晴" in weather_info
    temp_match = re.search(r"(\d+)℃", weather_info)
    temp = int(temp_match.group(1)) if temp_match else 25

    if is_sunny and temp >= 25:
        fit_tags = ["奶茶", "果茶", "减脂餐", "轻食", "沙拉", "冷饮"]
    elif "雨" in weather_info or temp <= 15:
        fit_tags = ["汉堡", "炸鸡", "热饮", "咖啡", "面条", "麻辣烫"]
    else:
        fit_tags = []

    all_fit_tags = fit_tags + [e for t in fit_tags for e in ITEM_EXPAND.get(t, [])]
    fit_result = tag in all_fit_tags or any(e in all_fit_tags for e in ITEM_EXPAND.get(tag, []))
    print(f"ℹ️  标签[{tag}] 天气适配：{fit_result}（当前天气：{weather_info}）")
    return fit_result


def _calculate_tag_score(tag: str, purchase_tags: Dict[str, List[str]], browse_tags: Dict[str, List[str]],
                         weather_info: Optional[str], user_purchase_history: Optional[str]) -> float:
    score = 0.0

    if tag in purchase_tags["core"]:
        score += WEIGHT_CONFIG["purchase_core"]
        score_type = "购买核心标签"
    elif tag in purchase_tags["extend"]:
        score += WEIGHT_CONFIG["purchase_extend"]
        score_type = "购买扩展标签"
    elif tag in browse_tags["core"]:
        score += WEIGHT_CONFIG["browse_core"]
        score_type = "浏览核心标签"
    elif tag in browse_tags["extend"]:
        score += WEIGHT_CONFIG["browse_extend"]
        score_type = "浏览扩展标签"
    else:
        score_type = "无匹配标签类型"

    frequency_score = 0.0
    if tag in purchase_tags["core"] or tag in purchase_tags["extend"]:
        if user_purchase_history:
            count_match = re.search(r"(\d+)次", user_purchase_history)
            if count_match:
                count = int(count_match.group(1))
                frequency_score = (count // 10) * WEIGHT_CONFIG["frequency_per_10"]
                frequency_score = min(frequency_score, 3)
    if frequency_score > 0:
        score += frequency_score
        print(f"ℹ️  标签[{tag}] 频次加分：{frequency_score}（基础分：{score - frequency_score}）")

    if _is_tag_fit_weather(tag, weather_info):
        score += WEIGHT_CONFIG["weather_fit"]
        print(f"ℹ️  标签[{tag}] 天气加分：{WEIGHT_CONFIG['weather_fit']}（当前得分：{score}）")

    print(f"ℹ️  标签[{tag}] 综合得分：{score}（类型：{score_type}）")
    return score


def _filter_diverse_tags(tags: List[str]) -> List[str]:
    fast_food_tags = ["汉堡", "炸鸡", "薯条", "烤串", "麻辣烫", "饺子"]
    fast_food_count = 0
    filtered_tags = []

    for tag in tags:  # 修复原错误tagss
        if tag in fast_food_tags:
            if fast_food_count < 2:
                filtered_tags.append(tag)
                fast_food_count += 1
        else:
            filtered_tags.append(tag)

    print(f"ℹ️  多样性过滤前标签：{tags}")
    print(f"ℹ️  多样性过滤后标签：{filtered_tags}")
    return filtered_tags


def _cold_start_recommend(weather_info: Optional[str]) -> str:
    print(f"ℹ️  无任何行为标签，启动冷启动推荐")

    weather_tags = []
    if weather_info and "晴" in weather_info and "25℃" in weather_info:
        weather_tags = ["奶茶", "减脂餐", "果茶"]
    else:
        weather_tags = ["汉堡", "炸鸡", "咖啡"]
    print(f"ℹ️  冷启动天气适配标签：{weather_tags}")

    all_data = kb.collection.get(include=["metadatas"])
    high_score_merchants = []
    for meta in all_data["metadatas"]:
        merchant_item_tags = _get_merchant_item_tags(meta)
        merchant_rating = float(_get_merchant_meta(meta)["rating"])
        if merchant_rating >= 4.7 and any(tag in merchant_item_tags for tag in weather_tags):
            high_score_merchants.append(meta)

    high_score_merchants.sort(key=lambda x: _get_merchant_meta(x)["rating"], reverse=True)
    valid_merchants = high_score_merchants[:3]
    print(f"ℹ️  冷启动高评分商户：{[m.get('name') for m in valid_merchants]}")

    if valid_merchants:
        prompt = _build_passive_prompt(valid_merchants, weather_info)
        # 使用带重试的LangChain调用
        response = call_llm_with_retry(
            prompt=prompt,
            llm_type="large"
        )
        return response
    else:
        return "为你推荐热门商户：肯德基（评分4.8）、喜茶（评分4.9）、轻食工坊（评分4.7）"


# ========== 主动推荐Prompt（替换为LangChain模板） ==========
def _build_passive_prompt(retrieved_metadatas: List[Dict], weather_info: Optional[str]) -> str:
    merchant_info = []
    for idx, meta in enumerate(retrieved_metadatas, 1):
        info = _get_merchant_meta(meta)
        merchant_line = (
            f"{idx}. 商户名称：{info['name']} | 评分：{info['rating']} | "
            f"招牌商品：{info['signature']} | 口味：{info['taste']} | "
            f"配送（原始数据）：{info['delivery']} | 优惠（原始数据）：{info['discount']}"
        )
        merchant_info.append(merchant_line)
    merchant_text = "\n".join(merchant_info)

    # 使用LangChain标准化Prompt模板
    prompt = active_recommend_prompt()
    input_data = {
        "merchant_text": merchant_text,
        "weather_desc": weather_info or "未知天气"
    }
    return prompt.format(**input_data)


# ========== 主流程 ==========
def generate_rag_response(
        user_query: str,
        user_purchase_history: Optional[str] = None,
        user_action: Optional[str] = None,
        weather_info: Optional[str] = None,
        session_id: Optional[str] = None  # 新增：兼容会话ID
) -> str:
    try:
        print(f"\n==================================================")
        print(f"📥 接收请求：查询={user_query} | 行为={user_action} | 历史={user_purchase_history}")
        print(f"==================================================")

        if not user_query.strip():
            print(f"==================================================")
            print(f"ℹ️  检测到用户无明确查询，启动主动推荐流程")
            print(f"==================================================")

            purchase_tags = _extract_action_tags(user_purchase_history, is_purchase=True)
            browse_tags = _extract_action_tags(user_action, is_purchase=False)

            all_tags = list(set(
                purchase_tags["core"] + purchase_tags["extend"] +
                browse_tags["core"] + browse_tags["extend"]
            ))

            if not all_tags:
                return _cold_start_recommend(weather_info)

            tag_scores = {}
            for tag in all_tags:
                tag_scores[tag] = _calculate_tag_score(
                    tag=tag,
                    purchase_tags=purchase_tags,
                    browse_tags=browse_tags,
                    weather_info=weather_info,
                    user_purchase_history=user_purchase_history
                )

            sorted_tag_items = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_tags = [tag for tag, score in sorted_tag_items]
            top_tags = _filter_diverse_tags(sorted_tags)[:3]
            print(f"ℹ️  最终优先推荐标签（Top3）：{top_tags}（得分：{[tag_scores[tag] for tag in top_tags]}）")

            matched_merchants = []
            for tag in top_tags:
                print(f"==================================================")
                print(f"ℹ️  为标签[{tag}]匹配商户")
                print(f"==================================================")
                expand_words = ITEM_EXPAND.get(tag, [])
                search_query = " ".join(list(set(["外卖", tag] + expand_words)))
                print(f"🔍 检索关键词：{search_query}")

                # 可选：切换为LangChain检索器
                # search_results = kb.search_with_retriever(search_query, top_k=34)
                search_results = kb.search(search_query, top_k=34)  # 保留原有检索

                if len(search_results["metadatas"]) < 34:
                    print(f"⚠️  检索结果不足34条，全量获取所有商户")
                    all_data = kb.collection.get(include=["metadatas"])
                    search_results["metadatas"] = all_data["metadatas"] if all_data["metadatas"] else []
                print(f"ℹ️  检索到商户数：{len(search_results['metadatas'])}")

                for meta in search_results["metadatas"]:
                    info = _get_merchant_meta(meta)
                    merchant_name = info["name"]
                    merchant_rating = float(info["rating"])
                    merchant_item_tags = _get_merchant_item_tags(meta)
                    print(f"ℹ️  校验商户：{merchant_name}（标签：{merchant_item_tags} | 评分：{merchant_rating}）")
                    item_match = tag in merchant_item_tags
                    rating_match = merchant_rating >= 4.4
                    if item_match and rating_match:
                        matched_merchants.append(meta)
                        print(f"✅ 商户[{merchant_name}]匹配成功")

            print(f"==================================================")
            print(f"ℹ️  主动推荐商户去重+排序")
            print(f"==================================================")
            unique_merchants = []
            merchant_names = set()
            sorted_merchants = sorted(matched_merchants, key=lambda x: _get_merchant_meta(x)["rating"], reverse=True)
            for merchant in sorted_merchants:
                merchant_name = _get_merchant_meta(merchant)["name"]
                if merchant_name not in merchant_names:
                    merchant_names.add(merchant_name)
                    unique_merchants.append(merchant)
                if len(unique_merchants) >= 3:
                    break
            print(f"ℹ️  最终推荐商户：{[m.get('name') for m in unique_merchants]}")

            if unique_merchants:
                prompt = _build_passive_prompt(unique_merchants, weather_info)
                # 使用LangChain调用大模型
                result = call_llm_with_retry(
                    prompt=prompt,
                    llm_type="large"
                )
                print(f"ℹ️  主动推荐结果：{result}")
                return result
            else:
                return "根据你的偏好和天气，暂时没有找到合适的外卖推荐，可尝试搜索具体品类~"

        # 有明确查询时的推荐流程
        demand = _parse_demand(user_query, user_action, user_purchase_history)
        target_items = demand["item"]
        target_core_type = demand["core_type"]

        if not target_items or all(item == "" for item in target_items):
            return "抱歉，未识别到您想吃的食物~"

        expand_words = []
        for item in target_items:
            expand_words.extend(ITEM_EXPAND.get(item, []))
        search_parts = [target_core_type] + target_items + expand_words
        search_query = " ".join(list(set(search_parts)))
        print(f"🔍 检索关键词（多商品+扩展）：{search_query}")

        # 可选：切换为LangChain检索器
        # search_results = kb.search_with_retriever(search_query, top_k=34)
        search_results = kb.search(search_query, top_k=34)  # 保留原有检索

        print(f"ℹ️  检索到 {len(search_results['metadatas'])} 条商户数据")

        if len(search_results["metadatas"]) < 34:
            print(f"⚠️  检索结果不完整，全量获取所有商户")
            try:
                all_data = kb.collection.get(include=["metadatas"])
                search_results["metadatas"] = all_data["metadatas"] if all_data["metadatas"] else []
                print(f"✅ 全量获取成功，共 {len(search_results['metadatas'])} 条商户")
            except Exception as e:
                print(f"❌ 全量获取失败：{str(e)}")

        filtered_metadatas = []
        for meta in search_results["metadatas"]:
            info = _get_merchant_meta(meta)
            merchant_name = info["name"]
            merchant_rating = float(info["rating"])
            merchant_item_tags = _get_merchant_item_tags(meta)
            print(f"ℹ️  校验商户：{merchant_name}（读取到的标签：{merchant_item_tags} | 评分：{merchant_rating}）")
            item_match = any(target_item in merchant_item_tags for target_item in target_items)
            rating_match = merchant_rating >= 4.4
            if item_match and rating_match:
                filtered_metadatas.append(meta)
                print(f"✅ 匹配成功：{merchant_name}")

        filtered_metadatas.sort(key=lambda x: _get_merchant_meta(x)["rating"], reverse=True)
        valid_metadatas = filtered_metadatas[:3]
        print(f"ℹ️  最终有效商户数：{len(valid_metadatas)}")

        if not valid_metadatas:
            return f"抱歉，未找到提供「{','.join(target_items)}」的相关商户~"

        prompt = _build_prompt(demand, valid_metadatas)
        # 使用LangChain调用大模型（带重试）
        result = call_llm_with_retry(
            prompt=prompt,
            llm_type="large"
        )
        return result

    except Exception as e:
        print(f"❌ 流程异常：{str(e)}")
        import traceback
        traceback.print_exc()
        return "抱歉，系统繁忙，请稍后再试~"