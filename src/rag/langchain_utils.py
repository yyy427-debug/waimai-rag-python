"""
LangChain 工具封装模块（兼容低版本LangChain）
核心功能：
1. 标准化Ollama模型调用（单例模式，避免重复初始化）
2. Prompt模板化构建
3. 统一LLM调用接口（带输出解析、自定义重试机制）
4. 嵌入模型封装（兼容nomic-embed-text）
"""
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List, Dict, Union
import logging
import time

# ===================== 基础配置 =====================
# 模型名称（与原有rag_engine.py保持一致）
SMALL_MODEL = "qwen3:1.7b"       # 小模型：用于标签提取（低token、快响应）
LARGE_MODEL = "qwen3:1.7b"       # 大模型：用于推荐结果生成
EMBED_MODEL = "nomic-embed-text:latest"  # 嵌入模型：用于向量生成

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("langchain_utils")

# ===================== 单例模型实例 =====================
# 单例模式：避免重复创建模型实例，提升性能
_llm_small: Optional[OllamaLLM] = None
_llm_large: Optional[OllamaLLM] = None
_embeddings: Optional[OllamaEmbeddings] = None

def get_llm_small(temperature: float = 0.3, max_tokens: int = 30) -> OllamaLLM:
    """获取小模型实例（用于标签提取）"""
    global _llm_small
    if _llm_small is None:
        logger.info(f"初始化小模型：{SMALL_MODEL} (temperature={temperature}, max_tokens={max_tokens})")
        _llm_small = OllamaLLM(
            model=SMALL_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    return _llm_small

def get_llm_large(temperature: float = 0.1, max_tokens: int = 500) -> OllamaLLM:
    """获取大模型实例（用于推荐结果生成）"""
    global _llm_large
    if _llm_large is None:
        logger.info(f"初始化大模型：{LARGE_MODEL} (temperature={temperature}, max_tokens={max_tokens})")
        _llm_large = OllamaLLM(
            model=LARGE_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
    return _llm_large

def get_embeddings() -> OllamaEmbeddings:
    """获取嵌入模型实例（兼容nomic-embed-text）"""
    global _embeddings
    if _embeddings is None:
        logger.info(f"初始化嵌入模型：{EMBED_MODEL}")
        _embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return _embeddings

# ===================== Prompt模板工具 =====================
def build_prompt(
    template_str: str,
    input_variables: List[str],
    partial_variables: Optional[Dict[str, str]] = None
) -> PromptTemplate:
    """
    创建标准化Prompt模板
    :param template_str: Prompt模板字符串（含变量占位符，如{user_query}）
    :param input_variables: 模板变量列表（如["user_query", "item_reference"]）
    :param partial_variables: 部分预填充变量（可选，如{"system_prompt": "你是外卖推荐助手"}）
    :return: 标准化的PromptTemplate实例
    """
    try:
        prompt = PromptTemplate(
            template=template_str,
            input_variables=input_variables,
            partial_variables=partial_variables or {}
        )
        logger.debug(f"成功创建Prompt模板，变量：{input_variables}")
        return prompt
    except Exception as e:
        logger.error(f"创建Prompt模板失败：{str(e)}", exc_info=True)
        raise ValueError(f"Prompt模板构建异常：{str(e)}")

def format_prompt(
    prompt: PromptTemplate,
    input_data: Dict[str, Union[str, List[str]]]
) -> str:
    """
    格式化Prompt模板（将变量填充为实际值）
    :param prompt: PromptTemplate实例
    :param input_data: 变量字典（如{"user_query": "想吃炸鸡", "item_reference": "炸鸡,汉堡,薯条"}）
    :return: 格式化后的完整Prompt字符串
    """
    try:
        # 处理列表类型变量（转为逗号分隔字符串）
        formatted_data = {}
        for k, v in input_data.items():
            if isinstance(v, list):
                formatted_data[k] = ",".join(v)
            else:
                formatted_data[k] = str(v)

        prompt_str = prompt.format(**formatted_data)
        logger.debug(f"Prompt格式化完成，长度：{len(prompt_str)}字符")
        return prompt_str
    except KeyError as e:
        logger.error(f"Prompt格式化失败：缺失变量 {e}", exc_info=True)
        raise KeyError(f"Prompt模板需要变量 {e}，但输入数据中未提供")
    except Exception as e:
        logger.error(f"Prompt格式化异常：{str(e)}", exc_info=True)
        raise

# ===================== LLM调用工具 =====================
def call_llm(
    prompt: Union[PromptTemplate, str],
    input_data: Optional[Dict[str, Union[str, List[str]]]] = None,
    llm_type: str = "large",
    output_parser: Optional[StrOutputParser] = None
) -> str:
    """
    统一调用LLM模型（封装调用逻辑，简化业务层代码）
    :param prompt: PromptTemplate实例 或 直接传入格式化后的字符串
    :param input_data: 模板变量字典（仅当prompt为PromptTemplate时需要）
    :param llm_type: 模型类型（small/large）
    :param output_parser: 输出解析器（默认使用StrOutputParser）
    :return: 模型返回的字符串结果（已清洗）
    """
    # 1. 选择模型
    llm = get_llm_large() if llm_type == "large" else get_llm_small()
    parser = output_parser or StrOutputParser()

    # 2. 处理Prompt
    if isinstance(prompt, PromptTemplate):
        if not input_data:
            raise ValueError("当prompt为PromptTemplate时，必须提供input_data填充变量")
        prompt_str = format_prompt(prompt, input_data)
    else:
        prompt_str = prompt.strip()

    # 3. 调用模型
    try:
        logger.info(f"调用{llm_type}模型，Prompt长度：{len(prompt_str)}字符")
        # 构建调用链：Prompt → LLM → 解析器
        result = llm.invoke(prompt_str)
        cleaned_result = parser.invoke(result).strip().replace("\n", "").replace("  ", " ")

        logger.info(f"模型调用成功，结果长度：{len(cleaned_result)}字符")
        return cleaned_result
    except Exception as e:
        logger.error(f"{llm_type}模型调用失败：{str(e)}", exc_info=True)
        raise RuntimeError(f"LLM调用异常：{str(e)}")

def call_llm_with_retry(
    prompt: Union[PromptTemplate, str],
    input_data: Optional[Dict[str, Union[str, List[str]]]] = None,
    llm_type: str = "large",
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> str:
    """
    带重试机制的LLM调用（兼容低版本LangChain，自定义重试逻辑）
    :param max_retries: 最大重试次数
    :param retry_delay: 重试间隔（秒）
    :return: 解析后的结果（重试失败则抛出异常）
    """
    # 1. 处理Prompt
    if isinstance(prompt, PromptTemplate):
        if not input_data:
            raise ValueError("当prompt为PromptTemplate时，必须提供input_data填充变量")
        prompt_str = format_prompt(prompt, input_data)
    else:
        prompt_str = prompt.strip()

    # 2. 自定义重试逻辑
    last_error = None
    for retry in range(max_retries + 1):
        try:
            logger.info(f"带重试调用{llm_type}模型（第{retry+1}次，共{max_retries+1}次）")
            llm = get_llm_large() if llm_type == "large" else get_llm_small()
            parser = StrOutputParser()

            result = llm.invoke(prompt_str)
            cleaned_result = parser.invoke(result).strip().replace("\n", "").replace("  ", " ")

            logger.info(f"带重试模型调用成功，结果：{cleaned_result[:50]}...")
            return cleaned_result
        except Exception as e:
            last_error = e
            logger.warning(f"第{retry+1}次调用失败：{str(e)}")
            if retry < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避

    # 所有重试失败
    logger.error(f"调用{llm_type}模型失败（已重试{max_retries}次）：{str(last_error)}", exc_info=True)
    raise RuntimeError(f"LLM调用失败（已重试{max_retries}次）：{str(last_error)}")

# ===================== 便捷工具函数 =====================
def extract_core_items_prompt() -> PromptTemplate:
    """快速获取「核心商品标签提取」的标准化Prompt模板（复用原有业务逻辑）"""
    template = """任务：根据用户需求，从【可选实物】中选3个最相关的，仅返回关键词，用逗号分隔。
用户需求：{user_query}
可选实物：{item_reference}
示例：用户说"想吃辣的"，返回"水煮鱼,剁椒鱼头,麻辣烫"；用户说"想喝冰的"，返回"冰淇淋,柠檬水,星冰乐"。
输出：仅3个关键词，用逗号分隔，无其他内容！"""
    return build_prompt(
        template_str=template,
        input_variables=["user_query", "item_reference"]
    )

def passive_recommend_prompt() -> PromptTemplate:
    """快速获取「被动推荐」的标准化Prompt模板（用户有明确查询）"""
    template = """严格按以下规则推荐外卖，禁止推荐无关商户：
    
1. 仅推荐实物标签含{demand_items}中任意一个的商户，无匹配则直接提示暂无；
2. 格式：每行为“商户名 - 推荐商品：[商品] | 配送：[信息] | 优惠：[信息] ”，最多3个，按评分降序；
3. 必须严格使用商户信息中的「配送（原始数据）」和「优惠（原始数据）」，**不得修改数字、文字或编造内容**；
4. 不额外解释，所有内容必须来自提供的商户信息。

商户信息：
{merchant_text}

用户需求：{demand_desc}
推荐结果："""
    return build_prompt(
        template_str=template,
        input_variables=["demand_items", "merchant_text", "demand_desc"]
    )

def active_recommend_prompt() -> PromptTemplate:
    """快速获取「主动推荐」的标准化Prompt模板（用户无明确查询）"""
    template = """基于你的历史购买偏好、浏览记录和当前天气，为你推荐合适的外卖：
1. 优先推荐你常买/浏览的品类，兼顾天气适配性；
2. 格式：每行为“商户名 - 推荐商品：[商品] | 配送：[信息] | 优惠：[信息] | 卖点（不超20字）”，最多3个，按评分降序；
3. 必须严格使用商户信息中的「配送（原始数据）」和「优惠（原始数据）」，**不得修改数字、文字或编造内容**；
4. 必须包含招牌商品、配送信息、优惠信息，卖点结合口味描述，不额外解释。

商户信息：
{merchant_text}

当前天气：{weather_desc}
推荐结果："""
    return build_prompt(
        template_str=template,
        input_variables=["merchant_text", "weather_desc"]
    )