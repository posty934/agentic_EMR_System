import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def create_llm(temperature: float = 0.0) -> ChatOpenAI:
    timeout_s = float(os.getenv("LLM_TIMEOUT", "90"))
    connect_timeout_s = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))

    timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)

    # 核心：trust_env=False，不继承系统代理环境变量
    http_client = httpx.Client(
        trust_env=False,
        timeout=timeout,
    )
    http_async_client = httpx.AsyncClient(
        trust_env=False,
        timeout=timeout,
    )

    return ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=temperature,
        max_retries=max_retries,
        http_client=http_client,
        http_async_client=http_async_client,
    )