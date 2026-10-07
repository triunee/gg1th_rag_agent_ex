from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
# print(api_key)

DEFAULT_LLM_MODEL = "gpt-5.4-mini"


def llm_connect(
    model: str = DEFAULT_LLM_MODEL,
    api_key: str = api_key,
    temperature: float = 0,
    max_tokens: int = 512
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )


def embedding_connect(
    model: str = "text-embedding-3-small",
    api_key: str = api_key,
):
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,  # MonoRouter 사용
    )


# 3.advanced_rag_ai_trend 노트북에서 사용하는 이름
def embedding_model():
    return embedding_connect()
