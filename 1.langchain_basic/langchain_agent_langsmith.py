from langchain.agents import create_agent
# 상위폴더 path 설정 (실행 위치와 무관하게 이 파일 기준으로 계산)
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(override=True, dotenv_path=ROOT_DIR / ".env")

from common_config import llm_connect
llm = llm_connect("gpt-5.4-mini")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)

print(result)