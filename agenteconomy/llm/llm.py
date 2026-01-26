from dotenv import load_dotenv
load_dotenv()

from litellm import Router
import os
import asyncio
from typing import Literal
import warnings

# 简单粗暴屏蔽所有 Pydantic 序列化警告
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
# 配置模型列表
model_list = [
    {
        "model_name": "simple",  # 简单推理模型
        "litellm_params": {
            "model": os.getenv("SIMPLE_MODEL", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("BASE_URL"),
        }
    },
    {
        "model_name": "strong",  # 强推理模型
        "litellm_params": {
            "model": os.getenv("STRONG_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("BASE_URL"),
        }
    }
]

# 初始化 Router
router = Router(model_list=model_list)


async def call_llm_simple(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    调用简单推理模型（适用于简单任务、快速响应）
    """
    response = await router.acompletion(
        model="simple",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


async def call_llm_strong(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    调用强推理模型（适用于复杂任务、需要深度推理）
    """
    response = await router.acompletion(
        model="strong",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


async def call_llm(
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.",
    model_type: Literal["simple", "strong"] = "simple"
) -> str:
    """
    统一的 LLM 调用接口，供所有实体和市场使用
    
    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词
        model_type: 模型类型，"simple" 为简单推理，"strong" 为强推理
    
    Returns:
        模型的回答内容
    """
    response = await router.acompletion(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


async def main():
    """示例：测试两种模型的调用"""
    # 测试简单推理模型
    print("=== 简单推理模型 ===")
    response_simple = await call_llm_simple("What is 2 + 2?")
    print(response_simple)
    
    # 测试强推理模型
    print("\n=== 强推理模型 ===")
    response_strong = await call_llm_strong("Explain the theory of relativity in simple terms.")
    print(response_strong)
    
    # 测试统一接口
    print("\n=== 统一接口调用 ===")
    response_unified = await call_llm("What is the capital of France?", model_type="simple")
    print(response_unified)


if __name__ == "__main__":
    asyncio.run(main())
