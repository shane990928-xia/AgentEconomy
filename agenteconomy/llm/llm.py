from dotenv import load_dotenv
load_dotenv()

from litellm import Router
import os
import asyncio
from pathlib import Path
from typing import Literal
import warnings

from agenteconomy.llm.agent_method import get_agent_method, normalize_agent_name

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

# LLM 并发限制信号量
# 优先使用环境变量，否则使用较高的默认值（400）
# 可以通过设置 LLM_MAX_CONCURRENCY 环境变量覆盖
_default_concurrency = 400  # 与 config.py 中的 max_llm_concurrent 默认值一致
_llm_max_concurrency = int(os.getenv("LLM_MAX_CONCURRENCY", str(_default_concurrency)))
_llm_semaphore = asyncio.Semaphore(_llm_max_concurrency)
_llm_pending_count = 0
_llm_pending_lock = asyncio.Lock()

# 打印并发配置信息（仅在直接运行或首次导入时显示）
import logging
_llm_logger = logging.getLogger("llm")
_llm_logger.info(f"LLM max concurrency initialized: {_llm_max_concurrency}")

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def configure_concurrency(max_concurrent: int) -> None:
    """
    动态配置 LLM 并发限制
    
    可以在 Simulator 初始化时调用，使用 config 中的 max_llm_concurrent 值
    
    Args:
        max_concurrent: 最大并发数
    """
    global _llm_semaphore, _llm_max_concurrency
    if max_concurrent > 0 and max_concurrent != _llm_max_concurrency:
        _llm_max_concurrency = max_concurrent
        _llm_semaphore = asyncio.Semaphore(max_concurrent)
        _llm_logger.info(f"LLM concurrency reconfigured to: {max_concurrent}")


def get_current_concurrency() -> int:
    """获取当前配置的最大并发数"""
    return _llm_max_concurrency


async def call_llm_simple(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    调用简单推理模型（适用于简单任务、快速响应）
    """
    return await call_llm(prompt, system_prompt=system_prompt, model_type="simple")


async def call_llm_strong(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    调用强推理模型（适用于复杂任务、需要深度推理）
    """
    return await call_llm(prompt, system_prompt=system_prompt, model_type="strong")


def _read_configured_agent_name() -> str | None:
    try:
        config_text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line != raw_line.lstrip():
            continue
        if not line.startswith("agent_name:"):
            continue
        agent_name = line.split(":", 1)[1].strip().strip("'\"")
        if agent_name.lower() in {"", "none", "null", "~"}:
            return None
        return normalize_agent_name(agent_name)
    return None


async def _call_direct_llm_raw(
    prompt: str,
    system_prompt: str,
    model_type: Literal["simple", "strong"],
) -> str:
    response = await router.acompletion(
        model=model_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


async def _call_agent_method(
    prompt: str,
    system_prompt: str,
    model_type: Literal["simple", "strong"],
    agent_name: str,
) -> str:
    agent_method = get_agent_method(agent_name)
    if agent_method is None:
        raise ValueError(f"Unsupported agent method: {agent_name}")
    return await agent_method(
        call_model=_call_direct_llm_raw,
        prompt=prompt,
        system_prompt=system_prompt,
        model_type=model_type,
    )


async def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model_type: Literal["simple", "strong"] = "simple",
    timeout: float = 180.0
) -> str:
    """
    统一的 LLM 调用接口，供所有实体和市场使用

    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词
        model_type: 模型类型，"simple" 为简单推理，"strong" 为强推理
        timeout: API调用超时时间（秒），不包括等待信号量的时间

    Returns:
        模型的回答内容
    """
    async with _llm_semaphore:
        # 超时只计算实际API调用时间，不包括等待信号量的时间
        configured_agent = _read_configured_agent_name()
        if get_agent_method(configured_agent) is None:
            return await asyncio.wait_for(
                _call_direct_llm_raw(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_type=model_type,
                ),
                timeout=timeout
            )

        return await asyncio.wait_for(
            _call_agent_method(
                prompt=prompt,
                system_prompt=system_prompt,
                model_type=model_type,
                agent_name=configured_agent,
            ),
            timeout=timeout
        )


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
