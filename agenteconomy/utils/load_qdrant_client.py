import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


DEFAULT_QDRANT_PATH = Path(__file__).resolve().parents[2] / "qdrant_database"

def load_client():
    """
    加载 Qdrant 客户端
    
    支持三种模式：
    - local: 本地文件存储
    - cloud: Qdrant Cloud（带超时配置）
    - docker: 本地 Docker 部署
    
    注意：实际查询并发由 ProductMarket._qdrant_semaphore 控制，
    这里只负责超时配置。超时默认 60s（原来 30s 在高并发下不够用）。
    """
    qdrant_mode = os.getenv("QDRANT_MODE", "local")
    
    # 超时配置（秒）：高并发场景下需要更大的超时
    # 因为请求会在信号量处排队，实际等待时间可能较长
    timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))
    
    if qdrant_mode == "local":
        qdrant_client = QdrantClient(
            path=os.getenv("QDRANT_PATH", str(DEFAULT_QDRANT_PATH))
        )
    elif qdrant_mode == "cloud":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=timeout,
            prefer_grpc=False,
        )
    elif qdrant_mode == "docker":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_DOCKER_URL"),
            timeout=timeout,
        )
    else:
        raise ValueError(f"Invalid QDRANT_MODE: {qdrant_mode}. Please set QDRANT_MODE to 'local', 'cloud', or 'docker'.")
    return qdrant_client
