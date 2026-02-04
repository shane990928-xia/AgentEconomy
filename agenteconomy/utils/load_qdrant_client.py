from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
import os

def load_client():
    """
    加载 Qdrant 客户端
    
    支持三种模式：
    - local: 本地文件存储
    - cloud: Qdrant Cloud（带超时和重试配置）
    - docker: 本地 Docker 部署
    """
    qdrant_mode = os.getenv("QDRANT_MODE")
    
    # 从环境变量获取超时配置（秒），默认值适合云端
    timeout = int(os.getenv("QDRANT_TIMEOUT", "30"))
    
    if qdrant_mode == "local":
        qdrant_client = QdrantClient(
            path=os.getenv("QDRANT_PATH")
        )
    elif qdrant_mode == "cloud":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=timeout,  # 连接和读取超时
            # gRPC 配置（如果使用 gRPC）
            prefer_grpc=False,  # HTTP 更稳定
        )
    elif qdrant_mode == "docker":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_DOCKER_URL"),
            timeout=timeout,
        )
    else:
        raise ValueError(f"Invalid QDRANT_MODE: {qdrant_mode}. Please set QDRANT_MODE to 'local', 'cloud', or 'docker'.")
    return qdrant_client