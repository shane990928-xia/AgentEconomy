from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
import os

def load_client():
    qdrant_mode = os.getenv("QDRANT_MODE")
    if qdrant_mode == "local":
        qdrant_client = QdrantClient(
            path=os.getenv("QDRANT_PATH")
        )
    elif qdrant_mode == "cloud":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    elif qdrant_mode == "docker":
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_DOCKER_URL")
        )
    else:
        raise ValueError(f"Invalid QDRANT_MODE: {qdrant_mode}. Please set QDRANT_MODE to 'local', 'cloud', or 'docker'.")
    return qdrant_client