#!/usr/bin/env python
"""Build or refresh the product embedding collection in Qdrant."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer

from agenteconomy.center.ProductMarket import (
    get_retailer_from_manufacturer,
    manufacturer_display_name,
    normalize_manufacturer_code,
)
from agenteconomy.utils.load_qdrant_client import load_client


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTS_CSV = REPO_ROOT / "agenteconomy" / "data" / "products_with_supply_chain_prices.csv"


def build_product_embedding_text(row: Dict[str, Any]) -> str:
    """Construct the searchable text used for product semantic matching."""
    fields = [
        row.get("Product Name"),
        row.get("Brand"),
        row.get("Category"),
        row.get("Industry_fixed") or row.get("Industry"),
        row.get("Description"),
    ]
    return " | ".join(str(value).strip() for value in fields if _has_value(value))


def stable_point_id(product_id: str) -> int:
    """Convert product IDs to stable unsigned integer point IDs for Qdrant."""
    digest = hashlib.sha1(str(product_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    product_id = str(row.get("Uniq Id") or "").strip()
    raw_industry = str(row.get("Industry_fixed") or "").strip()
    industry = normalize_manufacturer_code(raw_industry) or raw_industry
    industry_name = manufacturer_display_name(industry) or raw_industry
    retailer = str(row.get("Retailer_Code") or "").strip() or get_retailer_from_manufacturer(industry)
    return {
        "product_id": product_id,
        "name": str(row.get("Product Name") or "").strip(),
        "category": str(row.get("Category") or "").strip(),
        "industry": industry,
        "industry_name": industry_name,
        "retailer_code": retailer,
        "is_active": True,
    }


def embed_texts(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> Iterable[List[float]]:
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.clamp(pooled, min=-1e6, max=1e6)
        normalized = F.normalize(pooled, p=2, dim=1)
        for vector in normalized.detach().cpu().tolist():
            yield [float(value) for value in vector]


def recreate_collection(client: Any, collection_name: str, vector_size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def index_products(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file)

    model_path = args.model_path or os.getenv("MODEL_PATH")
    if not model_path:
        raise SystemExit("MODEL_PATH is required. Set it in .env or pass --model-path.")

    csv_path = Path(args.products_csv)
    collection_name = args.collection_name or os.getenv("QDRANT_COLLECTION_NAME", "products")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    probe = next(
        embed_texts(
            ["probe"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=1,
        )
    )
    vector_size = len(probe)

    client = load_client()
    recreate_collection(client, collection_name, vector_size, args.recreate)

    total = 0
    rows_seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=args.chunk_size):
        chunk_start = rows_seen
        rows_seen += len(chunk)
        if rows_seen <= args.start_row:
            continue
        if args.start_row > chunk_start:
            chunk = chunk.iloc[args.start_row - chunk_start :]
        rows = chunk.to_dict(orient="records")
        texts = [build_product_embedding_text(row) for row in rows]
        vectors = list(
            embed_texts(
                texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.embedding_batch_size,
            )
        )
        points = []
        for row, vector in zip(rows, vectors):
            product_id = str(row.get("Uniq Id") or "").strip()
            if not product_id:
                continue
            points.append(
                PointStruct(
                    id=stable_point_id(product_id),
                    vector=vector,
                    payload=build_payload(row),
                )
            )
        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            total += len(points)
            print(f"indexed={args.start_row + total}", flush=True)

    close = getattr(client, "close", None)
    if callable(close):
        close()
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--products-csv", default=str(DEFAULT_PRODUCTS_CSV))
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    parser.add_argument("--start-row", type=int, default=0, help="Skip CSV rows before this zero-based row offset")
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except TypeError:
        return True


if __name__ == "__main__":
    count = index_products(parse_args())
    print(f"completed_index_count={count}")
