from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))
model = AutoModel.from_pretrained(os.getenv("MODEL_PATH")).to(device)

def embedding(text: str):
    """
    Generate an embedding for the given text using the specified tokenizer and model.
    
    Args:
        text (str): The input text to be embedded.
    
    Returns:
        np.ndarray: The normalized embedding vector.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    pooled_output = mean_pooling(outputs, inputs['attention_mask']).squeeze(0)
    pooled_output = torch.nan_to_num(pooled_output.float(), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extreme values before normalization to prevent overflow
    pooled_output = torch.clamp(pooled_output, min=-1e6, max=1e6)

    # Normalize the output
    normalized_embedding = F.normalize(pooled_output, p=2, dim=0)
    vec = normalized_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    
    # Additional safety: handle any remaining non-finite values
    if not np.isfinite(vec).all():
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip the final vector to prevent overflow in downstream operations
    vec = np.clip(vec, -1.0, 1.0)
    
    # Final normalization with epsilon for numerical stability
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    else:
        # If norm is too small, return a zero vector (will be handled by Qdrant)
        vec = np.zeros_like(vec)
    
    return vec.tolist()

# mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
