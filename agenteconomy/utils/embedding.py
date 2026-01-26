from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))
model = AutoModel.from_pretrained(os.getenv("MODEL_PATH")).to(device)

def embedding(text: str) -> np.ndarray:
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

    # Normalize the output
    normalized_embedding = F.normalize(pooled_output, p=2, dim=0)
    
    return normalized_embedding.cpu().numpy().tolist()

# mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
