from sentence_transformers import SentenceTransformer
import numpy as np

def extract_text_embedding(text: str) -> np.ndarray:
    """
    Extracts the text embedding of the given text.
    """
    lm_model = 'sentence-transformers/stsb-roberta-base-v2'
    lm = SentenceTransformer(lm_model, device='cuda').eval()
    
    return lm.encode(text)