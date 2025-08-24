from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from ..config import EMB_MODEL
from ..logging import get_logger

logger = get_logger(__name__)

class Embedder:
    def __init__(self, model_name: str = EMB_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d)", self.dim)
        self.normalize = True

    def encode(self, texts: List[str]) -> List[List[float]]:
        v = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
