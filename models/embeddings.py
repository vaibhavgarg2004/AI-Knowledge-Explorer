from sentence_transformers import SentenceTransformer
import numpy as np

# models/embeddings.py.
class EmbeddingModel:
    """Wrapper around a sentence-transformers model to produce vectors."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """Return list of vectors for the provided texts."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # ensure numpy arrays
        return [np.array(vec).astype(float) for vec in embeddings]
