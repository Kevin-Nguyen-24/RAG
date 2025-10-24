# rag/retriever_embed.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class EmbeddingRetriever:
    def __init__(self, csv_path, model_dir):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer(model_dir)  # local path only
        corpus = (self.df["brand"].astype(str)+" "+self.df["model"].astype(str)+" "+
                  self.df.fillna("").astype(str).agg(" ".join, axis=1))
        self.embs = self.model.encode(list(corpus), normalize_embeddings=True)
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.embs)

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)
        dists, idx = self.nn.kneighbors(q, n_neighbors=k)
        hits = self.df.iloc[idx[0]].copy()
        hits["_score"] = 1 - dists[0]
        return hits
