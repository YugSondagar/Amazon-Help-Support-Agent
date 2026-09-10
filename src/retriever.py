"""
src/retriever.py
----------------
Purpose:
Retrieve top-k historical (customer message -> brand reply) pairs relevant to an incoming ticket
using a hybrid BM25 (lexical) + Sentence Transformers (dense embedding) similarity algorithm.

Why this file exists:
Grounding support agent replies in historical brand data prevents hallucination and ensures
compliance with brand-specific support policies (e.g. asking for DM order #).
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Default model for fast lightweight embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "amazon_help_cleaned.csv")


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword matching with dense sentence transformer vector search.
    """

    def __init__(self, data_path: str = DATA_PATH):
        """
        Loads clean dataset, tokenizes text for BM25, and computes dense vector embeddings.
        """
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Clean dataset not found at {self.data_path}. Run data/download_and_sample.py first.")

        self.df = pd.read_csv(self.data_path)
        self.corpus_messages = self.df['customer_message'].fillna('').tolist()

        # 1. Initialize BM25 Sparse Index
        self.tokenized_corpus = [msg.lower().split() for msg in self.corpus_messages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Initialize Dense Vector Index
        self.dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.corpus_embeddings = self.dense_model.encode(
            self.corpus_messages,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 3, bm25_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical pairs for a query using hybrid scoring.

        Hybrid Score = (bm25_weight * BM25_normalized) + ((1 - bm25_weight) * Cosine_Similarity)

        Args:
            query: The incoming customer message.
            top_k: Number of historical context pairs to return.
            bm25_weight: Weight allocation between BM25 (0.0 to 1.0) and Dense embedding.

        Returns:
            List of dicts containing customer_message, brand_reply, similarity_score, and intent.
        """
        if not query or not query.strip():
            return []

        query_clean = query.strip()

        # 1. Calculate BM25 Scores
        query_tokens = query_clean.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1.0
        bm25_scores_norm = bm25_scores / max_bm25

        # 2. Calculate Dense Cosine Similarity
        query_embedding = self.dense_model.encode([query_clean], convert_to_numpy=True, normalize_embeddings=True)[0]
        dense_scores = np.dot(self.corpus_embeddings, query_embedding)
        # Scale cosine similarity from [-1, 1] to [0, 1]
        dense_scores_norm = np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0)

        # 3. Hybrid Combination Score
        hybrid_scores = (bm25_weight * bm25_scores_norm) + ((1.0 - bm25_weight) * dense_scores_norm)

        # Get top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
                "intent": row.get("ground_truth_intent", "unknown"),
                "similarity_score": round(float(hybrid_scores[idx]), 4),
                "dense_score": round(float(dense_scores_norm[idx]), 4),
                "bm25_score": round(float(bm25_scores_norm[idx]), 4)
            })

        return results
