from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc1_index: Optional[int] = None,
        doc2_index: Optional[int] = None,
        doc1_text: Optional[str] = None,
        doc2_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents.

        Supports existing documents by index (from session.feature_matrix) or raw text
        vectorized using session.artifacts['count_vectorizer'].

        Args:
            doc1_index: Row index of first document in session.feature_matrix.
            doc2_index: Row index of second document in session.feature_matrix.
            doc1_text: Raw text string for first document.
            doc2_text: Raw text string for second document.

        Returns:
            Dict containing result_id and similarity_score.
        """
        def _get_vector(idx: Optional[int], txt: Optional[str], label: str):
            if idx is not None and txt is not None:
                raise ValueError(f"Provide either {label}_index or {label}_text, not both.")
            if idx is None and txt is None:
                raise ValueError(f"Must provide either {label}_index or {label}_text.")

            if idx is not None:
                if session.feature_matrix is None:
                    raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")
                if idx < 0 or idx >= session.feature_matrix.shape[0]:
                    raise ValueError(f"{label}_index {idx} is out of range.")
                return session.feature_matrix[idx]
            else:
                vectorizer = session.artifacts.get("count_vectorizer")
                if vectorizer is None:
                    raise ValueError("No fitted 'count_vectorizer' found in session.artifacts. Run build_dtm_tool first.")
                return vectorizer.transform([txt])

        vec1 = _get_vector(doc1_index, doc1_text, "doc1")
        vec2 = _get_vector(doc2_index, doc2_text, "doc2")

        # Handle zero vectors safely
        if vec1.nnz == 0 or vec2.nnz == 0:
            sim_score = 0.0
        else:
            sim_score = float(cosine_similarity(vec1, vec2)[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc1_index": doc1_index,
            "doc2_index": doc2_index,
            "doc1_text_provided": doc1_text is not None,
            "doc2_text_provided": doc2_text is not None,
            "similarity_score": round(sim_score, 4)
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={
                "doc1_index": doc1_index,
                "doc2_index": doc2_index,
                "doc1_text": doc1_text,
                "doc2_text": doc2_text
            },
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_k_variance: int = 20) -> Dict[str, Any]:
        """Calculates Pearson correlation matrix across top-variance terms and plots heatmap.

        Selects top_k_variance terms by variance from session.feature_matrix, computes Pearson
        correlation matrix, and sets a seaborn heatmap on session.pending_figure.

        Args:
            top_k_variance: Number of highest-variance terms to analyze (default 20).

        Returns:
            Dict containing result_id, n_terms, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        X_dense = session.feature_matrix.toarray()
        feature_names = session.feature_names
        n_features = X_dense.shape[1]

        if n_features == 0:
            raise ValueError("Feature matrix has 0 terms.")

        # Compute per-term variance across all documents
        variances = np.var(X_dense, axis=0, ddof=0)

        # Sort indices by variance descending
        sorted_indices = np.argsort(variances)[::-1]
        top_indices = sorted_indices[:min(top_k_variance, n_features)]

        selected_terms = [feature_names[i] for i in top_indices]
        X_selected = X_dense[:, top_indices]

        # Compute Pearson correlation matrix across selected term columns
        corr_matrix = np.corrcoef(X_selected, rowvar=False)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

        # Generate Seaborn Correlation Heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(selected_terms) * 0.8), max(5, len(selected_terms) * 0.8)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax
        )
        ax.set_title("Top-Variance Feature Correlation Matrix")
        plt.tight_layout()

        # Set pending_figure as required by spec contract
        session.pending_figure = fig

        corr_list = [[round(float(v), 4) for v in row] for row in corr_matrix]

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "n_terms": len(selected_terms),
            "terms": selected_terms,
            "correlation_matrix": corr_list
        }

        # Store full report
        df_full = pd.DataFrame(corr_list, index=selected_terms, columns=selected_terms)

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_k_variance": top_k_variance},
            summary=summary,
            full_report=df_full
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
