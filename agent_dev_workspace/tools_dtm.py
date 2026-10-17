from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_df: float = 1.0,
        min_df: int = 1,
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from session.dataframe text.

        Vectorizes text using CountVectorizer, stores the feature matrix and vocabulary
        in session.feature_matrix and session.feature_names, saves the fitted vectorizer
        in session.artifacts['count_vectorizer'], and reports matrix sparsity.

        Args:
            max_df: Maximum document frequency threshold (float for proportion, int for count).
            min_df: Minimum document frequency threshold (float for proportion, int for count).
            max_features: Maximum number of features/terms to build.
            ngram_range: Lower and upper boundary of the range of n-gram values, e.g., [1, 1].

        Returns:
            Dict containing summary of the built DTM including dimensions and sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataframe loaded in session or 'text' column is missing.")

        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert List[int] to tuple to satisfy CountVectorizer while maintaining schema safety
        ngram_tuple = tuple(ngram_range)

        vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_tuple
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Store in session attributes as required by spec
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "max_df": max_df,
                "min_df": min_df,
                "max_features": max_features,
                "ngram_range": ngram_range
            },
            summary=summary
        )

        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums term counts across all documents and returns top_n terms sorted descending by frequency.

        Args:
            top_n: Number of top terms to return in summary (default 20, None for all terms).

        Returns:
            Dict containing result_id, total_terms, and top term frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        # Compute column sums across DTM
        term_counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        feature_names = session.feature_names

        df_freq = pd.DataFrame({
            "term": feature_names,
            "frequency": term_counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        if top_n is not None and top_n > 0:
            df_summary = df_freq.head(top_n)
        else:
            df_summary = df_freq

        term_frequencies = dict(zip(df_summary["term"], df_summary["frequency"].astype(int)))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(feature_names),
            "top_n": top_n,
            "term_frequencies": term_frequencies
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=df_freq
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a raw count heatmap of a slice of session.feature_matrix.

        Slices top n_documents and n_terms in positional order, plots a seaborn heatmap,
        and sets session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to slice (default 20).
            n_documents: Number of documents (rows) to slice (default 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, included_terms, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        total_docs, total_terms = session.feature_matrix.shape
        eff_docs = min(n_documents, total_docs)
        eff_terms = min(n_terms, total_terms)

        dense_slice = session.feature_matrix[0:eff_docs, 0:eff_terms].toarray()
        included_terms = session.feature_names[0:eff_terms]
        doc_labels = [f"Doc {i}" for i in range(eff_docs)]

        # Generate Heatmap Plot
        fig, ax = plt.subplots(figsize=(max(6, eff_terms * 0.8), max(4, eff_docs * 0.4)))
        sns.heatmap(dense_slice, annot=True, fmt="d", xticklabels=included_terms, yticklabels=doc_labels, ax=ax)
        ax.set_title("Document-Term Matrix Raw Count Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending_figure attribute as required by plotting spec
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": eff_docs,
            "n_terms": eff_terms,
            "included_terms": included_terms,
            "matrix_slice": dense_slice.tolist()
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
