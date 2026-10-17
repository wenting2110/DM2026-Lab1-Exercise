from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal from session.feature_matrix.

        Calculates per-term variance across all documents. Returns kept and removed term lists
        and stores a full report DataFrame sorted descending by variance.

        Args:
            threshold: Minimum variance required for a term to be kept (default 0.15).

        Returns:
            Dict containing threshold, total_terms, kept_count, removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        # Compute per-term population variance (ddof=0) across axis 0
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)
        feature_names = session.feature_names

        # Build full report DataFrame sorted descending by variance
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        # Sort removed terms ascending by variance so lowest variance terms appear first
        removed_terms_asc = removed_df.sort_values(by="variance", ascending=True)["term"].tolist()

        # Truncate preview in summary to max 20 items to prevent LLM prompt bloat on large datasets
        summary_kept = kept_terms[:20]
        summary_removed = removed_terms_asc[:20]

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms_asc),
            "kept_terms": summary_kept,
            "removed_terms": summary_removed
        }

        # Store full report for visualize_result_tool (first col term, second col variance, sorted)
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculates Pearson correlation between term counts and a target class label.

        Computes Pearson correlation (pearson_r) for each term against a binary target class indicator.
        Filters terms by absolute correlation threshold (|r| >= threshold).

        Args:
            target_class: Target category label to correlate term counts against.
            threshold: Minimum absolute Pearson correlation required to keep a term (default 0.5).

        Returns:
            Dict containing target_class, threshold, total_terms, kept_count, removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        if session.labels is None:
            raise ValueError("No labels found in session.")

        y_binary = (np.array(session.labels) == target_class).astype(float)
        if np.std(y_binary) == 0:
            raise ValueError(f"Target class '{target_class}' not found or present in all documents.")

        X_dense = session.feature_matrix.toarray()
        feature_names = session.feature_names

        pearson_r_list = []
        std_y = np.std(y_binary)

        for j in range(X_dense.shape[1]):
            col = X_dense[:, j]
            std_col = np.std(col)
            if std_col == 0:
                r_val = 0.0
            else:
                cov = np.cov(col, y_binary, bias=True)[0, 1]
                r_val = float(cov / (std_col * std_y))
            pearson_r_list.append(round(r_val, 6))

        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_list
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["pearson_r"].abs() >= threshold
        kept_df = df_report[kept_mask]
        removed_df = df_report[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20]
        }

        # Store full report DataFrame (first col term, second col pearson_r)
        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between term counts and a target class label.

        Computes Spearman correlation (spearman_r) for each term against a binary target class indicator.
        Filters terms by absolute correlation threshold (|rho| >= threshold).

        Args:
            target_class: Target category label to correlate term counts against.
            threshold: Minimum absolute Spearman correlation required to keep a term (default 0.5).

        Returns:
            Dict containing target_class, threshold, total_terms, kept_count, removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        if session.labels is None:
            raise ValueError("No labels found in session.")

        y_binary = (np.array(session.labels) == target_class).astype(float)
        if np.std(y_binary) == 0:
            raise ValueError(f"Target class '{target_class}' not found or present in all documents.")

        X_dense = session.feature_matrix.toarray()
        feature_names = session.feature_names

        spearman_r_list = []

        for j in range(X_dense.shape[1]):
            col = X_dense[:, j]
            if np.std(col) == 0:
                rho_val = 0.0
            else:
                res = spearmanr(col, y_binary)
                stat = getattr(res, 'statistic', res[0])
                rho_val = float(stat) if not np.isnan(stat) else 0.0
            spearman_r_list.append(round(rho_val, 6))

        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r_list
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["spearman_r"].abs() >= threshold
        kept_df = df_report[kept_mask]
        removed_df = df_report[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20]
        }

        # Store full report DataFrame (first col term, second col spearman_r)
        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
