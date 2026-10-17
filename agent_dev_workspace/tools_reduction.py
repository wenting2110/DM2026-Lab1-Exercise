from typing import Dict, Any, Optional, Literal, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensions and plots 2D scatter plot colored by category.

        Supports PCA, t-SNE, and UMAP reduction methods. Always creates a 2D scatter plot
        colored by category and sets it on session.pending_figure.

        Args:
            method: Reduction algorithm ("pca", "tsne", or "umap", default "pca").
            n_components: Number of output dimensions (default 2).
            random_state: Seed for reproducible random output (default 42).
            perplexity: t-SNE perplexity parameter (default 30.0, override for small datasets).
            n_neighbors: UMAP neighbor count parameter (default 15, override for small datasets).

        Returns:
            Dict containing result_id, method, n_components, shape, and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            raise ValueError("No feature_matrix built in session. Run build_dtm_tool first.")

        X_dense = session.feature_matrix.toarray()
        n_samples = X_dense.shape[0]

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            # Adjust perplexity if larger than n_samples - 1
            eff_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=n_components,
                perplexity=eff_perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)
        elif method == "umap":
            # Adjust n_neighbors if larger than n_samples
            eff_neighbors = min(n_neighbors, max(2, n_samples))
            reducer = umap.UMAP(n_components=n_components, n_neighbors=eff_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
        else:
            raise ValueError(f"Unsupported reduction method '{method}'. Choose 'pca', 'tsne', or 'umap'.")

        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list
        if explained_variance_ratio is not None:
            session.artifacts["explained_variance_ratio"] = explained_variance_ratio

        # Determine document labels for plot coloring
        labels = session.labels if session.labels is not None else ["Uncategorized"] * n_samples

        # Mandatory Plotting Requirement: Scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        df_plot = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else [0] * n_samples,
            "category": labels
        })

        sns.scatterplot(data=df_plot, x="Dim 1", y="Dim 2", hue="category", style="category", s=80, ax=ax)
        ax.set_title(f"{method.upper()} Dimensionality Reduction (2D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        # Set pending_figure as required by spec contract
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "shape": list(coords.shape),
            "preview_coordinates": [[round(float(v), 4) for v in row] for row in coords[:3]],
            "explained_variance_ratio": explained_variance_ratio
        }

        # Store full report
        df_full = pd.DataFrame(coords, columns=[f"dim_{i+1}" for i in range(n_components)])
        df_full["category"] = labels

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary=summary,
            full_report=df_full
        )

        return summary

    @tool
    def binarize_labels_tool(column_name: str = "category_name") -> Dict[str, Any]:
        """Binarizes/one-hot encodes category labels using sklearn LabelBinarizer.

        Fits LabelBinarizer on the specified category column, saves the fitted transformer in
        session.artifacts['label_binarizer'], and stores the binary matrix in session.artifacts['onehot_labels'].

        Args:
            column_name: Name of the column containing category labels (default "category_name").

        Returns:
            Dict containing result_id, column_name, n_samples, n_classes, classes, shape, and class_mapping.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        target_col = column_name
        if target_col not in session.dataframe.columns:
            if "category" in session.dataframe.columns:
                target_col = "category"
            else:
                raise ValueError(f"Column '{column_name}' not found in session.dataframe.")

        labels = session.dataframe[target_col].astype(str).tolist()

        lb = LabelBinarizer()
        onehot_matrix = lb.fit_transform(labels)
        classes = lb.classes_.tolist()

        # Store fitted LabelBinarizer and binarized matrix in session artifacts
        session.artifacts["label_binarizer"] = lb
        session.artifacts["onehot_labels"] = onehot_matrix.tolist()

        class_mapping = {cls: i for i, cls in enumerate(classes)}

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "column_name": target_col,
            "n_samples": len(labels),
            "n_classes": len(classes),
            "classes": classes,
            "shape": list(onehot_matrix.shape),
            "class_mapping": class_mapping
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={"column_name": column_name},
            summary=summary
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
