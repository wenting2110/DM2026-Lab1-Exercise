import os
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    @tool
    def check_missing_tool(column_name: str = "text") -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe.

        Identifies null, NaN, or whitespace-only entries in the specified column,
        reporting the missing count, percentage, and row indices.

        Args:
            column_name: Name of the column to inspect for missing/empty values (default "text").

        Returns:
            Dict containing result_id, total_rows, missing_count, missing_pct, and missing_indices.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")
        
        if column_name not in session.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in session.dataframe.")

        df = session.dataframe
        total_rows = len(df)

        if total_rows == 0:
            missing_indices = []
        else:
            col_series = df[column_name]
            # Missing if null/NaN or if string after stripping is empty
            is_missing_mask = col_series.isna() | (col_series.astype(str).str.strip() == "")
            missing_indices = df.index[is_missing_mask].tolist()

        missing_count = len(missing_indices)
        missing_pct = float(100.0 * missing_count / total_rows) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "column_name": column_name,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"column_name": column_name},
            summary=summary
        )

        return summary

    @tool
    def tokenize_tool(column_name: str = "text", output_key: str = "unigrams") -> Dict[str, Any]:
        """Tokenizes document text into unigrams using nltk.word_tokenize.

        Lowercases text and splits each document into unigrams. Stores tokenized lists
        in session.artifacts[output_key] and updates session.dataframe[output_key].

        Args:
            column_name: Column in session.dataframe containing text to tokenize.
            output_key: Key under which tokenized unigrams are stored in session.artifacts.

        Returns:
            Dict containing summary of tokenization (n_documents, total_tokens, avg_tokens_per_doc, unique_unigrams).
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        if column_name not in session.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in session.dataframe.")

        # Ensure Punkt resource is available
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)

        tokenized_docs = []
        all_unigrams = set()
        total_tokens = 0

        for text in session.dataframe[column_name]:
            if text is None or (isinstance(text, float) and float('nan') == text) or str(text).strip() == "":
                tokens = []
            else:
                tokens = nltk.word_tokenize(str(text).lower())
            
            tokenized_docs.append(tokens)
            total_tokens += len(tokens)
            all_unigrams.update(tokens)

        session.artifacts[output_key] = tokenized_docs
        session.dataframe[output_key] = tokenized_docs

        n_documents = len(tokenized_docs)
        avg_tokens_per_doc = float(total_tokens / n_documents) if n_documents > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens_per_doc, 4),
            "unique_unigrams": len(all_unigrams),
            "artifact_key": output_key
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={"column_name": column_name, "output_key": output_key},
            summary=summary
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """Lists files and subdirectories within a given relative directory path.

        Excludes hidden files/folders and system cache directories. Guarded against path traversal and permission errors.

        Args:
            subdirectory: Relative directory path to inspect (default "" for current working directory).

        Returns:
            Dict containing subdirectories and files listed under the requested directory path.
        """
        base_dir = os.path.abspath(os.getcwd())
        target_path = os.path.abspath(os.path.join(base_dir, subdirectory))

        # Security check: prevent path traversal outside workspace root
        if os.path.commonpath([base_dir, target_path]) != base_dir:
            raise ValueError(f"Access denied: Path '{subdirectory}' attempts to escape workspace root.")

        if not os.path.exists(target_path):
            raise ValueError(f"Directory path '{subdirectory}' does not exist.")

        if not os.path.isdir(target_path):
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        try:
            entries = os.listdir(target_path)
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot read contents of directory '{subdirectory}'.")

        directories = []
        files = []

        for entry in entries:
            # Ignore hidden files/dirs and pycache
            if entry.startswith(".") or entry == "__pycache__":
                continue
            
            full_entry_path = os.path.join(target_path, entry)
            if os.path.isdir(full_entry_path):
                directories.append(entry)
            else:
                files.append(entry)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "n_directories": len(directories),
            "n_files": len(files)
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(subset: List[str] = ["text"], drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops them.

        When drop=True, drops every copy of duplicated rows (keep=False), matching Master's behavior.

        Args:
            subset: List of column names to check for duplication (default ["text"]).
            drop: If True, removes all copies of duplicated rows in-place (default False).

        Returns:
            Dict containing result_id, total_rows_before, duplicate_count, duplicate_indices, dropped, and rows_remaining.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        for col in subset:
            if col not in session.dataframe.columns:
                raise ValueError(f"Column '{col}' not found in session.dataframe.")

        df = session.dataframe
        total_rows_before = len(df)

        # Count extra duplicate rows
        duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())
        
        # Identify all row indices involved in duplication
        all_duplicate_mask = df.duplicated(subset=subset, keep=False)
        duplicate_indices = df.index[all_duplicate_mask].tolist()

        if drop and duplicate_count > 0:
            df.drop_duplicates(subset=subset, keep=False, inplace=True)

        rows_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "subset": subset,
            "total_rows_before": total_rows_before,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "rows_remaining": rows_remaining
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"subset": subset, "drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Subsamples session.dataframe using pandas DataFrame.sample.

        Updates session.dataframe and category labels with the sampled rows.

        Args:
            n: Number of items/rows to sample.
            frac: Fraction of items to sample (e.g. 0.5 for 50%).
            random_state: Seed for reproducible random sampling (default 42).

        Returns:
            Dict containing result_id, n, frac, random_state, total_rows_before, sampled_rows, and sampled_indices.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        if n is not None and n < 0:
            raise ValueError(f"Sample size 'n' must be non-negative, got {n}.")

        if frac is not None and frac < 0:
            raise ValueError(f"Sample fraction 'frac' must be non-negative, got {frac}.")

        if random_state is not None and random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {random_state}.")

        total_rows_before = len(session.dataframe)

        if n is None and frac is None:
            n = min(5, total_rows_before)

        sampled_df = session.dataframe.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session dataframe and category labels
        session.dataframe = sampled_df
        if "category_name" in sampled_df.columns:
            session.set_labels(sampled_df["category_name"].values)
        elif "category" in sampled_df.columns:
            session.set_labels(sampled_df["category"].values)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "total_rows_before": total_rows_before,
            "sampled_rows": len(sampled_df),
            "sampled_indices": sampled_indices
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "frac": frac, "random_state": random_state},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """Computes text length statistics and generates a box plot grouped by category.

        Computes character length describe() statistics overall and by category,
        and sets a seaborn boxplot figure on session.pending_figure.

        Args:
            text_column: Name of the column containing document text (default "text").
            category_column: Name of the column containing category labels (default "category_name").

        Returns:
            Dict containing overall_stats and per_category_stats summaries.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        df = session.dataframe.copy()
        # Compute character length
        df["text_length"] = df[text_column].fillna("").astype(str).apply(len)

        overall_stats = df["text_length"].describe().to_dict()

        per_category_stats = {}
        target_cat_col = category_column if category_column in df.columns else ("category" if "category" in df.columns else None)

        if target_cat_col is not None:
            for cat_val, group in df.groupby(target_cat_col):
                per_category_stats[str(cat_val)] = group["text_length"].describe().to_dict()

        # Build boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if target_cat_col is not None:
            sns.boxplot(data=df, x=target_cat_col, y="text_length", ax=ax)
            ax.set_xlabel(target_cat_col)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length")
        ax.set_title("Document Text Length Distribution")
        plt.tight_layout()

        # Assign figure to pending_figure as required by plotting spec
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={"text_column": text_column, "category_column": category_column},
            summary=summary
        )

        return summary

    @tool
    def inspect_data_tool(
        columns: Optional[List[str]] = None,
        n_rows: int = 10,
        max_char_len: int = 200
    ) -> Dict[str, Any]:
        """Peeks at top n_rows of session.dataframe for requested columns.

        Truncates long text fields to max_char_len characters to prevent prompt bloat.

        Args:
            columns: List of column names to inspect (default None for all columns).
            n_rows: Number of top rows to slice (default 10).
            max_char_len: Maximum character length per text cell before truncating (default 200).

        Returns:
            Dict containing result_id, total_rows, inspected_rows, columns, and data records.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        available_cols = list(session.dataframe.columns)
        if columns is None or len(columns) == 0:
            selected_cols = available_cols
        else:
            for col in columns:
                if col not in available_cols:
                    raise ValueError(f"Column '{col}' not found in session.dataframe. Available: {available_cols}")
            selected_cols = columns

        df_slice = session.dataframe[selected_cols].head(n_rows).copy()

        # Truncate text cells longer than max_char_len
        records = []
        for _, row in df_slice.iterrows():
            rec = {}
            for col in selected_cols:
                val = row[col]
                if isinstance(val, str) and len(val) > max_char_len:
                    rec[col] = val[:max_char_len] + "..."
                elif isinstance(val, (list, tuple)):
                    rec[col] = str(val)[:max_char_len]
                else:
                    rec[col] = val
            records.append(rec)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "inspected_rows": len(records),
            "columns": selected_cols,
            "data": records
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"columns": columns, "n_rows": n_rows, "max_char_len": max_char_len},
            summary=summary
        )

        return summary

    return [
        check_missing_tool,
        tokenize_tool,
        list_files_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool
    ]
