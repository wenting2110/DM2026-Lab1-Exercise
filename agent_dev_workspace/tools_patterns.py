import os
import ast
import tempfile
from typing import Dict, Any, Optional, Literal, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI pattern mining modules safely
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth
import PAMI.extras.convert.DF2DB as DF2DB

def _clean_itemset(itemset_obj) -> List[str]:
    raw_str = str(itemset_obj).strip()
    if raw_str.startswith("(") and raw_str.endswith(")"):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, (tuple, list)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    items = [item.strip(" '\"(),") for item in raw_str.replace("\t", " ").split()]
    return [i for i in items if i]

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent itemsets for a specific document category.

        Fits a category-local CountVectorizer(stop_words="english"), applies vocabulary
        filtering (variance, tfidf, or term_frequency), exports a transactional database,
        and mines patterns using PAMI algorithms (FPGrowth, FAE/TopK, or MaxFPGrowth).

        Args:
            category_name: Category label to filter session.dataframe by.
            filtering_method: Vocabulary filter method ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI algorithm ("fpgrowth", "topk", or "maxfpgrowth", default "fpgrowth").
            min_sup: Required for "fpgrowth" and "maxfpgrowth" (minimum absolute support count).
            k: Required for "topk" (number of top highest-support patterns to keep).

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, and mined patterns.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe loaded in session.")

        # Parameter validation per spec contract
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None:
                raise ValueError(f"min_sup (int) is required when algorithm='{algorithm}'.")
            if min_sup <= 0:
                raise ValueError(f"min_sup must be positive, got {min_sup}.")

        if algorithm == "topk":
            if k is None:
                raise ValueError("k (int) is required when algorithm='topk'.")
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}.")

        # Find category column
        cat_col = "category_name" if "category_name" in session.dataframe.columns else ("category" if "category" in session.dataframe.columns else None)
        if cat_col is None:
            raise ValueError("No category or category_name column in session.dataframe.")

        df_cat = session.dataframe[session.dataframe[cat_col] == category_name]
        if len(df_cat) == 0:
            raise ValueError(f"No documents found for category '{category_name}'.")

        texts = df_cat["text"].fillna("").astype(str).tolist()

        # Step 1: Fit category-local vectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError: # Empty vocabulary after stop words
            X_cat = None

        if X_cat is None or X_cat.shape[1] == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stop word removal"
            }
            session.store_result("mine_patterns_tool", args={"category_name": category_name}, summary=summary)
            return summary

        feature_names = vectorizer.get_feature_names_out()
        X_dense = X_cat.toarray()
        n_terms = X_dense.shape[1]

        # Step 2: Apply chosen vocabulary filtering method
        kept_indices = []
        if filtering_method == "variance":
            scores = np.var(X_dense, axis=0, ddof=0)
            if n_terms >= 2:
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                kept_indices = np.where((scores > p5) & (scores < p95))[0].tolist()
            else:
                kept_indices = []
        elif filtering_method == "term_frequency":
            scores = np.asarray(X_dense.sum(axis=0)).flatten()
            if n_terms >= 2:
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                kept_indices = np.where((scores > p5) & (scores < p95))[0].tolist()
            else:
                kept_indices = []
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(X_cat).toarray()
            scores = np.mean(tfidf, axis=0)
            p20 = np.percentile(scores, 20)
            kept_indices = np.where(scores >= p20)[0].tolist()

        # Step 3: Handle zero surviving terms cleanly
        if len(kept_indices) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", args={"category_name": category_name}, summary=summary)
            return summary

        # Step 4: Convert surviving terms to transactional DB
        kept_names = [feature_names[i] for i in kept_indices]
        X_filtered = X_dense[:, kept_indices]

        df_binary = pd.DataFrame((X_filtered >= 1).astype(int), columns=kept_names)

        # Remove completely empty rows from transactional DB
        df_binary = df_binary[df_binary.sum(axis=1) > 0]

        patterns = []
        if len(df_binary) > 0:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_db:
                db_path = tmp_db.name

            try:
                converter = DF2DB.DF2DB(df_binary)
                converter.convert2TransactionalDatabase(db_path, ">=", 1)

                if algorithm == "fpgrowth":
                    obj = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
                    obj.mine()
                    raw_pats = obj.getPatterns()
                elif algorithm == "topk":
                    obj = FAE.FAE(db_path, k=k, sep="\t")
                    obj.mine()
                    raw_pats = obj.getPatterns()
                elif algorithm == "maxfpgrowth":
                    obj = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
                    obj.mine()
                    raw_pats = obj.getPatterns()

                for itemset, sup in raw_pats.items():
                    item_list = _clean_itemset(itemset)
                    patterns.append({"pattern": item_list, "support": int(sup)})

            finally:
                if os.path.exists(db_path):
                    os.remove(db_path)

        # Step 5: Store artifact and result
        art_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[art_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
            "pattern_count": len(patterns),
            "patterns": patterns,
            "artifact_key": art_key
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary
        )

        return summary

    return [mine_patterns_tool]
