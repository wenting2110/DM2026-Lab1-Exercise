# Report

## Loading the data

The dataset was loaded with `load_dataset_tool` from `newdataset/Reddit-stock-sentiment.csv` (Result ID: `load_dataset_1`). `inspect_data_tool` (Result ID: `inspect_data_2`) was used to preview the first five rows, and `describe_data_tool` (Result ID: `describe_data_5`) was used to examine the distribution of document lengths. The results show that a small number of extremely long posts pull the mean upward, as shown below:

![Document Text Length Distribution](../plots/陳玟廷_homework_plot_describe_data_5.png)

`check_missing_tool` (Result ID: `check_missing_3`) and `check_duplicates_tool` (Result ID: `check_duplicates_7`) were then used to check data quality and remove duplicate rows. `build_dtm_tool` (Result ID: `build_dtm_8`) was used to build the document-term matrix, which has 16,659 non-zero entries and a sparsity of 99.52%. `dtm_heatmap_tool` (Result ID: `dtm_heatmap_10`) was used to visualize a 20×20 slice of the matrix, further confirming this high sparsity. As shown below, most terms appear at most once or twice in any single document:

![DTM Heatmap](../plots/陳玟廷_homework_plot_dtm_heatmap_10.png)

## Filtering

`term_frequency_tool` (Result ID: `term_frequency_9`) was used to inspect the top 20 most frequent terms across all documents, which turned out to be English stop words such as `the`, `to`, and `and`, since we did not remove stop words when building the DTM. `variance_filter_tool` was then run with threshold = 0.2 (Result ID: `variance_filter_12`). Because variance filtering ignores a term's relationship to the label and only considers how much its count fluctuates across documents, it likewise retained a large number of stop words.

Unlike variance filtering, `pearson_filter_tool` and `spearman_filter_tool` (both run with threshold = 0.1) measure how strongly each term's count is associated with a given target class.

| Target Class | Pearson Kept | Spearman Kept | Associated Theme |
|---|---|---|---|
| `"1.0"` | 41 (Result ID: `pearson_filter_14`) | 39 (Result ID: `spearman_filter_17`) | Leans toward technology-related topics |
| `"0.0"` | 10 (Result ID: `pearson_filter_15`) | 13 (Result ID: `spearman_filter_18`) | Political/economic vocabulary |
| `"-1.0"` | 18 (Result ID: `pearson_filter_16`) | 21 (Result ID: `spearman_filter_19`) | Leans toward geopolitical/political topics |

Interestingly, `he` and `his` were retained in both the `"0.0"` and `"-1.0"` classes, unlike most other stop words, which were filtered out.

The correlation matrix built by `feature_correlation_matrix_tool` (Result ID: `feature_correlation_23`) visually confirms this: `he` has a noticeably lower correlation with the other high-frequency terms, indicating that, unlike those terms, it does not scale with document length.

![feature_correlation](../plots/陳玟廷_homework_plot_feature_correlation_23.png)


## Dimensionality Reduction and Pattern Mining

Three dimensionality reduction methods were run via `reduce_dimensions_tool`, each projecting the data onto a 2D plane. In the PCA plot (Result ID: `reduce_dimensions_20`), the three classes are fully intermixed, only 31.8% of the variance is explained and the outliers show clearly in the plot. In the t-SNE (Result ID: `reduce_dimensions_21`) and UMAP (Result ID: `reduce_dimensions_22`) plots, the three classes are likewise evenly intermixed across the entire plot. None of the three methods reveal any visible clustering by class.

![PCA Dimensionality Reduction](../plots/陳玟廷_homework_plot_reduce_dimensions_20.png)
![t-SNE Dimensionality Reduction](../plots/陳玟廷_homework_plot_reduce_dimensions_21.png)
![UMAP Dimensionality Reduction](../plots/陳玟廷_homework_plot_reduce_dimensions_22.png)

`mine_patterns_tool` was called once per class, using `"variance"` as the filtering method and the default `"fpgrowth"` algorithm.

- **`"1.0"` (positive, 109 documents)**: with `min_sup=2` (Result ID: `mine_patterns_25`), 367 patterns were mined. The highest-support terms include `doing` (9), `way` (6), `economy` (6), and `tariffs` (5), suggesting that this group tends to discuss tariffs with an optimistic slant.

- **`"-1.0"` (negative, 315 documents)**: with `min_sup=2` (Result ID: `mine_patterns_26`), 721 patterns were mined. The highest-support terms include `war` (8), `country` (8), and `worse` (8). `war` and `country` echo the political-topic terms surfaced earlier by the Pearson/Spearman filters, suggesting this is a genuine thematic signal that shows up consistently across methods. `worse`, as a clearly negative-valence word, also serves as a sanity check that the method is picking up a real signal.

- **`"0.0"` (neutral, 423 documents)**: with `min_sup=4` (Result ID: `mine_patterns_30`), 129 patterns were obtained. The highest-support terms are `today` (8), `currently` (8), `thing` (8), `say` (8), and `deal` (8), along with trading-related terms such as `usd`, `ticker`, `calls`, and `wsb`. The vocabulary here leans clearly toward factual statements and trading mechanics.

