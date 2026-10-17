# report

## Loading the data

The 20-newsgroups dataset was loaded using `load_dataset_tool` across four categories, totaling 2,257 documents (result_id: `load_dataset_1`). Since running `mine_patterns_tool` on the full dataset had previously caused the kernel to crash due to memory exhaustion, `sample_data_tool` was used to sample 800 documents as the working set for subsequent analysis (result_id: `sample_data_2`), improving computational efficiency.

`inspect_data_tool` was then used to examine the category and text content of the first five rows (result_id: `inspect_data_3`), confirming the data format was correct. For data quality checks, `check_missing_tool` (result_id: `check_missing_4`) and `check_duplicates_tool` (result_id: `check_duplicates_5`) confirmed 0 missing values and 0 duplicate documents respectively, indicating the sampled subset was clean and required no additional preprocessing before further analysis.

Finally, `describe_data_tool` was used to summarize the statistical distribution of document length (result_id: `describe_data_6`). Overall, the mean (2,006.08) was notably higher than the median (1,232.0), and the maximum value (57,076.0) far exceeded the 75th percentile (2,053.0). Both indicators show that most documents cluster at shorter lengths while a small number of extremely long documents pull the mean upward. This pattern was consistent across the per-category statistics as well (mean exceeded median in every category), with `comp.graphics` showing the most extreme outlier; this distribution is clearly visible in the box plot below.

![Document Text Length Distribution](../plots/陳玟廷_agentic-pipeline_plot_describe_data_6.png)


## Filtering

To build a vectorized representation of documents and terms, `build_dtm_tool` was used to construct the document-term matrix (result_id: `build_dtm_7`). Within the 800-document subset, 20,320 unique terms were identified, with a matrix sparsity of 99.2042%, indicating that the vast majority of terms appear in only a small number of documents.

Examining overall term frequencies via `term_frequency_tool` (result_id: `term_frequency_8`) revealed that the top 20 most frequent terms were almost entirely generic English words (e.g., *the*, *of*, *to*, *and*), rather than content words related to any specific topic.

We further used `dtm_heatmap_tool` (result_id: `dtm_heatmap_9`) to confirm the high sparsity of the matrix. As shown in the 20×20 slice below, most terms appear only once, in a single document.

![DTM Heatmap](../plots/陳玟廷_agentic-pipeline_plot_dtm_heatmap_9.png)

Based on these observations, the next step was to filter out low-discriminative terms. After filtering with `variance_filter_tool` (result_id: `variance_filter_10`, threshold=0.01), 5,023 of the 20,320 terms were kept and 15,297 were removed; rare noise terms were successfully filtered out. 
Notably, however, generic stop words such as *the*, *of*, *to*, and *is* remained in the filtered results. This is because variation in document length causes these words' occurrence counts to fluctuate substantially (appearing frequently in long documents, rarely in short ones), giving them relatively high variance rather than being excluded by the variance filter as might intuitively be expected.

To examine how strongly individual terms correlate with a specific category, correlation filtering was performed against the `comp.graphics` category using `pearson_filter_tool` (result_id: `pearson_filter_11`, threshold=0.1) and `spearman_filter_tool` (result_id: `spearman_filter_12`, threshold=0.1).
The Pearson filter kept 542 of the 20,320 terms, while the Spearman filter kept 761. The top-ranked terms from both methods overlapped substantially, including *graphics*, *video*, *3d*, which related to the computer graphics topic.

To visualize the Pearson filter results more intuitively, `visualize_result_tool` was used to render `pearson_filter_11` as a bar chart, showing the top 20 terms ranked by correlation coefficient.
![Pearson Filter Top Terms](../plots/陳玟廷_agentic-pipeline_plot_turn68.png)


## Correlation analysis

To further examine how terms correlate with one another, `feature_correlation_matrix_tool` (result_id: `feature_correlation_13`) was used to compute a correlation matrix across the top 20 highest-variance terms. The results revealed two distinct co-occurrence mechanisms:

1. **Length-driven**: Correlations among stop words were generally high, likely because longer documents cause all generic words' counts to increase proportionally.
2. **Topic-driven**: In contrast, the content words *jpeg* and *image* showed a correlation of 0.9759, which is far higher than their correlation with any stop word, indicating that their co-occurrence pattern is driven by document topic rather than simply document length.

![Feature Correlation Heatmap](../plots/陳玟廷_agentic-pipeline_plot_feature_correlation_13.png)


## Pattern mining

Next, to explore term co-occurrence patterns within each category, `mine_patterns_tool` was used with tfidf filtering and the FPGrowth algorithm to mine frequent patterns for each of the four categories:

* comp.graphics: result_id `mine_patterns_14`, min_sup=30
* sci.med: result_id `mine_patterns_15`, min_sup=35
* alt.atheism: result_id `mine_patterns_16`, min_sup=25
* soc.religion.christian: result_id `mine_patterns_17`, min_sup=40

The results showed that the top-ranked frequent patterns in all four categories were almost entirely email header fields, such as *lines*, *subject*, *organization*, *edu*, *nntp*, and *host*. These field names appear at the top of nearly every post, giving them extremely high support despite carrying no topical information. For `comp.graphics`, the genuinely topic-related term *graphics* did not appear until rank 12 by support. This suggests that such header-field terms could reasonably be filtered out in the same way stop words are. 
Further comparing the ranking of the word *god* between `alt.atheism` and `soc.religion.christian`, *god* ranked 15th in `alt.atheism` but jumped to 5th in `soc.religion.christian`, showing that frequent pattern mining can still capture meaningful cross-category topical differences with some noise.


## Dimension reduction

For dimensionality reduction, the document-term matrix was reduced to two dimensions and visualized using PCA (result_id: `reduce_dimensions_18`), UMAP (result_id: `reduce_dimensions_19`), and t-SNE (result_id: `reduce_dimensions_20`).

![PCA](../plots/陳玟廷_agentic-pipeline_plot_reduce_dimensions_18.png)
![UMAP](../plots/陳玟廷_agentic-pipeline_plot_reduce_dimensions_19.png)
![t-SNE](../plots/陳玟廷_agentic-pipeline_plot_reduce_dimensions_20.png)

The PCA plot shows the majority of documents clustered tightly together, with only a few extreme outliers near the edge of the plot. 
In contrast, both non-linear methods, UMAP and t-SNE, outperformed PCA. Documents spread out into a band-like distribution, with some categories showing a degree of regional clustering tendency.


## Binarize labels, cosine similarity

Finally, `binarize_labels_tool` (result_id: `binarize_labels_21`) was used to one-hot encode the four category labels, producing a matrix of shape `[800, 4]`, with each category mapped to its own column (alt.atheism=0, comp.graphics=1, sci.med=2, soc.religion.christian=3).

To examine semantic similarity between documents, `cosine_similarity_tool` was used to compare three document pairs:

1. Document indices 0 and 2, both in `sci.med` (result_id: `cosine_similarity_23`, similarity 0.3608)
2. Document indices 0 and 1, from `sci.med` and `soc.religion.christian` respectively (result_id: `cosine_similarity_24`, similarity 0.26)
3. Document indices 1 and 3, both in `soc.religion.christian` (result_id: `cosine_similarity_25`, similarity 0.4063)

The results showed that same-category document pairs consistently had higher similarity than cross-category pairs, though the gap between the two was not especially large.


## Conclusion

The raw text data contains a large amount of noise unrelated to document topic, which dilutes the signal that should reflect topical differences between categories. This pipeline therefore applied a series of text mining tools to filter out this noise and uncover meaningful features that align with each category's underlying topic.

