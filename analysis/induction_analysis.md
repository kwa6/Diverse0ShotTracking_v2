



## DSG5K Induction Analysis

* 30 noises (-1 cluster) randomly sampled
* 7/30 incorrect
  * 4 vague
  * 2 hallucinated
  * 1 inappropriate slot name
* This is not substantially worse than baseline noise level, so we do not filter unclustered data

Although pilot induction slightly prefered the slot-only encoding strategy with sbert, manual review revealed a high degree of cluster impurity, such as

| Column 3                         | Column 4                                          |
|----------------------------------|---------------------------------------------------|
| expensive guest house            | No                                                |
| moderate priced guest houses     | 3                                                 |
| guest house type                 | Expensive                                         |
| moderate guest houses east side  | "carolina bed and breakfast", "warkworth house"   |
| moderate priced guest house      | hobsons house                                     |
| guest houses meeting criteria    | 12                                                |
| guest houses location            | East                                              |
| guest house help                 | True                                              |
| guest house 1                    | a and b guest house                               |
| guest house west side            | 2                                                 |


Missing slot errors:

| Category               | Count |
|------------------------|-------|
| Coreference            | 8     |
| Missing                | 5     |
| Slot Variant           | 23    |
| Destination/Departure  | 3     |
| No Domain              | 25    |
| Correct                | 17    |
| Value in Slot          | 7     |
| Incorrect Value        | 2     |


Approach pilot after grid search:

| Model   | Encoding | UMAP Used | Precision | Recall | F1 Score |
|---------|----------|-----------|-----------|--------|----------|
| sbert   | s        | No        | 0.429     | 0.935  | 0.588    |
| sbert   | sv       | No        | 0.453     | 0.742  | 0.562    |
| roberta | sv       | Yes       | 0.350     | 0.548  | 0.427    |
| roberta | s        | No        | 0.786     | 0.258  | 0.389    |
| roberta | ts       | Yes       | 0.286     | 0.290  | 0.288    |
| sbert   | ts       | Yes       | 0.263     | 0.258  | 0.261    |


## Previous Work evaluation

Using the previous work's automatic evaluation methodology, we can trivially achieve a competitive result (random grid search of hyperparameters, mwoz2.4 validation data, RoBERTa embedding centroids with cosine similarity threshold of 0.8 to automatically map predicted clusters to reference clusters).

**Difference: RoBERTa embeddings of values for matching are created by concatenating dialogue context with slot-value pair and encoding the value, since many values are abstractive. Previous works' predicted values were extractive and therefore embedded in the original context, and it is not specified how gold values were embedded**. 

| Parameter                  | Value    |
|----------------------------|----------|
| encoding_model             | sbert    |
| encoding_type              | s        |
| clustering_algorithm       | hdbscan  |
| dim_reduction_algorithm    | umap     |
| min_cluster_size           | 50       |
| cluster_merge_eps          | 0.0      |
| dim_reduction_n_neighbors  | 2        |
| dim_reduction_n_dimensions | 10       |
| dim_reduction_min_dist     | 0.9      |
| precision                  | 0.992248 |
| recall                     | 0.967742 |
| f1                         | 0.979842 |
| N                          | 129      |
