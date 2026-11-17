# Week 6 – Day 1: Embeddings + Vector Databases

## Overview
This project demonstrates cosine similarity and practical applications of text embeddings using Sentence Transformers.

## Technologies Used
- Python
- NumPy
- Pandas
- Sentence Transformers
- Scikit-learn
- K-Means

## Task 1: Cosine Similarity from Scratch

Implemented cosine similarity using NumPy and compared results with Scikit-learn.

### Results

| Vector A | Vector B | Cosine Similarity |
|---|---|---:|
| [1,2,3] | [2,4,6] | 1.0 |
| [1,0] | [0,1] | 0.0 |
| [1,0] | [-1,0] | -1.0 |

Normalized dot product: 0.478091

All three test cases matched Scikit-learn.

## Task 2: Use Cases Mini-Lab

### 1. Semantic Search
- 15 FAQ entries
- 3 search queries
- Top 3 results for each query
- Used cosine similarity to rank relevant FAQs.

### 2. Duplicate Detection
- 12 customer complaints
- Similarity threshold: 0.75
- Detected pairs:
  - Complaint 5 and 6: 0.7694
  - Complaint 11 and 12: 0.7966

### 3. K-Means Clustering
- 30 headlines
- 3 clusters

Cluster themes:
1. Finance and Economy
2. Technology and Innovation
3. Sports

### 4. Product Recommendation
- 15 product descriptions
- Retrieved top 3 recommendations.

Actual top results:
1. Laptop travel backpack – 0.6987
2. Ultralight laptop with long battery life – 0.5419
3. Gaming laptop with powerful graphics – 0.5216

## Bonus Challenge: Precision and Recall

Duplicate detection evaluation:

| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 0.3333 |
| F1-score | 0.5000 |

Confusion Matrix:

```text
[[6, 0],
 [4, 2]]
```

## Concepts Learned
- Cosine similarity
- Dot product and vector normalization
- Semantic search
- Duplicate detection
- K-Means clustering
- Product recommendation
- Precision and recall

## Conclusion
Successfully implemented cosine similarity and four practical embedding use cases using the all-MiniLM-L6-v2 model. Evaluated duplicate detection using precision, recall, and F1-score.