## Summary

For toxic comment classification, both logistic regression and naive Bayes were
chosen, with logistic regression showing significantly better results (F1-score
0.75 versus 0.28 for the toxic class), providing more balanced precision (0.92)
and recall (0.63) metrics.

The most effective preprocessing steps were stop word removal, lemmatization,
and TF-IDF vectorization, which helped identify significant words and eliminate
noise, achieving high overall accuracy (0.96 for logistic regression).

With additional resources, the model could be improved by focusing on
increasing recall for the toxic class (especially for naive Bayes, where it's
only 0.16), implementing transformer-based architectures (BERT, RoBERTa) or
class balancing methods to handle the clearly imbalanced dataset (28,671
non-toxic versus 3,244 toxic comments).

---

## Model Performance

### Logistic Regression

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0                | 0.96      | 0.99   | 0.98     | 28671   |
| 1                | 0.92      | 0.63   | 0.75     | 3244    |
| **Accuracy**     |           |        | 0.96     | 31915   |
| **Macro avg**    | 0.94      | 0.81   | 0.86     | 31915   |
| **Weighted avg** | 0.96      | 0.96   | 0.95     | 31915   |

### Multinomial Naive Bayes

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0                | 0.91      | 1.00   | 0.95     | 28671   |
| 1                | 0.99      | 0.16   | 0.28     | 3244    |
| **Accuracy**     |           |        | 0.91     | 31915   |
| **Macro avg**    | 0.95      | 0.58   | 0.62     | 31915   |
| **Weighted avg** | 0.92      | 0.91   | 0.89     | 31915   |