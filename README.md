# fda-classifiers
Classifying new medical devices by closest possible FDA approved predicate device for 501(k) filing

# Project Description
- Turned product blurbs into vectors (tf-idf + synonyms) and computed dot products against FDA product code titles/descriptions.
- Returned top candidates with scores and highlighted term overlap for quick memo use.
- Evaluated on a small holdout from public FDA device dictionaries; exported CSV with ranking + rationale.
Role: researcher; stack: python, scikit-learn/pandas.
