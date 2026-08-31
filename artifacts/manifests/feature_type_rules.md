# Feature-type audit rules


Rules (frozen, reproducible; do not change for convenience):

1. Start from observed pandas Series after OpenML dataframe load.
2. Categorical if ANY of:
   - dtype is pandas CategoricalDtype
   - dtype == object
   - dtype is boolean
3. Otherwise numeric if pandas considers the series numeric (excluding bool).
4. Continuous numeric: numeric AND >2 unique non-null values AND NOT integer-valued
   (integer-valued := all non-null values within 1e-9 of an integer).
5. Integer-valued numeric: numeric AND all non-null values near integers.
   These remain numeric under preprocessing (not ordinal-encoded) unless rule 2 applies.
6. Missingness indicators: one per source feature with any missing value in the audited
   table (TRAIN-fit in experiments; full-table estimate used for screening upper bound).
7. Encoded feature count estimate = n_numeric + n_categorical + missing_indicator_count
   (ordinal encoding keeps one column per categorical feature).
8. If OpenML categorical_indicator disagrees with rules 2–3, record
   dtype_metadata_disagreement and DO NOT silently override; screening uses observed
   rules 2–3, disagreement flagged for review.
