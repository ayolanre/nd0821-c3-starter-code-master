# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a reproducible scikit-learn RandomForestClassifier with 100
trees and a fixed random seed. Numeric Census features are coerced to numeric
values; categorical features are one-hot encoded with unknown categories
ignored at inference time.

## Intended Use

The model estimates whether an adult Census record belongs to the `<=50K` or
`>50K` income class. It is intended for educational experimentation and
pipeline/API demonstration, not for employment, credit, benefits, or other
individual decision-making.

## Training Data

The model is trained from the UCI Census Income dataset (`census.csv`). The
training script uses an 80/20 stratified train/test split with `random_state=42`.
The source data should be cleaned only through the reproducible preprocessing
code so that the raw input remains available for auditability.

## Evaluation Data

Evaluation uses the held-out 20 percent split produced by the training script.
The exact split and artifact metrics are printed by `starter.train_model` and
should be recorded here after training on the supplied dataset.

## Metrics

The implementation reports precision, recall, and F1 (called `fbeta` with
`beta=1`) for the positive class, and supports the same metrics for categorical
feature slices through `performance_on_slices`. On the supplied 20 percent
held-out split, the saved artifact achieved:

- Precision: `0.743865`
- Recall: `0.618622`
- F1/F-beta: `0.675487`

## Ethical Considerations

Income is a sensitive socioeconomic outcome, and the dataset contains
demographic attributes. Model outputs can reproduce historical sampling and
measurement bias. Do not use predictions as evidence of an individual's worth,
eligibility, or protected-class status. Slice metrics should be reviewed before
any research claim or deployment decision.

## Caveats and Recommendations

Performance depends on the supplied dataset and its label definition. Missing
or novel categories are handled technically, not semantically. Report overall
and slice metrics, inspect class imbalance, monitor drift, and obtain domain
and fairness review before any use beyond this exercise.
