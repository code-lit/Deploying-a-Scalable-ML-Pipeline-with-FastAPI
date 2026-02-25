# Model Card

## Model Details
This project trains a binary classifier to predict whether an individual's annual income is **>50K** or **<=50K** using the UCI Adult Census (Census Income) dataset.  
The model is a **RandomForestClassifier** (scikit-learn) trained on a mix of numeric features and one-hot encoded categorical features.

## Intended Use
This model is intended for **educational purposes** to demonstrate an end-to-end ML pipeline (preprocessing, training, evaluation, slice analysis, and API deployment).  
It is **not** intended to be used for real-world decision-making (e.g., hiring, lending, housing) due to bias and dataset limitations.

## Training Data
The training data is `data/census.csv` after preprocessing:
- whitespace trimmed from string fields
- missing values marked with `?` removed
- salary labels normalized (removes trailing periods)

After preprocessing, the dataset contains **30,162 rows** and **15 columns** (including the `salary` label).  
Train/test split: **24,129 train** and **6,033 test** (80/20, stratified by `salary`).

## Evaluation Data
The evaluation set is the held-out **test split** (6,033 rows) created from the same cleaned dataset.

## Metrics
The model is evaluated using:
- **Precision**
- **Recall**
- **F1 score** (fbeta with beta=1)

Test-set performance:
- Precision: **0.7392**
- Recall: **0.6245**
- F1: **0.6770**

In addition, performance is computed on **slices** of the test data for each categorical feature (e.g., each unique value of `education`, `workclass`, etc.). Slice metrics are written to `slice_output.txt`.

## Ethical Considerations
The dataset includes sensitive attributes (e.g., race, sex, native-country). Models trained on this data may learn and reproduce historical and societal biases present in the dataset.  
Predictions should not be used to make decisions that impact individuals.

## Caveats and Recommendations
- This model was trained on a historical dataset and may not generalize to modern populations.
- The dataset is known to include bias; even high accuracy does not imply fairness.
- If this model were to be used beyond this educational context, it would require:
  - fairness evaluation and mitigation
  - robust monitoring for drift
  - a careful review of legal and ethical implications
