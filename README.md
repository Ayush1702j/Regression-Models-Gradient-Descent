# Regression Models and Gradient Descent Optimization

## Real-World Application: Disease Progression Prediction

This project implements and evaluates multiple regression models for a real-world regression problem and investigates the performance of **Gradient Descent optimization for Linear Regression**.

The project was designed around the following academic objectives:

1. **Develop regression models for a real-world application and evaluate their performance using appropriate metrics.**
2. **Implement and analyze the performance of Gradient Descent optimization for Linear Regression.**

The implementation is organized as a reproducible Python project containing the dataset, modular source code, experimental results, graphs, and this detailed documentation.

---

## 1. Abstract

Regression is a supervised machine learning technique used when the output variable is continuous. In this project, regression is applied to a real-world **disease progression prediction** problem. The dataset contains baseline measurements for patients and a quantitative target representing disease progression one year after the baseline measurements.

Four regression approaches are developed and compared:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression

The models are evaluated using **Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score**.

In addition, Linear Regression is implemented from scratch using **Batch Gradient Descent**. The experiment investigates how different learning rates affect the optimization process, final cost, prediction error, and R² score.

The objective is not simply to obtain a numerical result, but to understand the relationship between model selection, preprocessing, optimization, evaluation metrics, convergence, and model limitations.

---

## 2. Problem Statement

A quantitative measure of disease progression must be predicted from a set of baseline patient measurements.

The problem can be represented as:

```text
Input features
     |
     v
Baseline measurements
     |
     v
Regression Model
     |
     v
Predicted disease-progression value
```

The project addresses two questions:

1. How do different regression models perform on the same dataset?
2. How does the learning rate affect Gradient Descent when training Linear Regression?

---

## 3. Motivation

A regression model can help identify relationships between input variables and a continuous outcome. However, selecting a model only because it produces a particular score is not sufficient.

This project therefore studies:

- model assumptions,
- preprocessing requirements,
- regularization,
- nonlinear feature relationships,
- prediction error,
- optimization using Gradient Descent,
- learning-rate behavior,
- convergence,
- and limitations of the experimental results.

The project also demonstrates the difference between using a ready-made machine learning estimator and understanding the underlying optimization algorithm.

---

## 4. Objectives

### Primary Objectives

- Implement multiple regression models.
- Apply the models to a real-world continuous prediction problem.
- Split the dataset into training and testing sets.
- Standardize input features using training data only.
- Evaluate models using appropriate regression metrics.
- Implement Linear Regression using Batch Gradient Descent from scratch.
- Experiment with multiple learning rates.
- Plot the Gradient Descent cost curve.
- Compare model performance and interpret the results.
- Document limitations, observations, and future improvements.

### Learning Objectives

After completing this project, the learner should understand:

- what regression means,
- how Linear Regression works,
- why nonlinear features may be useful,
- why regularization is introduced,
- how MAE, MSE, RMSE and R² differ,
- how Gradient Descent updates model parameters,
- how learning rate affects optimization,
- and why experimental results must be interpreted rather than simply reported.

---

# 5. Dataset Description

## Dataset Used

The project uses the **Diabetes Regression Dataset distributed with scikit-learn**.

The dataset contains:

- **442 observations**
- **10 input features**
- **1 continuous target variable**

The target is a quantitative measure of disease progression one year after baseline.

### Input Features

| Feature | Description in the dataset | Type |
|---|---|---|
| `age` | Age-related baseline feature | Numerical |
| `sex` | Sex-related encoded feature | Numerical |
| `bmi` | Body-mass-index-related feature | Numerical |
| `bp` | Blood-pressure-related feature | Numerical |
| `s1` | Serum measurement 1 | Numerical |
| `s2` | Serum measurement 2 | Numerical |
| `s3` | Serum measurement 3 | Numerical |
| `s4` | Serum measurement 4 | Numerical |
| `s5` | Serum measurement 5 | Numerical |
| `s6` | Serum measurement 6 | Numerical |
| `target` | Quantitative disease progression measure | Continuous target |

The feature values in the supplied scikit-learn version are already represented in a standardized form. The project still applies a separate training-set-based `StandardScaler` so that the experimental pipeline explicitly demonstrates correct scaling and ensures numerical conditioning for Gradient Descent.

### Dataset Characteristics

```text
Rows        : 442
Input cols  : 10
Target cols : 1
Problem     : Regression
Output      : Continuous numerical value
```

The CSV copy is included in the repository under:

```text
 data/diabetes_regression.csv
```

Therefore, the project can run without downloading the dataset during execution.

---

# 6. State of the Machine Learning Problem

A supervised regression problem can be represented as:

```text
X = input features
Y = continuous target

Training data:
(X_train, Y_train)

Testing data:
(X_test, Y_test)
```

The objective is to learn a function:

```text
f(X) -> Y
```

such that predictions on unseen test data are as close as possible to the actual target values.

---

# 7. Data Preprocessing

The preprocessing pipeline consists of the following steps:

```text
CSV Dataset
    |
    v
Separate features and target
    |
    v
Train/Test Split
    |
    +---- Training Data ----> Fit StandardScaler
    |                              |
    |                              v
    |                       Transform Training Data
    |
    +---- Testing Data ----> Transform using same scaler
    |
    v
Model Training
    |
    v
Prediction
    |
    v
Evaluation
```

## 7.1 Train-Test Split

The project uses:

- Test size = 20%
- Training size = 80%
- Random state = 42

The resulting split contains:

```text
Training samples = 353
Testing samples  = 89
```

The test set is kept separate from model fitting so that performance can be evaluated on unseen observations.

## 7.2 Feature Scaling

Standardization is performed using:

```text
z = (x - mean) / standard deviation
```

The scaler is fitted only on the training data and then applied to both training and testing data.

This prevents information from the test set from influencing preprocessing parameters.

Feature scaling is particularly important for Gradient Descent because features with substantially different scales can cause uneven parameter updates and slow optimization.

---

# 8. Regression Model 1 — Linear Regression

## 8.1 Definition

Linear Regression assumes that the target can be represented approximately as a weighted combination of the input features.

For multiple features:

```text
ŷ = b0 + b1x1 + b2x2 + ... + bnxn
```

where:

- `ŷ` = predicted target
- `b0` = intercept
- `bi` = coefficient of feature i
- `xi` = input feature

## 8.2 Objective

The model attempts to minimize prediction error, commonly using Mean Squared Error:

```text
MSE = (1/m) Σ(yi - ŷi)^2
```

## 8.3 Why Linear Regression Was Selected

Linear Regression is used as the **baseline model** because it is simple, interpretable, computationally efficient, and provides a reference against which more complex models can be compared.

## 8.4 Advantages

- Easy to understand.
- Fast to train.
- Coefficients are interpretable.
- Good baseline for regression experiments.

## 8.5 Limitations

- Assumes a linear relationship.
- Sensitive to influential observations.
- May underfit nonlinear relationships.
- Correlated features can make coefficient interpretation difficult.

---

# 9. Regression Model 2 — Polynomial Regression

## 9.1 Definition

Polynomial Regression extends Linear Regression by creating polynomial and interaction features.

For one feature:

```text
ŷ = b0 + b1x + b2x²
```

For multiple features, degree-2 transformation can contain:

- original features,
- squared features,
- pairwise interaction terms.

## 9.2 Configuration

The project uses:

```text
Polynomial degree = 2
```

The polynomial features are then passed to Linear Regression.

## 9.3 Why It Was Selected

A purely linear model may fail to capture nonlinear relationships. Polynomial Regression provides a controlled way to test whether additional nonlinear terms improve generalization.

## 9.4 Limitation

Increasing polynomial degree can dramatically increase the number of features. This can increase computational cost and the risk of overfitting.

Therefore, Polynomial Regression should not automatically be considered better simply because it is more flexible.

---

# 10. Regression Model 3 — Ridge Regression

## 10.1 Definition

Ridge Regression is Linear Regression with L2 regularization.

The objective can be represented as:

```text
J = MSE + λ Σ βj²
```

where `λ` controls the regularization strength.

## 10.2 Configuration

This project uses:

```text
alpha = 1.0
```

## 10.3 Purpose

Regularization discourages excessively large coefficients and can improve numerical stability and generalization.

## 10.4 Advantages

- Reduces coefficient magnitude.
- Helps with multicollinearity.
- Can reduce overfitting.
- Usually retains all features.

## 10.5 Limitation

Ridge does not normally force coefficients exactly to zero, so it is not a direct feature-selection method.

---

# 11. Regression Model 4 — Lasso Regression

## 11.1 Definition

Lasso Regression applies L1 regularization:

```text
J = MSE + λ Σ |βj|
```

## 11.2 Configuration

This project uses:

```text
alpha = 0.01
```

## 11.3 Purpose

L1 regularization can shrink some coefficients to zero. Therefore, Lasso can simultaneously perform regression and a form of feature selection.

## 11.4 Advantages

- Can produce sparse models.
- Can reduce unnecessary features.
- Provides regularization.

## 11.5 Limitations

- Performance depends on the regularization parameter.
- Strong regularization can remove useful features.
- Correlated features can make coefficient selection less stable.

---

# 12. Why Multiple Models Were Selected

The four models represent different modeling assumptions:

| Model | Main Idea | Purpose in Experiment |
|---|---|---|
| Linear Regression | Linear relationship | Baseline |
| Polynomial Regression | Nonlinear feature relationships | Test additional flexibility |
| Ridge | L2 regularization | Test coefficient shrinkage |
| Lasso | L1 regularization | Test sparsity/feature selection |

This selection allows the experiment to compare **baseline modeling, nonlinear expansion, L2 regularization, and L1 regularization** rather than testing several models that behave almost identically.

---

# 13. Evaluation Metrics

Regression models should be evaluated using metrics appropriate for continuous predictions.

## 13.1 Mean Absolute Error — MAE

```text
MAE = (1/n) Σ |yi - ŷi|
```

MAE represents the average absolute difference between actual and predicted values.

### Interpretation

- Lower MAE is better.
- MAE is relatively easy to interpret.
- Every error contributes proportionally to the metric.

---

## 13.2 Mean Squared Error — MSE

```text
MSE = (1/n) Σ (yi - ŷi)²
```

MSE squares each error before averaging.

### Interpretation

- Lower MSE is better.
- Large errors receive greater penalty.

---

## 13.3 Root Mean Squared Error — RMSE

```text
RMSE = √MSE
```

RMSE is useful because it is expressed in the same target units as the prediction error.

### Interpretation

- Lower RMSE is better.
- Large errors influence RMSE more strongly than MAE.

---

## 13.4 R² Score

```text
R² = 1 - SSres / SStot
```

R² measures the proportion of variation in the target explained by the model relative to a constant-mean baseline.

### Interpretation

- Higher R² is generally better for the same evaluation dataset.
- R² should be interpreted together with error metrics and the experimental context.

---

# 14. Gradient Descent

## 14.1 What is Gradient Descent?

Gradient Descent is an iterative optimization algorithm used to minimize a cost function.

Instead of directly calculating the optimal parameters, the algorithm starts with an initial parameter vector and repeatedly moves it in the direction that decreases the cost.

```text
Initial parameters
       |
       v
Calculate predictions
       |
       v
Calculate errors
       |
       v
Calculate gradient
       |
       v
Update parameters
       |
       v
Repeat
```

---

# 15. Gradient Descent Cost Function

For Linear Regression, the project uses:

```text
J(θ) = (1 / 2m) Σ (ŷi - yi)²
```

where:

- `J(θ)` = cost function
- `m` = number of training observations
- `ŷi` = predicted value
- `yi` = actual value
- `θ` = parameter vector

The factor `1/2` simplifies the derivative.

---

# 16. Gradient Calculation

Using matrix notation:

```text
Xb = [1  X]

ŷ = Xb θ

error = ŷ - y

gradient = (1/m) Xbᵀ error
```

The parameter update is:

```text
θ := θ - α gradient
```

where `α` is the learning rate.

---

# 17. Batch Gradient Descent Algorithm

The implementation in `src/models.py` follows this procedure:

```text
1. Add a column of ones to X for the intercept.
2. Initialize all parameters to zero.
3. Repeat for the selected number of iterations:
      a. Calculate predictions.
      b. Calculate prediction errors.
      c. Calculate the cost.
      d. Calculate the gradient.
      e. Update all parameters.
4. Store the cost after every iteration.
5. Use the final parameters for prediction.
```

### Pseudocode

```text
initialize θ = 0

for iteration = 1 to N:
    prediction = Xθ
    error = prediction - y
    cost = mean(error²) / 2
    gradient = Xᵀerror / m
    θ = θ - learning_rate × gradient

return θ
```

---

# 18. Learning Rate Experiment

The following learning rates are tested:

```text
0.001
0.01
0.05
0.1
```

The same maximum number of iterations is used for each experiment so that the effect of the learning rate can be compared under a controlled experimental setup.

The experiment records:

- final cost,
- MAE,
- RMSE,
- R²,
- and cost history.

The cost curve allows us to observe whether the optimization is moving toward a stable minimum.

---

# 19. Why Feature Scaling Matters for Gradient Descent

Gradient Descent updates all parameters using the gradient.

If one feature has a much larger numerical scale than another, its contribution to the gradient can dominate the update.

This can lead to:

- slower convergence,
- uneven updates,
- difficulty selecting a suitable learning rate.

Therefore, the project standardizes input features before training the custom Gradient Descent model.

The target is also standardized internally for stable optimization and transformed back to the original target scale before reporting prediction metrics.

---

# 20. Experimental Methodology

The experiment follows this procedure:

```text
Step 1: Load dataset
       ↓
Step 2: Separate X and y
       ↓
Step 3: Split into train and test data
       ↓
Step 4: Fit scaler using training data
       ↓
Step 5: Transform training and testing features
       ↓
Step 6: Train Linear Regression
       ↓
Step 7: Train Polynomial Regression
       ↓
Step 8: Train Ridge Regression
       ↓
Step 9: Train Lasso Regression
       ↓
Step 10: Calculate MAE/MSE/RMSE/R²
       ↓
Step 11: Implement custom Gradient Descent
       ↓
Step 12: Test multiple learning rates
       ↓
Step 13: Save CSV results
       ↓
Step 14: Generate graphs
       ↓
Step 15: Analyze results
```

---

# 21. Software and Hardware Requirements

## Software

- Python 3.10 or newer recommended
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Git
- GitHub

## Hardware

The project is small enough to run on a normal student laptop or desktop.

No GPU is required.

---

# 22. Project Architecture

```text
Regression_Gradient_Descent/
│
├── data/
│   └── diabetes_regression.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── results/
│   ├── model_results.csv
│   └── gradient_descent_results.csv
│
├── graphs/
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   ├── gradient_descent_cost.png
│   ├── learning_rate_vs_cost.png
│   └── learning_rate_iterations.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 23. Description of Source Files

## `main.py`

The main experiment driver.

It:

- loads the dataset,
- prepares the data,
- trains the regression models,
- evaluates the models,
- runs Gradient Descent experiments,
- saves results,
- and generates graphs.

## `src/data_preprocessing.py`

Responsible for:

- loading the CSV,
- separating features and target,
- train-test splitting,
- feature standardization.

## `src/models.py`

Contains:

- Linear Regression model definitions,
- Polynomial Regression pipeline,
- Ridge Regression,
- Lasso Regression,
- custom `LinearRegressionGD` implementation,
- metric calculation.

## `src/evaluation.py`

Contains reusable evaluation and CSV-export functions.

## `src/visualization.py`

Generates the experimental graphs.

---

# 24. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Regression-Gradient-Descent.git
cd Regression-Gradient-Descent
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 25. How to Run

Run:

```bash
python main.py
```

The program prints the regression results and Gradient Descent results in the terminal.

It also generates CSV files inside `results/` and graphs inside `graphs/`.

---

# 26. Expected Program Output

A representative run produces output similar to:

```text
========================================================================
REGRESSION MODELS + GRADIENT DESCENT EXPERIMENT
Real-world application: Disease Progression Prediction
========================================================================
Dataset shape: 442 rows x 11 columns
Training samples: 353
Testing samples : 89

REGRESSION MODEL RESULTS
...

GRADIENT DESCENT RESULTS
...

Generated files:
  results/gradient_descent_results.csv
  results/model_results.csv
  graphs/actual_vs_predicted.png
  graphs/gradient_descent_cost.png
  graphs/learning_rate_iterations.png
  graphs/learning_rate_vs_cost.png
  graphs/model_comparison.png
```

The exact metric values can vary if the data split or implementation configuration is changed. The CSV files generated by the actual run should be treated as the authoritative experimental results.

---

# 27. Experimental Results

The included run used:

```text
Random state : 42
Test size    : 20%
Train size   : 353
Test size    : 89
```

## 27.1 Regression Model Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 42.7941 | 2900.1936 | 53.8534 | 0.4526 |
| Polynomial Regression (Degree 2) | 43.5817 | 3096.0283 | 55.6420 | 0.4156 |
| Ridge Regression | 42.8120 | 2892.0146 | 53.7775 | 0.4541 |
| Lasso Regression | 42.7950 | 2898.3680 | 53.8365 | 0.4529 |

### Interpretation

The observed test results are quite close for Linear, Ridge, and Lasso Regression. Ridge has the lowest RMSE and the highest R² among these four models in this particular run, while Polynomial Regression has the largest error values and lower R².

This does **not** mean Ridge Regression is universally the best regression algorithm. It means that under this dataset, split, preprocessing pipeline, and selected hyperparameters, Ridge produced the strongest metrics in this particular experiment.

Polynomial Regression did not improve the test result in this configuration. The added degree-2 terms increased model flexibility but did not translate into better generalization on the test set.

---

# 28. Gradient Descent Results

The included run used 3,000 iterations for each learning rate.

| Learning Rate | Iterations | Final Cost | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 3000 | 0.238234 | 42.994994 | 53.604863 | 0.457645 |
| 0.01 | 3000 | 0.236856 | 42.865808 | 53.722332 | 0.455265 |
| 0.05 | 3000 | 0.235534 | 42.815545 | 53.785936 | 0.453974 |
| 0.1 | 3000 | 0.235382 | 42.799299 | 53.834207 | 0.452994 |

### Interpretation

The learning rates tested in this experiment all produced stable results under the selected feature scaling and iteration count.

The final optimization cost decreased as the learning rate increased across the tested values, although the prediction metrics do not improve monotonically in exactly the same order.

This is an important observation: **a lower training cost does not automatically guarantee the lowest test error**. Optimization quality and generalization performance are related but distinct concepts.

The 0.001 learning rate makes smaller parameter updates and therefore approaches the solution more slowly. Larger learning rates move more aggressively toward the minimum. However, increasing the learning rate indefinitely would not be safe; an excessively large learning rate can cause oscillation or divergence.

---

# 29. Graph Analysis

## 29.1 Actual vs Predicted Graph

`graphs/actual_vs_predicted.png` plots actual target values against predicted values for the model with the highest R² in the experiment.

A perfect prediction would place points directly on the diagonal reference line.

Points far from the line represent larger prediction errors.

The plot should therefore be interpreted together with MAE, RMSE, and R² rather than as a standalone measure.

---

## 29.2 Model Comparison Graph

`graphs/model_comparison.png` compares the main regression metrics across the four models.

The graph helps visually identify differences in:

- MAE,
- RMSE,
- and R².

Because MAE/RMSE and R² have different scales and meanings, the graph is intended for visual comparison rather than as a single combined score.

---

## 29.3 Gradient Descent Cost Curve

`graphs/gradient_descent_cost.png` shows the cost function across iterations for the 0.05 learning rate.

A generally decreasing curve indicates that the optimization process is reducing the training objective.

If the curve oscillates heavily or grows, the learning rate may be too large or the numerical setup may be unstable.

---

## 29.4 Learning Rate vs Cost

`graphs/learning_rate_vs_cost.png` compares the final optimization cost for different learning rates.

This graph demonstrates that learning rate is an important hyperparameter in iterative optimization.

---

# 30. Critical Comparative Analysis

## Linear Regression vs Polynomial Regression

Linear Regression provides a simple baseline. Polynomial Regression adds nonlinear and interaction terms.

In the observed experiment:

- Linear Regression achieved RMSE = 53.8534.
- Polynomial Regression achieved RMSE = 55.6420.

The polynomial model therefore did not improve the test performance in this configuration.

A possible explanation is that the added degree-2 features increased model flexibility without capturing a useful generalizable pattern in the available training data. Higher-dimensional polynomial features can also increase the risk of overfitting.

This result demonstrates why increasing model complexity should be supported by validation evidence rather than assumed to improve performance.

---

## Linear Regression vs Ridge Regression

The observed results were:

```text
Linear RMSE = 53.8534
Ridge  RMSE = 53.7775
```

Ridge produced a slightly lower RMSE and slightly higher R² in this run.

The difference is small, which indicates that regularization provided only a modest change under the selected alpha and dataset split.

A stronger conclusion would require repeated cross-validation and hyperparameter tuning.

---

## Linear Regression vs Lasso Regression

The observed results were very close:

```text
Linear R² = 0.4526
Lasso  R² = 0.4529
```

The small difference suggests that the selected Lasso regularization did not dramatically change predictive performance.

Lasso can still be useful when interpretability or sparse feature representations are important, even when its prediction score is similar to Linear Regression.

---

# 31. Critical Analysis of Gradient Descent

Gradient Descent provides an iterative way to obtain Linear Regression parameters.

The experiment shows several important concepts.

### Observation 1 — Learning Rate Matters

The learning rate determines the size of each update.

```text
Small α  → smaller updates → slower progress
Large α  → larger updates → faster progress, but greater instability risk
```

### Observation 2 — Cost and Test Metrics Are Different

The cost function is calculated on training data during optimization, whereas MAE/RMSE/R² reported here are calculated on the test set.

Therefore, a lower final training cost should not automatically be interpreted as better generalization.

### Observation 3 — Feature Scaling Helps

Scaling makes the numerical optimization more stable because the feature dimensions are placed on comparable scales.

### Observation 4 — Fixed Iteration Count

The learning-rate experiment uses the same maximum iteration count for every rate. This creates a controlled comparison, but it does not measure the exact number of iterations required for convergence.

A future version could implement an early-stopping criterion based on the change in cost or gradient norm.

---

# 32. Advantages of the Project

- Uses a real-world regression problem.
- Includes multiple regression models.
- Uses multiple evaluation metrics.
- Implements Gradient Descent manually.
- Tests multiple learning rates.
- Uses training-only feature scaling.
- Generates reproducible CSV results.
- Generates graphs automatically.
- Separates code into reusable modules.
- Includes detailed documentation.

---

# 33. Limitations

The current experiment has several limitations.

### 33.1 Single Train-Test Split

Only one random train-test split is used. Results can change with another split.

### 33.2 Limited Hyperparameter Search

The project uses selected values for Ridge and Lasso rather than an extensive cross-validation search.

### 33.3 Polynomial Degree

Only degree 2 is tested. Other degrees could be evaluated, but higher degrees may substantially increase feature count and overfitting risk.

### 33.4 Limited Gradient Descent Study

Only four learning rates are tested and the number of iterations is fixed.

### 33.5 Dataset Size

The dataset contains 442 observations, which is relatively small compared with many modern machine-learning datasets.

### 33.6 Generalization

The experimental result should not be interpreted as evidence that one model will always outperform another on different datasets.

---

# 34. Future Scope

The project can be extended using:

1. K-Fold Cross Validation.
2. Grid Search for Ridge and Lasso alpha.
3. More polynomial degrees.
4. Elastic Net Regression.
5. Robust Regression.
6. Early stopping for Gradient Descent.
7. Mini-Batch Gradient Descent.
8. Stochastic Gradient Descent.
9. Momentum-based optimization.
10. Adaptive learning-rate methods such as Adam for more advanced optimization experiments.
11. Feature importance and coefficient analysis.
12. Residual plots.
13. Confidence intervals and statistical analysis.
14. Multiple random train-test splits.
15. Hyperparameter sensitivity analysis.

---

# 35. Reproducibility

The project uses `random_state=42` for the train-test split.

This means the same split can be reproduced when the same software versions and dataset are used.

To reproduce the experiment:

```bash
pip install -r requirements.txt
python main.py
```

The generated files are:

```text
results/model_results.csv
results/gradient_descent_results.csv

graphs/actual_vs_predicted.png
graphs/model_comparison.png
graphs/gradient_descent_cost.png
graphs/learning_rate_vs_cost.png
graphs/learning_rate_iterations.png
```

---

# 36. GitHub Upload Instructions

## Step 1 — Create Repository

Go to GitHub and create a new repository named:

```text
Regression-Gradient-Descent
```

Do not create unnecessary files if you already have the project locally.

## Step 2 — Open Terminal in Project Folder

```bash
cd Regression_Gradient_Descent
```

## Step 3 — Initialize Git

```bash
git init
```

## Step 4 — Add Files

```bash
git add .
```

## Step 5 — Commit

```bash
git commit -m "Regression models and Gradient Descent implementation"
```

## Step 6 — Set Main Branch

```bash
git branch -M main
```

## Step 7 — Connect GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/Regression-Gradient-Descent.git
```

## Step 8 — Push

```bash
git push -u origin main
```

After the push, refresh your GitHub repository and verify that the folders and files are visible.

---

# 37. Recommended GitHub Repository Appearance

The repository should display approximately:

```text
Regression-Gradient-Descent
│
├── data
│   └── diabetes_regression.csv
│
├── graphs
│   ├── actual_vs_predicted.png
│   ├── gradient_descent_cost.png
│   ├── learning_rate_iterations.png
│   ├── learning_rate_vs_cost.png
│   └── model_comparison.png
│
├── results
│   ├── gradient_descent_results.csv
│   └── model_results.csv
│
├── src
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── models.py
│   └── visualization.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

This organization makes it easy for an evaluator to find the code, data, results, graphs, and documentation.

---

# 38. Screenshots to Include for the Academic Submission

For the final report/PDF/PPT, take screenshots of:

### Screenshot 1 — Dataset

Show:

```text
data/diabetes_regression.csv
```

or the first rows of the dataset.

### Screenshot 2 — Code Structure

Show the VS Code project structure.

### Screenshot 3 — Successful Execution

Show:

```text
python main.py
```

with the generated results visible.

### Screenshot 4 — Regression Results

Show the model metrics table.

### Screenshot 5 — Model Comparison Graph

Show:

```text
graphs/model_comparison.png
```

### Screenshot 6 — Gradient Descent Cost Curve

Show:

```text
graphs/gradient_descent_cost.png
```

### Screenshot 7 — Learning Rate Analysis

Show:

```text
graphs/learning_rate_vs_cost.png
```

### Screenshot 8 — GitHub Repository

Show the uploaded repository containing the code, data, results, graphs and README.

---

# 39. Rubric Mapping

The project is specifically structured around the supplied 10-mark rubric.

## Criterion 1 — Model Selection & Application — 2.5 Marks

Evidence provided:

- Real-world regression problem.
- Four regression approaches.
- Explanation of why each model was selected.
- Mathematical foundations.
- Data preprocessing methodology.
- Explanation of Gradient Descent.
- Application of each technique to the selected problem.

The README documents the relationship between the problem and each model rather than simply listing algorithms.

---

## Criterion 2 — Implementation, Output Quality & Analysis — 2.5 Marks

Evidence provided:

- Modular Python implementation.
- Dataset included in repository.
- Reproducible train-test split.
- Appropriate preprocessing.
- MAE, MSE, RMSE and R² calculations.
- Custom Gradient Descent implementation.
- Multiple learning-rate experiments.
- CSV result files.
- Automatically generated graphs.
- Successful terminal output.

---

## Criterion 3 — Critical Analysis & Evaluation — 2.5 Marks

Evidence provided:

- Model-to-model comparison.
- Metric interpretation.
- Polynomial model analysis.
- Regularization analysis.
- Learning-rate analysis.
- Cost-function interpretation.
- Discussion of training objective versus test performance.
- Limitations of a single train-test split.
- Future experimental improvements.

The analysis avoids treating one experimental result as a universal conclusion.

---

## Criterion 4 — Professionalism, Creativity, Communication & Reflection — 2.5 Marks

Evidence provided:

- Professional repository structure.
- Detailed README.
- Modular source code.
- Automatically generated visualizations.
- Experimental CSV outputs.
- Reproducibility instructions.
- GitHub upload instructions.
- Limitations and future scope.
- Reflection through interpretation of model and optimization behavior.

---

# 40. Reflection

This project demonstrates that implementing a machine-learning algorithm is not limited to calling a library function. A complete regression experiment requires understanding the problem, selecting appropriate models, preparing data correctly, defining evaluation metrics, interpreting results, and identifying limitations.

The Gradient Descent experiment was particularly useful because it demonstrates how model parameters can be learned iteratively. The learning-rate experiment also shows that optimization behavior and predictive generalization are not exactly the same concept.

The project also demonstrates why model complexity should be justified experimentally. Polynomial Regression added additional nonlinear terms, but the observed test results did not improve compared with the simpler models. Similarly, regularization produced only small changes under the selected hyperparameters.

Therefore, the main learning outcome is not simply identifying a model with a particular score. It is understanding the complete workflow from data preparation to model evaluation and critical interpretation.

---

# 41. Conclusion

This project developed and evaluated multiple regression models for a real-world continuous prediction problem and implemented Gradient Descent optimization for Linear Regression.

Linear Regression was used as the baseline, Polynomial Regression was used to investigate nonlinear feature expansion, Ridge Regression was used to study L2 regularization, and Lasso Regression was used to study L1 regularization and sparsity.

MAE, MSE, RMSE and R² were used to evaluate predictive performance. In the included experimental run, Ridge Regression produced a slightly lower RMSE and slightly higher R² than the other tested models, while Polynomial Regression produced weaker test metrics in the selected degree-2 configuration.

Gradient Descent was implemented from scratch and evaluated with several learning rates. The results demonstrate the relationship between learning rate, optimization cost, and test prediction metrics.

Overall, the project demonstrates a complete machine-learning regression workflow and provides experimental evidence, visualization, implementation details, critical analysis, and reproducibility information suitable for academic evaluation.

---

# 42. References

1. Scikit-learn Documentation — Regression, preprocessing, metrics and datasets.
2. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research, 2011.
3. Efron, Hastie, Johnstone and Tibshirani, *Least Angle Regression*, Annals of Statistics, 2004.
4. Géron, Aurélien, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, O'Reilly.
5. NumPy Documentation.
6. Pandas Documentation.
7. Matplotlib Documentation.

---

# 43. Academic Integrity Note

This repository is intended as an academic implementation and learning project. Experimental values shown in this README correspond to the included implementation and dataset configuration. If the code, random split, hyperparameters, or software environment is changed, the results should be regenerated and reported from the new execution.

The repository should be submitted together with the student's own explanation, screenshots, presentation/report, and understanding of the implementation.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Then check:

```text
results/
graphs/
```

for the generated experimental evidence.
