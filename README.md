# Regression Models and Gradient Descent for Disease Progression Prediction

## 1. Project Overview

This project implements and evaluates multiple **regression models** for a real-world machine learning application: **predicting a quantitative measure of disease progression** from baseline patient-related measurements.

The project also implements **Gradient Descent from scratch for Linear Regression** and studies how the learning rate affects optimization, convergence behavior, and test-set performance.

The work is designed to demonstrate both the **practical use of regression models** and the **mathematical optimization process behind Linear Regression**.

---

## 2. Assignment Objectives

The project addresses two main objectives:

### Objective 1
Develop regression models for a real-world application and evaluate their performance using suitable regression metrics.

### Objective 2
Implement Gradient Descent optimization for Linear Regression and analyze how different learning rates affect the training process and predictive performance.

The implementation therefore combines:

- Data preprocessing
- Multiple regression techniques
- Model training and prediction
- Regression evaluation metrics
- Custom Batch Gradient Descent
- Experimental comparison
- Visualization of model and optimization behavior

---

## 3. Problem Statement

In many real-world applications, the goal is not to predict a simple yes/no outcome but to estimate a **continuous numerical value**. Regression is appropriate for these tasks because it learns a relationship between input variables and a continuous target.

In this project, the target represents a **disease progression measurement**. The model receives several baseline measurements as input and estimates the progression value.

The central question is:

> How effectively can different regression models predict disease progression, and how does Gradient Descent behave when it is used to optimize the parameters of Linear Regression?

This problem is useful for demonstrating model comparison because different regression techniques make different assumptions about the relationship between the features and the target.

---

## 4. Application Context

The project uses a disease progression prediction dataset containing numerical baseline measurements. The dataset is commonly used for demonstrating regression methods and contains:

- Age-related information
- Sex-related information
- Body-mass-related measurement
- Blood-pressure-related measurement
- Six blood-serum measurements
- A continuous target representing disease progression

The goal here is **prediction and algorithm evaluation**, not clinical diagnosis. The models should be interpreted as machine-learning experiments and not as medical decision-making systems.

---

## 5. Dataset Description

The dataset used in this repository is stored in:

```text
 data/diabetes_regression.csv
```

### Dataset size

- Total records: **442**
- Input features: **10**
- Target column: **1**
- Total columns: **11**
- Training samples: **353**
- Testing samples: **89**
- Train-test split: **80% / 20%**
- Random state: **42**

### Columns

| Column | Description |
|---|---|
| `age` | Age-related baseline measurement |
| `sex` | Sex-related baseline measurement |
| `bmi` | Body Mass Index related measurement |
| `bp` | Blood-pressure related measurement |
| `s1` | Serum measurement 1 |
| `s2` | Serum measurement 2 |
| `s3` | Serum measurement 3 |
| `s4` | Serum measurement 4 |
| `s5` | Serum measurement 5 |
| `s6` | Serum measurement 6 |
| `target` | Continuous disease progression value to predict |

The feature values are already numerically encoded and standardized in the source dataset. The implementation also performs training-set-based scaling so that the optimization algorithms operate under consistent numerical conditions.

### Data quality check

The dataset contains **no missing values** in the stored CSV used by this experiment.

---

## 6. Why Regression?

The target variable is continuous rather than a class label. Therefore, the problem is formulated as a **supervised regression problem**.

A regression model attempts to learn a function:

\[
\hat{y}=f(X)
\]

where:

- \(X\) = input features
- \(y\) = actual target
- \(\hat{y}\) = predicted target

The quality of a regression model is measured by how close the predicted values are to the actual values.

---

## 7. Project Workflow

The complete workflow is:

```text
Dataset
   |
   v
Data Loading
   |
   v
Data Quality Check
   |
   v
Train/Test Split
   |
   v
Feature Scaling
   |
   +-------------------------------+
   |                               |
   v                               v
Classical Regression Models   Custom Gradient Descent
   |                               |
   |                               +--> Multiple Learning Rates
   |                               |
   +---------------+---------------+
                   |
                   v
              Predictions
                   |
                   v
           Evaluation Metrics
                   |
                   v
        CSV Results + Graphs
                   |
                   v
        Comparative Analysis
```

---

# 8. Regression Models Implemented

Four regression approaches are included in the experiment:

1. Linear Regression
2. Polynomial Regression of Degree 2
3. Ridge Regression
4. Lasso Regression

In addition, a custom implementation called **LinearRegressionGD** is used to train Linear Regression with Batch Gradient Descent.

---

## 9. Model 1 – Linear Regression

### Concept

Linear Regression assumes that the target can be approximated by a linear combination of the input features.

For multiple features:

\[
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
\]

where:

- \(b_0\) = intercept
- \(b_1,\ldots,b_n\) = coefficients
- \(x_1,\ldots,x_n\) = input features
- \(\hat y\) = predicted value

### Why it is included

Linear Regression provides a strong and interpretable **baseline model**. It gives a reference point against which more complex models can be compared.

### Strengths

- Simple to understand
- Fast to train
- Easy to interpret
- Suitable for a continuous target
- Useful as a baseline

### Limitations

- Assumes a linear relationship
- May not capture complex nonlinear patterns
- Can be sensitive to multicollinearity
- May underfit a nonlinear dataset

### Implementation

The project uses `sklearn.linear_model.LinearRegression` for the standard Linear Regression experiment.

---

## 10. Model 2 – Polynomial Regression

### Concept

Polynomial Regression extends a linear model by adding polynomial combinations of the input features.

For a simple one-variable example:

\[
\hat y=b_0+b_1x+b_2x^2
\]

For multiple variables, degree-2 polynomial features can include squared terms and pairwise interaction terms.

### Why degree 2?

Degree 2 is selected as a controlled increase in model flexibility. It allows nonlinear relationships to be tested without making the feature space unnecessarily large.

### Strengths

- Can represent nonlinear relationships
- More flexible than ordinary Linear Regression
- Easy to combine with a linear estimator

### Limitations

- Feature count can grow quickly
- More flexible models can overfit
- Extra features can make the model harder to interpret
- Higher-degree polynomials may become numerically unstable or overly complex

### Implementation

The project uses:

```text
PolynomialFeatures(degree=2)
        +
LinearRegression()
```

inside a scikit-learn pipeline.

---

## 11. Model 3 – Ridge Regression

### Concept

Ridge Regression is Linear Regression with **L2 regularization**.

A simplified objective is:

\[
J(\theta)=MSE+\lambda\sum_j\theta_j^2
\]

The additional penalty discourages very large coefficients.

### Why it is included

Ridge is useful when predictors may be correlated or when a model needs some control against coefficient magnitude and overfitting.

### Strengths

- Reduces coefficient magnitude
- Often improves numerical stability
- Can help when features are correlated
- Usually retains all features

### Limitation

Unlike Lasso, Ridge generally does not force coefficients exactly to zero, so it does not directly perform sparse feature selection.

### Implementation

The project uses:

```text
Ridge(alpha=1.0)
```

The value of `alpha` controls the strength of regularization.

---

## 12. Model 4 – Lasso Regression

### Concept

Lasso Regression uses **L1 regularization**:

\[
J(\theta)=MSE+\lambda\sum_j|\theta_j|
\]

The absolute-value penalty encourages some coefficients to become exactly or approximately zero.

### Why it is included

Lasso allows the experiment to study a different regularization strategy from Ridge and to observe how sparse solutions can be produced.

### Strengths

- Can reduce the effect of irrelevant features
- Can produce sparse coefficient vectors
- Provides an alternative to Ridge regularization

### Limitations

- Can be sensitive to regularization strength
- With strongly correlated predictors, Lasso may select one predictor and reduce others
- Too much regularization can cause underfitting

### Implementation

The project uses:

```text
Lasso(alpha=0.01, max_iter=20000)
```

---

# 13. Model Selection Logic

The models were selected to compare four different levels of modeling behavior:

| Model | Main Idea | Main Purpose in Experiment |
|---|---|---|
| Linear Regression | Direct linear relationship | Baseline and interpretability |
| Polynomial Regression | Nonlinear feature expansion | Test whether added flexibility helps |
| Ridge | L2 regularization | Control coefficient magnitude |
| Lasso | L1 regularization | Regularization and sparse coefficients |

This is more informative than testing several models that all behave in essentially the same way. The experiment compares a simple baseline, nonlinear expansion, and two distinct regularization strategies.

---

# 14. Data Preprocessing

Data preparation has a direct effect on model quality and optimization stability.

## 14.1 Loading data

The CSV file is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("data/diabetes_regression.csv")
```

The target column is separated from the features:

```python
X = df.drop(columns=["target"])
y = df["target"]
```

## 14.2 Train-test split

The data is divided into training and testing subsets:

- 80% training
- 20% testing
- `random_state=42`

The training set is used for learning model parameters, while the test set is kept for final evaluation.

## 14.3 Feature scaling

The project uses `StandardScaler`.

Conceptually:

\[
z=\frac{x-\mu}{\sigma}
\]

where:

- \(\mu\) = training-set mean
- \(\sigma\) = training-set standard deviation

The scaler is fitted only on the training data and then applied to the test data. This avoids using test-set statistics during training.

### Why scaling is especially important for Gradient Descent

Gradient Descent updates all parameters according to their gradients. If one feature has values on a much larger scale than another, the optimization landscape can become poorly conditioned and parameter updates can behave unevenly.

Scaling helps create a more numerically balanced optimization problem.

---

# 15. Evaluation Metrics

The experiment uses several regression metrics because no single metric describes every aspect of predictive performance.

## 15.1 MAE – Mean Absolute Error

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

MAE measures the average absolute difference between actual and predicted values.

### Interpretation

Lower MAE means that predictions are, on average, closer to the actual target.

### Advantage

MAE is easy to understand and is less dominated by very large errors than MSE.

---

## 15.2 MSE – Mean Squared Error

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

MSE squares every prediction error before averaging.

### Interpretation

Large errors receive disproportionately high penalties.

### Advantage

MSE is useful for optimization because it is differentiable and therefore works naturally with Gradient Descent.

---

## 15.3 RMSE – Root Mean Squared Error

\[
RMSE=\sqrt{MSE}
\]

RMSE is expressed in the same units as the target.

### Interpretation

A lower RMSE means better predictive agreement with the actual target, while larger errors are still penalized strongly.

---

## 15.4 R² – Coefficient of Determination

\[
R^2=1-\frac{SS_{res}}{SS_{tot}}
\]

R² describes how much variation in the target is explained by the model relative to a baseline based on the mean target.

A higher R² is generally associated with stronger explanatory performance on the evaluated dataset.

### Important note

R² should not be viewed alone. A model can have a similar R² to another model while showing different error behavior under MAE or RMSE.

---

# 16. Gradient Descent – Core Idea

Gradient Descent is an iterative optimization algorithm used to minimize a differentiable cost function.

Instead of directly solving for parameters in one operation, Gradient Descent repeatedly moves the parameters in the direction that reduces the cost.

The basic update rule is:

\[
\theta := \theta-\alpha\nabla J(\theta)
\]

where:

- \(\theta\) = parameter vector
- \(\alpha\) = learning rate
- \(\nabla J(\theta)\) = gradient of the cost function

The direction of the negative gradient points toward decreasing cost.

---

# 17. Linear Regression Cost Function for Gradient Descent

For the custom implementation, the model uses Batch Gradient Descent.

The feature matrix is augmented with a column of ones to represent the intercept:

\[
X_b=[1\ \ X]
\]

Predictions are calculated as:

\[
\hat y=X_b\theta
\]

The error vector is:

\[
e=\hat y-y
\]

The project uses the following half-MSE-style cost:

\[
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y_i-y_i)^2
\]

The factor \(1/2\) simplifies the derivative.

---

# 18. Gradient Calculation

The vectorized gradient used in the program is:

\[
\nabla J(\theta)=\frac{1}{m}X_b^T(X_b\theta-y)
\]

The parameter update becomes:

\[
\theta:=\theta-\alpha\frac{1}{m}X_b^T(X_b\theta-y)
\]

This formula is implemented directly using NumPy matrix operations.

---

# 19. Gradient Descent Algorithm Used in This Project

The custom implementation follows these steps:

```text
1. Receive scaled training features X and target y.
2. Add an intercept column of ones.
3. Initialize all parameters to zero.
4. Repeat for a fixed number of iterations:
      a. Calculate predictions.
      b. Calculate prediction errors.
      c. Calculate the cost.
      d. Calculate the gradient.
      e. Update the parameters.
      f. Store the cost in cost_history.
5. Return the learned parameters and training history.
6. Use the learned parameters to predict unseen test samples.
```

The implementation is a **Batch Gradient Descent** method because the gradient is computed using the complete training set in every iteration.

---

# 20. Why Batch Gradient Descent?

Batch Gradient Descent was selected because it is easy to understand mathematically and provides a stable learning curve for this relatively small dataset.

### Advantages

- Uses the full training set for each update
- Produces deterministic updates for a fixed dataset and starting point
- Gives a smooth cost trajectory
- Makes the relationship between the learning rate and convergence easy to study

### Limitation

For very large datasets, processing all training examples in every iteration can become computationally expensive. Mini-batch or stochastic methods are often more suitable for large-scale training.

---

# 21. Target Scaling in Gradient Descent

For the custom Gradient Descent experiment, the training target is standardized before optimization:

\[
y_{scaled}=\frac{y-y_{mean}}{y_{std}}
\]

The model therefore optimizes a numerically normalized target.

After prediction, the values are converted back to the original target scale:

\[
y_{original}=y_{scaled}\times y_{std}+y_{mean}
\]

### Why this is done

Scaling the target makes the magnitude of the loss and gradients easier to manage and helps the learning-rate experiments remain numerically stable.

### Important interpretation detail

The `Final Cost` reported in `gradient_descent_results.csv` is calculated on the **scaled training target**, while MAE and RMSE are calculated after transforming predictions back to the **original target scale**. Therefore, Final Cost should not be compared directly with the RMSE value as if they were the same metric.

---

# 22. Learning Rate Experiment

Four learning rates were tested:

```text
0.001
0.01
0.05
0.1
```

Each experiment uses:

```text
3000 iterations
```

The purpose is to observe how the learning rate changes:

- The final training cost
- The optimization trajectory
- Test-set MAE
- Test-set RMSE
- Test-set R²

---

# 23. Learning Rate Behavior

### Very small learning rate

A small learning rate produces small parameter updates. The optimization may move toward the minimum slowly.

### Moderate learning rate

A well-selected learning rate can reduce cost efficiently while maintaining stable updates.

### Large learning rate

A learning rate that is too large can overshoot the minimum, oscillate, or diverge depending on the optimization problem.

The exact behavior depends on feature scaling, the dataset, initialization, the cost function, and the number of iterations.

---

# 24. Why Multiple Metrics Are Needed

Consider two models:

- Model A may have slightly lower MAE.
- Model B may have slightly lower RMSE.
- Model C may have a slightly higher R².

Those outcomes describe different aspects of performance.

Therefore, the experiment reports all four major metrics instead of selecting a model from a single number.

The final interpretation should consider:

1. Average error magnitude
2. Sensitivity to large errors
3. Explained variance
4. Model complexity
5. Optimization behavior

---

# 25. Experimental Results – Regression Models

The recorded test-set results from the included experiment are:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 42.7941 | 2900.1936 | 53.8534 | 0.4526 |
| Polynomial Regression (Degree 2) | 43.5817 | 3096.0283 | 55.6420 | 0.4156 |
| Ridge Regression | 42.8120 | 2892.0146 | 53.7775 | 0.4541 |
| Lasso Regression | 42.7950 | 2898.3680 | 53.8365 | 0.4529 |

> Timing and numerical results can change slightly if the experiment is rerun in a different software environment or with changed preprocessing parameters. The table above records the results produced with the included configuration.

---

# 26. Interpretation of Regression Results

## Linear Regression

Linear Regression provides a baseline R² of approximately **0.4526** with an MAE of approximately **42.79**.

This shows that a linear model captures a meaningful portion of the relationship in the dataset, but a substantial amount of variation remains unexplained.

## Polynomial Regression

The degree-2 polynomial model records:

- Higher MAE
- Higher MSE
- Higher RMSE
- Lower R²

than the basic Linear Regression in this experiment.

This means that adding degree-2 polynomial terms did **not** improve generalization on the test set for this configuration.

A more complex feature representation does not automatically result in better predictive performance. The additional flexibility can increase variance or fit patterns that do not generalize to unseen data.

## Ridge Regression

Ridge achieves the lowest RMSE in the model-comparison table and the highest R² among the four standard regression models in this particular run:

- RMSE ≈ **53.78**
- R² ≈ **0.4541**

The improvement over ordinary Linear Regression is small rather than dramatic. This suggests that regularization provides a modest change for this dataset and parameter setting.

## Lasso Regression

Lasso performs very close to Linear Regression:

- MAE ≈ **42.7950**
- RMSE ≈ **53.8365**
- R² ≈ **0.4529**

The results suggest that, with `alpha=0.01`, L1 regularization changes the model only slightly relative to the baseline.

---

# 27. Important Comparative Observation

The results demonstrate that **greater model complexity is not automatically better**.

The degree-2 Polynomial Regression model has more representational flexibility but produces worse test-set performance than the simpler Linear Regression model in this experiment.

At the same time, Ridge changes the objective through regularization and produces a small improvement in RMSE and R².

This is an important machine-learning principle:

> Model complexity should be justified by generalization performance, not by complexity alone.

---

# 28. Experimental Results – Gradient Descent

The learning-rate experiments produced the following results:

| Learning Rate | Iterations | Final Cost | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 3000 | 0.238234 | 42.994994 | 53.604864 | 0.457645 |
| 0.01 | 3000 | 0.236856 | 42.865809 | 53.722332 | 0.455265 |
| 0.05 | 3000 | 0.235534 | 42.815545 | 53.785936 | 0.453974 |
| 0.1 | 3000 | 0.235382 | 42.799299 | 53.834207 | 0.452994 |

---

# 29. Interpretation of Gradient Descent Results

## Effect on training cost

The recorded final training cost decreases as the learning rate increases from 0.001 to 0.1 in this experiment:

```text
0.001  ->  0.238234
0.01   ->  0.236856
0.05   ->  0.235534
0.1    ->  0.235382
```

This indicates that, under the fixed 3000-iteration budget and current scaling, the larger learning rates moved the parameters closer to a lower training objective within the allotted iterations.

## Effect on test performance

The test results show a different pattern. The learning rate of 0.001 has the highest R² and lowest RMSE among the four Gradient Descent runs shown above.

This is an important observation because **lower training cost does not automatically mean better test-set performance**.

A model is ultimately evaluated on unseen data, and optimization quality and generalization are related but not identical objectives.

## What can be concluded

The tested learning rates are all stable under the current configuration. None of the recorded runs shows divergence in the final reported cost.

However, the experiment also shows why learning-rate selection is an optimization problem: different rates can reach different parameter states within a fixed iteration budget, and the resulting test errors can vary.

---

# 30. Gradient Descent Cost Curve

The graph:

```text
 graphs/gradient_descent_cost.png
```

plots the cost across iterations for the main demonstration learning rate (`0.05`).

### Purpose of the graph

The cost curve visually demonstrates whether the optimization process is moving toward lower error.

A typical successful curve has a rapidly decreasing section followed by a flatter region as the parameters approach a region of lower cost.

### How to interpret it

- Steep downward movement: parameters are learning quickly.
- Gradual downward movement: optimization continues but improvement is slower.
- Flat curve: the algorithm may be close to convergence or the learning rate may be too small for further practical progress.
- Oscillating or increasing curve: the learning rate or numerical setup may be unsuitable.

---

# 31. Actual vs Predicted Graph

The file:

```text
 graphs/actual_vs_predicted.png
```

shows the relationship between actual test targets and model predictions.

A strong prediction model should place predicted points close to the ideal relationship between actual and predicted values.

The plot is useful because numerical metrics alone do not show:

- Whether errors grow for larger target values
- Whether predictions are systematically too high or too low
- Whether there are clusters of difficult observations
- Whether the model compresses extreme values toward the center

The graph therefore complements MAE, RMSE, and R².

---

# 32. Model Comparison Graph

The file:

```text
 graphs/model_comparison.png
```

compares the performance metrics of the four classical regression models.

The visualization allows the user to quickly inspect differences that may be difficult to notice in raw numbers.

For MAE, MSE, and RMSE, lower values indicate smaller prediction error.

For R², higher values indicate stronger explained variation on the evaluated test set.

---

# 33. Learning Rate vs Cost Graph

The file:

```text
 graphs/learning_rate_vs_cost.png
```

compares the final Gradient Descent cost across learning rates.

In the recorded experiment, the final cost gradually decreases as the tested learning rate increases from 0.001 to 0.1.

However, this does not mean that increasing the learning rate indefinitely will continue to improve optimization. Beyond a stable range, large updates can overshoot a minimum and lead to oscillation or divergence.

---

# 34. Learning Rate and Iteration Comparison

The file:

```text
 graphs/learning_rate_iterations.png
```

visualizes the iteration budget associated with each learning-rate experiment.

All four experiments use 3000 iterations in the current implementation. This controlled setup makes learning rate the main experimental variable rather than changing both learning rate and iteration count at the same time.

---

# 35. Implementation Architecture

The project is split into small modules so that each responsibility is separated.

```text
main.py
  |
  +--> data_preprocessing.py
  |
  +--> models.py
  |
  +--> evaluation.py
  |
  +--> visualization.py
```

### `main.py`

Coordinates the entire experiment:

- Loads and prepares data
- Trains the four standard regression models
- Evaluates predictions
- Runs Gradient Descent experiments
- Saves result CSV files
- Generates graphs

### `src/data_preprocessing.py`

Responsible for:

- Loading the dataset
- Separating features and target
- Train-test splitting
- Standardization

### `src/models.py`

Contains:

- Standard regression model definitions
- Custom `LinearRegressionGD` class
- Gradient Descent training logic
- Prediction logic
- Metric helper

### `src/evaluation.py`

Handles model evaluation and result storage.

### `src/visualization.py`

Generates the project graphs automatically.

---

# 36. Project Structure

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
│   ├── gradient_descent_results.csv
│   └── complete_output.txt
│
├── graphs/
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   ├── gradient_descent_cost.png
│   ├── learning_rate_vs_cost.png
│   └── learning_rate_iterations.png
│
├── screenshots/
│   └── Add your execution screenshots here
│
├── main.py
├── requirements.txt
├── .gitignore
├── GITHUB_UPLOAD_GUIDE.md
└── README.md
```

---

# 37. Software Requirements

The project requires:

- Python 3.10+ recommended
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# 38. Running the Project

Open a terminal in the project folder:

```bash
cd Regression_Gradient_Descent
```

Install the packages:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python main.py
```

The program automatically creates or updates:

```text
results/model_results.csv
results/gradient_descent_results.csv
results/complete_output.txt
```

and the five PNG graphs inside the `graphs/` directory.

---

# 39. Expected Program Output

The included experiment reports:

```text
========================================================================
REGRESSION MODELS + GRADIENT DESCENT EXPERIMENT
Real-world application: Disease Progression Prediction
========================================================================
Dataset shape: 442 rows x 11 columns
Training samples: 353
Testing samples : 89

REGRESSION MODEL RESULTS
                           Model     MAE       MSE    RMSE     R2
               Linear Regression 42.7941 2900.1936 53.8534 0.4526
Polynomial Regression (Degree 2) 43.5817 3096.0283 55.6420 0.4156
                Ridge Regression 42.8120 2892.0146 53.7775 0.4541
                Lasso Regression 42.7950 2898.3680 53.8365 0.4529

GRADIENT DESCENT RESULTS
 Learning Rate  Iterations  Final Cost       MAE      RMSE       R2
      0.001000        3000    0.238234 42.994994 53.604864 0.457645
      0.010000        3000    0.236856 42.865809 53.722332 0.455265
      0.050000        3000    0.235534 42.815545 53.785936 0.453974
      0.100000        3000    0.235382 42.799299 53.834207 0.452994
```

These values are generated by the included code and correspond to the current dataset split and configuration.

---

# 40. Critical Discussion of the Experiment

## 40.1 Linear and Ridge are very close

The difference between Linear Regression and Ridge Regression is small. This suggests that the regularization penalty with `alpha=1.0` does not radically change the fitted solution for this dataset.

The small performance difference is still useful because it shows that regularization can affect the optimization objective without necessarily producing a dramatic change in prediction error.

## 40.2 Polynomial Regression did not help

The polynomial model has more features and more flexibility, but its test RMSE is higher and its R² is lower than the simpler models.

This is a useful experimental demonstration that increasing complexity can hurt generalization.

## 40.3 Lasso remains close to the baseline

Lasso produces almost the same performance as Linear Regression for the selected regularization value. A stronger `alpha` could create greater shrinkage, but that would be a new experiment and should be reported separately rather than assumed to be better.

## 40.4 Gradient Descent produces a competitive solution

The custom optimizer produces test metrics in the same general range as the standard Linear Regression baseline.

That demonstrates that the model parameters can be learned iteratively by minimizing the cost function rather than relying only on a library estimator.

## 40.5 Training objective and generalization are different

The learning rate of 0.1 achieves the lowest recorded final training cost among the tested rates, but the 0.001 run has the best test RMSE and R² in this particular experiment.

This distinction is important:

> Optimization minimizes the chosen training objective, while model evaluation measures how well the learned parameters generalize to unseen data.

The two goals are related, but they are not identical.

---

# 41. Complexity and Practical Trade-offs

| Method | Main Benefit | Main Cost / Limitation |
|---|---|---|
| Linear Regression | Simple and interpretable | Limited nonlinear modeling |
| Polynomial Regression | Captures nonlinear patterns | More features and overfitting risk |
| Ridge | Regularization and stability | Requires choosing `alpha` |
| Lasso | Regularization + sparsity | Can suppress correlated features |
| Gradient Descent | Shows optimization process and scalable idea | Requires learning-rate and iteration tuning |

---

# 42. Important Concepts Demonstrated

This project demonstrates several machine-learning principles.

### Bias and variance

A simple model may have higher bias and fail to capture useful structure. A more flexible model can reduce bias but may increase variance and overfit.

### Regularization

Ridge and Lasso modify the learning objective to discourage overly complex coefficient values.

### Generalization

Training performance alone is insufficient. Test-set performance is needed to estimate how the model behaves on unseen samples.

### Optimization

Gradient Descent provides an iterative strategy for minimizing an objective function.

### Hyperparameters

Learning rate, number of iterations, polynomial degree, and regularization strength affect model behavior and must be chosen deliberately.

---

# 43. Limitations

The current experiment has several limitations.

### Dataset size

The dataset contains 442 records, which is suitable for an academic demonstration but small compared with many industrial machine-learning problems.

### Single train-test split

The current results are based on one 80/20 split with `random_state=42`. A different split may produce different metrics.

### Limited hyperparameter search

Only one degree is tested for Polynomial Regression, and one regularization value is used for Ridge and Lasso.

### Fixed number of iterations

The Gradient Descent experiments use 3000 iterations for every learning rate. The implementation does not currently stop automatically when a convergence tolerance is reached.

### Limited feature engineering

No advanced feature construction or domain-specific feature engineering is performed.

### Synthetic interpretation risk

Although the dataset represents a real application domain, this project is an academic machine-learning experiment and should not be interpreted as a deployable clinical prediction system.

---

# 44. Future Improvements

Several improvements can make the project more advanced.

## Cross-validation

Use k-fold cross-validation to estimate performance across multiple train-validation splits instead of relying on a single split.

## Hyperparameter tuning

Search over:

- Ridge `alpha`
- Lasso `alpha`
- Polynomial degree
- Gradient Descent learning rate
- Number of iterations

## Early stopping

Stop Gradient Descent when the reduction in cost becomes smaller than a chosen tolerance.

Example condition:

\[
|J_t-J_{t-1}|<\epsilon
\]

## Mini-batch Gradient Descent

For larger datasets, mini-batches can reduce the cost of computing every update on the full training dataset.

## Learning-rate schedules

The learning rate can be reduced during training to combine rapid initial progress with stable final optimization.

## Feature analysis

Coefficient magnitude, permutation importance, or other explanatory methods could be used to investigate which input variables contribute most strongly to predictions.

## Error analysis

Residual plots can be added to investigate whether the model makes systematic errors for particular target ranges.

---

# 45. Reproducibility

The experiment is configured for reproducibility using:

```text
train_test_split(..., random_state=42)
```

This ensures that the same train-test partition is generated when the same dataset and software setup are used.

For exact reproduction of numerical results, use the same:

- Dataset file
- Python version
- Package versions
- Preprocessing steps
- Random state
- Model hyperparameters
- Learning rates
- Number of iterations

Small differences can still occur because of environment and dependency changes.

---

# 46. Generated Outputs

After running `python main.py`, the repository contains two major classes of outputs.

## Numerical outputs

```text
results/model_results.csv
results/gradient_descent_results.csv
results/complete_output.txt
```

These files make the experiment easy to inspect and reuse for report tables.

## Visual outputs

```text
graphs/actual_vs_predicted.png
graphs/model_comparison.png
graphs/gradient_descent_cost.png
graphs/learning_rate_vs_cost.png
graphs/learning_rate_iterations.png
```

These visualizations provide evidence of the experimental results and optimization behavior.

---

# 47. How the Code Works Internally

The most important custom class is:

```python
class LinearRegressionGD:
```

Its `fit()` method performs the optimization.

Conceptually:

```python
Xb = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(Xb.shape[1])

for _ in range(n_iterations):
    predictions = Xb @ theta
    errors = predictions - y
    cost = np.mean(errors ** 2) / 2.0
    gradient = (Xb.T @ errors) / m
    theta -= learning_rate * gradient
```

This compact block contains the essential mechanics of Gradient Descent:

1. Prediction
2. Error calculation
3. Cost calculation
4. Gradient calculation
5. Parameter update

The learned `theta` values are then used in `predict()`.

---

# 48. Why NumPy is Used for Gradient Descent

NumPy allows the implementation to express Gradient Descent with vectorized matrix operations instead of slow Python loops over every feature and every example.

For example:

```python
predictions = Xb @ theta
```

performs matrix-vector multiplication, while:

```python
gradient = (Xb.T @ errors) / m
```

computes all parameter gradients at once.

This makes the implementation concise, mathematically transparent, and computationally efficient for a dataset of this size.

---

# 49. Why Scikit-learn is Used

Scikit-learn is used for the standard regression estimators and common preprocessing utilities.

This has two benefits:

1. It provides reliable reference implementations for comparison.
2. It allows the custom Gradient Descent implementation to be evaluated against a standard Linear Regression workflow.

The project therefore combines **library-based models** with a **from-scratch optimization implementation**.

---

# 50. Ethical and Practical Considerations

Because the application domain is healthcare-related, model outputs should be interpreted carefully.

The project demonstrates regression algorithms and optimization techniques. It does not establish clinical validity, diagnosis capability, treatment recommendations, or safety for real patient use.

A real deployment would require additional work including validated clinical datasets, careful bias assessment, privacy controls, external validation, uncertainty analysis, domain-expert review, and appropriate regulatory and ethical oversight.

---

# 51. Conclusion

This project demonstrates how regression models can be applied to a real-world prediction problem and how their performance can be evaluated using quantitative metrics.

Four regression approaches were compared:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression

The experimental results show that the simpler models perform competitively on this dataset, while the degree-2 Polynomial model does not improve test performance in the current configuration. Ridge provides a small improvement in RMSE and R² compared with ordinary Linear Regression.

The second part of the project implements **Batch Gradient Descent from scratch** for Linear Regression. Multiple learning rates were tested to study optimization behavior. The experiments demonstrate that learning-rate selection affects the final training objective and the resulting test performance.

The project therefore connects the theory of regression with the practical mechanics of optimization:

```text
Regression Model
      ↓
Define Cost Function
      ↓
Compute Gradient
      ↓
Update Parameters
      ↓
Reduce Cost
      ↓
Generate Predictions
      ↓
Evaluate on Unseen Data
```

The main learning outcome is that a good machine-learning solution requires more than training a model. It also requires appropriate preprocessing, suitable metrics, careful optimization, experimental comparison, interpretation of results, and recognition of the limitations of the evidence.

---

# 52. References

1. Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
2. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
3. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
4. Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. *Least Angle Regression*. The Annals of Statistics, 2004.
5. NumPy documentation for numerical linear algebra and array operations.
6. Pandas documentation for data loading and manipulation.
7. Matplotlib documentation for data visualization.

---

## Author / Project Information

**Project:** Regression Models and Gradient Descent Optimization  
**Application:** Disease Progression Prediction  
**Language:** Python  
**Main Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
**Optimization Method:** Batch Gradient Descent  

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The generated results and graphs can then be found in:

```text
results/
graphs/
```
