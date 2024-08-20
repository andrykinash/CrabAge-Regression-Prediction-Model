# Crab Age Prediction using Machine Learning

## Overview

Using A large excel dataset, this project focuses on predicting the age of crabs using advanced regression techniques. We explore and compare different methods, including Stochastic Gradient Descent (SGD), Batch Gradient Descent (BGD), and the Normal Equation. Additionally, Locally Weighted Regression (LWR) is implemented to enhance prediction accuracy.

## Data Preprocessing

- **Label Encoding**: The 'Sex' feature is encoded numerically (`F=0`, `M=1`, `I=2`).
- **Normalization**: Features are standardized to ensure effective convergence of gradient-based algorithms.

## Regression Techniques

- **SGD (Stochastic Gradient Descent)**: Iterative optimization to minimize Mean Squared Error (MSE).
- **BGD (Batch Gradient Descent)**: Utilizes the entire dataset per iteration for stable convergence.
- **Normal Equation**: A closed-form solution providing optimal parameters without iterative optimization.
- **LWR (Locally Weighted Regression)**: Enhances predictions by weighting data points based on proximity to the query point.

## Results

The performance of various regression techniques was evaluated based on the Mean Squared Error (MSE) on a test dataset. The following are the key findings:

- **Stochastic Gradient Descent (SGD)**: 
  - Achieved a low MSE, indicating that the model converged effectively after 100 epochs.
  - **Final MSE**: `0.5546808449666433`

- **Batch Gradient Descent (BGD)**: 
  - Although BGD was stable, it showed higher MSE compared to other methods, suggesting possible overfitting or insufficient convergence.
  - **Final MSE**: `4.762894872529929`

- **Normal Equation**:
  - The Normal Equation provided a precise solution with a relatively low MSE, showing its effectiveness in linear regression problems.
  - **Final MSE**: `0.4917224723626753`

- **Locally Weighted Regression (LWR)**:
  - LWR was particularly effective, with a lower MSE when combined with the Normal Equation, demonstrating its strength in handling localized data variability.
  - **MSE for LWR (SGD)**: `2.29248784024601366`
  - **MSE for LWR (Normal Equation)**: `1.05150789284528`

## Conclusion

The comparison among the different regression methods reveals that while SGD and Normal Equation provide good overall performance, LWR with the Normal Equation offers the best balance between accuracy and localized prediction, making it the most robust approach for this dataset.

## Visualizations

To better understand the performance, visualizations such as cost history plots and MSE bar charts were generated to illustrate the results of each method:

- 
- 
- 


