# Data Science & Machine Learning Portfolio

This repository contains several independent analytics and modeling projects spanning deep learning, predictive modeling, image/audio classification, and statistical analysis.

## Projects

### Bird Sound Classification (CNNs)
- Classifies bird species from mel‑spectrograms.
- Binary and 12‑class models using SimpleCNN and DeepCNN.
- External MP3 inference via sliding‑window spectrograms.

### ACT Score Analysis (Education)
- Examines socioeconomic and funding predictors of ACT performance.
- Builds full and reduced regression models.
- Finds socioeconomic factors outperform funding variables.

### Election Results Prediction
- Predicts state‑level presidential outcomes (2000–2024).
- Random Forest achieves 83% national accuracy.
- Key predictors: turnout, unemployment, median income.

### Flower Image Classification (ResNet34)
- Five‑class flower classifier using transfer learning.
- Confusion‑matrix interpretation and model diagnostics.
- Strong overall accuracy with class‑specific weaknesses.

### Coronary Heart Disease Prediction (SVMs)
- SVM models (Linear, RBF, Polynomial) on NHIS 2022.
- Polynomial achieves highest accuracy; Linear best sensitivity.
- Strong predictors: age, physical activity, alcohol use.

### Weather Comparison: Seattle vs. Charlotte
- NOAA precipitation analysis (2018–2022).
- Seattle wetter in winter; Charlotte wetter in summer.
- Includes seasonal, monthly, and annual comparisons.

### Youth Marijuana Use Modeling (Decision Trees & Ensembles)
- Binary, multi‑class, and regression tasks on NSDUH 2023 youth data.
- Random Forest and Boosting outperform single trees.
- Peer norms and peer behavior are strongest predictors.

## Tooling
- Python (pandas, numpy, scikit‑learn, matplotlib, seaborn)
- PyTorch / fast.ai
- R (tidyverse, tree, randomForest, gbm, ROSE)
- Jupyter / RMarkdown
- NOAA, NHIS, NSDUH, Census, FRED datasets