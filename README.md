# **EV Adoption in Washington State**  
**_Note: The final technical report is included as `WA_EVs_Report.pdf`._**

> Analysis of county‑level EV adoption patterns in Washington State using PCA, clustering, and supervised machine learning models.

---

## **Project Overview**

This project investigates why electric vehicle (EV) adoption varies so widely across Washington State’s 39 counties. Using a combination of unsupervised and supervised learning methods, we identify structural patterns in EV ownership, cluster counties into meaningful archetypes, and build predictive models to determine which factors most strongly influence adoption.

- **Objective:**  
  Understand the drivers of EV adoption across Washington counties and identify county‑level archetypes using machine learning.

- **Domain:**  
  Transportation, Energy, Public Policy, Data Science

- **Key Techniques:**  
  - **Unsupervised Learning:** PCA, SVD, K‑Means, Hierarchical Clustering  
  - **Supervised Learning:** OLS, Ridge, Lasso, Decision Trees, Random Forests, Bagging, Gradient Boosting  
  - **Feature Engineering:** Log transforms, standardization, ratio features  
  - **Model Evaluation:** Cross‑validation, RMSE, R², feature importance

---

## **Project Structure**

```
SML_group_project/
│
├── blog_post/                 # Blog-style summary (Homework 4)
├── code/                      # Homework 4 code (PCA + clustering)
├── data/                      # Homework 4 data
│
└── final project/
    ├── code/                  # Final project code (full pipeline)
    ├── data/                  # Final project data (scaled + log-transformed)
    └── WA_EVs_Report.pdf      # Final technical report
```

---

## **Data**

- **Sources:**  
  - Washington State Department of Licensing (EV registrations)  
  - U.S. Department of Energy Alternative Fuels Station Locator (charging infrastructure)  
  - U.S. Census Bureau ACS (median household income)

- **Description:**  
  - 39 counties  
  - 16 engineered features including EV counts, charger counts, BEV/PHEV ratios, vehicle characteristics, and income  
  - Two processed datasets:  
    - `county_scaled.csv` — standardized features for PCA/clustering  
    - `county_log_transformed.csv` — log‑transformed counts for supervised learning

- **Format:** CSV files  
- **License:** Public government data (open use)

---

## **Analysis**

### **Unsupervised Learning**
- PCA to reduce 16 features → 3 components explaining 81% variance  
- SVD to validate PCA directions  
- K‑Means clustering (k=3) to identify county archetypes  
- Hierarchical clustering (Ward, Complete, Average, Single) for stability checks  
- Cluster profiling using z‑score heatmaps

### **Supervised Learning**
- Target: `total_EV_count`  
- Models: OLS, RidgeCV, LassoCV, Decision Tree, Bagging, Random Forest, Gradient Boosting  
- Removed BEV_count and PHEV_count to avoid data leakage  
- Cross‑validation for hyperparameter tuning  
- Feature importance analysis across tree‑based models

### **Reproducibility**
Run notebooks in this order:

1. **EDA + PCA + Clustering**  
2. **SVD validation**  
3. **Supervised learning models**  
4. **Model comparison + interpretation**

---

## **Results**

- **Three county archetypes** emerged:  
  1. High‑adoption urban counties  
  2. Mid‑range rural counties  
  3. Garfield County as a unique outlier  

- **Gradient Boosting** was the strongest predictive model:  
  - Test R² = **0.895**  
  - RMSE ≈ **21,520**

- **Most important predictors:**  
  - Charging infrastructure (Level 2 chargers, total chargers, DC fast chargers)  
  - Median household income  
  - BEV share and BEV‑to‑PHEV ratio  

- **Key insight:**  
  Different counties face different barriers — infrastructure shortages in urban counties vs. affordability challenges in rural counties.

---

## **Authors**

- **Alex Song** – `https://github.com/alexsong-lab`  
- **Gurpreet Kaur** – `https://github.com/Kaurgurpreet23`
- **Naomi Le Mouel** – `https://github.com/naomi-rlm`

---

## **License**

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## **Acknowledgements**

- **Libraries:** scikit‑learn, pandas, numpy, seaborn, matplotlib, scipy  
- **Textbook:** *An Introduction to Statistical Learning with Applications in Python*  
- **Data Sources:** WA DOL, DOE AFDC, US Census ACS  
- **Inspiration:** Seattle University MSDS program, DATA 5322 Statistical Machine Learning II
