# House Price Prediction

## Problem Statement
Predict residential property prices using machine learning based on 20,000+ housing records. The goal is to build a reliable model that estimates home values for buyers, sellers, and real estate platforms.

## Dataset
- **Source:** Kaggle - House Price Prediction Dataset
- **Records:** 20,640 residential properties
- **Features:** Location coordinates, house size, bedrooms, bathrooms, lot size, and more
- **Target:** House Price (continuous variable)

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Analyzed distributions of all numerical features
   - Identified outlier presence using box plots (IQR method) combined with domain knowledge
   - Visualized relationships between features and price

2. **Multicollinearity Analysis**
   - Used correlation matrix to detect highly correlated independent features
   - Identified feature redundancy (e.g., highly correlated room counts)

3. **Data Preprocessing**
   - Outlier treatment using domain-based thresholds
   - Feature selection based on statistical significance
   - Train-test split (70:30)

4. **Model Training & Evaluation**
   - Models tested: Linear Regression, Ridge, Lasso, XGBoost
   - XGBoost outperformed linear models due to non-linear relationships in housing data
   - Hyperparameter tuning via GridSearchCV (48 combinations)

## Results

| Model | Train R² | Test R² |
|-------|----------|---------|
| Linear Regression | 0.62 | 0.61 |
| Ridge | 0.63 | 0.62 |
| Lasso | 0.61 | 0.60 |
| **XGBoost** | **0.85** | **0.83** |

- **Final Model:** XGBoost
- **R² Score:** 0.83 (83% of price variation explained)
- **Overfitting Check:** Train-Test gap of only 0.02 — model generalizes well

## Business Impact

- An R² of 0.83 means the model performs like an experienced broker who has seen thousands of homes — reliable enough for price estimation tools
- Helps real estate platforms provide instant price quotes
- Assists buyers in identifying over/under-valued properties

## Tools Used

- Python (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Jupyter Notebook
- GridSearchCV for hyperparameter tuning

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `house_price.ipynb` in Jupyter Notebook
4. Run all cells

- **Source:**
- [Kaggle - House Price Prediction](https://www.kaggle.com/datasets/your-dataset-link-here)

## Author

**Anuj Singh Bhardwaj**
- [LinkedIn](https://www.linkedin.com/in/anuj-singh-bhardwaj/)
- [GitHub](https://github.com/Anuj-Singh-Bhardwaj)
