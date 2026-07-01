# Property Price Predictor (Linear Regression Starter)

A minimal, ready-to-run project to train your first machine learning model:
predicting property prices from a CSV dataset using linear regression.

## Project structure

```
property-price-predictor/
├── property_prices.csv           # sample dataset (300 properties)
├── train_linear_regression.py    # command-line: loads data, trains & evaluates model
├── streamlit_app.py              # web UI: interactive dashboard + predictor
├── requirements.txt
└── README.md
```

## Dataset columns

| Column        | Description                          |
|---------------|---------------------------------------|
| size_sqft     | Property size in square feet          |
| bedrooms      | Number of bedrooms                    |
| bathrooms     | Number of bathrooms                   |
| age_years     | Age of the property in years          |
| location      | Downtown / Suburb / Rural / Uptown / Waterfront |
| garage_spaces | Number of garage spaces               |
| price         | Sale price (target variable, in $)    |

> This is synthetic sample data so you can start training right away.
> Swap in your own CSV (same column names, or edit the script) once you're ready.

## Setup in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal in VS Code (`Ctrl+` ` or View > Terminal).
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the command-line training script:
   ```bash
   python train_linear_regression.py
   ```

   Or launch the interactive web UI instead:
   ```bash
   streamlit run streamlit_app.py
   ```
   This opens a browser tab at `http://localhost:8501` with two sections:
   - **Model performance** — MAE / RMSE / R² for bathrooms, garage spaces,
     and price, plus accuracy for location, all measured on a held-out test set
   - **Predict** — enter only `size`, `age`, and `bedrooms`; the app predicts
     everything else (bathrooms, garage spaces, location, and price) and
     displays it

   Under the hood: one multi-output `LinearRegression` predicts the numeric
   fields (bathrooms, garage spaces, price) from size/age/bedrooms, and a
   `LogisticRegression` classifier predicts the location category.

## What the script does

1. Loads `data/property_prices.csv` with pandas.
2. Splits data into training (80%) and test (20%) sets.
3. One-hot encodes the `location` column and builds a `LinearRegression` pipeline.
4. Trains the model and prints evaluation metrics: MAE, RMSE, R².
5. Saves the trained model to `property_price_model.joblib`.
6. Predicts the price of one example new property as a demo.

## Using your own data

Replace `property_prices.csv` with your own file, keeping the same
column names (or update `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` /
`TARGET` in `train_linear_regression.py` and `streamlit_app.py` to match your columns).

If your data is an Excel file (`.xlsx`) instead of CSV, change the load line to:
```python
df = pd.read_excel("property_prices.xlsx")
```
(requires `openpyxl` — add it to `requirements.txt`)

## Next steps once this works

- Try other models: `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`
- Add feature engineering (e.g. price per sqft, bedroom/bathroom ratio)
- Visualize predictions vs actual prices with matplotlib
- Do k-fold cross-validation instead of a single train/test split
