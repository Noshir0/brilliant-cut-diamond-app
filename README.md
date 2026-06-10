# 💎 Brilliant Cut — Diamond Price Intelligence

A Streamlit web app that predicts the price of a diamond from its attributes, built for a data-science mid-term project.

**The problem:** pricing diamonds by hand is slow and inconsistent. Brilliant Cut studies ~54,000 real diamonds and instantly estimates a fair market price for any stone — helping a retailer price inventory consistently and spot mis-priced diamonds.

## Features

The app has four pages (use the sidebar to switch):

- **Introduction** — the business case, a data preview, and a data-quality check.
- **Visualization** — a correlation heatmap, carat-vs-price scatter, and average price by quality grade.
- **Automated Report** — one-click exploratory summary of the whole dataset.
- **Prediction** — a linear-regression model that predicts price (R² ≈ 0.89, average error ≈ $859), shows which features drive the price, and includes a live "price a diamond" tool.

## The dataset

The classic **diamonds** dataset (`diamonds.csv`): 53,940 diamonds, 10 attributes (carat, cut, color, clarity, depth, table, x, y, z) and the target, **price** in USD.

## Run it locally

```bash
# 1. (optional) create an isolated environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt

# 3. launch the app
streamlit run streamlit_app.py
```

Then open the link it prints (usually http://localhost:8501).

## Files

| File | What it is |
|------|------------|
| `streamlit_app.py` | The app |
| `diamonds.csv` | The dataset |
| `diamond.png` | Header image |
| `requirements.txt` | Python dependencies |

## Built with

Python · Streamlit · pandas · NumPy · Matplotlib · Seaborn · scikit-learn
