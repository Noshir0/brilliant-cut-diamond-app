"""
💎 Brilliant Cut — Diamond Price Intelligence
Mid-Term Group Project | Streamlit + Linear Regression

Business case: An online diamond retailer needs a fast, consistent way to price
its inventory and to flag stones that are over- or under-priced versus the market.
This app explores the classic `diamonds` dataset (53,940 stones) and trains a
linear regression model that predicts a diamond's price from its physical and
quality attributes.

Run locally with:  streamlit run streamlit_app.py
"""

# ------------------------------------------------------------------
# Step 00 - Import the packages
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Brilliant Cut — Diamond Price Intelligence 💎",
    layout="wide",
)

CURRENCY = "$"


# ------------------------------------------------------------------
# Data loading (cached so it only reads from disk once)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")
    # The ggplot2 diamonds file occasionally carries an index column; drop it.
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


df = load_data()

CATEGORICAL_COLS = ["cut", "color", "clarity"]
NUMERIC_COLS = [c for c in df.columns if c not in CATEGORICAL_COLS]

# Human-readable ordering for the quality grades (worst -> best)
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]          # J (worst) -> D (best)
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("💎 Brilliant Cut")
st.sidebar.caption("Diamond Price Intelligence")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Visualization 📊", "Automated Report 📑", "Prediction 🔮"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Dataset: **diamonds** (ggplot2 / Kaggle) — 53,940 stones, 10 attributes.\n\n"
    "Goal: predict a diamond's **price** with linear regression."
)


# ==================================================================
# 01 - INTRODUCTION
# ==================================================================
if page == "Introduction 📘":
    try:
        st.image("diamond.png", use_container_width=True)
    except Exception:
        pass

    st.title("01 · Introduction 📘")
    st.markdown(
        """
        ### The business problem
        **Brilliant Cut** is an online diamond retailer. Pricing thousands of stones
        by hand is slow, inconsistent, and leaves money on the table — some diamonds
        are accidentally under-priced (lost margin) while others are over-priced
        (lost sales).

        **Our goal:** build a data-driven model that predicts a fair market **price**
        for any diamond from its measurable attributes, so the team can price
        inventory instantly and spot mis-priced stones.
        """
    )

    st.markdown("### The dataset")
    st.markdown(
        """
        Each row is one diamond. The columns are:

        | Column | Meaning |
        |---|---|
        | **carat** | Weight of the diamond |
        | **cut** | Quality of the cut (Fair → Ideal) |
        | **color** | Diamond colour, J (worst) → D (best) |
        | **clarity** | How clear it is (I1 worst → IF best) |
        | **depth** | Total depth %  |
        | **table** | Width of top facet relative to widest point |
        | **x / y / z** | Length, width, depth in mm |
        | **price** | 🎯 Price in US dollars (**what we predict**) |
        """
    )

    st.markdown("##### Data preview")
    rows = st.slider("Number of rows to display", 5, 25, 5)
    st.dataframe(df.head(rows), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Diamonds", f"{len(df):,}")
    c2.metric("Attributes", f"{df.shape[1]}")
    c3.metric("Avg price", f"{CURRENCY}{df['price'].mean():,.0f}")

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found — the dataset is clean.")
    else:
        st.warning("⚠️ Some missing values are present.")

    st.markdown("##### 📈 Summary statistics")
    if st.button("Show describe table"):
        st.dataframe(df.describe(), use_container_width=True)


# ==================================================================
# 02 - VISUALIZATION
# ==================================================================
elif page == "Visualization 📊":
    st.title("02 · Data Visualization 📊")
    st.write("Explore the relationships that drive a diamond's price.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔥 Correlation Heatmap", "💠 Carat vs Price", "📊 Price by Quality", "📈 Custom Chart"]
    )

    # --- Tab 1: correlation heatmap -------------------------------
    with tab1:
        st.subheader("Correlation between numeric features")
        df_numeric = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)
        st.info(
            "**Key insight:** `carat` is the strongest driver of price "
            f"(correlation ≈ {df['carat'].corr(df['price']):.2f}). "
            "The size measures x, y, z are also highly correlated with price — "
            "because bigger stones weigh more."
        )

    # --- Tab 2: carat vs price scatter ----------------------------
    with tab2:
        st.subheader("Carat vs Price (coloured by cut)")
        sample = df.sample(min(4000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=sample, x="carat", y="price", hue="cut",
            hue_order=CUT_ORDER, alpha=0.5, ax=ax,
        )
        ax.set_xlabel("Carat (weight)")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)
        st.info(
            "Price rises sharply with carat. For the **same** weight, better-cut "
            "diamonds (Ideal/Premium) tend to sit higher — quality adds a premium."
        )

    # --- Tab 3: average price by quality grade --------------------
    with tab3:
        st.subheader("Average price by quality grade")
        grade = st.selectbox("Choose a quality attribute", CATEGORICAL_COLS)
        order = {"cut": CUT_ORDER, "color": COLOR_ORDER, "clarity": CLARITY_ORDER}[grade]
        avg = df.groupby(grade)["price"].mean().reindex(order)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg.index, y=avg.values, hue=avg.index,
                    legend=False, palette="viridis", ax=ax)
        ax.set_xlabel(grade.capitalize())
        ax.set_ylabel("Average price ($)")
        st.pyplot(fig)
        st.caption(
            "Note: counter-intuitively, average price can fall as clarity/colour "
            "improve — because the very best grades are dominated by *small* stones. "
            "This is why we need a model that controls for carat."
        )

    # --- Tab 4: user-driven chart ---------------------------------
    with tab4:
        st.subheader("Build your own chart")
        col_x = st.selectbox("X-axis variable", df.columns, index=0)
        col_y = st.selectbox("Y-axis variable", df.columns,
                             index=list(df.columns).index("price"))
        chart_type = st.radio("Chart type", ["Scatter", "Line", "Bar"], horizontal=True)
        plot_df = df.sample(min(3000, len(df)), random_state=2).sort_values(col_x)
        fig, ax = plt.subplots(figsize=(10, 5))
        if chart_type == "Scatter":
            ax.scatter(plot_df[col_x], plot_df[col_y], alpha=0.4)
        elif chart_type == "Line":
            ax.plot(plot_df[col_x].values, plot_df[col_y].values)
        else:
            ax.bar(plot_df[col_x].astype(str), plot_df[col_y])
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        st.pyplot(fig)


# ==================================================================
# 03 - AUTOMATED REPORT
# ==================================================================
elif page == "Automated Report 📑":
    st.title("03 · Automated Report 📑")
    st.write(
        "Generate a one-click exploratory data analysis report. "
    )

    if st.button("Generate report"):
        with st.spinner("Generating report..."):
            try:
                from ydata_profiling import ProfileReport
                from streamlit_pandas_profiling import st_profile_report

                profile = ProfileReport(
                    df, title="Diamonds Data Report", explorative=True, minimal=True
                )
                st_profile_report(profile)
                export = profile.to_html()
                st.download_button(
                    "📥 Download full report",
                    data=export,
                    file_name="diamonds_report.html",
                    mime="text/html",
                )
            except Exception:
                # Lightweight fallback so the page always works
                st.subheader("Statistical summary")
                st.dataframe(df.describe(include="all"), use_container_width=True)

                st.subheader("Distribution of numeric columns")
                num = df.select_dtypes(include=np.number)
                num.hist(figsize=(12, 8))
                plt.tight_layout()
                st.pyplot(plt.gcf())

                st.subheader("Category counts")
                for c in CATEGORICAL_COLS:
                    st.write(f"**{c}**")
                    st.bar_chart(df[c].value_counts())


# ==================================================================
# 04 - PREDICTION (Linear Regression)
# ==================================================================
elif page == "Prediction 🔮":
    st.title("04 · Price Prediction with Linear Regression 🔮")

    # --- 01 Data preprocessing -----------------------------------
    df2 = df.copy().dropna()

    # Encode the categorical quality grades into numbers.
    encoders = {}
    for c in CATEGORICAL_COLS:
        le = LabelEncoder()
        df2[c] = le.fit_transform(df2[c])
        encoders[c] = le

    all_features = [c for c in df2.columns if c != "price"]

    st.sidebar.markdown("### ⚙️ Model settings")
    features_selection = st.sidebar.multiselect(
        "Select features (X)", all_features, default=all_features
    )
    target_selection = "price"
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2)
    selected_metrics = st.sidebar.multiselect(
        "Metrics to display",
        ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
        default=["R² Score", "Mean Absolute Error (MAE)"],
    )

    if not features_selection:
        st.warning("⬅️ Please select at least one feature in the sidebar.")
        st.stop()

    st.markdown(
        "We train a **linear regression** to predict `price` from the selected "
        "features, then evaluate it on a held-out test set."
    )

    # --- i) X and y ----------------------------------------------
    X = df2[features_selection]
    y = df2[target_selection]

    with st.expander("Preview the model inputs (X) and target (y)"):
        st.dataframe(X.head(), use_container_width=True)
        st.dataframe(y.head(), use_container_width=True)

    # --- ii) train/test split ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- 02 Model ------------------------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # --- 03 Evaluation -------------------------------------------
    st.subheader("Model performance")
    mcols = st.columns(len(selected_metrics) if selected_metrics else 1)
    i = 0
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        mcols[i].metric("R² Score", f"{r2:.3f}")
        i += 1
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        mcols[i].metric("MAE", f"{CURRENCY}{mae:,.0f}")
        i += 1
    if "Root Mean Squared Error (RMSE)" in selected_metrics:
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
        mcols[i].metric("RMSE", f"{CURRENCY}{rmse:,.0f}")
        i += 1

    mae_val = metrics.mean_absolute_error(y_test, predictions)
    st.success(
        f"On average our model's price estimate is off by "
        f"{CURRENCY}{mae_val:,.0f} — small relative to the average "
        f"diamond price of {CURRENCY}{df['price'].mean():,.0f}."
    )

    # --- Actual vs predicted plot --------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.4)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r", linewidth=2,
    )
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # --- Driving variables ---------------------------------------
    st.subheader("Which variables drive the price?")
    coefs = pd.Series(model.coef_, index=features_selection).sort_values()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    coefs.plot(kind="barh", ax=ax2, color="#3b82f6")
    ax2.set_xlabel("Effect on price ($ per unit)")
    ax2.set_title("Linear regression coefficients")
    st.pyplot(fig2)
    st.caption(
        "A positive coefficient pushes price up, a negative one pulls it down. "
        "`carat` has by far the largest positive effect — confirming that weight "
        "is the dominant driver of a diamond's price."
    )

    # --- Live price estimator ------------------------------------
    st.markdown("---")
    st.subheader("💰 Price a diamond")
    st.write("Enter a diamond's attributes to get an instant price estimate.")

    ip = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        ip["carat"] = st.number_input("Carat", 0.2, 5.0, 0.7, 0.05)
        ip["cut"] = st.selectbox("Cut", CUT_ORDER, index=4)
        ip["color"] = st.selectbox("Color", COLOR_ORDER, index=5)
    with c2:
        ip["clarity"] = st.selectbox("Clarity", CLARITY_ORDER, index=3)
        ip["depth"] = st.number_input("Depth %", 50.0, 75.0, 61.8, 0.1)
        ip["table"] = st.number_input("Table", 50.0, 75.0, 57.0, 0.1)
    with c3:
        ip["x"] = st.number_input("x (mm)", 0.0, 11.0, 5.7, 0.1)
        ip["y"] = st.number_input("y (mm)", 0.0, 11.0, 5.7, 0.1)
        ip["z"] = st.number_input("z (mm)", 0.0, 7.0, 3.5, 0.1)

    if st.button("Estimate price 💎"):
        row = {}
        for f in features_selection:
            val = ip.get(f, 0)
            if f in CATEGORICAL_COLS:
                val = int(encoders[f].transform([val])[0])
            row[f] = val
        x_new = pd.DataFrame([row])[features_selection]
        est = float(model.predict(x_new)[0])
        st.metric("Estimated price", f"{CURRENCY}{max(est, 0):,.0f}")

# source .venv/bin/activate && streamlit run streamlit_app.py
