
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Marketing Campaign Analytics",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_dataset.csv"
)

EVALUATION_FILE = (
    BASE_DIR
    / "reports"
    / "model_evaluation_summary.csv"
)

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_evaluation():
    if EVALUATION_FILE.exists():
        return pd.read_csv(EVALUATION_FILE)

    return pd.DataFrame()


try:
    df = load_data()
    evaluation = load_evaluation()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.title("📊 Multi-Brand Marketing Campaign Analytics")

st.subheader(
    "Marketing Campaign Performance Analysis and Prediction"
)

st.markdown(
    """
    An interactive analytics dashboard to analyse and compare
    marketing campaign performance across **Nykaa, Purplle and Tira**.
    """
)

st.divider()

# --------------------------------------------------
# 5. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.write(
    "Select the brands and campaign types you want to analyse."
)

brands = sorted(
    df["Brand"].dropna().unique().tolist()
)

campaign_types = sorted(
    df["Campaign_Type"].dropna().unique().tolist()
)

selected_brands = st.sidebar.multiselect(
    "Select Brands",
    options=brands,
    default=[],
    placeholder="Choose one or more brands"
)

selected_campaigns = st.sidebar.multiselect(
    "Select Campaign Types",
    options=campaign_types,
    default=[],
    placeholder="Choose one or more campaign types"
)

st.sidebar.divider()

st.sidebar.caption(
    "Select at least one brand and one campaign type "
    "to view the dashboard."
)

# --------------------------------------------------
# 6. FILTER VALIDATION
# --------------------------------------------------

if not selected_brands or not selected_campaigns:

    st.info(
        "👈 Please select at least one brand and one campaign type "
        "from the sidebar to view the dashboard."
    )

    st.stop()


filtered_df = df[
    df["Brand"].isin(selected_brands)
    & df["Campaign_Type"].isin(selected_campaigns)
].copy()


if filtered_df.empty:

    st.warning(
        "No campaign records match your selected filters. "
        "Please change your selections."
    )

    st.stop()


# --------------------------------------------------
# 7. CAMPAIGN OVERVIEW
# --------------------------------------------------

st.header("1. Campaign Overview")

total_campaigns = len(filtered_df)

total_brands = filtered_df["Brand"].nunique()

average_roi = filtered_df["ROI"].mean()

total_revenue = filtered_df["Revenue"].sum()

average_revenue = filtered_df["Revenue"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Campaign Records",
    f"{total_campaigns:,}"
)

col2.metric(
    "Selected Brands",
    total_brands
)

col3.metric(
    "Average ROI",
    f"{average_roi:.2f}"
)

col4.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)

st.divider()

# --------------------------------------------------
# 8. BRAND PERFORMANCE COMPARISON
# --------------------------------------------------

st.header("2. Brand Performance Comparison")

brand_summary = (
    filtered_df.groupby("Brand")
    .agg(
        Campaigns=("Brand", "count"),
        Average_ROI=("ROI", "mean"),
        Total_Revenue=("Revenue", "sum"),
        Average_Revenue=("Revenue", "mean")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Average ROI by Brand")

    st.bar_chart(
        brand_summary.set_index("Brand")[["Average_ROI"]]
    )

with col2:

    st.subheader("Total Revenue by Brand")

    st.bar_chart(
        brand_summary.set_index("Brand")[["Total_Revenue"]]
    )

st.subheader("Brand Performance Summary")

st.dataframe(
    brand_summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# 9. CAMPAIGN PERFORMANCE ANALYSIS
# --------------------------------------------------

st.header("3. Campaign Performance Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Campaign Type Distribution")

    campaign_counts = (
        filtered_df["Campaign_Type"]
        .value_counts()
        .rename_axis("Campaign Type")
        .to_frame("Campaign Count")
    )

    st.bar_chart(campaign_counts)

with col2:

    st.subheader("Average ROI by Campaign Type")

    campaign_roi = (
        filtered_df.groupby("Campaign_Type")["ROI"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(campaign_roi)

# --------------------------------------------------
# 10. CUSTOMER SEGMENT ANALYSIS
# --------------------------------------------------

st.subheader("Customer Segment Analysis")

segment_roi = (
    filtered_df.groupby("Customer_Segment")["ROI"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(segment_roi)

st.divider()

# --------------------------------------------------
# 11. ROI ANALYSIS
# --------------------------------------------------

st.header("4. Return on Investment Analysis")

profitable = (filtered_df["ROI"] > 0).sum()

loss_or_break_even = (filtered_df["ROI"] <= 0).sum()

positive_roi_percentage = (
    profitable / total_campaigns
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Positive ROI Records",
    f"{profitable:,}"
)

col2.metric(
    "Non-Positive ROI Records",
    f"{loss_or_break_even:,}"
)

col3.metric(
    "Positive ROI Percentage",
    f"{positive_roi_percentage:.2f}%"
)

st.subheader("ROI Distribution")

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(
    filtered_df["ROI"].dropna(),
    bins=50,
    edgecolor="black"
)

ax.set_xlabel("Return on Investment")

ax.set_ylabel("Campaign Frequency")

ax.set_title("Distribution of Campaign ROI")

ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)

plt.close(fig)

st.divider()

# --------------------------------------------------
# 12. MARKETING CHANNEL ANALYSIS
# --------------------------------------------------

st.header("5. Marketing Channel Analysis")

channel_roi = (
    filtered_df.groupby("Channel_Used")["ROI"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

st.subheader("Top 15 Channel Combinations by Average ROI")

st.bar_chart(channel_roi)

st.divider()

# --------------------------------------------------
# 13. MACHINE LEARNING MODEL EVALUATION
# --------------------------------------------------

st.header("6. Machine Learning Model Evaluation")

if not evaluation.empty:

    st.subheader("Consolidated Model Performance")

    st.dataframe(
        evaluation.round(4),
        use_container_width=True,
        hide_index=True
    )

    # Revenue Prediction

    st.subheader("Revenue Prediction – Linear Regression")

    revenue_result = evaluation[
        evaluation["Task"] == "Revenue Prediction"
    ]

    if not revenue_result.empty:

        result = revenue_result.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{result['MAE']:,.2f}"
        )

        col2.metric(
            "RMSE",
            f"{result['RMSE']:,.2f}"
        )

        col3.metric(
            "R² Score",
            f"{result['R2_Score']:.4f}"
        )

    # Profit/Loss Classification

    st.subheader("Profit/Loss Classification")

    classification_result = evaluation[
        evaluation["Task"] == "Profit/Loss Classification"
    ]

    if not classification_result.empty:

        st.dataframe(
            classification_result[
                [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1_Score"
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "Model evaluation report is not available."
    )

st.caption(
    "Note: Profit/Loss classification represents positive ROI "
    "versus non-positive ROI, based on the supplied ROI values."
)

st.divider()

# --------------------------------------------------
# 14. CAMPAIGN DATA EXPLORER
# --------------------------------------------------

st.header("7. Campaign Data Explorer")

st.write(
    "Explore campaign records based on your selected filters."
)

st.dataframe(
    filtered_df.head(1000),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 15. DOWNLOAD FILTERED DATA
# --------------------------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Campaign Data",
    data=csv_data,
    file_name="filtered_campaign_data.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# 16. FOOTER
# --------------------------------------------------

st.caption(
    "Multi-Brand Marketing Campaign Performance Analysis "
    "| Python | Pandas | Streamlit | Machine Learning"
)
