import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.cluster import KMeans

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI


# =========================
# Page config
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="wide")

# =========================
# Blue accent styling (light + consistent)
# =========================
st.markdown(
    """
    <style>
    .blue-accent {
        color: #2563EB;
        font-weight: 700;
        margin-top: 0.8rem;
    }
    .subtle {
        color: #1F2937;
        font-size: 0.95rem;
    }
    .report-box {
        background-color: #EFF6FF;
        padding: 1.1rem 1.2rem;
        border-radius: 10px;
        border-left: 6px solid #2563EB;
    }
    .section-title {
        font-weight: 800;
        font-size: 1.1rem;
        border-bottom: 1px solid #CBD5E1;
        padding-bottom: 6px;
        margin: 14px 0 8px 0;
    }
    code {
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Market Research Assistant")
st.caption("Generate a concise, Wikipedia-grounded industry briefing in three steps.")

# =========================
# Sidebar: API Key input (masked + show toggle)
# =========================
st.sidebar.header("API Key")
st.sidebar.write("Enter your OpenAI API key to run the report.")
show_key = st.sidebar.checkbox("Show API key", value=False)
user_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="default" if show_key else "password"
)

# =========================
# Sidebar: Model settings
# =========================
with st.sidebar.expander("Advanced settings", expanded=False):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# =========================
# Persistent controls (avoid full refresh)
# =========================
with st.sidebar.form("controls_form"):
    st.markdown("**Report preferences**")
    report_focus = st.selectbox(
        "Report focus",
        ["Acquisition screening", "Market overview", "Competitive positioning", "Risk & compliance"],
        index=0,
    )
    detail_level = st.select_slider(
        "Detail level",
        options=["Concise", "Balanced", "Deep"],
        value="Balanced"
    )

    st.markdown("**Clustering (synthetic data)**")
    k_clusters = st.slider("K-means clusters", min_value=2, max_value=6, value=3, step=1)

    apply_controls = st.form_submit_button("Apply settings")

if "report_focus_value" not in st.session_state:
    st.session_state.report_focus_value = report_focus
if "detail_level_value" not in st.session_state:
    st.session_state.detail_level_value = detail_level
if "k_clusters_value" not in st.session_state:
    st.session_state.k_clusters_value = k_clusters

if apply_controls:
    st.session_state.report_focus_value = report_focus
    st.session_state.detail_level_value = detail_level
    st.session_state.k_clusters_value = k_clusters

# =========================
# Helper functions
# =========================
def industry_is_valid(industry: str) -> bool:
    return bool(industry and industry.strip())


def retrieve_wikipedia_docs(industry: str, k: int = 5):
    retriever = WikipediaRetriever(top_k_results=k, lang="en")
    try:
        docs = retriever.get_relevant_documents(industry)
    except AttributeError:
        docs = retriever.invoke(industry)
    return docs[:k]


def extract_urls(docs):
    urls = []
    for d in docs:
        src = (d.metadata or {}).get("source", "")
        if src:
            urls.append(src)

    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)

    return unique[:5]


def build_sources_text(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        url = (d.metadata or {}).get("source", "")
        text = (d.page_content or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = text[:2600]

        parts.append(
            f"[Source {i}]\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT EXCERPT: {text}\n"
        )
    return "\n\n".join(parts)


def cap_500_words(text: str) -> str:
    words = (text or "").split()
    if len(words) <= 500:
        return text.strip()
    return " ".join(words[:500]).rstrip() + "…"


# =========================
# Synthetic Data Schemas
# =========================
def rand_date(start_year=2020, end_year=2025):
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    delta_days = (end_date - start_date).days
    return (start_date + pd.Timedelta(days=int(np.random.randint(0, delta_days)))).date().isoformat()

# ... ALL schemas and helper functions unchanged ...

# =========================
# API key handling
# =========================
api_key = (user_key or "").strip()
if not api_key:
    st.markdown(
        """
        <div style="
            background:#F8FAFC;
            border:1px solid #E2E8F0;
            color:#0F172A;
            padding:14px 16px;
            border-radius:10px;
        ">
            <strong>Almost there.</strong>
            Please enter your OpenAI API key in the sidebar to continue.
            <span style="color:#475569;">It stays on your machine.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key


# =========================
# UI — Q1
# =========================
st.markdown("<h3 class='blue-accent'>Step 1 — Choose an industry</h3>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Tip: be specific (e.g., “Fast fashion”, “Semiconductor industry”, “EV batteries”).</div>",
    unsafe_allow_html=True
)

with st.form("industry_form"):
    industry = st.text_input(
        "Industry",
        placeholder="Try: Fast fashion",
    )
    submitted = st.form_submit_button("Generate report")

if submitted:
    if not industry_is_valid(industry):
        st.warning("Please enter an industry to continue.")
        st.stop()

    if len(industry.strip()) < 3:
        st.info("Please provide a more specific industry name.")
        st.caption("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    st.success("Industry received. Fetching Wikipedia sources...")

    # Lookup immediately in Step 1
    docs = retrieve_wikipedia_docs(industry.strip(), k=5)
    urls = extract_urls(docs)

    if not urls:
        st.warning("I couldn't find reliable Wikipedia matches. Please be more specific or rephrase the industry you would like to research.")
        st.info("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    # =========================
    # Step 2 — only appears if results exist
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 2 — Top Wikipedia sources</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>These are the five most relevant pages used to generate the report.</div>",
        unsafe_allow_html=True
    )

    with st.expander("Show sources", expanded=True):
        shown = set()
        rank = 0
        for d in docs:
            src = (d.metadata or {}).get("source", "")
            title = (d.metadata or {}).get("title", "Untitled")
            if not src or src in shown:
                continue
            rank += 1
            shown.add(src)
            st.write(f"{rank}. {title} — {src}")
            if rank >= 5:
                break

    # =========================
    # Step 3 report/visuals
    # =========================
   
    with st.form("visual_controls"):
        st.markdown("<div class='section-title'>Visual Controls</div>", unsafe_allow_html=True)
        time_granularity = st.radio(
            "Trend granularity (affects M&A visuals)",
            options=["Monthly", "Annual"],
            horizontal=True
        )
        show_profit_pool = st.checkbox("Show Profit Pool by Segment", value=True)
        show_risk_view = st.checkbox("Show Risk vs Supply Concentration", value=True)
        show_cluster_view = st.checkbox("Show Clustering (K-means)", value=True)
        apply_visuals = st.form_submit_button("Apply visuals")

    if "time_granularity_value" not in st.session_state:
        st.session_state.time_granularity_value = time_granularity
    if "show_profit_pool_value" not in st.session_state:
        st.session_state.show_profit_pool_value = show_profit_pool
    if "show_risk_view_value" not in st.session_state:
        st.session_state.show_risk_view_value = show_risk_view
    if "show_cluster_view_value" not in st.session_state:
        st.session_state.show_cluster_view_value = show_cluster_view

    if apply_visuals:
        st.session_state.time_granularity_value = time_granularity
        st.session_state.show_profit_pool_value = show_profit_pool
        st.session_state.show_risk_view_value = show_risk_view
        st.session_state.show_cluster_view_value = show_cluster_view

# =========================
# Synthetic Dataset & M&A-Oriented Visuals
# =========================
st.markdown("<h3 class='blue-accent'>Synthetic Dataset & M&A-Oriented Visuals</h3>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>A synthetic dataset is generated and enriched with acquisition-style metrics for analyst screening.</div>",
    unsafe_allow_html=True
)

# Generate + enrich synthetic data (make sure these functions exist above)
synthetic_df = generate_synthetic_df(industry.strip(), rows=240)
synthetic_df = enrich_for_ma(synthetic_df, industry.strip())

# ---- Market Share (Top Companies)
st.markdown("<div class='section-title'>Market Share — Top Companies</div>", unsafe_allow_html=True)
st.write("Ranks companies by estimated market share within the synthetic sample to highlight potential leaders.")

share_df = (
    synthetic_df.groupby("company")["market_share_pct"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

st.altair_chart(
    alt.Chart(share_df)
    .mark_bar()
    .encode(
        x=alt.X("market_share_pct:Q", title="Market Share (%)"),
        y=alt.Y("company:N", sort="-x", title="Company"),
        tooltip=["company", "market_share_pct"],
    ),
    use_container_width=True
)
# ---- Growth vs EBITDA Margin
st.markdown("<div class='section-title'>Growth vs EBITDA Margin</div>", unsafe_allow_html=True)
st.write("Shows the trade-off between growth and profitability across synthetic entities.")
st.altair_chart(
    alt.Chart(synthetic_df)
    .mark_circle(size=70, opacity=0.8)
    .encode(
        x=alt.X("revenue_growth_pct:Q", title="Revenue Growth (%)"),
        y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
        color=alt.Color("segment:N", title="Segment"),
        tooltip=["company", "segment", "revenue_growth_pct", "ebitda_margin_pct"],
    ),
    use_container_width=True
)
# ---- Revenue Distribution
    st.markdown("<div class='section-title'>Revenue Distribution</div>", unsafe_allow_html=True)
    st.write("Shows how revenue is distributed across entities, highlighting size skew.")
    st.altair_chart(
        alt.Chart(synthetic_df)
        .mark_bar()
        .encode(
            x=alt.X("revenue_usd_m:Q", bin=alt.Bin(maxbins=20), title="Revenue (USD, millions)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"],
        ),
        use_container_width=True
    )

    # ---- Revenue Trend (Monthly/Annual)
    st.markdown("<div class='section-title'>Revenue Trend Over Time</div>", unsafe_allow_html=True)
    st.write("Tracks aggregate revenue trends over time, based on synthetic time signals.")
    if st.session_state.time_granularity_value == "Monthly":
        time_df = synthetic_df.groupby("month")["revenue_usd_m"].sum().reset_index()
        time_col = "month"
    else:
        time_df = synthetic_df.groupby("year")["revenue_usd_m"].sum().reset_index()
        time_col = "year"

    st.altair_chart(
        alt.Chart(time_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{time_col}:O", title=time_col.title()),
            y=alt.Y("revenue_usd_m:Q", title="Total Revenue (USD, millions)"),
            tooltip=[time_col, "revenue_usd_m"],
        ),
        use_container_width=True
    )

    # ---- Capex vs Margin
    st.markdown("<div class='section-title'>Capex Intensity vs Margin</div>", unsafe_allow_html=True)
    st.write("Identifies which players combine strong margins with capital efficiency.")
    st.altair_chart(
        alt.Chart(synthetic_df)
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x=alt.X("capex_intensity_pct:Q", title="Capex Intensity (%)"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "capex_intensity_pct", "ebitda_margin_pct"],
        ),
        use_container_width=True
    )

    # ---- Risk vs Supply Concentration (toggle)
    if st.session_state.show_risk_view_value:
        st.markdown("<div class='section-title'>Risk vs Supply Concentration</div>", unsafe_allow_html=True)
        st.write("Highlights exposure to supply‑chain concentration risk against composite risk scores.")
        st.altair_chart(
            alt.Chart(synthetic_df)
            .mark_circle(size=70, opacity=0.8)
            .encode(
                x=alt.X("supply_concentration:Q", title="Supply Concentration (0–1)"),
                y=alt.Y("risk_score:Q", title="Risk Score (0–1)"),
                color=alt.Color("segment:N", title="Segment"),
                tooltip=["company", "supply_concentration", "risk_score"],
            ),
            use_container_width=True
        )

    # ---- Segment Attractiveness
    st.markdown("<div class='section-title'>Segment Attractiveness</div>", unsafe_allow_html=True)
    st.write("Compares segments by a composite of growth, margin, and low risk.")
    seg_df = synthetic_df.groupby("segment").agg(
        avg_growth=("revenue_growth_pct", "mean"),
        avg_margin=("ebitda_margin_pct", "mean"),
        avg_risk=("risk_score", "mean")
    ).reset_index()
    seg_df["attractiveness"] = (seg_df["avg_growth"] * 0.4 + seg_df["avg_margin"] * 0.5 + (1 - seg_df["avg_risk"]) * 10)
    st.altair_chart(
        alt.Chart(seg_df)
        .mark_bar()
        .encode(
            x=alt.X("attractiveness:Q", title="Attractiveness Score"),
            y=alt.Y("segment:N", sort="-x", title="Segment"),
            tooltip=["segment", "attractiveness", "avg_growth", "avg_margin", "avg_risk"]
        ),
        use_container_width=True
    )

    # ---- Top 5 Acquisition Targets
    st.markdown("<div class='section-title'>Top 5 Acquisition Targets</div>", unsafe_allow_html=True)
    st.write("Ranks targets using a composite of growth, margin, and lower risk.")
    target_df = synthetic_df.copy()
    target_df["target_score"] = (
        target_df["revenue_growth_pct"] * 0.4 +
        target_df["ebitda_margin_pct"] * 0.5 +
        (1 - target_df["risk_score"]) * 10
    )
    top_targets = target_df.sort_values("target_score", ascending=False).head(5)
    st.dataframe(top_targets[["company", "segment", "region", "revenue_growth_pct", "ebitda_margin_pct", "risk_score", "target_score"]])

    # ---- Profit Pool by Segment (toggle)
    if st.session_state.show_profit_pool_value:
        st.markdown("<div class='section-title'>Profit Pool by Segment</div>", unsafe_allow_html=True)
        st.write("Estimates segment profit pools using revenue × margin as a proxy.")
        profit_df = synthetic_df.groupby("segment").apply(
            lambda d: (d["revenue_usd_m"] * (d["ebitda_margin_pct"] / 100)).sum()
        ).reset_index(name="profit_pool")
        st.altair_chart(
            alt.Chart(profit_df)
            .mark_bar()
            .encode(
                x=alt.X("profit_pool:Q", title="Profit Pool (proxy)"),
                y=alt.Y("segment:N", sort="-x", title="Segment"),
                tooltip=["segment", "profit_pool"]
            ),
            use_container_width=True
        )

    # ---- Margin vs Leverage
    st.markdown("<div class='section-title'>Margin vs Leverage</div>", unsafe_allow_html=True)
    st.write("Shows whether higher leverage correlates with margin performance.")
    st.altair_chart(
        alt.Chart(synthetic_df)
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x=alt.X("debt_to_equity:Q", title="Debt to Equity"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "debt_to_equity", "ebitda_margin_pct"],
        ),
        use_container_width=True
    )

    # ---- Top 5 Risks
    st.markdown("<div class='section-title'>Top 5 Risks</div>", unsafe_allow_html=True)
    st.write("Lists the highest‑risk entities to flag for diligence.")
    top_risks = synthetic_df.sort_values("risk_score", ascending=False).head(5)
    st.dataframe(top_risks[["company", "segment", "region", "risk_score", "supply_concentration"]])

    # ---- Profit Strategy Summary
    st.markdown("<div class='section-title'>Profit Strategy Summary</div>", unsafe_allow_html=True)
    st.write("Synthetic signals suggest where value is concentrated and which segments to prioritize.")
    st.write(
        f"- Highest attractiveness segment: **{seg_df.sort_values('attractiveness', ascending=False).iloc[0]['segment']}**"
    )
    st.write(
        f"- Largest profit pool: **{profit_df.sort_values('profit_pool', ascending=False).iloc[0]['segment']}**"
    )
    st.write(
        f"- Most risky segment: **{seg_df.sort_values('avg_risk', ascending=False).iloc[0]['segment']}**"
    )

    # =========================
    # Clustering (K-means) — toggle controlled
    # =========================
    if st.session_state.show_cluster_view_value:
        st.markdown("<h3 class='blue-accent'>Clustering (K-means)</h3>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtle'>Uses the same synthetic dataset to group entities by numeric characteristics.</div>",
            unsafe_allow_html=True
        )

        cluster_df = synthetic_df.select_dtypes(include=["number"]).copy()

        with st.form("cluster_controls"):
            st.markdown("**Cluster Controls**")
            cluster_fields = st.multiselect(
                "Fields used to cluster",
                options=cluster_df.columns.tolist(),
                default=["revenue_growth_pct", "ebitda_margin_pct", "capex_intensity_pct", "risk_score"]
            )
            cluster_x = st.selectbox("X-axis", options=cluster_df.columns.tolist(), index=cluster_df.columns.get_loc("revenue_growth_pct"))
            cluster_y = st.selectbox("Y-axis", options=cluster_df.columns.tolist(), index=cluster_df.columns.get_loc("ebitda_margin_pct"))
            apply_cluster = st.form_submit_button("Apply clustering")

        if "cluster_fields_value" not in st.session_state:
            st.session_state.cluster_fields_value = cluster_fields
        if "cluster_x_value" not in st.session_state:
            st.session_state.cluster_x_value = cluster_x
        if "cluster_y_value" not in st.session_state:
            st.session_state.cluster_y_value = cluster_y

        if apply_cluster:
            st.session_state.cluster_fields_value = cluster_fields
            st.session_state.cluster_x_value = cluster_x
            st.session_state.cluster_y_value = cluster_y

        if len(st.session_state.cluster_fields_value) >= 2:
            scaled = (cluster_df[st.session_state.cluster_fields_value] - cluster_df[st.session_state.cluster_fields_value].mean()) / (
                cluster_df[st.session_state.cluster_fields_value].std(ddof=0) + 1e-9
            )
            km = KMeans(n_clusters=st.session_state.k_clusters_value, n_init=10, random_state=42)
            clusters = km.fit_predict(scaled)
            plot_df = synthetic_df.copy()
            plot_df["cluster"] = clusters.astype(str)

            st.altair_chart(
                alt.Chart(plot_df)
                .mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X(f"{st.session_state.cluster_x_value}:Q", title=st.session_state.cluster_x_value),
                    y=alt.Y(f"{st.session_state.cluster_y_value}:Q", title=st.session_state.cluster_y_value),
                    color=alt.Color("cluster:N", title="Cluster"),
                    tooltip=["company", st.session_state.cluster_x_value, st.session_state.cluster_y_value, "cluster"],
                ),
                use_container_width=True
            )

            cluster_summary = plot_df.groupby("cluster")[st.session_state.cluster_fields_value].mean().reset_index()
            st.markdown("<div class='section-title'>Cluster Insights</div>", unsafe_allow_html=True)
            st.write("Average values per cluster to help compare strategic profiles.")
            st.dataframe(cluster_summary)
        else:
            st.warning("Select at least two numeric fields for clustering.")
