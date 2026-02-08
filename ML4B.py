import os
import re
import io
import random
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import openai
import altair as alt
from sklearn.cluster import KMeans

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI


# =========================
# Page config
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="wide")

# =========================
# Styling
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
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 1px solid #CBD5E1;
        padding-bottom: 4px;
        margin: 10px 0 6px 0;
        color: #0F172A;
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
# Sidebar: API Key + settings
# =========================
st.sidebar.header("API Key")
st.sidebar.write("Enter your OpenAI API key to run the report.")
show_key = st.sidebar.checkbox("Show API key", value=False)
user_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="default" if show_key else "password"
)

with st.sidebar.expander("Advanced settings", expanded=False):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

st.sidebar.header("Report Preferences")
report_focus = st.sidebar.selectbox(
    "Report focus",
    [
        "Balanced",
        "Acquisition fit",
        "Market size & growth",
        "Competitive landscape",
        "Regulation & risk",
        "Operations & supply chain",
    ],
)
detail_level = st.sidebar.select_slider(
    "Detail level",
    options=["Concise", "Standard", "Detailed"],
    value="Standard",
)
st.sidebar.caption("Temperature adjusts phrasing variety; facts must still come only from sources.")

st.sidebar.header("Data (CSV optional)")
uploaded_csv = st.sidebar.file_uploader("Upload CSV for clustering", type=["csv"])
k_clusters = st.sidebar.slider("K-means clusters", min_value=2, max_value=6, value=3, step=1)

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
# Wikipedia helpers
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
# Synthetic dataset generator
# =========================
random_seed = 42

def rand_date(start_year=2020, end_year=2025):
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    return (start_date + timedelta(days=random.randint(0, delta_days))).isoformat()

def schema_fast_fashion():
    brands = ["Zara","H&M","Shein","Forever 21","Uniqlo","Primark","Boohoo","Mango","ASOS"]
    product_types = ["T-shirt","Jeans","Dress","Hoodie","Sweater","Skirt","Shorts","Jacket"]
    materials = ["Cotton","Polyester","Viscose","Linen","Nylon","Acrylic","Wool","Blend"]
    countries = ["Bangladesh","Vietnam","China","India","Turkey","Cambodia","Pakistan","Indonesia"]
    seasons = ["Spring","Summer","Fall","Winter"]
    colors = ["Black","White","Blue","Red","Green","Beige","Gray","Pink","Yellow","Brown"]

    columns = [
        "id","brand","product_type","material","color","price_usd",
        "production_country","co2_kg","water_l","recycled_pct",
        "labor_rating","collection_season","release_date"
    ]

    def row(i):
        return [
            i,
            random.choice(brands),
            random.choice(product_types),
            random.choice(materials),
            random.choice(colors),
            round(random.uniform(4.99, 89.99), 2),
            random.choice(countries),
            round(random.uniform(1.5, 45.0), 2),
            round(random.uniform(50, 2500), 1),
            round(random.uniform(0, 60), 1),
            random.choice(["A","B","C","D","E"]),
            random.choice(seasons),
            rand_date(2022, 2025),
        ]

    return columns, row

def schema_generic(industry_name):
    columns = [
        "id","industry","entity","event_date","metric_a","metric_b",
        "metric_c","region","category","status"
    ]
    entities = ["Alpha","Beta","Gamma","Delta","Omega"]
    regions = ["NA","EU","APAC","LATAM","MEA"]
    categories = ["Standard","Premium","Enterprise","SMB"]
    status = ["active","inactive","pending","closed"]

    def row(i):
        return [
            i,
            industry_name,
            random.choice(entities),
            rand_date(2020, 2025),
            round(random.uniform(0, 1000), 2),
            round(random.uniform(0, 100), 2),
            round(random.uniform(0, 10), 2),
            random.choice(regions),
            random.choice(categories),
            random.choice(status),
        ]

    return columns, row

SCHEMAS = {
    "fast fashion": schema_fast_fashion,
}

def generate_synthetic_df(industry: str, rows: int = 200) -> pd.DataFrame:
    random.seed(random_seed)
    key = industry.strip().lower()
    if key in SCHEMAS:
        columns, row_fn = SCHEMAS[key]()
    else:
        columns, row_fn = schema_generic(industry)

    data = [row_fn(i) for i in range(1, rows + 1)]
    return pd.DataFrame(data, columns=columns)

# =========================
# Clustering helper
# =========================
def prepare_for_kmeans(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return None, None
    scaled = (numeric_df - numeric_df.mean()) / (numeric_df.std(ddof=0) + 1e-9)
    return numeric_df, scaled


# =========================
# UI — Step 1
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

    st.success("Industry received. Fetching Wikipedia sources...")

    # =========================
    # Step 2 — Wikipedia sources
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 2 — Top Wikipedia sources</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>These are the five most relevant pages used to generate the report.</div>",
        unsafe_allow_html=True
    )

    with st.spinner("Retrieving the five most relevant Wikipedia pages…"):
        docs = retrieve_wikipedia_docs(industry.strip(), k=5)
        urls = extract_urls(docs)

    if not urls:
        st.error("No Wikipedia pages found. Try a more specific industry term.")
        st.stop()

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
    # Step 3 — Industry report
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)

    sources_text = build_sources_text(docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    system_prompt = (
        "You are a market research assistant for a business analyst at a large corporation.\n"
        "The analyst is evaluating a potential acquisition target in this industry.\n"
        "Write a concise industry briefing STRICTLY based on the provided Wikipedia sources.\n"
        "Do NOT use outside knowledge.\n"
        "When you make a factual claim, add a citation in the form [Source #].\n"
        "If the sources do not support a claim, write: 'Not specified in the sources.'\n"
        "Keep the full report under 500 words."
    )

    user_prompt = (
        f"Industry: {industry.strip()}\n\n"
        "Context: You are preparing this for a business analyst evaluating an acquisition target in this industry.\n"
        f"Report focus: {report_focus}.\n"
        f"Detail level: {detail_level}.\n"
        "Write a <500 word business analyst briefing using ONLY the sources below.\n\n"
        "Required structure (use these headings):\n"
        "1) Executive snapshot (2–3 sentences)\n"
        "2) Scope and definition\n"
        "3) Value chain / key segments\n"
        "4) Demand drivers and primary use-cases\n"
        "5) Challenges / constraints / notable developments (only if stated)\n"
        "6) What to research next (3–5 bullet points)\n\n"
        "Rules:\n"
        "- Cite sources as [Source 1], [Source 2], etc.\n"
        "- Do not introduce facts not present in the sources.\n\n"
        f"SOURCES:\n{sources_text}"
    )

    with st.spinner("Generating industry briefing…"):
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        report = cap_500_words(response.content)

    report = re.sub(r"(?m)^#+\s*", "", report)
    report = re.sub(r"(?m)^\s*\d+\)\s*(.+)$", r"<div class=\"section-title\">\1</div>", report)
    report = re.sub(r"(?m)^\s*[-*]\s*", "", report).strip()

    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # Synthetic Dataset & Analyst-Useful Visuals
    # =========================
    st.markdown("<h3 class='blue-accent'>Synthetic Dataset & Analyst-Useful Visuals</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>A synthetic dataset is generated from the industry name and used to create analyst-friendly visuals.</div>",
        unsafe_allow_html=True
    )

    synthetic_df = generate_synthetic_df(industry.strip(), rows=200)

    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download synthetic dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"{industry.strip().lower().replace(' ', '_')}_synthetic.csv",
        mime="text/csv",
    )

    required_cols = {
        "price_usd", "co2_kg", "water_l", "recycled_pct",
        "brand", "material", "production_country"
    }
    if required_cols.issubset(set(synthetic_df.columns)):
        st.markdown("<div class='blue-accent'>Price Distribution</div>", unsafe_allow_html=True)
        hist = (
            alt.Chart(synthetic_df)
            .mark_bar()
            .encode(
                alt.X("price_usd:Q", bin=alt.Bin(maxbins=20), title="Price (USD)"),
                alt.Y("count()", title="Count"),
            )
        )
        st.altair_chart(hist, use_container_width=True)

        st.markdown("<div class='blue-accent'>Average Price by Brand</div>", unsafe_allow_html=True)
        avg_price = (
            synthetic_df.groupby("brand", as_index=False)["price_usd"].mean()
            .sort_values("price_usd", ascending=False)
        )
        st.altair_chart(
            alt.Chart(avg_price)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", sort="-y", title="Brand"),
                y=alt.Y("price_usd:Q", title="Avg Price (USD)"),
                tooltip=["brand", alt.Tooltip("price_usd:Q", format=".2f")],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>CO₂ vs Price (Environmental Cost Proxy)</div>", unsafe_allow_html=True)
        scatter = (
            alt.Chart(synthetic_df)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("price_usd:Q", title="Price (USD)"),
                y=alt.Y("co2_kg:Q", title="CO₂ (kg)"),
                tooltip=["brand", "price_usd", "co2_kg"],
            )
        )
        trend = scatter.transform_regression("price_usd", "co2_kg").mark_line(color="#2563EB")
        st.altair_chart(scatter + trend, use_container_width=True)

        st.markdown("<div class='blue-accent'>Production Country Concentration</div>", unsafe_allow_html=True)
        country_counts = (
            synthetic_df["production_country"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "country", "production_country": "count"})
        )
        st.altair_chart(
            alt.Chart(country_counts)
            .mark_bar()
            .encode(
                x=alt.X("country:N", sort="-y", title="Country"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["country", "count"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Recycled % by Material</div>", unsafe_allow_html=True)
        recycled = (
            synthetic_df.groupby("material", as_index=False)["recycled_pct"].mean()
            .sort_values("recycled_pct", ascending=False)
        )
        st.altair_chart(
            alt.Chart(recycled)
            .mark_bar()
            .encode(
                x=alt.X("material:N", sort="-y", title="Material"),
                y=alt.Y("recycled_pct:Q", title="Avg Recycled %"),
                tooltip=["material", alt.Tooltip("recycled_pct:Q", format=".1f")],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Water Usage vs Price</div>", unsafe_allow_html=True)
        water_scatter = (
            alt.Chart(synthetic_df)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("price_usd:Q", title="Price (USD)"),
                y=alt.Y("water_l:Q", title="Water (L)"),
                tooltip=["brand", "price_usd", "water_l"],
            )
        )
        st.altair_chart(water_scatter, use_container_width=True)

        st.markdown("<div class='blue-accent'>Synthetic Cluster Map</div>", unsafe_allow_html=True)
        st.write(
            "Groups synthetic records into 3 clusters using price, CO₂, water, and recycled content. "
            "This helps highlight distinct operational profiles within the dataset."
        )
        cluster_features = synthetic_df[["price_usd", "co2_kg", "water_l", "recycled_pct"]].copy()
        cluster_scaled = (cluster_features - cluster_features.mean()) / (cluster_features.std(ddof=0) + 1e-9)
        km_syn = KMeans(n_clusters=3, n_init=10, random_state=42)
        synthetic_df["synthetic_cluster"] = km_syn.fit_predict(cluster_scaled).astype(str)

        cluster_chart = (
            alt.Chart(synthetic_df)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X("price_usd:Q", title="Price (USD)"),
                y=alt.Y("co2_kg:Q", title="CO₂ (kg)"),
                color=alt.Color("synthetic_cluster:N", title="Cluster"),
                tooltip=["brand", "price_usd", "co2_kg", "water_l", "recycled_pct", "synthetic_cluster"],
            )
        )
        st.altair_chart(cluster_chart, use_container_width=True)

    else:
        st.info("Synthetic dataset schema did not include expected fields for analyst visuals.")

    # =========================
    # Clustering — only if CSV uploaded
    # =========================
    st.markdown("<h3 class='blue-accent'>Clustering (K-means)</h3>", unsafe_allow_html=True)
    if uploaded_csv is None:
        st.info("Upload a CSV to enable clustering.")
        df = None
    else:
        try:
            raw = uploaded_csv.getvalue().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
        except Exception:
            st.warning("Could not read the CSV. Please upload a valid CSV file.")
            df = None

    if df is not None:
        numeric_df, scaled = prepare_for_kmeans(df)
        if numeric_df is None:
            st.warning("CSV needs at least two numeric columns for clustering.")
        else:
            km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
            clusters = km.fit_predict(scaled)
            df_plot = numeric_df.copy()
            df_plot["cluster"] = clusters.astype(str)

            x_col, y_col = numeric_df.columns[:2]
            chart = (
                alt.Chart(df_plot)
                .mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col, title=y_col),
                    color=alt.Color("cluster:N", title="Cluster"),
                    tooltip=[x_col, y_col, "cluster"],
                )
            )
            st.altair_chart(chart, use_container_width=True)
