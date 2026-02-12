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


def schema_fast_fashion():
    brands = ["Zara", "H&M", "Shein", "Forever 21", "Uniqlo", "Primark", "Boohoo", "Mango", "ASOS"]
    product_types = ["T-shirt", "Jeans", "Dress", "Hoodie", "Sweater", "Skirt", "Shorts", "Jacket"]
    materials = ["Cotton", "Polyester", "Viscose", "Linen", "Nylon", "Acrylic", "Wool", "Blend"]
    countries = ["Bangladesh", "Vietnam", "China", "India", "Turkey", "Cambodia", "Pakistan", "Indonesia"]
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    colors = ["Black", "White", "Blue", "Red", "Green", "Beige", "Gray", "Pink", "Yellow", "Brown"]

    columns = [
        "id", "brand", "product_type", "material", "color", "price_usd",
        "production_country", "co2_kg", "water_l", "recycled_pct",
        "labor_rating", "collection_season", "release_date"
    ]

    def row(i):
        return [
            i,
            np.random.choice(brands),
            np.random.choice(product_types),
            np.random.choice(materials),
            np.random.choice(colors),
            round(float(np.random.uniform(4.99, 89.99)), 2),
            np.random.choice(countries),
            round(float(np.random.uniform(1.5, 45.0)), 2),
            round(float(np.random.uniform(50, 2500)), 1),
            round(float(np.random.uniform(0, 60)), 1),
            np.random.choice(["A", "B", "C", "D", "E"]),
            np.random.choice(seasons),
            rand_date(2022, 2025),
        ]

    return columns, row


def schema_healthcare():
    providers = ["St. Mary Hospital", "City Clinic", "MedPrime", "CarePlus", "BlueLeaf Health"]
    departments = ["Cardiology", "Oncology", "Pediatrics", "Orthopedics", "Neurology", "ER"]
    insurance = ["Private", "Medicare", "Medicaid", "Self-Pay"]
    diagnosis = ["Hypertension", "Diabetes", "Asthma", "Flu", "Arthritis", "Migraine"]

    columns = [
        "id", "provider", "department", "visit_date", "diagnosis",
        "length_of_stay_days", "total_cost_usd", "insurance_type", "readmitted"
    ]

    def row(i):
        return [
            i,
            np.random.choice(providers),
            np.random.choice(departments),
            rand_date(2021, 2025),
            np.random.choice(diagnosis),
            int(np.random.randint(0, 14)),
            round(float(np.random.uniform(120, 25000)), 2),
            np.random.choice(insurance),
            np.random.choice(["yes", "no"]),
        ]

    return columns, row


def schema_ecommerce():
    categories = ["Electronics", "Home", "Beauty", "Sports", "Toys", "Fashion", "Books"]
    channels = ["Web", "Mobile", "Marketplace"]
    regions = ["NA", "EU", "APAC", "LATAM"]

    columns = [
        "id", "order_date", "category", "unit_price_usd", "units",
        "channel", "region", "discount_pct", "shipping_days", "returned"
    ]

    def row(i):
        return [
            i,
            rand_date(2021, 2025),
            np.random.choice(categories),
            round(float(np.random.uniform(5, 1500)), 2),
            int(np.random.randint(1, 8)),
            np.random.choice(channels),
            np.random.choice(regions),
            round(float(np.random.uniform(0, 40)), 1),
            int(np.random.randint(1, 10)),
            np.random.choice(["yes", "no"]),
        ]

    return columns, row


def schema_semiconductors():
    companies = ["TSMC", "Samsung", "Intel", "SK Hynix", "Micron", "GlobalFoundries", "UMC"]
    segments = ["Foundry", "Memory", "Logic", "Analog", "Power", "RF"]
    nodes = ["5nm", "7nm", "10nm", "14nm", "22nm", "28nm", "40nm"]
    regions = ["US", "Taiwan", "Korea", "Japan", "EU", "China"]

    columns = [
        "id", "company", "segment", "node", "region", "wafer_starts_k",
        "yield_pct", "asp_usd", "capex_bil", "fab_utilization_pct", "date"
    ]

    def row(i):
        return [
            i,
            np.random.choice(companies),
            np.random.choice(segments),
            np.random.choice(nodes),
            np.random.choice(regions),
            round(float(np.random.uniform(10, 180)), 1),
            round(float(np.random.uniform(70, 99)), 1),
            round(float(np.random.uniform(600, 4000)), 1),
            round(float(np.random.uniform(0.5, 15)), 2),
            round(float(np.random.uniform(55, 95)), 1),
            rand_date(2021, 2025),
        ]

    return columns, row


def schema_ev_batteries():
    makers = ["CATL", "LG Energy", "Panasonic", "BYD", "SK On", "Samsung SDI"]
    chemistries = ["LFP", "NMC", "NCA"]
    regions = ["China", "Korea", "Japan", "EU", "US"]
    segments = ["Passenger EV", "Commercial EV", "Energy Storage"]

    columns = [
        "id", "maker", "chemistry", "segment", "region",
        "cost_per_kwh", "energy_density_whkg", "cycle_life",
        "capacity_gwh", "date"
    ]

    def row(i):
        return [
            i,
            np.random.choice(makers),
            np.random.choice(chemistries),
            np.random.choice(segments),
            np.random.choice(regions),
            round(float(np.random.uniform(70, 180)), 2),
            round(float(np.random.uniform(120, 300)), 1),
            int(np.random.randint(800, 4000)),
            round(float(np.random.uniform(1, 50)), 2),
            rand_date(2021, 2025),
        ]

    return columns, row


def schema_retail():
    banners = ["Walmart", "Target", "Carrefour", "Tesco", "Costco", "Aldi", "Lidl"]
    regions = ["NA", "EU", "APAC", "LATAM"]
    formats = ["Hypermarket", "Supermarket", "Warehouse", "Discount", "Online"]

    columns = [
        "id", "banner", "format", "region", "store_sales_usd_m",
        "same_store_growth_pct", "foot_traffic_idx", "basket_size_usd",
        "private_label_pct", "date"
    ]

    def row(i):
        return [
            i,
            np.random.choice(banners),
            np.random.choice(formats),
            np.random.choice(regions),
            round(float(np.random.uniform(5, 400)), 2),
            round(float(np.random.uniform(-5, 15)), 2),
            round(float(np.random.uniform(70, 130)), 1),
            round(float(np.random.uniform(15, 120)), 2),
            round(float(np.random.uniform(5, 35)), 1),
            rand_date(2021, 2025),
        ]

    return columns, row


def schema_logistics():
    carriers = ["DHL", "FedEx", "UPS", "Maersk", "MSC", "DP World", "XPO"]
    modes = ["Air", "Ocean", "Road", "Rail"]
    regions = ["NA", "EU", "APAC", "LATAM", "MEA"]

    columns = [
        "id", "carrier", "mode", "region", "shipment_volume_k",
        "on_time_pct", "cost_per_shipment_usd", "fuel_cost_index",
        "capacity_util_pct", "date"
    ]

    def row(i):
        return [
            i,
            np.random.choice(carriers),
            np.random.choice(modes),
            np.random.choice(regions),
            round(float(np.random.uniform(20, 700)), 1),
            round(float(np.random.uniform(75, 98)), 1),
            round(float(np.random.uniform(50, 900)), 2),
            round(float(np.random.uniform(80, 150)), 1),
            round(float(np.random.uniform(50, 95)), 1),
            rand_date(2021, 2025),
        ]

    return columns, row


def schema_generic(industry_name):
    columns = [
        "id", "industry", "entity", "event_date", "metric_a", "metric_b",
        "metric_c", "region", "category", "status"
    ]
    entities = ["Alpha", "Beta", "Gamma", "Delta", "Omega"]
    regions = ["NA", "EU", "APAC", "LATAM", "MEA"]
    categories = ["Standard", "Premium", "Enterprise", "SMB"]
    status = ["active", "inactive", "pending", "closed"]

    def row(i):
        return [
            i,
            industry_name,
            np.random.choice(entities),
            rand_date(2020, 2025),
            round(float(np.random.uniform(0, 1000)), 2),
            round(float(np.random.uniform(0, 100)), 2),
            round(float(np.random.uniform(0, 10)), 2),
            np.random.choice(regions),
            np.random.choice(categories),
            np.random.choice(status),
        ]

    return columns, row


SCHEMAS = {
    "fast fashion": schema_fast_fashion,
    "healthcare": schema_healthcare,
    "ecommerce": schema_ecommerce,
    "semiconductors": schema_semiconductors,
    "ev batteries": schema_ev_batteries,
    "retail": schema_retail,
    "logistics": schema_logistics,
}

SCHEMA_KEYWORDS = {
    "fast fashion": ["fashion", "apparel", "textile"],
    "healthcare": ["health", "medical", "hospital", "pharma"],
    "ecommerce": ["ecommerce", "e-commerce", "online retail", "marketplace"],
    "semiconductors": ["semiconductor", "chip", "foundry", "fab"],
    "ev batteries": ["battery", "ev", "electric vehicle", "lithium"],
    "retail": ["retail", "supermarket", "grocery", "store"],
    "logistics": ["logistics", "shipping", "freight", "supply chain"],
}


def pick_schema(industry: str):
    key = industry.strip().lower()
    if key in SCHEMAS:
        return SCHEMAS[key]
    for schema_name, kws in SCHEMA_KEYWORDS.items():
        if any(k in key for k in kws):
            return SCHEMAS[schema_name]
    return lambda: schema_generic(industry)


def generate_synthetic_df(industry: str, rows: int = 240) -> pd.DataFrame:
    np.random.seed(abs(hash(industry)) % (2**32))
    schema_fn = pick_schema(industry)
    columns, row_fn = schema_fn()
    rows_list = [row_fn(i + 1) for i in range(rows)]
    return pd.DataFrame(rows_list, columns=columns)


def enrich_for_ma(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(industry)) % (2**32))
    df = df.copy()

    if "company" not in df.columns:
        df["company"] = df.get("brand", df.get("provider", df.get("maker", df.get("carrier", "Company"))))

    df["company"] = df["company"].astype(str)

    df["segment"] = df.get("segment", None)
    if df["segment"].isnull().all():
        df["segment"] = rng.choice(
            ["Core", "Premium", "Value", "Emerging"], size=len(df), replace=True
        )

    df["region"] = df.get("region", None)
    if df["region"].isnull().all():
        df["region"] = rng.choice(["NA", "EU", "APAC", "LATAM", "MEA"], size=len(df), replace=True)

    date_col = None
    for c in ["release_date", "visit_date", "order_date", "date", "event_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        df["event_date"] = pd.to_datetime(
            rng.choice(pd.date_range("2021-01-01", "2025-12-31"), size=len(df))
        )
        date_col = "event_date"
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.to_period("M").astype(str)

    df["market_share_pct"] = np.clip(rng.normal(5, 2, len(df)), 0.2, 15)
    df["revenue_usd_m"] = np.clip(rng.normal(250, 120, len(df)), 20, 1200)
    df["revenue_growth_pct"] = np.clip(rng.normal(8, 6, len(df)), -10, 30)
    df["ebitda_margin_pct"] = np.clip(rng.normal(18, 7, len(df)), 2, 45)
    df["capex_intensity_pct"] = np.clip(rng.normal(6, 3, len(df)), 1, 20)
    df["debt_to_equity"] = np.clip(rng.normal(1.1, 0.6, len(df)), 0, 4.5)

    df["supply_concentration"] = np.clip(rng.normal(0.55, 0.2, len(df)), 0, 1)
    df["risk_score"] = np.clip(
        0.5 * (1 - df["supply_concentration"]) + 0.5 * (1 - (df["ebitda_margin_pct"] / 50)),
        0, 1
    )

    return df


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

    # =========================
    # Q2 — URLs of five most relevant Wikipedia pages
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
        st.warning("I couldn't find reliable Wikipedia matches. Please уточнить or rephrase the industry.")
        st.info("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    # ... rest of your app continues here ...
