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

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)

    return unique[:5]


def build_sources_text(docs) -> str:
    """
    Build context ONLY from the retrieved Wikipedia pages.
    We number sources so the model can cite like [Source 1].
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        url = (d.metadata or {}).get("source", "")
        text = (d.page_content or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = text[:2600]  # bounded context per source

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


def schema_healthcare():
    providers = ["St. Mary Hospital","City Clinic","MedPrime","CarePlus","BlueLeaf Health"]
    departments = ["Cardiology","Oncology","Pediatrics","Orthopedics","Neurology","ER"]
    insurance = ["Private","Medicare","Medicaid","Self-Pay"]
    diagnosis = ["Hypertension","Diabetes","Asthma","Flu","Arthritis","Migraine"]

    columns = [
        "id","provider","department","visit_date","diagnosis",
        "length_of_stay_days","total_cost_usd","insurance_type","readmitted"
    ]

    def row(i):
        return [
            i,
            random.choice(providers),
            random.choice(departments),
            rand_date(2021, 2025),
            random.choice(diagnosis),
            random.randint(0, 14),
            round(random.uniform(120, 25000), 2),
            random.choice(insurance),
            random.choice(["yes","no"]),
        ]

    return columns, row


def schema_ecommerce():
    categories = ["Electronics","Home","Beauty","Sports","Toys","Fashion","Books"]
    channels = ["Web","Mobile","Marketplace"]
    regions = ["NA","EU","APAC","LATAM"]

    columns = [
        "id","order_date","category","unit_price_usd","units",
        "channel","region","discount_pct","shipping_days","returned"
    ]

    def row(i):
        return [
            i,
            rand_date(2021, 2025),
            random.choice(categories),
            round(random.uniform(5, 1500), 2),
            random.randint(1, 8),
            random.choice(channels),
            random.choice(regions),
            round(random.uniform(0, 40), 1),
            random.randint(1, 10),
            random.choice(["yes","no"]),
        ]

    return columns, row


def schema_semiconductors():
    companies = ["TSMC","Samsung","Intel","GlobalFoundries","SMIC","Micron","SK hynix","Texas Instruments"]
    segments = ["Foundry","Memory","Logic","Analog","Power","Mixed-Signal"]
    nodes = [3, 5, 7, 10, 14, 28, 45, 65]
    countries = ["Taiwan","South Korea","United States","China","Japan","Germany","Singapore"]
    materials = ["Silicon","Gallium Nitride","Silicon Carbide","Copper","Low-k Dielectric"]

    columns = [
        "id","brand","segment","node_nm","wafer_cost_usd","yield_pct",
        "price_usd","production_country","material","co2_kg","water_l","recycled_pct","ship_date"
    ]

    def row(i):
        node = random.choice(nodes)
        yield_pct = round(random.uniform(70, 98), 1)
        wafer_cost = round(random.uniform(800, 20000), 2)
        price = round(wafer_cost / max(1, yield_pct / 100) / random.uniform(30, 120), 2)
        return [
            i,
            random.choice(companies),
            random.choice(segments),
            node,
            wafer_cost,
            yield_pct,
            price,
            random.choice(countries),
            random.choice(materials),
            round(random.uniform(5, 120), 2),
            round(random.uniform(200, 1800), 1),
            round(random.uniform(0, 35), 1),
            rand_date(2022, 2025),
        ]

    return columns, row


def schema_ev_batteries():
    companies = ["CATL","LG Energy Solution","Panasonic","BYD","SK On","Northvolt","Samsung SDI"]
    chemistries = ["LFP","NMC","NCA","LMFP","LTO"]
    formats = ["Pouch","Prismatic","Cylindrical"]
    countries = ["China","South Korea","Japan","United States","Germany","Sweden","Poland"]
    materials = ["Lithium","Nickel","Cobalt","Manganese","Phosphate","Graphite"]

    columns = [
        "id","brand","chemistry","cell_format","energy_density_whkg","cycle_life",
        "price_usd","production_country","material","co2_kg","water_l","recycled_pct","ship_date"
    ]

    def row(i):
        density = round(random.uniform(140, 280), 1)
        cycle_life = random.randint(800, 3500)
        price = round(random.uniform(55, 220), 2)
        return [
            i,
            random.choice(companies),
            random.choice(chemistries),
            random.choice(formats),
            density,
            cycle_life,
            price,
            random.choice(countries),
            random.choice(materials),
            round(random.uniform(8, 160), 2),
            round(random.uniform(120, 1600), 1),
            round(random.uniform(0, 45), 1),
            rand_date(2022, 2025),
        ]

    return columns, row


def schema_retail():
    retailers = ["Walmart","Target","Carrefour","Tesco","Costco","Kroger","Aldi","Lidl"]
    categories = ["Grocery","Home","Electronics","Apparel","Health","Beauty","Toys"]
    channels = ["In-store","Online","Omni"]
    countries = ["United States","United Kingdom","France","Germany","Canada","Spain","Italy"]
    materials = ["Paper","Plastic","Mixed","Recycled"]

    columns = [
        "id","brand","category","channel","basket_size","price_usd",
        "production_country","material","co2_kg","water_l","recycled_pct","order_date"
    ]

    def row(i):
        price = round(random.uniform(10, 350), 2)
        basket = random.randint(1, 40)
        return [
            i,
            random.choice(retailers),
            random.choice(categories),
            random.choice(channels),
            basket,
            price,
            random.choice(countries),
            random.choice(materials),
            round(random.uniform(1, 35), 2),
            round(random.uniform(20, 600), 1),
            round(random.uniform(5, 60), 1),
            rand_date(2022, 2025),
        ]

    return columns, row


def schema_logistics():
    providers = ["DHL","FedEx","UPS","Maersk","DB Schenker","Kuehne+Nagel","XPO"]
    modes = ["Air","Ocean","Rail","Road"]
    regions = ["NA","EU","APAC","LATAM","MEA"]
    countries = ["United States","Germany","China","Netherlands","Singapore","United Kingdom"]
    materials = ["Cardboard","Plastic","Reusable","Mixed"]

    columns = [
        "id","brand","mode","region","distance_km","price_usd",
        "production_country","material","co2_kg","water_l","recycled_pct","ship_date"
    ]

    def row(i):
        distance = random.randint(50, 12000)
        price = round(random.uniform(120, 9000), 2)
        return [
            i,
            random.choice(providers),
            random.choice(modes),
            random.choice(regions),
            distance,
            price,
            random.choice(countries),
            random.choice(materials),
            round(random.uniform(5, 220), 2),
            round(random.uniform(30, 900), 1),
            round(random.uniform(5, 55), 1),
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
    "healthcare": schema_healthcare,
    "ecommerce": schema_ecommerce,
    "e-commerce": schema_ecommerce,
    "semiconductors": schema_semiconductors,
    "semiconductor": schema_semiconductors,
    "ev batteries": schema_ev_batteries,
    "ev battery": schema_ev_batteries,
    "retail": schema_retail,
    "logistics": schema_logistics,
}

SCHEMA_KEYWORDS = [
    (["fast fashion", "apparel"], schema_fast_fashion),
    (["semiconductor", "chip", "foundry", "ic", "wafer", "fab"], schema_semiconductors),
    (["ev battery", "battery", "lithium", "lithium-ion", "lithium ion", "cell"], schema_ev_batteries),
    (["ecommerce", "e-commerce", "online retail", "marketplace"], schema_ecommerce),
    (["retail", "grocery", "department store", "supermarket"], schema_retail),
    (["logistics", "shipping", "freight", "supply chain", "warehouse", "3pl"], schema_logistics),
    (["healthcare", "hospital", "clinic"], schema_healthcare),
]


def pick_schema(industry: str):
    key = industry.strip().lower()
    if key in SCHEMAS:
        return SCHEMAS[key]()
    for keywords, schema_fn in SCHEMA_KEYWORDS:
        if any(kw in key for kw in keywords):
            return schema_fn()
    return schema_generic(industry)


def generate_synthetic_df(industry: str, rows: int = 200) -> pd.DataFrame:
    random.seed(random_seed)
    columns, row_fn = pick_schema(industry)

    data = [row_fn(i) for i in range(1, rows + 1)]
    return pd.DataFrame(data, columns=columns)

# =========================
# Report analysis helpers
# =========================
def split_report_sections(report_text: str):
    sections = []
    current_title = "Report"
    current_lines = []
    for line in report_text.splitlines():
        if re.match(r"^\s*\d+\)\s+", line):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = re.sub(r"^\s*\d+\)\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def section_confidence_score(section_text: str):
    # Heuristic: combine length and citation density
    words = section_text.split()
    word_count = max(1, len(words))
    citations = len(re.findall(r"\[Source\s+\d+\]", section_text))
    citation_density = min(1.0, citations / max(1, word_count / 60))
    length_score = min(1.0, word_count / 120)
    score = int((0.6 * citation_density + 0.4 * length_score) * 100)
    return max(10, min(100, score))

# =========================
# Clustering helpers
# =========================
def prepare_for_kmeans(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return None, None
    # z-score scaling
    scaled = (numeric_df - numeric_df.mean()) / (numeric_df.std(ddof=0) + 1e-9)
    return numeric_df, scaled



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
    # Q1 validation
    if not industry_is_valid(industry):
        st.warning("Please enter an industry to continue.")
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

    st.info("The report below is generated exclusively from the five Wikipedia pages listed above.")

    # =========================
    # Q3 — Industry report (<500 words), based on those five pages
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Business-analyst style briefing with traceable citations in the form [Source #].</div>",
        unsafe_allow_html=True
    )

    sources_text = build_sources_text(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # BA-brief prompt + explicit grounding + source citations
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

    # Remove markdown heading hashes if present
    report = re.sub(r"(?m)^#+\s*", "", report)
    report = re.sub(r"(?m)^\s*\d+\)\s*(.+)$", r"<div class=\"section-title\">\1</div>", report)
    report = re.sub(r"(?m)^\s*[-*]\s*", "", report).strip()

    word_count = len(report.split())
    st.caption(f"Word count: {word_count} / 500")

    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # Confidence scoring by section
    # =========================
    st.markdown("<h3 class='blue-accent'>Section Confidence Scores</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Heuristic confidence based on section length and citation density.</div>",
        unsafe_allow_html=True
    )
    sections = split_report_sections(report)
    if sections:
        conf_rows = []
        for title, text in sections:
            conf_rows.append(
                {
                    "Section": title,
                    "Confidence": section_confidence_score(text),
                    "Citations": len(re.findall(r"\[Source\s+\d+\]", text)),
                }
            )
        conf_df = pd.DataFrame(conf_rows)
        conf_chart = (
            alt.Chart(conf_df)
            .mark_bar()
            .encode(
                x=alt.X("Section:N", sort=None, title="Section"),
                y=alt.Y("Confidence:Q", title="Confidence (0–100)"),
                color=alt.Color("Confidence:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Section", "Confidence", "Citations"],
            )
        )
        st.altair_chart(conf_chart, use_container_width=True)
    else:
        st.caption("Could not detect section headings for confidence scoring.")

    # =========================
    # Visuals — synthetic dataset (analyst-friendly)
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

    # Only render enhanced visuals if expected columns exist
    required_cols = {
        "price_usd", "co2_kg", "water_l", "recycled_pct",
        "brand", "material", "production_country"
    }
    if required_cols.issubset(set(synthetic_df.columns)):
        # Price distribution
        st.markdown("<div class='blue-accent'>Price Distribution</div>", unsafe_allow_html=True)
        st.write("Shows how product prices are distributed across the synthetic catalog.")
        hist = (
            alt.Chart(synthetic_df)
            .mark_bar()
            .encode(
                alt.X("price_usd:Q", bin=alt.Bin(maxbins=20), title="Price (USD)"),
                alt.Y("count()", title="Count"),
            )
        )
        st.altair_chart(hist, use_container_width=True)

        # Avg price by brand
        st.markdown("<div class='blue-accent'>Average Price by Brand</div>", unsafe_allow_html=True)
        st.write("Compares brand positioning based on average price level.")
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

        # CO2 vs price (trendline)
        st.markdown("<div class='blue-accent'>CO₂ vs Price (Environmental Cost Proxy)</div>", unsafe_allow_html=True)
        st.write("Highlights whether higher prices correlate with higher (or lower) CO₂ impact.")
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

        # Production country concentration
        st.markdown("<div class='blue-accent'>Production Country Concentration</div>", unsafe_allow_html=True)
        st.write("Shows where production is concentrated in the synthetic dataset, highlighting geographic dependencies.")
        country_counts = synthetic_df["production_country"].value_counts().reset_index(name="count")
        country_counts = country_counts.rename(columns={"index": "country"})
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

        # Recycled % by material
        st.markdown("<div class='blue-accent'>Recycled % by Material</div>", unsafe_allow_html=True)
        st.write("Compares circularity across materials to identify sustainability leaders.")
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

        # Water usage vs price
        st.markdown("<div class='blue-accent'>Water Usage vs Price</div>", unsafe_allow_html=True)
        st.write("Shows whether pricing aligns with water intensity in production.")
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

        # Trend-style clustering on synthetic data
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

        # Price spread by material (boxplot for variety)
        st.markdown("<div class='blue-accent'>Price Spread by Material (Boxplot)</div>", unsafe_allow_html=True)
        st.write("Shows median and variability of prices by material to surface cost dispersion.")
        box = (
            alt.Chart(synthetic_df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("material:N", sort="-y", title="Material"),
                y=alt.Y("price_usd:Q", title="Price (USD)"),
            )
        )
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("Synthetic dataset schema did not include expected fields for analyst visuals.")

    # =========================
    # Clustering — only if CSV uploaded (K-means)
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

            # Pick first two numeric columns for a 2D view
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

            st.caption("Clusters are based on numeric columns only. Non-numeric columns are ignored.")

    # =========================
    # Source bias heatmap (citations by section)
    # =========================
    st.markdown("<h3 class='blue-accent'>Source Bias Heatmap</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Shows which sources are most cited across report sections.</div>",
        unsafe_allow_html=True
    )
    if sections:
        heat_rows = []
        for title, text in sections:
            for i in range(1, len(docs) + 1):
                count = len(re.findall(rf"\[Source\s+{i}\]", text))
                heat_rows.append({"Section": title, "Source": f"Source {i}", "Count": count})
        heat_df = pd.DataFrame(heat_rows)
        heat_chart = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("Source:N", title="Source"),
                y=alt.Y("Section:N", title="Section"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Citations"),
                tooltip=["Section", "Source", "Count"],
            )
        )
        st.altair_chart(heat_chart, use_container_width=True)
    else:
        st.caption("No sections detected to build heatmap.")
