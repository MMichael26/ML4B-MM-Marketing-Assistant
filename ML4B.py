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
# Local Development (VS Code) instructions
# =========================
with st.expander("Local development setup (optional)", expanded=False):
    st.markdown("<h3 class='blue-accent'>Local Development (VS Code)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'><b>Where the key goes (locally)</b><br>You include the key only in your local environment, not in code.</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'><b>Option A (recommended): environment variable</b><br><b>Mac/Linux</b></div>", unsafe_allow_html=True)
    st.code('export OPENAI_API_KEY="sk-..."', language="bash")

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


def enrich_for_ma(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    enriched = df.copy()
    seed = abs(hash(industry)) % (2**32)
    rng = random.Random(seed)

    if "brand" in enriched.columns:
        enriched["company"] = enriched["brand"]
    elif "provider" in enriched.columns:
        enriched["company"] = enriched["provider"]
    elif "entity" in enriched.columns:
        enriched["company"] = enriched["entity"]
    else:
        enriched["company"] = [f"{industry.title()} Co {i+1}" for i in range(len(enriched))]

    if "segment" in enriched.columns:
        enriched["segment"] = enriched["segment"]
    elif "category" in enriched.columns:
        enriched["segment"] = enriched["category"]
    elif "product_type" in enriched.columns:
        enriched["segment"] = enriched["product_type"]
    elif "mode" in enriched.columns:
        enriched["segment"] = enriched["mode"]
    else:
        enriched["segment"] = "General"

    if "region" not in enriched.columns:
        enriched["region"] = rng.choices(
            ["NA", "EU", "APAC", "LATAM", "MEA"],
            weights=[0.35, 0.25, 0.25, 0.1, 0.05],
            k=len(enriched),
        )

    companies = sorted(enriched["company"].unique().tolist())
    base = [rng.uniform(0.5, 8.0) for _ in companies]
    scale = 100.0 / sum(base)
    share_map = {c: round(b * scale, 2) for c, b in zip(companies, base)}
    enriched["market_share_pct"] = enriched["company"].map(share_map)

    rev_base = {c: rng.lognormvariate(3.2, 0.6) * 10 for c in companies}
    enriched["revenue_usd_m"] = [
        round(rev_base[c] * rng.uniform(0.7, 1.3), 2) for c in enriched["company"]
    ]

    enriched["revenue_growth_pct"] = [
        round(rng.uniform(-8, 25), 2) for _ in range(len(enriched))
    ]
    enriched["ebitda_margin_pct"] = [
        round(rng.uniform(5, 38), 2) for _ in range(len(enriched))
    ]
    enriched["capex_intensity_pct"] = [
        round(rng.uniform(2, 18), 2) for _ in range(len(enriched))
    ]
    enriched["debt_to_equity"] = [
        round(rng.uniform(0.0, 3.5), 2) for _ in range(len(enriched))
    ]

    enriched["supply_concentration"] = [
        round(rng.uniform(0.1, 0.9), 2) for _ in range(len(enriched))
    ]
    enriched["risk_score"] = [
        round(
            0.35 * (1 - min(1, g / 30)) +
            0.35 * min(1, d / 3.5) +
            0.30 * s,
            2
        )
        for g, d, s in zip(
            enriched["revenue_growth_pct"],
            enriched["debt_to_equity"],
            enriched["supply_concentration"],
        )
    ]

    return enriched

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
    if not industry_is_valid(industry):
        st.warning("Please enter an industry to continue.")
        st.stop()

    st.success("Industry received. Fetching Wikipedia sources...")

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

    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Business-analyst style briefing with traceable citations in the form [Source #].</div>",
        unsafe_allow_html=True
    )

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

    st.markdown("<h3 class='blue-accent'>Synthetic Dataset & M&A-Oriented Visuals</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>A synthetic dataset is generated and enriched with acquisition-style metrics for analyst screening.</div>",
        unsafe_allow_html=True
    )

    synthetic_df = generate_synthetic_df(industry.strip(), rows=240)
    synthetic_df = enrich_for_ma(synthetic_df, industry.strip())

    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download synthetic dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"{industry.strip().lower().replace(' ', '_')}_synthetic.csv",
        mime="text/csv",
    )

    required_cols = {
        "company","segment","region","market_share_pct","revenue_usd_m",
        "revenue_growth_pct","ebitda_margin_pct","capex_intensity_pct",
        "debt_to_equity","risk_score","supply_concentration"
    }
    if required_cols.issubset(set(synthetic_df.columns)):
        st.markdown("<div class='blue-accent'>Market Share (Top Companies)</div>", unsafe_allow_html=True)
        st.write("Highlights concentration and relative scale among leading companies.")
        share = (
            synthetic_df.groupby("company", as_index=False)["market_share_pct"].mean()
            .sort_values("market_share_pct", ascending=False)
            .head(10)
        )
        st.altair_chart(
            alt.Chart(share)
            .mark_bar()
            .encode(
                x=alt.X("company:N", sort="-y", title="Company"),
                y=alt.Y("market_share_pct:Q", title="Market Share (%)"),
                tooltip=["company", alt.Tooltip("market_share_pct:Q", format=".2f")],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Growth vs EBITDA Margin (Acquisition Lens)</div>", unsafe_allow_html=True)
        st.write("Positions companies by growth and profitability; top-right is most attractive.")
        gm = synthetic_df.groupby("company", as_index=False)[
            ["revenue_growth_pct","ebitda_margin_pct","market_share_pct"]
        ].mean()
        vline = gm["revenue_growth_pct"].median()
        hline = gm["ebitda_margin_pct"].median()
        base = alt.Chart(gm).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("revenue_growth_pct:Q", title="Revenue Growth (%)"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            size=alt.Size("market_share_pct:Q", title="Market Share"),
            tooltip=["company", "revenue_growth_pct", "ebitda_margin_pct", "market_share_pct"],
        )
        lines = (
            alt.Chart(pd.DataFrame({"x": [vline], "y": [hline]}))
            .mark_rule(color="#94A3B8")
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart(base + lines, use_container_width=True)

        st.markdown("<div class='blue-accent'>Revenue Distribution</div>", unsafe_allow_html=True)
        st.write("Shows how revenue is distributed across synthetic companies.")
        rev = (
            synthetic_df.groupby("company", as_index=False)["revenue_usd_m"].mean()
        )
        rev_chart = (
            alt.Chart(rev)
            .mark_bar()
            .encode(
                alt.X("revenue_usd_m:Q", bin=alt.Bin(maxbins=20), title="Revenue (USD, millions)"),
                alt.Y("count()", title="Count of Companies"),
            )
        )
        st.altair_chart(rev_chart, use_container_width=True)

        st.markdown("<div class='blue-accent'>Capex Intensity vs EBITDA Margin</div>", unsafe_allow_html=True)
        st.write("Reveals whether higher capex needs are correlated with stronger margins.")
        cm = synthetic_df.groupby("company", as_index=False)[
            ["capex_intensity_pct","ebitda_margin_pct"]
        ].mean()
        st.altair_chart(
            alt.Chart(cm)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X("capex_intensity_pct:Q", title="Capex Intensity (%)"),
                y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
                tooltip=["company", "capex_intensity_pct", "ebitda_margin_pct"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Risk vs Supply Concentration</div>", unsafe_allow_html=True)
        st.write("Surfaces exposure to concentrated supply chains and combined risk score.")
        risk = synthetic_df.groupby("company", as_index=False)[
            ["risk_score","supply_concentration"]
        ].mean()
        st.altair_chart(
            alt.Chart(risk)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X("supply_concentration:Q", title="Supply Concentration (0–1)"),
                y=alt.Y("risk_score:Q", title="Risk Score (0–1)"),
                tooltip=["company", "supply_concentration", "risk_score"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Segment Attractiveness</div>", unsafe_allow_html=True)
        st.write("Ranks segments using growth, margin, and market share proxies.")
        seg = synthetic_df.groupby("segment", as_index=False)[
            ["revenue_growth_pct","ebitda_margin_pct","market_share_pct"]
        ].mean()
        seg["attractiveness"] = (
            0.45 * seg["revenue_growth_pct"] +
            0.45 * seg["ebitda_margin_pct"] +
            0.10 * seg["market_share_pct"]
        )
        seg = seg.sort_values("attractiveness", ascending=False)
        st.altair_chart(
            alt.Chart(seg)
            .mark_bar()
            .encode(
                x=alt.X("segment:N", sort="-y", title="Segment"),
                y=alt.Y("attractiveness:Q", title="Attractiveness Score"),
                tooltip=["segment", "attractiveness"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Top 5 Acquisition Targets (Synthetic)</div>", unsafe_allow_html=True)
        st.write("Ranks companies using a composite of growth, margin, scale, and risk-adjusted penalties.")
        company_metrics = synthetic_df.groupby("company", as_index=False)[
            ["revenue_growth_pct","ebitda_margin_pct","market_share_pct",
             "revenue_usd_m","risk_score","debt_to_equity"]
        ].mean()
        company_metrics["ma_score"] = (
            0.30 * company_metrics["revenue_growth_pct"] +
            0.30 * company_metrics["ebitda_margin_pct"] +
            0.20 * company_metrics["market_share_pct"] +
            0.20 * (company_metrics["revenue_usd_m"] / company_metrics["revenue_usd_m"].max()) * 100 -
            15 * company_metrics["risk_score"] -
            3 * company_metrics["debt_to_equity"]
        )
        top5 = company_metrics.sort_values("ma_score", ascending=False).head(5)
        st.dataframe(
            top5[["company","ma_score","revenue_growth_pct","ebitda_margin_pct",
                  "market_share_pct","revenue_usd_m","risk_score"]],
            use_container_width=True,
        )
        st.altair_chart(
            alt.Chart(top5)
            .mark_bar()
            .encode(
                x=alt.X("company:N", sort="-y", title="Company"),
                y=alt.Y("ma_score:Q", title="M&A Score"),
                tooltip=["company", "ma_score"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Profit Pool by Segment</div>", unsafe_allow_html=True)
        st.write("Estimates where industry profit concentrates by combining revenue and EBITDA margin proxies.")
        profit_pool = synthetic_df.groupby("segment", as_index=False)[
            ["revenue_usd_m","ebitda_margin_pct"]
        ].mean()
        profit_pool["profit_pool_usd_m"] = (
            profit_pool["revenue_usd_m"] * (profit_pool["ebitda_margin_pct"] / 100.0)
        )
        profit_pool = profit_pool.sort_values("profit_pool_usd_m", ascending=False)
        st.altair_chart(
            alt.Chart(profit_pool)
            .mark_bar()
            .encode(
                x=alt.X("segment:N", sort="-y", title="Segment"),
                y=alt.Y("profit_pool_usd_m:Q", title="Estimated Profit Pool (USD, millions)"),
                tooltip=["segment", "profit_pool_usd_m"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Margin vs Leverage</div>", unsafe_allow_html=True)
        st.write("Shows whether profitability is paired with sustainable leverage levels.")
        md = synthetic_df.groupby("company", as_index=False)[
            ["ebitda_margin_pct","debt_to_equity","market_share_pct"]
        ].mean()
        st.altair_chart(
            alt.Chart(md)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X("debt_to_equity:Q", title="Debt to Equity"),
                y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
                size=alt.Size("market_share_pct:Q", title="Market Share"),
                tooltip=["company", "debt_to_equity", "ebitda_margin_pct", "market_share_pct"],
            ),
            use_container_width=True,
        )

        st.markdown("<div class='blue-accent'>Top 5 Risks (Synthetic)</div>", unsafe_allow_html=True)
        st.write("Highlights the highest-risk companies based on leverage, supply concentration, and weak growth.")
        risks = synthetic_df.groupby("company", as_index=False)[
            ["risk_score","debt_to_equity","supply_concentration","revenue_growth_pct","ebitda_margin_pct"]
        ].mean()
        risks = risks.sort_values("risk_score", ascending=False).head(5)
        risks["risk_reason"] = risks.apply(
            lambda r: "High leverage" if r["debt_to_equity"] > 2.0 else
                      "Supply concentration" if r["supply_concentration"] > 0.7 else
                      "Low growth" if r["revenue_growth_pct"] < 0 else
                      "Low margin" if r["ebitda_margin_pct"] < 10 else
                      "Composite risk",
            axis=1,
        )
        st.dataframe(
            risks[["company","risk_score","debt_to_equity","supply_concentration","revenue_growth_pct","ebitda_margin_pct","risk_reason"]],
            use_container_width=True,
        )
    else:
        st.info("Synthetic dataset missing acquisition-style columns for M&A visuals.")

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

            st.caption("Clusters are based on numeric columns only. Non-numeric columns are ignored.")

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
