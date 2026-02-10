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
# Page config + styling
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="wide")

st.markdown(
    """
    <style>
    .blue-accent { color:#2563EB; font-weight:700; margin-top:0.8rem; }
    .subtle { color:#1F2937; font-size:0.95rem; }
    .report-box { background:#EFF6FF; padding:1.1rem 1.2rem; border-radius:10px; border-left:6px solid #2563EB; }
    .section-title { font-weight:800; font-size:1.1rem; border-bottom:1px solid #CBD5E1; padding-bottom:6px; margin:14px 0 8px; }
    code { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Market Research Assistant")
st.caption("Generate a concise, Wikipedia‑grounded industry briefing in three steps.")

# =========================
# Sidebar: LLM + API Key (Q0)
# =========================
st.sidebar.header("Model & API Key")
llm_options = ["gpt-4o-mini"]
selected_llm = st.sidebar.selectbox("LLM", llm_options, index=0)
user_key = st.sidebar.text_input("OpenAI API Key", type="password")

# =========================
# Sidebar: Report preferences
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
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("**Clustering (synthetic data)**")
    k_clusters = st.slider("K-means clusters", 2, 6, 3, 1)

    apply_controls = st.form_submit_button("Apply settings")

if "report_focus_value" not in st.session_state:
    st.session_state.report_focus_value = report_focus
if "detail_level_value" not in st.session_state:
    st.session_state.detail_level_value = detail_level
if "k_clusters_value" not in st.session_state:
    st.session_state.k_clusters_value = k_clusters
if "temperature_value" not in st.session_state:
    st.session_state.temperature_value = temperature

if apply_controls:
    st.session_state.report_focus_value = report_focus
    st.session_state.detail_level_value = detail_level
    st.session_state.k_clusters_value = k_clusters
    st.session_state.temperature_value = temperature

# =========================
# Helper functions
# =========================
def industry_is_valid(industry: str) -> bool:
    return bool(industry and industry.strip())

def retrieve_wikipedia_docs(industry: str, k: int = 5):
    retriever = WikipediaRetriever(top_k_results=k, lang="en")
    try:
        return retriever.get_relevant_documents(industry)[:k]
    except AttributeError:
        return retriever.invoke(industry)[:k]

def extract_urls(docs):
    urls, seen = [], set()
    for d in docs:
        src = (d.metadata or {}).get("source", "")
        if src and src not in seen:
            urls.append(src)
            seen.add(src)
    return urls[:5]

def build_sources_text(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        url = (d.metadata or {}).get("source", "")
        text = re.sub(r"\s+", " ", (d.page_content or "").strip())[:2600]
        parts.append(f"[Source {i}]\nTITLE: {title}\nURL: {url}\nCONTENT EXCERPT: {text}\n")
    return "\n\n".join(parts)

def cap_500_words(text: str) -> str:
    words = (text or "").split()
    return " ".join(words[:500]).rstrip() + ("…" if len(words) > 500 else "")

# =========================
# Synthetic schema helpers
# =========================
def rand_date(start_year=2020, end_year=2025):
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    delta_days = (end_date - start_date).days
    return (start_date + pd.Timedelta(days=int(np.random.randint(0, delta_days)))).date().isoformat()

def make_schema(entity_col, entities, segment_col=None, segments=None, metrics=None, date_col="date"):
    metrics = metrics or []
    columns = ["id", entity_col]
    if segment_col:
        columns.append(segment_col)
    for m in metrics:
        columns.append(m[0])
    columns.append(date_col)

    def row(i):
        row_vals = [i, np.random.choice(entities)]
        if segment_col:
            row_vals.append(np.random.choice(segments))
        for name, low, high, dec in metrics:
            val = float(np.random.uniform(low, high))
            row_vals.append(round(val, dec))
        row_vals.append(rand_date(2021, 2025))
        return row_vals

    return columns, row

# =========================
# Expanded schema library
# =========================
def schema_fast_fashion():
    return make_schema("brand",
        ["Zara","H&M","Shein","Forever 21","Uniqlo","Primark","Boohoo","Mango","ASOS"],
        "segment", ["Design","Manufacturing","Retail","E‑commerce"],
        [("price_usd", 4.99, 89.99, 2), ("co2_kg", 1.5, 45.0, 2), ("water_l", 50, 2500, 1)]
    )

def schema_healthcare():
    return make_schema("provider",
        ["St. Mary Hospital","City Clinic","MedPrime","CarePlus","BlueLeaf Health"],
        "department", ["Cardiology","Oncology","Pediatrics","Orthopedics","Neurology"],
        [("length_of_stay_days", 0, 14, 0), ("total_cost_usd", 120, 25000, 2)]
    )

def schema_ecommerce():
    return make_schema("merchant",
        ["ShopLine","RetailHub","MarketBridge","NovaStore","PrimeCart"],
        "category", ["Electronics","Home","Beauty","Sports","Toys","Fashion","Books"],
        [("unit_price_usd", 5, 1500, 2), ("units", 1, 8, 0), ("discount_pct", 0, 40, 1)]
    )

def schema_semiconductors():
    return make_schema("company",
        ["TSMC","Samsung","Intel","SK Hynix","Micron","GlobalFoundries","UMC"],
        "segment", ["Foundry","Memory","Logic","Analog","Power","RF"],
        [("wafer_starts_k", 10, 180, 1), ("yield_pct", 70, 99, 1), ("capex_bil", 0.5, 15, 2)]
    )

def schema_ev_batteries():
    return make_schema("maker",
        ["CATL","LG Energy","Panasonic","BYD","SK On","Samsung SDI"],
        "segment", ["Passenger EV","Commercial EV","Energy Storage"],
        [("cost_per_kwh", 70, 180, 2), ("energy_density_whkg", 120, 300, 1), ("capacity_gwh", 1, 50, 2)]
    )

def schema_retail():
    return make_schema("banner",
        ["Walmart","Target","Carrefour","Tesco","Costco","Aldi","Lidl"],
        "format", ["Hypermarket","Supermarket","Warehouse","Discount","Online"],
        [("store_sales_usd_m", 5, 400, 2), ("same_store_growth_pct", -5, 15, 2), ("basket_size_usd", 15, 120, 2)]
    )

def schema_logistics():
    return make_schema("carrier",
        ["DHL","FedEx","UPS","Maersk","MSC","DP World","XPO"],
        "mode", ["Air","Ocean","Road","Rail"],
        [("shipment_volume_k", 20, 700, 1), ("on_time_pct", 75, 98, 1), ("cost_per_shipment_usd", 50, 900, 2)]
    )

def schema_financial_services():
    return make_schema("firm",
        ["CivicBank","UnionTrust","MetroBank","CapitalWay","NorthStar Finance"],
        "segment", ["Retail Banking","SME","Corporate","Wealth"],
        [("aum_usd_b", 5, 500, 2), ("net_interest_margin_pct", 1.5, 4.5, 2), ("cost_to_income_pct", 35, 75, 2)]
    )

def schema_energy():
    return make_schema("company",
        ["GridWorks","PowerCore","RenewCo","FluxEnergy","TerraPower"],
        "segment", ["Generation","Transmission","Distribution","Retail"],
        [("capacity_gw", 1, 50, 2), ("utilization_pct", 60, 95, 1), ("opex_usd_m", 10, 250, 2)]
    )

def schema_telecom():
    return make_schema("company",
        ["WaveCom","SignalNet","MetroTel","GlobalLink","SpectrumOne"],
        "segment", ["Wireless","Broadband","Enterprise","Infrastructure"],
        [("arpu_usd", 10, 80, 2), ("churn_pct", 0.5, 3.5, 2), ("capex_pct", 10, 30, 2)]
    )

def schema_pharma():
    return make_schema("company",
        ["Medigen","PharmaCore","LifeWell","AcuPharm","NovaRx"],
        "segment", ["Branded","Generics","Specialty","Distribution"],
        [("r_and_d_pct", 5, 25, 2), ("pipeline_assets", 2, 40, 0), ("gross_margin_pct", 30, 85, 2)]
    )

def schema_software():
    return make_schema("company",
        ["SoftBridge","CodeWave","AppCore","CloudMint","StackLabs"],
        "segment", ["SaaS","Enterprise","SMB","Developer Tools"],
        [("arr_usd_m", 10, 500, 2), ("churn_pct", 1, 10, 2), ("gross_margin_pct", 50, 90, 2)]
    )

def schema_media():
    return make_schema("company",
        ["StreamWorks","VistaMedia","Northlight Studios","EchoBroadcast","BrightCast"],
        "segment", ["Streaming","Broadcast","Studios","Advertising"],
        [("subs_m", 1, 60, 2), ("arpu_usd", 3, 20, 2), ("content_spend_usd_m", 10, 300, 2)]
    )

def schema_automotive():
    return make_schema("company",
        ["DriveWorks","AutoCore","VelocityMotors","RoadLine","TorqueAuto"],
        "segment", ["OEM","Aftermarket","EV","Mobility"],
        [("units_k", 10, 800, 1), ("gross_margin_pct", 5, 30, 2), ("r_and_d_pct", 2, 12, 2)]
    )

def schema_real_estate():
    return make_schema("company",
        ["PrimeEstate","MetroLiving","CivicHomes","HarborRealty","Northview Properties"],
        "segment", ["Residential","Commercial","Industrial","Property Mgmt"],
        [("occupancy_pct", 70, 98, 1), ("rent_usd_sqft", 10, 120, 2), ("cap_rate_pct", 3, 10, 2)]
    )

def schema_agriculture():
    return make_schema("company",
        ["FieldGrow","HarvestLine","AgriCore","GreenLeaf Farms","TerraAgri"],
        "segment", ["Inputs","Farming","Processing","Distribution"],
        [("yield_ton_ha", 1, 12, 2), ("cost_usd_ha", 100, 1500, 2), ("price_usd_ton", 80, 600, 2)]
    )

def schema_construction():
    return make_schema("company",
        ["BuildCore","StoneBridge","IronPeak","CivicConstruct","UrbanBuild"],
        "segment", ["Residential","Commercial","Infrastructure","Engineering"],
        [("backlog_usd_m", 20, 800, 2), ("margin_pct", 3, 18, 2), ("project_duration_m", 6, 48, 0)]
    )

def schema_mining():
    return make_schema("company",
        ["OreCore","DeepRock","TerraMine","AtlasMinerals","IronField"],
        "segment", ["Exploration","Extraction","Processing","Logistics"],
        [("ore_grade_pct", 0.3, 6, 2), ("cash_cost_usd_ton", 20, 200, 2), ("capex_usd_m", 10, 900, 2)]
    )

def schema_education():
    return make_schema("company",
        ["EduCore","BrightPath","LearnWorks","NovaAcademy","SkillPoint"],
        "segment", ["K‑12","Higher Ed","EdTech","Training"],
        [("enrollment_k", 1, 500, 1), ("completion_pct", 50, 95, 1), ("tuition_usd", 200, 8000, 2)]
    )

def schema_defense():
    return make_schema("company",
        ["AegisDefense","ShieldWorks","TitanSystems","IronGate Defense","NovaArmor"],
        "segment", ["Systems","Weapons","Cyber","Maintenance"],
        [("contract_usd_m", 50, 2000, 2), ("backlog_usd_m", 200, 5000, 2), ("margin_pct", 5, 25, 2)]
    )

def schema_utilities():
    return make_schema("company",
        ["CivicUtilities","GridLine","PowerFlow","MetroEnergy","CityLight"],
        "segment", ["Generation","Distribution","Customer Service","Grid"],
        [("customers_m", 0.1, 20, 2), ("outage_minutes", 20, 300, 1), ("capex_usd_m", 20, 1200, 2)]
    )

def schema_gaming():
    return make_schema("company",
        ["PixelForge","GameWave","Arcadia Studios","LevelUp Labs","PlayGrid"],
        "segment", ["Mobile","Console","PC","Live Services"],
        [("dau_m", 0.2, 50, 2), ("arppu_usd", 1, 30, 2), ("retention_pct", 15, 60, 1)]
    )

def schema_travel():
    return make_schema("company",
        ["SkyRoute","VoyageGroup","GlobalWays","BlueCompass","AeroTour"],
        "segment", ["Air","Rail","Tours","Online Travel"],
        [("bookings_m", 0.1, 15, 2), ("take_rate_pct", 5, 25, 2), ("cancellations_pct", 2, 20, 2)]
    )

def schema_hospitality():
    return make_schema("company",
        ["StayWell","UrbanLodge","VistaResorts","HarborHotels","PrimeStay"],
        "segment", ["Hotels","Resorts","Restaurants","Events"],
        [("occupancy_pct", 45, 95, 1), ("adr_usd", 60, 350, 2), ("revpar_usd", 30, 250, 2)]
    )

def schema_chemicals():
    return make_schema("company",
        ["ChemCore","SynthWorks","ApexChem","NovaMaterials","PolyLab"],
        "segment", ["Commodity","Specialty","Agrochem","Industrial"],
        [("volume_ktons", 5, 500, 1), ("margin_pct", 5, 30, 2), ("energy_cost_pct", 5, 35, 2)]
    )

def schema_materials():
    return make_schema("company",
        ["AlloyWorks","CoreMaterials","StonePeak","TerraMat","ForgeLine"],
        "segment", ["Metals","Polymers","Composites","Cement"],
        [("capacity_ktons", 10, 800, 1), ("utilization_pct", 50, 95, 1), ("price_usd_ton", 150, 1200, 2)]
    )

def schema_shipping():
    return make_schema("company",
        ["OceanLink","HarborLine","BlueRoute","SeaBridge","MaritimeCore"],
        "segment", ["Containers","Bulk","Ports","Logistics"],
        [("teu_k", 20, 1200, 1), ("freight_rate_usd", 400, 3500, 2), ("on_time_pct", 60, 95, 1)]
    )

def schema_consumer_goods():
    return make_schema("company",
        ["EverydayBrands","PureHome","BrightLife","NovaHouse","TrueEssentials"],
        "segment", ["Home","Personal Care","Beverages","Household"],
        [("sales_usd_m", 20, 1200, 2), ("gross_margin_pct", 20, 60, 2), ("distribution_pts", 100, 50000, 0)]
    )

def schema_aerospace():
    return make_schema("company",
        ["AeroDyne","SkyWorks","OrbitTech","StratoSystems","NovaFlight"],
        "segment", ["Commercial","Defense","Avionics","Maintenance"],
        [("orders_usd_m", 50, 3000, 2), ("backlog_usd_m", 200, 8000, 2), ("margin_pct", 5, 25, 2)]
    )

def schema_generic(industry_name):
    return make_schema("entity",
        ["Alpha","Beta","Gamma","Delta","Omega"],
        "segment", ["Core","Premium","Value","Emerging"],
        [("metric_a", 0, 1000, 2), ("metric_b", 0, 100, 2), ("metric_c", 0, 10, 2)]
    )

SCHEMAS = {
    "fast fashion": schema_fast_fashion,
    "healthcare": schema_healthcare,
    "ecommerce": schema_ecommerce,
    "semiconductors": schema_semiconductors,
    "ev batteries": schema_ev_batteries,
    "retail": schema_retail,
    "logistics": schema_logistics,
    "financial services": schema_financial_services,
    "energy": schema_energy,
    "telecom": schema_telecom,
    "pharma": schema_pharma,
    "software": schema_software,
    "media": schema_media,
    "automotive": schema_automotive,
    "real estate": schema_real_estate,
    "agriculture": schema_agriculture,
    "construction": schema_construction,
    "mining": schema_mining,
    "education": schema_education,
    "defense": schema_defense,
    "utilities": schema_utilities,
    "gaming": schema_gaming,
    "travel": schema_travel,
    "hospitality": schema_hospitality,
    "chemicals": schema_chemicals,
    "materials": schema_materials,
    "shipping": schema_shipping,
    "consumer goods": schema_consumer_goods,
    "aerospace": schema_aerospace,
}

SCHEMA_KEYWORDS = {
    "fast fashion": ["fashion","apparel","textile"],
    "healthcare": ["health","medical","hospital","pharma"],
    "ecommerce": ["ecommerce","e-commerce","online retail","marketplace"],
    "semiconductors": ["semiconductor","chip","foundry","fab"],
    "ev batteries": ["battery","ev","electric vehicle","lithium"],
    "retail": ["retail","supermarket","grocery","store"],
    "logistics": ["logistics","shipping","freight","supply chain"],
    "financial services": ["bank","finance","fintech","insurance","financial"],
    "energy": ["energy","power","utility","renewable","oil","gas"],
    "telecom": ["telecom","wireless","broadband","carrier"],
    "pharma": ["pharma","drug","biotech"],
    "software": ["software","saas","cloud","app"],
    "media": ["media","streaming","broadcast","studio"],
    "automotive": ["auto","vehicle","car","ev"],
    "real estate": ["real estate","property","housing"],
    "agriculture": ["agriculture","farming","agri"],
    "construction": ["construction","infrastructure","engineering"],
    "mining": ["mining","minerals","ore"],
    "education": ["education","school","university","edtech"],
    "defense": ["defense","military","aerospace"],
    "utilities": ["utilities","grid","power","electric"],
    "gaming": ["gaming","games","esports"],
    "travel": ["travel","tourism","airline"],
    "hospitality": ["hotel","hospitality","resort"],
    "chemicals": ["chemical","chemicals"],
    "materials": ["materials","metals","composites"],
    "shipping": ["shipping","maritime","ports"],
    "consumer goods": ["consumer","fmcg","household"],
    "aerospace": ["aerospace","aviation","space"],
}

def pick_schema(industry: str):
    key = industry.strip().lower()
    if key in SCHEMAS:
        return SCHEMAS[key]
    for name, kws in SCHEMA_KEYWORDS.items():
        if any(k in key for k in kws):
            return SCHEMAS[name]
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

    def make_generic_company_names(industry: str, n: int):
        base = industry.title()
        prefixes = ["Global","Prime","Blue","North","Apex","Summit","Civic","Urban","Atlas","Core"]
        suffixes = ["Holdings","Group","Capital","Partners","Industries","Solutions","Systems","Ventures","Corp","Labs"]
        return [f"{rng.choice(prefixes)} {base} {rng.choice(suffixes)}" for _ in range(n)]

    if "company" not in df.columns:
        for col in ["brand","provider","maker","carrier","entity","firm","banner","merchant"]:
            if col in df.columns:
                df["company"] = df[col]
                break
        else:
            df["company"] = make_generic_company_names(industry, len(df))

    df["segment"] = df.get("segment", None)
    if df["segment"].isnull().all():
        df["segment"] = rng.choice(["Core","Premium","Value","Emerging"], size=len(df), replace=True)

    df["region"] = df.get("region", None)
    if df["region"].isnull().all():
        df["region"] = rng.choice(["NA","EU","APAC","LATAM","MEA"], size=len(df), replace=True)

    date_col = "date" if "date" in df.columns else None
    if date_col is None:
        df["event_date"] = pd.to_datetime(rng.choice(pd.date_range("2021-01-01", "2025-12-31"), size=len(df)))
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
# API Key handling
# =========================
api_key = (user_key or "").strip()
if not api_key:
    st.markdown(
        """
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;color:#0F172A;padding:14px 16px;border-radius:10px;">
            <strong>Almost there.</strong> Please enter your OpenAI API key in the sidebar to continue.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
llm = ChatOpenAI(model=selected_llm, temperature=st.session_state.temperature_value, api_key=user_key)

# =========================
# Step 1
# =========================
st.markdown("<h3 class='blue-accent'>Step 1 — Choose an industry</h3>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Tip: be specific (e.g., “Fast fashion”, “Semiconductor industry”, “EV batteries”).</div>", unsafe_allow_html=True)

with st.form("industry_form"):
    industry = st.text_input("Industry", placeholder="Try: Fast fashion")
    submitted = st.form_submit_button("Generate report")

if submitted:
    if not industry_is_valid(industry):
        st.warning("Please enter an industry to continue.")
        st.stop()

    if len(industry.strip()) < 3:
        st.info("Please provide a more specific industry name.")
        st.caption("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    st.session_state.industry_value = industry.strip()
    docs = retrieve_wikipedia_docs(st.session_state.industry_value, k=5)
    urls = extract_urls(docs)

    if not urls:
        st.warning("I couldn't find reliable Wikipedia matches. Please be more specific or rephrase the industry.")
        st.info("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    st.session_state.docs_value = docs

# =========================
# Step 2 + Step 3 (persisted)
# =========================
if "industry_value" in st.session_state and "docs_value" in st.session_state:
    industry = st.session_state.industry_value
    docs = st.session_state.docs_value

    st.markdown("<h3 class='blue-accent'>Step 2 — Top Wikipedia sources</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>These are the five most relevant pages used to generate the report.</div>", unsafe_allow_html=True)
    with st.expander("Show sources", expanded=True):
        shown, rank = set(), 0
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

    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Business‑analyst style briefing with citations [Source #].</div>", unsafe_allow_html=True)

    sources_text = build_sources_text(docs)

    system_prompt = (
        "You are a market research assistant for a business analyst at a large corporation.\n"
        "Write a concise industry briefing STRICTLY based on the provided Wikipedia sources.\n"
        "Every paragraph must include at least one citation [Source #].\n"
        "If the sources do not support a claim, write: 'Not specified in the sources.'\n"
        "Keep the report under 500 words."
    )

    focus_map = {
        "Acquisition screening": "Focus on M&A relevance, strategic fit, and competitive landscape.",
        "Market overview": "Focus on market definition, scope, and broad industry structure.",
        "Competitive positioning": "Focus on segments, key players, and competitive dynamics.",
        "Risk & compliance": "Focus on regulatory, operational, and reputational risks."
    }
    detail_map = {
        "Concise": "Use brief, tight language.",
        "Balanced": "Use balanced depth with clear headings.",
        "Deep": "Add more detail within the 500-word limit."
    }

    user_prompt = (
        f"Industry: {industry.strip()}\n\n"
        f"{focus_map.get(st.session_state.report_focus_value, '')}\n"
        f"{detail_map.get(st.session_state.detail_level_value, '')}\n"
        "Required structure:\n"
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
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}]
        )
        report = cap_500_words(response.content)

    report = re.sub(r"(?m)^#+\s*", "", report)
    report = re.sub(r"(?m)^\s*\d+\)\s*(.+)$", r"<div class='section-title'>\1</div>", report).strip()
    report = report.replace("- **", "").replace("**", "")

    st.caption(f"Word count: {len(report.split())} / 500")
    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # Synthetic Dataset & M&A Visuals
    # =========================
    st.markdown("<h3 class='blue-accent'>Synthetic Dataset & M&A‑Oriented Visuals</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Synthetic data enriched with acquisition‑style metrics.</div>", unsafe_allow_html=True)

    synthetic_df = enrich_for_ma(generate_synthetic_df(industry.strip(), 240), industry.strip())

    st.markdown("<div class='section-title'>Market Share — Top Companies</div>", unsafe_allow_html=True)
    share_df = synthetic_df.groupby("company")["market_share_pct"].mean().sort_values(ascending=False).head(10).reset_index()
    st.altair_chart(alt.Chart(share_df).mark_bar().encode(
        x=alt.X("market_share_pct:Q", title="Market Share (%)"),
        y=alt.Y("company:N", sort="-x", title="Company"),
        tooltip=["company", "market_share_pct"]), use_container_width=True)

    st.markdown("<div class='section-title'>Growth vs EBITDA Margin</div>", unsafe_allow_html=True)
    st.altair_chart(alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
        x=alt.X("revenue_growth_pct:Q", title="Revenue Growth (%)"),
        y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
        color=alt.Color("segment:N", title="Segment"),
        tooltip=["company","segment","revenue_growth_pct","ebitda_margin_pct"]), use_container_width=True)

    st.markdown("<div class='section-title'>Revenue Trend Over Time</div>", unsafe_allow_html=True)
    time_df = synthetic_df.groupby("month")["revenue_usd_m"].sum().reset_index()
    st.altair_chart(alt.Chart(time_df).mark_line(point=True).encode(
        x=alt.X("month:O", title="Month"),
        y=alt.Y("revenue_usd_m:Q", title="Total Revenue (USD, millions)"),
        tooltip=["month","revenue_usd_m"]), use_container_width=True)

    st.markdown("<div class='section-title'>Top 5 Acquisition Targets</div>", unsafe_allow_html=True)
    target_df = synthetic_df.copy()
    target_df["target_score"] = target_df["revenue_growth_pct"]*0.4 + target_df["ebitda_margin_pct"]*0.5 + (1-target_df["risk_score"])*10
    st.dataframe(target_df.sort_values("target_score", ascending=False).head(5)[["company","segment","region","revenue_growth_pct","ebitda_margin_pct","risk_score","target_score"]])

    st.markdown("<div class='section-title'>Top 5 Risks</div>", unsafe_allow_html=True)
    st.dataframe(synthetic_df.sort_values("risk_score", ascending=False).head(5)[["company","segment","region","risk_score","supply_concentration"]])

    st.markdown("<h3 class='blue-accent'>Clustering (K-means)</h3>", unsafe_allow_html=True)

    cluster_df = synthetic_df.select_dtypes(include=["number"]).copy()
    with st.form("cluster_controls"):
        cluster_fields = st.multiselect(
            "Fields used to cluster",
            options=cluster_df.columns.tolist(),
            default=["revenue_growth_pct","ebitda_margin_pct","capex_intensity_pct","risk_score"]
        )
        cluster_x = st.selectbox("X-axis", options=cluster_df.columns.tolist(), index=cluster_df.columns.get_loc("revenue_growth_pct"))
        cluster_y = st.selectbox("Y-axis", options=cluster_df.columns.tolist(), index=cluster_df.columns.get_loc("ebitda_margin_pct"))
        apply_cluster = st.form_submit_button("Apply clustering")

    if "cluster_fields_value" not in st.session_state:
        st.session_state.cluster_fields_value = cluster_fields
        st.session_state.cluster_x_value = cluster_x
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

        st.altair_chart(alt.Chart(plot_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X(f"{st.session_state.cluster_x_value}:Q", title=st.session_state.cluster_x_value),
            y=alt.Y(f"{st.session_state.cluster_y_value}:Q", title=st.session_state.cluster_y_value),
            color=alt.Color("cluster:N", title="Cluster"),
            tooltip=["company", st.session_state.cluster_x_value, st.session_state.cluster_y_value, "cluster"]),
            use_container_width=True
        )
    else:
        st.warning("Select at least two numeric fields for clustering.")
