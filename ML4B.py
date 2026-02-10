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

    industry_key = industry.strip().lower()

    name_banks = {
        "food": ["FreshFoods", "HarvestGroup", "TasteCo", "KitchenWorks", "AgriFoods", "PantryBrands"],
        "grocery": ["MarketFresh", "DailyGrocer", "ValueFoods", "UrbanMart", "GreenBasket"],
        "retail": ["MetroRetail", "ValueMart", "ShopLine", "PrimeStores", "EverydayRetail"],
        "tech": ["TechNova", "InnovaSoft", "NextCloud", "QuantumLabs", "VectorSystems"],
        "software": ["SoftBridge", "CodeWave", "AppCore", "CloudMint", "StackLabs"],
        "health": ["CarePlus", "MediCore", "HealthFirst", "WellPoint", "NovaHealth"],
        "biotech": ["BioNova", "GeneWorks", "HelixLabs", "NeoPharma", "CellCore"],
        "pharma": ["Medigen", "PharmaCore", "LifeWell", "AcuPharm", "NovaRx"],
        "fintech": ["PayStream", "FinEdge", "NeoLedger", "FlowBank", "CreditSpring"],
        "bank": ["CivicBank", "UnionTrust", "MetroBank", "NorthStar Finance", "CapitalWay"],
        "insurance": ["ShieldSure", "PrimeCover", "SuretyOne", "HarborRisk", "PolicyGuard"],
        "energy": ["GridWorks", "PowerCore", "RenewCo", "FluxEnergy", "TerraPower"],
        "oil": ["PetroCore", "DeltaEnergy", "ApexOil", "PrimePetro", "AtlasFuel"],
        "renewable": ["SunGrid", "WindPeak", "GreenSpark", "EcoVolt", "Solaris"],
        "logistics": ["RouteLine", "FreightFlow", "CargoLink", "ShipPro", "TransitCore"],
        "telecom": ["WaveCom", "SignalNet", "MetroTel", "GlobalLink", "SpectrumOne"],
        "media": ["StreamWorks", "VistaMedia", "Northlight Studios", "EchoBroadcast", "BrightCast"],
        "hospitality": ["StayWell", "UrbanLodge", "VistaResorts", "HarborHotels", "PrimeStay"],
        "travel": ["SkyRoute", "VoyageGroup", "GlobalWays", "BlueCompass", "AeroTour"],
        "automotive": ["DriveWorks", "AutoCore", "VelocityMotors", "RoadLine", "TorqueAuto"],
        "aerospace": ["AeroDyne", "SkyWorks", "OrbitTech", "StratoSystems", "NovaFlight"],
        "construction": ["BuildCore", "StoneBridge", "IronPeak", "CivicConstruct", "UrbanBuild"],
        "real estate": ["PrimeEstate", "MetroLiving", "CivicHomes", "HarborRealty", "Northview Properties"],
        "agriculture": ["FieldGrow", "HarvestLine", "AgriCore", "GreenLeaf Farms", "TerraAgri"],
        "chemical": ["ChemCore", "SynthWorks", "ApexChem", "NovaMaterials", "PolyLab"],
        "materials": ["AlloyWorks", "CoreMaterials", "StonePeak", "TerraMat", "ForgeLine"],
        "semiconductor": ["SiliconWorks", "ChipCore", "NanoFab", "LogicFoundry", "IonSemis"],
        "fashion": ["StyleHub", "TrendLane", "UrbanWear", "NovaFashion", "LuxeLine"],
        "consumer": ["EverydayBrands", "PureHome", "BrightLife", "NovaHouse", "TrueEssentials"],
        "education": ["EduCore", "BrightPath", "LearnWorks", "NovaAcademy", "SkillPoint"],
        "security": ["SecureNet", "CyberShield", "AegisTech", "GuardLine", "IronGate"],
        "mining": ["OreCore", "DeepRock", "TerraMine", "AtlasMinerals", "IronField"],
        "defense": ["AegisDefense", "ShieldWorks", "TitanSystems", "IronGate Defense", "NovaArmor"],
        "gaming": ["PixelForge", "GameWave", "Arcadia Studios", "LevelUp Labs", "PlayGrid"],
        "shipping": ["OceanLink", "HarborLine", "BlueRoute", "SeaBridge", "MaritimeCore"],
        "utilities": ["CivicUtilities", "GridLine", "PowerFlow", "MetroEnergy", "CityLight"],
        "electronics": ["NovaElectro", "CircuitWorks", "PulseTech", "BrightWave", "ElectraCore"],
    }

    segment_banks = {
        "food": ["Production", "Processing", "Distribution", "Retail", "Food Service"],
        "grocery": ["Wholesale", "Retail", "Private Label", "Supply Chain"],
        "retail": ["Online", "Stores", "Private Label", "Logistics"],
        "tech": ["Platforms", "Infrastructure", "Applications", "Services"],
        "software": ["SaaS", "Enterprise", "SMB", "Developer Tools"],
        "health": ["Providers", "Services", "Devices", "Payers"],
        "biotech": ["R&D", "Clinical", "Manufacturing", "Commercial"],
        "pharma": ["Branded", "Generics", "Distribution", "Specialty"],
        "fintech": ["Payments", "Lending", "Wealth", "Infrastructure"],
        "bank": ["Retail Banking", "SME", "Corporate", "Wealth"],
        "insurance": ["Life", "Property & Casualty", "Health", "Reinsurance"],
        "energy": ["Generation", "Transmission", "Distribution", "Retail"],
        "renewable": ["Solar", "Wind", "Storage", "Grid Services"],
        "logistics": ["Freight", "Last‑Mile", "Warehousing", "Cold Chain"],
        "telecom": ["Wireless", "Broadband", "Enterprise", "Infrastructure"],
        "media": ["Streaming", "Broadcast", "Studios", "Advertising"],
        "hospitality": ["Hotels", "Resorts", "Restaurants", "Events"],
        "travel": ["Air", "Rail", "Tours", "Online Travel"],
        "automotive": ["OEM", "Aftermarket", "EV", "Mobility"],
        "aerospace": ["Commercial", "Defense", "Avionics", "Maintenance"],
        "construction": ["Residential", "Commercial", "Infrastructure", "Engineering"],
        "real estate": ["Residential", "Commercial", "Industrial", "Property Mgmt"],
        "agriculture": ["Inputs", "Farming", "Processing", "Distribution"],
        "chemical": ["Commodity", "Specialty", "Agrochem", "Industrial"],
        "materials": ["Metals", "Polymers", "Composites", "Cement"],
        "semiconductor": ["Foundry", "Memory", "Logic", "Analog"],
        "fashion": ["Design", "Manufacturing", "Retail", "E‑commerce"],
        "consumer": ["Home", "Personal Care", "Beverages", "Household"],
        "education": ["K‑12", "Higher Ed", "EdTech", "Training"],
        "security": ["Software", "Hardware", "Services", "Identity"],
        "mining": ["Exploration", "Extraction", "Processing", "Logistics"],
        "defense": ["Systems", "Weapons", "Cyber", "Maintenance"],
        "gaming": ["Mobile", "Console", "PC", "Live Services"],
        "shipping": ["Containers", "Bulk", "Ports", "Logistics"],
        "utilities": ["Generation", "Distribution", "Customer Service", "Grid"],
        "electronics": ["Consumer", "Industrial", "Components", "Devices"],
    }

    def pick_name_base(industry_key: str):
        for key, bank in name_banks.items():
            if key in industry_key:
                return bank
        return [f"{industry.title()} Co", f"{industry.title()} Group", f"{industry.title()} Holdings"]

    def pick_segment_base(industry_key: str):
        for key, bank in segment_banks.items():
            if key in industry_key:
                return bank
        return ["Core", "Premium", "Value", "Emerging"]

    if "company" not in df.columns:
        if "brand" in df.columns:
            df["company"] = df["brand"]
        elif "provider" in df.columns:
            df["company"] = df["provider"]
        elif "maker" in df.columns:
            df["company"] = df["maker"]
        elif "carrier" in df.columns:
            df["company"] = df["carrier"]
        elif "entity" in df.columns:
            df["company"] = df["entity"]
        else:
            base_names = pick_name_base(industry_key)
            df["company"] = [f"{rng.choice(base_names)} {i+1}" for i in range(len(df))]

    df["company"] = df["company"].astype(str)

    df["segment"] = df.get("segment", None)
    if df["segment"].isnull().all():
        segment_names = pick_segment_base(industry_key)
        df["segment"] = rng.choice(segment_names, size=len(df), replace=True)

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
# UI — Step 1
# =========================
st.markdown("<h3 class='blue-accent'>Step 1 — Choose an industry</h3>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Tip: be specific (e.g., “Fast fashion”, “Semiconductor industry”, “EV batteries”).</div>",
    unsafe_allow_html=True
)

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

    st.success("Industry received. Fetching Wikipedia sources...")

    docs = retrieve_wikipedia_docs(industry.strip(), k=5)
    urls = extract_urls(docs)

    if not urls:
        st.warning("I couldn't find reliable Wikipedia matches. Please be more specific or rephrase the industry you would like to research.")
        st.info("Examples: “Fast fashion”, “Semiconductor industry”, “EV battery market”.")
        st.stop()

    # Step 2
    st.markdown("<h3 class='blue-accent'>Step 2 — Top Wikipedia sources</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>These are the five most relevant pages used to generate the report.</div>", unsafe_allow_html=True)

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

    # Step 3
    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Business-analyst style briefing with traceable citations in the form [Source #].</div>", unsafe_allow_html=True)

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
        "Context: You are preparing this for a business analyst evaluating an acquisition target in this industry.\n"
        f"{focus_map.get(st.session_state.report_focus_value, '')}\n"
        f"{detail_map.get(st.session_state.detail_level_value, '')}\n"
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
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}]
        )
        report = cap_500_words(response.content)

    report = re.sub(r"(?m)^#+\s*", "", report)
    report = re.sub(r"(?m)^\s*\d+\)\s*(.+)$", r"<div class=\"section-title\">\1</div>", report).strip()
    report = report.replace("- **", "").replace("**", "")

    st.caption(f"Word count: {len(report.split())} / 500")
    st.markdown(f"<div class='report-box'>{report.replace('\\n','<br>')}</div>", unsafe_allow_html=True)

    # =========================
    # Synthetic Dataset & M&A-Oriented Visuals
    # =========================
    st.markdown("<h3 class='blue-accent'>Synthetic Dataset & M&A-Oriented Visuals</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>A synthetic dataset is generated and enriched with acquisition-style metrics for analyst screening.</div>", unsafe_allow_html=True)

    synthetic_df = generate_synthetic_df(industry.strip(), rows=240)
    synthetic_df = enrich_for_ma(synthetic_df, industry.strip())

    # ---- Market Share (Top Companies)
    st.markdown("<div class='section-title'>Market Share — Top Companies</div>", unsafe_allow_html=True)
    st.write("Ranks companies by estimated market share within the synthetic sample to highlight potential leaders.")
    share_df = (
        synthetic_df.groupby("company")["market_share_pct"]
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    st.altair_chart(
        alt.Chart(share_df).mark_bar().encode(
            x=alt.X("market_share_pct:Q", title="Market Share (%)"),
            y=alt.Y("company:N", sort="-x", title="Company"),
            tooltip=["company", "market_share_pct"]
        ),
        use_container_width=True
    )

    # ---- Growth vs EBITDA Margin
    st.markdown("<div class='section-title'>Growth vs EBITDA Margin</div>", unsafe_allow_html=True)
    st.write("Shows the trade-off between growth and profitability across synthetic entities.")
    st.altair_chart(
        alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("revenue_growth_pct:Q", title="Revenue Growth (%)"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "segment", "revenue_growth_pct", "ebitda_margin_pct"]
        ),
        use_container_width=True
    )

    # ---- Revenue Distribution
    st.markdown("<div class='section-title'>Revenue Distribution</div>", unsafe_allow_html=True)
    st.write("Shows how revenue is distributed across entities, highlighting size skew.")
    st.altair_chart(
        alt.Chart(synthetic_df).mark_bar().encode(
            x=alt.X("revenue_usd_m:Q", bin=alt.Bin(maxbins=20), title="Revenue (USD, millions)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count()"]
        ),
        use_container_width=True
    )

    # ---- Revenue Trend (Monthly)
    st.markdown("<div class='section-title'>Revenue Trend Over Time</div>", unsafe_allow_html=True)
    st.write("Tracks aggregate revenue trends over time, based on synthetic time signals.")
    time_df = synthetic_df.groupby("month")["revenue_usd_m"].sum().reset_index()
    st.altair_chart(
        alt.Chart(time_df).mark_line(point=True).encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("revenue_usd_m:Q", title="Total Revenue (USD, millions)"),
            tooltip=["month", "revenue_usd_m"]
        ),
        use_container_width=True
    )

    # ---- Capex vs Margin
    st.markdown("<div class='section-title'>Capex Intensity vs Margin</div>", unsafe_allow_html=True)
    st.write("Identifies which players combine strong margins with capital efficiency.")
    st.altair_chart(
        alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("capex_intensity_pct:Q", title="Capex Intensity (%)"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "capex_intensity_pct", "ebitda_margin_pct"]
        ),
        use_container_width=True
    )

    # ---- Risk vs Supply Concentration
    st.markdown("<div class='section-title'>Risk vs Supply Concentration</div>", unsafe_allow_html=True)
    st.write("Highlights exposure to supply-chain concentration risk against composite risk scores.")
    st.altair_chart(
        alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("supply_concentration:Q", title="Supply Concentration (0–1)"),
            y=alt.Y("risk_score:Q", title="Risk Score (0–1)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "supply_concentration", "risk_score"]
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
        alt.Chart(seg_df).mark_bar().encode(
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

    # ---- Profit Pool by Segment
    st.markdown("<div class='section-title'>Profit Pool by Segment</div>", unsafe_allow_html=True)
    st.write("Estimates segment profit pools using revenue × margin as a proxy.")
    profit_df = synthetic_df.groupby("segment").apply(
        lambda d: (d["revenue_usd_m"] * (d["ebitda_margin_pct"] / 100)).sum()
    ).reset_index(name="profit_pool")
    st.altair_chart(
        alt.Chart(profit_df).mark_bar().encode(
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
        alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("debt_to_equity:Q", title="Debt to Equity"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company", "debt_to_equity", "ebitda_margin_pct"]
        ),
        use_container_width=True
    )

    # ---- Top 5 Risks
    st.markdown("<div class='section-title'>Top 5 Risks</div>", unsafe_allow_html=True)
    st.write("Lists the highest-risk entities to flag for diligence.")
    top_risks = synthetic_df.sort_values("risk_score", ascending=False).head(5)
    st.dataframe(top_risks[["company", "segment", "region", "risk_score", "supply_concentration"]])

    # ---- Profit Strategy Summary
    st.markdown("<div class='section-title'>Profit Strategy Summary</div>", unsafe_allow_html=True)
    st.write("Synthetic signals suggest where value is concentrated and which segments to prioritize.")
    st.write(f"- Highest attractiveness segment: **{seg_df.sort_values('attractiveness', ascending=False).iloc[0]['segment']}**")
    st.write(f"- Largest profit pool: **{profit_df.sort_values('profit_pool', ascending=False).iloc[0]['segment']}**")
    st.write(f"- Most risky segment: **{seg_df.sort_values('avg_risk', ascending=False).iloc[0]['segment']}**")

    # =========================
    # Clustering (K-means)
    # =========================
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
            alt.Chart(plot_df).mark_circle(size=70, opacity=0.8).encode(
                x=alt.X(f"{st.session_state.cluster_x_value}:Q", title=st.session_state.cluster_x_value),
                y=alt.Y(f"{st.session_state.cluster_y_value}:Q", title=st.session_state.cluster_y_value),
                color=alt.Color("cluster:N", title="Cluster"),
                tooltip=["company", st.session_state.cluster_x_value, st.session_state.cluster_y_value, "cluster"]
            ),
            use_container_width=True
        )

        cluster_summary = plot_df.groupby("cluster")[st.session_state.cluster_fields_value].mean().reset_index()
        st.markdown("<div class='section-title'>Cluster Insights</div>", unsafe_allow_html=True)
        st.write("Average values per cluster to help compare strategic profiles.")
        st.dataframe(cluster_summary)
    else:
        st.warning("Select at least two numeric fields for clustering.")
