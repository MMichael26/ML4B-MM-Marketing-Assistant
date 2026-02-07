import os
import re
import io
import pandas as pd
import streamlit as st
import requests
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
# Local Development (VS Code) instructions
# =========================
with st.expander("Local development setup (optional)", expanded=False):
    st.markdown("<h3 class='blue-accent'>Local Development (VS Code)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'><b>Where the key goes (locally)</b><br>You include the key only in your local environment, not in code.</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'><b>Option A (recommended): environment variable</b><br><b>Mac/Linux</b></div>", unsafe_allow_html=True)
    st.code('export OPENAI_API_KEY="sk-..."', language="bash")

# =========================
# Sidebar: API Key input
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

st.sidebar.header("External data (optional keys)")
fred_api_key = st.sidebar.text_input("FRED API Key (optional)", type="password")
comtrade_token = st.sidebar.text_input("UN Comtrade Token (optional)", type="password")

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


def prepare_for_kmeans(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return None, None
    scaled = (numeric_df - numeric_df.mean()) / (numeric_df.std(ddof=0) + 1e-9)
    return numeric_df, scaled


# =========================
# External data helpers (free sources)
# =========================
def worldbank_latest_indicator(iso2: str, indicator: str):
    url = f"https://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}"
    try:
        r = requests.get(url, params={"format": "json", "per_page": 60}, timeout=12)
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        for row in data[1]:
            if row.get("value") is not None:
                return {"year": row.get("date"), "value": row.get("value")}
    except Exception:
        return None
    return None


def oecd_latest_series(dataset: str, series_key: str):
    url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset}/{series_key}.json"
    try:
        r = requests.get(url, timeout=12)
        data = r.json()
        obs = data["dataSets"][0]["observations"]
        if not obs:
            return None
        latest_key = max(obs.keys(), key=lambda k: int(k.split(":")[-1]))
        latest_val = obs[latest_key][0]
        return latest_val
    except Exception:
        return None


def fred_latest(series_id: str, api_key: str):
    if not api_key:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    try:
        r = requests.get(
            url,
            params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
            timeout=12,
        )
        data = r.json()
        obs = [o for o in data.get("observations", []) if o.get("value") not in (".", None)]
        if not obs:
            return None
        latest = obs[-1]
        return {"date": latest["date"], "value": float(latest["value"])}
    except Exception:
        return None


def comtrade_latest(token: str, reporter: str = "840", commodity: str = "TOTAL"):
    if not token:
        return None
    url = "https://comtradeapi.worldbank.org/v1/get/HS"
    params = {
        "reporterCode": reporter,
        "year": "2022",
        "cmdCode": commodity,
        "flowCode": "M",
        "token": token,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "data" in data and data["data"]:
            return data["data"][0]
    except Exception:
        return None
    return None


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
    # Step 3 — Report (Wikipedia-only)
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Industry Report based on wikipedia sources (Under 500 words).</div>",
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


    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # Visuals — real external sources only
    # =========================
    st.markdown("<h3 class='blue-accent'>Industry visuals (external real-world data)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Sources: World Bank, OECD, UN Comtrade, FRED.</div>", unsafe_allow_html=True)

    wb_gdp = worldbank_latest_indicator("USA", "NY.GDP.MKTP.CD")
    wb_ind = worldbank_latest_indicator("USA", "NV.IND.MANF.ZS")
    if wb_gdp and wb_ind:
        st.markdown("<div class='blue-accent'>US Macro Context (World Bank)</div>", unsafe_allow_html=True)
        st.bar_chart(
            {"Metric": ["GDP (US$)", "Manufacturing % of GDP"], "Value": [wb_gdp["value"], wb_ind["value"]]},
            x="Metric",
            y="Value",
        )

    oecd_val = oecd_latest_series("SNA_TABLE2", "USA.A")
    if oecd_val is not None:
        st.markdown("<div class='blue-accent'>OECD Indicator (Sample)</div>", unsafe_allow_html=True)
        st.write(f"Latest OECD value (sample series): {oecd_val}")

    comtrade = comtrade_latest(comtrade_token)
    if comtrade:
        st.markdown("<div class='blue-accent'>UN Comtrade Sample (Imports)</div>", unsafe_allow_html=True)
        st.write(comtrade)

    fred = fred_latest("INDPRO", fred_api_key)
    if fred:
        st.markdown("<div class='blue-accent'>FRED Industrial Production Index</div>", unsafe_allow_html=True)
        st.line_chart({"Date": [fred["date"]], "Index": [fred["value"]]}, x="Date", y="Index")

    # =========================
    # Clustering — only if CSV uploaded
    # =========================
    st.markdown("<h3 class='blue-accent'>Clustering (K-means)</h3>", unsafe_allow_html=True)
    if uploaded_csv is None:
        st.info("Upload a CSV to enable clustering.")
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
