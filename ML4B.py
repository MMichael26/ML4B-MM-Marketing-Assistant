import os
import re
from collections import Counter
import streamlit as st
import requests
import pandas as pd
import openai

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI


# =========================
# Page config + styling
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="wide")

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
    .soft-box {
        background:#F8FAFC;
        border:1px solid #E2E8F0;
        color:#0F172A;
        padding:14px 16px;
        border-radius:10px;
    }
    .soft-muted {
        color:#475569;
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
st.sidebar.caption("Leave blank to use the default key (if configured).")
show_key = st.sidebar.checkbox("Show API key", value=False)
user_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="default" if show_key else "password"
)

with st.sidebar.expander("Advanced settings", expanded=False):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# =========================
# API key handling (default + allow override)
# =========================
DEFAULT_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
api_key = (user_key or "").strip() or DEFAULT_KEY

if not api_key:
    st.markdown(
        """
        <div class="soft-box">
            <strong>Almost there.</strong>
            Please enter your OpenAI API key in the sidebar to continue.
            <span class="soft-muted">It stays on your machine.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# =========================
# Helper functions (Wikipedia for report only)
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
# External data helpers (for visuals only)
# =========================
def wikidata_find_industry_qid(industry_label: str) -> str:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": industry_label,
        "language": "en",
        "format": "json",
        "limit": 1
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("search"):
            return data["search"][0]["id"]
    except Exception:
        return ""
    return ""


def wikidata_companies_by_industry(industry_qid: str, limit: int = 40):
    query = f"""
    SELECT ?company ?companyLabel ?countryLabel ?iso2 ?sitelinks ?inception WHERE {{
      ?company wdt:P31/wdt:P279* wd:Q4830453 .
      ?company wdt:P452 wd:{industry_qid} .
      OPTIONAL {{ ?company wdt:P159 ?hq . ?hq wdt:P17 ?hqCountry . }}
      OPTIONAL {{ ?company wdt:P17 ?country . }}
      BIND(COALESCE(?hqCountry, ?country) AS ?countryFinal)
      OPTIONAL {{ ?countryFinal wdt:P297 ?iso2 . }}
      OPTIONAL {{ ?company wikibase:sitelinks ?sitelinks . }}
      OPTIONAL {{ ?company wdt:P571 ?inception . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    rows = []
    try:
        r = requests.get(url, params={"query": query}, headers=headers, timeout=20)
        data = r.json()
        for b in data["results"]["bindings"]:
            rows.append({
                "company": b.get("companyLabel", {}).get("value", ""),
                "country": b.get("countryLabel", {}).get("value", "Unknown"),
                "iso2": b.get("iso2", {}).get("value", ""),
                "sitelinks": int(float(b.get("sitelinks", {}).get("value", "0"))),
                "inception": b.get("inception", {}).get("value", "")
            })
    except Exception:
        return []
    return rows


def worldbank_latest_gdp(iso2: str):
    url = f"https://api.worldbank.org/v2/country/{iso2}/indicator/NY.GDP.MKTP.CD"
    try:
        r = requests.get(url, params={"format": "json", "per_page": 60}, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        for row in data[1]:
            if row.get("value") is not None:
                return {"year": row.get("date"), "value": row.get("value")}
    except Exception:
        return None
    return None


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
        for u in urls:
            st.write(u)

    st.info("The report below is generated exclusively from the five Wikipedia pages listed above.")

    # =========================
    # Step 3 — Industry report
    # =========================
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
        "Keep the full report under 500 words.\n"
        "Use plain text headings without markdown hashes."
    )

    user_prompt = (
        f"Industry: {industry.strip()}\n\n"
        "Context: You are preparing this for a business analyst evaluating an acquisition target in this industry.\n"
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
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            report = cap_500_words(response.content)
        except openai.AuthenticationError:
            st.markdown(
                """
                <div class="soft-box">
                    <strong>We couldn’t verify that key.</strong>
                    Please double‑check it and try again.
                    <span class="soft-muted">If you just updated it, refresh the page.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()
        except Exception:
            st.markdown(
                """
                <div class="soft-box">
                    <strong>Small hiccup.</strong>
                    We couldn’t finish the report just now.
                    <span class="soft-muted">Please try again in a moment.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()

    report = re.sub(r"(?m)^#+\s*", "", report).strip()
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
    # Visuals — external real-world data only
    # =========================
    st.markdown("<h3 class='blue-accent'>Industry visuals (external real-world data)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Visuals below use free external sources (Wikidata + World Bank).</div>",
        unsafe_allow_html=True
    )

    industry_qid = wikidata_find_industry_qid(industry.strip())
    companies = wikidata_companies_by_industry(industry_qid, limit=60) if industry_qid else []

    if not companies:
        st.info("External company data could not be retrieved for this industry.")
    else:
        companies_df = pd.DataFrame(companies)

        # Chart 1: Top companies by Wikidata prominence (sitelinks)
        top_companies = (
            companies_df[['company', 'sitelinks']]
            .dropna()
            .sort_values('sitelinks', ascending=False)
            .head(10)
        )
        if not top_companies.empty:
            st.markdown("<div class='blue-accent'>Top Companies (Wikidata prominence proxy)</div>", unsafe_allow_html=True)
            st.bar_chart(top_companies.set_index('company'))
            st.caption("Prominence proxy based on number of Wikipedia sitelinks per company.")

        # Chart 2: HQ country distribution
        country_counts = (
            companies_df[['country']]
            .replace('', 'Unknown')
            .value_counts()
            .reset_index(name='count')
            .head(10)
        )
        if not country_counts.empty:
            st.markdown("<div class='blue-accent'>Company HQ by Country (Wikidata)</div>", unsafe_allow_html=True)
            st.bar_chart(country_counts.set_index('country'))

        # Chart 3: Company founding decade distribution
        years = []
        for v in companies_df['inception'].fillna(''):
            m = re.search(r'^(\\d{4})', str(v))
            if m:
                years.append(int(m.group(1)))
        if years:
            decades = [f'{(y // 10) * 10}s' for y in years]
            decade_counts = pd.Series(decades).value_counts().sort_index().head(12)
            st.markdown("<div class='blue-accent'>Company Founding Decades</div>", unsafe_allow_html=True)
            st.bar_chart(decade_counts)

        # Chart 4: GDP of HQ countries (World Bank)
        iso2_list = (
            companies_df[['country', 'iso2']]
            .dropna()
            .drop_duplicates()
            .head(8)
        )
        gdp_rows = []
        for _, row in iso2_list.iterrows():
            iso2 = str(row['iso2']).strip()
            if not iso2:
                continue
            gdp = worldbank_latest_gdp(iso2)
            if gdp:
                gdp_rows.append({
                    'country': row['country'],
                    'gdp': gdp['value'],
                    'year': gdp['year']
                })
        if gdp_rows:
            gdp_df = pd.DataFrame(gdp_rows).sort_values('gdp', ascending=False)
            st.markdown("<div class='blue-accent'>GDP of HQ Countries (World Bank)</div>", unsafe_allow_html=True)
            st.bar_chart(gdp_df.set_index('country')[['gdp']])
            latest_year = gdp_df['year'].astype(str).max()
            st.caption(f"GDP (current US$), latest available year per country (max year shown: {latest_year}).")
