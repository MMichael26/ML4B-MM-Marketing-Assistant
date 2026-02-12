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
# Synthetic schema helpers (keep your full schema library below)
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

# ... (keep your full expanded schema list here) ...

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

    user_prompt = (
        f"Industry: {industry.strip()}\n\n"
        "Required structure (HEADINGS MUST BE ON THEIR OWN LINE):\n"
        "### Executive Snapshot\n"
        "(2–3 sentences)\n\n"
        "### Scope and Definition\n\n"
        "### Value Chain / Key Segments\n\n"
        "### Demand Drivers and Primary Use-Cases\n\n"
        "### Challenges / Constraints / Notable Developments\n\n"
        "### What to Research Next\n"
        "(3–5 bullets)\n\n"
        "Rules:\n"
        "- Put each heading on its own line.\n"
        "- Leave a blank line after each heading.\n"
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

    # Force headings onto new lines even if model fails
    report = re.sub(r"(Executive Snapshot|Scope and Definition|Value Chain / Key Segments|Demand Drivers and Primary Use-Cases|Challenges / Constraints / Notable Developments|What to Research Next)", r"\n### \\1", report)
    report = re.sub(r"(?m)^###\s*(.+)$", r"<div class='section-title'>\1</div>", report)
    report = report.replace("**", "").strip()

    st.caption(f"Word count: {len(report.split())} / 500")
    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # (Keep your visuals + clustering sections below unchanged)
    
    # =========================
    # Synthetic Dataset & M&A Visuals
    # =========================
    st.markdown("<h3 class='blue-accent'>Synthetic Dataset & M&A‑Oriented Visuals</h3>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Synthetic data enriched with acquisition‑style metrics.</div>", unsafe_allow_html=True)

    synthetic_df = enrich_for_ma(generate_synthetic_df(industry.strip(), 240), industry.strip())

    st.markdown("<div class='section-title'>Market Share — Top Companies</div>", unsafe_allow_html=True)
    share_df = (
        synthetic_df.groupby("company")["market_share_pct"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    st.altair_chart(
        alt.Chart(share_df).mark_bar().encode(
            x=alt.X("market_share_pct:Q", title="Market Share (%)"),
            y=alt.Y("company:N", sort="-x", title="Company"),
            tooltip=["company", "market_share_pct"]
        ),
        use_container_width=True
    )

    st.markdown("<div class='section-title'>Growth vs EBITDA Margin</div>", unsafe_allow_html=True)
    st.altair_chart(
        alt.Chart(synthetic_df).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("revenue_growth_pct:Q", title="Revenue Growth (%)"),
            y=alt.Y("ebitda_margin_pct:Q", title="EBITDA Margin (%)"),
            color=alt.Color("segment:N", title="Segment"),
            tooltip=["company","segment","revenue_growth_pct","ebitda_margin_pct"]
        ),
        use_container_width=True
    )

    st.markdown("<div class='section-title'>Revenue Trend Over Time</div>", unsafe_allow_html=True)
    time_df = synthetic_df.groupby("month")["revenue_usd_m"].sum().reset_index()
    st.altair_chart(
        alt.Chart(time_df).mark_line(point=True).encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("revenue_usd_m:Q", title="Total Revenue (USD, millions)"),
            tooltip=["month","revenue_usd_m"]
        ),
        use_container_width=True
    )

    st.markdown("<div class='section-title'>Top 5 Acquisition Targets</div>", unsafe_allow_html=True)
    target_df = synthetic_df.copy()
    target_df["target_score"] = (
        target_df["revenue_growth_pct"]*0.4
        + target_df["ebitda_margin_pct"]*0.5
        + (1 - target_df["risk_score"])*10
    )
    st.dataframe(
        target_df.sort_values("target_score", ascending=False)
        .head(5)[["company","segment","region","revenue_growth_pct","ebitda_margin_pct","risk_score","target_score"]]
    )

    st.markdown("<div class='section-title'>Top 5 Risks</div>", unsafe_allow_html=True)
    st.dataframe(
        synthetic_df.sort_values("risk_score", ascending=False)
        .head(5)[["company","segment","region","risk_score","supply_concentration"]]
    )

    st.markdown("<h3 class='blue-accent'>Clustering (K-means)</h3>", unsafe_allow_html=True)

    cluster_df = synthetic_df.select_dtypes(include=["number"]).copy()
    with st.form("cluster_controls"):
        cluster_fields = st.multiselect(
            "Fields used to cluster",
            options=cluster_df.columns.tolist(),
            default=["revenue_growth_pct","ebitda_margin_pct","capex_intensity_pct","risk_score"]
        )
        cluster_x = st.selectbox(
            "X-axis",
            options=cluster_df.columns.tolist(),
            index=cluster_df.columns.get_loc("revenue_growth_pct")
        )
        cluster_y = st.selectbox(
            "Y-axis",
            options=cluster_df.columns.tolist(),
            index=cluster_df.columns.get_loc("ebitda_margin_pct")
        )
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

        st.altair_chart(
            alt.Chart(plot_df).mark_circle(size=70, opacity=0.8).encode(
                x=alt.X(f"{st.session_state.cluster_x_value}:Q", title=st.session_state.cluster_x_value),
                y=alt.Y(f"{st.session_state.cluster_y_value}:Q", title=st.session_state.cluster_y_value),
                color=alt.Color("cluster:N", title="Cluster"),
                tooltip=["company", st.session_state.cluster_x_value, st.session_state.cluster_y_value, "cluster"]
            ),
            use_container_width=True
        )
    else:
        st.warning("Select at least two numeric fields for clustering.")
