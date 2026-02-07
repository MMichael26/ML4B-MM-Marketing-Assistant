import os
import re
from collections import Counter
import streamlit as st
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
    if not industry or not industry.strip():
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
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        try:
            docs = retriever.get_relevant_documents(industry.strip())
        except AttributeError:
            docs = retriever.invoke(industry.strip())
        docs = docs[:5]

        urls = []
        seen = set()
        for d in docs:
            src = (d.metadata or {}).get("source", "")
            if src and src not in seen:
                urls.append(src)
                seen.add(src)

    if not urls:
        st.error("No Wikipedia pages found. Try a more specific industry term.")
        st.stop()

    st.info("The report and visuals below are generated exclusively from the five Wikipedia pages listed above.")

    # =========================
    # Step 3 — Industry report
    # =========================
    st.markdown("<h3 class='blue-accent'>Step 3 — Industry report (under 500 words)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Business-analyst style briefing with traceable citations in the form [Source #].</div>",
        unsafe_allow_html=True
    )

    # Build sources text inline
    parts = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        url = (d.metadata or {}).get("source", "")
        text = (d.page_content or "").strip()
        text = re.sub(r"\s+", " ", text)[:2600]
        parts.append(
            f"[Source {i}]\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT EXCERPT: {text}\n"
        )
    sources_text = "\n\n".join(parts)

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
            report = response.content or ""
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

    # Remove markdown heading hashes if present
    report = re.sub(r"(?m)^#+\s*", "", report).strip()

    # Cap to 500 words inline
    words = report.split()
    if len(words) > 500:
        report = " ".join(words[:500]).rstrip() + "…"

    word_count = len(report.split())
    st.caption(f"Word count: {word_count} / 500")

    # Tabs
    tab_report, tab_visuals, tab_sources = st.tabs(["Report", "Visuals", "Sources"])

    with tab_report:
        st.markdown(
            f"""
            <div class="report-box">
            {report.replace("\n", "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_visuals:
        st.markdown("<div class='subtle'>Visuals derived from the Wikipedia source excerpts.</div>", unsafe_allow_html=True)

        # Visual 1: Source coverage by excerpt length
        titles = [(d.metadata or {}).get("title", f"Source {i+1}") for i, d in enumerate(docs)]
        lengths = [len((d.page_content or "")) for d in docs]
        st.markdown("<div class='blue-accent'>Source Coverage (Excerpt Length)</div>", unsafe_allow_html=True)
        st.bar_chart({"Source": titles, "Excerpt length": lengths}, x="Source", y="Excerpt length")

        # Visual 2: Top terms across sources
        all_text = " ".join((d.page_content or "") for d in docs)
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", all_text.lower())
        stop = {
            "the","and","for","with","that","this","from","are","was","were","has","have",
            "had","its","their","which","into","also","such","than","over","under","between",
            "about","after","before","these","those","other","more","most","used","use","using",
            "industry","market","company","companies","products","product"
        }
        counts = Counter(t for t in tokens if t not in stop)
        top_terms = counts.most_common(8)
        if top_terms:
            terms = [t[0].title() for t in top_terms]
            values = [t[1] for t in top_terms]
            st.markdown("<div class='blue-accent'>Top Topics Mentioned</div>", unsafe_allow_html=True)
            st.bar_chart({"Topic": terms, "Mentions": values}, x="Topic", y="Mentions")
        else:
            st.caption("Not enough text to extract key topics.")

        # Visual 3: Relative coverage index (derived from sources)
        if lengths:
            total = sum(lengths)
            share = [round((l / total) * 100, 2) if total else 0 for l in lengths]
            st.markdown("<div class='blue-accent'>Relative Coverage Index (Derived)</div>", unsafe_allow_html=True)
            st.area_chart({"Source": titles, "Coverage %": share}, x="Source", y="Coverage %")
            st.caption("Derived from excerpt lengths to show relative coverage per source.")

    with tab_sources:
        for u in urls:
            st.write(u)
