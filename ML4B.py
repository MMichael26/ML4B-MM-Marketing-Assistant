import os
import re
import streamlit as st

from langchain_openai import ChatOpenAI
from ml4b_helpers import validate_industry_with_llm, get_top_5_wikipedia_pages


# =========================
# Page config
# =========================
st.set_page_config(page_title="The Best Market Research Assistant", layout="wide")

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
    code {
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("The Best Market Research Assistant")
st.caption("Designed to provide you with quick industry briefings.")

# =========================
# API key handling (Streamlit Secrets or sidebar)
# =========================
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Fallback: sidebar input if no API key found
if not api_key:
    st.sidebar.header("🔑 API Key")
    st.sidebar.write("Enter your OpenAI API key to use this app.")
    show_key = st.sidebar.checkbox("Show API key", value=False)
    user_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="default" if show_key else "password"
    )
    api_key = user_key

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Add it to Streamlit Secrets or enter it in the sidebar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# =========================
# Helper function for Q3
# =========================
def build_sources_text(docs) -> str:
    """Build context from Wikipedia pages for Q3 report generation."""
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
    """Ensure report is under 500 words."""
    words = (text or "").split()
    if len(words) <= 500:
        return text.strip()
    return " ".join(words[:500]).rstrip() + "…"


# =========================
# UI — Q1: Industry Validation
# =========================
st.markdown("<h3 class='blue-accent'>Provide an Industry</h3>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>The assistant checks that a valid industry is provided. If not, it requests an update.</div>",
    unsafe_allow_html=True
)

industry_input = st.text_input("Industry (e.g., Fast fashion, Semiconductor industry, Renewable energy)")

if st.button("Validate & Continue", type="primary"):
    
    # Q1: Validate with LLM
    with st.spinner("🤖 Validating industry input..."):
        validation = validate_industry_with_llm(industry_input, api_key)
    
    if not validation['is_valid']:
        # Invalid input
        st.error(validation['message'])
        st.stop()
    
    # Valid input - proceed
    st.success(validation['message'])
    industry = validation['industry_name']
    
    # =========================
    # Q2: Wikipedia URLs (Top 5 with LLM Ranking)
    # =========================
    st.markdown("<h3 class='blue-accent'>Top 5 Wikipedia URLs</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>The assistant retrieves Wikipedia pages and uses AI to rank the five most relevant.</div>",
        unsafe_allow_html=True
    )

    with st.spinner("🔍 Searching and ranking Wikipedia pages..."):
        docs, urls = get_top_5_wikipedia_pages(industry, api_key)

    if not urls or len(urls) < 5:
        st.error("Could not retrieve 5 Wikipedia pages. Try a more specific industry term.")
        st.stop()

    st.success(f"✅ Found and ranked {len(urls)} Wikipedia pages")
    
    for i, url in enumerate(urls, 1):
        st.write(f"{i}. {url}")

    st.info("📊 The report below is generated exclusively from these five Wikipedia pages.")

    # =========================
    # Q3: Industry Report (<500 words)
    # =========================
    st.markdown("<h3 class='blue-accent'>Industry Report (<500 words)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Business-analyst style briefing with source citations [Source #].</div>",
        unsafe_allow_html=True
    )

    sources_text = build_sources_text(docs)

    # UPDATED: Use gpt-4o with temperature 0.7 for better quality
    llm = ChatOpenAI(
        model="gpt-4o",           
        temperature=0.7            
    )

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
        f"Industry: {industry}\n\n"
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

    with st.spinner("✍️ Generating industry briefing..."):
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        report = cap_500_words(response.content)

    word_count = len(report.split())
    
    if word_count <= 500:
        st.caption(f"✅ Word count: {word_count} / 500")
    else:
        st.caption(f"⚠️ Word count: {word_count} / 500 (slightly over)")

    st.markdown(
        f"""
        <div class="report-box">
        {report.replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Download button
    st.download_button(
        label="⬇️ Download Report",
        data=report,
        file_name=f"{industry.replace(' ', '_')}_report.txt",
        mime="text/plain"
    )
```

---
