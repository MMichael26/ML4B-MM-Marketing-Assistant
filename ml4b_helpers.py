"""
Helper functions for ML4B Market Research Assistant
Contains Q1 (validation) and Q2 (Wikipedia ranking) implementations
"""

import os
import re
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever


# =========================
# Q1: Industry Validation with LLM
# =========================

def validate_industry_with_llm(user_input: str, api_key: str) -> dict:
    """
    Q1: Validate that user has provided a valid industry using LLM.
    
    Args:
        user_input: Raw string from user
        api_key: OpenAI API key
        
    Returns:
        dict with: is_valid (bool), industry_name (str), message (str)
    """
    
    # Check for empty input first
    if not user_input or user_input.strip() == "":
        return {
            'is_valid': False,
            'industry_name': None,
            'message': "⚠️ Please provide an industry name."
        }
    
    # Use LLM to validate if input is actually an industry
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0  # Deterministic for validation
    )
    
    system_prompt = """You are an industry classification expert.
Determine if the user's input represents a valid industry or business sector.

Valid industries include:
- Traditional: automotive, pharmaceutical, healthcare, retail, manufacturing
- Technology: software, AI, cybersecurity, semiconductor, cloud computing  
- Emerging: renewable energy, cryptocurrency, biotechnology, electric vehicles
- Services: financial services, hospitality, education, e-commerce
- Niche markets: luxury goods, fast fashion, organic food, gaming

Invalid inputs include:
- Greetings: "hello", "hi"
- Questions: "help me", "what should I do"
- Gibberish: "asdfgh", "123"
- Commands: "start", "begin"

Respond in exactly 3 lines:
VALID: [Yes/No]
INDUSTRY: [Standardized industry name if valid, or "None" if invalid]
REASON: [Brief one-sentence explanation]"""

    user_prompt = f"Validate this input: {user_input}"
    
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Parse response
        response_text = response.content.strip()
        lines = response_text.split('\n')
        
        # Extract validation result
        is_valid = False
        industry_name = None
        reason = ""
        
        for line in lines:
            if 'VALID:' in line.upper():
                is_valid = 'Yes' in line or 'YES' in line
            elif 'INDUSTRY:' in line.upper():
                industry_name = line.split(':', 1)[1].strip()
                if industry_name.lower() == 'none':
                    industry_name = None
            elif 'REASON:' in line.upper():
                reason = line.split(':', 1)[1].strip()
        
        if is_valid and industry_name:
            return {
                'is_valid': True,
                'industry_name': industry_name,
                'message': f"✅ Valid industry: {industry_name}"
            }
        else:
            return {
                'is_valid': False,
                'industry_name': None,
                'message': f"❌ Invalid input. {reason}\n\nPlease enter a valid industry (e.g., 'automotive', 'fintech', 'renewable energy')."
            }
            
    except Exception as e:
        return {
            'is_valid': False,
            'industry_name': None,
            'message': f"⚠️ Error during validation: {str(e)}"
        }


# =========================
# Q2: Wikipedia Retrieval + LLM Ranking
# =========================

def get_top_5_wikipedia_pages(industry: str, api_key: str) -> tuple:
    """
    Q2: Retrieve Wikipedia pages and use LLM to rank the top 5 most relevant.
    
    Args:
        industry: Validated industry name
        api_key: OpenAI API key
        
    Returns:
        tuple: (docs_list, urls_list)
            docs_list: List of 5 Document objects
            urls_list: List of 5 URLs (strings)
    """
    
    # Step 1: Generate diverse search queries using LLM
    search_queries = generate_search_queries_with_llm(industry, api_key)
    
    # Step 2: Retrieve Wikipedia documents
    retriever = WikipediaRetriever(top_k_results=10, lang="en")
    
    all_docs = []
    seen_titles = set()
    
    for query in search_queries:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                title = (doc.metadata or {}).get('title', '')
                if title and title not in seen_titles:
                    all_docs.append(doc)
                    seen_titles.add(title)
        except Exception as e:
            print(f"Error retrieving query '{query}': {e}")
            continue
    
    # Need at least 5 documents
    if len(all_docs) < 5:
        # Fallback: try basic search
        try:
            fallback_docs = retriever.get_relevant_documents(industry)
            for doc in fallback_docs:
                title = (doc.metadata or {}).get('title', '')
                if title and title not in seen_titles:
                    all_docs.append(doc)
                    seen_titles.add(title)
        except:
            pass
    
    # Step 3: Use LLM to rank by relevance
    if len(all_docs) >= 5:
        ranked_docs = rank_docs_with_llm(industry, all_docs[:20], api_key)  # Limit to 20 for ranking
    else:
        ranked_docs = all_docs
    
    # Get top 5
    top_5_docs = ranked_docs[:5]
    
    # Extract URLs
    urls = []
    for doc in top_5_docs:
        url = (doc.metadata or {}).get("source", "")
        if url:
            urls.append(url)
    
    return top_5_docs, urls


def generate_search_queries_with_llm(industry: str, api_key: str) -> list:
    """
    Generate diverse Wikipedia search queries using LLM.
    """
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.3  # Slightly creative for query diversity
    )
    
    prompt = f"""Generate 4 diverse Wikipedia search queries for researching the {industry} industry.

Include queries about:
1. The industry itself
2. Major companies  
3. Technology/products
4. Market trends

Return ONLY the search queries, one per line, no numbering."""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        
        # Ensure we have at least the industry name
        if not queries:
            queries = [industry]
        
        return queries[:4]
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        # Fallback queries
        return [industry, f"{industry} industry", f"{industry} companies", f"{industry} market"]


def rank_docs_with_llm(industry: str, docs: list, api_key: str) -> list:
    """
    Use LLM to rank Wikipedia documents by relevance for market research.
    """
    
    if not docs:
        return []
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0  # Deterministic ranking
    )
    
    # Create document summaries for ranking
    doc_summaries = []
    for i, doc in enumerate(docs):
        title = (doc.metadata or {}).get('title', 'Unknown')
        preview = doc.page_content[:200].replace('\n', ' ') if doc.page_content else ""
        doc_summaries.append(f"{i}. {title}: {preview}...")
    
    docs_text = "\n\n".join(doc_summaries)
    
    prompt = f"""You are ranking Wikipedia pages for business market research on the {industry} industry.

Rank by:
1. Direct industry relevance
2. Market data & statistics  
3. Company information
4. Technology & trends
5. Economic context

Return ONLY the numbers of the top 5 most relevant pages, comma-separated.
Example: 3,7,1,12,5

Do not explain. Just numbers.

Pages:
{docs_text}

Top 5 page numbers:"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Parse response - extract numbers
        response_text = response.content.strip()
        # Remove non-numeric characters except commas
        clean_text = ''.join(c for c in response_text if c.isdigit() or c == ',')
        ranked_indices = [int(x.strip()) for x in clean_text.split(',') if x.strip()]
        
    except Exception as e:
        print(f"Ranking error: {e}")
        # Fallback: return first 5
        ranked_indices = list(range(min(5, len(docs))))
    
    # Build ranked results
    ranked_docs = []
    for idx in ranked_indices:
        if 0 <= idx < len(docs):
            ranked_docs.append(docs[idx])
    
    # Fill remaining slots if needed
    for doc in docs:
        if len(ranked_docs) >= 5:
            break
        if doc not in ranked_docs:
            ranked_docs.append(doc)
    
    return ranked_docs[:5]
