from data_loader import ElasticGroundwaterSearch
import streamlit as st
import glob
import os
import json
import re
from groq import Groq
from typing import List, Dict, Any

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def detect_numeric_query(query: str) -> Dict[str, Any]:
    """
    Detect if query is asking for specific numeric ranges on groundwater parameters.
    """
    query_lower = query.lower().strip()

    # Default return (not numeric)
    result: Dict[str, Any] = {
        "is_numeric_query": False,
        "parameter": None,
        "type": None,
        "min_value": None,
        "max_value": None,
        "limit": None
    }

    # Keywords mapped to parameters
    parameter_keywords = {
        "rainfall_total_mm": ["rainfall", "rain", "precipitation", "mm"],
        "total_gw_availability_ham": ["total gw", "total groundwater", "total water", "availability", "ham", "groundwater availability"],
        "net_gw_availability_ham": ["net gw", "net groundwater", "net water", "net availability"],
        "stage_of_development_percent": ["development", "stage", "exploitation", "percent", "%", "stage of development"],
    }

    # Detect parameter
    detected_parameter = None
    for param, keywords in parameter_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_parameter = param
            break

    if not detected_parameter:
        return result  # no parameter detected

    # Regex patterns
    above_pattern = r"(?:above|greater than|more than|over|>)\s*(\d+(?:\.\d+)?)"
    below_pattern = r"(?:below|less than|under|<)\s*(\d+(?:\.\d+)?)"
    between_pattern = r"between\s*(\d+(?:\.\d+)?)\s*(?:and|to)\s*(\d+(?:\.\d+)?)"
    top_pattern = r"(?:top|highest|best|lowest|least|min|max)\s*(\d+)"

    # Apply regex
    above_match = re.search(above_pattern, query_lower)
    below_match = re.search(below_pattern, query_lower)
    between_match = re.search(between_pattern, query_lower)
    top_match = re.search(top_pattern, query_lower)

    # Detect query type
    if above_match:
        result.update({
            "is_numeric_query": True,
            "parameter": detected_parameter,
            "type": "above",
            "min_value": float(above_match.group(1)),
            "limit": int(top_match.group(1)) if top_match else None
        })
    elif below_match:
        result.update({
            "is_numeric_query": True,
            "parameter": detected_parameter,
            "type": "below",
            "max_value": float(below_match.group(1)),
            "limit": int(top_match.group(1)) if top_match else None
        })
    elif between_match:
        result.update({
            "is_numeric_query": True,
            "parameter": detected_parameter,
            "type": "between",
            "min_value": float(between_match.group(1)),
            "max_value": float(between_match.group(2)),
            "limit": int(top_match.group(1)) if top_match else None
        })
    elif top_match:
        if any(word in query_lower for word in ["high", "above", "maximum", "most", "max"]):
            result.update({
                "is_numeric_query": True,
                "parameter": detected_parameter,
                "type": "top_highest",
                "limit": int(top_match.group(1))
            })
        elif any(word in query_lower for word in ["low", "lowest", "minimum", "least", "min"]):
            result.update({
                "is_numeric_query": True,
                "parameter": detected_parameter,
                "type": "top_lowest",
                "limit": int(top_match.group(1))
            })

    return result

def safe_format(value, fmt=".1f", default=0):
    """Safely format numeric values with fallback"""
    if value is None or value == "N/A":
        value = default
    try:
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(default)

def get_latest_json():
    """Get the most recent JSON file"""
    files = glob.glob("rajasthan_groundwater_blocks.json")
    if not files:
        raise FileNotFoundError("No JSON files found. Run extract_groundwater.py first.")
    return max(files, key=os.path.getctime)

@st.cache_resource(show_spinner=False)
def load_data():
    """Load and cache the search engine"""
    json_file = get_latest_json()
    return ElasticGroundwaterSearch(json_file)

def format_context_for_llm(results: List[Dict[str, Any]], query: str, search_type: str = "hybrid") -> str:
    """Enhanced context formatting with clear data structure"""
    if not results:
        return "No relevant data found for your query."

    # Filter out summary/total records (additional safety check)
    filtered_results = []
    for r in results:
        block_name = r.get("block_name", "").strip().lower()
        if block_name not in ["total", "district_total", "state_total", ""]:
            filtered_results.append(r)

    if not filtered_results:
        return "No individual block data found."

    context_parts = [
        f"USER QUERY: {query}",
        f"FOUND {len(filtered_results)} blocks matching the criteria:",
        ""
    ]

    # Group by district for better organization
    by_district = {}
    for r in filtered_results:
        district = r.get("district_name", "Unknown")
        if district not in by_district:
            by_district[district] = []
        by_district[district].append(r)

    # Sort districts by name for consistent output
    for district in sorted(by_district.keys()):
        district_results = by_district[district]
        context_parts.append(f"{district} DISTRICT:")

        for r in district_results:
            block = r.get("block_name", "Unknown")
            category = r.get("total_category", "Not Categorized")
            rainfall = safe_format(r.get("rainfall_total_mm"), ".0f", "No data")
            total_gw = safe_format(r.get("total_gw_availability_ham"), ".2f", "No data")
            net_gw = safe_format(r.get("net_gw_availability_ham"), ".2f", "No data")
            stage = safe_format(r.get("stage_of_development_percent"), ".1f", "No data")

            context_parts.append(
                f"- {block}: Category={category}, Rainfall={rainfall}mm, Total_GW={total_gw}HAM, Net_GW={net_gw}HAM, Development={stage}%"
            )
        context_parts.append("")

    return "\n".join(context_parts)

def generate_enhanced_prompt(context: str, query: str, numeric_info: Dict = None) -> str:
    """Generate improved prompt focusing on clear tabular output"""

    filter_info = ""
    if numeric_info and numeric_info.get("is_numeric_query"):
        param = numeric_info["parameter"]
        query_type = numeric_info["type"]
        if query_type == "above":
            filter_info = f"Showing only blocks with {param} above {numeric_info['min_value']}."
        elif query_type == "below":
            filter_info = f"Showing only blocks with {param} below {numeric_info['max_value']}."
        elif query_type == "between":
            filter_info = f"Showing only blocks with {param} between {numeric_info['min_value']} and {numeric_info['max_value']}."
        elif "top" in query_type:
            direction = "highest" if "highest" in query_type else "lowest"
            filter_info = f"Showing top {numeric_info['limit']} blocks with {direction} {param}."

    return f"""You are analyzing groundwater data for blocks in Rajasthan, India.

{context}

{filter_info}

USER QUESTION: {query}

INSTRUCTIONS:
1. Present the data in a clear markdown table format
2. Include columns: District, Block, Category
3. Sort the results logically (by rainfall, district name, or relevance to query)
4. After the table, provide a brief summary with key insights
5. Do not include any "does not meet criteria" statements - all data shown already meets the query criteria
6. Use actual numbers from the data, not approximations
7. Also include columns as asked by user from these: Rainfall (mm), Total GW (HAM), Net GW (HAM), Development (%)

GROUNDWATER CATEGORIES:
- safe: Sustainable groundwater usage
- semi_critical: Moderate stress levels
- critical: High stress, needs attention
- over_exploited: Unsustainable usage, urgent action needed

Please create a table and summary:"""

def display_search_results(results: List[Dict[str, Any]], search_type: str):
    """Display search results in an expandable section"""
    if not results:
        st.info("No results found.")
        return

    with st.expander(f"View Raw Data ({len(results)} blocks found)", expanded=False):
        for i, result in enumerate(results, 1):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.write(f"**{result.get('district_name', 'Unknown')}**")
                st.write(f"Block: {result.get('block_name', 'N/A')}")

            with col2:
                st.write(f"Category: **{result.get('total_category', 'N/A')}**")
                st.write(f"Rainfall: {safe_format(result.get('rainfall_total_mm'), '.0f')} mm")

            with col3:
                if "_relevance_score" in result:
                    st.write(f"Score: {result['_relevance_score']:.2f}")
                elif "_similarity_score" in result:
                    st.write(f"Similarity: {result['_similarity_score']:.3f}")

            st.divider()

def main():
    st.set_page_config(
        page_title="Groundwater Intelligence",
        page_icon="💧",
        layout="wide"
    )

    st.title("💧 Rajasthan Groundwater Intelligence")
    st.markdown("*AI-powered groundwater data analysis*")

    # Initialize search engine
    try:
        with st.spinner("Loading groundwater database..."):
            engine = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if query := st.chat_input("Ask me about Rajasthan groundwater (e.g., 'rainfall above 800mm', 'over-exploited blocks', 'Jaipur district analysis')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Perform search based on selected method
        with st.chat_message("assistant"):
            with st.spinner("Searching database..."):
                try:
                    # Check for district first to run targeted queries
                    detected_district = engine.detect_district_in_query(query)
                    # Check if this is a numeric parameter query
                    numeric_query = detect_numeric_query(query)

                    if numeric_query["is_numeric_query"]:
                        param = numeric_query["parameter"]
                        query_type = numeric_query["type"]
                        limit = int(numeric_query["limit"]) if numeric_query["limit"] else None

                        # For district-specific searches, get district blocks first, then filter
                        if detected_district:
                            district_variants = [detected_district, detected_district.upper(), detected_district.lower(), detected_district.title()]
                            district_blocks = []

                            for variant in district_variants:
                                district_blocks = engine.get_district_blocks(variant)
                                if district_blocks:
                                    break

                            if district_blocks:
                                # Apply numeric filtering to district blocks
                                filtered_results = []
                                for block in district_blocks:
                                    param_value = block.get(param, 0)

                                    if query_type == "above" and param_value >= numeric_query["min_value"]:
                                        filtered_results.append(block)
                                    elif query_type == "below" and param_value <= numeric_query["max_value"]:
                                        filtered_results.append(block)
                                    elif query_type == "between" and numeric_query["min_value"] <= param_value <= numeric_query["max_value"]:
                                        filtered_results.append(block)

                                # Sort results
                                if query_type in ["above", "between"] or query_type == "top_highest":
                                    filtered_results.sort(key=lambda x: x.get(param, 0), reverse=True)
                                else:
                                    filtered_results.sort(key=lambda x: x.get(param, 0))

                                # Limit results if specified
                                if limit:
                                    filtered_results = filtered_results[:limit]

                                results = filtered_results
                                search_type = f"district-filtered numeric"
                            else:
                                st.error(f"Could not find any blocks for district: {detected_district}")
                                results = []

                        else:
                            # Use global search methods for non-district queries
                            if query_type == "above":
                                results = engine.search_parameter_range(
                                    parameter=param,
                                    min_value=numeric_query["min_value"],
                                    top_k=limit,
                                    sort_order="desc"
                                )
                            elif query_type == "below":
                                results = engine.search_parameter_range(
                                    parameter=param,
                                    max_value=numeric_query["max_value"],
                                    top_k=limit,
                                    sort_order="asc"
                                )
                            elif query_type == "between":
                                results = engine.search_parameter_range(
                                    parameter=param,
                                    min_value=numeric_query["min_value"],
                                    max_value=numeric_query["max_value"],
                                    top_k=limit,
                                    sort_order="desc"
                                )
                            elif query_type in ["top_highest", "top_lowest"]:
                                highest = query_type == "top_highest"
                                results = engine.search_top_blocks_by_parameter(
                                    parameter=param,
                                    top_k=limit,
                                    highest=highest
                                )

                    else:
                        # Execute normal search
                        if detected_district:
                            # For text searches with district, get district blocks and do text matching
                            all_results = engine.search(query, top_k=50, min_score=0.0)
                            results = [r for r in all_results if r.get('district_name', '').upper() == detected_district.upper()][:20]
                        else:
                            # Normal search without district filtering
                            results = engine.search(query, top_k=20, min_score=0.1)

                    # Debug: Log search results count
                    st.info(f"Found {len(results)} blocks matching criteria")
                    if not results:
                        answer = "No blocks found matching your criteria. Please try a different search or check if the district name is correct."
                    else:
                        # Format context and generate LLM response
                        context = format_context_for_llm(results, query)
                        prompt = generate_enhanced_prompt(context, query, numeric_query if numeric_query["is_numeric_query"] else None)

                        # Get LLM response
                        completion = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=1200,
                        )
                        answer = completion.choices[0].message.content

                    # Display the answer
                    st.markdown(answer)

                    # Display search results
                    display_search_results(results, "search")

                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error during search: {str(e)}"
                    st.error(error_msg)

                    # Fallback: show raw results if available
                    try:
                        fallback_results = engine.search(query, top_k=5)
                        if fallback_results:
                            st.info("Here's some data that might help:")
                            display_search_results(fallback_results, "fallback")
                    except:
                        st.error("Unable to retrieve data. Please try again.")

    # Quick action buttons
    st.markdown("---")
    st.subheader("Quick Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("High Rainfall Areas"):
            if "auto_query" not in st.session_state:
                st.session_state.auto_query = "blocks with rainfall above 800 mm"
                st.rerun()

    with col2:
        if st.button("Low Rainfall Areas"):
            if "auto_query" not in st.session_state:
                st.session_state.auto_query = "blocks with rainfall below 400 mm"
                st.rerun()

    with col3:
        if st.button("Critical Zones"):
            if "auto_query" not in st.session_state:
                st.session_state.auto_query = "over-exploited blocks"
                st.rerun()

    with col4:
        if st.button("Safe Blocks"):
            if "auto_query" not in st.session_state:
                st.session_state.auto_query = "safe category blocks"
                st.rerun()

    # Handle auto-triggered queries
    if "auto_query" in st.session_state:
        query = st.session_state.auto_query
        del st.session_state.auto_query

        # Add to chat and process
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

if __name__ == "__main__":
    main()
