"""
Main Entry Point - CrewAI Vacation Planner Streamlit App

This file coordinates the Streamlit UI, theme styles, page configuration,
sidebar configurations, and event handlers. It delegates core business logic
to the other modular files:
- utils.py: Date parsing, defaults, and timeline checklist utilities.
- crew_planner.py: Multi-agent orchestration using CrewAI and validation logic.
- cost_calculator.py: Interactive math-based cost calculator component.

Run this app using:
    streamlit run app.py
"""

# Warning control
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

# Import modular components
from utils import (
    setup_safe_console_output,
    generate_default_demo_dates,
    generate_booking_timeline
)

# Apply safe writer globally for Windows Unicode symbol handling
setup_safe_console_output()

# Load environment variables (such as OPENAI_API_KEY from local .env)
load_dotenv()
from crew_planner import (
    validate_destination_quick,
    run_vacation_planner
)
from cost_calculator import render_cost_calculator

# Streamlit Page UI
# ----------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="Vacation Planner",
        page_icon="🌴",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Theme Styling — Noto Sans loaded for full Unicode currency symbol support (₹ € £ ¥ etc.)
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700;800&family=Noto+Sans+Symbols+2&display=swap" rel="stylesheet">
        <style>
        /* Apply Noto Sans globally — covers all currency symbols */
        html, body, [class*="css"], .stMarkdown, .stMetric, .stText {
            font-family: 'Noto Sans', 'Noto Sans Symbols 2', 'Segoe UI', Arial, sans-serif !important;
        }
        .main {
            background-color: #f8fafc;
        }
        .stButton>button {
            background-color: #0284c7;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: all 0.2s ease-in-out;
            width: 100%;
            font-family: 'Noto Sans', sans-serif !important;
        }
        .stButton>button:hover {
            background-color: #0369a1;
            border: none;
            color: white;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        /* Metric card styling */
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 18px;
            border-radius: 10px;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
            border: 1px solid #e2e8f0;
        }
        /* Metric value — larger, bold, Noto Sans for proper currency glyph rendering */
        div[data-testid="stMetricValue"] > div {
            font-family: 'Noto Sans', 'Noto Sans Symbols 2', 'Segoe UI Symbol', Arial, sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
            letter-spacing: -0.01em;
        }
        /* Metric label */
        div[data-testid="stMetricLabel"] > div {
            font-family: 'Noto Sans', sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        h1 {
            color: #0f172a;
            font-weight: 800;
            font-family: 'Noto Sans', sans-serif !important;
        }
        h2 {
            color: #0284c7;
            font-weight: 700;
            font-family: 'Noto Sans', sans-serif !important;
        }
        .sidebar .sidebar-content {
            background-color: #0f172a;
        }
        .stForm {
            border-radius: 12px;
            background-color: white;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            border: 1px solid #f1f5f9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Title & Subtitle
    st.title("🌴 Vacation Planner")
    st.markdown("##### *Optimized concurrent execution workflow powered by CrewAI*")

    # Fetch API keys from environment (default values)
    default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
    default_tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    # Sidebar Configuration (left panel)
    st.sidebar.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=400&auto=format&fit=crop", width="stretch")
    
    # Use environment variables directly (hidden from UI)
    openai_api_key = default_openai_api_key
    tavily_api_key = default_tavily_api_key
    
    st.sidebar.markdown("### 🤖 CrewAI Multi-Agent Team")
    st.sidebar.markdown("""
    This app orchestrates **4 specialized AI agents** operating in parallel and sequential workflows:

    - 🔍 **Research Specialist**: Real-time web search for flight tiers, lodging rates, transport, & attractions.
    - 📅 **Itinerary Planner**: Crafts detailed day-by-day schedules (*Async Parallel Execution*).
    - 💰 **Financial Coordinator**: Calculates local currency budgets & 5 money saving tips (*Async Parallel Execution*).
    - 📋 **Trip Director**: Merges all agent outputs into a polished final master deliverable.
    """)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🛠️ Agent Tools Suite")
    st.sidebar.markdown("""
    - 🌐 `TrackedTavilySearchTool`: Live Tavily web search with URL citation tracking.
    - 🧮 `CalculatorTool`: Fast mathematical expression evaluator.
    - 📊 `TripCostCalculatorTool`: Structured budget math engine shared with the UI calculator.
    """)

    st.sidebar.markdown("---")

    # Tavily Web Search status badge in sidebar
    st.sidebar.markdown("### 🌐 Live Web Search Status")
    if tavily_api_key:
        st.sidebar.success("✅ Tavily API Connected — Agents will query real-time web data.")
    else:
        st.sidebar.warning(
            "⚠️ **Tavily API Key not set.**\n\n"
            "Agents will use internal LLM knowledge base.\n\n"
            "Get a free key at [app.tavily.com](https://app.tavily.com)."
        )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 💡 Tips for Best Results")
    st.sidebar.info("""
    - **Be specific with your destination**: State exact cities or areas (e.g., *'Plan a 7-day trip to Kyoto and Osaka'*).
    - **Define traveler composition**: Specify *'1 adult'*, *'2 adults'*, or *'2 adults, 2 kids (ages 8 and 5)'*. The agents adapt pacing, lodging, and child-friendliness!
    - **Specify preferences**: State your travel style, budget tier, or special interests (e.g. *food, anime, hiking*).
    """)

    # Setup default dates
    default_demo_dates = generate_default_demo_dates()

    # Top-level tabs: Vacation Planner vs Cost Calculator
    tab_planner, tab_calc = st.tabs(["\U0001f334 Vacation Planner", "\U0001f9ee Cost Calculator"])

    with tab_planner:
        # Plan parameters section
        col1, col2 = st.columns([1.1, 1.8])

    with col1:
        st.markdown("### 📍 Vacation Specifications")
        
        with st.form("planning_form"):
            # Source & Target Destination Inputs
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                origin_city = st.text_input(
                    "🛫 Source / Departure Location",
                    value="New Delhi, India (DEL)",
                    placeholder="e.g. New Delhi, India OR New York (JFK)",
                    help="Departure airport/city for flight route and airfare calculation."
                )
            with loc_col2:
                destination_city = st.text_input(
                    "🛬 Target Destination",
                    value="Rome, Italy",
                    placeholder="e.g. Tokyo, Japan OR Rome, Italy",
                    help="Destination city or country for your vacation."
                )

            # Vacation Goal
            vacation_goal_input = st.text_area(
                "Vacation Goal & Experiences",
                value="Plan a 7-day family vacation focusing on Rome, Florence, and Venice.",
                placeholder="e.g. Explore historical landmarks, local cuisine, and anime shops",
                height=80
            )
            
            # Travel Dates selection
            use_custom_date = st.checkbox("Use custom date description (e.g. 'a week during summer break')", value=False)
            
            if use_custom_date:
                dates = st.text_input("Travel Dates / Period", value="June 15 to June 22, 2026")
            else:
                dates_input = st.date_input(
                    "Select Travel Dates",
                    value=(datetime.now() + timedelta(days=90), datetime.now() + timedelta(days=105)),
                    min_value=datetime.now()
                )
                
                if isinstance(dates_input, tuple) and len(dates_input) == 2:
                    start_d, end_d = dates_input
                    dates = f"{start_d.strftime('%Y-%m-%d')} to {end_d.strftime('%Y-%m-%d')}"
                elif isinstance(dates_input, tuple) and len(dates_input) == 1:
                    dates = dates_input[0].strftime('%Y-%m-%d')
                else:
                    dates = dates_input.strftime('%Y-%m-%d') if dates_input else default_demo_dates
            
            # Traveler Composition (dynamic input)
            family_size = st.text_input(
                "Travel Party Size & Composition",
                value="2 adults, 2 children (ages 8 and 5)",
                placeholder="e.g. 1 adult OR 2 adults, 2 kids ages 10 and 6"
            )

            # Preferences
            preferences = st.text_area(
                "Preferences & Constraints",
                value="Moderate budget, kid-friendly activities, avoid long overnight travel, prefer central accommodations",
                placeholder="e.g. Budget-friendly, food focus, close to train stations",
                height=80
            )
            
            submit_btn = st.form_submit_button("🚀 Generate Vacation Plan")

        # Add quick demo buttons outside the form
        st.markdown("#### ⚡ Quick Actions")
        demo_btn = st.button("🇮🇹 Run Default Demo (Italy Family Vacation)")

    # Trigger execution on form submit or demo click
    trigger_execution = False
    if submit_btn:
        if not openai_api_key:
            st.error("⚠️ **OpenAI API Key missing!** Please set the `OPENAI_API_KEY` variable in a `.env` file in the project folder to run this application.")
        else:
            trigger_execution = True

    if demo_btn:
        if not openai_api_key:
            st.error("⚠️ **OpenAI API Key missing!** Please set the `OPENAI_API_KEY` variable in a `.env` file in the project folder to run this application.")
        else:
            # Pre-filled demo values
            vacation_goal = "Plan a 7-day family vacation to Italy focusing on Rome, Florence, and Venice."
            dates = default_demo_dates
            family_size = "2 adults, 2 children (ages 8 and 5)"
            preferences = "Moderate budget, kid-friendly activities, avoid long overnight travel, prefer central accommodations"
            trigger_execution = True

    if trigger_execution:
        # Clear any previous plan results so stale data is never shown after a new run
        if "plan_result" in st.session_state:
            del st.session_state["plan_result"]
        # Also clear any previous validation state
        if "validation_state" in st.session_state:
            del st.session_state["validation_state"]
        if "city_clarification_goal" in st.session_state:
            del st.session_state["city_clarification_goal"]

        # Step 1: Validate destination before launching the full planner
        with st.spinner("🔍 Validating destination..."):
            val_status, val_message = validate_destination_quick(vacation_goal, openai_api_key)

        if val_status == "INVALID_COUNTRY":
            # Show error and stop — do NOT run the planner
            st.session_state["validation_state"] = "INVALID_COUNTRY"
            st.session_state["validation_message"] = val_message

        elif val_status == "NEEDS_CITY":
            # Ask user to clarify the city before proceeding
            st.session_state["validation_state"] = "NEEDS_CITY"
            st.session_state["validation_message"] = val_message
            # Store the pending form values so we can reuse them after city is entered
            st.session_state["pending_goal"] = vacation_goal
            st.session_state["pending_dates"] = dates
            st.session_state["pending_family"] = family_size
            st.session_state["pending_preferences"] = preferences

        else:
            # VALID — run the planner with origin city context
            full_preferences = f"Departing from: {origin_city}. {preferences}" if origin_city else preferences
            try:
                plan_data = run_vacation_planner(
                    vacation_goal=vacation_goal,
                    dates=dates,
                    family_size=family_size,
                    preferences=full_preferences,
                    api_key=openai_api_key,
                    tavily_api_key=tavily_api_key
                )
                st.session_state.plan_result = plan_data
            except Exception as e:
                st.error(f"An error occurred while running the planner: {e}")

    # --- Validation feedback shown inside col1 (below the form) ---
    with col1:
        # Show invalid country error
        if st.session_state.get("validation_state") == "INVALID_COUNTRY":
            st.error(f"⚠️ **Invalid destination detected.**\n\n{st.session_state.get('validation_message', '')}\n\nPlease correct the destination in the Vacation Goal field and try again.")

        # Show city clarification prompt
        elif st.session_state.get("validation_state") == "NEEDS_CITY":
            st.warning(f"📍 **More detail needed.**\n\n{st.session_state.get('validation_message', '')}")
            city_input = st.text_input(
                "Which specific city or area would you like to visit?",
                placeholder="e.g. Jaipur, Mumbai, Udaipur",
                key="city_clarification_input"
            )
            if st.button("✅ Continue with this city", key="city_confirm_btn") and city_input.strip():
                # Merge city into the original goal and re-run
                original_goal = st.session_state.get("pending_goal", "")
                updated_goal = original_goal.rstrip(". ") + f", specifically visiting {city_input.strip()}."
                # Clear validation state so we don't loop
                del st.session_state["validation_state"]
                del st.session_state["validation_message"]
                full_preferences = f"Departing from: {origin_city}. {preferences}" if origin_city else preferences
                try:
                    plan_data = run_vacation_planner(
                        vacation_goal=updated_goal,
                        dates=st.session_state.get("pending_dates", dates),
                        family_size=st.session_state.get("pending_family", family_size),
                        preferences=full_preferences,
                        api_key=openai_api_key,
                        tavily_api_key=tavily_api_key
                    )
                    st.session_state.plan_result = plan_data
                except Exception as e:
                    st.error(f"An error occurred while running the planner: {e}")

    # Render results on the right column if they exist in state
    with col2:
        st.markdown("### 🗺️ Generated Travel Plan")
        
        if "plan_result" in st.session_state:
            res = st.session_state.plan_result
            
            # Travel metadata metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Destination Currency", res["currency"])
            with m_col2:
                st.metric("Travel Dates", res["enhanced_dates"])
            with m_col3:
                st.metric("Travel Party", res["enhanced_family"])
                
            st.markdown("---")

            # Extract Money Saving Tips from plan
            plan_text = res["result"]
            tips_section = ""
            tips_marker = "## Money Saving Tips"
            if tips_marker in plan_text:
                parts = plan_text.split(tips_marker, 1)
                tips_section = parts[1].strip()
                # Also trim at the next ## heading if present so we don't double-display
                import re as _re
                next_heading = _re.search(r'\n##\s', tips_section)
                if next_heading:
                    tips_section = tips_section[:next_heading.start()].strip()

            # 1. Main Markdown Travel Plan (strip out the Money Saving Tips section to avoid duplication)
            display_text = plan_text.split(tips_marker)[0].strip() if tips_marker in plan_text else plan_text
            st.markdown(display_text)

            # 2. Display dynamically calculated booking timeline (with ASAP corrections)
            booking_timeline_html = generate_booking_timeline(res["enhanced_dates"])
            if booking_timeline_html:
                with st.expander("📅 Calculated Booking Milestones & Deadlines", expanded=True):
                    st.markdown(booking_timeline_html)

            # 3. Web Sources used by Tavily (only shown when Tavily was active)
            sources = res.get("sources", [])
            if sources:
                with st.expander(f"🌐 {len(sources)} Web Source(s) Used by Agents", expanded=False):
                    st.caption("The following pages were fetched live by Tavily during plan generation:")
                    # Group sources by the search query that triggered them
                    seen_queries = {}
                    for s in sources:
                        q = s.get("query", "General search")
                        seen_queries.setdefault(q, []).append(s)
                    for query, srcs in seen_queries.items():
                        st.markdown(f"**Search:** *{query}*")
                        for s in srcs:
                            label = s["title"] if s["title"] and s["title"] != s["url"] else s["url"]
                            st.markdown(f"  - [{label}]({s['url']})")
                        st.markdown("")

            # 4. Money Saving Tips at the bottom
            if tips_section:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                        border: 1px solid #86efac;
                        border-left: 4px solid #16a34a;
                        border-radius: 10px;
                        padding: 18px 22px;
                        margin-bottom: 18px;
                    ">
                        <div style="font-family:'Noto Sans',sans-serif; font-size:1rem; font-weight:700; color:#15803d; margin-bottom:10px;">
                            💡 Money Saving Tips
                        </div>
                """, unsafe_allow_html=True)
                st.markdown(tips_section)
                st.markdown("</div>", unsafe_allow_html=True)

            # 5. Download plan as markdown file
            st.download_button(
                label="📥 Download Travel Plan (Markdown)",
                data=res["result"],
                file_name="my_vacation_plan.md",
                mime="text/markdown"
            )
        else:
            st.info("👈 Enter specifications and click 'Generate Vacation Plan' or click 'Run Default Demo' to begin.")
            st.image("https://images.unsplash.com/photo-1488646953014-85cb44e25828?q=80&w=600&auto=format&fit=crop", use_container_width=True)

    # Tab 2: Standalone Cost Calculator
    with tab_calc:
        render_cost_calculator()
