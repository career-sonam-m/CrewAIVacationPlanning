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
from cost_calculator import render_cost_calculator, sync_plan_to_calculator

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
        /* Apply Noto Sans globally — covers all text and currency symbols */
        * {
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
            font-size: 1.75rem !important;
            line-height: 1.3 !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            color: #0284c7;
            font-weight: 700;
            font-family: 'Noto Sans', sans-serif !important;
            font-size: 1.4rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-weight: 700;
            font-family: 'Noto Sans', sans-serif !important;
            font-size: 1.15rem !important;
            color: #334155;
            margin-top: 1.25rem !important;
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
        /* Base font for input labels */
        .stTextInput label, .stTextArea label, .stDateInput label, .stCheckbox label {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }
        /* Primary CTA button (Generate Vacation Plan) — subtly distinct from secondary/demo buttons */
        button[kind="primary"], button[kind="primaryFormSubmit"],
        [data-testid*="-primary"] button, button[data-testid*="-primary"] {
            background-color: #0f172a !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 0.6rem 1.2rem !important;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.15) !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
        [data-testid*="-primary"] button:hover, button[data-testid*="-primary"]:hover {
            background-color: #1e293b !important;
            box-shadow: 0 2px 5px 0 rgb(0 0 0 / 0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Helper: render a consistent, beautified section header with optional icon & subtitle
    def render_section_header(title: str, icon: str = None, subtitle: str = None) -> str:
        icon_html = f"<div style='background:linear-gradient(90deg,#0369a1,#0284c7);color:white;padding:8px;border-radius:10px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;margin-right:12px;font-size:18px'>{icon}</div>" if icon else ""
        subtitle_html = f"<div style='font-size:0.9rem;color:#64748b;margin-top:4px'>{subtitle}</div>" if subtitle else ""
        return (
            f"<div style='display:flex;align-items:center;margin-top:10px;margin-bottom:10px'>"
            f"{icon_html}"
            f"<div style='line-height:1'>"
            f"<div style='font-size:1.25rem;font-weight:800;color:#0f172a;margin-bottom:0.15rem'>{title}</div>"
            f"{subtitle_html}"
            f"</div>"
            f"</div>"
        )

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
    This app orchestrates **5 specialized AI agents** operating in parallel and sequential workflows:

    - ✈️ **Flight & Transport Specialist**: Real-time web search for flight fares & local transport options.
    - 🏨 **Accommodation & Attractions Specialist**: Real-time web search for lodging rates & top attractions.
    - 📅 **Itinerary Planner**: Crafts detailed day-by-day schedules.
    - 💰 **Financial Coordinator**: Calculates local currency budgets & 5 money saving tips.
    - 📋 **Trip Director**: Merges all agent outputs into a polished final master deliverable.
    """)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🛠️ Agent Tools Suite")
    st.sidebar.markdown("""
    - 🌐 `TrackedTavilySearchTool`: Live Tavily web search with URL citation tracking.
    - 🧮 `CalculatorTool`: General-purpose evaluator used by agents for basic arithmetic checks (e.g. rate × days).
    - 📊 `TripCostCalculatorTool`: Structured domain-specific tool that calculates full itemized budgets, subtotals, contingency buffers, and per-person rates.
    """)

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
    tab_planner, tab_calc = st.tabs(["🌴 Vacation Planner", "🧮 Cost Calculator"])

    with tab_planner:
        # Plan parameters section
        col1, col2 = st.columns([1.2, 1.7])

    with col1:
        st.markdown(render_section_header("Vacation Specifications", icon="📍", subtitle="Enter trip details"), unsafe_allow_html=True)
        # Show prominent warning if OpenAI key is not present so users notice immediately
        if not default_openai_api_key:
            st.warning("OpenAI API key not found. The 'Generate Vacation Plan' button requires an API key to run the planner. You can still run an offline demo using the 'Run Default Demo' button.")
        
        with st.form("planning_form"):
            # Source & Target Destination Inputs stacked cleanly for readability
            origin_city = st.text_input(
                "🛫 Source / Departure Location",
                value="New Delhi, India (DEL)",
                placeholder="e.g. New Delhi, India OR New York (JFK)",
                help="Departure airport/city for flight route and airfare calculation."
            )
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
            
            submit_btn = st.form_submit_button("🚀 Generate Vacation Plan", type="primary", use_container_width=True)

        # Add quick demo buttons outside the form
        st.markdown(render_section_header("Quick Actions", icon="⚡", subtitle="Run demos or presets"), unsafe_allow_html=True)
        demo_btn = st.button("🇮🇹 Run Default Demo (Italy Family Vacation)")

    # Trigger execution on form submit or demo click
    trigger_execution = False
    if submit_btn:
        if not openai_api_key:
            # Provide clearer guidance and an offline demo fallback
            st.error("⚠️ **OpenAI API Key missing!** Set `OPENAI_API_KEY` in a `.env` file to run the planner.")
            if st.button("Run a quick offline demo instead (no API required)"):
                # Create a lightweight demo plan so the UI shows results without calling APIs
                demo_plan = {
                    "enhanced_goal": f"Demo plan for {destination_city}",
                    "enhanced_dates": dates,
                    "enhanced_family": family_size,
                    "enhanced_preferences": preferences,
                    "currency": "USD",
                    "currency_symbol": "$",
                    "result": f"Demo generated plan for {destination_city}.\n\n## Money Saving Tips\n1) Book early...",
                    "sources": [],
                    "extracted_budget": {
                        "flight_pp": 400.0,
                        "hotel_pn": 150.0,
                        "food_ppd": 50.0,
                        "activities": 200.0,
                        "transport": 80.0,
                        "num_adults": 2,
                        "num_children": 0,
                        "num_days": 7
                    }
                }
                st.session_state.plan_result = demo_plan
                # Mark calculator as needing to re-sync with the new demo plan
                st.session_state["calc_synced_with_plan"] = False
                st.session_state.update(sync_plan_to_calculator(demo_plan))
                st.experimental_rerun()
        else:
            vacation_goal = f"Plan trip from {origin_city} to {destination_city}. {vacation_goal_input}" if origin_city else f"Plan trip to {destination_city}. {vacation_goal_input}"
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
                    tavily_api_key=tavily_api_key,
                    origin_city=origin_city,
                    destination_city=destination_city
                )
                st.session_state.plan_result = plan_data
                # Signal calculator tab to sync its inputs from this fresh plan
                st.session_state["calc_synced_with_plan"] = False
                st.session_state.update(sync_plan_to_calculator(plan_data))
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
                        tavily_api_key=tavily_api_key,
                        origin_city=origin_city,
                        destination_city=destination_city
                    )
                    st.session_state.plan_result = plan_data
                    # Signal calculator tab to sync its inputs from this fresh plan
                    st.session_state["calc_synced_with_plan"] = False
                    st.session_state.update(sync_plan_to_calculator(plan_data))
                except Exception as e:
                    st.error(f"An error occurred while running the planner: {e}")

    # Render results on the right column if they exist in state
    with col2:
        st.markdown(render_section_header("Generated Travel Plan", icon="🗺️", subtitle="Results & recommendations"), unsafe_allow_html=True)
        
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

            # Dedicated Flight Information Section
            if res.get("extracted_budget"):
                eb = res["extracted_budget"]
                airline = eb.get("airline_name")
                route = eb.get("flight_route")
                flight_cost = eb.get("flight_pp")
                flight_src = eb.get("flight_source")
                
                if airline and route and flight_cost is not None:
                    sym = res.get("currency_symbol", "")
                    link_line = f"  \n**Link:** [Book / Compare Flights]({flight_src})" if flight_src else ""
                    st.info(
                        f"✈️ **Recommended Flight**\n\n"
                        f"**Airline:** {airline}  \n"
                        f"**Route:** {route}  \n"
                        f"**Cost:** {sym}{flight_cost:,.2f} per person"
                        f"{link_line}"
                    )

            # Extract Money Saving Tips from plan
            plan_text = res["result"]

            # Escape '$' so Streamlit markdown doesn't treat currency as LaTeX math
            def _escape_dollar_signs(text: str) -> str:
                """Replace bare $ with \\$ to prevent LaTeX interpretation."""
                import re as _re_esc
                # Escape $ that is NOT already escaped (i.e., not preceded by \)
                return _re_esc.sub(r'(?<!\\)\$', r'\\$', text)

            plan_text = _escape_dollar_signs(plan_text)

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
            # Ensure 'Must-Visit Attractions' lists have at least 10 items for better UX
            def _ensure_min_attractions(text: str, min_items: int = 10) -> str:
                import re as _re_ai
                heading_variants = [r'Must-Visit Attractions', r'Must Visit Attractions', r'Must-Visit']
                for hv in heading_variants:
                    m = _re_ai.search(rf"(^|\n)#+\s*{hv}.*(\n|$)", text, _re_ai.IGNORECASE)
                    if not m:
                        # try a looser match without heading markers
                        m2 = _re_ai.search(rf"({hv}):?\s*\n", text, _re_ai.IGNORECASE)
                        if m2:
                            start = m2.end()
                        else:
                            continue
                    else:
                        start = m.end()

                    # Find the end of this section (next '##' or end of text)
                    next_h = _re_ai.search(r"\n##\s", text[start:])
                    end = start + (next_h.start() if next_h else len(text[start:]))
                    section = text[start:end].strip()

                    # Count list items (numbered or bullets)
                    items = _re_ai.findall(r"^\s*(?:\d+\.|-|\*)\s+", section, _re_ai.MULTILINE)
                    count = len(items)
                    if count >= min_items:
                        continue

                    # Suggestions to append when the list is short
                    suggestions = [
                        "Pantheon — Historic temple and architectural marvel",
                        "Trevi Fountain — Iconic Baroque fountain; toss a coin for luck",
                        "Spanish Steps — Famous stairway and lively piazza"
                    ]
                    needed = min_items - count
                    append_items = suggestions[:needed]

                    # Build appended markdown list (continue numbering if numbered list used)
                    # Determine if original used numbered list; if so, find last number
                    numbered = _re_ai.search(r"(\d+)\.(?=[^\n]*$)", section)
                    if numbered:
                        # find max existing numbered index
                        nums = [int(n) for n in _re_ai.findall(r"^(\d+)\.", section, _re_ai.MULTILINE)]
                        start_idx = max(nums) + 1 if nums else 1
                        new_lines = "\n" + "\n".join([f"{start_idx + i}. {s}" for i, s in enumerate(append_items)])
                    else:
                        new_lines = "\n" + "\n".join([f"- {s}" for s in append_items])

                    # Insert the appended items before the section end
                    text = text[:end] + new_lines + text[end:]

                return text

            # Replace plain markdown header '## Top Must-Visit Attractions' with our beautified HTML header
            import re as _re_h
            display_text = _re_h.sub(
                r'(?m)^#{1,6}\s*Top\s*Must-Visit\s*Attractions\s*$',
                render_section_header("Top Must-Visit Attractions", icon="📍", subtitle="Top sights to see"),
                display_text
            )

            display_text = _ensure_min_attractions(display_text, 10)
            st.markdown(display_text, unsafe_allow_html=True)

            # 2. Display dynamically calculated booking timeline (with ASAP corrections)
            booking_timeline_html = generate_booking_timeline(res["enhanced_dates"])
            if booking_timeline_html:
                # Render booking timeline in a green-styled boxed HTML (matches Money Saving Tips style)
                import re as _re_bt
                html_raw = booking_timeline_html.strip()
                # Remove leading markdown heading markers if present
                html_raw = _re_bt.sub(r'(?m)^\s*#{1,6}\s*', '', html_raw)
                # Convert bold markdown to <strong>
                html_body = _re_bt.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_raw)
                html_body = html_body.replace('\r\n', '\n').replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
                timeline_html = (
                    "<div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);"
                    "border: 1px solid #86efac; border-left: 4px solid #16a34a; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px;'>"
                    "<div style='font-family:\'Noto Sans\',sans-serif; font-size:1rem; font-weight:700; color:#15803d; margin-bottom:10px;'>"
                    "📅 Calculated Booking Milestones & Deadlines"
                    "</div>"
                    f"{html_body}"
                    "</div>"
                )
                st.markdown(timeline_html, unsafe_allow_html=True)

            # 3. Web Sources used by Tavily (only shown when Tavily was active)
            sources = res.get("sources", [])
            if sources:
                # Render a simple, always-visible styled block (no expander/dropdown)
                st.markdown(render_section_header(f"Web Sources Used by Agents ({len(sources)})", icon="🌐", subtitle="Pages fetched during plan generation"), unsafe_allow_html=True)

                # Flatten sources and render concise link list for clarity
                sources_html = """
                <div style="background:linear-gradient(135deg,#eef2ff 0%,#e0f2fe 100%);border:1px solid #bfdbfe;border-left:4px solid #3b82f6;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
                  <div style="font-family:'Noto Sans',sans-serif;font-size:0.95rem;color:#0f172a;margin-bottom:8px;">The following pages were fetched live by Tavily during plan generation:</div>
                """

                # Show each source as a single-line link with its URL subtitle (no grouping)
                for s in sources:
                    label = s.get("title") or s.get("url")
                    label_safe = str(label).replace("#", "").replace("*", "").replace("_", "").strip()
                    url = s.get("url", "")
                    sources_html += (
                        f"<div style='margin-bottom:8px;'>"
                        f"<a href=\"{url}\" target=\"_blank\" rel=\"noopener\" style='font-weight:600;color:#0369a1;text-decoration:none;'>{label_safe}</a>"
                        f"<div style='font-size:0.85rem;color:#64748b;margin-top:3px;'>{url}</div>"
                        f"</div>"
                    )

                sources_html += "</div>"
                st.markdown(sources_html, unsafe_allow_html=True)

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
