"""
CrewAI Planner Orchestrator - CrewAI Vacation Planner

This module defines the CrewAI multi-agent system including:
- 4 specialized agents (Research Specialist, Itinerary Planner, Financial Coordinator, Trip Director)
- 4 workflow tasks (Research, Itinerary [async], Budget [async], Coordination)
- Custom TrackedTavilySearchTool with URL capture
- Pre-flight destination validation (direct LLM call for speed)
- Orchestrator function `run_vacation_planner(...)`
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Type
from datetime import datetime
import streamlit as st
from langchain_openai import ChatOpenAI

# CrewAI multi-agent framework
from crewai import Agent, Task, Crew, Process

# CrewAI custom tool support
try:
    from crewai.tools import BaseTool
except ImportError:
    try:
        from crewai_tools import BaseTool
    except ImportError:
        # langchain_core.tools.BaseTool is what CrewAgentExecutor actually validates tools against.
        from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Helper utilities & calculation tools
from utils import get_currency_symbol, validate_and_adjust_dates
from cost_calculator import CalculatorTool, TripCostCalculatorTool

# Global mutable list that captures URLs fetched by Tavily during a planning run.
# Cleared before each run so sources are always fresh for the current plan.
_active_search_sources = []


# ---------------------------------------------------------------------------
# Custom Tavily Search Tool for CrewAI (with URL tracking)
# ---------------------------------------------------------------------------

class TavilySearchInput(BaseModel):
    """Input schema for the Tavily web search tool."""
    query: str = Field(..., description="The search query to look up on the web.")


class TrackedTavilySearchTool(BaseTool):
    """CrewAI-compatible Tavily search tool that records every URL into _active_search_sources."""
    name: str = "Web Search"
    description: str = (
        "Search the web for current travel information including flight prices, "
        "hotel rates, attraction tickets, transport options, and travel advisories. "
        "Use this tool to find real-time data for vacation planning."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> str:
        """Perform a Tavily search and capture source URLs before returning results."""
        global _active_search_sources
        try:
            from tavily import TavilyClient
            api_key = os.environ.get("TAVILY_API_KEY", "")
            if not api_key:
                return "No Tavily API key available. Using general knowledge only."

            client = TavilyClient(api_key=api_key)
            max_results = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
            # If the query appears to be flight-related, prioritize MakeMyTrip results
            is_flight_query = False
            ql = query.lower()
            if any(x in ql for x in ("flight", "round trip", "round-trip", "fare", "airline", "cheap flight", "flight options")):
                is_flight_query = True

            # Run a site-specific search for MakeMyTrip first when flight queries are detected
            response = None
            if is_flight_query:
                try:
                    site_query = f"{query} site:makemytrip.com"
                    site_resp = client.search(
                        query=site_query,
                        max_results=min(2, max_results),
                        search_depth="basic",
                        include_answer=True,
                        include_raw_content=False,
                    )
                    # Prepend site-specific results into the returned results if available
                    response = site_resp
                except Exception:
                    response = None

            # Fallback to the general search (or supplement it)
            general_resp = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
                include_raw_content=False,
            )
            # Merge site-specific results (if any) with general results, preferring site results
            if response and isinstance(response, dict):
                # Combine results with de-duplication by URL
                site_results = response.get("results", [])
                gen_results = general_resp.get("results", []) if isinstance(general_resp, dict) else []
                merged = []
                seen_urls = set()
                for r in site_results + gen_results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        merged.append(r)
                        seen_urls.add(url)
                # Build a combined response dict similar to Tavily's shape
                response = {"results": merged, "answer": (response.get("answer", "") or general_resp.get("answer", ""))}
            else:
                response = general_resp

            # Track URLs for source citation in the UI
            results = response.get("results", [])
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "") or url
                if url and not any(s["url"] == url for s in _active_search_sources):
                    _active_search_sources.append({
                        "url": url,
                        "title": title,
                        "query": query
                    })

            # Format results for the agent
            output = ""
            answer = response.get("answer", "")
            if answer:
                output += f"Summary: {answer}\n\n"
            for r in results:
                output += f"Source: {r.get('title', '')} ({r.get('url', '')})\n"
                output += f"Content: {r.get('content', '')}\n\n"
            return output if output else "No results found."

        except Exception as e:
            return f"Search failed: {str(e)}. Falling back to general knowledge."


def create_web_search_tool(tavily_api_key):
    """Creates a tracked Tavily web search tool for CrewAI agents if an API key is provided.
    Returns None if no key is available so agents fall back to LLM knowledge."""
    if not tavily_api_key:
        return None
    try:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        return TrackedTavilySearchTool()
    except Exception as e:
        print(f"[Warning] Could not initialize Tavily search tool: {e}")
        return None


# ---------------------------------------------------------------------------
# Pre-flight Destination Validation (fast direct LLM call — no Crew needed)
# ---------------------------------------------------------------------------

def validate_destination_quick(vacation_goal, api_key):
    """Lightweight pre-flight validation using a single direct LLM call.
    Uses ChatOpenAI.invoke() for speed — spinning up a full CrewAI Crew for a
    single yes/no question would add ~5s of unnecessary overhead.

    Returns a tuple: (status, message)
    - status: 'VALID' | 'INVALID_COUNTRY' | 'NEEDS_CITY'
    - message: human-readable explanation to show the user"""
    os.environ["OPENAI_API_KEY"] = api_key
    model_name = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("PLANNER_TEMPERATURE", "0.0"))
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    prompt = (
        "You are an expert geographer validator. Validate the vacation destination in this goal:\n"
        f"Goal: {vacation_goal}\n\n"
        "Return EXACTLY ONE of the following three response formats:\n\n"
        "VALID: [brief confirmation of the destination]\n"
        "INVALID_COUNTRY: [explain why the country or place name appears wrong, misspelled, or does not exist]\n"
        "NEEDS_CITY: [name of state/region only provided] — Please specify which city in [state/region] you would like to visit.\n\n"
        "Rules:\n"
        "- If a specific city, well-known resort area, island, or landmark is mentioned: return VALID\n"
        "- Popular tourist regions without a city (e.g. 'Tuscany', 'Bali', 'Goa', 'Patagonia', 'Scottish Highlands') are VALID\n"
        "- If only a country name is given without any city/area: return NEEDS_CITY\n"
        "- If only a state or province name is given (e.g. 'Rajasthan', 'California', 'Bavaria') without a city: return NEEDS_CITY\n"
        "- If the country name is misspelled, made-up, or clearly invalid: return INVALID_COUNTRY\n"
        "- If no destination can be identified at all in the goal: return NEEDS_CITY"
    )

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()

        if result.startswith("VALID"):
            return ("VALID", result.replace("VALID:", "").strip())
        elif result.startswith("INVALID_COUNTRY"):
            return ("INVALID_COUNTRY", result.replace("INVALID_COUNTRY:", "").strip())
        elif result.startswith("NEEDS_CITY"):
            return ("NEEDS_CITY", result.replace("NEEDS_CITY:", "").strip())
        else:
            return ("VALID", "")
    except Exception:
        return ("VALID", "")


# ---------------------------------------------------------------------------
# CrewAI Multi-Agent Vacation Planner
# ---------------------------------------------------------------------------

def run_vacation_planner(vacation_goal, dates, family_size, preferences, api_key, tavily_api_key="", origin_city="", destination_city=""):
    """Orchestrates the vacation planning pipeline using CrewAI multi-agent architecture.

    Architecture:
    - Pre-flight: Direct ChatOpenAI calls for input enhancement & currency detection (fast)
    - Main Crew: 4 specialized agents with 4 tasks (including async parallel execution)

    Args:
        origin_city: Departure city/airport for flight route pricing.
        destination_city: Target destination city for the vacation.
    """
    # Build explicit flight route string for agent prompts
    flight_route = ""
    if origin_city and destination_city:
        flight_route = f"{origin_city} → {destination_city}"
    elif origin_city:
        flight_route = f"departing from {origin_city}"
    elif destination_city:
        flight_route = f"flying to {destination_city}"
    global _active_search_sources
    _active_search_sources = []  # Reset source collector for this run

    # -----------------------------
    # Simple on-disk cache to speed up repeated runs
    # Cache key is derived from user-visible inputs (not API keys)
    cache_dir = Path.cwd() / ".vacation_cache"
    try:
        cache_dir.mkdir(exist_ok=True)
    except Exception:
        cache_dir = None

    cache_key = None
    cache_file = None
    if cache_dir:
        m = hashlib.sha256()
        m.update(str(vacation_goal).encode("utf-8"))
        m.update(str(dates).encode("utf-8"))
        m.update(str(family_size).encode("utf-8"))
        m.update(str(preferences).encode("utf-8"))
        m.update(str(origin_city).encode("utf-8"))
        m.update(str(destination_city).encode("utf-8"))
        # include whether Tavily is available as it affects results
        m.update(b"tavily" if tavily_api_key else b"notavily")
        cache_key = m.hexdigest()
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                # Return cached result immediately to save time
                return cached
            except Exception:
                pass

    os.environ["OPENAI_API_KEY"] = api_key
    model_name = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("PLANNER_TEMPERATURE", "0.0"))
    # Tunable agent execution parameters to trade speed vs quality
    # NOTE: now that tools are correctly recognized by crewai (see BaseTool import fix),
    # agents actually invoke them (searches, calculations) and need more than 1-2 steps to finish.
    default_max_iter = int(os.getenv("PLANNER_MAX_ITER", "8"))
    default_max_exec_time = int(os.getenv("PLANNER_MAX_EXEC_TIME", "120"))

    # Direct LLM for fast pre-flight calls (no Crew overhead)
    llm_direct = ChatOpenAI(model=model_name, temperature=temperature)
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Initialize Tavily web search tool (optional — agents fall back to LLM knowledge if not provided)
    web_search_tool = create_web_search_tool(tavily_api_key)

    # Streamlit status indicator: use a simple placeholder container (no chevron/expander)
    progress_placeholder = st.empty()
    progress_container = progress_placeholder.container()

    with progress_container:
        # ==================================================================
        # PRE-FLIGHT PHASE — Direct LLM calls for speed
        # ==================================================================
        st.write("🕵️ Running Optimized AI Planner (estimating 15-25 seconds)...")
        st.write("🔮 **Step 1/3**: Enhancing inputs & detecting destination local currency...")

        # --- Combined Single Pre-flight Call (Input Enhancement & Currency Detection) ---
        enhanced_goal = vacation_goal
        detected_currency = "Unknown (Unknown)"
        try:
            prompt_preflight = (
                "You are a Travel Pre-flight Specialist. Process inputs and detect destination currency.\n"
                f"Goal: {vacation_goal}\n"
                f"Dates: {dates}\n"
                f"Travelers: {family_size}\n"
                f"Preferences: {preferences}\n\n"
                "Return EXACTLY in this format:\n"
                "ENHANCED_GOAL: [enhanced concise goal description]\n"
                "CURRENCY: [Currency Code] ([Symbol])\n\n"
                "Examples for Currency:\n"
                "- Tokyo, Japan -> CURRENCY: JPY (¥)\n"
                "- Paris, France -> CURRENCY: EUR (€)\n"
                "- New York / Disney USA -> CURRENCY: USD ($)\n"
                "- India -> CURRENCY: INR (₹)"
            )
            response = llm_direct.invoke(prompt_preflight)
            res_text = response.content.strip()

            for line in res_text.split('\n'):
                if line.startswith("ENHANCED_GOAL:"):
                    enhanced_goal = line.replace("ENHANCED_GOAL:", "").strip()
                elif line.startswith("CURRENCY:"):
                    detected_currency = line.replace("CURRENCY:", "").strip()
        except Exception as e:
            st.error(f"Pre-flight setup failed: {e}")

        # Keep traveler composition and dates strictly preserved from user inputs
        enhanced_family = family_size
        enhanced_dates = validate_and_adjust_dates(dates)
        currency_symbol = get_currency_symbol(detected_currency)

        # Compute ONE authoritative trip-day count from the actual date range so every agent
        # (research/itinerary/financial/coordination) uses the same number instead of each
        # independently inferring it — this is what previously caused the Financial Coordinator to
        # use a different day count (e.g. from wording like "7-day trip") than the exported JSON.
        trip_num_days = 7
        try:
            if " to " in enhanced_dates:
                start_str, end_str = enhanced_dates.split(" to ")
                start_dt = datetime.strptime(start_str.strip(), "%Y-%m-%d")
                end_dt = datetime.strptime(end_str.strip(), "%Y-%m-%d")
                trip_num_days = max(1, (end_dt - start_dt).days)
        except Exception:
            pass
        trip_duration_line = (
            f"TRIP DURATION: exactly {trip_num_days} days. Use this EXACT number for every "
            "day-based calculation (hotel nights, food days, itinerary days). Do NOT recount or "
            "re-infer this from the dates or from any wording elsewhere in the goal text.\n"
        )

        # ==================================================================
        # MAIN CREW PHASE — CrewAI Multi-Agent Architecture
        # ==================================================================
        st.write("🗺️ **Step 2/3**: CrewAI agents researching options (accommodations, transit, attractions)...")

        # --- Define CrewAI Tools & Agents ---

        calculator_tool = CalculatorTool()
        trip_budget_calculator = TripCostCalculatorTool()
        search_tools = [web_search_tool] if web_search_tool else []
        financial_tools = search_tools + [calculator_tool, trip_budget_calculator]

        flight_transport_agent = Agent(
            role="Flight & Transport Research Specialist",
            goal=(
                f"Research flights and local transport for: {enhanced_goal}. "
                f"FLIGHT ROUTE: {flight_route} (round-trip). "
                f"Find flights with specific airline names and local transport options "
                f"with pricing in local currency ({detected_currency}). "
                f"Tailor all recommendations for the travel party: {enhanced_family}."
            ),
            backstory=(
                "You are a seasoned travel researcher specializing in flights and ground transport. "
                "You always research flights for the EXACT route specified, "
                "naming specific airlines that operate on that route. "
                "You always consider the composition of the travel party "
                "when making recommendations — solo travelers get different suggestions than families with kids."
            ),
            tools=search_tools,
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=default_max_iter,
            max_execution_time=default_max_exec_time,
        )

        stay_attractions_agent = Agent(
            role="Accommodation & Attractions Research Specialist",
            goal=(
                f"Research accommodations and top attractions for: {enhanced_goal}. "
                f"Find accommodations and 10 must-visit attractions "
                f"with pricing in local currency ({detected_currency}). "
                f"Tailor all recommendations for the travel party: {enhanced_family}."
            ),
            backstory=(
                "You are a seasoned travel researcher with deep expertise in global destinations. "
                "You specialize in finding diverse accommodation options and hidden-gem attractions. "
                "You always consider the composition of the travel party "
                "when making recommendations — solo travelers get different suggestions than families with kids."
            ),
            tools=search_tools,
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=default_max_iter,
            max_execution_time=default_max_exec_time,
        )

        itinerary_agent = Agent(
            role="Itinerary & Logistics Planner",
            goal=(
                f"Build a detailed day-by-day itinerary for: {enhanced_goal}. "
                f"Dates: {enhanced_dates}. Travel party: {enhanced_family}. "
                "Include morning/afternoon/evening activities, travel times, and rest buffers."
            ),
            backstory=(
                "You are an expert travel logistics planner who creates realistic, well-paced daily schedules. "
                "You understand that families with children need slower pacing and rest breaks, while adult-only "
                "travelers can handle more packed itineraries. You always factor in travel time between locations."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=default_max_iter,
            max_execution_time=default_max_exec_time,
        )

        financial_agent = Agent(
            role="Financial Coordinator & Budget Analyst",
            goal=(
                f"Create a realistic, itemized budget for: {enhanced_goal}. "
                f"All amounts must be in local currency ({detected_currency}). "
                f"Calculate costs for the full travel party: {enhanced_family}."
            ),
            backstory=(
                "You are a meticulous financial analyst specializing in travel budgets. "
                "You calculate costs based on real-world pricing, factoring in seasonality, "
                "party size (including whether children need separate tickets/seats), and destination cost of living. "
                "Use the Calculator and Trip Budget Calculator tools to perform accurate sum totals and itemized breakdowns."
            ),
            tools=financial_tools,
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=default_max_iter,
            max_execution_time=default_max_exec_time,
        )

        coordination_agent = Agent(
            role="Trip Coordination Director",
            goal=(
                f"Compile all research, itinerary, and budget data into a polished final travel plan. "
                f"Ensure all amounts use local currency ({detected_currency}) with symbol ({currency_symbol}). "
                f"The plan must be tailored for: {enhanced_family}."
            ),
            backstory=(
                "You are a senior travel coordinator who synthesizes complex travel data into clear, "
                "actionable travel plans. You verify budget totals using the Calculator and Trip Budget Calculator tools, ensure consistency across sections, "
                "and format the final deliverable for easy sharing with travelers."
            ),
            tools=[calculator_tool, trip_budget_calculator],
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=default_max_iter,
            max_execution_time=default_max_exec_time,
        )

        # --- Define CrewAI Tasks ---
        # Flights/transport and accommodation/attractions are independent — running them as two
        # separate async subagent tasks lets them execute concurrently instead of one long serial task.

        flight_transport_task = Task(
            description=(
                f"Research flights and local transport for:\n"
                f"Goal: {enhanced_goal}\n"
                f"FLIGHT ROUTE: {flight_route} (round-trip)\n"
                f"Dates: {enhanced_dates}\n"
                f"{trip_duration_line}"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "DYNAMIC PRICING INSTRUCTIONS:\n"
                "1. Analyze travel dates for seasonality impact on pricing\n"
                "2. Provide CURRENT estimated costs (not generic/historical prices)\n"
                "3. Note if travel dates fall during peak season (affects all prices by 30-80%)\n"
                "4. Compare realistic price ranges for different booking windows\n\n"
                "Provide DIVERSE and SPECIFIC recommendations:\n"
                f"- 2 flight options for the EXACT route {flight_route} (round-trip):\n"
                "  * Option 1: Budget airline — NAME THE SPECIFIC AIRLINE (e.g. IndiGo, Ryanair, AirAsia)\n"
                "  * Option 2: Full-service carrier — NAME THE SPECIFIC AIRLINE (e.g. Air India, Lufthansa, Emirates)\n"
                "  * For EACH option: state airline name, whether direct or connecting, flight duration, "
                "and estimated round-trip fare per person based on travel dates and seasonality\n"
                "  * Calculate total flight cost for the whole party\n"
                "- Local transport options with pricing:\n"
                "  * Rail passes, car rental, ride-sharing and public transit costs\n\n"
                f"Use the LOCAL CURRENCY ({detected_currency}) for all pricing.\n"
                "If any option requires online bookings, flag it with 'ACTION REQUIRED'."
            ),
            expected_output=(
                "A research report covering: 2 flight options with prices and airline names, "
                f"and local transport options with costs — all in {detected_currency}."
            ),
            agent=flight_transport_agent,
            async_execution=True,
        )

        stay_attractions_task = Task(
            description=(
                f"Research accommodations and top attractions for:\n"
                f"Goal: {enhanced_goal}\n"
                f"Dates: {enhanced_dates}\n"
                f"{trip_duration_line}"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "Provide DIVERSE and SPECIFIC recommendations:\n"
                "- 2-3 accommodation types with SPECIFIC, UNIQUE hotels/properties:\n"
                "  * Avoid generic chain hotels when possible\n"
                "  * Provide options suitable for the composition of the party\n"
                f"  * Estimate per-night rates in LOCAL CURRENCY ({detected_currency}) for dates specified\n"
                "- 10 must-visit attractions:\n"
                "  * Attraction name, location, visit duration, cost per person in LOCAL CURRENCY\n"
                f"  * IMPORTANT: If {enhanced_family} has NO children/kids, do NOT recommend child-centric attractions\n"
                "  * Booking requirements and money-saving tips\n\n"
                f"Use the LOCAL CURRENCY ({detected_currency}) for all pricing.\n"
                "If any option requires online bookings, flag it with 'ACTION REQUIRED'."
            ),
            expected_output=(
                "A research report covering: accommodation recommendations with nightly rates "
                f"and 10 must-visit attractions with per-person pricing — all in {detected_currency}."
            ),
            agent=stay_attractions_agent,
            async_execution=True,
        )

        itinerary_task = Task(
            description=(
                f"Build a detailed daily itinerary for the travel party:\n"
                f"Goal: {enhanced_goal}\n"
                f"Dates: {enhanced_dates}\n"
                f"{trip_duration_line}"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "Use the research findings from the Research Specialists to inform your schedule.\n\n"
                "For each day list:\n"
                "- Morning / Afternoon / Evening activities\n"
                "- Estimated travel time between stops\n"
                "- Notes tailored to the traveler composition\n"
                "- Time buffers for rest or self-exploration\n\n"
                "Return a balanced, realistic daily schedule."
            ),
            expected_output=(
                "A day-by-day itinerary covering each travel day with morning/afternoon/evening "
                "activities, travel times between stops, and practical notes for the travel party."
            ),
            agent=itinerary_agent,
            context=[flight_transport_task, stay_attractions_task],
            async_execution=True,
        )

        current_date_str = datetime.now().strftime("%Y-%m-%d")

        budget_task = Task(
            description=(
                f"Create a REALISTIC and ACCURATE budget estimate for:\n"
                f"Goal: {enhanced_goal}\n"
                f"FLIGHT ROUTE: {flight_route} (round-trip)\n"
                f"Dates: {enhanced_dates}\n"
                f"{trip_duration_line}"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "Use the research findings from the Research Specialists.\n"
                "CRITICAL: Use your Calculator tool for all math calculations (multiplication of rates, summing categories, contingency percentage).\n\n"
                "FLIGHT COSTS:\n"
                f"1. The flight route is: {flight_route} (round-trip). Use this EXACT route for pricing.\n"
                "2. Name the SPECIFIC AIRLINE(S) for each option (e.g. Air India, Emirates, IndiGo).\n"
                "3. Determine a realistic economy class round-trip fare per person for this specific route.\n"
                "4. Factor in the travel season: high season routes command higher prices.\n"
                "5. Count the total number of travelers. Children aged 2+ require a full paid seat.\n"
                "6. Calculate using Calculator tool: [per-person fare] × [total travelers] = Total flight cost.\n"
                "7. Show this calculation explicitly with airline name and route.\n"
                "8. Cite source: Tavily search URLs if available, otherwise state 'based on market trends'.\n\n"
                "ACCOMMODATION:\n"
                "- Based on party size, determine number of rooms required.\n"
                "- Calculate using Calculator tool: [nightly rate] × [rooms] × [nights] = Total accommodation cost.\n\n"
                "FOOD & DINING:\n"
                "- Estimate daily food spend per person based on destination cost of living.\n"
                "- Calculate using Calculator tool: [daily food cost] × [people] × [days] = Total food cost.\n\n"
                "ACTIVITIES & LOCAL TRANSPORT:\n"
                "- Use attraction ticket prices from research for the specific party composition.\n"
                "- Estimate local transport costs for the trip duration.\n\n"
                "GRAND TOTAL:\n"
                "- Use Calculator tool to sum ALL categories + 12% contingency.\n"
                "- Show minimum and maximum range.\n"
                "- Display prominently.\n\n"
                f"Use the LOCAL CURRENCY ({detected_currency}) for all amounts.\n"
                "Format all amounts with the correct currency symbol.\n"
                "Provide two practical cost-saving tips specific to this route and party."
            ),
            expected_output=(
                f"An itemized budget breakdown in {detected_currency} covering flights, accommodation, "
                "food, activities, and transport with explicit calculations, subtotal, 12% contingency, "
                "and a prominently displayed grand total with min/max range."
            ),
            agent=financial_agent,
            context=[flight_transport_task, stay_attractions_task],
            async_execution=True,
        )

        coordination_task = Task(
            description=(
                "Assemble the FINAL travel plan from all agent outputs.\n\n"
                f"Travel Dates: {enhanced_dates}\n"
                f"{trip_duration_line}"
                f"Current Date (Today): {current_date_str}\n\n"
                "CRITICAL INSTRUCTIONS FOR FINAL OUTPUT:\n"
                f"1. ALL currency amounts must use LOCAL CURRENCY ({detected_currency}) with symbol ({currency_symbol})\n"
                "2. Sum all cost categories from the budget breakdown:\n"
                "   - Flights (total for all travelers)\n"
                "   - Accommodation (total for all nights)\n"
                "   - Local transport (total for entire trip)\n"
                "   - Food (total for entire trip)\n"
                "   - Activities and attractions\n"
                "   - Contingency fund (EXACTLY 12% of subtotal — must match the Financial Coordinator's figure)\n"
                "3. Calculate: FINAL TOTAL = Sum of all above categories\n"
                "4. Display prominently:\n"
                f"   'Final Estimated Budget: [AMOUNT] [CURRENCY CODE] ({currency_symbol})'\n"
                "5. Use your Calculator tool to verify the total matches the exact sum of the breakdown table\n"
                "6. Cite flight cost sources (Tavily URLs or general market trends)\n\n"
                "PRODUCE A FINAL DELIVERABLE CONTAINING:\n"
                f"- Executive summary (1-2 paragraphs). Explicitly state travel party: {enhanced_family}\n"
                f"- Top 10 Must-Visit Attractions with descriptions for {enhanced_family}\n"
                f"  * IMPORTANT: If {enhanced_family} has NO kids/children, no child-centric activities\n"
                f"- Complete Budget Breakdown in LOCAL CURRENCY ({detected_currency}) — clearly itemized. FOR FLIGHTS: explicitly state the flight route ({flight_route}) and the specific airline name(s) recommended.\n"
                "- Risk notes and contingencies\n"
                "- Final total estimated cost prominently displayed with currency code and symbol\n"
                "- MONEY SAVING TIPS: End with a section titled '## Money Saving Tips' with exactly 5 practical tips:\n"
                "  1) flight booking strategy, 2) accommodation saving, 3) food/dining saving,\n"
                "  4) transport saving, 5) activities/attractions saving\n"
                "  Include estimated saving amount in local currency where possible.\n"
                "- DO NOT write any booking checklist or booking timeline section — that is handled separately.\n"
                "- RAW DATA EXPORT: At the very end of your response, output a strict JSON block enclosed in ```json ... ``` tags containing exactly these keys in local currency (costs as floats, text as strings, counts as integers):\n"
                '  {"airline_name": "<Specific recommended airline>", "flight_route": "<origin to destination>", "flight_pp": <round-trip flight cost PER PERSON (do NOT put total cost)>, "hotel_pn": <total accommodation cost DIVIDED BY num_days below, so hotel_pn * num_days equals the accommodation total you displayed>, "food_ppd": <food cost per person per day>, "activities": <total activities cost>, "transport": <total local transport cost>, "num_adults": <total adults>, "num_children": <total children>, "num_days": '
                f'<MUST be exactly {trip_num_days}, the trip duration given above — do not use any other number>, "contingency_pct": <the exact contingency percentage you used, e.g. 12.0>}}\n\n'
                "Format clearly for sharing with travelers. Ensure all monetary amounts use the correct currency symbol."
            ),
            expected_output=(
                "A polished, complete travel plan document with: executive summary, attractions list, "
                "day-by-day itinerary, itemized budget with grand total, risk notes, and 5 money saving tips. "
                f"All amounts in {detected_currency} with {currency_symbol} symbol."
            ),
            agent=coordination_agent,
            context=[flight_transport_task, stay_attractions_task, itinerary_task, budget_task],
        )

        # --- Assemble and Run the CrewAI Crew ---

        st.write("🗓️ **Step 3/3**: CrewAI agents researching, generating itinerary & budget concurrently, then compiling final plan...")

        vacation_crew = Crew(
            agents=[flight_transport_agent, stay_attractions_agent, itinerary_agent, financial_agent, coordination_agent],
            tasks=[flight_transport_task, stay_attractions_task, itinerary_task, budget_task, coordination_task],
            process=Process.sequential,
            llm=llm,
            cache=True,
            memory=False,
            verbose=False,
        )

        extracted_budget = None
        try:
            crew_output = vacation_crew.kickoff()
            result = str(crew_output)
            
            # Extract JSON block from output
            import re, json
            match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
            if match:
                try:
                    extracted_budget = json.loads(match.group(1))
                    # Remove the JSON block from the final result string
                    result = result.replace(match.group(0), "").strip()
                except Exception:
                    pass
            # If we successfully parsed an extracted_budget, try to annotate it with a source URL for key prices
            if extracted_budget and isinstance(extracted_budget, dict):
                # Prefer any tracked Tavily source that looks like a flight booking site (makemytrip, cleartrip, skyscanner, kayak, momondo)
                preferred_hosts = ["makemytrip.com", "cleartrip.com", "skyscanner.net", "skyscanner.com", "kayak.com", "momondo.com", "google.com"]
                flight_src = None
                for src in _active_search_sources:
                    u = src.get("url", "") or ""
                    for h in preferred_hosts:
                        if h in u:
                            flight_src = u
                            break
                    if flight_src:
                        break
                # Fallback: if no preferred host, pick first source that mentions flights
                if not flight_src:
                    for src in _active_search_sources:
                        q = (src.get("query") or "").lower()
                        if "flight" in q or "fare" in q or "flight" in (src.get("title") or "").lower():
                            flight_src = src.get("url")
                            break
                if flight_src:
                    try:
                        extracted_budget["flight_source"] = flight_src
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"CrewAI execution failed: {e}")
            result = f"Error during plan generation: {str(e)}"

        # Replace status with a success message in the placeholder
        try:
            progress_placeholder.success("✨ Travel plan generated successfully!")
        except Exception:
            # Fallback: write a success line in the container
            progress_container.write("✨ Travel plan generated successfully!")

    result_obj = {
        "enhanced_goal": enhanced_goal,
        "enhanced_dates": enhanced_dates,
        "enhanced_family": enhanced_family,
        "enhanced_preferences": preferences,
        "currency": detected_currency,
        "currency_symbol": currency_symbol,
        "result": result,
        "sources": list(_active_search_sources),  # URLs fetched by Tavily during this run
        "extracted_budget": extracted_budget
    }

    # Save to cache (best-effort)
    if cache_file is not None:
        try:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(result_obj, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return result_obj
