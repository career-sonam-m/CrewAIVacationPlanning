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
from typing import Type
from datetime import datetime
import streamlit as st
from langchain_openai import ChatOpenAI

# CrewAI multi-agent framework
from crewai import Agent, Task, Crew, Process

# CrewAI custom tool support
from crewai.tools import BaseTool
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
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
                include_raw_content=False,
            )

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
    llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)

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

def run_vacation_planner(vacation_goal, dates, family_size, preferences, api_key, tavily_api_key=""):
    """Orchestrates the vacation planning pipeline using CrewAI multi-agent architecture.

    Architecture:
    - Pre-flight: Direct ChatOpenAI calls for input enhancement & currency detection (fast)
    - Main Crew: 4 specialized agents with 4 tasks (including async parallel execution)
    """
    global _active_search_sources
    _active_search_sources = []  # Reset source collector for this run

    os.environ["OPENAI_API_KEY"] = api_key
    model_name = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("PLANNER_TEMPERATURE", "0.0"))

    # Direct LLM for fast pre-flight calls (no Crew overhead)
    llm_direct = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)

    # Initialize Tavily web search tool (optional — agents fall back to LLM knowledge if not provided)
    web_search_tool = create_web_search_tool(tavily_api_key)

    # Streamlit status indicator with optimized timeline updates
    progress_status = st.status("🕵️ Running Optimized AI Planner (estimating 15-25 seconds)...", expanded=True)

    with progress_status:
        # ==================================================================
        # PRE-FLIGHT PHASE — Direct LLM calls for speed
        # ==================================================================
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

        # ==================================================================
        # MAIN CREW PHASE — CrewAI Multi-Agent Architecture
        # ==================================================================
        st.write("🗺️ **Step 2/3**: CrewAI agents researching options (accommodations, transit, attractions)...")

        # --- Define CrewAI Tools & Agents ---

        calculator_tool = CalculatorTool()
        trip_budget_calculator = TripCostCalculatorTool()
        search_tools = [web_search_tool] if web_search_tool else []
        financial_tools = search_tools + [calculator_tool, trip_budget_calculator]

        research_agent = Agent(
            role="Travel & Attractions Research Specialist",
            goal=(
                f"Research comprehensive travel options for: {enhanced_goal}. "
                f"Find flights, accommodations, transport, and 6-7 must-visit attractions "
                f"with pricing in local currency ({detected_currency}). "
                f"Tailor all recommendations for the travel party: {enhanced_family}."
            ),
            backstory=(
                "You are a seasoned travel researcher with deep expertise in global destinations. "
                "You specialize in finding diverse accommodation options, competitive flight deals, "
                "and hidden-gem attractions. You always consider the composition of the travel party "
                "when making recommendations — solo travelers get different suggestions than families with kids."
            ),
            tools=search_tools,
            llm=model_name,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=120,
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
            llm=model_name,
            verbose=False,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=120,
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
            llm=model_name,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=120,
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
            llm=model_name,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=120,
        )

        # --- Define CrewAI Tasks ---

        research_task = Task(
            description=(
                f"Research travel options and top attractions for:\n"
                f"Goal: {enhanced_goal}\n"
                f"Dates: {enhanced_dates}\n"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "DYNAMIC PRICING INSTRUCTIONS:\n"
                "1. Analyze travel dates for seasonality impact on pricing\n"
                "2. Provide CURRENT estimated costs (not generic/historical prices)\n"
                "3. Note if travel dates fall during peak season (affects all prices by 30-80%)\n"
                "4. Compare realistic price ranges for different booking windows\n\n"
                "Provide DIVERSE and SPECIFIC recommendations:\n"
                "- 2 flight options:\n"
                "  * Option 1: Budget airline (lower cost)\n"
                "  * Option 2: Full-service carrier (higher cost)\n"
                "  * Include estimated pricing based on travel dates and seasonality for the whole party\n"
                "- 2-3 accommodation types with SPECIFIC, UNIQUE hotels/properties:\n"
                "  * Avoid generic chain hotels when possible\n"
                "  * Provide options suitable for the composition of the party\n"
                f"  * Estimate per-night rates in LOCAL CURRENCY ({detected_currency}) for dates specified\n"
                "- Local transport options with pricing:\n"
                "  * Rail passes, car rental, ride-sharing and public transit costs\n"
                "- 6-7 must-visit attractions:\n"
                "  * Attraction name, location, visit duration, cost per person in LOCAL CURRENCY\n"
                f"  * IMPORTANT: If {enhanced_family} has NO children/kids, do NOT recommend child-centric attractions\n"
                "  * Booking requirements and money-saving tips\n\n"
                f"Use the LOCAL CURRENCY ({detected_currency}) for all pricing.\n"
                "If any option requires online bookings, flag it with 'ACTION REQUIRED'."
            ),
            expected_output=(
                "A comprehensive research report covering: flight options with prices, "
                "accommodation recommendations with nightly rates, local transport options with costs, "
                f"and 6-7 must-visit attractions with per-person pricing — all in {detected_currency}."
            ),
            agent=research_agent,
        )

        itinerary_task = Task(
            description=(
                f"Build a detailed daily itinerary for the travel party:\n"
                f"Goal: {enhanced_goal}\n"
                f"Dates: {enhanced_dates}\n"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "Use the research findings from the Research Specialist to inform your schedule.\n\n"
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
            context=[research_task],
            async_execution=True,
        )

        current_date_str = datetime.now().strftime("%Y-%m-%d")

        budget_task = Task(
            description=(
                f"Create a REALISTIC and ACCURATE budget estimate for:\n"
                f"Goal: {enhanced_goal}\n"
                f"Dates: {enhanced_dates}\n"
                f"Travel party: {enhanced_family}\n"
                f"Preferences: {preferences}\n\n"
                "Use the research findings from the Research Specialist.\n"
                "CRITICAL: Use your Calculator tool for all math calculations (multiplication of rates, summing categories, contingency percentage).\n\n"
                "FLIGHT COSTS:\n"
                "1. Identify the specific origin and destination from the vacation goal.\n"
                "2. Determine a realistic economy class round-trip fare per person.\n"
                "3. Factor in the travel season: high season routes command higher prices.\n"
                "4. Count the total number of travelers. Children aged 2+ require a full paid seat.\n"
                "5. Calculate using Calculator tool: [per-person fare] × [total travelers] = Total flight cost.\n"
                "6. Show this calculation explicitly.\n"
                "7. Cite source: Tavily search URLs if available, otherwise state 'based on market trends'.\n\n"
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
            context=[research_task],
            async_execution=True,
        )

        coordination_task = Task(
            description=(
                "Assemble the FINAL travel plan from all agent outputs.\n\n"
                f"Travel Dates: {enhanced_dates}\n"
                f"Current Date (Today): {current_date_str}\n\n"
                "CRITICAL INSTRUCTIONS FOR FINAL OUTPUT:\n"
                f"1. ALL currency amounts must use LOCAL CURRENCY ({detected_currency}) with symbol ({currency_symbol})\n"
                "2. Sum all cost categories from the budget breakdown:\n"
                "   - Flights (total for all travelers)\n"
                "   - Accommodation (total for all nights)\n"
                "   - Local transport (total for entire trip)\n"
                "   - Food (total for entire trip)\n"
                "   - Activities and attractions\n"
                "   - Contingency fund (10-15% of subtotal)\n"
                "3. Calculate: FINAL TOTAL = Sum of all above categories\n"
                "4. Display prominently:\n"
                f"   'Final Estimated Budget: [AMOUNT] [CURRENCY CODE] ({currency_symbol})'\n"
                "5. Use your Calculator tool to verify the total matches the exact sum of the breakdown table\n"
                "6. Cite flight cost sources (Tavily URLs or general market trends)\n\n"
                "PRODUCE A FINAL DELIVERABLE CONTAINING:\n"
                f"- Executive summary (1-2 paragraphs). Explicitly state travel party: {enhanced_family}\n"
                f"- Top 6-7 Must-Visit Attractions with descriptions for {enhanced_family}\n"
                f"  * IMPORTANT: If {enhanced_family} has NO kids/children, no child-centric activities\n"
                f"- Complete Budget Breakdown in LOCAL CURRENCY ({detected_currency}) — clearly itemized\n"
                "- Risk notes and contingencies\n"
                "- Final total estimated cost prominently displayed with currency code and symbol\n"
                "- MONEY SAVING TIPS: End with a section titled '## Money Saving Tips' with exactly 5 practical tips:\n"
                "  1) flight booking strategy, 2) accommodation saving, 3) food/dining saving,\n"
                "  4) transport saving, 5) activities/attractions saving\n"
                "  Include estimated saving amount in local currency where possible.\n"
                "- DO NOT write any booking checklist or booking timeline section — that is handled separately.\n\n"
                "Format clearly for sharing with travelers. Ensure all monetary amounts use the correct currency symbol."
            ),
            expected_output=(
                "A polished, complete travel plan document with: executive summary, attractions list, "
                "day-by-day itinerary, itemized budget with grand total, risk notes, and 5 money saving tips. "
                f"All amounts in {detected_currency} with {currency_symbol} symbol."
            ),
            agent=coordination_agent,
            context=[research_task, itinerary_task, budget_task],
        )

        # --- Assemble and Run the CrewAI Crew ---

        st.write("🗓️ **Step 3/3**: CrewAI agents generating itinerary & budget concurrently, then compiling final plan...")

        vacation_crew = Crew(
            agents=[research_agent, itinerary_agent, financial_agent, coordination_agent],
            tasks=[research_task, itinerary_task, budget_task, coordination_task],
            process=Process.sequential,
            cache=True,
            memory=False,
            verbose=False,
        )

        try:
            crew_output = vacation_crew.kickoff()
            result = str(crew_output)
        except Exception as e:
            st.error(f"CrewAI execution failed: {e}")
            result = f"Error during plan generation: {str(e)}"

        progress_status.update(label="✨ Travel plan generated successfully!", state="complete", expanded=False)

    return {
        "enhanced_goal": enhanced_goal,
        "enhanced_dates": enhanced_dates,
        "enhanced_family": enhanced_family,
        "enhanced_preferences": preferences,
        "currency": detected_currency,
        "currency_symbol": currency_symbol,
        "result": result,
        "sources": list(_active_search_sources)  # URLs fetched by Tavily during this run
    }
