# 🌴 CrewAI Vacation Planner

AI-powered vacation planner built with **CrewAI**, **OpenAI**, and **Streamlit**.

It creates personalized trip plans, day-by-day itineraries, local-currency budgets, booking timelines, and deep-linked flight searches using robust multi-agent orchestration.

## Features
- **4-Agent CrewAI Workflow**: Operates sequentially and concurrently to research, map schedules, and tally itemized pricing.
- **Dynamic Google Flights Link**: Automatically parses your source city, destination, and selected travel dates to compile exact, live round-trip flight booking routes.
- **Safe Evaluation Sandbox**: Safe AST-evaluation logic (A03 Injection Protected) for error-free math calculations.
- **Strict Budget Consistency**: Programmatic schema assertions to ensure per-person and total day metrics are perfectly matched.

## Agents (crew_planner.py)
- 🔍 **Research Specialist** — Real-time web search for flights, lodging, transport, & attractions.
- 📅 **Itinerary Planner** — Day-by-day schedule logic.
- 💰 **Financial Coordinator** — Itemized budgets in local currencies.
- 📋 **Trip Director** — Consolidates coordinates into the final master plan.

## Agent tools
- 🌐 `TrackedTavilySearchTool` — Live Tavily web search with source URL tracking (optional; needs `TAVILY_API_KEY`).
- 🧮 `CalculatorTool` — General arithmetic evaluator for quick checks (e.g. rate × days).
- 📊 `TripCostCalculatorTool` — Itemized budget calculator: category totals, subtotal, contingency, grand total, and per-person cost.

## Project structure
- [app.py](app.py): Clean, single-page Streamlit application.
- [crew_planner.py](crew_planner.py): Self-contained agent orchestration and custom calculation math engines.
- [utils.py](utils.py): Date validation and dynamic currency symbol parsers.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
3. Run locally:
   ```bash
   streamlit run app.py
   ```

## Optional settings
Set these in `.env` to tune the planner:

| Variable | Default | Description |
|---|---|---|
| `PLANNER_MODEL` | `gpt-4o-mini` | OpenAI model used by the agents |
| `PLANNER_TEMPERATURE` | `0.0` | Model temperature |
| `PLANNER_MAX_ITER` | `15` | Max iterations per agent |
| `PLANNER_MAX_EXEC_TIME` | `300` | Max seconds per agent |

If a run ends with "Agent stopped due to iteration limit or time limit", raise the last two values.

## Caching
Results are cached as JSON in `.vacation_cache/`, keyed by the trip inputs. Delete that folder to force a fresh plan for the same inputs.
