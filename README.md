# 🌴 CrewAI Vacation Planner

AI-powered vacation planner built with **CrewAI**, **OpenAI**, and **Streamlit**.

It creates personalized trip plans, day-by-day itineraries, local-currency budgets, booking timelines, and deep-linked flight searches using robust multi-agent orchestration.

## Features
- **4-Agent CrewAI Workflow**: Operates sequentially and concurrently to research, map schedules, and tally itemized pricing.
- **Dynamic Google Flights Link**: Automatically parses your source city, destination, and selected travel dates to compile exact, live round-trip flight booking routes.
- **Safe Evaluation Sandbox**: Safe AST-evaluation logic (A03 Injection Protected) for error-free math calculations.
- **Strict Budget Consistency**: Programmatic schema assertions to ensure per-person and total day metrics are perfectly matched.
- **Hugging Face ready**: Configured with Docker setup to run on huggingface spaces cleanly.

## Agents (crew_planner.py)
- 🔍 **Research Specialist** — Real-time web search for flights, lodging, transport, & attractions.
- 📅 **Itinerary Planner** — Day-by-day schedule logic.
- 💰 **Financial Coordinator** — Itemized budgets in local currencies.
- 📋 **Trip Director** — Consolidates coordinates into the final master plan.

## Project structure
- [app.py](app.py): Clean, single-page Streamlit application.
- [crew_planner.py](crew_planner.py): Self-contained agent orchestration and custom calculation math engines.
- [utils.py](utils.py): Date validation and dynamic currency symbol parsers.
- [Dockerfile](Dockerfile): Production container configuration compiled for port 7860.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   # Optional CrewAI agent limits (defaults: 15 iterations / 300 seconds)
   PLANNER_MAX_ITER=15
   PLANNER_MAX_EXEC_TIME=300
   ```
3. Run locally:
   ```bash
   streamlit run app.py
   ```
   *For Docker deployments (Hugging Face Spaces), the container exposes port 7860.*
