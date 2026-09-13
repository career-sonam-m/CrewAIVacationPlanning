# 🌴 CrewAI Vacation Planner

AI-powered vacation planner built with **CrewAI**, **OpenAI**, and **Streamlit**.

It creates personalized trip plans, day-by-day itineraries, local-currency budgets, and booking timelines using a multi-agent workflow.

## Features
- Multi-agent research, itinerary, and budgeting flow
- Destination validation and date handling
- Dynamic currency detection
- Live web search citations
- Synced planner + cost calculator tabs
- CrewAI compatibility fixes for newer versions

## Project structure
- [app.py](app.py): Streamlit app and main UI
- [crew_planner.py](crew_planner.py): agent orchestration and planning logic
- [cost_calculator.py](cost_calculator.py): budget math and calculator tab
- [utils.py](utils.py): shared helper functions

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
3. Run:
   ```bash
   streamlit run app.py
   ```

## Notes
- The app has two tabs: **Vacation Planner** and **Cost Calculator**.
- The calculator now syncs with the main planner values so it stays aligned with the current trip inputs.
