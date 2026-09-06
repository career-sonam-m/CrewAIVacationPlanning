"""
Cost Calculator UI & CrewAI Tool Component - CrewAI Vacation Planner

This module implements:
1. calculate_trip_budget: Pure Python math engine for trip cost calculations.
2. TripCostCalculatorTool: CrewAI BaseTool wrapper for multi-agent execution.
3. render_cost_calculator: Streamlit UI tab for interactive cost calculations.
"""

from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import streamlit as st


# ---------------------------------------------------------------------------
# Core Calculation Engine
# ---------------------------------------------------------------------------

def calculate_trip_budget(
    num_adults: int,
    num_children: int,
    num_days: int,
    flight_pp: float,
    hotel_pn: float,
    food_ppd: float,
    activities: float = 0.0,
    transport: float = 0.0,
    misc: float = 0.0,
    contingency_pct: float = 12.0,
    currency_symbol: str = "₹"
) -> dict:
    """Core mathematical engine for calculating itemized trip budgets."""
    total_travelers = max(1, num_adults + num_children)
    flights_total = flight_pp * total_travelers
    hotel_total = hotel_pn * num_days
    food_total = food_ppd * total_travelers * num_days
    subtotal = flights_total + hotel_total + food_total + activities + transport + misc
    contingency_amt = subtotal * (contingency_pct / 100.0)
    grand_total = subtotal + contingency_amt
    per_person = grand_total / total_travelers if total_travelers > 0 else 0.0

    return {
        "total_travelers": total_travelers,
        "num_adults": num_adults,
        "num_children": num_children,
        "num_days": num_days,
        "flights_total": flights_total,
        "hotel_total": hotel_total,
        "food_total": food_total,
        "activities": activities,
        "transport": transport,
        "misc": misc,
        "subtotal": subtotal,
        "contingency_amt": contingency_amt,
        "contingency_pct": contingency_pct,
        "grand_total": grand_total,
        "per_person": per_person,
        "currency_symbol": currency_symbol
    }


# ---------------------------------------------------------------------------
# CrewAI BaseTool Wrapper
# ---------------------------------------------------------------------------

class TripCostInput(BaseModel):
    """Input parameters for calculating itemized trip budget."""
    num_days: int = Field(..., description="Duration of the trip in days")
    flight_pp: float = Field(..., description="Round-trip flight fare per person in local currency")
    hotel_pn: float = Field(..., description="Total room cost per night for all rooms in local currency")
    food_ppd: float = Field(..., description="Average daily food spend per person in local currency")
    num_adults: int = Field(default=2, description="Number of adult travelers")
    num_children: int = Field(default=0, description="Number of child travelers")
    activities: float = Field(default=0.0, description="Total cost for all tickets, tours, and activities")
    transport: float = Field(default=0.0, description="Total local transport cost (taxi, train, transit) for entire trip")
    misc: float = Field(default=0.0, description="Optional extra or shopping budget")
    contingency_pct: float = Field(default=12.0, description="Contingency buffer percentage (5-20%)")
    currency_symbol: str = Field(default="₹", description="Currency symbol (e.g. $, €, ¥, ₹)")


class TripCostCalculatorTool(BaseTool):
    """CrewAI-compatible tool for calculating precise itemized trip budgets using cost_calculator math engine."""
    name: str = "Trip Budget Calculator"
    description: str = (
        "Calculates exact itemized travel budgets including flights, lodging, dining, "
        "activities, local transit, subtotal, contingency, grand total, and per-person cost."
    )
    args_schema: Type[BaseModel] = TripCostInput

    def _run(
        self,
        num_days: int,
        flight_pp: float,
        hotel_pn: float,
        food_ppd: float,
        num_adults: int = 2,
        num_children: int = 0,
        activities: float = 0.0,
        transport: float = 0.0,
        misc: float = 0.0,
        contingency_pct: float = 12.0,
        currency_symbol: str = "₹"
    ) -> str:
        res = calculate_trip_budget(
            num_adults=num_adults,
            num_children=num_children,
            num_days=num_days,
            flight_pp=flight_pp,
            hotel_pn=hotel_pn,
            food_ppd=food_ppd,
            activities=activities,
            transport=transport,
            misc=misc,
            contingency_pct=contingency_pct,
            currency_symbol=currency_symbol
        )
        sym = res["currency_symbol"]
        return (
            f"=== Trip Budget Calculation Result ===\n"
            f"Travelers: {res['total_travelers']} ({res['num_adults']} adults, {res['num_children']} children)\n"
            f"Duration: {res['num_days']} days\n"
            f"Flights: {sym}{res['flights_total']:,.2f} ({sym}{flight_pp:,.2f} x {res['total_travelers']} travelers)\n"
            f"Accommodation: {sym}{res['hotel_total']:,.2f} ({sym}{hotel_pn:,.2f} x {res['num_days']} nights)\n"
            f"Food & Dining: {sym}{res['food_total']:,.2f} ({sym}{food_ppd:,.2f} x {res['total_travelers']}p x {res['num_days']}d)\n"
            f"Activities: {sym}{res['activities']:,.2f}\n"
            f"Transport: {sym}{res['transport']:,.2f}\n"
            f"Miscellaneous: {sym}{res['misc']:,.2f}\n"
            f"Subtotal: {sym}{res['subtotal']:,.2f}\n"
            f"Contingency ({res['contingency_pct']}%): {sym}{res['contingency_amt']:,.2f}\n"
            f"GRAND TOTAL: {sym}{res['grand_total']:,.2f}\n"
            f"Per Person: {sym}{res['per_person']:,.2f}"
        )


class CalculatorInput(BaseModel):
    """Input schema for the general arithmetic Calculator tool."""
    expression: str = Field(..., description="Mathematical expression string to calculate (e.g. '150 * 5 + 400 + 120').")


class CalculatorTool(BaseTool):
    """CrewAI-compatible tool for general mathematical arithmetic."""
    name: str = "Calculator"
    description: str = (
        "Useful for calculating travel budgets, multiplying daily rates by number of days or travelers, "
        "summing category totals, and computing contingency percentages. "
        "Input must be a mathematical expression like '150 * 5 + 400' or '2500 * 0.15'."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Safely evaluate mathematical expression."""
        try:
            clean_expr = expression.replace(",", "").replace("$", "").replace("¥", "").replace("€", "").replace("₹", "").strip()
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(clean_expr, {"__builtins__": None}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error for '{expression}': {str(e)}"


# ---------------------------------------------------------------------------
# Streamlit Interactive Tab Component
# ---------------------------------------------------------------------------

def render_cost_calculator():
    """Standalone interactive trip cost calculator — no AI needed, pure Python math."""
    st.markdown("### 🧭 Trip Cost Calculator")
    st.markdown("*Enter your estimated costs below for an instant itemized breakdown and grand total.*")
    st.markdown("")

    inp_col, res_col = st.columns([1, 1.4], gap="large")

    with inp_col:
        st.markdown("#### 👥 Travelers & Duration")
        num_adults   = st.number_input("Adults",   min_value=1, max_value=20, value=2, key="calc_adults")
        num_children = st.number_input("Children", min_value=0, max_value=20, value=0, key="calc_children")
        num_days     = st.number_input("Trip Duration (days)", min_value=1, max_value=180, value=7, key="calc_days")
        currency_sym = st.text_input("Currency Symbol", value="₹", placeholder="₹, €, £, ¥, $", key="calc_currency",
                                     help="Enter the symbol for your destination currency")

        # Track currency switch to update default values in session state
        if "calc_prev_currency" not in st.session_state:
            st.session_state.calc_prev_currency = currency_sym

        # Determine dynamic defaults based on currency symbol
        is_high_value_currency = currency_sym in ["₹", "¥", "Unknown", ""] or not any(c in currency_sym for c in ["$", "€", "£"])
        
        default_flight = 85000.0 if is_high_value_currency else 1000.0
        default_hotel  = 8000.0  if is_high_value_currency else 150.0
        default_food   = 2500.0  if is_high_value_currency else 50.0
        default_activities = 15000.0 if is_high_value_currency else 200.0
        default_transport  = 6000.0  if is_high_value_currency else 80.0

        if st.session_state.calc_prev_currency != currency_sym:
            st.session_state.calc_prev_currency = currency_sym
            st.session_state.calc_flight = default_flight
            st.session_state.calc_hotel = default_hotel
            st.session_state.calc_food = default_food
            st.session_state.calc_activities = default_activities
            st.session_state.calc_transport = default_transport
            st.session_state.calc_misc = 0.0
            st.rerun()

        st.markdown("#### 💰 Cost Inputs")
        flight_pp    = st.number_input(f"Flight — round-trip per person ({currency_sym})",
                                       min_value=0.0, value=default_flight, step=10.0 if not is_high_value_currency else 1000.0, key="calc_flight",
                                       help="Economy round-trip cost per traveler")
        hotel_pn     = st.number_input(f"Hotel — per night, all rooms ({currency_sym})",
                                       min_value=0.0, value=default_hotel, step=10.0 if not is_high_value_currency else 500.0, key="calc_hotel",
                                       help="Total room cost per night for the whole party")
        food_ppd     = st.number_input(f"Food — per person per day ({currency_sym})",
                                       min_value=0.0, value=default_food, step=5.0 if not is_high_value_currency else 100.0, key="calc_food",
                                       help="Average daily food spend per traveler")
        activities   = st.number_input(f"Activities & Attractions — total ({currency_sym})",
                                       min_value=0.0, value=default_activities, step=10.0 if not is_high_value_currency else 500.0, key="calc_activities",
                                       help="Total tickets, tours, and experiences for all travelers")
        transport    = st.number_input(f"Local Transport — total ({currency_sym})",
                                       min_value=0.0, value=default_transport, step=10.0 if not is_high_value_currency else 500.0, key="calc_transport",
                                       help="Taxis, metro passes, trains for the whole trip")
        misc         = st.number_input(f"Miscellaneous / Shopping ({currency_sym})",
                                       min_value=0.0, value=0.0, step=10.0 if not is_high_value_currency else 500.0, key="calc_misc",
                                       help="Optional extra spending budget")
        contingency_pct = st.slider("Contingency Buffer (%)", min_value=5, max_value=25, value=12, key="calc_cont",
                                    help="Extra buffer for unexpected costs, added on top of subtotal")

    # ---- Use Centralized Calculation Engine ----
    res = calculate_trip_budget(
        num_adults=num_adults,
        num_children=num_children,
        num_days=num_days,
        flight_pp=flight_pp,
        hotel_pn=hotel_pn,
        food_ppd=food_ppd,
        activities=activities,
        transport=transport,
        misc=misc,
        contingency_pct=contingency_pct,
        currency_symbol=currency_sym
    )

    sym = res["currency_symbol"] or "₹"
    total_travelers = res["total_travelers"]
    flights_total = res["flights_total"]
    hotel_total = res["hotel_total"]
    food_total = res["food_total"]
    subtotal = res["subtotal"]
    contingency_amt = res["contingency_amt"]
    grand_total = res["grand_total"]
    per_person = res["per_person"]

    def fmt(n):
        return f"{sym}{n:,.0f}"

    with res_col:
        st.markdown("#### 📊 Itemized Breakdown")

        rows = [
            ("Flights",                f"{fmt(flight_pp)} × {total_travelers} travelers", fmt(flights_total)),
            ("Accommodation",          f"{fmt(hotel_pn)} × {num_days} nights",           fmt(hotel_total)),
            ("Food & Dining",          f"{fmt(food_ppd)} × {total_travelers}p × {num_days}d", fmt(food_total)),
            ("Activities",             "Total (all travelers)",                           fmt(activities)),
            ("Local Transport",        "Total (entire trip)",                             fmt(transport)),
            ("Miscellaneous",          "Optional",                                        fmt(misc)),
            ("Subtotal",               "",                                                fmt(subtotal)),
            (f"Contingency ({contingency_pct}%)", "",                                    fmt(contingency_amt)),
        ]

        table_html = """
        <style>
        .calc-table { width:100%; border-collapse:collapse; font-family:'Noto Sans',sans-serif; font-size:0.88rem; }
        .calc-table th { background:#0284c7; color:white; padding:8px 10px; text-align:left; }
        .calc-table td { padding:7px 10px; border-bottom:1px solid #e2e8f0; }
        .calc-table tr:nth-child(even) { background:#f8fafc; }
        .calc-table .subtotal td { background:#dbeafe; font-weight:600; border-top:2px solid #0284c7; }
        .calc-table .contingency td { background:#fef9c3; font-style:italic; }
        .calc-table .amount { text-align:right; font-weight:600; }
        </style>
        <table class="calc-table">
        <tr><th>Category</th><th>Calculation</th><th class="amount">Amount</th></tr>
        """
        for i, (cat, calc, amt) in enumerate(rows):
            row_class = ""
            if cat == "Subtotal": row_class = " class='subtotal'"
            elif cat.startswith("Contingency"): row_class = " class='contingency'"
            table_html += f"<tr{row_class}><td>{cat}</td><td style='color:#64748b'>{calc}</td><td class='amount'>{amt}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("")

        if subtotal > 0:
            st.markdown("#### 📊 Cost Distribution")
            categories = [
                ("Flights",       flights_total, "#0284c7"),
                ("Accommodation", hotel_total,   "#7c3aed"),
                ("Food",          food_total,    "#059669"),
                ("Activities",    activities,    "#d97706"),
                ("Transport",     transport,     "#db2777"),
                ("Misc",          misc,          "#64748b"),
            ]
            for label, amount, colour in categories:
                if amount > 0:
                    pct = int((amount / subtotal) * 100)
                    bar_html = f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                      <div style="width:90px;font-size:0.8rem;font-family:'Noto Sans',sans-serif;color:#475569">{label}</div>
                      <div style="flex:1;background:#e2e8f0;border-radius:6px;height:14px;overflow:hidden;">
                        <div style="width:{pct}%;background:{colour};height:100%;border-radius:6px;"></div>
                      </div>
                      <div style="width:50px;text-align:right;font-size:0.8rem;font-weight:600;color:#0f172a">{pct}%</div>
                    </div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("")

        gcard, pcard = st.columns(2)
        with gcard:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0284c7,#0369a1);color:white;
                        border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:0.78rem;font-weight:600;opacity:0.85;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:6px;">Grand Total</div>
              <div style="font-size:1.8rem;font-weight:800;font-family:'Noto Sans',sans-serif;">{fmt(grand_total)}</div>
              <div style="font-size:0.75rem;opacity:0.8;margin-top:4px;">incl. {contingency_pct}% contingency</div>
            </div>""", unsafe_allow_html=True)
        with pcard:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#059669,#047857);color:white;
                        border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:0.78rem;font-weight:600;opacity:0.85;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:6px;">Per Person</div>
              <div style="font-size:1.8rem;font-weight:800;font-family:'Noto Sans',sans-serif;">{fmt(per_person)}</div>
              <div style="font-size:0.75rem;opacity:0.8;margin-top:4px;">{total_travelers} traveler(s), {num_days} days</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        summary_lines = [
            f"Trip Cost Summary",
            f"{'='*40}",
            f"Travelers : {num_adults} adult(s), {num_children} child(ren)",
            f"Duration  : {num_days} days",
            f"Currency  : {sym}",
            f"{'='*40}",
            f"Flights        : {fmt(flights_total)}  ({fmt(flight_pp)} x {total_travelers} travelers)",
            f"Accommodation  : {fmt(hotel_total)}  ({fmt(hotel_pn)} x {num_days} nights)",
            f"Food           : {fmt(food_total)}  ({fmt(food_ppd)} x {total_travelers}p x {num_days}d)",
            f"Activities     : {fmt(activities)}",
            f"Transport      : {fmt(transport)}",
            f"Miscellaneous  : {fmt(misc)}",
            f"Subtotal       : {fmt(subtotal)}",
            f"Contingency    : {fmt(contingency_amt)} ({contingency_pct}%)",
            f"{'='*40}",
            f"GRAND TOTAL    : {fmt(grand_total)}",
            f"Per Person     : {fmt(per_person)}",
        ]
        st.download_button(
            label="📥 Download Cost Summary (.txt)",
            data="\n".join(summary_lines),
            file_name="trip_cost_summary.txt",
            mime="text/plain",
            key="calc_download"
        )
