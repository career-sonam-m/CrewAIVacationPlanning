"""
Cost Calculator UI & CrewAI Tool Component - CrewAI Vacation Planner

This module implements:
1. calculate_trip_budget: Pure Python math engine for trip cost calculations.
2. TripCostCalculatorTool: CrewAI BaseTool wrapper for multi-agent execution.
3. render_cost_calculator: Streamlit UI tab for interactive cost calculations.
"""

from typing import Type
import ast
import operator
import re
from pydantic import BaseModel, Field
import streamlit as st

try:
    from crewai.tools import BaseTool
except ImportError:
    try:
        from crewai_tools import BaseTool
    except ImportError:
        # langchain_core.tools.BaseTool is what CrewAgentExecutor actually validates tools against.
        from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Safe arithmetic expression evaluator (no eval()/exec() — AST allowlist only)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {"abs": abs, "round": round, "min": min, "max": max}


def _eval_ast_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_ast_node(node.left), _eval_ast_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_eval_ast_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError("Unsupported or unsafe expression element")


def safe_eval_expression(expression: str) -> float:
    """Evaluate a plain arithmetic expression without using eval()/exec()."""
    parsed = ast.parse(expression, mode="eval").body
    return _eval_ast_node(parsed)


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
    flights: float = Field(default=0.0, description="Deprecated. Alias for flight_pp. Do not use.")
    flight_route: str = Field(default="", description="Deprecated. Do not use.")


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
        currency_symbol: str = "₹",
        flights: float = 0.0,
        flight_route: str = ""
    ) -> str:
        # Gracefully handle older model calls that pass flight totals into the wrong parameter name
        f_pp = flight_pp if flight_pp > 0 else flights
        res = calculate_trip_budget(
            num_adults=num_adults,
            num_children=num_children,
            num_days=num_days,
            flight_pp=f_pp,
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
        """Safely evaluate a mathematical expression using an AST allowlist (no eval())."""
        try:
            clean_expr = expression.replace("$", "").replace("¥", "").replace("€", "").replace("₹", "").strip()
            # Strip thousands-separator commas (e.g. "1,500" or "12,000.50") without breaking
            # multi-arg calls like round(x, 2) or max(1,2,3) — only matches groups of exactly 3 digits.
            clean_expr = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', clean_expr)
            result = safe_eval_expression(clean_expr)
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

    # Treat the planner's extracted budget as the source of truth whenever a new main-page plan arrives.
    if "plan_result" in st.session_state and not st.session_state.get("calc_synced_with_plan", False):
        plan = st.session_state.get("plan_result", {})
        try:
            st.session_state.update(sync_plan_to_calculator(plan))
        except Exception:
            pass
        st.session_state["calc_synced_with_plan"] = True

    inp_col, res_col = st.columns([1, 1.4], gap="large")

    with inp_col:
        st.markdown("#### 👥 Travelers & Duration")
        num_adults   = st.number_input("Adults",   min_value=1, max_value=20, value=2, key="calc_adults")
        num_children = st.number_input("Children", min_value=0, max_value=20, value=0, key="calc_children")
        num_days     = st.number_input("Trip Duration (days)", min_value=1, max_value=180, value=7, key="calc_days")
        # Currency dropdown with common world currencies
        currency_options = [
            "$ - USD (US Dollar)",
            "€ - EUR (Euro)",
            "£ - GBP (British Pound)",
            "₹ - INR (Indian Rupee)",
            "¥ - JPY (Japanese Yen)",
            "¥ - CNY (Chinese Yuan)",
            "A$ - AUD (Australian Dollar)",
            "C$ - CAD (Canadian Dollar)",
            "CHF - CHF (Swiss Franc)",
            "₩ - KRW (South Korean Won)",
            "S$ - SGD (Singapore Dollar)",
            "R$ - BRL (Brazilian Real)",
            "฿ - THB (Thai Baht)",
            "₺ - TRY (Turkish Lira)",
            "د.إ - AED (UAE Dirham)",
            "RM - MYR (Malaysian Ringgit)",
            "₱ - PHP (Philippine Peso)",
            "kr - SEK (Swedish Krona)",
            "zł - PLN (Polish Zloty)",
            "R - ZAR (South African Rand)",
        ]
        # Prefer any previously-selected currency stored in session state so tabs stay in sync
        initial_currency = st.session_state.get("calc_currency_select")
        if initial_currency and initial_currency in currency_options:
            initial_index = currency_options.index(initial_currency)
        else:
            initial_index = 0

        selected_currency = st.selectbox(
            "Currency",
            options=currency_options,
            index=initial_index,
            key="calc_currency_select",
            help="Select the currency for your destination"
        )
        # Extract just the symbol from the selected option (everything before ' - ')
        currency_sym = selected_currency.split(" - ")[0].strip()

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

        # When returning from the main planner, prefer values already in session state
        flight_default = st.session_state.get("calc_flight", default_flight)
        hotel_default = st.session_state.get("calc_hotel", default_hotel)
        food_default = st.session_state.get("calc_food", default_food)
        activities_default = st.session_state.get("calc_activities", default_activities)
        transport_default = st.session_state.get("calc_transport", default_transport)
        misc_default = st.session_state.get("calc_misc", 0.0)

        if st.session_state.calc_prev_currency != currency_sym:
            st.session_state.calc_prev_currency = currency_sym
            # Only overwrite inputs if they weren't explicitly set by the planner run
            if "calc_flight" not in st.session_state:
                st.session_state.calc_flight = default_flight
            if "calc_hotel" not in st.session_state:
                st.session_state.calc_hotel = default_hotel
            if "calc_food" not in st.session_state:
                st.session_state.calc_food = default_food
            if "calc_activities" not in st.session_state:
                st.session_state.calc_activities = default_activities
            if "calc_transport" not in st.session_state:
                st.session_state.calc_transport = default_transport
            if "calc_misc" not in st.session_state:
                st.session_state.calc_misc = 0.0
            st.rerun()

        st.markdown("#### 💰 Cost Inputs")
        flight_pp    = st.number_input(f"Flight — round-trip per person ({currency_sym})",
                           min_value=0.0, value=flight_default, step=10.0 if not is_high_value_currency else 1000.0, key="calc_flight",
                           help="Economy round-trip cost per traveler")
        hotel_pn     = st.number_input(f"Hotel — per night, all rooms ({currency_sym})",
                           min_value=0.0, value=hotel_default, step=10.0 if not is_high_value_currency else 500.0, key="calc_hotel",
                           help="Total room cost per night for the whole party")
        food_ppd     = st.number_input(f"Food — per person per day ({currency_sym})",
                           min_value=0.0, value=food_default, step=5.0 if not is_high_value_currency else 100.0, key="calc_food",
                           help="Average daily food spend per traveler")
        activities   = st.number_input(f"Activities & Attractions — total ({currency_sym})",
                           min_value=0.0, value=activities_default, step=10.0 if not is_high_value_currency else 500.0, key="calc_activities",
                           help="Total tickets, tours, and experiences for all travelers")
        transport    = st.number_input(f"Local Transport — total ({currency_sym})",
                           min_value=0.0, value=transport_default, step=10.0 if not is_high_value_currency else 500.0, key="calc_transport",
                           help="Taxis, metro passes, trains for the whole trip")
        misc         = st.number_input(f"Miscellaneous / Shopping ({currency_sym})",
                           min_value=0.0, value=misc_default, step=10.0 if not is_high_value_currency else 500.0, key="calc_misc",
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


def sync_plan_to_calculator(plan_data: dict) -> dict:
    """Extract budget data from plan and sync it to calculator session state.

    Args:
        plan_data: Dictionary containing the vacation plan with extracted_budget

    Returns:
        Dictionary of calculator state values to update
    """
    if not plan_data or not plan_data.get("extracted_budget"):
        return {}

    eb = plan_data["extracted_budget"]
    sym = plan_data.get("currency_symbol", "$")

    currency_map = {
        "$": "$ - USD (US Dollar)",
        "€": "€ - EUR (Euro)",
        "£": "£ - GBP (British Pound)",
        "₹": "₹ - INR (Indian Rupee)",
        "¥": "¥ - JPY (Japanese Yen)",
        "A$": "A$ - AUD (Australian Dollar)",
        "C$": "C$ - CAD (Canadian Dollar)",
        "CHF": "CHF - CHF (Swiss Franc)",
        "₩": "₩ - KRW (South Korean Won)",
        "S$": "S$ - SGD (Singapore Dollar)",
        "R$": "R$ - BRL (Brazilian Real)",
        "฿": "฿ - THB (Thai Baht)",
        "₺": "₺ - TRY (Turkish Lira)",
        "د.إ": "د.إ - AED (UAE Dirham)",
        "RM": "RM - MYR (Malaysian Ringgit)",
        "₱": "₱ - PHP (Philippine Peso)",
        "kr": "kr - SEK (Swedish Krona)",
        "zł": "zł - PLN (Polish Zloty)",
        "R": "R - ZAR (South African Rand)",
    }

    currency_select = currency_map.get(sym, "$ - USD (US Dollar)")

    return {
        "calc_flight": float(eb.get("flight_pp", 0.0)),
        "calc_hotel": float(eb.get("hotel_pn", 0.0)),
        "calc_food": float(eb.get("food_ppd", 0.0)),
        "calc_activities": float(eb.get("activities", 0.0)),
        "calc_transport": float(eb.get("transport", 0.0)),
        "calc_misc": float(eb.get("misc", 0.0)),
        "calc_adults": int(eb.get("num_adults", 2)),
        "calc_children": int(eb.get("num_children", 0)),
        "calc_days": int(eb.get("num_days", 7)),
        "calc_cont": float(eb.get("contingency_pct", 12.0)),
        "calc_currency_select": currency_select,
        "calc_prev_currency": sym,
    }
