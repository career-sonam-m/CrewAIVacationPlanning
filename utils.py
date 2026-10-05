"""
Utility Functions - CrewAI Vacation Planner

This module contains stateless helper functions for travel date processing,
booking checklist timeline HTML generation, destination local currency symbol extraction,
and console Unicode safety.
"""

import re
import sys
from datetime import datetime, timedelta

# Booking milestones configuration offsets (in days before travel date) and task labels
BOOKING_MILESTONE_OFFSETS = {
    "flights":           (120, "Book international/long flights"),
    "travel_insurance":  (90,  "Purchase travel insurance"),
    "accommodations":    (60,  "Book hotels/accommodations"),
    "transport":         (45,  "Book car rentals/rail passes"),
    "attractions":       (30,  "Book attraction tickets and passes"),
    "dining":            (21,  "Make dining reservations"),
    "airport_transfers": (14,  "Arrange airport transfers and logistics"),
}


class UnicodeSafeWriter:
    """Safe stdout/stderr writer wrapper for Windows console to prevent UnicodeEncodeError
    when printing non-ASCII characters, emojis, or currency symbols (e.g. ₹, €, ¥)."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        encoding = getattr(self.stream, 'encoding', 'utf-8') or 'utf-8'
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            safe_data = data.encode(encoding, errors='replace').decode(encoding)
            self.stream.write(safe_data)

    def flush(self):
        if hasattr(self.stream, 'flush'):
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_safe_console_output():
    """Apply safe unicode stdout/stderr writer globally across OS environments."""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            return
        except Exception:
            pass
    sys.stdout = UnicodeSafeWriter(sys.stdout)
    sys.stderr = UnicodeSafeWriter(sys.stderr)


def get_currency_symbol(currency_str: str) -> str:
    """Extract just the symbol from currency string dynamically.
    Instead of hardcoding a list, we look for text enclosed in parentheses."""
    if not currency_str:
        return "₹"
    match = re.search(r'\(([^)]+)\)', currency_str)
    if match:
        return match.group(1).strip()
    return currency_str.strip()


def validate_and_adjust_dates(date_str: str) -> str:
    """Validate that travel dates are in the future relative to the current date.
    Adjusts them if needed while preserving the duration of the trip."""
    try:
        # Remove parenthetical descriptions (e.g., "(a week during summer break)")
        date_str_clean = date_str.split('(')[0].strip()
        
        # Try multiple date separators
        separator = None
        if " to " in date_str_clean:
            separator = " to "
        elif " - " in date_str_clean:
            separator = " - "
        elif "-" in date_str_clean and "to" not in date_str_clean:
            parts = date_str_clean.split("-")
            if len(parts) > 2:  # Multiple dashes, likely date range
                separator = "-"
        
        if separator:
            parts = date_str_clean.split(separator)
            start_date_str = parts[0].strip()
            end_date_str = parts[1].strip() if len(parts) > 1 else parts[0].strip()
            
            start_year_match = re.search(r'\b(20\d{2})\b', start_date_str)
            end_year_match = re.search(r'\b(20\d{2})\b', end_date_str)
            if not start_year_match and end_year_match:
                year = end_year_match.group(1)
                start_date_str = f"{start_date_str}, {year}"
            elif start_year_match and not end_year_match:
                year = start_year_match.group(1)
                end_date_str = f"{end_date_str}, {year}"
        else:
            start_date_str = date_str_clean.strip()
            end_date_str = date_str_clean.strip()
        
        start_date = None
        end_date = None
        
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            pass
        
        if not start_date:
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", 
                       "%d %B %Y", "%d %b %Y", "%B %d", "%d %B"]:
                try:
                    if "20" in start_date_str or "202" in start_date_str:
                        start_date = datetime.strptime(start_date_str, fmt)
                    else:
                        parsed = datetime.strptime(start_date_str, fmt)
                        current_year = datetime.now().year
                        start_date = parsed.replace(year=current_year)
                        if start_date < datetime.now():
                            start_date = start_date.replace(year=current_year + 1)
                    break
                except ValueError:
                    continue
        
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            pass
        
        if not end_date:
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
                       "%d %B %Y", "%d %b %Y", "%B %d", "%d %B"]:
                try:
                    if "20" in end_date_str or "202" in end_date_str:
                        end_date = datetime.strptime(end_date_str, fmt)
                    else:
                        parsed = datetime.strptime(end_date_str, fmt)
                        if start_date:
                            end_date = parsed.replace(year=start_date.year)
                        else:
                            current_year = datetime.now().year
                            end_date = parsed.replace(year=current_year)
                    break
                except ValueError:
                    continue
        
        if not start_date:
            print(f"[Warning] Could not parse date format: {date_str}")
            return date_str
        
        current_date = datetime.now()
        
        if start_date < current_date:
            days_past = (current_date - start_date).days
            new_start_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if end_date and end_date != start_date:
                try:
                    duration = (end_date - start_date).days
                    new_end_date = new_start_date + timedelta(days=duration)
                    new_date_str = f"{new_start_date.strftime('%Y-%m-%d')} to {new_end_date.strftime('%Y-%m-%d')}"
                except Exception:
                    new_date_str = new_start_date.strftime('%Y-%m-%d')
            else:
                new_date_str = new_start_date.strftime('%Y-%m-%d')
            
            print(f"[Warning] Original date {date_str} is in the past ({days_past} days ago)")
            print(f"[Success] Adjusted to: {new_date_str}")
            return new_date_str
        
        if end_date and end_date != start_date:
            return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            return start_date.strftime('%Y-%m-%d')
        
    except Exception as e:
        print(f"[Error] Date parsing error: {e}")
        return date_str


def generate_booking_timeline(date_str: str) -> str:
    """Generate dynamic booking timeline based on travel dates.
    Shows ASAP cleanly (without any past date) for items whose deadline has passed."""
    try:
        if " to " in date_str:
            start_date_str = date_str.split(" to ")[0].strip()
        else:
            start_date_str = date_str.strip()

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

        timeline = {}
        for key, (days, description) in BOOKING_MILESTONE_OFFSETS.items():
            timeline[key] = (start_date - timedelta(days=days), description)

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timeline_str = "\n#### Actionable Booking Checklist\n\n"

        for item, (deadline, description) in timeline.items():
            if deadline <= today:
                timeline_str += f"- **ASAP** ⚡: {description}\n"
            else:
                timeline_str += f"- **{deadline.strftime('%B %d, %Y')}**: {description}\n"

        return timeline_str

    except Exception as e:
        print(f"[Warning] Could not generate booking timeline: {e}")
        return ""


def generate_default_demo_dates() -> str:
    """Generate dynamic default demo dates (3 months from now for 15 days)"""
    today = datetime.now()
    start_date = today + timedelta(days=90)
    end_date = start_date + timedelta(days=15)
    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
