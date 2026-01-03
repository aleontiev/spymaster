"""
Market Hours Utilities.

DST-aware timezone handling for US equity market hours.
"""
from datetime import datetime, date as date_type
from typing import Tuple, Union

import arrow

# Market duration in seconds (9:30 AM - 4:00 PM ET = 6.5 hours)
MARKET_DURATION_SECS = 6.5 * 3600  # 23400 seconds

# Early close duration in seconds (9:30 AM - 1:00 PM ET = 3.5 hours)
EARLY_CLOSE_DURATION_SECS = 3.5 * 3600  # 12600 seconds


def is_early_close_day(date_input: Union[str, date_type, datetime]) -> bool:
    """
    Check if a date is an early close day (market closes at 1:00 PM ET).

    NYSE early close days:
    - Day before Independence Day (July 3, or July 2 if July 4 is on Saturday)
    - Black Friday (day after Thanksgiving - 4th Thursday of November)
    - Christmas Eve (December 24, unless it falls on a weekend)

    Args:
        date_input: Date to check

    Returns:
        True if the market closes early on this date
    """
    if isinstance(date_input, str):
        d = datetime.strptime(date_input, "%Y-%m-%d").date()
    elif isinstance(date_input, datetime):
        d = date_input.date()
    else:
        d = date_input

    month, day, weekday = d.month, d.day, d.weekday()

    # Christmas Eve - Dec 24 (Mon-Fri only)
    if month == 12 and day == 24 and weekday < 5:
        return True

    # Day before Independence Day
    # July 3 normally, but if July 4 is on Saturday, it's July 2 (Friday)
    if month == 7:
        if day == 3 and weekday < 5:  # July 3 if it's a weekday
            return True
        # If July 4 is Saturday, July 3 is Friday (early close on July 2 is not standard)
        # Actually NYSE closes early on July 3 regardless, and if July 4 is Sat/Sun,
        # the holiday is observed on Friday/Monday but July 3 is still early close if weekday

    # Black Friday (day after Thanksgiving = 4th Thursday of November + 1)
    if month == 11 and weekday == 4:  # Friday
        # Check if this is the 4th Friday (which follows the 4th Thursday)
        # 4th Thursday is between day 22-28, so 4th Friday is between day 23-29
        if 23 <= day <= 29:
            return True

    return False


def get_market_hours_utc(date_input: Union[str, date_type, datetime]) -> Tuple[datetime, datetime]:
    """
    Get market open and close times in UTC for a given date.

    Uses arrow library for proper DST handling:
    - During EDT (Mar-Nov): Market open 9:30 AM EDT = 13:30 UTC
    - During EST (Nov-Mar): Market open 9:30 AM EST = 14:30 UTC

    Handles early close days (1:00 PM ET close):
    - Christmas Eve (Dec 24)
    - Day before Independence Day (July 3)
    - Black Friday (day after Thanksgiving)

    Args:
        date_input: Date string (YYYY-MM-DD), date object, or datetime

    Returns:
        Tuple of (market_open_utc, market_close_utc) as naive datetime objects in UTC
    """
    # Parse date to string format for arrow
    if isinstance(date_input, str):
        date_str = date_input
    elif isinstance(date_input, datetime):
        date_str = date_input.strftime("%Y-%m-%d")
    else:
        date_str = date_input.strftime("%Y-%m-%d")

    # Determine close time based on early close status
    close_time = "13:00" if is_early_close_day(date_input) else "16:00"

    # Create market open/close in US/Eastern using arrow (handles DST automatically)
    market_open_et = arrow.get(f"{date_str} 09:30", "YYYY-MM-DD HH:mm", tzinfo="US/Eastern")
    market_close_et = arrow.get(f"{date_str} {close_time}", "YYYY-MM-DD HH:mm", tzinfo="US/Eastern")

    # Convert to UTC and return as naive datetime (for pandas compatibility)
    market_open_utc = market_open_et.to("UTC").naive
    market_close_utc = market_close_et.to("UTC").naive

    return market_open_utc, market_close_utc


def get_market_duration_secs(date_input: Union[str, date_type, datetime]) -> float:
    """
    Get market duration in seconds for a given date.

    Args:
        date_input: Date to check

    Returns:
        Duration in seconds (12600 for early close, 23400 for normal days)
    """
    if is_early_close_day(date_input):
        return EARLY_CLOSE_DURATION_SECS
    return MARKET_DURATION_SECS
