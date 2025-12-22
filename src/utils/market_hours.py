"""
Market Hours Utilities.

DST-aware timezone handling for US equity market hours.
"""
from datetime import datetime, date as date_type
from typing import Tuple, Union

import arrow

# Market duration in seconds (9:30 AM - 4:00 PM ET = 6.5 hours)
MARKET_DURATION_SECS = 6.5 * 3600  # 23400 seconds


def get_market_hours_utc(date_input: Union[str, date_type, datetime]) -> Tuple[datetime, datetime]:
    """
    Get market open and close times in UTC for a given date.

    Uses arrow library for proper DST handling:
    - During EDT (Mar-Nov): Market open 9:30 AM EDT = 13:30 UTC
    - During EST (Nov-Mar): Market open 9:30 AM EST = 14:30 UTC

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

    # Create market open/close in US/Eastern using arrow (handles DST automatically)
    market_open_et = arrow.get(f"{date_str} 09:30", "YYYY-MM-DD HH:mm", tzinfo="US/Eastern")
    market_close_et = arrow.get(f"{date_str} 16:00", "YYYY-MM-DD HH:mm", tzinfo="US/Eastern")

    # Convert to UTC and return as naive datetime (for pandas compatibility)
    market_open_utc = market_open_et.to("UTC").naive
    market_close_utc = market_close_et.to("UTC").naive

    return market_open_utc, market_close_utc
