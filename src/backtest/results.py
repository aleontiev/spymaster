"""
Backtest result printing and formatting utilities.
"""
from datetime import date, datetime
from typing import Dict, List


def print_results(result: Dict, start_date: date, end_date: date) -> None:
    """
    Print results for unified backtest.

    Args:
        result: Backtest result dict with 'yearly', 'monthly', 'trades' keys
        start_date: Start date of backtest
        end_date: End date of backtest
    """
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Yearly summary
    print(f"\n{'Year':<8} {'Trades':>8} {'Win%':>8} {'Return%':>10} {'P&L $':>12} {'Long':>6} {'Short':>6}")
    print("-" * 70)

    total_trades = 0
    total_pnl = 0.0

    for y in result["yearly"]:
        year = y["year"]
        trades = y["trades"]
        win_rate = y["win_rate"]
        return_pct = y["total_return_pct"]
        pnl = y["total_pnl_dollars"]
        longs = y["long_trades"]
        shorts = y["short_trades"]

        print(f"{year:<8} {trades:>8} {win_rate:>7.1f}% {return_pct:>+9.1f}% ${pnl:>10,.0f} {longs:>6} {shorts:>6}")

        total_trades += trades
        total_pnl += pnl

    print("-" * 70)

    # Calculate overall win rate
    all_trades = result.get("trades", [])
    if all_trades:
        wins = sum(1 for t in all_trades if t.pnl_pct > 0)
        overall_win_rate = wins / len(all_trades) * 100
    else:
        overall_win_rate = 0

    final_capital = result["yearly"][-1]["final_capital"] if result["yearly"] else 100000
    overall_return = (final_capital - 100000) / 100000 * 100

    print(f"{'TOTAL':<8} {total_trades:>8} {overall_win_rate:>7.1f}% {overall_return:>+9.1f}% ${total_pnl:>10,.0f}")

    # Heuristic model stats
    heuristic_rejections = result.get("heuristic_rejections", 0)
    if heuristic_rejections > 0:
        print(f"\nHeuristic Model: Rejected {heuristic_rejections} signals (candle color mismatch)")

    # Monthly breakdown
    print("\n" + "=" * 80)
    print("MONTHLY BREAKDOWN")
    print("=" * 80)

    for year in range(start_date.year, end_date.year + 1):
        monthly = [m for m in result["monthly"] if m["year"] == year and m.get("trades", 0) > 0]
        if not monthly:
            continue

        print(f"\n  {year}:")
        print(f"  {'Month':<8} {'Trades':>8} {'Win%':>8} {'Return%':>10} {'P&L $':>12} {'Long':>6} {'Short':>6}")
        print(f"  {'-'*66}")

        for m in sorted(monthly, key=lambda x: x["month"]):
            month_name = datetime(2000, m["month"], 1).strftime("%b")
            trades = m.get("trades", 0)
            win_rate = m.get("win_rate", 0)
            return_pct = m.get("total_return_pct", 0)
            pnl = m.get("total_pnl_dollars", 0)
            longs = m.get("long_trades", 0)
            shorts = m.get("short_trades", 0)

            print(f"  {month_name:<8} {trades:>8} {win_rate:>7.1f}% {return_pct:>+9.1f}% ${pnl:>10,.0f} {longs:>6} {shorts:>6}")
