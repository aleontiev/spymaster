"""
Backtest multi-scale percentile policy fusion with triple barrier exits.

Combines three independently-trained percentile policy models (15m, 5m, 1m horizons)
into a single unified trading bot using 3/3 voting fusion.

Signal Logic:
- All 3 models must agree on direction (LONG or SHORT)
- If any model signals the opposite direction, no trade

Usage:
    uv run python backtest_multi_percentile.py --start-date 2024-01-01 --end-date 2024-12-31
"""
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.backtest import (
    BacktestConfig,
    MultiPercentileBacktester,
    PositionType,
    print_results,
)
from src.execution import FilteredExecutor
from src.strategy.fusion_config import (
    FusedSignal,
    FusionConfig,
    PercentileModelConfig,
    Signal,
    TradingWindow,
)
from src.strategy.multi_percentile_executor import MultiPercentileExecutor


def load_data(start_date: date, end_date: date) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load normalized data, raw closes, and OHLCV data."""
    from src.data.dag.loader import load_dataset

    print(f"Loading data from {start_date} to {end_date}...")

    df = load_dataset(
        dataset="training-1m-normalized",
        underlying="SPY",
        start_date=start_date,
        end_date=end_date,
    )

    raw_df = load_dataset(
        dataset="training-1m-raw",
        underlying="SPY",
        start_date=start_date,
        end_date=end_date,
    )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    if "timestamp" in raw_df.columns:
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
        raw_df = raw_df.set_index("timestamp").sort_index()

    common_idx = df.index.intersection(raw_df.index)
    df = df.loc[common_idx]
    raw_closes = raw_df.loc[common_idx, "close"].values

    # Extract OHLCV for charting
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    available_ohlcv = [c for c in ohlcv_cols if c in raw_df.columns]
    ohlcv_df = raw_df.loc[common_idx, available_ohlcv].copy()

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    exclude_cols = {'timestamp', 'date', 'datetime', 'index', 'symbol', 'underlying'}
    feature_cols = sorted([
        col for col in df.columns
        if col.lower() not in exclude_cols
        and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ])
    df = df[feature_cols]

    print(f"Loaded {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
    print(f"Features: {df.shape[1]} columns (sorted to match training)")

    return df, raw_closes, ohlcv_df


def run_unified_backtest(
    executor: MultiPercentileExecutor,
    df: pd.DataFrame,
    raw_closes: np.ndarray,
    config: BacktestConfig,
    device: torch.device,
    start_date: date,
    end_date: date,
    ohlcv_df: Optional[pd.DataFrame] = None,
    generate_html: bool = False,
    output_dir: Optional[Path] = None,
    command: Optional[str] = None,
    use_heuristic: bool = True,
) -> Dict:
    """Run unified backtest using all 3 models with 2/3 agreement logic."""
    from src.backtest.html_report import SignalMarker, TradeMarker, generate_html_report

    # Use all 3 models
    all_models = {"1m", "5m", "15m"}
    filtered_executor = FilteredExecutor(executor, all_models)

    all_monthly_results = []
    all_yearly_results = []
    all_trades = []
    all_signals = {"1m": [], "5m": [], "15m": []}
    total_heuristic_rejections = 0

    for year in range(start_date.year, end_date.year + 1):
        yearly_trades = []
        yearly_capital = config.initial_capital

        # Determine month range for this year based on date range
        start_month = start_date.month if year == start_date.year else 1
        end_month = end_date.month if year == end_date.year else 12

        for month in range(start_month, end_month + 1):
            mask = (df.index.year == year) & (df.index.month == month)
            if mask.sum() < 100:
                continue

            backtester = MultiPercentileBacktester(filtered_executor, config, device, use_heuristic=use_heuristic)

            if generate_html:
                metrics, month_signals = backtester.run_backtest_with_signals(
                    df, raw_closes, year, month, executor, starting_capital=yearly_capital, ohlcv_df=ohlcv_df
                )
                for model, sigs in month_signals.items():
                    all_signals[model].extend(sigs)
            else:
                metrics = backtester.run_backtest(df, raw_closes, year, month, starting_capital=yearly_capital, ohlcv_df=ohlcv_df)

            total_heuristic_rejections += backtester.heuristic_rejections

            if metrics.get("trades", 0) > 0:
                yearly_capital = metrics["final_capital"]
                yearly_trades.extend(backtester.trades)
                all_trades.extend(backtester.trades)

            monthly_result = {
                "year": year,
                "month": month,
                **metrics
            }
            all_monthly_results.append(monthly_result)

        # Yearly totals
        if yearly_trades:
            wins = [t.pnl_pct for t in yearly_trades if t.pnl_pct > 0]
            yearly_result = {
                "year": year,
                "trades": len(yearly_trades),
                "win_rate": len(wins) / len(yearly_trades) * 100,
                "total_pnl_dollars": sum(t.pnl_dollars for t in yearly_trades),
                "total_return_pct": (yearly_capital - config.initial_capital) / config.initial_capital * 100,
                "final_capital": yearly_capital,
                "long_trades": len([t for t in yearly_trades if t.position_type == PositionType.CALL]),
                "short_trades": len([t for t in yearly_trades if t.position_type == PositionType.PUT]),
            }
        else:
            yearly_result = {
                "year": year,
                "trades": 0,
                "win_rate": 0,
                "total_pnl_dollars": 0,
                "total_return_pct": 0,
                "final_capital": config.initial_capital,
                "long_trades": 0,
                "short_trades": 0,
            }
        all_yearly_results.append(yearly_result)

    # Generate HTML report
    html_path = None
    if generate_html and ohlcv_df is not None and output_dir is not None:
        trade_markers = []
        for t in all_trades:
            parent_ts = int(t.parent_entry_time.timestamp()) if t.parent_entry_time else 0
            trade_markers.append(TradeMarker(
                entry_time=int(t.entry_time.timestamp()),
                exit_time=int(t.exit_time.timestamp()),
                entry_price=t.entry_underlying,
                exit_price=t.exit_underlying,
                is_long=t.position_type == PositionType.CALL,
                is_win=t.pnl_pct > 0,
                pnl_pct=t.pnl_pct,
                pnl_dollars=t.pnl_dollars,
                agreeing_models=getattr(t, 'agreeing_models', ()),
                exit_reason=t.exit_reason,
                capital_after=t.capital_after,
                contract=getattr(t, 'contract', ''),
                entry_option_price=t.entry_option_price,
                exit_option_price=t.exit_option_price,
                num_contracts=getattr(t, 'num_contracts', 10),
                is_runner=getattr(t, 'is_runner', False),
                parent_entry_time=parent_ts,
            ))

        # Generate filename: SPY-{START_DATE}-{END_DATE}-{TIMESTAMP}-backtest.html
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"SPY-{start_str}-{end_str}-{timestamp}-backtest.html"
        html_path = output_dir / html_filename

        generate_html_report(
            ohlcv_df=ohlcv_df,
            trades=trade_markers,
            signals=all_signals,
            combination_name="1m+5m+15m (unified)",
            output_path=html_path,
            title=f"SPY Backtest ({start_date} to {end_date})",
            command=command,
        )
        print(f"  HTML report saved to: {html_path}")

    return {
        "monthly": all_monthly_results,
        "yearly": all_yearly_results,
        "html_path": str(html_path) if html_path else None,
        "trades": all_trades,
        "heuristic_rejections": total_heuristic_rejections,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest multi-scale percentile policy fusion")
    parser.add_argument("--start-date", type=str, default="2024-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--no-trade-first-minutes", type=int, default=15)
    parser.add_argument("--no-trade-last-minutes", type=int, default=15)
    parser.add_argument("--stop-loss-pct", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML reports with interactive charts")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory for HTML reports")
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                        help="Use heuristic model for final confirmation (default: True)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                        help="Disable heuristic model confirmation")

    # Model checkpoint paths (for reproducibility in reports)
    parser.add_argument("--15m-jepa", type=str,
                        default="data/checkpoints/lejepa-60-15/lejepa_best.pt",
                        help="Path to 15m LeJEPA checkpoint")
    parser.add_argument("--15m-policy", type=str,
                        default="data/checkpoints/percentile-60-15-v2/entry_policy_best.pt",
                        help="Path to 15m policy checkpoint")
    parser.add_argument("--5m-jepa", type=str,
                        default="data/checkpoints/lejepa-15-5-v1/lejepa_best.pt",
                        help="Path to 5m LeJEPA checkpoint")
    parser.add_argument("--5m-policy", type=str,
                        default="data/checkpoints/percentile-15-5-v1/entry_policy_best.pt",
                        help="Path to 5m policy checkpoint")
    parser.add_argument("--1m-jepa", type=str,
                        default="data/checkpoints/lejepa-5-1-v1/lejepa_best.pt",
                        help="Path to 1m LeJEPA checkpoint")
    parser.add_argument("--1m-policy", type=str,
                        default="data/checkpoints/percentile-5-1-v1/entry_policy_best.pt",
                        help="Path to 1m policy checkpoint")

    # Threshold arguments for each model
    parser.add_argument("--15m-long-threshold", type=float, default=0.48,
                        help="15m model long threshold (higher = fewer signals)")
    parser.add_argument("--15m-short-threshold", type=float, default=0.56,
                        help="15m model short threshold (higher = fewer signals)")
    parser.add_argument("--5m-long-threshold", type=float, default=0.49,
                        help="5m model long threshold (higher = fewer signals)")
    parser.add_argument("--5m-short-threshold", type=float, default=0.51,
                        help="5m model short threshold (higher = fewer signals)")
    parser.add_argument("--1m-long-threshold", type=float, default=0.50,
                        help="1m model long threshold (higher = fewer signals)")
    parser.add_argument("--1m-short-threshold", type=float, default=0.55,
                        help="1m model short threshold (higher = fewer signals)")

    args = parser.parse_args()

    # Build full command with ALL explicit parameters for reproducibility
    full_command = f"""uv run python backtest_multi_percentile.py \\
    --start-date {args.start_date} \\
    --end-date {args.end_date} \\
    --no-trade-first-minutes {args.no_trade_first_minutes} \\
    --no-trade-last-minutes {args.no_trade_last_minutes} \\
    --stop-loss-pct {args.stop_loss_pct} \\
    --15m-jepa {getattr(args, '15m_jepa')} \\
    --15m-policy {getattr(args, '15m_policy')} \\
    --15m-long-threshold {getattr(args, '15m_long_threshold')} \\
    --15m-short-threshold {getattr(args, '15m_short_threshold')} \\
    --5m-jepa {getattr(args, '5m_jepa')} \\
    --5m-policy {getattr(args, '5m_policy')} \\
    --5m-long-threshold {getattr(args, '5m_long_threshold')} \\
    --5m-short-threshold {getattr(args, '5m_short_threshold')} \\
    --1m-jepa {getattr(args, '1m_jepa')} \\
    --1m-policy {getattr(args, '1m_policy')} \\
    --1m-long-threshold {getattr(args, '1m_long_threshold')} \\
    --1m-short-threshold {getattr(args, '1m_short_threshold')} \\
    {'--use-heuristic' if args.use_heuristic else '--no-heuristic'} \\
    --html --output-dir {args.output_dir}"""

    # Parse date strings to date objects
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nRunning unified 3-model backtest (3/3 agreement logic)")
    print(f"Heuristic confirmation: {'ENABLED' if args.use_heuristic else 'DISABLED'}")

    # Create output directory if generating HTML
    output_dir = None
    if args.html:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"HTML reports will be saved to: {output_dir}")

    # Create fusion config
    trading_window = TradingWindow(
        open_after_minutes=args.no_trade_first_minutes,
        close_before_minutes=args.no_trade_last_minutes,
    )
    fusion_config = FusionConfig(
        weights=(0.333, 0.333, 0.333),
        min_agreement=0.40,
        trading_window=trading_window,
        stop_loss_pct=args.stop_loss_pct,
    )

    # Build model configs from CLI arguments (allows overriding default paths)
    # Access args with getattr since argparse converts dashes to underscores for some
    jepa_15m = getattr(args, '15m_jepa', None) or "data/checkpoints/lejepa-60-15/lejepa_best.pt"
    policy_15m = getattr(args, '15m_policy', None) or "data/checkpoints/percentile-60-15-v2/entry_policy_best.pt"
    jepa_5m = getattr(args, '5m_jepa', None) or "data/checkpoints/lejepa-15-5-v1/lejepa_best.pt"
    policy_5m = getattr(args, '5m_policy', None) or "data/checkpoints/percentile-15-5-v1/entry_policy_best.pt"
    jepa_1m = getattr(args, '1m_jepa', None) or "data/checkpoints/lejepa-5-1-v1/lejepa_best.pt"
    policy_1m = getattr(args, '1m_policy', None) or "data/checkpoints/percentile-5-1-v1/entry_policy_best.pt"

    # Get thresholds from args
    threshold_15m_long = getattr(args, '15m_long_threshold')
    threshold_15m_short = getattr(args, '15m_short_threshold')
    threshold_5m_long = getattr(args, '5m_long_threshold')
    threshold_5m_short = getattr(args, '5m_short_threshold')
    threshold_1m_long = getattr(args, '1m_long_threshold')
    threshold_1m_short = getattr(args, '1m_short_threshold')

    custom_model_configs = [
        PercentileModelConfig(
            name="15m",
            lejepa_path=jepa_15m,
            policy_path=policy_15m,
            context_len=60,
            horizon_minutes=15,
            long_threshold=threshold_15m_long,
            short_threshold=threshold_15m_short,
            base_weight=0.333,
            min_context_minutes=60,
        ),
        PercentileModelConfig(
            name="5m",
            lejepa_path=jepa_5m,
            policy_path=policy_5m,
            context_len=15,
            horizon_minutes=5,
            long_threshold=threshold_5m_long,
            short_threshold=threshold_5m_short,
            base_weight=0.333,
            min_context_minutes=15,
        ),
        PercentileModelConfig(
            name="1m",
            lejepa_path=jepa_1m,
            policy_path=policy_1m,
            context_len=5,
            horizon_minutes=1,
            long_threshold=threshold_1m_long,
            short_threshold=threshold_1m_short,
            base_weight=0.333,
            min_context_minutes=5,
        ),
    ]

    # Load all models (we'll filter which ones to use per combination)
    print("\nLoading models...")
    executor = MultiPercentileExecutor.from_configs(
        model_configs=custom_model_configs,
        fusion_config=fusion_config,
        device=str(device),
    )

    print("\nModel Configuration:")
    for name, info in executor.get_model_info().items():
        print(f"  {name}: context={info['context_len']}, horizon={info['horizon_minutes']}, "
              f"thresholds=(L:{info['long_threshold']}, S:{info['short_threshold']})")

    # Load data
    df, raw_closes, ohlcv_df = load_data(start_date, end_date)

    # Backtest config
    backtest_config = BacktestConfig(
        stop_loss_pct=args.stop_loss_pct,
    )

    # Run unified backtest with all 3 models
    print(f"\n{'='*60}")
    print("RUNNING UNIFIED BACKTEST")
    print("=" * 60)

    result = run_unified_backtest(
        executor=executor,
        df=df,
        raw_closes=raw_closes,
        config=backtest_config,
        device=device,
        start_date=start_date,
        end_date=end_date,
        ohlcv_df=ohlcv_df,  # Always pass for heuristic model
        generate_html=args.html,
        output_dir=output_dir,
        command=full_command if args.html else None,
        use_heuristic=args.use_heuristic,
    )

    # Print results
    print_results(result, start_date, end_date)

    # Print HTML report path
    if args.html and result.get("html_path"):
        print("\n" + "=" * 60)
        print("HTML REPORT GENERATED")
        print("=" * 60)
        print(f"  {result['html_path']}")


if __name__ == "__main__":
    main()
