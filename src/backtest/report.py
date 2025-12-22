"""
HTML Report Generator for Backtesting.

Generates interactive HTML reports with:
- Zoomable candlestick charts with trade markers
- Performance metrics summary
- Trade history table
- Equity curve visualization
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd

from src.backtest.engine import Trade, PositionType, BacktestConfig


def generate_json_report(
    df: pd.DataFrame,
    trades: List[Trade],
    metrics: Dict[str, float],
    config: BacktestConfig,
    equity_curve: List[tuple],
    output_path: str = "backtest_report.json",
    title: str = "Spymaster Backtest Report",
    strategy_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
    checkpoints: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Generate a JSON backtest report for use with the webui.

    Args:
        df: Market data DataFrame with OHLCV
        trades: List of completed trades
        metrics: Performance metrics dictionary
        config: Backtest configuration
        equity_curve: List of (timestamp, equity) tuples
        output_path: Path to save JSON report
        title: Report title
        strategy_id: Optional UUID of the strategy this backtest belongs to
        strategy_name: Optional name of the strategy (for display)
        checkpoints: Optional dict of checkpoint entries with id and path

    Returns:
        Path to generated report
    """
    # Prepare data sections
    candle_data = _prepare_candle_data_dict(df)
    trade_markers = _prepare_trade_markers_list(trades)
    equity_data = _prepare_equity_data_list(equity_curve)
    trades_table = _prepare_trades_table(trades)

    # Build report structure
    report = {
        "title": title,
        "generated_at": datetime.now().isoformat(),
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "checkpoints": checkpoints,
        "date_range": {
            "start": df.index.min().strftime("%Y-%m-%d"),
            "end": df.index.max().strftime("%Y-%m-%d"),
        },
        "total_bars": len(df),
        "metrics": {
            "total_return": metrics.get("return_pct", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "final_equity": metrics.get("final_equity", config.initial_capital),
            "total_trades": int(metrics.get("total_trades", 0)),
            "winning_trades": len([t for t in trades if t.pnl > 0]),
            "losing_trades": len([t for t in trades if t.pnl < 0]),
            "win_rate": metrics.get("win_rate", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "avg_win": metrics.get("avg_win", 0),
            "avg_loss": metrics.get("avg_loss", 0),
            "expectancy": metrics.get("expectancy", 0),
            "risk_reward": metrics.get("risk_reward", 0),
        },
        "config": {
            "initial_capital": config.initial_capital,
            "max_position_size": config.max_position_size,
            "slippage_bps": config.slippage_bps,
            "commission_per_contract": config.commission_per_contract,
            "spread_pct": config.spread_pct,
            "options_multiplier": config.options_multiplier,
        },
        "candles": candle_data,
        "trade_markers": trade_markers,
        "equity_curve": equity_data,
        "trades": trades_table,
    }

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"JSON report saved to {output_path}")
    return str(output_path)


def generate_html_report(
    df: pd.DataFrame,
    trades: List[Trade],
    metrics: Dict[str, float],
    config: BacktestConfig,
    equity_curve: List[tuple],
    output_path: str = "backtest_report.html",
    title: str = "Spymaster Backtest Report",
) -> str:
    """
    Generate an interactive HTML backtest report.

    Args:
        df: Market data DataFrame with OHLCV
        trades: List of completed trades
        metrics: Performance metrics dictionary
        config: Backtest configuration
        equity_curve: List of (timestamp, equity) tuples
        output_path: Path to save HTML report
        title: Report title

    Returns:
        Path to generated report
    """
    # Prepare candlestick data (sample if too large)
    candle_data = _prepare_candle_data(df)

    # Prepare trade markers
    trade_markers = _prepare_trade_markers(trades)

    # Prepare equity curve data
    equity_data = _prepare_equity_data(equity_curve)

    # Prepare trades table
    trades_table = _prepare_trades_table(trades)

    # Generate HTML
    html = _generate_html(
        title=title,
        candle_data=candle_data,
        trade_markers=trade_markers,
        equity_data=equity_data,
        trades_table=trades_table,
        metrics=metrics,
        config=config,
        start_date=df.index.min().strftime("%Y-%m-%d"),
        end_date=df.index.max().strftime("%Y-%m-%d"),
        total_bars=len(df),
    )

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    print(f"Report saved to {output_path}")
    return str(output_path)


def _prepare_candle_data(df: pd.DataFrame, max_points: int = 10000) -> str:
    """Prepare candlestick data as JSON string for Plotly."""
    return json.dumps(_prepare_candle_data_dict(df, max_points))


def _prepare_candle_data_dict(df: pd.DataFrame, max_points: int = 10000) -> dict:
    """Prepare candlestick data as dict for JSON export."""
    # Sample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]

    return {
        "timestamps": [ts.isoformat() for ts in df.index],
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist() if "volume" in df.columns else [],
    }


def _prepare_trade_markers(trades: List[Trade]) -> str:
    """Prepare trade entry/exit markers as JSON string."""
    return json.dumps(_prepare_trade_markers_list(trades))


def _prepare_trade_markers_list(trades: List[Trade]) -> List[Dict]:
    """Prepare trade entry/exit markers as list for JSON export."""
    markers = []

    for trade in trades:
        # Entry marker
        markers.append({
            "timestamp": trade.entry_timestamp.isoformat(),
            "price": trade.entry_price,
            "type": "entry",
            "position": trade.position_type.value,
            "quantity": trade.quantity,
            "pnl": None,
        })

        # Exit marker
        markers.append({
            "timestamp": trade.exit_timestamp.isoformat(),
            "price": trade.exit_price,
            "type": "exit",
            "position": trade.position_type.value,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
        })

    return markers


def _prepare_equity_data(equity_curve: List[tuple]) -> str:
    """Prepare equity curve data as JSON string."""
    return json.dumps(_prepare_equity_data_list(equity_curve))


def _prepare_equity_data_list(equity_curve: List[tuple]) -> List[Dict]:
    """Prepare equity curve data as list for JSON export."""
    if not equity_curve:
        return []

    return [
        {"timestamp": ts.isoformat(), "equity": equity}
        for ts, equity in equity_curve
    ]


def _prepare_trades_table(trades: List[Trade]) -> List[Dict]:
    """Prepare trades data for HTML table."""
    return [
        {
            "id": trade.trade_id,
            "type": trade.position_type.value.upper(),
            "quantity": trade.quantity,
            "entry_time": trade.entry_timestamp.strftime("%Y-%m-%d %H:%M"),
            "exit_time": trade.exit_timestamp.strftime("%Y-%m-%d %H:%M"),
            "entry_price": f"${trade.entry_price:.2f}",
            "exit_price": f"${trade.exit_price:.2f}",
            "pnl": trade.pnl,
            "pnl_formatted": f"${trade.pnl:+,.2f}",
            "holding_mins": int(trade.holding_period_minutes),
        }
        for trade in trades
    ]


def _generate_html(
    title: str,
    candle_data: str,
    trade_markers: str,
    equity_data: str,
    trades_table: List[Dict],
    metrics: Dict[str, float],
    config: BacktestConfig,
    start_date: str,
    end_date: str,
    total_bars: int,
) -> str:
    """Generate the complete HTML report."""

    # Format metrics for display
    total_trades = int(metrics.get("total_trades", 0))
    win_rate = metrics.get("win_rate", 0) * 100
    total_pnl = metrics.get("total_pnl", 0)
    return_pct = metrics.get("return_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0) * 100
    profit_factor = metrics.get("profit_factor", 0)
    avg_win = metrics.get("avg_win", 0)
    avg_loss = metrics.get("avg_loss", 0)
    expectancy = metrics.get("expectancy", 0)
    risk_reward = metrics.get("risk_reward", 0)
    final_equity = metrics.get("final_equity", config.initial_capital)

    # Count winning/losing trades
    winning_trades = len([t for t in trades_table if t["pnl"] > 0])
    losing_trades = len([t for t in trades_table if t["pnl"] < 0])

    # Generate trades table HTML
    trades_rows = ""
    for t in trades_table:
        pnl_class = "positive" if t["pnl"] > 0 else "negative" if t["pnl"] < 0 else ""
        type_class = "call" if t["type"] == "CALL" else "put"
        trades_rows += f"""
        <tr>
            <td>{t["id"]}</td>
            <td class="{type_class}">{t["type"]}</td>
            <td>{t["quantity"]}</td>
            <td>{t["entry_time"]}</td>
            <td>{t["entry_price"]}</td>
            <td>{t["exit_time"]}</td>
            <td>{t["exit_price"]}</td>
            <td class="{pnl_class}">{t["pnl_formatted"]}</td>
            <td>{t["holding_mins"]} min</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #a1a1aa;
            font-size: 1.1rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}

        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .metric-label {{
            color: #a1a1aa;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .positive {{ color: #4ade80; }}
        .negative {{ color: #f87171; }}
        .call {{ color: #60a5fa; }}
        .put {{ color: #f472b6; }}

        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}

        .chart-title {{
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #e4e4e7;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}

        .section-title {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: #e4e4e7;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th {{
            background: rgba(255, 255, 255, 0.1);
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            color: #a1a1aa;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }}

        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .config-label {{
            color: #a1a1aa;
        }}

        .config-value {{
            font-weight: 600;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #71717a;
            font-size: 0.85rem;
        }}

        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 10px;
            font-size: 0.9rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-marker {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .legend-marker.call-entry {{ background: #60a5fa; }}
        .legend-marker.put-entry {{ background: #f472b6; }}
        .legend-marker.exit {{ background: #fbbf24; border: 2px solid #fff; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">{start_date} to {end_date} | {total_bars:,} bars | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if return_pct >= 0 else 'negative'}">{return_pct:+.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${final_equity:,.2f}</div>
                <div class="metric-label">Final Equity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sharpe:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">-{max_dd:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if expectancy >= 0 else 'negative'}">${expectancy:+,.2f}</div>
                <div class="metric-label">Expectancy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{risk_reward:.2f}</div>
                <div class="metric-label">Risk/Reward</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">${avg_win:,.2f}</div>
                <div class="metric-label">Avg Win</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">${avg_loss:,.2f}</div>
                <div class="metric-label">Avg Loss</div>
            </div>
        </div>

        <!-- Price Chart with Trades -->
        <div class="chart-container">
            <h2 class="chart-title">SPY Price Chart with Trade Markers</h2>
            <div id="priceChart" style="height: 500px;"></div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-marker call-entry"></div>
                    <span>CALL Entry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker put-entry"></div>
                    <span>PUT Entry</span>
                </div>
                <div class="legend-item">
                    <div class="legend-marker exit"></div>
                    <span>Exit</span>
                </div>
            </div>
        </div>

        <!-- Equity Curve -->
        <div class="chart-container">
            <h2 class="chart-title">Equity Curve</h2>
            <div id="equityChart" style="height: 300px;"></div>
        </div>

        <!-- Trades Table -->
        <div class="section">
            <h2 class="section-title">Trade History ({winning_trades} wins, {losing_trades} losses)</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Type</th>
                            <th>Qty</th>
                            <th>Entry Time</th>
                            <th>Entry Price</th>
                            <th>Exit Time</th>
                            <th>Exit Price</th>
                            <th>P&L</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Configuration -->
        <div class="section">
            <h2 class="section-title">Backtest Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <span class="config-label">Initial Capital</span>
                    <span class="config-value">${config.initial_capital:,.2f}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Max Position Size</span>
                    <span class="config-value">{config.max_position_size * 100:.1f}%</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Slippage</span>
                    <span class="config-value">{config.slippage_bps:.1f} bps</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Commission</span>
                    <span class="config-value">${config.commission_per_contract:.2f}/contract</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Bid-Ask Spread</span>
                    <span class="config-value">{config.spread_pct:.1f}%</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Options Multiplier</span>
                    <span class="config-value">{config.options_multiplier}x</span>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by Spymaster Backtest Engine</p>
        </footer>
    </div>

    <script>
        // Parse data
        const candleData = {candle_data};
        const tradeMarkers = {trade_markers};
        const equityData = {equity_data};

        // Create candlestick chart
        const candlestick = {{
            type: 'candlestick',
            x: candleData.timestamps,
            open: candleData.open,
            high: candleData.high,
            low: candleData.low,
            close: candleData.close,
            increasing: {{ line: {{ color: '#4ade80' }}, fillcolor: '#4ade80' }},
            decreasing: {{ line: {{ color: '#f87171' }}, fillcolor: '#f87171' }},
            name: 'SPY',
        }};

        // Separate entry and exit markers
        const callEntries = tradeMarkers.filter(m => m.type === 'entry' && m.position === 'call');
        const putEntries = tradeMarkers.filter(m => m.type === 'entry' && m.position === 'put');
        const exits = tradeMarkers.filter(m => m.type === 'exit');

        // For entries/exits, we need to map option prices to approximate SPY prices
        // Since we don't have SPY price stored, we'll use annotations instead

        const callEntryMarkers = {{
            type: 'scatter',
            mode: 'markers',
            x: callEntries.map(m => m.timestamp),
            y: callEntries.map(m => {{
                // Find closest candle
                const idx = candleData.timestamps.findIndex(t => t >= m.timestamp);
                return idx >= 0 ? candleData.low[idx] * 0.999 : null;
            }}),
            marker: {{
                symbol: 'triangle-up',
                size: 12,
                color: '#60a5fa',
            }},
            name: 'CALL Entry',
            text: callEntries.map(m => `CALL Entry: ${{m.price.toFixed(2)}} x${{m.quantity}}`),
            hoverinfo: 'text+x',
        }};

        const putEntryMarkers = {{
            type: 'scatter',
            mode: 'markers',
            x: putEntries.map(m => m.timestamp),
            y: putEntries.map(m => {{
                const idx = candleData.timestamps.findIndex(t => t >= m.timestamp);
                return idx >= 0 ? candleData.low[idx] * 0.999 : null;
            }}),
            marker: {{
                symbol: 'triangle-down',
                size: 12,
                color: '#f472b6',
            }},
            name: 'PUT Entry',
            text: putEntries.map(m => `PUT Entry: ${{m.price.toFixed(2)}} x${{m.quantity}}`),
            hoverinfo: 'text+x',
        }};

        const exitMarkers = {{
            type: 'scatter',
            mode: 'markers',
            x: exits.map(m => m.timestamp),
            y: exits.map(m => {{
                const idx = candleData.timestamps.findIndex(t => t >= m.timestamp);
                return idx >= 0 ? candleData.high[idx] * 1.001 : null;
            }}),
            marker: {{
                symbol: 'x',
                size: 10,
                color: '#fbbf24',
                line: {{ width: 2 }},
            }},
            name: 'Exit',
            text: exits.map(m => `Exit: ${{m.price.toFixed(2)}} | P&L: ${{m.pnl >= 0 ? '+' : ''}}${{m.pnl.toFixed(2)}}`),
            hoverinfo: 'text+x',
        }};

        const priceLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e4e4e7' }},
            xaxis: {{
                type: 'date',
                rangeslider: {{ visible: false }},
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true,
                fixedrange: false,
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true,
                title: 'Price ($)',
                fixedrange: false,
            }},
            margin: {{ l: 60, r: 20, t: 20, b: 40 }},
            showlegend: true,
            legend: {{
                orientation: 'h',
                y: -0.15,
                x: 0.5,
                xanchor: 'center',
            }},
            hovermode: 'x unified',
            dragmode: 'pan',
        }};

        // Store original ranges for reset
        let priceChartOriginalRanges = null;

        Plotly.newPlot('priceChart', [candlestick, callEntryMarkers, putEntryMarkers, exitMarkers], priceLayout, {{
            responsive: true,
            scrollZoom: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage'],
            modeBarButtonsToAdd: [{{
                name: 'Reset View',
                icon: {{
                    width: 24,
                    height: 24,
                    path: 'M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z'
                }},
                click: function(gd) {{
                    if (priceChartOriginalRanges) {{
                        Plotly.relayout(gd, priceChartOriginalRanges);
                    }} else {{
                        Plotly.relayout(gd, {{'xaxis.autorange': true, 'yaxis.autorange': true}});
                    }}
                }}
            }}],
            displaylogo: false,
        }});

        // Capture original ranges after initial render
        const priceChartEl = document.getElementById('priceChart');
        priceChartEl.on('plotly_afterplot', function() {{
            if (!priceChartOriginalRanges) {{
                const layout = priceChartEl.layout;
                priceChartOriginalRanges = {{
                    'xaxis.range': layout.xaxis.range ? [...layout.xaxis.range] : null,
                    'yaxis.range': layout.yaxis.range ? [...layout.yaxis.range] : null,
                    'xaxis.autorange': !layout.xaxis.range,
                    'yaxis.autorange': !layout.yaxis.range,
                }};
            }}
        }});

        // Create equity curve
        if (equityData.length > 0) {{
            const equityTrace = {{
                type: 'scatter',
                mode: 'lines',
                x: equityData.map(d => d.timestamp),
                y: equityData.map(d => d.equity),
                line: {{ color: '#60a5fa', width: 2 }},
                fill: 'tozeroy',
                fillcolor: 'rgba(96, 165, 250, 0.1)',
                name: 'Equity',
            }};

            const equityLayout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#e4e4e7' }},
                xaxis: {{
                    type: 'date',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true,
                    fixedrange: false,
                }},
                yaxis: {{
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true,
                    title: 'Equity ($)',
                    fixedrange: false,
                }},
                margin: {{ l: 60, r: 20, t: 20, b: 40 }},
                showlegend: false,
                dragmode: 'pan',
            }};

            // Store original ranges for reset
            let equityChartOriginalRanges = null;

            Plotly.newPlot('equityChart', [equityTrace], equityLayout, {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage'],
                modeBarButtonsToAdd: [{{
                    name: 'Reset View',
                    icon: {{
                        width: 24,
                        height: 24,
                        path: 'M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z'
                    }},
                    click: function(gd) {{
                        if (equityChartOriginalRanges) {{
                            Plotly.relayout(gd, equityChartOriginalRanges);
                        }} else {{
                            Plotly.relayout(gd, {{'xaxis.autorange': true, 'yaxis.autorange': true}});
                        }}
                    }}
                }}],
                displaylogo: false,
            }});

            // Capture original ranges after initial render
            const equityChartEl = document.getElementById('equityChart');
            equityChartEl.on('plotly_afterplot', function() {{
                if (!equityChartOriginalRanges) {{
                    const layout = equityChartEl.layout;
                    equityChartOriginalRanges = {{
                        'xaxis.range': layout.xaxis.range ? [...layout.xaxis.range] : null,
                        'yaxis.range': layout.yaxis.range ? [...layout.yaxis.range] : null,
                        'xaxis.autorange': !layout.xaxis.range,
                        'yaxis.autorange': !layout.yaxis.range,
                    }};
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    return html
