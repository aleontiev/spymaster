# Spymaster - 0DTE SPY Options Trading Bot

Intraday trading bot for SPY 0DTE (Zero Days to Expiration) options using LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture).

## Setup

```bash
# Install dependencies
uv sync
```

## Data Download

Download historical SPY data from Polygon.io flat files:

```bash
# Download stocks data (2020-2025)
uv run python scripts/download_stocks_flatfiles.py --from-year 2020 --to-year 2025 --workers 4

# Download options data (2022-2025)
uv run python scripts/download_options_flatfiles.py --from-year 2022 --to-year 2025 --workers 4
```

## Training

### 1. Train LeJEPA (Self-Supervised Representation Learning)

Train the LeJEPA model on combined stocks + options data:

```bash
uv run python train_jepa.py \
  --data_dir data/polygon/stocks \
  --options_dir data/polygon/options \
  --pattern "SPY_202*.parquet" \
  --epochs 100 \
  --batch_size 2048 \
  --embedding_dim 512 \
  --num_heads 8 \
  --num_layers 6 \
  --ff_dim 2048 \
  --lr 1e-4 \
  --warmup_epochs 10 \
  --save_every 10 \
  --log_every 50 \
  --compile
```

Key parameters:
- `--batch_size 2048`: Required for SIGReg stability (must be >= embedding_dim)
- `--compile`: Enable torch.compile() for faster training
- `--pattern`: Glob pattern for parquet files

Output: `checkpoints/lejepa_best.pt`

### 2. Train Policy Network (Trading Decisions)

Train the policy network on top of frozen LeJEPA embeddings:

```bash
uv run python train_policy.py \
  --lejepa_checkpoint checkpoints/lejepa_best.pt \
  --data_dir data/polygon/stocks \
  --options_dir data/polygon/options \
  --pattern "SPY_202*.parquet" \
  --patch_length 32 \
  --stride 8 \
  --epochs 50 \
  --batch_size 256 \
  --policy_hidden_dim 256 \
  --policy_layers 3 \
  --lr 1e-4 \
  --lookahead 10 \
  --threshold_pct 0.1 \
  --save_every 10 \
  --log_every 10
```

Key parameters:
- `--lejepa_checkpoint`: Path to pre-trained LeJEPA model
- `--lookahead 10`: Labels based on price 10 minutes in the future
- `--threshold_pct 0.1`: 0.1% move threshold for BUY_CALL/BUY_PUT vs HOLD
- `--patch_length` and `--stride`: Must match LeJEPA training for cache reuse

Output: `checkpoints/policy_best.pt`

### 3. Backtest Strategy

Run the trained policy on historical data:

```bash
uv run python -m src.backtest.runner \
  --lejepa checkpoints/lejepa_best.pt \
  --policy checkpoints/policy_best.pt \
  --start_date 2024-10-01 \
  --end_date 2024-12-01 \
  --initial_capital 100000 \
  --max_position_pct 0.05 \
  --force_trades \
  --verbose
```

#### Backtest Parameters

**Required:**
- `--lejepa`: Path to LeJEPA checkpoint
- `--policy`: Path to policy checkpoint

**Data Source:**
- `--data_dir`: Directory with stocks parquet files (default: `data/polygon/stocks`)
- `--options_dir`: Directory with options parquet files (default: `data/polygon/options`)
- `--pattern`: Glob pattern for parquet files (default: `SPY_*.parquet`)
- `--start_date`: Start date for backtest in YYYY-MM-DD format
- `--end_date`: End date for backtest in YYYY-MM-DD format

**Capital & Position Sizing:**
- `--initial_capital`: Starting capital for simulation (default: `100000`)
- `--max_position_pct`: Maximum position size as fraction of capital (default: `0.05`)

**Trading Behavior:**
- `--confidence_threshold`: Minimum confidence to execute non-HOLD action (default: `0.0`)
- `--force_trades`: Override HOLD predictions when BUY signal exceeds min_signal_threshold
- `--action_cooldown`: Minutes to wait between actions (default: `5`)
- `--min_signal_threshold`: Minimum probability for non-HOLD action when using --force_trades (default: `0.20`)
- `--stochastic`: Use probabilistic action selection instead of argmax

**Output:**
- `--verbose`: Show individual trades, position opens/closes, and daily details
- `--report`: Path to save HTML report (default: `reports/backtest_YYYYMMDD_YYYYMMDD.html`)
- `--no_report`: Disable HTML report generation
- `--seed`: Random seed for reproducibility (default: `42`)
- `--device`: Device to run on (`auto`, `cuda`, or `cpu`)

#### Example: Backtest Specific Date Range

```bash
uv run python -m src.backtest.runner \
  --lejepa checkpoints/lejepa_best.pt \
  --policy checkpoints/policy_best.pt \
  --start_date 2024-11-01 \
  --end_date 2024-12-01 \
  --force_trades \
  --verbose
```

#### Backtest Output

The backtest produces:
- **Daily Summary**: SPY open/close prices, account equity, return %, trades per day
- **Trade Log** (with `--verbose`): Entry/exit prices, P&L, holding period for each trade
- **Policy Predictions**: Distribution of HOLD/BUY_CALL/BUY_PUT/CLOSE actions
- **Performance Metrics**:
  - Total P&L and Return %
  - Win Rate and Expectancy (expected $ per trade)
  - Risk/Reward Ratio (avg win / avg loss)
  - Sharpe Ratio (annualized)
  - Max Drawdown
  - Profit Factor (gross profit / gross loss)

#### Interactive HTML Report

By default, the backtest generates an interactive HTML report with:
- **Zoomable Candlestick Chart**: SPY price with trade entry/exit markers
  - Blue triangles: CALL entries
  - Pink triangles: PUT entries
  - Yellow X markers: Exits with P&L hover info
- **Equity Curve**: Visual representation of account value over time
- **Trade History Table**: Sortable table of all trades with P&L
- **Performance Dashboard**: Key metrics displayed in an easy-to-read grid
- **Configuration Summary**: Backtest parameters used

Reports are saved to `reports/` directory by default. Open in any browser to view.

#### Notes

- Backtest automatically filters to regular market hours (9:30 AM - 4:00 PM ET)
- All positions are closed at end of day (0DTE options expire at close)
- No trading in final 30 minutes (theta decay too aggressive)
- Includes realistic simulation: 10 bps slippage, $0.65/contract commission, bid-ask spreads

## Architecture

```
src/
├── data/           # Data loading and preprocessing
│   ├── combined_loader.py   # Stocks + options combined loader
│   ├── parquet_loader.py    # Stocks-only parquet loader
│   └── processing.py        # MarketPatch feature engineering
├── model/          # LeJEPA architecture
│   ├── lejepa.py            # Main LeJEPA model
│   ├── encoders.py          # Context/Target encoders
│   ├── predictor.py         # Latent predictor
│   └── sigreg.py            # SIGReg regularization
├── strategy/       # Trading strategy
│   └── policy.py            # Policy network
└── execution/      # Trade execution (Alpaca)
```

## Hardware Requirements

- GPU: NVIDIA RTX 4090 (24GB VRAM) recommended
- RAM: 64GB system RAM for in-memory dataset
- Storage: ~50GB for historical data

## References

- LeJEPA: Joint-Embedding Predictive Architecture with SIGReg regularization
- Polygon.io: Market data provider
- Alpaca: Brokerage API for execution
