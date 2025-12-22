# Implementation Plan

## **Phase 1: Infrastructure & Data Pipeline**

### **1.1 Environment Setup**
- [x] Create `pyproject.toml` with dependencies (torch, numpy, pandas, polygon-api-client, alpaca-trade-api, pydantic) for use with `uv`.
- [x] Check CUDA availability and Flash Attention support:
  ```python
  import torch
  torch.backends.cuda.flash_sdp_enabled()
````

* [x] Create `.env` template for API Keys.

### **1.2 Polygon.io Data Client (`src/data/polygon.py`)**

* [x] Implement `fetch_historical_aggregates` (OHLCV for SPY).
* [x] Implement `fetch_options_chain` (Snapshots for 0DTE chains).
* [x] Implement `fetch_trades_quotes` (High-frequency tick data).

### **1.3 Data Processing & In-Memory Loading (`src/data/processing.py`)**

* [x] Implement `MarketPatch` class: Converts time-series window to tensor.
* [x] Feature Engineering: Log-returns, Normalized Volume, OB Imbalance, Time-encoding.
* [x] Optimization: Implement `InMemoryDataset` that pre-loads all tensors into CPU RAM.
* [x] Implement `DataLoader` with `pin_memory=True` and large batch support.

## **Phase 2: LeJEPA Model Architecture**

### **2.1 Encoders (`src/model/encoders.py`)**

* [x] Implement **ContextEncoder** (Transformer, `batch_first=True`).
* [x] Implement **TargetEncoder** (EMA wrapper).

### **2.2 Predictor (`src/model/predictor.py`)**

* [x] Implement **LatentPredictor** (MLP or Transformer head).

### **2.3 Loss Functions (`src/model/loss.py`)**

* [x] Implement **PredictionLoss** (MSE).
* [x] **CRITICAL:** Implement **SIGRegLoss** using *exact covariance* (no sketching).
* [x] Ensure operations support **BFloat16** (avoid FP16 overflow in matmul).

### **2.4 Model Wrapper (`src/model/lejepa.py`)**

* [x] Create **LeJEPA** class.
* [x] Add `.compile()` method to wrap submodules with `torch.compile`.

## **Phase 3: Self-Supervised Training**

### **3.1 Training Loop (`train_jepa.py`)**

* [x] Setup `torch.amp` with `dtype=torch.bfloat16`.
* [x] Initialize AdamW optimizers & LR schedulers (warmup + cosine decay).
* [x] **Training Step:**

  * Context Forward (Masked).
  * Target Forward (Unmasked, No Grad).
  * Predict & Calculate Loss.
  * Scaled Backprop (GradScaler optional for BF16).
  * EMA Update.
* [x] Implement checkpointing (full config save/load).

### **3.2 Validation & Visualization (`visualize_embeddings.py`)**

* [x] PCA visualization of embeddings (detect collapse).
* [x] Embedding statistics: effective rank, dimension variance, cosine similarity.
* [x] Track loss curves (train/val logged per epoch).

## **Phase 4: Downstream Policy (The Trader)**

### **4.1 Policy Network (`src/model/policy.py`)**

* [x] Lightweight MLP head on top of **Frozen LeJEPA**.
* [x] Output action probabilities (HOLD, BUY_CALL, BUY_PUT, CLOSE).
* [x] Optional value head for actor-critic RL.
* [x] Optional position sizing head.

### **4.2 Training (`train_policy.py`)**

* [x] Train policy with frozen embeddings.
* [x] Use large batches (embeddings cached/pre-extracted).
* [x] Synthetic label generation from future returns.
* [x] Per-class accuracy tracking.

## **Phase 5: Backtesting & Simulation**

### **5.1 Simulator Engine (`src/backtest/engine.py`)**

* [x] Mock order execution with realistic slippage.
* [x] Options pricing model with theta decay (accelerates near expiry).
* [x] Position tracking and P&L calculation.
* [x] Bid-ask spread simulation.

### **5.2 Strategy Runner (`src/backtest/runner.py`)**

* [x] Connect LeJEPA + Policy to backtest engine.
* [x] Run policy on test data with action cooldowns.
* [x] Compute Sharpe, Drawdown, Win Rate, Profit Factor.
* [x] Per-day trading with 0DTE expiry handling.


## **Phase 6: Live Execution Framework**

### **6.1 Alpaca Wrapper (`src/execution/alpaca_client.py`)**

* [x] Implement `submit_order` (limit orders only).
* [x] Implement Risk Manager with stop-loss/take-profit.
* [x] Position tracking and account monitoring.
* [x] OCC option symbol formatting.

### **6.2 Real-Time Orchestrator (`main.py`)**

* [x] Async event loop using `asyncio`.
* [x] Real-time inference pipeline:
  * Buffer ticks → patch.
  * LeJEPA inference (compilable).
  * Policy → action.
* [x] Market hours detection.
* [x] Kill switch and graceful shutdown.
* [x] Dry run mode for testing.

## **Phase 7: Deployment & Monitoring**

### **7.1 Logging & Safety**

* [x] Structured logging (e.g., JSON logs, log rotation).
* [x] Kill switch (SIGUSR1 signal handler).
* [x] Paper-trading dry run mode.
* [ ] Prometheus/Grafana metrics (optional).

## **Phase 8: GEX Flow Engine (ThetaData Integration)**

### **8.1 ThetaData Client Extensions (`src/data/thetadata_client.py`)**

* [x] Implement `fetch_greeks_for_date()` - 1-minute Greeks from ThetaData V3
* [x] Implement `fetch_trade_quotes()` - Combined trade+quote data with start_time/end_time
* [x] Implement `fetch_trade_quotes_parallel()` - Parallelized 1-minute chunk fetching

### **8.2 Lee-Ready Trade Classifier (`src/data/gex_flow_engine.py`)**

* [x] Implement `LeeReadyClassifier` with quote rule (bid/ask) + tick rule fallback
* [x] Vectorized trade classification: sign = -1 (buy aggressor), +1 (sell aggressor)

### **8.3 GEX Flow Engine Core (`src/data/gex_flow_engine.py`)**

* [x] Implement `GEXFlowEngine` class with 10 flow-based features:
  - net_gamma_flow: Σ(Sign × Vol × Γ × S × 100)
  - dist_to_zero_gex: Current_Price - Zero_GEX_Price
  - cumulative_net_gex: Running sum of net_gamma_flow
  - dist_to_pos_gex_wall: Distance to max positive GEX strike
  - dist_to_neg_gex_wall: Distance to max negative GEX strike
  - net_delta_flow: Σ(Sign × Vol × Δ × S × 100)
  - anchored_vwap_z: (Price - VWAP_Day) / StdDev_Intraday
  - gamma_sentiment_ratio: Net_Gamma_Flow / Σ|Gamma_Flow|
  - vwap_divergence: ln(VWAP_30m) - ln(VWAP_Day)
  - dist_to_zero_dex: Current_Price - Zero_DEX_Price
* [x] Trade aggregation to 1-minute intervals
* [x] GEX/DEX wall detection algorithms

### **8.4 Historical Download Script (`scripts/download_gex_flow.py`)**

* [x] Create download script with date range support
* [x] Concurrent API requests with asyncio.Semaphore
* [x] Resume capability with --skip-existing flag
* [x] Output: `data/gex_flow/SPY_thetadata_1m_combined_YYYY-MM-DD.parquet`

### **8.5 Loader Integration (`src/data/loader.py`)**

* [x] Add `load_gex_flow_for_date()` function
* [x] Update `load_combined_day()` with `use_gex_flow` parameter
* [x] Add normalization for new GEX flow features:
  - Flow features: signed log transform
  - Distance features: normalize to % of spot, clip to [-5, 5]
  - Z-score features: clip extremes
  - Ratio features: already bounded [-1, 1]
  - Divergence: scale and clip

### **8.6 Feature Processing (`src/data/processing.py`)**

* [x] Add new feature types to OPTIONS_FEATURE_SPEC
* [x] Implement normalization handlers for gex_flow, dex_flow, zscore, divergence types

### **8.7 Testing**

* [x] Unit tests for LeeReadyClassifier (6 tests)
* [x] Unit tests for GEXFlowEngine (4 tests)
* [x] Integration tests for feature computation (3 tests)

