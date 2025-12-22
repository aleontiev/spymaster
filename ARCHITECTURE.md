# SpyMaster Architecture

## Overview

SpyMaster is an intraday trading bot for SPY 0DTE (Zero Days to Expiration) options, powered by **LeJEPA** (Latent-Euclidean Joint-Embedding Predictive Architecture). The system learns latent market dynamics through self-supervised learning, then uses these representations to drive a trading policy.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SPYMASTER TRADING SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA LAYER                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Polygon.io    │    │   Synthetic     │    │      MarketPatch            │  │
│  │   API Client    │    │   Generator     │    │      Processor              │  │
│  │                 │    │                 │    │                             │  │
│  │  • OHLCV bars   │    │  • GBM prices   │    │  • Log returns              │  │
│  │  • Options chain│    │  • Volume       │    │  • Volume normalization     │  │
│  │  • Tick data    │    │  • Bid/Ask      │    │  • Order book imbalance     │  │
│  └────────┬────────┘    └────────┬────────┘    │  • Time-to-close encoding   │  │
│           │                      │             │  • Cyclic time features     │  │
│           └──────────┬───────────┘             └──────────────┬──────────────┘  │
│                      │                                        │                  │
│                      ▼                                        │                  │
│           ┌─────────────────────┐                             │                  │
│           │  InMemoryDataset    │◄────────────────────────────┘                  │
│           │  (Pre-loaded RAM)   │                                                │
│           │                     │                                                │
│           │  • Context patches  │                                                │
│           │  • Target patches   │                                                │
│           │  • pin_memory=True  │                                                │
│           └──────────┬──────────┘                                                │
└──────────────────────┼───────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LeJEPA MODEL LAYER                                  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         Self-Supervised Training                         │    │
│  │                                                                          │    │
│  │   Context Patches                              Target Patches            │    │
│  │   [B, T, 11]                                   [B, T, 11]                │    │
│  │        │                                            │                    │    │
│  │        ▼                                            ▼                    │    │
│  │  ┌───────────────┐                         ┌───────────────┐             │    │
│  │  │    Context    │                         │    Target     │             │    │
│  │  │    Encoder    │────── EMA Update ──────▶│    Encoder    │             │    │
│  │  │     (f_θ)     │         (τ=0.996)       │     (f_ξ)     │             │    │
│  │  │               │                         │               │             │    │
│  │  │  Transformer  │                         │  Transformer  │             │    │
│  │  │  6 layers     │                         │  (frozen)     │             │    │
│  │  │  8 heads      │                         │  no gradient  │             │    │
│  │  └───────┬───────┘                         └───────┬───────┘             │    │
│  │          │                                         │                     │    │
│  │          ▼                                         ▼                     │    │
│  │   Context Embedding                         Target Embedding             │    │
│  │   [B, 512]                                  [B, 512]                     │    │
│  │          │                                         │                     │    │
│  │          ▼                                         │                     │    │
│  │  ┌───────────────┐                                 │                     │    │
│  │  │   Predictor   │                                 │                     │    │
│  │  │     (g_φ)     │                                 │                     │    │
│  │  │               │                                 │                     │    │
│  │  │  3-layer MLP  │                                 │                     │    │
│  │  │  2048 hidden  │                                 │                     │    │
│  │  └───────┬───────┘                                 │                     │    │
│  │          │                                         │                     │    │
│  │          ▼                                         │                     │    │
│  │   Predicted Embedding                              │                     │    │
│  │   [B, 512]                                         │                     │    │
│  │          │                                         │                     │    │
│  │          └────────────────┬────────────────────────┘                     │    │
│  │                           │                                              │    │
│  │                           ▼                                              │    │
│  │                 ┌───────────────────┐                                    │    │
│  │                 │    LeJEPA Loss    │                                    │    │
│  │                 │                   │                                    │    │
│  │                 │  L = L_pred + λ·L_SIGReg                               │    │
│  │                 │                   │                                    │    │
│  │                 │  L_pred = MSE(ŷ, y)                                    │    │
│  │                 │  L_SIGReg = ‖C - I‖²_F                                 │    │
│  │                 └───────────────────┘                                    │    │
│  │                                                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                       │
                       │ Frozen Embeddings
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              POLICY LAYER                                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         PolicyWithLeJEPA                                 │    │
│  │                                                                          │    │
│  │   Market Patches                                                         │    │
│  │   [B, T, 11]                                                             │    │
│  │        │                                                                 │    │
│  │        ▼                                                                 │    │
│  │  ┌───────────────┐                                                       │    │
│  │  │ Frozen LeJEPA │                                                       │    │
│  │  │ Context Enc.  │                                                       │    │
│  │  └───────┬───────┘                                                       │    │
│  │          │                                                               │    │
│  │          ▼                                                               │    │
│  │   Embeddings [B, 512]                                                    │    │
│  │          │                                                               │    │
│  │          ▼                                                               │    │
│  │  ┌───────────────┐                                                       │    │
│  │  │ Policy Network│                                                       │    │
│  │  │               │                                                       │    │
│  │  │  2-layer MLP  │                                                       │    │
│  │  │  256 hidden   │                                                       │    │
│  │  │  LayerNorm    │                                                       │    │
│  │  └───────┬───────┘                                                       │    │
│  │          │                                                               │    │
│  │          ├────────────────┬────────────────┐                             │    │
│  │          ▼                ▼                ▼                             │    │
│  │   ┌───────────┐    ┌───────────┐    ┌───────────┐                        │    │
│  │   │  Action   │    │   Value   │    │ Position  │                        │    │
│  │   │   Head    │    │   Head    │    │   Size    │                        │    │
│  │   │           │    │           │    │   Head    │                        │    │
│  │   │ softmax   │    │  scalar   │    │  sigmoid  │                        │    │
│  │   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                        │    │
│  │         │                │                │                              │    │
│  │         ▼                ▼                ▼                              │    │
│  │    Action Probs      Value Est.      Position %                          │    │
│  │    [HOLD, BUY_CALL,  V(s)            [0, 1]                              │    │
│  │     BUY_PUT, CLOSE]                                                      │    │
│  │                                                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTION LAYER                                       │
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  Alpaca Client  │    │  Risk Manager   │    │    Real-Time Orchestrator   │  │
│  │                 │    │                 │    │                             │  │
│  │  • Limit orders │    │  • Position     │    │  • Async event loop         │  │
│  │  • Mid-price    │    │    limits       │    │  • Tick buffer → patch      │  │
│  │    pegging      │    │  • Loss stops   │    │  • Compiled inference       │  │
│  │  • No market    │    │  • Theta decay  │    │  • Action dispatch          │  │
│  │    orders       │    │    awareness    │    │                             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer (`src/data/`)

| Component | File | Description |
|-----------|------|-------------|
| **PolygonClient** | `polygon.py` | REST API client for historical OHLCV, options chains, and tick data |
| **SyntheticMarketData** | `synthetic.py` | Generates realistic SPY-like data using geometric Brownian motion |
| **MarketPatch** | `processing.py` | Transforms raw OHLCV into feature tensors with 11 dimensions |
| **InMemoryDataset** | `processing.py` | Pre-loads all patches into RAM for fast training |

#### Feature Engineering (11 dimensions per timestep)

```
┌────────────────────────────────────────────────────────┐
│  Feature Vector [T, 11]                                │
├────────────────────────────────────────────────────────┤
│  [0-4]  OHLCV: Open, High, Low, Close, Volume          │
│  [5]    Log Return: log(close_t / close_{t-1})         │
│  [6]    Normalized Volume: z-score over rolling window │
│  [7]    Order Book Imbalance: (bid-ask)/(bid+ask)      │
│  [8]    Time to Close: hours until 4PM (normalized)    │
│  [9]    Time Sin: sin(2π × fraction_of_day)            │
│  [10]   Time Cos: cos(2π × fraction_of_day)            │
└────────────────────────────────────────────────────────┘
```

### 2. LeJEPA Model Layer (`src/model/`)

| Component | File | Description |
|-----------|------|-------------|
| **ContextEncoder** | `encoders.py` | Causal Transformer (GPT-style) processing past patches |
| **TargetEncoder** | `encoders.py` | EMA copy of context encoder (momentum τ=0.996) |
| **LatentPredictor** | `predictor.py` | MLP predicting target embedding from context |
| **LeJEPALoss** | `loss.py` | Combined prediction loss + SIGReg regularization |
| **LeJEPA** | `lejepa.py` | Complete model wrapper with training/inference methods |

#### LeJEPA vs I-JEPA

The key differentiator is **SIGReg** (Sketched Isotropic Gaussian Regularization):

```
L_total = L_pred + λ × L_SIGReg

Where:
  L_pred = ‖ŷ - y‖²₂           (MSE between predicted and target embeddings)
  L_SIGReg = ‖C - I‖²_F        (Frobenius norm of covariance deviation from identity)
  C = (1/B) × Z^T × Z          (Empirical covariance of embeddings)
```

This regularization prevents representation collapse by forcing embeddings toward an isotropic Gaussian distribution.

### 3. Policy Layer (`src/strategy/`)

| Component | File | Description |
|-----------|------|-------------|
| **PolicyNetwork** | `policy.py` | MLP head outputting action probabilities |
| **PolicyWithLeJEPA** | `policy.py` | Combined frozen encoder + trainable policy |
| **TradingAction** | `policy.py` | Action enum: HOLD, BUY_CALL, BUY_PUT, CLOSE |

#### Action Space

```
┌─────────────────────────────────────────────┐
│  TradingAction Enum                         │
├─────────────────────────────────────────────┤
│  0: HOLD           - No action              │
│  1: BUY_CALL       - Buy call option        │
│  2: BUY_PUT        - Buy put option         │
│  3: CLOSE_POSITION - Exit current position  │
└─────────────────────────────────────────────┘
```

### 4. Execution Layer (`src/execution/`)

| Component | File | Description |
|-----------|------|-------------|
| **AlpacaClient** | `alpaca_client.py` | Order submission with limit orders only |
| **RiskManager** | `alpaca_client.py` | Position limits, loss stops, theta awareness |

### 5. Backtest Layer (`src/backtest/`)

| Component | File | Description |
|-----------|------|-------------|
| **SimulatorEngine** | `engine.py` | Mock Alpaca API with realistic slippage |
| **StrategyRunner** | `engine.py` | Runs policy on historical data |

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

Phase 1: Self-Supervised Pre-training (train_jepa.py)
─────────────────────────────────────────────────────
    ┌──────────────┐
    │ Raw Market   │
    │ Data         │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────┐
    │ Context      │     │ Target       │
    │ Patches      │     │ Patches      │
    │ (past)       │     │ (future)     │
    └──────┬───────┘     └──────┬───────┘
           │                    │
           ▼                    ▼
    ┌──────────────┐     ┌──────────────┐
    │ Context      │────▶│ Target       │
    │ Encoder      │ EMA │ Encoder      │
    └──────┬───────┘     └──────┬───────┘
           │                    │
           ▼                    │
    ┌──────────────┐            │
    │ Predictor    │            │
    └──────┬───────┘            │
           │                    │
           ▼                    ▼
    ┌──────────────────────────────────┐
    │ LeJEPA Loss                      │
    │ (MSE + SIGReg)                   │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────┐
    │ AdamW +      │
    │ Cosine LR    │
    │ BFloat16 AMP │
    └──────────────┘

Phase 2: Policy Training (train_policy.py)
──────────────────────────────────────────
    ┌──────────────┐
    │ Market       │
    │ Patches      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ FROZEN       │
    │ LeJEPA       │
    │ Encoder      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Embeddings   │
    │ (cached)     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────┐
    │ Policy       │────▶│ Actions +    │
    │ Network      │     │ Rewards      │
    └──────┬───────┘     └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Policy       │
    │ Gradient     │
    │ (PPO/A2C)    │
    └──────────────┘
```

## Hardware Optimization (RTX 4090 + 64GB RAM)

| Optimization | Implementation |
|--------------|----------------|
| **Precision** | BFloat16 via `torch.amp.autocast` |
| **Compilation** | `torch.compile(mode="reduce-overhead")` |
| **Data Loading** | In-memory dataset, `pin_memory=True`, `num_workers=0` |
| **Batch Size** | 2048-4096 (critical for SIGReg stability) |
| **Memory** | All patches pre-loaded into 64GB RAM |

## File Structure

```
spymaster/
├── src/
│   ├── data/
│   │   ├── polygon.py        # Polygon.io API client
│   │   ├── synthetic.py      # Synthetic data generator
│   │   └── processing.py     # MarketPatch & InMemoryDataset
│   ├── model/
│   │   ├── encoders.py       # Context & Target encoders
│   │   ├── predictor.py      # Latent predictor MLP
│   │   ├── loss.py           # Prediction + SIGReg losses
│   │   └── lejepa.py         # Complete LeJEPA model
│   ├── strategy/
│   │   └── policy.py         # Policy network
│   ├── execution/
│   │   └── alpaca_client.py  # Order execution (TODO)
│   └── backtest/
│       └── engine.py         # Backtesting engine (TODO)
├── train_jepa.py             # Self-supervised training
├── train_policy.py           # Policy training (TODO)
├── visualize_embeddings.py   # PCA visualization
├── checkpoints/              # Model checkpoints
└── visualizations/           # PCA plots
```

## Key Design Decisions

1. **LeJEPA over I-JEPA**: SIGReg regularization prevents representation collapse without requiring careful masking strategies.

2. **Frozen Encoder for Policy**: Separates representation learning from policy learning, allowing stable embeddings.

3. **Time-to-Close Feature**: Critical for 0DTE options where theta decay accelerates exponentially in final hours.

4. **Limit Orders Only**: Market orders on options have severe slippage; always peg to mid-price.

5. **Large Batch Sizes**: Required for stable covariance estimation in SIGReg (batch_size >= embedding_dim).
