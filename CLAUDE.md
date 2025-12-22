# Agent Guidelines (CLAUDE.md, GEMINI.md, CODEX.md, etc)

## 1. Project Mission

Build a sophisticated **intraday trading bot** for **SPY 0DTE (Zero Days to Expiration) options**.

- **Core Technology:** LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture)  
- **Goal:** Predict latent market dynamics to inform a buy/sell policy for maximizing profit while managing Theta decay  
- **Stack:** Python, PyTorch (Model), Polygon.io (Data), Alpaca (Execution)  
- **Hardware Profile:** NVIDIA RTX 4090 (24GB VRAM) + 64GB System RAM  

---

## 2. Coding Standards & Style

### Type Hinting
Mandatory for all functions.  
Example:  
```python
def train(loader: DataLoader) -> float:
````

### Docstrings

Use docstrings for classes and complex methods.

### Modularity

Enforce strict separation of concerns:

```
src/data       → Ingestion and "Patching"
src/model      → LeJEPA architecture and Loss
src/strategy   → Policy heads (RL/Classification) and logic
src/execution  → Async API wrappers for Alpaca/Polygon
```

### Error Handling

* **Data feeds:** Fail gracefully
* **Trading execution:** Fail fast (safety first)

### Testing

* Unit tests for model shapes and logic
* Integration tests for API wrapper correctness

---

## 3. Technical Specifications: LeJEPA

### Key Note

We implement **LeJEPA**, not I-JEPA.
Differentiator: **Regularization term (SIGReg)**.

---

### ### Architecture Overview

#### **1. Context Encoder** ( (f_\theta) \ )

A causal Transformer (GPT-style blocks) processing past market patches.

#### **2. Target Encoder** ( (f_\xi) \ )

EMA copy of the context encoder:

[
\xi \leftarrow \tau\xi + (1-\tau)\theta
]

#### **3. Predictor** ( (g_\phi) \ )

Lightweight MLP or Transformer predicting the embedding of a *future* patch.

---

### Loss Function (Defining “Le” in LeJEPA)

Total Loss:

[
L = L_{pred} + \lambda L_{SIGReg}
]

#### **1. Prediction Loss (L_{pred})**

L2 distance between predicted and target embeddings:

[
L_{pred} = ||\hat{y} - y||_2^2
]

#### **2. SIGReg (Sketched Isotropic Gaussian Regularization)**

Forces embeddings toward an isotropic Gaussian distribution.

* Covariance (exact, no sketching unless dim > 4096):

[
C = \frac{1}{B} Z^T Z
]

* Regularization loss:

[
L_{SIGReg} = ||C - I||_F^2
]

---

## 4. Hardware Optimizations (RTX 4090 / 64GB RAM)

### Precision

Use **BFloat16 (torch.bfloat16)** to avoid gradient underflow.

### Compilation

Use:

`torch.compile(model, mode="reduce-overhead")`

### Data Loading

* In-memory dataset (load all tensors into RAM)
* num_workers=0 or persistent_workers=True
* pin_memory=True
* Batch size: **2048–4096** (critical for SIGReg stability)

## 5. Financial Domain Knowledge

### 0DTE Nuance

Theta decay accelerates **exponentially** in the final 2 hours.
**Time-to-close** must be a feature in every patch.

### Data Structure

* Use **log-returns** or **z-scored windows**, NOT raw prices
* Include **Order Book Imbalance:**
  [
  \frac{\text{Bid Vol - Ask Vol}}{\text{Bid Vol + Ask Vol}}
  ]
* Include **Options Flow:** Put/Call ratios (Polygon)

### Execution

* **Never use market orders** for options
* Use **limit orders** pegged to mid-price + slippage tolerance

## 6. Development Tools & Commands

### Using `uv`

**IMPORTANT:** Always use `uv run` to execute Python commands, pytest, or any Python scripts.

Examples:
```bash
# Running Python scripts
uv run python src/data/synthetic.py

# Running pytest
uv run pytest tests/

# Running specific test files
uv run pytest tests/test_model.py -v

# Running the test runner
uv run python run_tests.py
```

**Never** use:
- `python` or `python3` directly
- `pytest` directly
- `pip install` (use `uv add` or `uv sync` instead)

## 7. Agent Interaction Rules

* Check **PLAN.md** before starting any tasks, update PLAN.md when completing tasks (check off completed tasks)
* If APIs change (Polygon/Alpaca), check docs before concluding failure.
* **Always use `uv run` for all Python commands** (see section 6)
