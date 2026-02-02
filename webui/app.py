#!/usr/bin/env python3
"""
Spymaster Data Browser - Web UI for browsing checkpoints, parquet files, and reports.

Usage:
    uv run python webui/app.py

Then open http://localhost:8050 in your browser.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DATA_DIR = BASE_DIR / "data" / "checkpoints"  # Actual .pt files
REPORTS_DIR = BASE_DIR / "reports"

# Add project root to path for imports
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, template_folder="templates", static_folder="static")


# =============================================================================
# Registry Helper
# =============================================================================

_registry = None

def get_registry():
    """Get or create dataset registry instance."""
    global _registry
    if _registry is None:
        from src.data.dag.registry import DatasetRegistry
        _registry = DatasetRegistry()
    return _registry


def get_dataset_path(dataset: str, underlying: str) -> Path:
    """Get the base path for a dataset and underlying using the registry."""
    registry = get_registry()
    if not registry.exists(dataset):
        return None
    config = registry.get(dataset)
    # Extract base path from pattern (remove date placeholders)
    pattern = config.path_pattern
    # Pattern like "data/stocks-1m/{underlying}/{date:%Y-%m}/{date:%d}.parquet"
    # We want "data/stocks-1m/{underlying}"
    base_pattern = pattern.split("/{date:")[0] if "/{date:" in pattern else pattern.rsplit("/", 1)[0]
    base_path = Path(base_pattern.replace("{underlying}", underlying))
    return base_path


def is_daily_dataset(dataset: str) -> bool:
    """Check if a dataset has daily (not intraday) granularity."""
    registry = get_registry()
    if not registry.exists(dataset):
        return "-1d" in dataset or "-day" in dataset
    config = registry.get(dataset)
    return config.granularity == "yearly" or "-1d" in dataset or "-day" in dataset


def is_intraday_dataset(dataset: str) -> bool:
    """Check if a dataset has intraday (1m) granularity."""
    return "-1m" in dataset and not is_daily_dataset(dataset)


def get_time_resolution(dataset: str) -> str:
    """Get human-readable time resolution for a dataset."""
    registry = get_registry()
    if registry.exists(dataset):
        return registry.get_time_resolution(dataset)
    if "-1m" in dataset:
        return "1 minute"
    if "-1d" in dataset or "-day" in dataset:
        return "daily"
    return "unknown"


# =============================================================================
# API Routes
# =============================================================================

@app.route("/api/checkpoints")
def api_checkpoints():
    """List all checkpoints with their configs from database."""
    from src.managers.checkpoint_manager import get_manager

    mgr = get_manager()
    checkpoints = []

    for info in mgr.list_all():
        # Build config dict from CheckpointConfig
        config = info.config.to_dict() if info.config else None

        # List .pt files from data/checkpoints/ dir
        files = []
        for f in info.files:
            if f.is_file():
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        checkpoints.append({
            "name": info.name,
            "status": info.status,
            "config": config,
            "files": sorted(files, key=lambda x: x["name"]),
            "total_size": info.total_size,
        })

    return jsonify(checkpoints)


@app.route("/api/checkpoint/<name>")
def api_checkpoint_detail(name: str):
    """Get detailed info about a specific checkpoint from database."""
    from src.managers.checkpoint_manager import get_manager

    mgr = get_manager()
    info = mgr.get(name)

    if not info:
        return jsonify({"error": "Checkpoint not found"}), 404

    # Build config dict from CheckpointConfig
    config = info.config.to_dict() if info.config else None

    # List files
    files = []
    for f in info.files:
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    return jsonify({
        "name": info.name,
        "status": info.status,
        "config": config,
        "files": sorted(files, key=lambda x: x["name"]),
        "total_size": info.total_size,
    })


@app.route("/api/data/summary")
def api_data_summary():
    """Get summary of data structure with file counts and date ranges."""
    import re

    def get_date_range(files: List[Path], pattern: str) -> Optional[Dict]:
        """Extract date range from files matching pattern."""
        dates = []
        date_regex = re.compile(r"(\d{4}-\d{2}-\d{2})")
        for f in files:
            match = date_regex.search(f.name)
            if match:
                dates.append(match.group(1))
        if dates:
            dates.sort()
            return {"start": dates[0], "end": dates[-1], "count": len(dates)}
        return None

    summary = {"categories": [], "total_files": 0, "total_size": 0}

    # Define categories to summarize
    categories = [
        ("stocks", "STOCKS-1M", "1-minute stock aggregates"),
        ("options", "OPTIONS-1M", "1-minute options aggregates"),
        ("oi", "OI-0DTE", "0DTE open interest data"),
        ("cache", "Combined", "Cached combined data"),
    ]

    for dir_name, data_type, description in categories:
        dir_path = DATA_DIR / dir_name
        if not dir_path.exists():
            continue

        files = list(dir_path.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        date_range = get_date_range(files, data_type)

        cat_info = {
            "name": dir_name,
            "data_type": data_type,
            "description": description,
            "file_count": len(files),
            "total_size": total_size,
            "date_range": date_range,
        }
        summary["categories"].append(cat_info)
        summary["total_files"] += len(files)
        summary["total_size"] += total_size

    # Check for legacy polygon directory
    legacy_dir = DATA_DIR / "polygon"
    if legacy_dir.exists():
        legacy_files = list(legacy_dir.rglob("*.parquet"))
        if legacy_files:
            summary["legacy"] = {
                "path": "data/polygon",
                "file_count": len(legacy_files),
                "total_size": sum(f.stat().st_size for f in legacy_files),
                "note": "Legacy data - can be removed after verifying migration",
            }

    return jsonify(summary)


@app.route("/api/pipeline")
def api_pipeline():
    """Get pipeline data from database for DAG visualization."""
    from src.db.database import get_db

    try:
        db = get_db()
        datasets = db.list_datasets()

        # Build node and edge data
        nodes = []
        edges = []

        # Track dependents for each dataset
        dependents_map = {}
        for ds in datasets:
            for dep in ds.dependencies:
                if dep.dependency_name not in dependents_map:
                    dependents_map[dep.dependency_name] = []
                dependents_map[dep.dependency_name].append(ds.name)

        for ds in datasets:
            # Build fields dict
            fields = {}
            for f in ds.fields:
                fields[f.name] = {
                    "type": f.type,
                    "description": f.description or "",
                    "source": f.source,
                    "unit": f.unit,
                }

            # Build dependencies list
            dependencies = []
            for dep in ds.dependencies:
                dependencies.append({
                    "name": dep.dependency_name,
                    "relation": dep.relation,
                    "days": dep.days,
                    "include_current": dep.include_current,
                    "required": dep.required,
                })
                # Add edge
                edges.append({
                    "from": dep.dependency_name,
                    "to": ds.name,
                    "relation": dep.relation,
                    "days": dep.days,
                })

            node = {
                "name": ds.name,
                "type": ds.type,
                "description": ds.description or "",
                "provider": ds.provider,
                "path_pattern": ds.path_pattern,
                "granularity": ds.granularity,
                "ephemeral": ds.ephemeral,
                "computation": ds.computation,
                "dependencies": [d["name"] for d in dependencies],
                "dependency_specs": dependencies,
                "dependents": dependents_map.get(ds.name, []),
                "fields": fields,
                "field_count": len(fields),
            }
            nodes.append(node)

        # Count by type
        source_count = sum(1 for n in nodes if n["type"] == "source")
        computed_count = sum(1 for n in nodes if n["type"] == "computed")
        total_fields = sum(n["field_count"] for n in nodes)

        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "source_count": source_count,
                "computed_count": computed_count,
                "total_fields": total_fields,
            }
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/data")
def api_data_tree():
    """Get the data directory tree structure."""
    def get_dir_size(path: Path) -> int:
        """Calculate total size of directory."""
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, OSError):
            pass
        return total

    def count_files(path: Path) -> int:
        """Count files recursively."""
        count = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    count += 1
        except (PermissionError, OSError):
            pass
        return count

    def build_tree(path: Path, depth: int = 0, max_depth: int = 5) -> Dict:
        if depth > max_depth:
            return {"name": path.name, "type": "directory", "truncated": True}

        result = {
            "name": path.name,
            "path": str(path.relative_to(BASE_DIR)),
            "type": "directory" if path.is_dir() else "file",
        }

        if path.is_dir():
            children = []
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            # Files to hide from the data browser
            hidden_files = {'spymaster.db', '.gitkeep', '.DS_Store'}

            # Include all items - frontend handles virtual scrolling
            for item in items:
                if item.name in hidden_files:
                    continue
                children.append(build_tree(item, depth + 1, max_depth))

            result["children"] = children
            result["count"] = len(items)
            # Add total file count and size for top-level dirs
            if depth <= 1:
                result["total_files"] = count_files(path)
                result["total_size"] = get_dir_size(path)
        else:
            result["size"] = path.stat().st_size
            result["modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        return result

    return jsonify(build_tree(DATA_DIR))


@app.route("/api/file/info")
def api_file_info():
    """Get info about a file (supports .pt PyTorch files and others)."""
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No path provided"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Security check
    try:
        full_path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    stat = full_path.stat()
    info = {
        "name": full_path.name,
        "path": file_path,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "type": full_path.suffix[1:] if full_path.suffix else "unknown",
    }

    # Handle .pt files (PyTorch checkpoints)
    if full_path.suffix == ".pt":
        try:
            import torch
            checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
            info["pytorch_info"] = {}

            if isinstance(checkpoint, dict):
                info["pytorch_info"]["keys"] = list(checkpoint.keys())

                # Get common metadata - expanded list including LeJEPA metrics
                metric_keys = [
                    # Common training metrics
                    "epoch", "step", "global_step",
                    # Loss metrics
                    "best_val_loss", "val_loss", "train_loss", "loss",
                    # LeJEPA-specific metrics
                    "val_pred_loss", "val_sigreg_loss", "val_total_loss",
                    "train_pred_loss", "train_sigreg_loss", "train_total_loss",
                    "pred_loss", "sigreg_loss", "total_loss",
                    # Entry/Exit policy metrics
                    "val_accuracy", "train_accuracy", "accuracy",
                    "val_f1", "train_f1", "f1_score",
                    # Best metrics
                    "best_val_pred_loss", "best_val_sigreg_loss", "best_val_total_loss",
                    "best_accuracy", "best_f1",
                ]
                for key in metric_keys:
                    if key in checkpoint:
                        val = checkpoint[key]
                        if hasattr(val, "item"):
                            val = val.item()
                        info["pytorch_info"][key] = val

                # Get model state dict info
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    info["pytorch_info"]["num_parameters"] = sum(
                        p.numel() for p in state_dict.values() if hasattr(p, "numel")
                    )
                    info["pytorch_info"]["layers"] = len(state_dict)
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    info["pytorch_info"]["num_parameters"] = sum(
                        p.numel() for p in state_dict.values() if hasattr(p, "numel")
                    )
                    info["pytorch_info"]["layers"] = len(state_dict)
                elif "context_encoder_state_dict" in checkpoint:
                    # LeJEPA-specific: has separate encoder state dicts
                    state_dict = checkpoint["context_encoder_state_dict"]
                    info["pytorch_info"]["context_encoder_params"] = sum(
                        p.numel() for p in state_dict.values() if hasattr(p, "numel")
                    )
                    info["pytorch_info"]["context_encoder_layers"] = len(state_dict)
                    if "predictor_state_dict" in checkpoint:
                        pred_dict = checkpoint["predictor_state_dict"]
                        info["pytorch_info"]["predictor_params"] = sum(
                            p.numel() for p in pred_dict.values() if hasattr(p, "numel")
                        )
                    # Total params
                    total_params = info["pytorch_info"].get("context_encoder_params", 0)
                    if "target_encoder_state_dict" in checkpoint:
                        target_dict = checkpoint["target_encoder_state_dict"]
                        total_params += sum(
                            p.numel() for p in target_dict.values() if hasattr(p, "numel")
                        )
                    total_params += info["pytorch_info"].get("predictor_params", 0)
                    info["pytorch_info"]["num_parameters"] = total_params
            else:
                info["pytorch_info"]["type"] = str(type(checkpoint).__name__)
        except Exception as e:
            info["pytorch_info"] = {"error": str(e)}

    return jsonify(info)


@app.route("/api/pytorch/key")
def api_pytorch_key():
    """Get detailed info about a specific key in a PyTorch checkpoint."""
    file_path = request.args.get("path")
    key_name = request.args.get("key")

    if not file_path or not key_name:
        return jsonify({"error": "path and key parameters required"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists() or not str(full_path).endswith(".pt"):
        return jsonify({"error": "File not found or not a .pt file"}), 404

    # Security check
    try:
        full_path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    try:
        import torch
        import numpy as np

        checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)

        if not isinstance(checkpoint, dict) or key_name not in checkpoint:
            return jsonify({"error": f"Key '{key_name}' not found"}), 404

        value = checkpoint[key_name]
        result = {"key": key_name, "type": type(value).__name__}

        # Handle different value types
        if isinstance(value, torch.Tensor):
            result["data_type"] = "tensor"
            result["dtype"] = str(value.dtype)
            result["shape"] = list(value.shape)
            result["numel"] = value.numel()
            result["device"] = str(value.device)

            # For small tensors, include the data
            if value.numel() <= 10000:
                if value.dim() <= 2:
                    # Convert to nested list for display
                    arr = value.cpu().numpy()
                    if value.dim() == 0:
                        result["value"] = float(arr)
                    elif value.dim() == 1:
                        result["data"] = arr.tolist()
                        result["display"] = "array"
                    else:
                        result["data"] = arr.tolist()
                        result["display"] = "table"
                        result["columns"] = [f"col_{i}" for i in range(arr.shape[1])]
                else:
                    # Multi-dimensional: show stats
                    result["display"] = "stats"
                    arr = value.cpu().numpy().flatten()
                    result["stats"] = {
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                    }
            else:
                # Large tensor: show stats only
                result["display"] = "stats"
                arr = value.cpu().numpy().flatten()
                result["stats"] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }

        elif isinstance(value, dict):
            result["data_type"] = "dict"
            result["display"] = "dict"
            # For state_dict, show layer info
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                result["is_state_dict"] = True
                result["layers"] = []
                for k, v in value.items():
                    result["layers"].append({
                        "name": k,
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                        "params": v.numel()
                    })
                result["total_params"] = sum(v.numel() for v in value.values())
            else:
                # Regular dict: show keys and types
                result["items"] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        result["items"][k] = f"Tensor{list(v.shape)}"
                    elif isinstance(v, (int, float, str, bool)):
                        result["items"][k] = v
                    elif isinstance(v, (list, tuple)):
                        result["items"][k] = f"{type(v).__name__}[{len(v)}]"
                    else:
                        result["items"][k] = type(v).__name__

        elif isinstance(value, (list, tuple)):
            result["data_type"] = "sequence"
            result["length"] = len(value)
            if len(value) <= 100:
                # Check if it's a list of simple types
                if all(isinstance(x, (int, float, str, bool, type(None))) for x in value):
                    result["display"] = "array"
                    result["data"] = list(value)
                elif all(isinstance(x, torch.Tensor) for x in value):
                    result["display"] = "tensor_list"
                    result["tensors"] = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in value]
                else:
                    result["display"] = "types"
                    result["item_types"] = [type(x).__name__ for x in value[:20]]
            else:
                result["display"] = "summary"
                result["item_types"] = [type(x).__name__ for x in value[:10]] + ["..."]

        elif isinstance(value, (int, float, bool)):
            result["data_type"] = "scalar"
            result["display"] = "value"
            result["value"] = value

        elif isinstance(value, str):
            result["data_type"] = "string"
            result["display"] = "value"
            result["value"] = value

        elif value is None:
            result["data_type"] = "none"
            result["display"] = "value"
            result["value"] = None

        else:
            result["data_type"] = "unknown"
            result["display"] = "repr"
            result["value"] = repr(value)[:1000]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parquet/preview")
def api_parquet_preview():
    """Preview a parquet file with pagination."""
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No path provided"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists() or not str(full_path).endswith(".parquet"):
        return jsonify({"error": "File not found or not a parquet file"}), 404

    # Security check - ensure path is within BASE_DIR
    try:
        full_path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 100))
    sort_by = request.args.get("sort_by")
    sort_dir = request.args.get("sort_dir", "asc")

    try:
        # Read parquet file
        df = pd.read_parquet(full_path)
        total_rows = len(df)

        # Reset index to make it a column (preserves timestamp index)
        if df.index.name is not None:
            df = df.reset_index()

        # Get schema info
        schema = []
        for col in df.columns:
            schema.append({
                "name": col,
                "dtype": str(df[col].dtype),
            })

        # Sort if requested (before pagination so we sort the whole file)
        if sort_by and sort_by in df.columns:
            ascending = sort_dir != "desc"
            df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

        # Paginate
        df_slice = df.iloc[offset:offset + limit]

        # Convert to records, handling timestamps
        records = []
        for _, row in df_slice.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                elif hasattr(val, "isoformat"):
                    record[col] = val.isoformat()
                elif hasattr(val, "item"):
                    record[col] = val.item()
                else:
                    record[col] = val
            records.append(record)

        return jsonify({
            "path": file_path,
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "schema": schema,
            "data": records,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parquet/stats")
def api_parquet_stats():
    """Get statistics for a parquet file."""
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No path provided"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        df = pd.read_parquet(full_path)

        # Reset index to include it in stats
        if df.index.name is not None:
            df = df.reset_index()

        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "columns": {},
        }

        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
            }

            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                col_stats.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                })
            elif "datetime" in str(df[col].dtype):
                col_stats.update({
                    "min": df[col].min().isoformat() if not pd.isna(df[col].min()) else None,
                    "max": df[col].max().isoformat() if not pd.isna(df[col].max()) else None,
                })

            stats["columns"][col] = col_stats

        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parquet/histogram")
def api_parquet_histogram():
    """Get histogram data for a specific column in a parquet file."""
    import numpy as np

    file_path = request.args.get("path")
    column = request.args.get("column")
    bins = int(request.args.get("bins", 50))

    if not file_path or not column:
        return jsonify({"error": "path and column required"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        df = pd.read_parquet(full_path)

        # Reset index to include it
        if df.index.name is not None:
            df = df.reset_index()

        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found"}), 404

        col_data = df[column].dropna()

        # Handle datetime columns
        if "datetime" in str(df[column].dtype):
            # Convert to numeric for histogram
            col_numeric = col_data.astype(np.int64) / 1e9  # Convert to seconds
            counts, edges = np.histogram(col_numeric, bins=bins)
            # Convert edges back to ISO strings
            edges_dt = pd.to_datetime(edges * 1e9)
            return jsonify({
                "column": column,
                "dtype": str(df[column].dtype),
                "bins": bins,
                "counts": counts.tolist(),
                "edges": [e.isoformat() for e in edges_dt],
                "is_datetime": True,
            })

        # Handle numeric columns
        if df[column].dtype in ["int64", "float64", "int32", "float32"]:
            counts, edges = np.histogram(col_data, bins=bins)
            return jsonify({
                "column": column,
                "dtype": str(df[column].dtype),
                "bins": bins,
                "counts": counts.tolist(),
                "edges": edges.tolist(),
                "is_datetime": False,
            })

        # Handle categorical/string columns - value counts
        value_counts = col_data.value_counts().head(bins)
        return jsonify({
            "column": column,
            "dtype": str(df[column].dtype),
            "is_categorical": True,
            "labels": value_counts.index.tolist(),
            "counts": value_counts.values.tolist(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/data")
def api_chart_data():
    """Get chart data from parquet files for TradingView visualization.

    Supports multiple data types:
    - stocks: OHLCV candlestick chart
    - options: Aggregated options volume by call/put
    - oi: Open interest distribution by strike
    - gex_flow: GEX flow line charts
    - cache/raw: Combined training data with all features
    """
    import numpy as np

    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No path provided"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Security check
    try:
        full_path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    try:
        df = pd.read_parquet(full_path)

        # Reset index to get timestamp as column
        if df.index.name is not None:
            df = df.reset_index()

        # Detect data type based on path and columns
        data_type = _detect_data_type(file_path, df.columns.tolist())

        if data_type == "stocks":
            return _chart_data_stocks(df, file_path)
        elif data_type == "options":
            return _chart_data_options(df, file_path)
        elif data_type == "greeks":
            return _chart_data_greeks(df, file_path)
        elif data_type == "oi":
            return _chart_data_oi(df, file_path)
        elif data_type == "gex_flow":
            return _chart_data_gex_flow(df, file_path)
        elif data_type == "training_raw":
            return _chart_data_training_raw(df, file_path)
        elif data_type == "training_normalized":
            return _chart_data_training_normalized(df, file_path)
        else:
            # Default: try OHLCV if available
            return _chart_data_ohlcv(df, file_path)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _detect_data_type(file_path: str, columns: List[str]) -> str:
    """Detect data type from file path and columns."""
    path_lower = file_path.lower()

    # Check path patterns first (order matters - more specific patterns first)
    if "gex_flow" in path_lower or "gex-flow" in path_lower:
        return "gex_flow"
    if "training-1m-raw" in path_lower or "cache/raw" in path_lower:
        return "training_raw"
    if "training-1m-normalized" in path_lower:
        return "training_normalized"
    if "greeks-1m" in path_lower or "/greeks/" in path_lower:
        return "greeks"  # Greeks have dedicated schema with 'right' column
    if "oi-day" in path_lower or "/oi/" in path_lower:
        return "oi"
    if "options-1m" in path_lower or "/options/" in path_lower:
        return "options"
    if "stocks-1m" in path_lower or "/stocks/" in path_lower:
        return "stocks"

    # Detect by columns
    if "net_gamma_flow" in columns and "dist_to_zero_gex" in columns:
        return "gex_flow"
    if "open_interest" in columns and "strike" in columns:
        return "oi"
    if "ticker" in columns and any("SPY" in str(c) for c in columns):
        if len(columns) < 15:  # Basic OHLCV
            return "stocks"
        return "options"

    # Default to OHLCV if we have price columns
    if all(c in columns for c in ["open", "high", "low", "close"]):
        return "stocks"

    return "unknown"


def _to_unix(ts, to_eastern: bool = True) -> int:
    """Convert timestamp to Unix seconds for TradingView.

    Args:
        ts: Timestamp to convert
        to_eastern: If True, adjust so chart displays Eastern time labels
                   (14:30 UTC will show as 09:30 on the chart)
    """
    import pytz

    pt = pd.Timestamp(ts)

    if to_eastern:
        # Convert to Eastern time for display
        # lightweight-charts shows time in UTC by default
        # We want 14:30 UTC to display as 09:30 (Eastern)
        eastern = pytz.timezone('US/Eastern')

        if pt.tzinfo is not None:
            # Has timezone - convert to Eastern
            pt_eastern = pt.astimezone(eastern)
        else:
            # No timezone - assume UTC
            pt_utc = pt.replace(tzinfo=pytz.UTC)
            pt_eastern = pt_utc.astimezone(eastern)

        # Create a naive timestamp with the Eastern hour/minute
        # Then treat it as UTC so the chart shows Eastern labels
        eastern_naive = pt_eastern.replace(tzinfo=None)
        # Get offset from UTC (in seconds) - we add this to "fake" the Eastern time
        offset_seconds = pt_eastern.utcoffset().total_seconds()

        # Return the original timestamp adjusted by the offset
        # This makes the chart display Eastern time
        return int(pt.timestamp() + offset_seconds)

    return int(pt.timestamp())


def _chart_data_stocks(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for stocks OHLCV files."""
    import numpy as np

    # Get timestamp column
    time_col = 'window_start' if 'window_start' in df.columns else 'timestamp'
    if time_col not in df.columns:
        time_col = df.columns[0]

    df = df.sort_values(time_col)

    # Compute VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum()
    df['vwap'] = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    # Build OHLCV data - filter out rows with NaN values
    ohlcv = [{"time": _to_unix(row[time_col]), "open": float(row['open']),
              "high": float(row['high']), "low": float(row['low']), "close": float(row['close'])}
             for _, row in df.iterrows()
             if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close'])]

    volume = [{"time": _to_unix(row[time_col]), "value": float(row['volume']),
               "color": "#26a69a" if row['close'] >= row['open'] else "#ef5350"}
              for _, row in df.iterrows()
              if pd.notna(row['open']) and pd.notna(row['close']) and pd.notna(row['volume'])]

    vwap = [{"time": _to_unix(row[time_col]), "value": float(row['vwap'])}
            for _, row in df.iterrows() if not pd.isna(row['vwap'])]

    return jsonify({
        "path": file_path,
        "chart_type": "stocks",
        "ohlcv": ohlcv,
        "volume": volume,
        "vwap": vwap,
        "metadata": {
            "rows": len(df),
            "start_time": _to_unix(df[time_col].iloc[0]) if len(df) > 0 else None,
            "end_time": _to_unix(df[time_col].iloc[-1]) if len(df) > 0 else None,
        }
    })


def _chart_data_options(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for options files - aggregated by call/put."""
    from src.data.option_parser import parse_option_tickers_vectorized

    time_col = 'window_start' if 'window_start' in df.columns else 'timestamp'
    if time_col not in df.columns:
        return jsonify({"error": "No timestamp column found"}), 400

    # Parse option tickers to get call/put
    parsed = parse_option_tickers_vectorized(df['ticker'].values)
    df['option_type'] = parsed['option_type']
    df['strike'] = parsed['strike']

    # Aggregate by timestamp and option type
    agg = df.groupby([time_col, 'option_type']).agg({
        'volume': 'sum',
        'close': 'mean',  # Average price
    }).reset_index()

    # Get timestamps
    timestamps = sorted(agg[time_col].unique())

    # Build call/put volume series
    call_volume = []
    put_volume = []
    call_premium = []
    put_premium = []

    for ts in timestamps:
        t = _to_unix(ts)
        ts_data = agg[agg[time_col] == ts]

        call_data = ts_data[ts_data['option_type'] == 'C']
        put_data = ts_data[ts_data['option_type'] == 'P']

        call_vol = float(call_data['volume'].sum()) if len(call_data) > 0 else 0
        put_vol = float(put_data['volume'].sum()) if len(put_data) > 0 else 0

        call_volume.append({"time": t, "value": call_vol, "color": "#26a69a"})
        put_volume.append({"time": t, "value": -put_vol, "color": "#ef5350"})

    # Put/Call ratio
    pc_ratio = []
    for i, ts in enumerate(timestamps):
        call_v = call_volume[i]["value"]
        put_v = abs(put_volume[i]["value"])
        ratio = put_v / call_v if call_v > 0 else 0
        pc_ratio.append({"time": _to_unix(ts), "value": ratio})

    return jsonify({
        "path": file_path,
        "chart_type": "options",
        "call_volume": call_volume,
        "put_volume": put_volume,
        "pc_ratio": pc_ratio,
        "metadata": {
            "rows": len(df),
            "unique_strikes": int(df['strike'].nunique()),
            "start_time": _to_unix(timestamps[0]) if timestamps else None,
            "end_time": _to_unix(timestamps[-1]) if timestamps else None,
        }
    })


def _chart_data_greeks(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for greeks files - aggregated by call/put.

    Greeks files have 'right' column (C/P or CALL/PUT) instead of needing ticker parsing.
    """
    time_col = 'timestamp' if 'timestamp' in df.columns else 'window_start'
    if time_col not in df.columns:
        return jsonify({"error": "No timestamp column found"}), 400

    # Greeks files have 'right' column directly (C/P or CALL/PUT)
    if 'right' not in df.columns:
        return jsonify({"error": "Missing 'right' column for call/put type"}), 400

    # Normalize right column to C/P
    df['option_type'] = df['right'].str.upper().str[0]  # Take first char: C or P

    # Aggregate by timestamp and option type
    agg = df.groupby([time_col, 'option_type']).agg({
        'delta': 'sum',  # Total delta exposure
        'gamma': 'sum',  # Total gamma exposure
        'implied_vol': 'mean',  # Average IV
    }).reset_index()

    # Also get count of contracts by type
    counts = df.groupby([time_col, 'option_type']).size().reset_index(name='count')
    agg = agg.merge(counts, on=[time_col, 'option_type'])

    # Get timestamps
    timestamps = sorted(agg[time_col].unique())

    # Build call/put series
    call_delta = []
    put_delta = []
    avg_iv = []

    for ts in timestamps:
        t = _to_unix(ts)
        ts_data = agg[agg[time_col] == ts]

        call_data = ts_data[ts_data['option_type'] == 'C']
        put_data = ts_data[ts_data['option_type'] == 'P']

        call_d = float(call_data['delta'].sum()) if len(call_data) > 0 else 0
        put_d = float(put_data['delta'].sum()) if len(put_data) > 0 else 0

        call_delta.append({"time": t, "value": call_d, "color": "#26a69a"})
        put_delta.append({"time": t, "value": put_d, "color": "#ef5350"})

        # Average IV across all options
        all_iv = ts_data['implied_vol'].mean() if len(ts_data) > 0 else 0
        avg_iv.append({"time": t, "value": float(all_iv) * 100})  # Convert to percentage

    return jsonify({
        "path": file_path,
        "chart_type": "greeks",
        "call_delta": call_delta,
        "put_delta": put_delta,
        "avg_iv": avg_iv,
        "metadata": {
            "rows": len(df),
            "unique_strikes": int(df['strike'].nunique()) if 'strike' in df.columns else 0,
            "start_time": _to_unix(timestamps[0]) if timestamps else None,
            "end_time": _to_unix(timestamps[-1]) if timestamps else None,
        }
    })


def _chart_data_oi(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for open interest files - OI by strike."""
    # OI data is typically a snapshot, show by strike
    if 'strike' not in df.columns or 'open_interest' not in df.columns:
        return jsonify({"error": "Missing strike or open_interest columns"}), 400

    # Aggregate by strike and right (call/put)
    agg = df.groupby(['strike', 'right']).agg({
        'open_interest': 'sum'
    }).reset_index()

    # Sort by strike
    agg = agg.sort_values('strike')

    strikes = sorted(agg['strike'].unique())

    # Build OI data by strike
    call_oi = []
    put_oi = []
    net_oi = []

    for strike in strikes:
        strike_data = agg[agg['strike'] == strike]
        # Handle both 'C'/'P' and 'CALL'/'PUT' formats
        call_data = strike_data[strike_data['right'].str.upper().str.startswith('C')]
        put_data = strike_data[strike_data['right'].str.upper().str.startswith('P')]

        call_val = int(call_data['open_interest'].sum()) if len(call_data) > 0 else 0
        put_val = int(put_data['open_interest'].sum()) if len(put_data) > 0 else 0

        call_oi.append({"strike": float(strike), "value": call_val})
        put_oi.append({"strike": float(strike), "value": put_val})
        net_oi.append({"strike": float(strike), "value": call_val - put_val})

    # Get timestamp if available
    ts = None
    if 'timestamp' in df.columns:
        ts = _to_unix(df['timestamp'].iloc[0])
    elif 'date' in df.columns:
        ts = _to_unix(pd.Timestamp(df['date'].iloc[0]))

    return jsonify({
        "path": file_path,
        "chart_type": "oi",
        "call_oi": call_oi,
        "put_oi": put_oi,
        "net_oi": net_oi,
        "strikes": [float(s) for s in strikes],
        "metadata": {
            "rows": len(df),
            "total_call_oi": sum(d["value"] for d in call_oi),
            "total_put_oi": sum(d["value"] for d in put_oi),
            "timestamp": ts,
        }
    })


def _chart_data_gex_flow(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for GEX flow files."""
    # GEX flow has timestamp index
    if df.index.name == 'timestamp' or 'timestamp' in df.columns:
        time_col = 'timestamp' if 'timestamp' in df.columns else df.index.name
        if time_col == df.index.name:
            df = df.reset_index()
            time_col = 'timestamp'
    else:
        # Try first column as timestamp
        time_col = df.columns[0]

    df = df.sort_values(time_col)

    # Build line series for each GEX metric
    def build_series(col_name: str, scale: float = 1.0) -> List[Dict]:
        if col_name not in df.columns:
            return []
        return [{"time": _to_unix(row[time_col]), "value": float(row[col_name]) * scale}
                for _, row in df.iterrows() if not pd.isna(row[col_name])]

    # Price line (if available)
    price_data = build_series('underlying_price')

    # GEX flow metrics
    net_gamma_flow = build_series('net_gamma_flow')
    cumulative_net_gex = build_series('cumulative_net_gex')
    dist_to_zero_gex = build_series('dist_to_zero_gex')
    dist_to_pos_wall = build_series('dist_to_pos_gex_wall')
    dist_to_neg_wall = build_series('dist_to_neg_gex_wall')

    # DEX metrics
    net_delta_flow = build_series('net_delta_flow')
    dist_to_zero_dex = build_series('dist_to_zero_dex')

    # Other metrics
    gamma_sentiment = build_series('gamma_sentiment_ratio')
    atm_iv_data = build_series('atm_iv', scale=100)  # ATM IV as percentage
    vwap_z = build_series('anchored_vwap_z')

    # GEX level prices for overlay on price chart
    # Use new direct price fields if available, fall back to distance-based calculation
    zero_gex_level = []
    zero_dex_level = []
    pos_gamma_wall = []
    neg_gamma_wall = []

    # Check which fields are available
    has_new_fields = 'zero_gex_price' in df.columns or 'positive_gws' in df.columns or 'pos_gex_wall_strike' in df.columns
    has_price = 'underlying_price' in df.columns

    for _, row in df.iterrows():
        ts = _to_unix(row[time_col])
        price = float(row['underlying_price']) if has_price else 0

        # Zero GEX level
        if 'zero_gex_price' in df.columns and not pd.isna(row['zero_gex_price']) and row['zero_gex_price'] != 0:
            zero_gex_level.append({"time": ts, "value": float(row['zero_gex_price'])})
        elif has_price and 'dist_to_zero_gex' in df.columns and not pd.isna(row['dist_to_zero_gex']) and row['dist_to_zero_gex'] != 0:
            zero_gex_level.append({"time": ts, "value": price - float(row['dist_to_zero_gex'])})

        # Zero DEX level
        if 'zero_dex_price' in df.columns and not pd.isna(row['zero_dex_price']) and row['zero_dex_price'] != 0:
            zero_dex_level.append({"time": ts, "value": float(row['zero_dex_price'])})
        elif has_price and 'dist_to_zero_dex' in df.columns and not pd.isna(row['dist_to_zero_dex']) and row['dist_to_zero_dex'] != 0:
            zero_dex_level.append({"time": ts, "value": price - float(row['dist_to_zero_dex'])})

        # Positive GEX wall
        if 'pos_gex_wall_strike' in df.columns and not pd.isna(row['pos_gex_wall_strike']) and row['pos_gex_wall_strike'] != 0:
            pos_gamma_wall.append({"time": ts, "value": float(row['pos_gex_wall_strike'])})
        elif has_price and 'dist_to_pos_gex_wall' in df.columns and not pd.isna(row['dist_to_pos_gex_wall']) and row['dist_to_pos_gex_wall'] != 0:
            pos_gamma_wall.append({"time": ts, "value": price - float(row['dist_to_pos_gex_wall'])})

        # Negative GEX wall
        if 'neg_gex_wall_strike' in df.columns and not pd.isna(row['neg_gex_wall_strike']) and row['neg_gex_wall_strike'] != 0:
            neg_gamma_wall.append({"time": ts, "value": float(row['neg_gex_wall_strike'])})
        elif has_price and 'dist_to_neg_gex_wall' in df.columns and not pd.isna(row['dist_to_neg_gex_wall']) and row['dist_to_neg_gex_wall'] != 0:
            neg_gamma_wall.append({"time": ts, "value": price - float(row['dist_to_neg_gex_wall'])})

    return jsonify({
        "path": file_path,
        "chart_type": "gex_flow",
        "price": price_data,
        "net_gamma_flow": net_gamma_flow,
        "cumulative_net_gex": cumulative_net_gex,
        "dist_to_zero_gex": dist_to_zero_gex,
        "dist_to_pos_wall": dist_to_pos_wall,
        "dist_to_neg_wall": dist_to_neg_wall,
        "net_delta_flow": net_delta_flow,
        "dist_to_zero_dex": dist_to_zero_dex,
        "gamma_sentiment": gamma_sentiment,
        "atm_iv": atm_iv_data,
        "vwap_z": vwap_z,
        # GEX level prices for overlay on price chart
        "zero_gex_level": zero_gex_level,
        "zero_dex_level": zero_dex_level,
        "pos_gamma_wall": pos_gamma_wall,
        "neg_gamma_wall": neg_gamma_wall,
        "metadata": {
            "rows": len(df),
            "start_time": _to_unix(df[time_col].iloc[0]) if len(df) > 0 else None,
            "end_time": _to_unix(df[time_col].iloc[-1]) if len(df) > 0 else None,
        }
    })


def _chart_data_training_raw(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for raw training files (combined OHLCV + features)."""
    import numpy as np
    from pathlib import Path

    time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    df = df.sort_values(time_col)

    # Ensure timestamp is timezone-aware for filtering
    if not hasattr(df[time_col].dtype, 'tz') or df[time_col].dt.tz is None:
        df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize('UTC')
    else:
        df[time_col] = pd.to_datetime(df[time_col]).dt.tz_convert('UTC')

    # Determine market hours in UTC based on the date (DST-aware)
    # Market hours: 9:30 AM - 4:00 PM ET
    # In EST (winter): 14:30 - 21:00 UTC
    # In EDT (summer): 13:30 - 20:00 UTC
    import pytz
    eastern = pytz.timezone('US/Eastern')

    # Get the date from the data to determine DST offset
    sample_date = df[time_col].iloc[0].date() if len(df) > 0 else None
    if sample_date:
        # Create a datetime at market open in Eastern time
        market_open_et = eastern.localize(pd.Timestamp(sample_date).replace(hour=9, minute=30).to_pydatetime())
        market_close_et = eastern.localize(pd.Timestamp(sample_date).replace(hour=16, minute=0).to_pydatetime())
        # Convert to UTC to get the actual hours
        market_open_utc = market_open_et.astimezone(pytz.UTC)
        market_close_utc = market_close_et.astimezone(pytz.UTC)
        market_start_hour = market_open_utc.hour
        market_start_minute = market_open_utc.minute
        market_end_hour = market_close_utc.hour
    else:
        # Fallback to EST (winter) hours
        market_start_hour = 14
        market_start_minute = 30
        market_end_hour = 21

    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute

    # Premarket: before market open UTC
    premarket_mask = (df['hour'] < market_start_hour) | \
                     ((df['hour'] == market_start_hour) & (df['minute'] < market_start_minute))

    # Market hours: market open to close UTC
    market_mask = ((df['hour'] > market_start_hour) | \
                   ((df['hour'] == market_start_hour) & (df['minute'] >= market_start_minute))) & \
                  (df['hour'] < market_end_hour)

    premarket_df = df[premarket_mask]
    market_df = df[market_mask]

    # Check for OHLCV
    has_ohlcv = all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume'])

    result = {
        "path": file_path,
        "chart_type": "training_raw",
        "metadata": {
            "rows": len(market_df),
            "total_rows": len(df),
            "premarket_rows": len(premarket_df),
            "start_time": _to_unix(market_df[time_col].iloc[0]) if len(market_df) > 0 else None,
            "end_time": _to_unix(market_df[time_col].iloc[-1]) if len(market_df) > 0 else None,
        }
    }

    # Calculate premarket high/low from the premarket range candle
    # New structure (8 candles at 9:22-9:29 AM ET):
    # - 9:22: T-3 day, 9:23: blank, 9:24: T-2 day, 9:25: blank
    # - 9:26: T-1 day, 9:27: blank, 9:28: premarket range, 9:29: blank
    # The premarket range candle is at 14:28 UTC (9:28 AM ET)
    premarket_found = False
    if has_ohlcv and len(premarket_df) > 0:
        premarket_range_mask = (premarket_df['hour'] == 14) & (premarket_df['minute'] == 28)
        premarket_range = premarket_df[premarket_range_mask]

        if len(premarket_range) > 0:
            row = premarket_range.iloc[0]
            if row['high'] > 0 and row['low'] > 0:
                result["premarket_high"] = float(row['high'])
                result["premarket_low"] = float(row['low'])
                premarket_found = True

    # Fallback: if no premarket candle found, try to fetch from Alpaca for current day
    if not premarket_found and has_ohlcv and len(market_df) > 0:
        try:
            from datetime import date as date_type
            query_date = market_df[time_col].iloc[0].date()
            today = date_type.today()

            # Only fetch from Alpaca for today's data
            if query_date == today:
                # Extract underlying from file path
                path_parts = Path(file_path).parts
                underlying = "SPY"
                for i, part in enumerate(path_parts):
                    if part in ("training-1m-raw", "training-1m-normalized", "cache"):
                        if i + 1 < len(path_parts):
                            underlying = path_parts[i + 1]
                        break

                from src.data.providers.alpaca import AlpacaProvider
                import pytz

                provider = AlpacaProvider()
                et = pytz.timezone("US/Eastern")

                # Premarket: 4:00 AM - 9:30 AM ET
                premarket_start = et.localize(pd.Timestamp(query_date).replace(hour=4, minute=0))
                premarket_end = et.localize(pd.Timestamp(query_date).replace(hour=9, minute=30))

                bars = provider.get_bars(
                    underlying,
                    premarket_start.astimezone(pytz.UTC),
                    premarket_end.astimezone(pytz.UTC),
                    timeframe="1Min",
                    feed="sip"
                )

                if bars and len(bars) > 0:
                    pm_high = max(b.get('high', 0) for b in bars)
                    pm_low = min(b.get('low', float('inf')) for b in bars if b.get('low', 0) > 0)
                    if pm_high > 0 and pm_low < float('inf'):
                        result["premarket_high"] = float(pm_high)
                        result["premarket_low"] = float(pm_low)
        except Exception as e:
            # Silently ignore premarket fetch errors
            pass

    # Calculate 3-day high/low: prior 3 trading days only (separate from premarket)
    try:
        if len(market_df) > 0:
            query_date = market_df[time_col].iloc[0].date()

            # Extract underlying from file path (e.g., data/training-1m-raw/SLV/2025-12/29.parquet -> SLV)
            path_parts = Path(file_path).parts
            underlying = "SPY"  # default
            for i, part in enumerate(path_parts):
                if part in ("training-1m-raw", "training-1m-normalized"):
                    if i + 1 < len(path_parts):
                        underlying = path_parts[i + 1]
                    break

            # Load stocks-1d data for the past 3 trading days
            # May need to look at current year + previous year for dates at start of year
            stocks_1d_base = Path(f"/home/ant/code/spymaster/data/stocks-1d/{underlying}")
            stocks_frames = []
            for year in [query_date.year, query_date.year - 1]:
                year_path = stocks_1d_base / f"{year}.parquet"
                if year_path.exists():
                    stocks_df = pd.read_parquet(year_path)
                    stocks_df['date'] = pd.to_datetime(stocks_df['date']).dt.date
                    stocks_frames.append(stocks_df)

            if stocks_frames:
                stocks_1d = pd.concat(stocks_frames, ignore_index=True)
                stocks_1d = stocks_1d.sort_values('date')
                # Get last 3 trading days before query_date
                prior_days = stocks_1d[stocks_1d['date'] < query_date].tail(3)
                if len(prior_days) > 0:
                    result["three_day_high"] = float(prior_days['high'].max())
                    result["three_day_low"] = float(prior_days['low'].min())
    except Exception as e:
        # Silently ignore if we can't load 3-day data
        pass

    if has_ohlcv and len(market_df) > 0:
        # OHLCV data - market hours only
        # Filter out rows with zero/NaN OHLC values (causes "Value is null" error in charts)
        result["ohlcv"] = [{"time": _to_unix(row[time_col]), "open": float(row['open']),
                           "high": float(row['high']), "low": float(row['low']), "close": float(row['close'])}
                          for _, row in market_df.iterrows()
                          if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close'])
                          and row['open'] > 0 and row['high'] > 0]
        # Volume bars with daily high highlighting
        # Gold (#FFD700) for green bars that are current daily high
        # Purple (#9333EA) for red bars that are current daily high
        # First bar keeps original color even if it's the daily high
        volume_data = []
        max_volume_so_far = 0.0
        for idx, (_, row) in enumerate(market_df.iterrows()):
            # Skip rows with NaN or invalid values
            if pd.isna(row['open']) or pd.isna(row['close']) or pd.isna(row['volume']) or row['open'] <= 0:
                continue
            vol = float(row['volume'])
            is_up = row['close'] >= row['open']

            # Check if this is a new daily high (and not the first bar)
            is_daily_high = vol > max_volume_so_far and idx > 0
            max_volume_so_far = max(max_volume_so_far, vol)

            if is_daily_high:
                color = "#FFD700" if is_up else "#9333EA"  # Gold / Purple
            else:
                color = "#26a69a" if is_up else "#ef5350"  # Green / Red

            volume_data.append({"time": _to_unix(row[time_col]), "value": vol, "color": color})
        result["volume"] = volume_data

        # Use pre-computed VWAP from training-1m-raw if available, otherwise compute it
        mdf = market_df.copy()
        if 'vwap' in mdf.columns:
            # Use pre-computed VWAP from training-1m-raw
            result["vwap"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap'])}
                             for _, row in mdf.iterrows() if not pd.isna(row.get('vwap', np.nan)) and row['open'] > 0 and row['vwap'] > 0]
        else:
            # Compute VWAP dynamically
            typical_price = (mdf['high'] + mdf['low'] + mdf['close']) / 3
            cumulative_tp_vol = (typical_price * mdf['volume']).cumsum()
            cumulative_vol = mdf['volume'].cumsum()
            mdf['vwap'] = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
            result["vwap"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap'])}
                             for _, row in mdf.iterrows() if not pd.isna(row.get('vwap', np.nan)) and row['open'] > 0]

        # VWAP bands (from training-1m-raw)
        if 'vwap_upper_1' in mdf.columns:
            result["vwap_upper_1"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap_upper_1'])}
                                      for _, row in mdf.iterrows() if not pd.isna(row.get('vwap_upper_1', np.nan)) and row['open'] > 0 and row['vwap_upper_1'] > 0]
            result["vwap_lower_1"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap_lower_1'])}
                                      for _, row in mdf.iterrows() if not pd.isna(row.get('vwap_lower_1', np.nan)) and row['open'] > 0 and row['vwap_lower_1'] > 0]
            result["vwap_upper_2"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap_upper_2'])}
                                      for _, row in mdf.iterrows() if not pd.isna(row.get('vwap_upper_2', np.nan)) and row['open'] > 0 and row['vwap_upper_2'] > 0]
            result["vwap_lower_2"] = [{"time": _to_unix(row[time_col]), "value": float(row['vwap_lower_2'])}
                                      for _, row in mdf.iterrows() if not pd.isna(row.get('vwap_lower_2', np.nan)) and row['open'] > 0 and row['vwap_lower_2'] > 0]

        # Rolling VWAP (computed dynamically)
        typical_price = (mdf['high'] + mdf['low'] + mdf['close']) / 3
        rolling_tp_vol = (typical_price * mdf['volume']).rolling(window=30, min_periods=1).sum()
        rolling_vol = mdf['volume'].rolling(window=30, min_periods=1).sum()
        mdf['rolling_vwap'] = rolling_tp_vol / rolling_vol.replace(0, np.nan)

        result["rolling_vwap"] = [{"time": _to_unix(row[time_col]), "value": float(row['rolling_vwap'])}
                                  for _, row in mdf.iterrows() if not pd.isna(row.get('rolling_vwap', np.nan)) and row['open'] > 0]

        # Opening Range (OR) high/low - from pre-computed columns (populated after first 15 min)
        # Extend line to first candle for visual continuity
        first_time = _to_unix(mdf.iloc[0][time_col]) if len(mdf) > 0 else None
        if 'or_high' in mdf.columns:
            or_high_data = [{"time": _to_unix(row[time_col]), "value": float(row['or_high'])}
                            for _, row in mdf.iterrows() if not pd.isna(row.get('or_high', np.nan)) and row['open'] > 0 and row['or_high'] > 0]
            if or_high_data and first_time and or_high_data[0]["time"] > first_time:
                or_high_data.insert(0, {"time": first_time, "value": or_high_data[0]["value"]})
            result["or_high"] = or_high_data
        if 'or_low' in mdf.columns:
            or_low_data = [{"time": _to_unix(row[time_col]), "value": float(row['or_low'])}
                           for _, row in mdf.iterrows() if not pd.isna(row.get('or_low', np.nan)) and row['open'] > 0 and row['or_low'] > 0]
            if or_low_data and first_time and or_low_data[0]["time"] > first_time:
                or_low_data.insert(0, {"time": first_time, "value": or_low_data[0]["value"]})
            result["or_low"] = or_low_data

        # Premarket (PM) high/low - from pre-computed columns
        # Extend line to first candle for visual continuity
        if 'pm_high' in mdf.columns:
            pm_high_data = [{"time": _to_unix(row[time_col]), "value": float(row['pm_high'])}
                            for _, row in mdf.iterrows() if not pd.isna(row.get('pm_high', np.nan)) and row['open'] > 0 and row['pm_high'] > 0]
            if pm_high_data and first_time and pm_high_data[0]["time"] > first_time:
                pm_high_data.insert(0, {"time": first_time, "value": pm_high_data[0]["value"]})
            result["pm_high"] = pm_high_data
        if 'pm_low' in mdf.columns:
            pm_low_data = [{"time": _to_unix(row[time_col]), "value": float(row['pm_low'])}
                           for _, row in mdf.iterrows() if not pd.isna(row.get('pm_low', np.nan)) and row['open'] > 0 and row['pm_low'] > 0]
            if pm_low_data and first_time and pm_low_data[0]["time"] > first_time:
                pm_low_data.insert(0, {"time": first_time, "value": pm_low_data[0]["value"]})
            result["pm_low"] = pm_low_data

    # Use market_df for all remaining features
    df_features = market_df

    # Options flow
    if 'atm_call_volume' in df_features.columns:
        result["call_volume"] = [{"time": _to_unix(row[time_col]),
                                  "value": float(row.get('atm_call_volume', 0) or 0) + float(row.get('otm_call_volume', 0) or 0),
                                  "color": "#26a69a"}
                                 for _, row in df_features.iterrows()]
        result["put_volume"] = [{"time": _to_unix(row[time_col]),
                                 "value": -(float(row.get('atm_put_volume', 0) or 0) + float(row.get('otm_put_volume', 0) or 0)),
                                 "color": "#ef5350"}
                                for _, row in df_features.iterrows()]

    # Net Trade Aggression (separate chart - volume histogram)
    # Bullish = (call buys - call sells) + (put sells - put buys)
    # Bearish = opposite
    # Show actual volume values (positive = bullish, negative = bearish)
    if 'call_buy_volume' in df_features.columns:
        net_aggression = []
        for _, row in df_features.iterrows():
            call_buy = float(row.get('call_buy_volume', 0) or 0)
            call_sell = float(row.get('call_sell_volume', 0) or 0)
            put_buy = float(row.get('put_buy_volume', 0) or 0)
            put_sell = float(row.get('put_sell_volume', 0) or 0)

            # Bullish signals: buying calls + selling puts
            # Bearish signals: selling calls + buying puts
            # Result is the actual net aggression volume (not normalized)
            net_bullish = (call_buy - call_sell) + (put_sell - put_buy)

            net_aggression.append({
                "time": _to_unix(row[time_col]),
                "value": net_bullish,
                "color": "#22c55e" if net_bullish >= 0 else "#ef4444"  # green if bullish, red if bearish
            })
        result["net_aggression"] = net_aggression

    # GEX Regime Strength for subchart (histogram with green/red bars)
    # Combines position (above/below zero GEX) with magnitude - replaces total_gex
    if 'gex_regime_strength' in df_features.columns:
        result["gex_regime_strength"] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row['gex_regime_strength']),
                "color": "#22c55e" if row['gex_regime_strength'] >= 0 else "#ef4444"
            }
            for _, row in df_features.iterrows()
            if not pd.isna(row['gex_regime_strength'])
        ]

    # GEX features
    for col in ['net_gamma_flow', 'cumulative_net_gex', 'dist_to_zero_gex', 'net_delta_flow']:
        if col in df_features.columns:
            result[col] = [{"time": _to_unix(row[time_col]), "value": float(row[col])}
                          for _, row in df_features.iterrows() if not pd.isna(row[col])]

    # Net GEX with color: Green when positive (stabilizing), Purple when negative (amplifying)
    if 'net_gex' in df_features.columns:
        result['net_gex'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row['net_gex']),
                "color": "#22c55e" if row['net_gex'] >= 0 else "#a855f7"  # green-500 / purple-500
            }
            for _, row in df_features.iterrows() if not pd.isna(row['net_gex'])
        ]

    # Net DEX with color: Teal when positive, Red when negative
    if 'net_dex' in df_features.columns:
        result['net_dex'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row['net_dex']),
                "color": "#14b8a6" if row['net_dex'] >= 0 else "#ef4444"  # teal-500 / red-500
            }
            for _, row in df_features.iterrows() if not pd.isna(row['net_dex'])
        ]

    # ATM IV (At The Money Implied Volatility)
    if 'atm_iv' in df_features.columns:
        result["atm_iv"] = [{"time": _to_unix(row[time_col]), "value": float(row['atm_iv']) * 100}
                       for _, row in df_features.iterrows() if not pd.isna(row['atm_iv'])]

    # GEX price levels for charting (use new direct price fields, fall back to distance-based if not available)
    # Zero GEX level
    if 'zero_gex_price' in df_features.columns:
        result["zero_gex_level"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['zero_gex_price'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['zero_gex_price']) and row['zero_gex_price'] != 0
        ]
    elif 'dist_to_zero_gex' in df_features.columns and 'close' in df_features.columns:
        # Fallback to old distance-based calculation
        result["zero_gex_level"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['close']) - float(row['dist_to_zero_gex'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['dist_to_zero_gex']) and row['dist_to_zero_gex'] != 0
        ]

    # Zero DEX level
    if 'zero_dex_price' in df_features.columns:
        result["zero_dex_level"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['zero_dex_price'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['zero_dex_price']) and row['zero_dex_price'] != 0
        ]
    elif 'dist_to_zero_dex' in df_features.columns and 'close' in df_features.columns:
        # Fallback to old distance-based calculation
        result["zero_dex_level"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['close']) - float(row['dist_to_zero_dex'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['dist_to_zero_dex']) and row['dist_to_zero_dex'] != 0
        ]

    # Weighted Centroids (GWS/DWS)
    # Positive GWS (gamma-weighted centroid of positive GEX - stabilizing anchor)
    if 'positive_gws' in df_features.columns:
        result["positive_gws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['positive_gws'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['positive_gws']) and row['positive_gws'] != 0
        ]
    elif 'pos_gex_wall_strike' in df_features.columns:
        # Fallback to old field name
        result["positive_gws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['pos_gex_wall_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['pos_gex_wall_strike']) and row['pos_gex_wall_strike'] != 0
        ]

    # Negative GWS (gamma-weighted centroid of negative GEX - volatility anchor)
    if 'negative_gws' in df_features.columns:
        result["negative_gws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['negative_gws'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['negative_gws']) and row['negative_gws'] != 0
        ]
    elif 'neg_gex_wall_strike' in df_features.columns:
        # Fallback to old field name
        result["negative_gws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['neg_gex_wall_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['neg_gex_wall_strike']) and row['neg_gex_wall_strike'] != 0
        ]

    # -DWS: Delta-weighted support (centroid of delta exposure below spot)
    if 'negative_dws' in df_features.columns:
        result["negative_dws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['negative_dws'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['negative_dws']) and row['negative_dws'] != 0
        ]

    # +DWS: Delta-weighted resistance (centroid of delta exposure above spot)
    if 'positive_dws' in df_features.columns:
        result["positive_dws"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['positive_dws'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['positive_dws']) and row['positive_dws'] != 0
        ]

    # Wall Levels (max exposure strikes)
    # Gamma Call Wall (strike with max positive GEX - price magnet)
    if 'gamma_call_wall' in df_features.columns:
        result["gamma_call_wall"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['gamma_call_wall'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['gamma_call_wall']) and row['gamma_call_wall'] != 0
        ]

    # Gamma Put Wall (strike with max negative GEX - trap door)
    if 'gamma_put_wall' in df_features.columns:
        result["gamma_put_wall"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['gamma_put_wall'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['gamma_put_wall']) and row['gamma_put_wall'] != 0
        ]

    # DEX Call Wall (strike with max positive DEX - dealer ceiling)
    if 'dex_call_wall' in df_features.columns:
        result["dex_call_wall"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['dex_call_wall'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['dex_call_wall']) and row['dex_call_wall'] != 0
        ]

    # DEX Put Wall (strike with max negative DEX - dealer floor)
    if 'dex_put_wall' in df_features.columns:
        result["dex_put_wall"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['dex_put_wall'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['dex_put_wall']) and row['dex_put_wall'] != 0
        ]

    # Market Velocity (hedging force from order flow)
    if 'market_velocity' in df_features.columns:
        result["market_velocity"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['market_velocity'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['market_velocity'])
        ]

    # === Volume Category ===
    # Most Active Strike (strike with highest options volume)
    if 'most_active_strike' in df_features.columns:
        result["most_active_strike"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['most_active_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['most_active_strike']) and row['most_active_strike'] != 0
        ]

    # Call Weighted Strike (+WS - volume-weighted average call strike)
    if 'call_weighted_strike' in df_features.columns:
        result["call_weighted_strike"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['call_weighted_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['call_weighted_strike']) and row['call_weighted_strike'] != 0
        ]

    # Put Weighted Strike (-WS - volume-weighted average put strike)
    if 'put_weighted_strike' in df_features.columns:
        result["put_weighted_strike"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['put_weighted_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['put_weighted_strike']) and row['put_weighted_strike'] != 0
        ]

    # Max Call Strike (MCS - strike with highest call volume)
    if 'max_call_strike' in df_features.columns:
        result["max_call_strike"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['max_call_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['max_call_strike']) and row['max_call_strike'] != 0
        ]

    # Max Put Strike (MPS - strike with highest put volume)
    if 'max_put_strike' in df_features.columns:
        result["max_put_strike"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['max_put_strike'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['max_put_strike']) and row['max_put_strike'] != 0
        ]

    # Point of Control (POC - single strike with highest total volume)
    if 'point_of_control' in df_features.columns:
        result["point_of_control"] = [
            {"time": _to_unix(row[time_col]), "value": float(row['point_of_control'])}
            for _, row in df_features.iterrows()
            if not pd.isna(row['point_of_control']) and row['point_of_control'] != 0
        ]

    # Stock Net Trade Volume (buy - sell) - per-minute imbalance
    # Handle both old name (stock_trade_ofi) and new name (stock_net_trade_volume)
    ntv_col = 'stock_net_trade_volume' if 'stock_net_trade_volume' in df_features.columns else 'stock_trade_ofi'
    if ntv_col in df_features.columns:
        result['stock_ofi'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row[ntv_col]),
                "color": "#22c55e" if row[ntv_col] >= 0 else "#ef4444"
            }
            for _, row in df_features.iterrows()
            if not pd.isna(row[ntv_col])
        ]

    return jsonify(result)


def _chart_data_training_normalized(df: pd.DataFrame, file_path: str) -> Any:
    """Generate chart data for normalized training files."""
    time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    df = df.sort_values(time_col)

    # Normalized data shows log returns, so we need different visualization
    result = {
        "path": file_path,
        "chart_type": "training_normalized",
        "metadata": {
            "rows": len(df),
            "start_time": _to_unix(df[time_col].iloc[0]) if len(df) > 0 else None,
            "end_time": _to_unix(df[time_col].iloc[-1]) if len(df) > 0 else None,
        }
    }

    # Show log returns as line chart
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            result[f"{col}_norm"] = [{"time": _to_unix(row[time_col]), "value": float(row[col])}
                                     for _, row in df.iterrows() if not pd.isna(row[col])]

    # GEX features (already normalized)
    # Net GEX with color: Green when positive (stabilizing), Purple when negative (amplifying)
    if 'net_gex' in df.columns:
        result['net_gex'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row['net_gex']),
                "color": "#22c55e" if row['net_gex'] >= 0 else "#a855f7"  # green-500 / purple-500
            }
            for _, row in df.iterrows() if not pd.isna(row['net_gex'])
        ]

    # Net DEX with color: Teal when positive, Red when negative
    if 'net_dex' in df.columns:
        result['net_dex'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row['net_dex']),
                "color": "#14b8a6" if row['net_dex'] >= 0 else "#ef4444"  # teal-500 / red-500
            }
            for _, row in df.iterrows() if not pd.isna(row['net_dex'])
        ]

    # Stock Net Trade Volume - per-minute imbalance
    # Handle both old name (stock_trade_ofi) and new name (stock_net_trade_volume)
    ntv_col = 'stock_net_trade_volume' if 'stock_net_trade_volume' in df.columns else 'stock_trade_ofi'
    if ntv_col in df.columns:
        result['stock_ofi'] = [
            {
                "time": _to_unix(row[time_col]),
                "value": float(row[ntv_col]),
                "color": "#22c55e" if row[ntv_col] >= 0 else "#ef4444"  # green / red
            }
            for _, row in df.iterrows() if not pd.isna(row[ntv_col])
        ]

    gex_cols = [
        'gamma_flow', 'delta_flow', 'market_velocity',
        'zero_gex_price', 'zero_dex_price',
        'positive_gws', 'negative_gws', 'negative_dws', 'positive_dws',
        'gamma_call_wall', 'gamma_put_wall', 'dex_call_wall', 'dex_put_wall',
    ]
    for col in gex_cols:
        if col in df.columns:
            result[col] = [{"time": _to_unix(row[time_col]), "value": float(row[col])}
                          for _, row in df.iterrows() if not pd.isna(row[col])]

    return jsonify(result)


def _chart_data_ohlcv(df: pd.DataFrame, file_path: str) -> Any:
    """Fallback: try to generate basic OHLCV chart."""
    import numpy as np

    # Check for required columns
    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        return jsonify({"error": "File does not have OHLCV data"}), 400

    # Find timestamp column
    time_col = None
    for col in ['timestamp', 'window_start', 'time', 'date']:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]

    df = df.sort_values(time_col)

    # Filter out rows with NaN values
    ohlcv = [{"time": _to_unix(row[time_col]), "open": float(row['open']),
              "high": float(row['high']), "low": float(row['low']), "close": float(row['close'])}
             for _, row in df.iterrows()
             if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close'])]

    volume = []
    if 'volume' in df.columns:
        volume = [{"time": _to_unix(row[time_col]), "value": float(row['volume']),
                   "color": "#26a69a" if row['close'] >= row['open'] else "#ef5350"}
                  for _, row in df.iterrows()
                  if pd.notna(row['open']) and pd.notna(row['close']) and pd.notna(row['volume'])]

    return jsonify({
        "path": file_path,
        "chart_type": "ohlcv",
        "ohlcv": ohlcv,
        "volume": volume,
        "metadata": {
            "rows": len(df),
            "start_time": _to_unix(df[time_col].iloc[0]) if len(df) > 0 else None,
            "end_time": _to_unix(df[time_col].iloc[-1]) if len(df) > 0 else None,
        }
    })


@app.route("/api/reports")
def api_reports():
    """List all report files."""
    reports = []

    if REPORTS_DIR.exists():
        for f in sorted(REPORTS_DIR.iterdir()):
            if f.is_file():
                stat = f.stat()
                reports.append({
                    "name": f.name,
                    "path": str(f.relative_to(BASE_DIR)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": f.suffix[1:] if f.suffix else "unknown",
                })

    return jsonify(reports)


@app.route("/api/backtests")
def api_backtests():
    """List all backtest JSON files."""
    backtests = []

    if REPORTS_DIR.exists():
        for f in sorted(REPORTS_DIR.glob("*.json")):
            stat = f.stat()
            # Try to load summary info from the JSON
            summary = {"name": f.stem}
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    summary["title"] = data.get("title", f.stem)
                    summary["generated_at"] = data.get("generated_at")
                    summary["date_range"] = data.get("date_range")
                    summary["strategy_id"] = data.get("strategy_id")
                    summary["strategy_name"] = data.get("strategy_name")
                    summary["checkpoints"] = data.get("checkpoints")
                    if "metrics" in data:
                        summary["total_return"] = data["metrics"].get("total_return")
                        summary["total_trades"] = data["metrics"].get("total_trades")
                        summary["win_rate"] = data["metrics"].get("win_rate")
            except (json.JSONDecodeError, KeyError):
                pass

            backtests.append({
                "name": f.name,
                "path": str(f.relative_to(BASE_DIR)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                **summary,
            })

    return jsonify(backtests)


@app.route("/api/backtest/<path:name>")
def api_backtest_data(name: str):
    """Get backtest JSON data."""
    # Handle both with and without .json extension
    if not name.endswith(".json"):
        name = f"{name}.json"

    backtest_path = REPORTS_DIR / name
    if not backtest_path.exists():
        return jsonify({"error": "Backtest not found"}), 404

    try:
        with open(backtest_path) as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 500


@app.route("/api/report/<path:name>")
def api_report_content(name: str):
    """Get report content."""
    report_path = REPORTS_DIR / name
    if not report_path.exists():
        return jsonify({"error": "Report not found"}), 404

    return send_file(report_path)


@app.route("/api/files/list")
def api_files_list():
    """List files in a directory with optional pattern matching."""
    dir_path = request.args.get("path", "data")
    pattern = request.args.get("pattern", "*")

    full_path = BASE_DIR / dir_path
    if not full_path.exists() or not full_path.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    files = []
    for f in sorted(full_path.glob(pattern))[:500]:
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f.relative_to(BASE_DIR)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    return jsonify({
        "directory": dir_path,
        "pattern": pattern,
        "files": files,
        "count": len(files),
    })


# =============================================================================
# Strategy API Routes
# =============================================================================

@app.route("/api/strategies")
def api_strategies():
    """List all strategies."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import list_strategies
    return jsonify(list_strategies())


@app.route("/api/strategy/<name>")
def api_strategy_detail(name: str):
    """Get detailed info about a specific strategy."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import get_strategy

    strategy = get_strategy(name)
    if strategy is None:
        return jsonify({"error": "Strategy not found"}), 404

    return jsonify(strategy)


@app.route("/api/strategy/id/<strategy_id>")
def api_strategy_by_id(strategy_id: str):
    """Get a strategy by its UUID."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import get_strategy_by_id

    strategy = get_strategy_by_id(strategy_id)
    if strategy is None:
        return jsonify({"error": "Strategy not found"}), 404

    return jsonify(strategy)


@app.route("/api/strategy", methods=["POST"])
def api_create_strategy():
    """Create a new strategy."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import create_strategy

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required_fields = ["name", "description", "lejepa_checkpoint", "entry_policy_checkpoint"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        strategy = create_strategy(
            name=data["name"],
            description=data["description"],
            lejepa_checkpoint=data["lejepa_checkpoint"],
            entry_policy_checkpoint=data["entry_policy_checkpoint"],
            exit_mode=data.get("exit_mode", "continuous_signal"),
            exit_policy_checkpoint=data.get("exit_policy_checkpoint"),
            exit_config=data.get("exit_config"),
            backtest_config=data.get("backtest_config"),
        )
        return jsonify(strategy), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy/<name>", methods=["PUT"])
def api_update_strategy(name: str):
    """Update an existing strategy."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import update_strategy

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    strategy = update_strategy(name, data)
    if strategy is None:
        return jsonify({"error": "Strategy not found"}), 404

    return jsonify(strategy)


@app.route("/api/strategy/<name>", methods=["DELETE"])
def api_delete_strategy(name: str):
    """Delete a strategy."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import delete_strategy

    if delete_strategy(name):
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Strategy not found"}), 404


@app.route("/api/checkpoint/<name>/log")
def api_checkpoint_log(name: str):
    """Get the training log for a checkpoint."""
    from src.managers.checkpoint_manager import get_manager

    mgr = get_manager()
    info = mgr.get(name)

    if not info:
        return jsonify({"error": "Checkpoint not found"}), 404

    # Check data directory for log file
    data_dir = CHECKPOINTS_DATA_DIR / name
    log_path = None
    if data_dir.exists():
        for log_name in ["log.txt", "training.log"]:
            potential_log = data_dir / log_name
            if potential_log.exists():
                log_path = potential_log
                break

    if not log_path:
        return jsonify({"error": "No log file found", "content": None})

    try:
        with open(log_path, "r") as f:
            content = f.read()
        return jsonify({"content": content, "size": log_path.stat().st_size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pytorch/embeddings")
def api_pytorch_embeddings():
    """
    Extract and visualize embeddings from a LeJEPA checkpoint.

    Query params:
        - path: Path to .pt checkpoint file
        - num_samples: Number of samples to extract (default: 1000)
        - reduction: Dimensionality reduction method (pca, tsne, umap) (default: pca)
        - perplexity: t-SNE perplexity (default: 30)
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    file_path = request.args.get("path")
    num_samples = int(request.args.get("num_samples", 1000))
    reduction = request.args.get("reduction", "pca")
    perplexity = int(request.args.get("perplexity", 30))

    if not file_path:
        return jsonify({"error": "No path provided"}), 400

    full_path = BASE_DIR / file_path
    if not full_path.exists() or not str(full_path).endswith(".pt"):
        return jsonify({"error": "File not found or not a .pt file"}), 404

    # Security check
    try:
        full_path.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    try:
        import torch
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        from src.model.lejepa import LeJEPA
        from src.data.dag.loader import load_normalized_data

        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, config = LeJEPA.load_checkpoint(str(full_path), device=device)
        model.eval()

        # Get model dimensions
        embedding_dim = config.get("embedding_dim", 128)
        input_dim = model.input_proj.weight.shape[1]
        context_len = 90  # Default context length

        # Load recent data for embedding extraction
        # Get the most recent available data
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        df = load_normalized_data(
            stocks_dir="data/stocks",
            options_dir="data/options",
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or len(df) < context_len + num_samples:
            return jsonify({"error": "Not enough data available"}), 400

        # Filter features to match model input dimension
        feature_cols = list(df.columns)
        new_features = ['vwap', 'dist_vwap', 'dist_ma20', 'dist_ma50', 'dist_ma200', 'vol_regime', 'day_of_week']
        if len(feature_cols) > input_dim:
            feature_cols = [c for c in feature_cols if c not in new_features]

        data = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Add zero columns if needed for removed features
        if data.shape[1] < input_dim:
            missing = input_dim - data.shape[1]
            zeros = torch.zeros(len(data), missing)
            data = torch.cat([data, zeros], dim=1)

        # Extract embeddings
        embeddings = []
        labels = []  # Direction labels for coloring
        timestamps = []

        # Sample indices
        valid_start = context_len
        valid_end = len(df) - 5  # Need 5 minutes for label

        if valid_end <= valid_start:
            return jsonify({"error": "Not enough data for context"}), 400

        # Sample evenly across the data
        indices = np.linspace(valid_start, valid_end - 1, min(num_samples, valid_end - valid_start)).astype(int)

        close_idx = feature_cols.index('close') if 'close' in feature_cols else 0

        with torch.no_grad():
            batch_size = 256
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                patches = []

                for idx in batch_indices:
                    patch = data[idx - context_len:idx]
                    patches.append(patch)

                    # Get direction label (next 5 minutes)
                    future_return = data[idx:idx + 5, close_idx].sum().item()
                    labels.append(1 if future_return > 0 else 0)

                    # Get timestamp
                    if hasattr(df.index, '__getitem__'):
                        ts = df.index[idx]
                        if hasattr(ts, 'isoformat'):
                            timestamps.append(ts.isoformat())
                        else:
                            timestamps.append(str(ts))
                    else:
                        timestamps.append(str(idx))

                patches = torch.stack(patches).to(device)
                emb = model.encode(patches)
                embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)

        # Dimensionality reduction
        if reduction == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1), random_state=42)
            coords = reducer.fit_transform(embeddings)
        elif reduction == "pca":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_.tolist()
        else:
            # Default to PCA
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_.tolist()

        # Calculate embedding statistics
        emb_stats = {
            "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
            "std_norm": float(np.linalg.norm(embeddings, axis=1).std()),
            "mean_per_dim": embeddings.mean(axis=0).tolist()[:10],  # First 10 dims
            "std_per_dim": embeddings.std(axis=0).tolist()[:10],
        }

        # Calculate covariance eigenvalues for collapse detection
        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        emb_stats["top_eigenvalues"] = eigenvalues[:10].tolist()
        emb_stats["effective_rank"] = float(np.sum(eigenvalues > 0.01 * eigenvalues[0]))

        result = {
            "num_samples": len(embeddings),
            "embedding_dim": embedding_dim,
            "reduction": reduction,
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "labels": labels.tolist(),
            "timestamps": timestamps,
            "stats": emb_stats,
        }

        if reduction == "pca":
            result["explained_variance"] = explained_var

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/checkpoint/<name>/strategies")
def api_checkpoint_strategies(name: str):
    """Get strategies that use a specific checkpoint."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.utils.strategy_manager import list_strategies, get_strategy, get_checkpoint_path

    checkpoint_path = f"checkpoints/{name}"
    related_strategies = []

    for strategy_summary in list_strategies():
        strategy = get_strategy(strategy_summary["name"])
        if strategy:
            checkpoints = strategy.get("checkpoints", {})
            # Check if any checkpoint path contains this checkpoint name
            for cp_type, cp_entry in checkpoints.items():
                # Handle both old format (string) and new format (dict with id and path)
                cp_path = get_checkpoint_path(cp_entry)
                if cp_path and checkpoint_path in cp_path:
                    related_strategies.append({
                        "id": strategy.get("id"),
                        "name": strategy["name"],
                        "description": strategy.get("description", ""),
                        "checkpoint_role": cp_type,  # lejepa, entry_policy, exit_policy
                        "exit_mode": strategy.get("exit_mode"),
                        "num_backtests": len(strategy.get("backtests", [])),
                    })
                    break

    return jsonify(related_strategies)


# =============================================================================
# Dataset Browse API
# =============================================================================

@app.route("/api/dataset/underlyings")
def api_dataset_underlyings():
    """Get list of available underlyings detected from data folder."""
    underlyings = set()

    # Get all datasets from registry
    registry = get_registry()
    all_datasets = registry.list_datasets()

    for ds_name in all_datasets:
        config = registry.get(ds_name)
        # Extract dataset folder from path pattern
        pattern = config.path_pattern
        # Pattern like "data/stocks-1m/{underlying}/..." - extract "data/stocks-1m"
        parts = pattern.split("/{underlying}")
        if parts:
            ds_path = Path(parts[0])
            if ds_path.exists():
                for item in ds_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        # Filter to uppercase symbols (likely underlyings)
                        if item.name.isupper():
                            underlyings.add(item.name)

    # Sort with SPY first, then alphabetically
    sorted_underlyings = sorted(underlyings, key=lambda x: (x != "SPY", x))

    return jsonify({"underlyings": sorted_underlyings})


@app.route("/api/dataset/stats")
def api_dataset_stats():
    """Get statistics for a dataset: size, file count, date count, frequency."""
    dataset = request.args.get("dataset", "stocks-1m")
    underlying = request.args.get("underlying", "SPY")

    # Get base path from registry
    base_path = get_dataset_path(dataset, underlying)
    if not base_path or not base_path.exists():
        return jsonify(None)

    # Collect stats
    total_size = 0
    file_count = 0
    dates = set()

    # Daily datasets store by year file
    is_daily = is_daily_dataset(dataset)

    for f in base_path.rglob("*.parquet"):
        total_size += f.stat().st_size
        file_count += 1
        # Extract date from path
        if is_daily:
            # Year file like 2024.parquet
            dates.add(f.stem)
        else:
            # Date file like 01.parquet in YYYY-MM folder
            try:
                month_folder = f.parent.name  # YYYY-MM
                day = f.stem  # DD
                dates.add(f"{month_folder}-{day}")
            except Exception:
                pass

    # Determine frequency from registry
    frequency = get_time_resolution(dataset)

    # Format size
    def format_size(size_bytes):
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.1f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes} B"

    # Find date range
    sorted_dates = sorted(dates)
    first_date = sorted_dates[0] if sorted_dates else None
    last_date = sorted_dates[-1] if sorted_dates else None

    return jsonify({
        "size_formatted": format_size(total_size),
        "size_bytes": total_size,
        "file_count": file_count,
        "date_count": len(dates),
        "frequency": frequency,
        "first_date": first_date,
        "last_date": last_date,
    })


@app.route("/api/dataset/calendar")
def api_dataset_calendar():
    """Get calendar view of available dates for a dataset."""
    from calendar import monthrange, weekday
    from datetime import date, timedelta, datetime
    import pytz

    dataset = request.args.get("dataset", "stocks-1m")
    underlying = request.args.get("underlying", "SPY")
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")

    # Parse dates
    try:
        if start_date_str:
            start_date = date.fromisoformat(start_date_str)
        else:
            start_date = date.today() - timedelta(days=30)
        if end_date_str:
            end_date = date.fromisoformat(end_date_str)
        else:
            end_date = date.today()
    except ValueError:
        start_date = date.today() - timedelta(days=30)
        end_date = date.today()

    # Expected rows for intraday datasets (full trading day = 390 minutes)
    FULL_DAY_ROWS = 390  # 9:30 AM - 4:00 PM
    PREMARKET_START_HOUR = 4  # 4:00 AM ET
    MARKET_OPEN_HOUR = 9.5  # 9:30 AM ET
    MARKET_CLOSE_HOUR = 16  # 4:00 PM ET

    # Get current time in ET for "today" calculation
    try:
        et_tz = pytz.timezone("America/New_York")
        now_et = datetime.now(et_tz)
        today = now_et.date()
        current_hour = now_et.hour + now_et.minute / 60
    except Exception:
        today = date.today()
        current_hour = 12  # Default to noon

    def get_expected_rows_for_date(d: date) -> int:
        """Get expected rows for a given date based on time of day."""
        if d < today:
            return FULL_DAY_ROWS  # Historical day should have full data
        elif d == today:
            # Calculate expected rows based on current time
            if current_hour < PREMARKET_START_HOUR:
                return 0  # Before premarket
            elif current_hour < MARKET_OPEN_HOUR:
                # Premarket: ~5.5 hours = 330 minutes but we only track some
                return int((current_hour - PREMARKET_START_HOUR) * 60)
            elif current_hour < MARKET_CLOSE_HOUR:
                # Market hours: calculate minutes since open
                return int((current_hour - MARKET_OPEN_HOUR) * 60)
            else:
                return FULL_DAY_ROWS  # After close
        else:
            return 0  # Future date

    # Get base path from registry
    base_path = get_dataset_path(dataset, underlying)
    if not base_path or not base_path.exists():
        return jsonify({"months": [], "stats": None})

    # Collect available dates with their sizes and row counts
    available_dates = {}  # date_str -> {"size": int, "rows": int}
    is_daily = is_daily_dataset(dataset)
    is_intraday = is_intraday_dataset(dataset)

    for f in base_path.rglob("*.parquet"):
        try:
            if is_daily:
                # Year file - expand to all trading days we have data for
                year = int(f.stem)
                try:
                    df = pd.read_parquet(f, columns=["date"] if "date" in pd.read_parquet(f, columns=[]).columns else [])
                    if "date" in df.columns:
                        for d in df["date"]:
                            if isinstance(d, pd.Timestamp):
                                d = d.date()
                            date_str = str(d)
                            available_dates[date_str] = {
                                "size": available_dates.get(date_str, {}).get("size", 0) + f.stat().st_size // max(1, len(df)),
                                "rows": 1,  # Daily data = 1 row per day
                            }
                except Exception:
                    pass
            else:
                # Date file like DD.parquet in YYYY-MM folder
                month_folder = f.parent.name  # YYYY-MM
                day = f.stem  # DD
                date_str = f"{month_folder}-{day}"
                file_size = f.stat().st_size

                # Count rows for intraday data to calculate completeness
                row_count = 0
                if is_intraday and file_size > 0:
                    try:
                        # Use pyarrow for fast row count without loading data
                        import pyarrow.parquet as pq
                        row_count = pq.ParquetFile(f).metadata.num_rows
                    except Exception:
                        try:
                            row_count = len(pd.read_parquet(f, columns=[]))
                        except Exception:
                            row_count = 0

                available_dates[date_str] = {"size": file_size, "rows": row_count}
        except Exception:
            pass

    # Build calendar months
    months = []
    current = date(start_date.year, start_date.month, 1)

    def format_size(size_bytes):
        if size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.1f}M"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f}K"
        return f"{size_bytes}B"

    while current <= end_date:
        year, month = current.year, current.month
        _, days_in_month = monthrange(year, month)
        first_weekday = weekday(year, month, 1)  # 0=Mon, 6=Sun
        # Convert to Sunday-first: (first_weekday + 1) % 7
        start_offset = (first_weekday + 1) % 7

        days = []
        for day in range(1, days_in_month + 1):
            d = date(year, month, day)
            # Check if it's a market day (weekday, not holiday)
            # Weekday: Mon=0, Tue=1, ..., Fri=4, Sat=5, Sun=6
            is_market_day = d.weekday() < 5  # Monday through Friday

            if d < start_date or d > end_date:
                days.append({"day": day, "date": str(d), "hasData": False, "size": None, "completeness": 0, "status": None, "isMarketDay": is_market_day})
                continue

            date_str = str(d)
            date_info = available_dates.get(date_str)
            has_data = date_info is not None
            size = format_size(date_info["size"]) if has_data else None

            # Calculate completeness for intraday datasets
            completeness = 0.0
            status = None  # "complete", "partial", "behind"

            if has_data and is_intraday:
                expected_rows = get_expected_rows_for_date(d)
                actual_rows = date_info.get("rows", 0)

                if expected_rows > 0:
                    completeness = min(1.0, actual_rows / expected_rows)

                    # Determine status
                    if completeness >= 0.98:  # Allow small margin
                        status = "complete"  # Green - caught up
                    elif d == today:
                        # For today, check if we're behind current time
                        if actual_rows >= expected_rows - 5:  # Within 5 minutes
                            status = "partial"  # Still filling, but on track
                        else:
                            status = "behind"  # Yellow - behind
                    elif d < today:
                        status = "behind"  # Historical but incomplete = behind
                    else:
                        status = "partial"  # Future - N/A
            elif has_data and is_daily:
                completeness = 1.0
                status = "complete"

            days.append({
                "day": day,
                "date": date_str,
                "hasData": has_data,
                "size": size,
                "completeness": completeness,
                "status": status,
                "rows": date_info.get("rows", 0) if has_data else 0,
                "isMarketDay": is_market_day,
            })

        months.append({
            "key": f"{year}-{month:02d}",
            "label": current.strftime("%B %Y"),
            "startOffset": start_offset,
            "days": days,
        })

        # Move to next month
        if month == 12:
            current = date(year + 1, 1, 1)
        else:
            current = date(year, month + 1, 1)

    # Calculate stats
    total_files = len(available_dates)
    total_size = sum(info["size"] for info in available_dates.values())
    trading_days = len([d for d in available_dates.keys() if start_date <= date.fromisoformat(d) <= end_date]) if available_dates else 0

    # Estimate total trading days in range (rough: 252/365 ratio)
    total_days = (end_date - start_date).days + 1
    estimated_trading_days = int(total_days * 252 / 365)
    coverage_pct = (trading_days / estimated_trading_days * 100) if estimated_trading_days > 0 else 0

    stats = {
        "total_files": total_files,
        "total_size_formatted": format_size(total_size) if total_size > 0 else "0 B",
        "dates_with_data": trading_days,
        "coverage_pct": min(100, coverage_pct),
    }

    return jsonify({"months": months, "stats": stats})


@app.route("/api/dataset/file")
def api_dataset_file():
    """Get file info for a specific date in a dataset."""
    dataset = request.args.get("dataset", "stocks-1m")
    underlying = request.args.get("underlying", "SPY")
    date_str = request.args.get("date", "")

    if not date_str:
        return jsonify(None)

    try:
        from datetime import date as date_type
        d = date_type.fromisoformat(date_str)
    except ValueError:
        return jsonify(None)

    # Get file path using registry
    registry = get_registry()
    if registry.exists(dataset):
        file_path = Path(registry.resolve_path(dataset, underlying, d))
    else:
        # Fallback for unknown datasets
        is_daily = is_daily_dataset(dataset)
        if is_daily:
            file_path = DATA_DIR / dataset / underlying / f"{d.year}.parquet"
        else:
            file_path = DATA_DIR / dataset / underlying / f"{d.year}-{d.month:02d}" / f"{d.day:02d}.parquet"

    if not file_path.exists():
        return jsonify(None)

    # Get file info
    stat = file_path.stat()
    size = stat.st_size

    # Get row count
    rows = None
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(file_path)
        rows = pf.metadata.num_rows
    except Exception:
        pass

    def format_size(size_bytes):
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.1f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes} B"

    # Return path relative to BASE_DIR to match tree structure
    try:
        relative_path = str(file_path.relative_to(BASE_DIR))
    except ValueError:
        relative_path = str(file_path)

    return jsonify({
        "name": file_path.name,
        "path": relative_path,
        "size": size,
        "size_formatted": format_size(size),
        "rows": rows,
    })


# =============================================================================
# Page Routes
# =============================================================================

@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/checkpoints")
def checkpoints_page():
    """Checkpoints browser page."""
    return render_template("checkpoints.html")


@app.route("/data")
@app.route("/data/<tab>")
def data_page(tab=None):
    """Data browser page with optional tab (browse, files, pipeline)."""
    return render_template("data.html")


@app.route("/reports")
def reports_page():
    """Reports browser page."""
    return render_template("reports.html")


@app.route("/parquet")
def parquet_page():
    """Parquet viewer page."""
    return render_template("parquet.html")


@app.route("/strategies")
def strategies_page():
    """Strategies browser page."""
    return render_template("strategies.html")


@app.route("/backtests")
def backtests_page():
    """Backtests list page."""
    return render_template("backtests.html")


# =============================================================================
# Charts Page API Endpoints
# =============================================================================

@app.route("/api/charts/search")
def api_charts_search():
    """Search for tickers - returns common tickers matching query."""
    query = request.args.get("q", "").upper().strip()
    if not query:
        return jsonify([])

    # Common tickers for quick search (cached in memory)
    common_tickers = [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "AMD", "name": "Advanced Micro Devices"},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF"},
        {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"},
        {"symbol": "VXX", "name": "iPath S&P 500 VIX Short-Term Futures ETN"},
        {"symbol": "GLD", "name": "SPDR Gold Trust"},
        {"symbol": "SLV", "name": "iShares Silver Trust"},
        {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
        {"symbol": "XLF", "name": "Financial Select Sector SPDR Fund"},
        {"symbol": "XLE", "name": "Energy Select Sector SPDR Fund"},
        {"symbol": "XLK", "name": "Technology Select Sector SPDR Fund"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "BAC", "name": "Bank of America Corporation"},
        {"symbol": "WMT", "name": "Walmart Inc."},
        {"symbol": "V", "name": "Visa Inc."},
        {"symbol": "MA", "name": "Mastercard Inc."},
    ]

    # Filter by symbol or name containing the query
    results = []
    for ticker in common_tickers:
        if query in ticker["symbol"] or query.lower() in ticker["name"].lower():
            results.append(ticker)
            if len(results) >= 10:
                break

    # Sort by exact match first, then by how early the match appears
    results.sort(key=lambda x: (0 if x["symbol"] == query else 1, x["symbol"]))
    return jsonify(results[:10])


@app.route("/api/charts/ohlcv")
def api_charts_ohlcv():
    """Get OHLCV data for quick initial chart rendering."""
    from datetime import date as date_type

    date_str = request.args.get("date")
    underlying = request.args.get("underlying", "SPY").upper()

    if not date_str:
        date_str = date_type.today().isoformat()

    try:
        # Try to load from stocks-1m parquet
        stocks_path = DATA_DIR / "stocks-1m" / underlying / f"{date_str[:7]}" / f"{date_str[-2:]}.parquet"

        if stocks_path.exists():
            df = pd.read_parquet(stocks_path)

            # Determine time column - check common names
            time_col = None
            for col_name in ['timestamp', 'window_start', 'time', 'datetime']:
                if col_name in df.columns:
                    time_col = col_name
                    break
            if time_col is None and df.index.name and 'time' in df.index.name.lower():
                time_col = df.index.name
                df = df.reset_index()
            if time_col is None:
                time_col = 'window_start'  # Default fallback

            df = df.sort_values(time_col)

            # Ensure timestamp is timezone-aware
            if not hasattr(df[time_col].dtype, 'tz') or df[time_col].dt.tz is None:
                df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize('UTC')
            else:
                df[time_col] = pd.to_datetime(df[time_col]).dt.tz_convert('UTC')

            # Filter to market hours (9:30 AM - 4:00 PM ET)
            df['hour'] = df[time_col].dt.hour
            df['minute'] = df[time_col].dt.minute
            market_mask = ((df['hour'] > 14) | ((df['hour'] == 14) & (df['minute'] >= 30))) & (df['hour'] < 21)
            df = df[market_mask]

            # Build candlestick and volume data
            candles = []
            volume = []
            for _, row in df.iterrows():
                # Skip rows with incomplete OHLC data (causes "Value is null" error in charts)
                if pd.isna(row["open"]) or pd.isna(row["high"]) or pd.isna(row["low"]) or pd.isna(row["close"]):
                    continue
                ts = int(row[time_col].timestamp())
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row.get("volume", 0)) if pd.notna(row.get("volume", 0)) else 0
                candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})
                volume.append({
                    "time": ts,
                    "value": v,
                    "color": "rgba(34, 197, 94, 0.5)" if c >= o else "rgba(239, 68, 68, 0.5)"
                })

            return jsonify({
                "status": "ok",
                "source": "parquet",
                "underlying": underlying,
                "date": date_str,
                "candles": candles,
                "volume": volume,
                "rows": len(candles),
            })

        # If no local data, fetch from Alpaca
        from src.data.providers.alpaca import AlpacaProvider
        provider = AlpacaProvider()
        target_date = date_type.fromisoformat(date_str)
        df = provider.fetch_bars_range(underlying, target_date, target_date, feed="sip")

        if df is not None and not df.empty:
            candles = []
            volume = []
            for ts, row in df.iterrows():
                # Handle both Timestamp and int index types
                if hasattr(ts, 'timestamp'):
                    time_val = int(ts.timestamp())
                elif isinstance(ts, (int, float)):
                    time_val = int(ts)
                else:
                    time_val = int(pd.Timestamp(ts).timestamp())

                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row.get("volume", 0))
                candles.append({"time": time_val, "open": o, "high": h, "low": l, "close": c})
                volume.append({
                    "time": time_val,
                    "value": v,
                    "color": "rgba(34, 197, 94, 0.5)" if c >= o else "rgba(239, 68, 68, 0.5)"
                })
            return jsonify({
                "status": "ok",
                "source": "alpaca",
                "underlying": underlying,
                "date": date_str,
                "candles": candles,
                "volume": volume,
                "rows": len(candles),
            })

        return jsonify({
            "status": "no_data",
            "underlying": underlying,
            "date": date_str,
            "message": "No data available for this date",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/charts/data")
def api_charts_data():
    """Get full training-1m-raw data with all features."""
    from datetime import date as date_type

    date_str = request.args.get("date")
    underlying = request.args.get("underlying", "SPY").upper()

    if not date_str:
        date_str = date_type.today().isoformat()

    try:
        # Try training-1m-raw path
        raw_path = DATA_DIR / "training-1m-raw" / underlying / f"{date_str[:7]}" / f"{date_str[-2:]}.parquet"

        if not raw_path.exists():
            # Check cache path
            raw_path = DATA_DIR / "cache" / "raw" / underlying / f"{date_str}.parquet"

        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            return _chart_data_training_raw(df, str(raw_path))

        # If no training data, return status indicating data needs to be loaded
        return jsonify({
            "status": "not_loaded",
            "underlying": underlying,
            "date": date_str,
            "message": "Training data not available. Trigger /api/charts/load to generate.",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/charts/load", methods=["POST"])
def api_charts_load():
    """Trigger DAG loader to generate training data for a date."""
    from datetime import date as date_type

    data = request.get_json() or {}
    date_str = data.get("date") or request.args.get("date")
    underlying = (data.get("underlying") or request.args.get("underlying", "SPY")).upper()

    if not date_str:
        date_str = date_type.today().isoformat()

    try:
        from src.data.dag.loader import DAGLoader

        loader = DAGLoader()
        target_date = date_type.fromisoformat(date_str)

        # Load training-1m-raw for this date
        result = loader.load_day("training-1m-raw", underlying, target_date)

        if result:
            return jsonify({
                "status": "ok",
                "message": f"Data loaded for {underlying} on {date_str}",
                "path": str(result) if hasattr(result, "__str__") else "loaded",
            })
        else:
            return jsonify({
                "status": "failed",
                "message": f"Failed to load data for {underlying} on {date_str}",
            })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/charts/realtime")
def api_charts_realtime():
    """Get realtime price/volume data from Alpaca."""
    import arrow
    from datetime import date as date_type

    underlying = request.args.get("underlying", "SPY").upper()

    try:
        from src.data.providers.alpaca import AlpacaProvider
        from src.utils.market_hours import get_market_hours_utc

        provider = AlpacaProvider()

        # Get current time
        now = arrow.now("UTC")
        today = now.date()

        # Check if within market hours (including extended hours 4 AM - 8 PM ET)
        # Extended hours: 4:00 AM - 8:00 PM ET = 9:00 - 1:00 UTC (next day)
        # We'll be generous and check regular market hours only for simplicity
        market_open, market_close = get_market_hours_utc(today)

        # Check if market is open (with 30 min buffer for pre/post)
        is_market_hours = market_open <= now.naive <= market_close

        # Get latest bar and quote
        bar = provider.get_latest_bar(underlying, feed="sip")
        quote = provider.get_latest_quote(underlying, feed="sip")

        result = {
            "underlying": underlying,
            "timestamp": now.isoformat(),
            "is_market_hours": is_market_hours,
            "market_open": market_open.isoformat() if market_open else None,
            "market_close": market_close.isoformat() if market_close else None,
        }

        if bar:
            result["bar"] = {
                "time": int(bar["timestamp"].timestamp()) if bar.get("timestamp") else None,
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
                "vwap": bar.get("vwap"),
            }

        if quote:
            result["quote"] = {
                "bid": quote.get("bid_price"),
                "ask": quote.get("ask_price"),
                "bid_size": quote.get("bid_size"),
                "ask_size": quote.get("ask_size"),
                "mid": (quote.get("bid_price", 0) + quote.get("ask_price", 0)) / 2 if quote.get("bid_price") and quote.get("ask_price") else None,
            }

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/charts/sync-minute", methods=["POST"])
def api_charts_sync_minute():
    """Sync a completed minute's data with full feature computation."""
    from datetime import date as date_type

    data = request.get_json() or {}
    date_str = data.get("date") or request.args.get("date")
    minute = data.get("minute")  # Format: "HH:MM"
    underlying = (data.get("underlying") or request.args.get("underlying", "SPY")).upper()

    if not date_str or not minute:
        return jsonify({"error": "date and minute parameters required"}), 400

    try:
        # For now, trigger a full day reload (future: incremental minute sync)
        from src.data.dag.loader import DAGLoader

        loader = DAGLoader()
        target_date = date_type.fromisoformat(date_str)

        result = loader.load_day("training-1m-raw", underlying, target_date)

        return jsonify({
            "status": "ok" if result else "failed",
            "underlying": underlying,
            "date": date_str,
            "minute": minute,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/charts")
def charts_page():
    """Fullscreen charts page with realtime updates."""
    from datetime import date
    return render_template("charts.html", default_date=date.today().isoformat(), default_underlying="SPY")


@app.route("/backtest/<path:name>")
def backtest_viewer_page(name: str):
    """Backtest viewer page."""
    return render_template("backtest_viewer.html", backtest_name=name)


@app.route("/live")
def live_page():
    """Live paper trading dashboard."""
    return render_template("live.html")


# =============================================================================
# Live Trading API Endpoints
# =============================================================================

_trading_service = None
_simulation_service = None
_active_mode = "none"  # "none" | "live" | "sim"


def _get_trading_service():
    """Get the TradingService singleton (lazy import)."""
    global _trading_service
    if _trading_service is None:
        from src.execution.trading_service import TradingService
        _trading_service = TradingService.instance()
    return _trading_service


def _get_simulation_service():
    """Get the SimulationService singleton (lazy import)."""
    global _simulation_service
    if _simulation_service is None:
        from src.execution.simulation_service import SimulationService
        _simulation_service = SimulationService.instance()
    return _simulation_service


def _get_active_service():
    """Get whichever service is currently active, or None."""
    if _active_mode == "live":
        return _get_trading_service()
    elif _active_mode == "sim":
        return _get_simulation_service()
    return None


@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    """Start the paper trader or simulation."""
    global _active_mode
    data = request.get_json() or {}
    mode = data.get("mode", "live")

    try:
        if mode == "sim":
            service = _get_simulation_service()
            service.start(
                underlying=data.get("underlying", "SPY"),
                start_date=data.get("start_date"),
                end_date=data.get("end_date"),
                capital=data.get("capital", 25000.0),
                speed=data.get("speed", 30.0),
                use_heuristic=data.get("use_heuristic", True),
                dry_run=True,
            )
            _active_mode = "sim"
        else:
            service = _get_trading_service()
            service.start(
                capital=data.get("capital", 25000.0),
                session_id=data.get("session_id", "paper_trading"),
                use_heuristic=data.get("use_heuristic", True),
                dry_run=data.get("dry_run", False),
            )
            _active_mode = "live"
        return jsonify({"ok": True, "mode": _active_mode})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    """Stop whichever service is active."""
    global _active_mode
    service = _get_active_service()
    if service is not None:
        service.stop()
    _active_mode = "none"
    return jsonify({"ok": True})


@app.route("/api/live/snapshot")
def api_live_snapshot():
    """Get current state snapshot from the active service."""
    service = _get_active_service()
    if service is None:
        # Return a minimal stopped snapshot without touching the database
        return jsonify({
            "is_running": False,
            "error": None,
            "started_at": None,
            "config": {},
            "timestamp": None,
        })
    snapshot = service.get_snapshot()
    return jsonify(snapshot)


@app.route("/api/live/trades")
def api_live_trades():
    """Get trade history and daily stats from the active service."""
    service = _get_active_service()
    if service is None:
        return jsonify({
            "trades": [],
            "daily_trades": 0,
            "daily_pnl": 0.0,
            "capital": 25000.0,
            "win_rate": 0.0,
        })
    trades = service.get_trades()
    return jsonify(trades)


@app.route("/api/live/candles")
def api_live_candles():
    """Get full chart data (OHLCV, volume, VWAP, OR) for TrainingChartView."""
    service = _get_active_service()
    if service is None:
        return jsonify({})
    return jsonify(service.get_chart_data())


@app.route("/api/live/speed", methods=["POST"])
def api_live_speed():
    """Change simulation playback speed."""
    if _active_mode != "sim":
        return jsonify({"ok": False, "error": "Not in simulation mode"}), 400
    data = request.get_json() or {}
    speed = data.get("speed", 30.0)
    service = _get_simulation_service()
    service.set_speed(speed)
    return jsonify({"ok": True, "speed": speed})


@app.route("/api/live/seek", methods=["POST"])
def api_live_seek():
    """Seek to a specific bar in the simulation."""
    if _active_mode != "sim":
        return jsonify({"ok": False, "error": "Not in simulation mode"}), 400
    data = request.get_json() or {}
    bar_index = data.get("bar_index", 0)
    service = _get_simulation_service()
    service.seek(bar_index)
    return jsonify({"ok": True, "bar_index": bar_index})


@app.route("/api/live/pause", methods=["POST"])
def api_live_pause():
    """Pause or resume the simulation."""
    if _active_mode != "sim":
        return jsonify({"ok": False, "error": "Not in simulation mode"}), 400
    data = request.get_json() or {}
    paused = data.get("paused", True)
    service = _get_simulation_service()
    if paused:
        service.pause()
    else:
        service.resume()
    return jsonify({"ok": True, "paused": paused})


# =============================================================================
# Main
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8050, debug: bool = True):
    """Run the web UI server."""
    # Ensure template directory exists
    (Path(__file__).parent / "templates").mkdir(exist_ok=True)
    (Path(__file__).parent / "static").mkdir(exist_ok=True)

    print(f"Starting Spymaster Web UI...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Checkpoints data: {CHECKPOINTS_DATA_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"\nOpen http://{host}:{port} in your browser")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
