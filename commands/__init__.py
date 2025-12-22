"""
Commands package - CLI command modules for SpyMaster.

Each module contains a Typer app for a specific command group.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

# Shared console instance
console = Console()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
BACKTESTS_DIR = PROJECT_ROOT / "backtests"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DATA_DIR = PROJECT_ROOT / "data"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_timestamp(timestamp: float) -> str:
    """Format timestamp as human-readable date."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "Trained": "green",
        "In Progress": "yellow",
        "Pending": "blue",
        "Not Found": "red",
        "Empty": "dim",
        "Running": "yellow",
        "Completed": "green",
        "Failed": "red",
    }
    return colors.get(status, "white")


def _get_dag_loader():
    """Get or create a DAGLoader instance."""
    from src.data.dag import DAGLoader
    return DAGLoader()
