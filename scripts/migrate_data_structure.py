#!/usr/bin/env python3
"""
Migrate data from flat structure to hierarchical structure.

Old structure: data/stocks/SPY_STOCKS-1M_2021-01-04.parquet
New structure: data/stocks-1m/SPY/2021-01/04.parquet

This script:
1. Scans existing data directories
2. Parses filenames to extract date and underlying
3. Creates new directory structure
4. Moves/copies files to new locations
5. Verifies data integrity
"""

import argparse
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Migration mappings: (old_dir, old_pattern, new_dir)
MIGRATIONS = [
    # Source data
    ("data/stocks", r"SPY_STOCKS-1M_(\d{4}-\d{2}-\d{2})\.parquet", "data/stocks-1m"),
    ("data/options", r"SPY_OPTIONS-1M_(\d{4}-\d{2}-\d{2})\.parquet", "data/options-1m"),
    ("data/oi", r"SPY_OI-0DTE_(\d{4}-\d{2}-\d{2})\.parquet", "data/oi-day"),
    ("data/greeks", r"SPY_GREEKS-1M_(\d{4}-\d{2}-\d{2})\.parquet", "data/greeks-1m"),
    # Computed data
    ("data/gex_flow", r"SPY_thetadata_1m_combined_(\d{4}-\d{2}-\d{2})\.parquet", "data/options-flow-1m"),
    # Training data (cache)
    ("data/cache/raw", r"SPY_raw_(\d{4}-\d{2}-\d{2})\.parquet", "data/training-1m-raw"),
    ("data/cache/normalized", r"SPY_v\d+_(\d{4}-\d{2}-\d{2})\.parquet", "data/training-1m-normalized"),
]


@dataclass
class MigrationTask:
    """Represents a single file migration task."""
    src_path: Path
    dst_path: Path
    underlying: str
    date_str: str


def parse_filename(filename: str, pattern: str) -> Optional[Tuple[str, str]]:
    """Parse filename to extract underlying and date.

    Returns (underlying, date_str) or None if no match.
    """
    match = re.match(pattern, filename)
    if match:
        date_str = match.group(1)
        # For now, all files are SPY
        return ("SPY", date_str)
    return None


def build_new_path(base_dir: str, underlying: str, date_str: str) -> Path:
    """Build new hierarchical path from date string.

    Example: base_dir=data/stocks-1m, underlying=SPY, date_str=2021-01-04
    Returns: data/stocks-1m/SPY/2021-01/04.parquet
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year_month = date.strftime("%Y-%m")
    day = date.strftime("%d")
    return Path(base_dir) / underlying / year_month / f"{day}.parquet"


def discover_migrations(project_root: Path, dry_run: bool = False) -> List[MigrationTask]:
    """Discover all files that need to be migrated."""
    tasks = []

    for old_dir, pattern, new_dir in MIGRATIONS:
        old_path = project_root / old_dir
        if not old_path.exists():
            print(f"  Skipping {old_dir} (does not exist)")
            continue

        for file_path in old_path.glob("*.parquet"):
            result = parse_filename(file_path.name, pattern)
            if result:
                underlying, date_str = result
                new_path = project_root / build_new_path(new_dir, underlying, date_str)
                tasks.append(MigrationTask(
                    src_path=file_path,
                    dst_path=new_path,
                    underlying=underlying,
                    date_str=date_str,
                ))

    return tasks


def migrate_file(task: MigrationTask, copy: bool = False, dry_run: bool = False) -> Tuple[bool, str]:
    """Migrate a single file.

    Returns (success, message).
    """
    try:
        if dry_run:
            return (True, f"Would {'copy' if copy else 'move'}: {task.src_path} -> {task.dst_path}")

        # Create destination directory
        task.dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if destination already exists
        if task.dst_path.exists():
            # Verify same size
            if task.dst_path.stat().st_size == task.src_path.stat().st_size:
                return (True, f"Already exists (same size): {task.dst_path}")
            else:
                return (False, f"Destination exists with different size: {task.dst_path}")

        if copy:
            shutil.copy2(task.src_path, task.dst_path)
        else:
            shutil.move(task.src_path, task.dst_path)

        return (True, f"{'Copied' if copy else 'Moved'}: {task.src_path.name} -> {task.dst_path}")

    except Exception as e:
        return (False, f"Error migrating {task.src_path}: {e}")


def migrate_all(
    project_root: Path,
    copy: bool = False,
    dry_run: bool = False,
    max_workers: int = 8,
) -> Tuple[int, int]:
    """Migrate all files.

    Returns (success_count, error_count).
    """
    print("=" * 60)
    print("Data Structure Migration")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN' if dry_run else ('COPY' if copy else 'MOVE')}")
    print()

    # Discover files to migrate
    print("Discovering files to migrate...")
    tasks = discover_migrations(project_root, dry_run)
    print(f"Found {len(tasks)} files to migrate")
    print()

    if not tasks:
        print("No files to migrate.")
        return (0, 0)

    # Group by dataset for reporting
    by_dataset = {}
    for task in tasks:
        dataset = task.dst_path.parent.parent.parent.name
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(task)

    print("Files by dataset:")
    for dataset, dataset_tasks in sorted(by_dataset.items()):
        print(f"  {dataset}: {len(dataset_tasks)} files")
    print()

    # Migrate files
    print("Migrating files...")
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(migrate_file, task, copy, dry_run): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            success, message = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"  ERROR: {message}")

    print()
    print("=" * 60)
    print(f"Migration complete: {success_count} succeeded, {error_count} failed")
    print("=" * 60)

    return (success_count, error_count)


def verify_migration(project_root: Path) -> bool:
    """Verify that new structure has expected files."""
    print()
    print("Verifying new structure...")

    new_dirs = [
        "data/stocks-1m",
        "data/options-1m",
        "data/oi-day",
        "data/greeks-1m",
        "data/options-flow-1m",
        "data/training-1m-raw",
        "data/training-1m-normalized",
    ]

    for dir_name in new_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            # Count files recursively
            file_count = sum(1 for _ in dir_path.rglob("*.parquet"))
            print(f"  {dir_name}: {file_count} files")
        else:
            print(f"  {dir_name}: NOT CREATED")

    return True


def cleanup_old_structure(project_root: Path, dry_run: bool = False) -> None:
    """Remove old directory structure after successful migration."""
    old_dirs = [
        "data/stocks",
        "data/options",
        "data/oi",
        "data/greeks",
        "data/gex_flow",
        "data/cache/raw",
        "data/cache/normalized",
    ]

    print()
    print("Cleaning up old structure...")

    for dir_name in old_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            # Check if directory is empty or only has hidden files
            remaining = list(dir_path.glob("*.parquet"))
            if remaining:
                print(f"  {dir_name}: {len(remaining)} files remaining, skipping cleanup")
            else:
                if dry_run:
                    print(f"  Would remove: {dir_name}")
                else:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  Removed: {dir_name}")
                    except Exception as e:
                        print(f"  Error removing {dir_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data to new hierarchical structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (safer but uses more disk space)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old directories after migration",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Run migration
    success, errors = migrate_all(
        project_root,
        copy=args.copy,
        dry_run=args.dry_run,
        max_workers=args.workers,
    )

    # Verify
    if not args.dry_run and success > 0:
        verify_migration(project_root)

    # Cleanup old structure
    if args.cleanup and not args.dry_run and errors == 0:
        cleanup_old_structure(project_root, dry_run=args.dry_run)

    # Exit with error code if there were failures
    exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
