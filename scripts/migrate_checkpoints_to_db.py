#!/usr/bin/env python3
"""Migrate checkpoint metadata from JSON files to database.

This script:
1. Reads all checkpoint config.json files from checkpoints/
2. Inserts them into the SQLite database
3. Verifies the migration
4. Optionally removes the old JSON files
"""

import json
from pathlib import Path

from src.db import get_db, Database


def migrate_checkpoints(dry_run: bool = False, remove_old: bool = False) -> None:
    """Migrate checkpoint configs from JSON to database."""
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "checkpoints"

    if not checkpoints_dir.exists():
        print("No checkpoints directory found - nothing to migrate")
        return

    # Get database
    db = get_db()

    # Find all config.json files
    config_files = list(checkpoints_dir.glob("*/config.json"))
    print(f"Found {len(config_files)} checkpoint config files to migrate")

    migrated = []
    skipped = []

    for config_path in config_files:
        checkpoint_name = config_path.parent.name

        # Check if already exists in database
        if db.checkpoint_exists(checkpoint_name):
            print(f"  SKIP: {checkpoint_name} (already in database)")
            skipped.append(checkpoint_name)
            continue

        # Read JSON config
        with open(config_path) as f:
            config = json.load(f)

        print(f"  Migrating: {checkpoint_name}")

        if dry_run:
            print(f"    [DRY RUN] Would migrate with config: {config}")
            continue

        # Extract fields
        hp = config.get("hyperparameters", {})
        data = config.get("data", {})

        # Merge any data-specific fields into hyperparameters for storage
        hp_with_data = dict(hp)
        for key in ["lejepa_checkpoint", "label_distribution"]:
            if key in data:
                hp_with_data[key] = data[key]

        # Create in database
        checkpoint = db.create_checkpoint(
            name=config["name"],
            model_type=config.get("model_type", "lejepa"),
            status=config.get("status", "pending"),
            hyperparameters=hp_with_data,
            underlying=data.get("underlying", "SPY"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            data_path=f"data/checkpoints/{config['name']}",
        )

        # Update metrics if present
        metrics = config.get("metrics", {})
        if metrics:
            db.update_checkpoint(config["name"], metrics=metrics)

        migrated.append(checkpoint_name)
        print(f"    Created: {checkpoint.id}")

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  Migrated: {len(migrated)}")
    print(f"  Skipped:  {len(skipped)}")

    # Verify migration
    if migrated:
        print("\nVerifying migration...")
        for name in migrated:
            cp = db.get_checkpoint(name)
            if cp:
                print(f"  ✓ {name}: {cp.model_type}, {cp.status}")
            else:
                print(f"  ✗ {name}: NOT FOUND")

    # Remove old files if requested
    if remove_old and not dry_run and migrated:
        print("\nRemoving old JSON files...")
        for name in migrated:
            config_path = checkpoints_dir / name / "config.json"
            if config_path.exists():
                config_path.unlink()
                print(f"  Removed: {config_path}")

            # Remove directory if empty
            checkpoint_dir = checkpoints_dir / name
            if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                checkpoint_dir.rmdir()
                print(f"  Removed empty directory: {checkpoint_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate checkpoints to database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--remove-old", action="store_true", help="Remove old JSON files after migration")
    args = parser.parse_args()

    migrate_checkpoints(dry_run=args.dry_run, remove_old=args.remove_old)
