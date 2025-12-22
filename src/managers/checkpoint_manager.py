"""Checkpoint manager - uses database for metadata storage.

Data files (.pt models) are stored in data/checkpoints/{name}/
Metadata (configuration, metrics) is stored in the database.
"""

import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.db import get_db, Database, Checkpoint


def _generate_id() -> str:
    """Generate a new UUID for a record."""
    return str(uuid.uuid4())


@dataclass
class CheckpointConfig:
    """Configuration for a checkpoint (for API compatibility)."""
    name: str
    id: str = field(default_factory=_generate_id)
    model_type: str = "lejepa"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "created": self.created,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "data": self.data,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointConfig":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", "unknown"),
            model_type=data.get("model_type", "lejepa"),
            created=data.get("created", ""),
            status=data.get("status", "pending"),
            hyperparameters=data.get("hyperparameters", {}),
            data=data.get("data", {}),
            metrics=data.get("metrics", {}),
        )

    @classmethod
    def from_db_model(cls, checkpoint: Checkpoint) -> "CheckpointConfig":
        """Create CheckpointConfig from database Checkpoint model."""
        # Extract data config from hyperparameters
        hp = checkpoint.hyperparameters or {}
        data_config = {
            "underlying": checkpoint.underlying,
            "start_date": checkpoint.start_date,
            "end_date": checkpoint.end_date,
        }
        # Copy extra data fields from hyperparameters
        for key in ["lejepa_checkpoint", "label_distribution"]:
            if key in hp:
                data_config[key] = hp.pop(key)

        return cls(
            id=checkpoint.id,
            name=checkpoint.name,
            model_type=checkpoint.model_type,
            created=checkpoint.created_at.isoformat() if checkpoint.created_at else "",
            status=checkpoint.status,
            hyperparameters=hp,
            data=data_config,
            metrics=checkpoint.metrics or {},
        )


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    name: str
    path: Path
    status: str
    config: Optional[CheckpointConfig]
    total_size: int
    modified: float
    files: List[Path]


class CheckpointManager:
    """Manager for model checkpoints using database for metadata."""

    def __init__(self, checkpoints_dir: Path, project_root: Path, db: Optional[Database] = None):
        self.checkpoints_dir = checkpoints_dir  # Legacy directory for config files
        self.project_root = project_root
        self.data_checkpoints_dir = project_root / "data" / "checkpoints"
        self.data_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._db = db

    @property
    def db(self) -> Database:
        """Get database instance (lazy initialization)."""
        if self._db is None:
            self._db = get_db()
        return self._db

    def _get_status_from_files(self, name: str) -> str:
        """Determine checkpoint training status from files."""
        data_dir = self.data_checkpoints_dir / name

        if not data_dir.exists():
            return "pending"

        # Check for trained model files
        best_pt = data_dir / "lejepa_best.pt"
        entry_best = data_dir / "entry_policy_best.pt"
        exit_best = data_dir / "exit_policy_best.pt"

        if best_pt.exists() or entry_best.exists() or exit_best.exists():
            return "trained"

        # Check for in-progress training
        latest_pt = data_dir / "lejepa_latest.pt"
        if latest_pt.exists():
            return "in_progress"

        return "pending"

    def list_all(self) -> List[CheckpointInfo]:
        """List all checkpoints from database."""
        checkpoints = []
        db_checkpoints = self.db.list_checkpoints()

        for cp in db_checkpoints:
            info = self._checkpoint_to_info(cp)
            if info:
                checkpoints.append(info)

        return checkpoints

    def _checkpoint_to_info(self, checkpoint: Checkpoint) -> Optional[CheckpointInfo]:
        """Convert database Checkpoint to CheckpointInfo."""
        data_dir = self.data_checkpoints_dir / checkpoint.name

        # Calculate file info
        all_files = []
        total_size = 0

        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    all_files.append(f)
                    total_size += f.stat().st_size

        # Get actual status from files
        file_status = self._get_status_from_files(checkpoint.name)
        # Use database status if it's more specific, otherwise use file status
        status = checkpoint.status if checkpoint.status in ["trained", "failed"] else file_status

        config = CheckpointConfig.from_db_model(checkpoint)
        modified = data_dir.stat().st_mtime if data_dir.exists() else 0

        return CheckpointInfo(
            name=checkpoint.name,
            path=data_dir,
            status=status,
            config=config,
            total_size=total_size,
            modified=modified,
            files=all_files,
        )

    def get(self, name: str) -> Optional[CheckpointInfo]:
        """Get checkpoint information by name."""
        checkpoint = self.db.get_checkpoint(name)
        if not checkpoint:
            return None
        return self._checkpoint_to_info(checkpoint)

    def exists(self, name: str) -> bool:
        """Check if checkpoint exists in database."""
        return self.db.checkpoint_exists(name)

    def create(
        self,
        name: str,
        model_type: str = "lejepa",
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        patch_length: int = 32,
        underlying: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 2048,
        lr: float = 1e-4,
        lambda_sigreg: float = 0.1,
        # Entry policy specific
        lejepa_checkpoint: Optional[str] = None,
        lookahead: int = 15,
        min_roi_threshold: float = 20.0,
        otm_buffer: float = 1.2,
        focal_alpha_hold: float = 0.35,
        focal_alpha_atm: float = 1.0,
        focal_alpha_otm: float = 1.2,
        slippage_pct: float = 0.5,
        execution_delay: int = 1,
        # Partial patches
        include_partial_patches: bool = False,
        min_partial_length: int = 8,
    ) -> CheckpointConfig:
        """Create a new checkpoint configuration."""
        # Build hyperparameters dict
        hyperparameters = {
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "patch_length": patch_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lambda_sigreg": lambda_sigreg,
            "include_partial_patches": include_partial_patches,
            "min_partial_length": min_partial_length,
        }

        # Add entry-specific hyperparameters
        if model_type in ["entry-3class", "entry-5class"]:
            hyperparameters.update({
                "lookahead": lookahead,
                "focal_alpha_hold": focal_alpha_hold,
            })
            if lejepa_checkpoint:
                hyperparameters["lejepa_checkpoint"] = lejepa_checkpoint

        if model_type == "entry-5class":
            hyperparameters.update({
                "min_roi_threshold": min_roi_threshold,
                "otm_buffer": otm_buffer,
                "focal_alpha_atm": focal_alpha_atm,
                "focal_alpha_otm": focal_alpha_otm,
                "slippage_pct": slippage_pct,
                "execution_delay": execution_delay,
            })

        # Create in database
        checkpoint = self.db.create_checkpoint(
            name=name,
            model_type=model_type,
            status="pending",
            hyperparameters=hyperparameters,
            underlying=underlying,
            start_date=start_date,
            end_date=end_date,
            data_path=f"data/checkpoints/{name}",
        )

        # Create data directory
        data_dir = self.data_checkpoints_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)

        return CheckpointConfig.from_db_model(checkpoint)

    def remove(self, name: str) -> bool:
        """Remove a checkpoint from database and filesystem."""
        # Delete from database
        deleted = self.db.delete_checkpoint(name)

        # Also remove data directory
        data_dir = self.data_checkpoints_dir / name
        if data_dir.exists():
            shutil.rmtree(data_dir)

        # Remove legacy config directory if exists
        legacy_dir = self.checkpoints_dir / name
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)

        return deleted

    def build_train_command(
        self,
        name: str,
        resume: bool = False,
    ) -> Optional[List[str]]:
        """Build the training command for a checkpoint."""
        info = self.get(name)
        if not info or not info.config:
            return None

        hp = info.config.hyperparameters
        data = info.config.data
        model_type = info.config.model_type

        # Map model_type to train.py --model-type argument
        model_type_map = {
            "lejepa": "jepa",
            "entry-3class": "entry-3class",
            "entry-5class": "entry-5class",
        }
        train_model_type = model_type_map.get(model_type, "jepa")

        cmd = [
            "uv", "run", "python", "train.py",
            "--model-type", train_model_type,
            "--checkpoint-id", name,
            "--patch-length", str(hp.get("patch_length", 32)),
            "--epochs", str(hp.get("epochs", 10)),
            "--batch-size", str(hp.get("batch_size", 2048)),
            "--lr", str(hp.get("learning_rate", 1e-4)),
            "--underlying", data.get("underlying", "SPY"),
        ]

        # LeJEPA specific args
        if model_type == "lejepa":
            cmd.extend([
                "--embedding-dim", str(hp.get("embedding_dim", 256)),
                "--num-layers", str(hp.get("num_layers", 4)),
                "--num-heads", str(hp.get("num_heads", 8)),
                "--lambda-sigreg", str(hp.get("lambda_sigreg", 0.1)),
            ])

        # Entry policy specific args
        if model_type in ["entry-3class", "entry-5class"]:
            lejepa_ckpt = data.get("lejepa_checkpoint")
            if lejepa_ckpt:
                cmd.extend(["--lejepa-checkpoint", lejepa_ckpt])
            cmd.extend([
                "--lookahead", str(hp.get("lookahead", 15)),
                "--focal-alpha-hold", str(hp.get("focal_alpha_hold", 0.35)),
            ])

        # Entry-5class specific args
        if model_type == "entry-5class":
            cmd.extend([
                "--min-roi-threshold", str(hp.get("min_roi_threshold", 20.0)),
                "--otm-buffer", str(hp.get("otm_buffer", 1.2)),
                "--focal-alpha-atm", str(hp.get("focal_alpha_atm", 1.0)),
                "--focal-alpha-otm", str(hp.get("focal_alpha_otm", 1.2)),
                "--slippage-pct", str(hp.get("slippage_pct", 0.5)),
                "--execution-delay", str(hp.get("execution_delay", 1)),
            ])
            if hp.get("no_time_decay", False):
                cmd.append("--no-time-decay")

        # Common data args
        if data.get("start_date"):
            cmd.extend(["--start-date", data["start_date"]])
        if data.get("end_date"):
            cmd.extend(["--end-date", data["end_date"]])
        if resume:
            cmd.append("--resume")

        # Partial patches
        if hp.get("include_partial_patches", False):
            cmd.append("--include-partial-patches")
            min_partial = hp.get("min_partial_length", 8)
            if min_partial != 8:
                cmd.extend(["--min-partial-length", str(min_partial)])

        return cmd

    def train(
        self,
        name: str,
        resume: bool = False,
        background: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Start training a checkpoint."""
        cmd = self.build_train_command(name, resume)
        if not cmd:
            return None

        if background:
            return subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            subprocess.run(cmd, cwd=self.project_root)
            return None

    def get_training_logs(self, name: str) -> Optional[str]:
        """Get training logs for a checkpoint."""
        data_dir = self.data_checkpoints_dir / name
        log_file = data_dir / "training.log"

        if log_file.exists():
            with open(log_file) as f:
                return f.read()
        return None

    def update(
        self,
        name: str,
        model_type: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        patch_length: Optional[int] = None,
        underlying: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_sigreg: Optional[float] = None,
    ) -> Optional[CheckpointConfig]:
        """Update an existing checkpoint configuration."""
        checkpoint = self.db.get_checkpoint(name)
        if not checkpoint:
            return None

        # Build update kwargs
        update_kwargs = {}

        if model_type is not None:
            update_kwargs["model_type"] = model_type
        if underlying is not None:
            update_kwargs["underlying"] = underlying
        if start_date is not None:
            update_kwargs["start_date"] = start_date
        if end_date is not None:
            update_kwargs["end_date"] = end_date

        # Update hyperparameters
        hp = dict(checkpoint.hyperparameters or {})
        if embedding_dim is not None:
            hp["embedding_dim"] = embedding_dim
        if num_layers is not None:
            hp["num_layers"] = num_layers
        if num_heads is not None:
            hp["num_heads"] = num_heads
        if patch_length is not None:
            hp["patch_length"] = patch_length
        if epochs is not None:
            hp["epochs"] = epochs
        if batch_size is not None:
            hp["batch_size"] = batch_size
        if lr is not None:
            hp["learning_rate"] = lr
        if lambda_sigreg is not None:
            hp["lambda_sigreg"] = lambda_sigreg

        update_kwargs["hyperparameters"] = hp

        updated = self.db.update_checkpoint(name, **update_kwargs)
        if updated:
            return CheckpointConfig.from_db_model(updated)
        return None

    def set_label_distribution(
        self,
        name: str,
        train_distribution: Dict[str, int],
        val_distribution: Optional[Dict[str, int]] = None,
    ) -> Optional[CheckpointConfig]:
        """Set the label distribution for a checkpoint."""
        checkpoint = self.db.get_checkpoint(name)
        if not checkpoint:
            return None

        # Store in hyperparameters
        hp = dict(checkpoint.hyperparameters or {})
        hp["label_distribution"] = {
            "train": train_distribution,
        }
        if val_distribution:
            hp["label_distribution"]["val"] = val_distribution

        updated = self.db.update_checkpoint(name, hyperparameters=hp)
        if updated:
            return CheckpointConfig.from_db_model(updated)
        return None

    def get_or_create(
        self,
        name: str,
        model_type: str = "lejepa",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get existing checkpoint or create a new one."""
        info = self.get(name)
        if info and info.config:
            return {
                "id": info.config.id,
                "name": info.name,
                "path": str(self.data_checkpoints_dir / name),
                "config": info.config.to_dict(),
                "metrics": info.config.metrics,
                "status": info.status,
            }

        # Filter config to only include valid create() parameters
        valid_params = {
            "embedding_dim", "num_layers", "num_heads", "patch_length",
            "underlying", "start_date", "end_date", "epochs", "batch_size",
            "lr", "lambda_sigreg", "lejepa_checkpoint", "lookahead",
            "min_roi_threshold", "otm_buffer", "focal_alpha_hold",
            "focal_alpha_atm", "focal_alpha_otm", "slippage_pct",
            "execution_delay", "include_partial_patches", "min_partial_length",
        }
        filtered_config = {k: v for k, v in (config or {}).items() if k in valid_params}

        # Create new checkpoint
        checkpoint_config = self.create(
            name=name,
            model_type=model_type,
            **filtered_config,
        )

        return {
            "id": checkpoint_config.id,
            "name": name,
            "path": str(self.data_checkpoints_dir / name),
            "config": checkpoint_config.to_dict(),
            "metrics": checkpoint_config.metrics,
            "status": checkpoint_config.status,
        }

    def get_best_checkpoint(self, name: str) -> Optional[Path]:
        """Get the path to the best checkpoint file."""
        info = self.get(name)
        if not info:
            return None

        model_type = info.config.model_type if info.config else "lejepa"
        data_dir = self.data_checkpoints_dir / name

        # Determine best checkpoint filename based on model type
        if "lejepa" in model_type:
            filename = "lejepa_best.pt"
        elif "entry" in model_type:
            filename = "entry_policy_best.pt"
        elif "exit" in model_type:
            filename = "exit_policy_best.pt"
        else:
            filename = None

        if filename and data_dir.exists():
            best_file = data_dir / filename
            if best_file.exists():
                return best_file

        # Try common patterns
        if data_dir.exists():
            for pattern in ["*_best.pt", "best.pt"]:
                matches = list(data_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def get_config_path(self, name: str) -> Optional[Path]:
        """Get path to config - now returns None since configs are in database."""
        # For backwards compatibility, return legacy path if it exists
        legacy_config = self.checkpoints_dir / name / "config.json"
        if legacy_config.exists():
            return legacy_config
        return None

    def set_training_complete(
        self,
        name: str,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "trained",
    ) -> Optional[CheckpointConfig]:
        """Mark a checkpoint as training complete and update metrics."""
        checkpoint = self.db.get_checkpoint(name)
        if not checkpoint:
            return None

        # Merge metrics
        existing_metrics = dict(checkpoint.metrics or {})
        if metrics:
            existing_metrics.update(metrics)

        updated = self.db.update_checkpoint(
            name,
            status=status,
            metrics=existing_metrics,
        )

        if updated:
            return CheckpointConfig.from_db_model(updated)
        return None


# =============================================================================
# Module-level helper functions (for backward compatibility)
# =============================================================================


def get_manager() -> CheckpointManager:
    """Get a CheckpointManager instance with default paths."""
    project_root = Path(__file__).parent.parent.parent
    return CheckpointManager(project_root / "checkpoints", project_root)


def resolve_checkpoint_file(identifier: str) -> Optional[Path]:
    """Resolve a checkpoint identifier to a .pt file path.

    Args:
        identifier: Can be:
            - A direct path to a .pt file
            - A checkpoint name (looks up best checkpoint)

    Returns:
        Path to checkpoint file or None if not found
    """
    # Check if it's a direct path
    path = Path(identifier)
    if path.exists() and path.suffix == ".pt":
        return path

    # Try to look up by name
    mgr = get_manager()
    best = mgr.get_best_checkpoint(identifier)
    if best:
        return best

    return None
