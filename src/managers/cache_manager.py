"""Cache manager - handles data cache operations."""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class CacheStats:
    """Statistics for a cache type."""
    cache_type: str
    file_count: int
    total_size: int
    date_range: Tuple[str, str]  # (min_date, max_date)
    files: List[Path]


@dataclass
class CacheFile:
    """Information about a single cache file."""
    path: Path
    underlying: str
    tag: str
    date: str
    size: int
    modified: float


class CacheManager:
    """Manager for data cache."""

    def __init__(
        self,
        cache_dir: Path,
        data_dir: Path,
        project_root: Path,
    ):
        self.base_cache_dir = cache_dir
        self.cache_dir = cache_dir / "normalized"  # Normalized cache
        self.raw_cache_dir = cache_dir / "raw"
        self.data_dir = data_dir
        self.project_root = project_root

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    def _parse_filename(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """Parse filename to extract underlying, tag, and date.

        Expected format: UNDERLYING_TAG_DATE.parquet
        e.g., SPY_v2_2025-01-01.parquet -> (SPY, v2, 2025-01-01)
        """
        if not filename.endswith(".parquet"):
            return None

        stem = filename.replace(".parquet", "")
        parts = stem.split("_")

        if len(parts) < 3:
            return None

        # Date is always at the end
        date = parts[-1]
        # Underlying is first
        underlying = parts[0]
        # Tag is everything in between
        tag = "_".join(parts[1:-1])

        return underlying, tag, date

    def _extract_date(self, filename: str) -> Optional[str]:
        """Extract date from filename."""
        parsed = self._parse_filename(filename)
        if parsed:
            return parsed[2]
        return None

    def get_normalized_stats(self) -> CacheStats:
        """Get statistics for normalized cache."""
        files = list(self.cache_dir.glob("*.parquet"))

        dates = []
        total_size = 0

        for f in files:
            total_size += f.stat().st_size
            date = self._extract_date(f.name)
            if date:
                dates.append(date)

        date_range = (min(dates), max(dates)) if dates else ("", "")

        return CacheStats(
            cache_type="normalized",
            file_count=len(files),
            total_size=total_size,
            date_range=date_range,
            files=files,
        )

    def get_raw_stats(self) -> CacheStats:
        """Get statistics for raw cache."""
        files = list(self.raw_cache_dir.glob("*.parquet"))

        dates = []
        total_size = 0

        for f in files:
            total_size += f.stat().st_size
            date = self._extract_date(f.name)
            if date:
                dates.append(date)

        date_range = (min(dates), max(dates)) if dates else ("", "")

        return CacheStats(
            cache_type="raw",
            file_count=len(files),
            total_size=total_size,
            date_range=date_range,
            files=files,
        )

    def get_all_stats(self) -> List[CacheStats]:
        """Get statistics for all cache types."""
        return [
            self.get_normalized_stats(),
            self.get_raw_stats(),
        ]

    def list_files(
        self,
        cache_type: str = "normalized",
        underlying: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[CacheFile]:
        """List cache files with optional filtering."""
        if cache_type == "normalized":
            directory = self.cache_dir
        elif cache_type == "raw":
            directory = self.raw_cache_dir
        else:
            return []

        files = []
        for f in directory.glob("*.parquet"):
            parsed = self._parse_filename(f.name)
            if not parsed:
                continue

            file_underlying, tag, date = parsed

            # Apply filters
            if underlying and file_underlying != underlying:
                continue
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            files.append(CacheFile(
                path=f,
                underlying=file_underlying,
                tag=tag,
                date=date,
                size=f.stat().st_size,
                modified=f.stat().st_mtime,
            ))

        return sorted(files, key=lambda x: x.date)

    def clear(
        self,
        cache_type: str = "all",
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Clear cache files.

        Args:
            cache_type: "all", "normalized", or "raw"
            date: Clear files for a specific date
            start_date: Clear files from this date (inclusive)
            end_date: Clear files up to this date (inclusive)

        Returns:
            Number of files deleted
        """
        cleared = 0
        directories = []

        if cache_type in ["all", "normalized"]:
            directories.append(self.cache_dir)
        if cache_type in ["all", "raw"]:
            directories.append(self.raw_cache_dir)

        for directory in directories:
            for f in directory.glob("*.parquet"):
                file_date = self._extract_date(f.name)

                # Apply date filters
                if date and file_date != date:
                    continue
                if start_date and file_date and file_date < start_date:
                    continue
                if end_date and file_date and file_date > end_date:
                    continue

                f.unlink()
                cleared += 1

        # Also clear combined dataset .pt files when clearing all
        if cache_type == "all" and not date and not start_date and not end_date:
            for f in self.base_cache_dir.glob("*.pt"):
                f.unlink()
                cleared += 1

        return cleared

    def clear_all(self) -> int:
        """Clear all cache files."""
        return self.clear(cache_type="all")

    def clear_date(self, date: str) -> int:
        """Clear cache files for a specific date."""
        return self.clear(cache_type="all", date=date)

    def clear_date_range(self, start_date: str, end_date: str) -> int:
        """Clear cache files within a date range."""
        return self.clear(cache_type="all", start_date=start_date, end_date=end_date)

    def build_command(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        workers: int = 8,
    ) -> List[str]:
        """Build the cache build command."""
        cmd = [
            "uv", "run", "python", "scripts/build_cache.py",
            "--workers", str(workers),
        ]

        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

        return cmd

    def build(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        workers: int = 8,
        background: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Build/rebuild the data cache.

        Returns:
            Popen object if background=True, None otherwise
        """
        cmd = self.build_command(start_date, end_date, workers)

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

    def get_missing_dates(
        self,
        start_date: str,
        end_date: str,
        underlying: str = "SPY",
    ) -> List[str]:
        """Get list of dates that are missing from cache.

        This compares against available raw data.
        """
        # Get cached dates
        cached_dates = set()
        for f in self.list_files("normalized", underlying=underlying):
            cached_dates.add(f.date)

        # Get all available dates from raw data
        stocks_dir = self.data_dir / "stocks"
        available_dates = set()

        if stocks_dir.exists():
            for f in stocks_dir.glob(f"{underlying}_*.parquet"):
                date = self._extract_date(f.name)
                if date and start_date <= date <= end_date:
                    available_dates.add(date)

        # Return missing dates
        missing = available_dates - cached_dates
        return sorted(missing)

    def get_storage_summary(self) -> dict:
        """Get overall storage summary."""
        normalized = self.get_normalized_stats()
        raw = self.get_raw_stats()

        return {
            "normalized": {
                "files": normalized.file_count,
                "size_bytes": normalized.total_size,
                "date_range": normalized.date_range,
            },
            "raw": {
                "files": raw.file_count,
                "size_bytes": raw.total_size,
                "date_range": raw.date_range,
            },
            "total_size_bytes": normalized.total_size + raw.total_size,
        }
