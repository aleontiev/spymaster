#!/usr/bin/env python3
"""
Test runner script for LeJEPA project.
Run all tests or specific test suites.
"""
import sys
import subprocess
from pathlib import Path


def run_tests(test_path: str = "", verbose: bool = True, markers: str = "") -> int:
    """
    Run pytest with specified options.

    Args:
        test_path: Specific test file or directory to run
        verbose: Enable verbose output
        markers: Pytest marker expression (e.g., "not slow")

    Returns:
        Exit code
    """
    cmd = ["pytest"]

    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    if verbose:
        cmd.append("-v")

    if markers:
        cmd.extend(["-m", markers])

    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    except ImportError:
        pass

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LeJEPA tests")
    parser.add_argument(
        "test",
        nargs="?",
        default="",
        help="Specific test file or directory (default: all tests)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    parser.add_argument(
        "-m", "--markers",
        default="",
        help="Run tests matching marker expression (e.g., 'not slow')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (exclude slow and cuda tests)",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Run only data processing tests",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Run only model tests",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run only end-to-end integration tests",
    )

    args = parser.parse_args()

    # Determine test path
    test_path = args.test

    if args.data:
        test_path = "tests/test_data_processing.py"
    elif args.model:
        test_path = "tests/test_model.py"
    elif args.e2e:
        test_path = "tests/test_end_to_end.py"

    # Determine markers
    markers = args.markers

    if args.quick:
        markers = "not slow and not cuda"

    # Run tests
    verbose = not args.quiet
    return run_tests(test_path, verbose=verbose, markers=markers)


if __name__ == "__main__":
    sys.exit(main())
