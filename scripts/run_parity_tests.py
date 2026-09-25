#!/usr/bin/env python3
# pyright: basic, reportPrivateImportUsage=false
"""
Robust test runner for parity tests with timeout handling and granular logging.

Usage:
    python scripts/run_parity_tests.py [--test-class CLASS] [--timeout SECONDS] [--log-dir DIR]
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Runs pytest tests with timeout and detailed logging."""

    def __init__(self, timeout: int = 300, log_dir: str = "test_logs"):
        self.timeout = timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def discover_tests(self, test_class: Optional[str] = None) -> List[str]:
        """Discover test methods to run."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/unit/test_feature_statistics_parity.py",
        ]
        if test_class:
            # Filter by class name in the test path
            cmd.extend(["-k", f"TestFeatureStatisticsParity{test_class}"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Test discovery failed: {result.stderr}")
            return []

        tests = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("tests/"):
                tests.append(line)
        return tests

    def run_single_test(self, test_path: str) -> Dict[str, Any]:
        """Run a single test with timeout and capture output."""
        test_name = test_path.split("::")[-1]
        log_file = (
            self.log_dir / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logger.info(f"Running test: {test_name}")
        logger.info(f"Log file: {log_file}")

        cmd = [sys.executable, "-m", "pytest", test_path, "-x", "-v", "--tb=short"]

        start_time = time.time()
        result = {
            "test": test_name,
            "test_path": test_path,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "status": "running",
            "duration": 0,
            "log_file": str(log_file),
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }

        try:
            with open(log_file, "w") as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Timeout: {self.timeout}s\n")
                f.write("=" * 80 + "\n\n")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                # Poll with timeout
                while True:
                    try:
                        stdout, _ = proc.communicate(timeout=5)
                        if stdout:
                            f.write(stdout)
                            f.flush()
                            result["stdout"] += stdout
                        break
                    except subprocess.TimeoutExpired:
                        # Check overall timeout
                        if time.time() - start_time > self.timeout:
                            proc.kill()
                            stdout, _ = proc.communicate()
                            if stdout:
                                f.write(stdout)
                                result["stdout"] += stdout
                            result["status"] = "timeout"
                            result["duration"] = time.time() - start_time
                            logger.error(
                                f"Test {test_name} timed out after {self.timeout}s"
                            )
                            return result
                        # Continue polling
                        continue

                result["returncode"] = proc.returncode
                result["duration"] = time.time() - start_time
                result["status"] = "passed" if proc.returncode == 0 else "failed"

                if proc.returncode == 0:
                    logger.info(f"Test {test_name} PASSED in {result['duration']:.1f}s")
                else:
                    logger.error(
                        f"Test {test_name} FAILED in {result['duration']:.1f}s"
                    )

        except Exception as e:
            result["status"] = "error"
            result["duration"] = time.time() - start_time
            result["stderr"] = str(e)
            logger.error(f"Test {test_name} ERROR: {e}")

        # Write final status to log
        with open(log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            f.write(f"Return code: {result['returncode']}\n")

        return result

    def run_tests(self, test_class: Optional[str] = None) -> Dict[str, Any]:
        """Run all discovered tests."""
        logger.info("Discovering tests...")
        tests = self.discover_tests(test_class)

        if not tests:
            logger.warning("No tests discovered")
            return {
                "status": "no_tests",
                "results": [],
                "total": 0,
                "passed": 0,
                "failed": 0,
                "timeout": 0,
                "error": 0,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": 0,
            }

        logger.info(f"Found {len(tests)} tests to run")

        summary = {
            "total": len(tests),
            "passed": 0,
            "failed": 0,
            "timeout": 0,
            "error": 0,
            "start_time": datetime.now().isoformat(),
            "results": [],
        }

        for i, test in enumerate(tests, 1):
            logger.info(f"[{i}/{len(tests)}] Running {test}")
            result = self.run_single_test(test)
            self.results.append(result)
            summary["results"].append(result)

            if result["status"] == "passed":
                summary["passed"] += 1
            elif result["status"] == "failed":
                summary["failed"] += 1
            elif result["status"] == "timeout":
                summary["timeout"] += 1
            else:
                summary["error"] += 1

            # Save progress after each test
            self.save_summary(summary)

        summary["end_time"] = datetime.now().isoformat()
        summary["total_duration"] = sum(r["duration"] for r in summary["results"])

        logger.info(
            f"Test run complete: {summary['passed']} passed, {summary['failed']} failed, {summary['timeout']} timeout, {summary['error']} errors"
        )
        return summary

    def save_summary(self, summary: Dict[str, Any]):
        """Save summary to JSON file."""
        summary_file = (
            self.log_dir
            / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run parity tests with timeout and logging"
    )
    parser.add_argument(
        "--test-class", choices=["DetVsCur", "CurVsCol"], help="Run specific test class"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)",
    )
    parser.add_argument(
        "--log-dir",
        default="test_logs",
        help="Directory for test logs (default: test_logs)",
    )
    parser.add_argument("--test", help="Run specific test by name")

    args = parser.parse_args()

    runner = TestRunner(timeout=args.timeout, log_dir=args.log_dir)

    if args.test:
        # Run single test
        result = runner.run_single_test(args.test)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "passed" else 1)
    else:
        # Run all tests in class
        test_class = args.test_class if args.test_class else None
        summary = runner.run_tests(test_class)

        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary["failed"] == 0 and summary["timeout"] == 0 else 1)


if __name__ == "__main__":
    main()
