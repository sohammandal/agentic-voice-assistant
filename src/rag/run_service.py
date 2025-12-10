"""
Embedding Service Runner - Start, stop, and manage the embedding service.

This script provides programmatic control over the embedding service,
allowing other scripts to start/stop it as needed.

Usage:
    # From command line
    python -m src.rag.run_service start
    python -m src.rag.run_service stop
    python -m src.rag.run_service status

    # From Python
    from src.rag.run_service import EmbeddingServiceManager

    manager = EmbeddingServiceManager()
    manager.start()
    # ... use the service ...
    manager.stop()
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100
PID_FILE = Path(__file__).parent / ".embedding_service.pid"
PORT_FILE = Path(__file__).parent / ".embedding_service.port"


class EmbeddingServiceManager:
    """Manager for starting, stopping, and checking the embedding service."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the service manager.

        Args:
            host: Host to bind the service to
            port: Port to bind the service to
        """
        self.host = host
        self.port = port
        self._process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        """Get the base URL of the service."""
        # Try to read actual port from file
        actual_port = self._read_port() or self.port
        return f"http://{self.host}:{actual_port}"

    def _write_pid(self, pid: int):
        """Write the process ID to file."""
        PID_FILE.write_text(str(pid))

    def _read_pid(self) -> Optional[int]:
        """Read the process ID from file."""
        if PID_FILE.exists():
            try:
                return int(PID_FILE.read_text().strip())
            except (ValueError, FileNotFoundError):
                return None
        return None

    def _clear_pid(self):
        """Clear the PID file."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _write_port(self, port: int):
        """Write the port to file."""
        PORT_FILE.write_text(str(port))

    def _read_port(self) -> Optional[int]:
        """Read the port from file."""
        if PORT_FILE.exists():
            try:
                return int(PORT_FILE.read_text().strip())
            except (ValueError, FileNotFoundError):
                return None
        return None

    def _clear_port(self):
        """Clear the port file."""
        if PORT_FILE.exists():
            PORT_FILE.unlink()

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        if sys.platform == "win32":
            try:
                # On Windows, use tasklist
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
                )
                return str(pid) in result.stdout
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def is_running(self) -> bool:
        """Check if the service is running and responding."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def is_ready(self) -> bool:
        """Check if the service is running and model is loaded."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("model_loaded", False)
            return False
        except requests.exceptions.RequestException:
            return False

    def start(self, wait_for_ready: bool = True, timeout: int = 600) -> bool:
        """
        Start the embedding service.

        Args:
            wait_for_ready: If True, wait until model is loaded
            timeout: Maximum seconds to wait for service to be ready (default 600s for first-time model download)

        Returns:
            True if service started successfully
        """
        # Check if already running
        if self.is_running():
            logger.info("Embedding service is already running")
            return True

        # Clean up stale PID file
        old_pid = self._read_pid()
        if old_pid and not self._is_process_running(old_pid):
            self._clear_pid()
            self._clear_port()

        logger.info(f"Starting embedding service on {self.host}:{self.port}...")

        # Get the path to the embedding_service module
        service_module = "src.rag.embedding_service"

        # Start the service as a subprocess
        # Use the same Python interpreter
        python_exe = sys.executable

        # Build command
        cmd = [
            python_exe,
            "-m",
            service_module,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Get workspace root (parent of src)
        workspace_root = Path(__file__).parent.parent.parent

        # Start process
        if sys.platform == "win32":
            # On Windows, use CREATE_NEW_PROCESS_GROUP for proper signal handling
            self._process = subprocess.Popen(
                cmd,
                cwd=workspace_root,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self._process = subprocess.Popen(
                cmd,
                cwd=workspace_root,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        # Save PID and port
        self._write_pid(self._process.pid)
        self._write_port(self.port)

        logger.info(f"Service started with PID {self._process.pid}")

        if wait_for_ready:
            return self.wait_for_ready(timeout=timeout)

        return True

    def wait_for_ready(self, timeout: int = 600) -> bool:
        """
        Wait for the service to be ready (model loaded).

        Args:
            timeout: Maximum seconds to wait (default 600s for first-time model download)

        Returns:
            True if service is ready, False if timeout
        """
        start_time = time.time()
        check_interval = 2

        logger.info("Waiting for embedding service to be ready (model loading)...")

        while time.time() - start_time < timeout:
            if self.is_ready():
                logger.info("Embedding service is ready!")
                return True

            # Check if process died
            pid = self._read_pid()
            if pid and not self._is_process_running(pid):
                logger.error("Service process died unexpectedly")
                self._clear_pid()
                self._clear_port()
                return False

            time.sleep(check_interval)

        logger.error(f"Timeout waiting for service to be ready ({timeout}s)")
        return False

    def stop(self) -> bool:
        """
        Stop the embedding service.

        Returns:
            True if service stopped successfully
        """
        pid = self._read_pid()

        if not pid:
            logger.info("No PID file found, service may not be running")
            return True

        if not self._is_process_running(pid):
            logger.info("Service process not running, cleaning up")
            self._clear_pid()
            self._clear_port()
            return True

        logger.info(f"Stopping embedding service (PID {pid})...")

        try:
            if sys.platform == "win32":
                # On Windows, use taskkill
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)], capture_output=True
                )
            else:
                os.kill(pid, signal.SIGTERM)

                # Wait for graceful shutdown
                for _ in range(10):
                    if not self._is_process_running(pid):
                        break
                    time.sleep(0.5)
                else:
                    # Force kill if still running
                    os.kill(pid, signal.SIGKILL)

            self._clear_pid()
            self._clear_port()
            logger.info("Embedding service stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping service: {e}")
            return False

    def restart(self, wait_for_ready: bool = True, timeout: int = 600) -> bool:
        """
        Restart the embedding service.

        Args:
            wait_for_ready: If True, wait until model is loaded
            timeout: Maximum seconds to wait for service to be ready

        Returns:
            True if service restarted successfully
        """
        self.stop()
        time.sleep(1)  # Brief pause
        return self.start(wait_for_ready=wait_for_ready, timeout=timeout)

    def status(self) -> dict:
        """
        Get the status of the embedding service.

        Returns:
            Dictionary with status information
        """
        pid = self._read_pid()
        port = self._read_port() or self.port

        status = {
            "pid": pid,
            "port": port,
            "process_running": pid is not None and self._is_process_running(pid),
            "service_responding": False,
            "model_loaded": False,
            "model_name": None,
            "device": None,
        }

        try:
            response = requests.get(f"http://{self.host}:{port}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                status["service_responding"] = True
                status["model_loaded"] = health.get("model_loaded", False)
                status["model_name"] = health.get("model_name")
                status["device"] = health.get("device")
        except requests.exceptions.RequestException:
            pass

        return status


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage the embedding service")
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Action to perform",
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for model to load"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for waiting (default 600s for first-time model download)",
    )

    args = parser.parse_args()

    manager = EmbeddingServiceManager(host=args.host, port=args.port)

    if args.action == "start":
        success = manager.start(wait_for_ready=not args.no_wait, timeout=args.timeout)
        sys.exit(0 if success else 1)

    elif args.action == "stop":
        success = manager.stop()
        sys.exit(0 if success else 1)

    elif args.action == "restart":
        success = manager.restart(wait_for_ready=not args.no_wait, timeout=args.timeout)
        sys.exit(0 if success else 1)

    elif args.action == "status":
        status = manager.status()
        print("\nEmbedding Service Status:")
        print(f"  PID: {status['pid']}")
        print(f"  Port: {status['port']}")
        print(f"  Process Running: {status['process_running']}")
        print(f"  Service Responding: {status['service_responding']}")
        print(f"  Model Loaded: {status['model_loaded']}")
        print(f"  Model Name: {status['model_name']}")
        print(f"  Device: {status['device']}")
        sys.exit(0)


if __name__ == "__main__":
    main()
