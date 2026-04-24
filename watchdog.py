#!/usr/bin/env python3
"""TetraMem Watchdog — monitors health endpoint, auto-restarts on failure."""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s %(message)s",
)
log = logging.getLogger("tetramem.watchdog")

HEALTH_URL = os.environ.get("TETRAMEM_HEALTH_URL", "http://127.0.0.1:8000/api/v1/health")
CHECK_INTERVAL = int(os.environ.get("TETRAMEM_WATCHDOG_INTERVAL", "15"))
FAIL_THRESHOLD = int(os.environ.get("TETRAMEM_WATCHDOG_FAIL_COUNT", "3"))
RESTART_COOLDOWN = int(os.environ.get("TETRAMEM_WATCHDOG_COOLDOWN", "30"))
SERVICE_NAME = os.environ.get("TETRAMEM_SERVICE", "tetramem-api")
MAX_RESTARTS_PER_HOUR = int(os.environ.get("TETRAMEM_WATCHDOG_MAX_RESTARTS", "10"))

_consecutive_failures = 0
_restart_timestamps = []
_running = True


def _signal_handler(signum, frame):
    global _running
    _running = False
    log.info("Watchdog stopping...")


def check_health():
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            status = data.get("status", "unknown")
            if status == "ok":
                return True, data
            return False, data
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
        return False, {"error": str(e)}
    except Exception as e:
        return False, {"error": str(e)}


def restart_service():
    now = time.time()
    global _restart_timestamps
    _restart_timestamps = [t for t in _restart_timestamps if now - t < 3600]
    if len(_restart_timestamps) >= MAX_RESTARTS_PER_HOUR:
        log.critical(
            "Max restarts per hour reached (%d) — NOT restarting",
            MAX_RESTARTS_PER_HOUR,
        )
        return False

    log.warning("Restarting %s...", SERVICE_NAME)
    try:
        subprocess.run(
            ["systemctl", "restart", SERVICE_NAME],
            capture_output=True, text=True, timeout=30,
        )
        _restart_timestamps.append(now)
        log.info("Restart command sent. Waiting %ds for recovery...", RESTART_COOLDOWN)
        time.sleep(RESTART_COOLDOWN)
        return True
    except subprocess.TimeoutExpired:
        log.error("Restart command timed out")
        return False
    except Exception as e:
        log.error("Restart failed: %s", e)
        return False


def run_watchdog():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    log.info("Watchdog started: url=%s interval=%ds fail_threshold=%d",
             HEALTH_URL, CHECK_INTERVAL, FAIL_THRESHOLD)

    global _consecutive_failures

    while _running:
        healthy, data = check_health()

        if healthy:
            if _consecutive_failures > 0:
                log.info("Service recovered after %d failures", _consecutive_failures)
            _consecutive_failures = 0
            uptime = data.get("uptime_seconds", 0)
            degradation = data.get("degradation_level", 0)
            if degradation > 0:
                log.warning("Degradation level: %d (issues: %s)",
                           degradation, json.dumps(data.get("checks", {})))
        else:
            _consecutive_failures += 1
            log.warning("Health check failed (%d/%d): %s",
                       _consecutive_failures, FAIL_THRESHOLD,
                       json.dumps(data, default=str)[:200])

            if _consecutive_failures >= FAIL_THRESHOLD:
                log.critical(
                    "Service unhealthy for %d consecutive checks — triggering restart",
                    _consecutive_failures,
                )
                restarted = restart_service()
                if restarted:
                    _consecutive_failures = 0
                else:
                    _consecutive_failures = 0
                    log.error("Restart failed or rate-limited. Will retry next cycle.")

        for _ in range(CHECK_INTERVAL):
            if not _running:
                break
            time.sleep(1)

    log.info("Watchdog stopped")


if __name__ == "__main__":
    run_watchdog()
