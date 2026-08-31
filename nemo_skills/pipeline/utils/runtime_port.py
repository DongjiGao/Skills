# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coordinate collision-free runtime ports between co-scheduled jobs."""

import argparse
import fcntl
import os
import socket
import sys
import time
from pathlib import Path

PORT_MIN = 20000
PORT_COUNT = 45000
PORT_STEP = 6151
DEFAULT_MAX_ATTEMPTS = 128
DEFAULT_WAIT_TIMEOUT = 600.0


def _read_port(state_file: Path) -> int | None:
    try:
        value = int(state_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if not 1 <= value <= 65535:
        return None
    return value


def _candidate_ports(base_port: int, max_attempts: int):
    if not PORT_MIN <= base_port < PORT_MIN + PORT_COUNT:
        raise ValueError(f"base_port must be in [{PORT_MIN}, {PORT_MIN + PORT_COUNT}), got {base_port}")
    for attempt in range(max_attempts):
        yield PORT_MIN + ((base_port - PORT_MIN + attempt * PORT_STEP) % PORT_COUNT)


def _port_is_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def allocate_runtime_port(
    state_file: str | os.PathLike[str],
    base_port: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    """Select a free candidate once and publish it atomically for clients."""
    state_file = Path(state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = state_file.with_suffix(state_file.suffix + ".lock")
    with lock_file.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selected = _read_port(state_file)
        if selected is not None:
            return selected

        for port in _candidate_ports(base_port, max_attempts):
            if not _port_is_available(port):
                continue
            temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
            temporary.write_text(f"{port}\n")
            os.replace(temporary, state_file)
            return port

    raise RuntimeError(f"Unable to find a free runtime port after {max_attempts} candidates")


def wait_for_runtime_port(
    state_file: str | os.PathLike[str],
    timeout: float = DEFAULT_WAIT_TIMEOUT,
    poll_interval: float = 0.05,
) -> int:
    """Wait for the server-side allocator and return its published port."""
    state_file = Path(state_file)
    deadline = time.monotonic() + timeout
    while True:
        selected = _read_port(state_file)
        if selected is not None:
            return selected
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for runtime port in {state_file}")
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    allocate = subparsers.add_parser("allocate")
    allocate.add_argument("--state-file", required=True)
    allocate.add_argument("--base-port", required=True, type=int)
    allocate.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    wait = subparsers.add_parser("wait")
    wait.add_argument("--state-file", required=True)
    wait.add_argument("--timeout", type=float, default=DEFAULT_WAIT_TIMEOUT)
    args = parser.parse_args()

    if args.command == "allocate":
        port = allocate_runtime_port(args.state_file, args.base_port, args.max_attempts)
        if port != args.base_port:
            print(
                f"NeMo Skills runtime port {args.base_port} was occupied; using {port} instead",
                file=sys.stderr,
                flush=True,
            )
    else:
        port = wait_for_runtime_port(args.state_file, args.timeout)
    print(port)


