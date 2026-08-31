#!/usr/bin/env python
"""Write artifacts/environment/hardware.json without touching system Python."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "environment" / "hardware.json"


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()


def main() -> int:
    gpu_csv = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    parts = [p.strip() for p in gpu_csv.split(",")]
    smi = sh("nvidia-smi")
    cuda = None
    for line in smi.splitlines():
        if "CUDA Version:" in line:
            cuda = line.split("CUDA Version:")[-1].strip().split()[0]
            break
    cpu = None
    for line in sh("lscpu").splitlines():
        if "Model name" in line or "Nombre del modelo" in line:
            cpu = line.split(":", 1)[1].strip()
            break
    disk = sh(f"df -B1 {ROOT}").splitlines()[-1].split()
    mem_b = int(sh("free -b | awk '/Mem:/{print $2}'"))
    py = {}
    for exe in ["python3.10", "python3.11", "python3.12", "python3"]:
        try:
            path = sh(f"command -v {exe}")
            ver = sh(f"{exe} --version")
            py[exe] = {"path": path, "version": ver}
        except Exception:
            pass
    hw = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": {
            "uname": platform.uname()._asdict(),
            "pretty_name": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'"),
        },
        "cpu": {"model": cpu, "logical_cpus": int(sh("nproc"))},
        "ram": {"total_bytes": mem_b, "total_gib": round(mem_b / (1024**3), 2)},
        "gpu": {
            "model": parts[0],
            "vram": parts[1],
            "driver": parts[2],
            "cuda_runtime_reported_by_nvidia_smi": cuda,
        },
        "python_available": py,
        "disk": {
            "filesystem": disk[0],
            "size_bytes": int(disk[1]),
            "used_bytes": int(disk[2]),
            "available_bytes": int(disk[3]),
            "available_gib": round(int(disk[3]) / (1024**3), 2),
            "use_percent": disk[4],
            "mount": disk[5],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hw, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
