"""CUDA pinned-memory copy optimisation benchmark.

Experiments:
  1. pageable vs pinned H2D          — single-transfer latency comparison
  2. pageable vs pinned D2H          — single-transfer latency comparison
  3. pinned H2D + compute overlap    — copy on stream A, matmul on stream B
  4. pinned D2H + compute overlap    — compute on stream A, copy out on stream B
  5. multi-batch pinned throughput   — sustained transfer bandwidth
  6. pin-memory registration overhead — cost of cudaHostRegister
"""

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path.home() / "benchmarks"
OUTPUT_DIR = ROOT / "outputs"


def setup_env() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mem_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


@dataclass
class CopyResult:
    experiment: str
    direction: str
    memory_type: str
    tensor_size_mb: float
    num_elements: int
    latency_mean_ms: float
    latency_p95_ms: float
    bandwidth_gb_s: float
    overlap_speedup: float
    gpu_mem_mb: float
    status: str = "ok"
    error_message: str = ""


# ── Experiment 1 & 2: Pageable vs Pinned H2D / D2H ──────────────────────────


def _measure_copy(cpu_tensor: torch.Tensor, direction: str, warmup: int, repeat: int,
                  pinned: bool = False) -> list[float]:
    latencies: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if direction == "H2D":
        for _ in range(warmup):
            gpu = cpu_tensor.to("cuda", non_blocking=pinned)
            torch.cuda.synchronize()
        for _ in range(repeat):
            start.record()
            gpu = cpu_tensor.to("cuda", non_blocking=pinned)
            end.record()
            torch.cuda.synchronize()
            latencies.append(event_ms(start, end))
    else:  # D2H
        gpu = cpu_tensor.to("cuda")
        cpu_out = torch.empty_like(cpu_tensor)
        for _ in range(warmup):
            cpu_out.copy_(gpu, non_blocking=pinned)
            torch.cuda.synchronize()
        for _ in range(repeat):
            start.record()
            cpu_out.copy_(gpu, non_blocking=pinned)
            end.record()
            torch.cuda.synchronize()
            latencies.append(event_ms(start, end))

    return latencies


def run_pageable_vs_pinned(size_mb: int, warmup: int, repeat: int) -> list[dict]:
    rows: list[dict] = []
    numel = (size_mb * 1024 * 1024) // 4  # float32
    cpu_pageable = torch.randn(numel, dtype=torch.float32)
    cpu_pinned = cpu_pageable.clone().pin_memory()

    tensor_bytes = float(cpu_pageable.numel() * 4)

    for direction in ("H2D", "D2H"):
        for label, tensor, is_pinned in [
            ("pageable", cpu_pageable, False),
            ("pinned", cpu_pinned, True),
        ]:
            lat = _measure_copy(tensor, direction, warmup, repeat, pinned=is_pinned)
            mean_ms = float(np.mean(lat))
            bw = (tensor_bytes / (mean_ms / 1000.0)) / 1e9 if mean_ms > 0 else 0.0
            rows.append({
                "experiment": f"{direction}_{label}",
                "direction": direction,
                "memory_type": label,
                "tensor_size_mb": round(tensor_bytes / (1024 * 1024), 2),
                "num_elements": cpu_pageable.numel(),
                "latency_mean_ms": round(mean_ms, 6),
                "latency_p95_ms": round(pct(lat, 95), 6),
                "bandwidth_gb_s": round(bw, 4),
                "overlap_speedup": 0.0,
                "gpu_mem_mb": 0.0,
            })
    return rows


# ── Experiment 3 & 4: Pinned copy + compute overlap ──────────────────────────


def _pinned_overlap_benchmark(size: int, iters: int, direction: str,
                              warmup: int, repeat: int) -> tuple[list[float], list[float]]:
    """
    direction='H2D': stream A does H2D, stream B does matmul
    direction='D2H': stream A does matmul, stream B does D2H
    Returns (sequential_latencies, overlap_latencies).
    """
    a_cpu = torch.randn(size, size, dtype=torch.float32).pin_memory()
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    c_cpu = torch.empty(size, size, dtype=torch.float32).pin_memory()

    s_copy = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()
    wall_s = torch.cuda.Event(enable_timing=True)
    wall_e = torch.cuda.Event(enable_timing=True)

    # --- sequential ---
    seq_lat: list[float] = []
    for _ in range(warmup):
        a = a_cpu.to("cuda", non_blocking=True)
        for _ in range(iters):
            a = torch.matmul(a, b)
        c_cpu.copy_(a, non_blocking=True)
        torch.cuda.synchronize()
    for _ in range(repeat):
        torch.cuda.synchronize()
        wall_s.record()
        a = a_cpu.to("cuda", non_blocking=True)
        for _ in range(iters):
            a = torch.matmul(a, b)
        c_cpu.copy_(a, non_blocking=True)
        wall_e.record()
        torch.cuda.synchronize()
        seq_lat.append(event_ms(wall_s, wall_e))

    # --- overlapped ---
    ov_lat: list[float] = []
    for _ in range(warmup):
        with torch.cuda.stream(s_copy):
            a = a_cpu.to("cuda", non_blocking=True)
        with torch.cuda.stream(s_comp):
            _a = torch.randn(size, size, device="cuda")
            for _ in range(iters):
                _a = torch.matmul(_a, b)
            if direction == "D2H":
                c_cpu.copy_(_a, non_blocking=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(s_copy):
            if direction == "D2H":
                c_cpu.copy_(a, non_blocking=True)
        with torch.cuda.stream(s_comp):
            for _ in range(iters):
                a = torch.matmul(a, b)
        torch.cuda.synchronize()

    for _ in range(repeat):
        torch.cuda.synchronize()
        wall_s.record()
        with torch.cuda.stream(s_copy):
            x = a_cpu.to("cuda", non_blocking=True)
        with torch.cuda.stream(s_comp):
            y = torch.randn(size, size, device="cuda")
            for _ in range(iters):
                y = torch.matmul(y, b)
            if direction == "D2H":
                c_cpu.copy_(y, non_blocking=True)
        with torch.cuda.stream(s_copy):
            if direction == "D2H":
                c_cpu.copy_(x, non_blocking=True)
        for _ in range(iters):
            x = torch.matmul(x, b)
        wall_e.record()
        torch.cuda.synchronize()
        ov_lat.append(event_ms(wall_s, wall_e))

    return seq_lat, ov_lat


def run_copy_compute_overlap(size: int, iters: int, warmup: int, repeat: int) -> list[dict]:
    rows: list[dict] = []
    for direction in ("H2D", "D2H"):
        seq, ov = _pinned_overlap_benchmark(size, iters, direction, warmup, repeat)
        seq_ms = float(np.mean(seq)) if seq else 0.0
        ov_ms = float(np.mean(ov)) if ov else 0.0
        speedup = seq_ms / ov_ms if ov_ms > 0 else 1.0
        rows.append({
            "experiment": f"overlap_{direction}",
            "direction": direction,
            "memory_type": "pinned",
            "tensor_size_mb": round((size * size * 4) / (1024 * 1024), 2),
            "num_elements": size * size,
            "latency_mean_ms": round(ov_ms, 6),
            "latency_p95_ms": round(pct(ov, 95) if ov else 0.0, 6),
            "bandwidth_gb_s": 0.0,
            "overlap_speedup": round(speedup, 4),
            "gpu_mem_mb": 0.0,
        })
        print(f"  overlap_{direction}: sequential={seq_ms:.4f}ms  overlapped={ov_ms:.4f}ms  speedup={speedup:.2f}x")
    return rows


# ── Experiment 5: Multi-batch pinned throughput ─────────────────────────────


def _pinned_throughput(size: int, batches: int, warmup: int, repeat: int) -> tuple[float, float]:
    """Sustained H2D bandwidth with pinned memory batching."""
    tensors = [torch.randn(size, size, dtype=torch.float32).pin_memory() for _ in range(batches)]
    wall_s = torch.cuda.Event(enable_timing=True)
    wall_e = torch.cuda.Event(enable_timing=True)
    gpu_tensors: list[torch.Tensor] = []

    for _ in range(warmup):
        gpu_tensors = [t.to("cuda", non_blocking=True) for t in tensors]
        torch.cuda.synchronize()
        gpu_tensors.clear()

    latencies: list[float] = []
    for _ in range(repeat):
        wall_s.record()
        gpu_tensors = [t.to("cuda", non_blocking=True) for t in tensors]
        wall_e.record()
        torch.cuda.synchronize()
        latencies.append(event_ms(wall_s, wall_e))
        gpu_tensors.clear()

    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    total_bytes = sum(t.numel() * 4 for t in tensors)
    bw = (total_bytes / (mean_ms / 1000.0)) / 1e9 if mean_ms > 0 else 0.0
    return mean_ms, bw


# ── Experiment 6: Pin-memory registration overhead ──────────────────────────


def _pin_overhead(size_mb: int, repeat: int) -> tuple[float, float]:
    """Measure cudaHostRegister / pin_memory() overhead."""
    numel = (size_mb * 1024 * 1024) // 4
    torch.cuda.synchronize()

    # pin timing
    pin_lat: list[float] = []
    for _ in range(repeat):
        t = torch.randn(numel, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_pinned = t.pin_memory()
        torch.cuda.synchronize()
        pin_lat.append((time.perf_counter() - t0) * 1000.0)
        del t, t_pinned

    # unpin timing (pageable clone as baseline for allocation)
    page_lat: list[float] = []
    for _ in range(repeat):
        t = torch.randn(numel, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t2 = t.clone()
        torch.cuda.synchronize()
        page_lat.append((time.perf_counter() - t0) * 1000.0)
        del t, t2

    return float(np.mean(pin_lat)), float(np.mean(page_lat))


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA pinned-memory copy benchmark")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "pinned_memory_results.csv"))
    parser.add_argument("--size-mb", type=int, nargs="+", default=[1, 16, 64, 256],
                        help="Tensor sizes in MB for H2D/D2H comparison")
    parser.add_argument("--overlap-size", type=int, default=2048, help="Matrix size for overlap benchmark")
    parser.add_argument("--overlap-iters", type=int, default=5, help="Matmul iterations in overlap benchmark")
    parser.add_argument("--throughput-batches", type=int, default=16, help="Batch count for throughput test")
    parser.add_argument("--throughput-size", type=int, default=256, help="Matrix size for throughput test")
    parser.add_argument("--pin-overhead-mb", type=int, default=256, help="Tensor size for pin overhead test")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    setup_env()
    print(f"CUDA pinned-memory copy benchmark")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print()

    all_rows: list[dict] = []

    # Exps 1 & 2: pageable vs pinned H2D/D2H
    print("── pageable vs pinned H2D/D2H ──")
    for size_mb in args.size_mb:
        rows = run_pageable_vs_pinned(size_mb, args.warmup, args.repeat)
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['experiment']:18s}  size={r['tensor_size_mb']:6.1f}MB  "
                  f"lat={r['latency_mean_ms']:8.4f}ms  bw={r['bandwidth_gb_s']:7.2f}GB/s")

    # Exps 3 & 4: copy + compute overlap
    print("\n── pinned copy + compute overlap ──")
    rows = run_copy_compute_overlap(args.overlap_size, args.overlap_iters, args.warmup, args.repeat)
    all_rows.extend(rows)

    # Exp 5: multi-batch throughput
    print("\n── multi-batch pinned throughput ──")
    mean_ms, bw = _pinned_throughput(args.throughput_size, args.throughput_batches, args.warmup, args.repeat)
    total_mb = args.throughput_batches * args.throughput_size * args.throughput_size * 4 / (1024 * 1024)
    all_rows.append({
        "experiment": "throughput_batched",
        "direction": "H2D",
        "memory_type": "pinned",
        "tensor_size_mb": round(total_mb, 2),
        "num_elements": args.throughput_batches * args.throughput_size * args.throughput_size,
        "latency_mean_ms": round(mean_ms, 6),
        "latency_p95_ms": 0.0,
        "bandwidth_gb_s": round(bw, 4),
        "overlap_speedup": 0.0,
        "gpu_mem_mb": 0.0,
    })
    print(f"  {args.throughput_batches} × {args.throughput_size}² tensors:  "
          f"{mean_ms:.4f}ms  {bw:.2f} GB/s  ({total_mb:.0f} MB total)")

    # Exp 6: pin overhead
    print("\n── pin-memory registration overhead ──")
    pin_ms, page_ms = _pin_overhead(args.pin_overhead_mb, args.repeat)
    all_rows.append({
        "experiment": "pin_overhead",
        "direction": "N/A",
        "memory_type": "pinned",
        "tensor_size_mb": float(args.pin_overhead_mb),
        "num_elements": (args.pin_overhead_mb * 1024 * 1024) // 4,
        "latency_mean_ms": round(pin_ms, 6),
        "latency_p95_ms": 0.0,
        "bandwidth_gb_s": 0.0,
        "overlap_speedup": 0.0,
        "gpu_mem_mb": 0.0,
    })
    all_rows.append({
        "experiment": "pin_overhead_baseline",
        "direction": "N/A",
        "memory_type": "pageable",
        "tensor_size_mb": float(args.pin_overhead_mb),
        "num_elements": (args.pin_overhead_mb * 1024 * 1024) // 4,
        "latency_mean_ms": round(page_ms, 6),
        "latency_p95_ms": 0.0,
        "bandwidth_gb_s": 0.0,
        "overlap_speedup": round(pin_ms / page_ms, 2) if page_ms > 0 else 0.0,
        "gpu_mem_mb": 0.0,
    })
    print(f"  pin_memory({args.pin_overhead_mb}MB): {pin_ms:.4f}ms")
    print(f"  clone (pageable baseline):         {page_ms:.4f}ms")
    print(f"  pin/clone ratio:                   {pin_ms / page_ms:.2f}x" if page_ms > 0 else "")

    # Write CSV
    fields = [
        "experiment", "direction", "memory_type", "tensor_size_mb", "num_elements",
        "latency_mean_ms", "latency_p95_ms", "bandwidth_gb_s", "overlap_speedup",
        "gpu_mem_mb", "status", "error_message",
    ]
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
