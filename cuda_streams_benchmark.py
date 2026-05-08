"""CUDA streams concurrency benchmark.

Experiments:
  1. single-stream baseline  – all work on default stream
  2. N-stream concurrent      – distribute identical kernels across N streams
  3. compute-copy overlap     – overlap H2D/D2H with compute on separate streams
  4. stream sync overhead     – measure cudaStreamSynchronize / event overhead

Outputs a CSV compatible with the existing operator_results.csv schema so that
operator_pair_summary / run_full_benchmark can consume it.
"""

import argparse
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ── UTF-8 stdout for Windows ────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path.home() / "benchmarks"
OUTPUT_DIR = ROOT / "outputs"
PROFILE_DIR = OUTPUT_DIR / "ort_profiles"


def setup_env() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# ── Experiment kernels ──────────────────────────────────────────────────────
# Each kernel is a function (size, iters) -> None that burns GPU compute.


def _gevm_kernel(size: int, iters: int) -> None:
    """GEMM-like square matmul — high compute density."""
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    for _ in range(iters):
        a = torch.matmul(a, b)


def _vector_kernel(size: int, iters: int) -> None:
    """Element-wise vector add — low compute, high bandwidth."""
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    for _ in range(iters):
        a = a + b


def _conv2d_kernel(size: int, iters: int) -> None:
    """Conv2d — typical CNN workload."""
    x = torch.randn(1, 64, size, size, device="cuda", dtype=torch.float32)
    conv = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False).cuda().eval()
    for _ in range(iters):
        x = conv(x)


KERNELS = {
    "gevm": _gevm_kernel,
    "vector": _vector_kernel,
    "conv2d": _conv2d_kernel,
}

# ── Helpers ─────────────────────────────────────────────────────────────────


def torch_event_time(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Return elapsed milliseconds between two CUDA events."""
    return start.elapsed_time(end)


def reset_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


def peak_mem_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "experiment",
        "backend",
        "num_streams",
        "kernel",
        "matrix_size",
        "iters_per_stream",
        "total_work_iters",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_min_ms",
        "latency_max_ms",
        "speedup_vs_baseline",
        "gpu_mem_mb",
        "status",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class StreamResult:
    experiment: str
    num_streams: int
    kernel: str
    matrix_size: int
    iters_per_stream: int
    total_work_iters: int
    latencies: list[float] = field(default_factory=list)
    gpu_mem: float = 0.0
    status: str = "ok"
    error_message: str = ""

    def to_row(self, baseline_ms: float = 0.0) -> dict[str, object]:
        if not self.latencies:
            return {
                "experiment": self.experiment,
                "backend": "torch_cuda",
                "num_streams": self.num_streams,
                "kernel": self.kernel,
                "matrix_size": self.matrix_size,
                "iters_per_stream": self.iters_per_stream,
                "total_work_iters": self.total_work_iters,
                "latency_mean_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_min_ms": 0.0,
                "latency_max_ms": 0.0,
                "speedup_vs_baseline": 0.0,
                "gpu_mem_mb": self.gpu_mem,
                "status": "error",
                "error_message": self.error_message,
            }
        arr = self.latencies
        mean_val = float(np.mean(arr))
        return {
            "experiment": self.experiment,
            "backend": "torch_cuda",
            "num_streams": self.num_streams,
            "kernel": self.kernel,
            "matrix_size": self.matrix_size,
            "iters_per_stream": self.iters_per_stream,
            "total_work_iters": self.total_work_iters,
            "latency_mean_ms": round(mean_val, 6),
            "latency_p50_ms": round(pct(arr, 50), 6),
            "latency_p95_ms": round(pct(arr, 95), 6),
            "latency_min_ms": round(float(np.min(arr)), 6),
            "latency_max_ms": round(float(np.max(arr)), 6),
            "speedup_vs_baseline": round(baseline_ms / mean_val, 4) if baseline_ms > 0 and mean_val > 0 else 1.0,
            "gpu_mem_mb": round(self.gpu_mem, 2),
            "status": self.status,
            "error_message": self.error_message,
        }


# ── Experiment 1: Single-stream baseline ───────────────────────────────────


def run_single_stream(kernel_fn, matrix_size: int, iters: int, warmup: int, repeat: int) -> list[float]:
    """All work on the default stream. Returns list of per-repeat latencies (ms)."""
    latencies: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        kernel_fn(matrix_size, iters)

    for _ in range(repeat):
        torch.cuda.synchronize()
        start_ev.record()
        kernel_fn(matrix_size, iters)
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(torch_event_time(start_ev, end_ev))

    return latencies


# ── Experiment 2: N-stream concurrent ───────────────────────────────────────


def run_multi_stream(kernel_fn, matrix_size: int, iters: int, num_streams: int,
                     warmup: int, repeat: int) -> list[float]:
    """Launch identical kernels on N streams concurrently. Returns wall-time latencies (ms)."""
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    latencies: list[float] = []
    wall_start = torch.cuda.Event(enable_timing=True)
    wall_end = torch.cuda.Event(enable_timing=True)

    def _dispatch() -> None:
        for s in streams:
            with torch.cuda.stream(s):
                kernel_fn(matrix_size, iters)

    # Warmup
    for _ in range(warmup):
        _dispatch()
    torch.cuda.synchronize()

    for _ in range(repeat):
        torch.cuda.synchronize()
        wall_start.record()
        _dispatch()
        wall_end.record()
        torch.cuda.synchronize()
        latencies.append(torch_event_time(wall_start, wall_end))

    return latencies


# ── Experiment 3: Compute-copy overlap ──────────────────────────────────────


def run_copy_compute_overlap(matrix_size: int, iters: int, warmup: int, repeat: int) -> list[float]:
    """
    Stream A: H2D → compute → D2H (serial per chunk)
    Stream B: H2D → compute → D2H (serial per chunk)
    Both streams run concurrently — measure wall time.
    """
    latencies: list[float] = []
    wall_start = torch.cuda.Event(enable_timing=True)
    wall_end = torch.cuda.Event(enable_timing=True)

    def _overlap_work() -> None:
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        # Pre-allocate
        a_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
        b_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
        c_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
        d_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)

        out1_cpu = torch.empty(matrix_size, matrix_size, dtype=torch.float32)
        out2_cpu = torch.empty(matrix_size, matrix_size, dtype=torch.float32)

        with torch.cuda.stream(s1):
            a = a_cpu.to("cuda", non_blocking=True)
            b = b_cpu.to("cuda", non_blocking=True)
            for _ in range(iters):
                a = torch.matmul(a, b)
            out1_cpu.copy_(a, non_blocking=True)

        with torch.cuda.stream(s2):
            c = c_cpu.to("cuda", non_blocking=True)
            d = d_cpu.to("cuda", non_blocking=True)
            for _ in range(iters):
                c = torch.matmul(c, d)
            out2_cpu.copy_(c, non_blocking=True)

        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _overlap_work()

    for _ in range(repeat):
        torch.cuda.synchronize()
        wall_start.record()
        _overlap_work()
        wall_end.record()
        torch.cuda.synchronize()
        latencies.append(torch_event_time(wall_start, wall_end))

    return latencies


# ── Experiment 4: Stream sync overhead ──────────────────────────────────────


def run_sync_overhead(num_streams: int, warmup: int, repeat: int) -> list[float]:
    """Measure raw cudaStreamSynchronize / event overhead (no kernel)."""
    latencies: list[float] = []
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    def _sync_all() -> None:
        for s in streams:
            s.synchronize()

    for _ in range(warmup):
        _sync_all()

    for _ in range(repeat):
        start_ev.record()
        _sync_all()
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(torch_event_time(start_ev, end_ev))

    return latencies


# ── Main ────────────────────────────────────────────────────────────────────


def run_all(args: argparse.Namespace) -> list[StreamResult]:
    results: list[StreamResult] = []

    for kernel_name in args.kernels:
        kernel_fn = KERNELS[kernel_name]

        # --- baseline: single stream ---
        reset_peak()
        bl_lat = run_single_stream(kernel_fn, args.matrix_size, args.iters, args.warmup, args.repeat)
        bl_ms = float(np.mean(bl_lat)) if bl_lat else 0.0
        results.append(StreamResult(
            experiment="single_stream",
            num_streams=1,
            kernel=kernel_name,
            matrix_size=args.matrix_size,
            iters_per_stream=args.iters,
            total_work_iters=args.iters,
            latencies=bl_lat,
            gpu_mem=peak_mem_mb(),
        ))

        # --- N-stream concurrent (2, 4, 8, ...) ---
        for n in args.num_streams_list:
            if n <= 1:
                continue
            reset_peak()
            lat = run_multi_stream(kernel_fn, args.matrix_size, args.iters, n, args.warmup, args.repeat)
            r = StreamResult(
                experiment=f"multi_stream_{n}",
                num_streams=n,
                kernel=kernel_name,
                matrix_size=args.matrix_size,
                iters_per_stream=args.iters,
                total_work_iters=args.iters * n,
                latencies=lat,
                gpu_mem=peak_mem_mb(),
            )
            results.append(r)
            mean_val = float(np.mean(lat)) if lat else 0.0
            print(f"  {kernel_name:10s} streams={n:2d}  wall_ms={mean_val:.4f}  speedup_vs_baseline={bl_ms / mean_val:.2f}x" if mean_val > 0 and bl_ms > 0 else f"  {kernel_name:10s} streams={n:2d}  wall_ms={mean_val:.4f}")

    # --- copy-compute overlap ---
    reset_peak()
    ov_lat = run_copy_compute_overlap(args.matrix_size, args.iters, args.warmup, args.repeat)
    results.append(StreamResult(
        experiment="copy_compute_overlap",
        num_streams=2,
        kernel="gevm",
        matrix_size=args.matrix_size,
        iters_per_stream=args.iters,
        total_work_iters=args.iters * 2,
        latencies=ov_lat,
        gpu_mem=peak_mem_mb(),
    ))

    # --- sync overhead ---
    for n in args.num_streams_list:
        reset_peak()
        sync_lat = run_sync_overhead(n, args.warmup, args.repeat)
        results.append(StreamResult(
            experiment=f"sync_overhead_{n}",
            num_streams=n,
            kernel="none",
            matrix_size=0,
            iters_per_stream=0,
            total_work_iters=0,
            latencies=sync_lat,
            gpu_mem=peak_mem_mb(),
        ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA streams concurrency benchmark")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "cuda_streams_results.csv"))
    parser.add_argument("--kernels", default="gevm,vector,conv2d", help="Comma-separated kernel names")
    parser.add_argument("--matrix-size", type=int, default=1024, help="Matrix/vector size for kernels")
    parser.add_argument("--iters", type=int, default=10, help="Kernel iterations per launch")
    parser.add_argument("--num-streams", default="2,4,8", help="Comma-separated stream counts")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    args.kernels = [k.strip() for k in args.kernels.split(",") if k.strip() in KERNELS]
    args.num_streams_list = [int(x.strip()) for x in args.num_streams.split(",") if x.strip().isdigit()]

    setup_env()

    print(f"CUDA streams concurrency benchmark")
    print(f"  kernels: {args.kernels}")
    print(f"  matrix_size: {args.matrix_size}")
    print(f"  iters: {args.iters}")
    print(f"  streams: {args.num_streams_list}")
    print(f"  warmup={args.warmup} repeat={args.repeat}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print()

    results = run_all(args)

    # Compute speedups (baseline = single_stream per kernel)
    baselines: dict[str, float] = {}
    for r in results:
        if r.experiment == "single_stream":
            baselines[r.kernel] = float(np.mean(r.latencies)) if r.latencies else 0.0

    rows = []
    for r in results:
        bl = baselines.get(r.kernel, 0.0)
        rows.append(r.to_row(bl))

    out_path = Path(args.output)
    write_csv(rows, out_path)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary table
    print(f"\n{'Experiment':<25s} {'Kernel':<10s} {'Streams':>7s} {'Mean(ms)':>10s} {'Speedup':>8s}")
    print("-" * 65)
    for row in rows:
        print(f"{row['experiment']:<25s} {row['kernel']:<10s} {row['num_streams']:>7d} {row['latency_mean_ms']:>10.4f} {row['speedup_vs_baseline']:>7.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
