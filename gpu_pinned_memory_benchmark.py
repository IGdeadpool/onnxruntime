import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_runtime import detect_runtime


ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/home/l/benchmarks" if Path("/home/l").exists() else str(Path.home() / "benchmarks")))
OUTPUT_DIR = ROOT / "outputs"
FIELDS = [
    "experiment",
    "backend",
    "direction",
    "memory_type",
    "tensor_size_mb",
    "num_elements",
    "latency_mean_ms",
    "latency_p95_ms",
    "bandwidth_gb_s",
    "overlap_speedup",
    "status",
    "error_message",
]


def setup_env() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available; this benchmark requires CUDA or ROCm through PyTorch.")


def backend_name() -> str:
    return detect_runtime().torch_backend


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def wall_ms(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def measure_copy(numel: int, direction: str, pinned: bool, warmup: int, repeat: int) -> list[float]:
    torch.cuda.reset_peak_memory_stats()
    cpu_src = torch.randn(numel, dtype=torch.float32, pin_memory=pinned)
    gpu_src = torch.randn(numel, device="cuda", dtype=torch.float32)
    cpu_dst = torch.empty(numel, dtype=torch.float32, pin_memory=pinned)
    gpu_dst = torch.empty(numel, device="cuda", dtype=torch.float32)

    def h2d() -> None:
        gpu_dst.copy_(cpu_src, non_blocking=pinned)

    def d2h() -> None:
        cpu_dst.copy_(gpu_src, non_blocking=pinned)

    fn = h2d if direction == "H2D" else d2h
    for _ in range(warmup):
        fn()
    return [wall_ms(fn) for _ in range(repeat)]


def pageable_vs_pinned(size_mb: int, warmup: int, repeat: int, backend: str) -> list[dict[str, object]]:
    numel = (size_mb * 1024 * 1024) // 4
    bytes_moved = float(numel * 4)
    rows: list[dict[str, object]] = []
    for direction in ("H2D", "D2H"):
        for memory_type, pinned in (("pageable", False), ("pinned", True)):
            lat = measure_copy(numel, direction, pinned, warmup, repeat)
            mean_ms = float(np.mean(lat)) if lat else 0.0
            bw = bytes_moved / (mean_ms / 1000.0) / 1e9 if mean_ms > 0 else 0.0
            rows.append(
                {
                    "experiment": f"{direction}_{memory_type}",
                    "backend": backend,
                    "direction": direction,
                    "memory_type": memory_type,
                    "tensor_size_mb": size_mb,
                    "num_elements": numel,
                    "latency_mean_ms": round(mean_ms, 6),
                    "latency_p95_ms": round(pct(lat, 95), 6),
                    "bandwidth_gb_s": round(bw, 4),
                    "overlap_speedup": 0.0,
                    "status": "ok",
                    "error_message": "",
                }
            )
    return rows


def copy_compute_overlap(size: int, iters: int, direction: str, warmup: int, repeat: int) -> tuple[list[float], list[float]]:
    cpu = torch.randn(size, size, dtype=torch.float32, pin_memory=True)
    gpu_copy_src = torch.randn(size, size, device="cuda", dtype=torch.float32)
    gpu_copy_dst = torch.empty(size, size, device="cuda", dtype=torch.float32)
    cpu_dst = torch.empty(size, size, dtype=torch.float32, pin_memory=True)
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def compute() -> None:
        x = a
        for _ in range(iters):
            x = torch.matmul(x, b)

    def copy() -> None:
        if direction == "H2D":
            gpu_copy_dst.copy_(cpu, non_blocking=True)
        else:
            cpu_dst.copy_(gpu_copy_src, non_blocking=True)

    def serial() -> None:
        copy()
        compute()

    def overlapped() -> None:
        with torch.cuda.stream(copy_stream):
            copy()
        with torch.cuda.stream(compute_stream):
            compute()

    for _ in range(warmup):
        serial()
        overlapped()
    return [wall_ms(serial) for _ in range(repeat)], [wall_ms(overlapped) for _ in range(repeat)]


def pin_overhead(size_mb: int, repeat: int) -> tuple[float, float]:
    numel = (size_mb * 1024 * 1024) // 4
    pin_lat: list[float] = []
    clone_lat: list[float] = []
    for _ in range(repeat):
        t = torch.randn(numel, dtype=torch.float32)
        start = time.perf_counter()
        pinned = t.pin_memory()
        pin_lat.append((time.perf_counter() - start) * 1000.0)
        del pinned

        start = time.perf_counter()
        clone = t.clone()
        clone_lat.append((time.perf_counter() - start) * 1000.0)
        del clone, t
    return float(np.mean(pin_lat)), float(np.mean(clone_lat))


def pinned_throughput(size: int, batches: int, warmup: int, repeat: int) -> tuple[float, float, float]:
    tensors = [torch.randn(size, size, dtype=torch.float32, pin_memory=True) for _ in range(batches)]
    outputs = [torch.empty(size, size, device="cuda", dtype=torch.float32) for _ in range(batches)]

    def copy_all() -> None:
        for src, dst in zip(tensors, outputs):
            dst.copy_(src, non_blocking=True)

    for _ in range(warmup):
        copy_all()
    lat = [wall_ms(copy_all) for _ in range(repeat)]
    mean_ms = float(np.mean(lat)) if lat else 0.0
    total_mb = batches * size * size * 4 / (1024 * 1024)
    bw = (total_mb * 1024 * 1024) / (mean_ms / 1000.0) / 1e9 if mean_ms > 0 else 0.0
    return mean_ms, bw, total_mb


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_rows(rows: list[dict[str, object]], output: Path, new_rows: list[dict[str, object]]) -> None:
    rows.extend(new_rows)
    write_csv(rows, output)
    for row in new_rows:
        print(
            "[OK] "
            f"{row['experiment']} direction={row['direction']} size_mb={row['tensor_size_mb']} "
            f"mean_ms={row['latency_mean_ms']} bandwidth_gb_s={row['bandwidth_gb_s']} output={output}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU pinned memory transfer benchmark for CUDA and ROCm.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "gpu_pinned_memory_results.csv"))
    parser.add_argument("--size-mb", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--overlap-size", type=int, default=256)
    parser.add_argument("--overlap-iters", type=int, default=1)
    parser.add_argument("--throughput-batches", type=int, default=4)
    parser.add_argument("--throughput-size", type=int, default=128)
    parser.add_argument("--pin-overhead-mb", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    setup_env()
    require_gpu()
    backend = backend_name()
    print("GPU pinned memory benchmark")
    print(f"backend={backend}")
    print(f"device={torch.cuda.get_device_name(0)}")
    rows: list[dict[str, object]] = []
    output = Path(args.output)
    write_csv(rows, output)
    print(
        f"planned_size_mb={','.join(map(str, args.size_mb))} overlap_size={args.overlap_size} "
        f"overlap_iters={args.overlap_iters} warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )
    print(f"output={output} (created before the first measurement)", flush=True)

    for size_mb in args.size_mb:
        print(f"[START] pageable_vs_pinned size_mb={size_mb}", flush=True)
        record_rows(rows, output, pageable_vs_pinned(size_mb, args.warmup, args.repeat, backend))
        torch.cuda.empty_cache()

    for direction in ("H2D", "D2H"):
        print(f"[START] overlap_{direction} size={args.overlap_size}", flush=True)
        serial, overlapped = copy_compute_overlap(args.overlap_size, args.overlap_iters, direction, args.warmup, args.repeat)
        serial_ms = float(np.mean(serial)) if serial else 0.0
        overlap_ms = float(np.mean(overlapped)) if overlapped else 0.0
        record_rows(
            rows,
            output,
            [{
                "experiment": f"overlap_{direction}",
                "backend": backend,
                "direction": direction,
                "memory_type": "pinned",
                "tensor_size_mb": round(args.overlap_size * args.overlap_size * 4 / (1024 * 1024), 2),
                "num_elements": args.overlap_size * args.overlap_size,
                "latency_mean_ms": round(overlap_ms, 6),
                "latency_p95_ms": round(pct(overlapped, 95), 6),
                "bandwidth_gb_s": 0.0,
                "overlap_speedup": round(serial_ms / overlap_ms, 4) if overlap_ms > 0 else 1.0,
                "status": "ok",
                "error_message": "",
            }],
        )
        torch.cuda.empty_cache()

    print("[START] throughput_batched", flush=True)
    throughput_ms, throughput_bw, total_mb = pinned_throughput(
        args.throughput_size, args.throughput_batches, args.warmup, args.repeat
    )
    record_rows(
        rows,
        output,
        [{
            "experiment": "throughput_batched",
            "backend": backend,
            "direction": "H2D",
            "memory_type": "pinned",
            "tensor_size_mb": round(total_mb, 2),
            "num_elements": args.throughput_batches * args.throughput_size * args.throughput_size,
            "latency_mean_ms": round(throughput_ms, 6),
            "latency_p95_ms": 0.0,
            "bandwidth_gb_s": round(throughput_bw, 4),
            "overlap_speedup": 0.0,
            "status": "ok",
            "error_message": "",
        }],
    )

    print("[START] pin_overhead", flush=True)
    pin_ms, clone_ms = pin_overhead(args.pin_overhead_mb, args.repeat)
    record_rows(
        rows,
        output,
        [{
            "experiment": "pin_overhead",
            "backend": backend,
            "direction": "N/A",
            "memory_type": "pinned",
            "tensor_size_mb": args.pin_overhead_mb,
            "num_elements": (args.pin_overhead_mb * 1024 * 1024) // 4,
            "latency_mean_ms": round(pin_ms, 6),
            "latency_p95_ms": 0.0,
            "bandwidth_gb_s": 0.0,
            "overlap_speedup": round(pin_ms / clone_ms, 4) if clone_ms > 0 else 0.0,
            "status": "ok",
            "error_message": "",
        }],
    )

    write_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
