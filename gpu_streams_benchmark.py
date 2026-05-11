import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmark_runtime import detect_runtime


ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/home/l/benchmarks" if Path("/home/l").exists() else str(Path.home() / "benchmarks")))
OUTPUT_DIR = ROOT / "outputs"
FIELDS = [
    "experiment",
    "backend",
    "kernel",
    "num_streams",
    "matrix_size",
    "iters_per_stream",
    "total_work_iters",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_min_ms",
    "latency_max_ms",
    "speedup_vs_serial",
    "gpu_mem_mb",
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


def wall_ms(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


class GemmKernel:
    def __init__(self, size: int, iters: int) -> None:
        self.iters = iters
        self.a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        self.b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    def compute(self) -> None:
        a = self.a
        for _ in range(self.iters):
            a = torch.matmul(a, self.b)
        self.a = a


class VectorAddKernel:
    def __init__(self, size: int, iters: int) -> None:
        self.iters = iters
        self.a = torch.randn(size, device="cuda", dtype=torch.float32)
        self.b = torch.randn(size, device="cuda", dtype=torch.float32)

    def compute(self) -> None:
        a = self.a
        for _ in range(self.iters):
            a = a + self.b
        self.a = a


class Conv2dKernel:
    def __init__(self, size: int, iters: int) -> None:
        self.iters = iters
        self.x = torch.randn(1, 64, size, size, device="cuda", dtype=torch.float32)
        self.conv = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False).to("cuda").eval()

    def compute(self) -> None:
        x = self.x
        for _ in range(self.iters):
            x = self.conv(x)
        self.x = x


KERNELS = {
    "gemm": GemmKernel,
    "vectorAdd": VectorAddKernel,
    "conv2d": Conv2dKernel,
}


@dataclass
class Result:
    experiment: str
    kernel: str
    num_streams: int
    matrix_size: int
    iters_per_stream: int
    total_work_iters: int
    latencies: list[float]
    backend: str
    baseline_ms: float = 0.0
    status: str = "ok"
    error_message: str = ""

    def row(self) -> dict[str, object]:
        mean_ms = float(np.mean(self.latencies)) if self.latencies else 0.0
        return {
            "experiment": self.experiment,
            "backend": self.backend,
            "kernel": self.kernel,
            "num_streams": self.num_streams,
            "matrix_size": self.matrix_size,
            "iters_per_stream": self.iters_per_stream,
            "total_work_iters": self.total_work_iters,
            "latency_mean_ms": round(mean_ms, 6),
            "latency_p50_ms": round(pct(self.latencies, 50), 6),
            "latency_p95_ms": round(pct(self.latencies, 95), 6),
            "latency_min_ms": round(float(np.min(self.latencies)), 6) if self.latencies else 0.0,
            "latency_max_ms": round(float(np.max(self.latencies)), 6) if self.latencies else 0.0,
            "speedup_vs_serial": round(self.baseline_ms / mean_ms, 4) if self.baseline_ms > 0 and mean_ms > 0 else 1.0,
            "gpu_mem_mb": round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 2),
            "status": self.status,
            "error_message": self.error_message,
        }


def serial_prealloc(kernel_cls: type, size: int, iters: int, work_count: int) -> Callable[[], None]:
    instances = [kernel_cls(size, iters) for _ in range(work_count)]

    def run() -> None:
        for item in instances:
            item.compute()

    return run


def multistream_prealloc(kernel_cls: type, size: int, iters: int, num_streams: int) -> Callable[[], None]:
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    instances = [kernel_cls(size, iters) for _ in range(num_streams)]

    def run() -> None:
        for stream, item in zip(streams, instances):
            with torch.cuda.stream(stream):
                item.compute()

    return run


def measure(fn: Callable[[], None], warmup: int, repeat: int) -> list[float]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    return [wall_ms(fn) for _ in range(repeat)]


def copy_compute_overlap(size: int, iters: int, warmup: int, repeat: int) -> tuple[list[float], list[float]]:
    cpu = torch.randn(size, size, dtype=torch.float32).pin_memory()
    gpu_dst = torch.empty(size, size, device="cuda", dtype=torch.float32)
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def compute() -> None:
        x = a
        for _ in range(iters):
            x = torch.matmul(x, b)

    def copy_h2d() -> None:
        gpu_dst.copy_(cpu, non_blocking=True)

    def serial() -> None:
        copy_h2d()
        compute()

    def overlapped() -> None:
        with torch.cuda.stream(copy_stream):
            copy_h2d()
        with torch.cuda.stream(compute_stream):
            compute()

    for _ in range(warmup):
        serial()
        overlapped()
    return [wall_ms(serial) for _ in range(repeat)], [wall_ms(overlapped) for _ in range(repeat)]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_result(rows: list[dict[str, object]], output: Path, result: Result) -> None:
    row = result.row()
    rows.append(row)
    write_csv(rows, output)
    print(
        "[OK] "
        f"{row['experiment']} kernel={row['kernel']} streams={row['num_streams']} "
        f"mean_ms={row['latency_mean_ms']} speedup={row['speedup_vs_serial']} output={output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU stream concurrency benchmark for CUDA and ROCm.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "gpu_streams_results.csv"))
    parser.add_argument("--kernels", default="vectorAdd,gemm")
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--num-streams", default="2,4")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    setup_env()
    require_gpu()
    backend = backend_name()
    kernels = [name.strip() for name in args.kernels.split(",") if name.strip() in KERNELS]
    stream_counts = [int(x.strip()) for x in args.num_streams.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    output = Path(args.output)
    write_csv(rows, output)

    print("GPU streams benchmark")
    print(f"backend={backend}")
    print(f"device={torch.cuda.get_device_name(0)}")
    planned = len(kernels) * (1 + len(stream_counts) * 2) + 2
    print(
        f"planned_experiments={planned} kernels={','.join(kernels)} streams={','.join(map(str, stream_counts))} "
        f"matrix_size={args.matrix_size} iters={args.iters} warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )
    print(f"output={output} (created before the first measurement)", flush=True)

    for kernel_name in kernels:
        kernel_cls = KERNELS[kernel_name]
        print(f"[START] single_stream_prealloc kernel={kernel_name}", flush=True)
        single_fn = serial_prealloc(kernel_cls, args.matrix_size, args.iters, 1)
        single_lat = measure(single_fn, args.warmup, args.repeat)
        record_result(rows, output, Result("single_stream_prealloc", kernel_name, 1, args.matrix_size, args.iters, args.iters, single_lat, backend))
        del single_fn
        torch.cuda.empty_cache()
        for n in stream_counts:
            print(f"[START] serial_{n}_work_items kernel={kernel_name}", flush=True)
            serial_fn = serial_prealloc(kernel_cls, args.matrix_size, args.iters, n)
            serial_lat = measure(serial_fn, args.warmup, args.repeat)
            serial_ms = float(np.mean(serial_lat)) if serial_lat else 0.0
            record_result(rows, output, Result(f"serial_{n}_work_items", kernel_name, 1, args.matrix_size, args.iters, args.iters * n, serial_lat, backend))
            del serial_fn
            torch.cuda.empty_cache()

            print(f"[START] multi_stream_{n} kernel={kernel_name}", flush=True)
            multi_fn = multistream_prealloc(kernel_cls, args.matrix_size, args.iters, n)
            multi_lat = measure(multi_fn, args.warmup, args.repeat)
            record_result(rows, output, Result(f"multi_stream_{n}", kernel_name, n, args.matrix_size, args.iters, args.iters * n, multi_lat, backend, serial_ms))
            del multi_fn
            torch.cuda.empty_cache()

    print("[START] copy_compute_serial/overlap kernel=gemm+h2d", flush=True)
    serial_copy, overlap_copy = copy_compute_overlap(args.matrix_size, args.iters, args.warmup, args.repeat)
    serial_ms = float(np.mean(serial_copy)) if serial_copy else 0.0
    record_result(rows, output, Result("copy_compute_serial", "gemm+h2d", 1, args.matrix_size, args.iters, args.iters, serial_copy, backend))
    record_result(rows, output, Result("copy_compute_overlap", "gemm+h2d", 2, args.matrix_size, args.iters, args.iters, overlap_copy, backend, serial_ms))

    write_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
