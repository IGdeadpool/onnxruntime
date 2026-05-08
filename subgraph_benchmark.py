import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_runtime import RuntimeConfig, detect_runtime
from operator_benchmark import compare_outputs, gbps, onnx_graph_ops, pct, tflops


ROOT = Path("/home/l/benchmarks")
OUTPUT_DIR = ROOT / "outputs"
ONNX_DIR = ROOT / "models" / "onnx_subgraphs"
PROFILE_DIR = OUTPUT_DIR / "ort_profiles"
RUNTIME: RuntimeConfig = detect_runtime()


def setup_env() -> None:
    os.environ.setdefault("LD_LIBRARY_PATH", "/opt/rocm-7.2.0/lib:/opt/rocm/lib:/usr/lib/wsl/lib")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn: Callable[[], object], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_p50_ms": pct(times, 50),
        "latency_p95_ms": pct(times, 95),
        "latency_min_ms": float(np.min(times)),
        "latency_max_ms": float(np.max(times)),
    }


def time_once(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - start) * 1000.0


def mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def tensor_bytes(*shapes: tuple[int, ...], dtype_bytes: int = 4) -> int:
    return sum(int(np.prod(shape)) * dtype_bytes for shape in shapes)


class ConvBnRelu(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class TransformerMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden)
        return self.out(context)


def subgraph_specs(batch: int, seq_len: int, shape_profile: str) -> list[dict[str, object]]:
    if shape_profile == "large":
        channels, hw, hidden, intermediate, heads = 256, 56, 1024, 4096, 16
    else:
        channels, hw, hidden, intermediate, heads = 64, 56, 768, 3072, 12

    x4 = torch.randn(batch, channels, hw, hw)
    x3 = torch.randn(batch, seq_len, hidden)
    conv_flops = 2.0 * batch * channels * hw * hw * channels * 3 * 3
    mlp_flops = 2.0 * batch * seq_len * hidden * intermediate * 2
    attn_flops = (
        2.0 * batch * seq_len * hidden * hidden * 4
        + 4.0 * batch * heads * seq_len * seq_len * (hidden // heads)
    )

    return [
        {
            "subgraph_name": "conv_bn_relu",
            "model_family": "resnet",
            "module": ConvBnRelu(channels).eval(),
            "inputs": (x4,),
            "input_names": ["input"],
            "layout": "nchw",
            "attributes": f"channels={channels};h={hw};w={hw};conv=3x3;bn=true;relu=true",
            "flops": conv_flops,
            "bytes_moved": tensor_bytes(tuple(x4.shape), tuple(x4.shape)) * 3,
        },
        {
            "subgraph_name": "resnet_basic_block",
            "model_family": "resnet",
            "module": BasicBlock(channels).eval(),
            "inputs": (x4,),
            "input_names": ["input"],
            "layout": "nchw",
            "attributes": f"channels={channels};h={hw};w={hw};conv_count=2;residual=true",
            "flops": conv_flops * 2,
            "bytes_moved": tensor_bytes(tuple(x4.shape), tuple(x4.shape)) * 5,
        },
        {
            "subgraph_name": "transformer_mlp",
            "model_family": "distilbert",
            "module": TransformerMlp(hidden, intermediate).eval(),
            "inputs": (x3,),
            "input_names": ["input"],
            "layout": "bsh",
            "attributes": f"hidden={hidden};intermediate={intermediate};gelu=true",
            "flops": mlp_flops,
            "bytes_moved": tensor_bytes(tuple(x3.shape), tuple(x3.shape)) * 3,
        },
        {
            "subgraph_name": "self_attention",
            "model_family": "distilbert",
            "module": SelfAttentionBlock(hidden, heads).eval(),
            "inputs": (x3,),
            "input_names": ["input"],
            "layout": "bsh",
            "attributes": f"hidden={hidden};heads={heads};seq_len={seq_len};softmax=true",
            "flops": attn_flops,
            "bytes_moved": tensor_bytes(tuple(x3.shape), tuple(x3.shape)) * 5,
        },
    ]


def output_shape(outputs: object) -> str:
    if isinstance(outputs, torch.Tensor):
        return str(tuple(outputs.shape))
    if isinstance(outputs, (list, tuple)):
        return ";".join(str(tuple(item.shape)) for item in outputs if isinstance(item, torch.Tensor))
    return ""


def row(
    *,
    spec: dict[str, object],
    backend: str,
    provider: str,
    batch_size: int,
    seq_len: int,
    repeat_id: int,
    shape_profile: str,
    stats: dict[str, float],
    out_shape: str,
    graph_ops: str = "",
    session_create_ms: float | str = "",
    first_run_ms: float | str = "",
    profile_path: str = "",
    correctness_status: str = "",
    max_abs_error: float | str = "",
    max_rel_error: float | str = "",
    correctness_message: str = "",
    status: str = "ok",
    error_message: str = "",
) -> dict[str, object]:
    latency = stats.get("latency_mean_ms", 0.0)
    inputs = spec["inputs"]
    assert isinstance(inputs, tuple)
    return {
        "subgraph_name": spec["subgraph_name"],
        "model_family": spec["model_family"],
        "backend": backend,
        "provider": provider,
        "repeat_id": repeat_id,
        "shape_profile": shape_profile,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": "fp32",
        "layout": spec["layout"],
        "input_shape": ";".join(str(tuple(item.shape)) for item in inputs),
        "output_shape": out_shape,
        "graph_ops": graph_ops,
        "attributes": spec["attributes"],
        "session_create_ms": session_create_ms,
        "first_run_ms": first_run_ms,
        "latency_mean_ms": latency,
        "latency_p50_ms": stats.get("latency_p50_ms", 0.0),
        "latency_p95_ms": stats.get("latency_p95_ms", 0.0),
        "latency_min_ms": stats.get("latency_min_ms", 0.0),
        "latency_max_ms": stats.get("latency_max_ms", 0.0),
        "tflops": tflops(float(spec["flops"]), latency),
        "bandwidth_gb_s": gbps(float(spec["bytes_moved"]), latency),
        "gpu_mem_mb": mem_mb() if backend.startswith("torch") else "",
        "correctness_status": correctness_status,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "correctness_message": correctness_message,
        "profile_path": profile_path,
        "status": status,
        "error_message": error_message,
    }


def export_onnx(module: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path, input_names: list[str]) -> Path:
    module.eval().cpu()
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=input_names,
        output_names=["output"],
        opset_version=18,
    )
    return path


def run_onnx(
    path: Path,
    feed: dict[str, np.ndarray],
    warmup: int,
    iters: int,
    profile_name: str,
    providers: list[str],
) -> tuple[dict[str, float], str, str, float, float, str, list[np.ndarray]]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(PROFILE_DIR / profile_name)
    start = time.perf_counter()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    session_create_ms = (time.perf_counter() - start) * 1000.0
    provider = ",".join(sess.get_providers())
    first_run_ms = time_once(lambda: sess.run(None, feed))
    stats = bench(lambda: sess.run(None, feed), warmup, iters)
    outputs = sess.run(None, feed)
    out_shape = ";".join(str(tuple(out.shape)) for out in outputs)
    profile_path = sess.end_profiling()
    return stats, provider, out_shape, session_create_ms, first_run_ms, profile_path, outputs


def run_torch(
    specs: list[dict[str, object]],
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    repeat_id: int,
    shape_profile: str,
) -> list[dict[str, object]]:
    dev = device()
    rows: list[dict[str, object]] = []
    for spec in specs:
        try:
            module = spec["module"]
            inputs = spec["inputs"]
            assert isinstance(module, nn.Module)
            assert isinstance(inputs, tuple)
            module = module.to(dev).eval()
            dev_inputs = tuple(item.to(dev) for item in inputs)
            with torch.no_grad():
                ref = module(*dev_inputs)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                stats = bench(lambda: module(*dev_inputs), warmup, iters)
            rows.append(
                row(
                    spec=spec,
                    backend=RUNTIME.torch_backend,
                    provider=RUNTIME.device_name,
                    batch_size=batch,
                    seq_len=seq_len,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    stats=stats,
                    out_shape=output_shape(ref),
                    correctness_status="reference",
                    correctness_message="torch_eager_reference",
                )
            )
        except Exception as exc:
            rows.append(
                row(
                    spec=spec,
                    backend=RUNTIME.torch_backend,
                    provider=RUNTIME.device_name,
                    batch_size=batch,
                    seq_len=seq_len,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    stats={},
                    out_shape="",
                    status="error",
                    error_message=str(exc),
                )
            )
    return rows


def run_onnxruntime(
    specs: list[dict[str, object]],
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    repeat_id: int,
    shape_profile: str,
    correctness_rtol: float,
    correctness_atol: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        name = str(spec["subgraph_name"])
        module = spec["module"]
        inputs = spec["inputs"]
        input_names = spec["input_names"]
        assert isinstance(module, nn.Module)
        assert isinstance(inputs, tuple)
        assert isinstance(input_names, list)
        path = ONNX_DIR / f"subgraph_{name}_{shape_profile}_bs{batch}_seq{seq_len}_rep{repeat_id}.onnx"
        try:
            export_onnx(module, inputs, path, input_names)
            feed = {n: t.detach().cpu().numpy().astype(np.float32) for n, t in zip(input_names, inputs)}
            profile_name = f"ort_subgraph_{name}_{shape_profile}_chain1_bs{batch}_rep{repeat_id}"
            stats, provider, out_shape, session_create_ms, first_run_ms, profile_path, ort_outputs = run_onnx(
                path,
                feed,
                warmup,
                iters,
                profile_name,
                RUNTIME.onnx_providers,
            )
            with torch.no_grad():
                reference_outputs = module.cpu().eval()(*inputs)
            correctness = compare_outputs(reference_outputs, ort_outputs, rtol=correctness_rtol, atol=correctness_atol)
            rows.append(
                row(
                    spec=spec,
                    backend="onnxruntime",
                    provider=provider,
                    batch_size=batch,
                    seq_len=seq_len,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    stats=stats,
                    out_shape=out_shape,
                    graph_ops=onnx_graph_ops(path),
                    session_create_ms=session_create_ms,
                    first_run_ms=first_run_ms,
                    profile_path=profile_path,
                    correctness_status=str(correctness["correctness_status"]),
                    max_abs_error=correctness["max_abs_error"],
                    max_rel_error=correctness["max_rel_error"],
                    correctness_message=str(correctness["correctness_message"]),
                )
            )
        except Exception as exc:
            rows.append(
                row(
                    spec=spec,
                    backend="onnxruntime",
                    provider="",
                    batch_size=batch,
                    seq_len=seq_len,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    stats={},
                    out_shape="",
                    status="error",
                    error_message=str(exc),
                )
            )
    return rows


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subgraph_name",
        "model_family",
        "backend",
        "provider",
        "repeat_id",
        "shape_profile",
        "batch_size",
        "seq_len",
        "dtype",
        "layout",
        "input_shape",
        "output_shape",
        "graph_ops",
        "attributes",
        "session_create_ms",
        "first_run_ms",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_min_ms",
        "latency_max_ms",
        "tflops",
        "bandwidth_gb_s",
        "gpu_mem_mb",
        "correctness_status",
        "max_abs_error",
        "max_rel_error",
        "correctness_message",
        "profile_path",
        "status",
        "error_message",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in fields})


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-subgraph benchmark for PyTorch and ONNX Runtime providers.")
    parser.add_argument("--backends", default="all", help="all,torch,onnx")
    parser.add_argument("--batches", default="1,8,16")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shape-profile", choices=["standard", "large"], default="large")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "subgraph_results.csv"))
    parser.add_argument("--onnx-backend", default="auto", help="auto,migraphx,cuda,cpu,custom")
    parser.add_argument("--onnx-providers", default="auto", help="auto or comma list such as CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--device-label", default="auto", help="auto or a stable label such as rx9070xt_rocm/rtx3080_cuda")
    parser.add_argument("--correctness-rtol", type=float, default=1e-3)
    parser.add_argument("--correctness-atol", type=float, default=1e-4)
    args = parser.parse_args()

    global RUNTIME
    RUNTIME = detect_runtime(args.onnx_backend, args.onnx_providers, args.device_label)
    setup_env()
    print(
        f"Runtime device_label={RUNTIME.device_label} device_name={RUNTIME.device_name} "
        f"torch_backend={RUNTIME.torch_backend} onnx_backend={RUNTIME.onnx_backend} "
        f"onnx_providers={','.join(RUNTIME.onnx_providers)}"
    )

    backends = {x.strip() for x in args.backends.split(",") if x.strip()}
    run_torch_backend = "all" in backends or "torch" in backends
    run_onnx_backend = "all" in backends or "onnx" in backends
    rows: list[dict[str, object]] = []
    for repeat_id in range(1, args.repeat + 1):
        for batch in parse_csv_ints(args.batches):
            torch.manual_seed(1000 + repeat_id * 100 + batch)
            specs = subgraph_specs(batch, args.seq_len, args.shape_profile)
            if run_torch_backend:
                print(f"Running torch subgraphs repeat={repeat_id} batch={batch}")
                rows.extend(run_torch(specs, batch, args.seq_len, args.warmup, args.iters, repeat_id, args.shape_profile))
            if run_onnx_backend:
                print(f"Running onnx subgraphs repeat={repeat_id} batch={batch}")
                rows.extend(
                    run_onnxruntime(
                        specs,
                        batch,
                        args.seq_len,
                        args.warmup,
                        args.iters,
                        repeat_id,
                        args.shape_profile,
                        args.correctness_rtol,
                        args.correctness_atol,
                    )
                )

    output = Path(args.output)
    write_rows(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
