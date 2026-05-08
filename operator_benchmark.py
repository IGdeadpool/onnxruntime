import argparse
import csv
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_runtime import RuntimeConfig, detect_runtime

ROOT = Path("/home/l/benchmarks")
OUTPUT_DIR = ROOT / "outputs"
ONNX_DIR = ROOT / "models" / "onnx_ops"
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


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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


def torch_backend_name() -> str:
    return RUNTIME.torch_backend


def tflops(flops: float, latency_ms: float) -> float:
    if flops <= 0 or latency_ms <= 0:
        return 0.0
    return flops / (latency_ms / 1000.0) / 1e12


def gbps(bytes_moved: float, latency_ms: float) -> float:
    if bytes_moved <= 0 or latency_ms <= 0:
        return 0.0
    return bytes_moved / (latency_ms / 1000.0) / 1e9


def tensor_bytes(*shapes: tuple[int, ...], dtype_bytes: int = 4) -> int:
    return sum(int(np.prod(shape)) * dtype_bytes for shape in shapes)


def correctness_metrics(reference: np.ndarray, actual: np.ndarray, rtol: float, atol: float) -> dict[str, object]:
    ref = np.asarray(reference)
    got = np.asarray(actual)
    if ref.shape != got.shape:
        return {
            "correctness_status": "shape_mismatch",
            "max_abs_error": "",
            "max_rel_error": "",
            "correctness_message": f"reference_shape={ref.shape};actual_shape={got.shape}",
        }
    if not np.issubdtype(ref.dtype, np.number) or not np.issubdtype(got.dtype, np.number):
        ok = bool(np.array_equal(ref, got))
        return {
            "correctness_status": "ok" if ok else "mismatch",
            "max_abs_error": 0.0 if ok else "",
            "max_rel_error": 0.0 if ok else "",
            "correctness_message": "exact_equal" if ok else "non_numeric_mismatch",
        }
    ref64 = ref.astype(np.float64)
    got64 = got.astype(np.float64)
    diff = np.abs(ref64 - got64)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(ref64), 1e-12)
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    ok = bool(np.allclose(ref64, got64, rtol=rtol, atol=atol, equal_nan=True))
    return {
        "correctness_status": "ok" if ok else "mismatch",
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "correctness_message": f"rtol={rtol};atol={atol}",
    }


def compare_outputs(reference_outputs: object, actual_outputs: list[np.ndarray], rtol: float, atol: float) -> dict[str, object]:
    if isinstance(reference_outputs, torch.Tensor):
        refs = [reference_outputs.detach().cpu().numpy()]
    elif isinstance(reference_outputs, (list, tuple)):
        refs = [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            for item in reference_outputs
        ]
    else:
        refs = [np.asarray(reference_outputs)]
    if len(refs) != len(actual_outputs):
        return {
            "correctness_status": "output_count_mismatch",
            "max_abs_error": "",
            "max_rel_error": "",
            "correctness_message": f"reference_outputs={len(refs)};actual_outputs={len(actual_outputs)}",
        }
    metrics = [correctness_metrics(ref, got, rtol, atol) for ref, got in zip(refs, actual_outputs)]
    bad = [m for m in metrics if m["correctness_status"] != "ok"]
    max_abs = max((float(m["max_abs_error"]) for m in metrics if m["max_abs_error"] != ""), default=0.0)
    max_rel = max((float(m["max_rel_error"]) for m in metrics if m["max_rel_error"] != ""), default=0.0)
    return {
        "correctness_status": "ok" if not bad else bad[0]["correctness_status"],
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "correctness_message": ";".join(str(m["correctness_message"]) for m in metrics),
    }


def row(
    *,
    op_name: str,
    backend: str,
    dtype: str,
    layout: str,
    input_shape: str,
    output_shape: str,
    attributes: str,
    batch_size: int,
    stats: dict[str, float],
    flops: float = 0.0,
    bytes_moved: float = 0.0,
    provider: str = "",
    graph_ops: str = "",
    repeat_id: int = 1,
    shape_profile: str = "standard",
    chain_len: int = 1,
    session_create_ms: float | str = "",
    first_run_ms: float | str = "",
    latency_per_op_mean_ms: float | str = "",
    profile_path: str = "",
    status: str = "ok",
    error_message: str = "",
    correctness_status: str = "",
    max_abs_error: float | str = "",
    max_rel_error: float | str = "",
    correctness_message: str = "",
) -> dict[str, object]:
    if latency_per_op_mean_ms == "":
        latency_per_op_mean_ms = stats.get("latency_mean_ms", 0.0) / max(chain_len, 1)
    return {
        "op_name": op_name,
        "backend": backend,
        "provider": provider,
        "repeat_id": repeat_id,
        "shape_profile": shape_profile,
        "chain_len": chain_len,
        "dtype": dtype,
        "layout": layout,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "graph_ops": graph_ops,
        "attributes": attributes,
        "batch_size": batch_size,
        "session_create_ms": session_create_ms,
        "first_run_ms": first_run_ms,
        "latency_mean_ms": stats.get("latency_mean_ms", 0.0),
        "latency_per_op_mean_ms": latency_per_op_mean_ms,
        "latency_p50_ms": stats.get("latency_p50_ms", 0.0),
        "latency_p95_ms": stats.get("latency_p95_ms", 0.0),
        "latency_min_ms": stats.get("latency_min_ms", 0.0),
        "latency_max_ms": stats.get("latency_max_ms", 0.0),
        "tflops": tflops(flops, stats.get("latency_mean_ms", 0.0)),
        "bandwidth_gb_s": gbps(bytes_moved, stats.get("latency_mean_ms", 0.0)),
        "gpu_mem_mb": mem_mb() if backend in ("torch_rocm", "torch_cuda") else "",
        "correctness_status": correctness_status,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "correctness_message": correctness_message,
        "profile_path": profile_path,
        "status": status,
        "error_message": error_message,
    }


class AddModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MatMulModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


class GeluModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class SoftmaxModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class EmbeddingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(30522, 768)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class ChainModule(nn.Module):
    def __init__(self, module: nn.Module, chain_len: int) -> None:
        super().__init__()
        self.ops = nn.ModuleList([module if i == 0 else self._clone_module(module) for i in range(chain_len)])

    def _clone_module(self, module: nn.Module) -> nn.Module:
        import copy

        return copy.deepcopy(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x


class AddChainModule(nn.Module):
    def __init__(self, chain_len: int) -> None:
        super().__init__()
        self.chain_len = chain_len

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for _ in range(self.chain_len):
            x = x + y
        return x


class MatMulChainModule(nn.Module):
    def __init__(self, chain_len: int) -> None:
        super().__init__()
        self.chain_len = chain_len

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for _ in range(self.chain_len):
            x = torch.matmul(x, y)
        return x


def export_onnx_once(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    names: list[str],
    use_cache: bool = True,
) -> Path:
    if use_cache and path.exists() and path.stat().st_size > 0:
        return path
    module.eval().cpu()
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=names,
        output_names=["output"],
        opset_version=18,
    )
    return path


def onnx_graph_ops(path: Path) -> str:
    import onnx

    model = onnx.load(str(path))
    ops = [node.op_type for node in model.graph.node]
    return ";".join(ops)


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
    sess = ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=providers,
    )
    session_create_ms = (time.perf_counter() - start) * 1000.0
    provider = ",".join(sess.get_providers())
    first_run_ms = time_once(lambda: sess.run(None, feed))
    stats = bench(lambda: sess.run(None, feed), warmup, iters)
    outputs = sess.run(None, feed)
    output_shape = ";".join(str(tuple(out.shape)) for out in outputs)
    profile_path = sess.end_profiling()
    return stats, provider, output_shape, session_create_ms, first_run_ms, profile_path, outputs


def conv_cases(shape_profile: str) -> list[tuple[int, int, int, int, int, int, int]]:
    if shape_profile == "large":
        return [
            (64, 128, 112, 112, 3, 1, 1),
            (128, 256, 56, 56, 3, 1, 1),
            (256, 256, 56, 56, 1, 1, 0),
            (256, 512, 28, 28, 3, 1, 1),
        ]
    return [
        (3, 64, 224, 224, 7, 2, 3),
        (64, 64, 56, 56, 3, 1, 1),
        (64, 128, 56, 56, 3, 2, 1),
        (256, 64, 56, 56, 1, 1, 0),
    ]


def torch_conv2d(batch: int, warmup: int, iters: int, repeat_id: int, shape_profile: str) -> list[dict[str, object]]:
    dev = device()
    rows = []
    for cin, cout, h, w, k, stride, pad in conv_cases(shape_profile):
        x = torch.randn(batch, cin, h, w, device=dev)
        op = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False).eval().to(dev)
        y = op(x)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bench(lambda: op(x), warmup, iters)
        oh, ow = int(y.shape[2]), int(y.shape[3])
        flops = 2.0 * batch * cout * oh * ow * cin * k * k
        bytes_moved = tensor_bytes(tuple(x.shape), tuple(y.shape), tuple(op.weight.shape))
        rows.append(
            row(
                op_name="conv2d",
                backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
                dtype="fp32",
                layout="nchw",
                input_shape=str(tuple(x.shape)),
                output_shape=str(tuple(y.shape)),
                attributes=f"cin={cin};cout={cout};k={k};stride={stride};pad={pad}",
                batch_size=batch,
                repeat_id=repeat_id,
                shape_profile=shape_profile,
                stats=stats,
                flops=flops,
                bytes_moved=bytes_moved,
                correctness_status="reference",
                correctness_message="torch_eager_reference",
            )
        )
    return rows


def torch_elementwise(batch: int, warmup: int, iters: int, repeat_id: int, shape_profile: str) -> list[dict[str, object]]:
    dev = device()
    rows = []
    shape = (batch, 512, 112, 112) if shape_profile == "large" else (batch, 256, 56, 56)

    x = torch.randn(*shape, device=dev)
    y = torch.randn(*shape, device=dev)
    ops: list[tuple[str, Callable[[], torch.Tensor], str, float]] = [
        ("relu", lambda: F.relu(x), "dim=all", tensor_bytes(shape, shape)),
        ("add", lambda: x + y, "broadcast=false", tensor_bytes(shape, shape, shape)),
        ("gelu", lambda: F.gelu(x), "approximate=none", tensor_bytes(shape, shape)),
    ]
    for name, fn, attrs, bytes_moved in ops:
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bench(fn, warmup, iters)
        rows.append(
            row(
                op_name=name,
                backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
                dtype="fp32",
                layout="nchw" if len(shape) == 4 else "nd",
                input_shape=str(shape),
                output_shape=str(tuple(out.shape)),
                attributes=attrs,
                batch_size=batch,
                repeat_id=repeat_id,
                shape_profile=shape_profile,
                stats=stats,
                bytes_moved=bytes_moved,
            )
        )
    return rows


def torch_pool_norm(batch: int, warmup: int, iters: int, repeat_id: int, shape_profile: str) -> list[dict[str, object]]:
    dev = device()
    rows = []
    pool_shape = (batch, 128, 112, 112) if shape_profile == "large" else (batch, 64, 56, 56)
    x = torch.randn(*pool_shape, device=dev)
    modules: list[tuple[str, nn.Module, str]] = [
        ("batchnorm2d", nn.BatchNorm2d(pool_shape[1]).eval().to(dev), f"channels={pool_shape[1]}"),
        ("maxpool2d", nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval().to(dev), "k=3;stride=2;pad=1"),
        ("avgpool2d", nn.AvgPool2d(kernel_size=7, stride=1).eval().to(dev), "k=7;stride=1"),
    ]
    for name, mod, attrs in modules:
        y = mod(x)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bench(lambda mod=mod: mod(x), warmup, iters)
        rows.append(
            row(
                op_name=name,
                backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
                dtype="fp32",
                layout="nchw",
                input_shape=str(tuple(x.shape)),
                output_shape=str(tuple(y.shape)),
                attributes=attrs,
                batch_size=batch,
                repeat_id=repeat_id,
                shape_profile=shape_profile,
                stats=stats,
                bytes_moved=tensor_bytes(tuple(x.shape), tuple(y.shape)),
            )
        )
    return rows


def torch_linear_matmul(batch: int, warmup: int, iters: int, repeat_id: int, shape_profile: str) -> list[dict[str, object]]:
    dev = device()
    rows = []

    in_features, out_features = (4096, 4096) if shape_profile == "large" else (2048, 1000)
    x = torch.randn(batch, in_features, device=dev)
    linear = nn.Linear(in_features, out_features, bias=True).eval().to(dev)
    y = linear(x)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = bench(lambda: linear(x), warmup, iters)
    rows.append(
        row(
            op_name="linear",
            backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
            dtype="fp32",
            layout="nc",
            input_shape=str(tuple(x.shape)),
            output_shape=str(tuple(y.shape)),
            attributes=f"in={in_features};out={out_features};bias=true",
            batch_size=batch,
            repeat_id=repeat_id,
            shape_profile=shape_profile,
            stats=stats,
            flops=2.0 * batch * in_features * out_features,
            bytes_moved=tensor_bytes(tuple(x.shape), tuple(y.shape), tuple(linear.weight.shape)),
        )
    )

    m, k, n = (256, 1024, 1024) if shape_profile == "large" else (128, 768, 768)
    a = torch.randn(batch, m, k, device=dev)
    b = torch.randn(batch, k, n, device=dev)
    c = torch.bmm(a, b)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = bench(lambda: torch.bmm(a, b), warmup, iters)
    rows.append(
        row(
            op_name="batch_matmul",
            backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
            dtype="fp32",
            layout="bmn_bnk",
            input_shape=f"{tuple(a.shape)};{tuple(b.shape)}",
            output_shape=str(tuple(c.shape)),
            attributes=f"m={m};k={k};n={n}",
            batch_size=batch,
            repeat_id=repeat_id,
            shape_profile=shape_profile,
            stats=stats,
            flops=2.0 * batch * m * k * n,
            bytes_moved=tensor_bytes(tuple(a.shape), tuple(b.shape), tuple(c.shape)),
        )
    )
    return rows


def torch_transformer_ops(batch: int, seq_len: int, warmup: int, iters: int, repeat_id: int, shape_profile: str) -> list[dict[str, object]]:
    dev = device()
    rows = []
    hidden = 1024 if shape_profile == "large" else 768
    x = torch.randn(batch, seq_len, hidden, device=dev)

    ln = nn.LayerNorm(hidden).eval().to(dev)
    emb = nn.Embedding(30522, hidden).eval().to(dev)
    ids = torch.randint(0, 30522, (batch, seq_len), device=dev)

    ops: list[tuple[str, Callable[[], torch.Tensor], str, str, str, float]] = [
        ("layernorm", lambda: ln(x), str(tuple(x.shape)), f"hidden={hidden}", "bsh", tensor_bytes(tuple(x.shape), tuple(x.shape))),
        ("softmax", lambda: F.softmax(x, dim=-1), str(tuple(x.shape)), "dim=-1", "bsh", tensor_bytes(tuple(x.shape), tuple(x.shape))),
        ("embedding", lambda: emb(ids), str(tuple(ids.shape)), f"vocab=30522;hidden={hidden}", "bs", tensor_bytes(tuple(ids.shape), (batch, seq_len, hidden))),
    ]

    for name, fn, input_shape, attrs, layout, bytes_moved in ops:
        y = fn()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bench(fn, warmup, iters)
        rows.append(
            row(
                op_name=name,
                backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
                dtype="fp32" if name != "embedding" else "int64/fp32",
                layout=layout,
                input_shape=input_shape,
                output_shape=str(tuple(y.shape)),
                attributes=attrs,
                batch_size=batch,
                repeat_id=repeat_id,
                shape_profile=shape_profile,
                stats=stats,
                bytes_moved=bytes_moved,
            )
        )
    return rows


def torch_basic_ops(
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    repeat_id: int,
    shape_profile: str,
    chain_len: int,
) -> list[dict[str, object]]:
    dev = device()
    rows = []
    specs: list[tuple[str, nn.Module, tuple[torch.Tensor, ...], str, str, float, float, int]] = []

    elem_shape = (batch, 512, 112, 112) if shape_profile == "large" else (batch, 64, 56, 56)
    x4 = torch.randn(*elem_shape, device=dev)
    specs.append(
        (
            "relu",
            ChainModule(nn.ReLU().eval(), chain_len).eval().to(dev),
            (x4,),
            "nchw",
            f"shape={elem_shape}",
            0.0,
            tensor_bytes(tuple(x4.shape), tuple(x4.shape)) * chain_len,
            chain_len,
        )
    )

    add_shape = (batch, 512, 112, 112) if shape_profile == "large" else (batch, 256, 56, 56)
    a4 = torch.randn(*add_shape, device=dev)
    b4 = torch.randn(*add_shape, device=dev)
    specs.append(
        (
            "add",
            AddChainModule(chain_len).eval().to(dev),
            (a4, b4),
            "nchw",
            f"shape={add_shape}",
            0.0,
            tensor_bytes(tuple(a4.shape), tuple(b4.shape), tuple(a4.shape)) * chain_len,
            chain_len,
        )
    )

    hidden = 1024 if shape_profile == "large" else 768
    gelu_x = torch.randn(batch, seq_len, hidden, device=dev)
    specs.append(
        (
            "gelu",
            ChainModule(GeluModule().eval(), chain_len).eval().to(dev),
            (gelu_x,),
            "bsh",
            f"shape=(N,{seq_len},{hidden})",
            0.0,
            tensor_bytes(tuple(gelu_x.shape), tuple(gelu_x.shape)) * chain_len,
            chain_len,
        )
    )
    specs.append(
        (
            "softmax",
            ChainModule(SoftmaxModule().eval(), chain_len).eval().to(dev),
            (gelu_x,),
            "bsh",
            f"shape=(N,{seq_len},{hidden});dim=-1",
            0.0,
            tensor_bytes(tuple(gelu_x.shape), tuple(gelu_x.shape)) * chain_len,
            chain_len,
        )
    )

    m, k, n = (256, 1024, 1024) if shape_profile == "large" else (seq_len, 768, 768)
    mm_a = torch.randn(batch, m, k, device=dev)
    mm_b = torch.randn(batch, k, n, device=dev)
    specs.append(
        (
            "batch_matmul",
            MatMulChainModule(chain_len).eval().to(dev),
            (mm_a, mm_b),
            "bmn_bnk",
            f"m={m};k={k};n={n}",
            2.0 * batch * m * k * n * chain_len,
            tensor_bytes(tuple(mm_a.shape), tuple(mm_b.shape), (batch, m, n)) * chain_len,
            chain_len,
        )
    )

    cin, cout, h, w, ksize, stride, pad = (
        (256, 256, 56, 56, 3, 1, 1)
        if shape_profile == "large"
        else (64, 64, 56, 56, 3, 1, 1)
    )
    conv = ChainModule(nn.Conv2d(cin, cout, kernel_size=ksize, stride=stride, padding=pad, bias=False).eval(), chain_len).eval().to(dev)
    conv_x = torch.randn(batch, cin, h, w, device=dev)
    conv_y = conv(conv_x)
    specs.append(
        (
            "conv2d",
            conv,
            (conv_x,),
            "nchw",
            f"cin={cin};cout={cout};k={ksize};stride={stride};pad={pad}",
            2.0 * batch * cout * int(conv_y.shape[2]) * int(conv_y.shape[3]) * cin * ksize * ksize * chain_len,
            tensor_bytes(tuple(conv_x.shape), tuple(conv_y.shape)) * chain_len,
            chain_len,
        )
    )

    bn_channels = 128 if shape_profile == "large" else 64
    bn_hw = 112 if shape_profile == "large" else 56
    bn = ChainModule(nn.BatchNorm2d(bn_channels).eval(), chain_len).eval().to(dev)
    bn_x = torch.randn(batch, bn_channels, bn_hw, bn_hw, device=dev)
    specs.append(
        (
            "batchnorm2d",
            bn,
            (bn_x,),
            "nchw",
            f"channels={bn_channels}",
            0.0,
            tensor_bytes(tuple(bn_x.shape), tuple(bn_x.shape)) * chain_len,
            chain_len,
        )
    )

    pool_x = torch.randn(batch, bn_channels, bn_hw, bn_hw, device=dev)
    specs.append(
        (
            "maxpool2d",
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval().to(dev),
            (pool_x,),
            "nchw",
            "k=3;stride=2;pad=1;effective_chain=1",
            0.0,
            tensor_bytes(tuple(pool_x.shape), (batch, bn_channels, bn_hw // 2, bn_hw // 2)),
            1,
        )
    )
    avg_x = torch.randn(batch, bn_channels, bn_hw, bn_hw, device=dev)
    specs.append(
        (
            "avgpool2d",
            nn.AvgPool2d(kernel_size=7, stride=1).eval().to(dev),
            (avg_x,),
            "nchw",
            "k=7;stride=1;effective_chain=1",
            0.0,
            tensor_bytes(tuple(avg_x.shape), (batch, bn_channels, bn_hw - 6, bn_hw - 6)),
            1,
        )
    )

    in_features, out_features = (4096, 4096) if shape_profile == "large" else (2048, 2048)
    linear = ChainModule(nn.Linear(in_features, out_features, bias=True).eval(), chain_len).eval().to(dev)
    linear_x = torch.randn(batch, in_features, device=dev)
    specs.append(
        (
            "linear",
            linear,
            (linear_x,),
            "nc",
            f"in={in_features};out={out_features};bias=true",
            2.0 * batch * in_features * out_features * chain_len,
            tensor_bytes(tuple(linear_x.shape), (batch, out_features)) * chain_len,
            chain_len,
        )
    )

    layernorm = ChainModule(nn.LayerNorm(hidden).eval(), chain_len).eval().to(dev)
    ln_x = torch.randn(batch, seq_len, hidden, device=dev)
    specs.append(
        (
            "layernorm",
            layernorm,
            (ln_x,),
            "bsh",
            f"hidden={hidden}",
            0.0,
            tensor_bytes(tuple(ln_x.shape), tuple(ln_x.shape)) * chain_len,
            chain_len,
        )
    )

    emb = EmbeddingModule().eval().to(dev)
    ids = torch.randint(0, 30522, (batch, seq_len), dtype=torch.long, device=dev)
    specs.append(
        (
            "embedding",
            emb,
            (ids,),
            "bs",
            "vocab=30522;hidden=768;effective_chain=1",
            0.0,
            tensor_bytes(tuple(ids.shape), (batch, seq_len, 768)),
            1,
        )
    )

    for name, module, inputs, layout, attrs, flops, bytes_moved, effective_chain_len in specs:
        y = module(*inputs)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bench(lambda module=module, inputs=inputs: module(*inputs), warmup, iters)
        rows.append(
            row(
                op_name=name,
                backend=torch_backend_name() if dev.type == "cuda" else "torch_cpu",
                dtype="int64/fp32" if any(t.dtype == torch.long for t in inputs) else "fp32",
                layout=layout,
                input_shape=";".join(str(tuple(t.shape)) for t in inputs),
                output_shape=str(tuple(y.shape)),
                attributes=attrs,
                batch_size=batch,
                repeat_id=repeat_id,
                shape_profile=shape_profile,
                chain_len=effective_chain_len,
                stats=stats,
                flops=flops,
                bytes_moved=bytes_moved,
                correctness_status="reference",
                correctness_message="torch_eager_reference",
            )
        )
    return rows


def onnx_basic_ops(
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    repeat_id: int,
    shape_profile: str,
    chain_len: int,
    correctness_rtol: float,
    correctness_atol: float,
) -> list[dict[str, object]]:
    rows = []
    specs: list[tuple[str, nn.Module, tuple[torch.Tensor, ...], list[str], str, str, float, float, int]] = []

    elem_shape = (batch, 512, 112, 112) if shape_profile == "large" else (batch, 64, 56, 56)
    x4 = torch.randn(*elem_shape)
    specs.append(("relu", ChainModule(nn.ReLU().eval(), chain_len).eval(), (x4,), ["input"], "nchw", f"shape={elem_shape}", 0.0, tensor_bytes(tuple(x4.shape), tuple(x4.shape)) * chain_len, chain_len))

    add_shape = (batch, 512, 112, 112) if shape_profile == "large" else (batch, 256, 56, 56)
    a4 = torch.randn(*add_shape)
    b4 = torch.randn(*add_shape)
    specs.append(("add", AddChainModule(chain_len).eval(), (a4, b4), ["x", "y"], "nchw", f"shape={add_shape}", 0.0, tensor_bytes(tuple(a4.shape), tuple(b4.shape), tuple(a4.shape)) * chain_len, chain_len))

    hidden = 1024 if shape_profile == "large" else 768
    gelu_x = torch.randn(batch, seq_len, hidden)
    specs.append(("gelu", ChainModule(GeluModule().eval(), chain_len).eval(), (gelu_x,), ["input"], "bsh", f"shape=(N,{seq_len},{hidden})", 0.0, tensor_bytes(tuple(gelu_x.shape), tuple(gelu_x.shape)) * chain_len, chain_len))
    specs.append(("softmax", ChainModule(SoftmaxModule().eval(), chain_len).eval(), (gelu_x,), ["input"], "bsh", f"shape=(N,{seq_len},{hidden});dim=-1", 0.0, tensor_bytes(tuple(gelu_x.shape), tuple(gelu_x.shape)) * chain_len, chain_len))

    m, k, n = (256, 1024, 1024) if shape_profile == "large" else (seq_len, 768, 768)
    mm_a = torch.randn(batch, m, k)
    mm_b = torch.randn(batch, k, n)
    specs.append(
        (
            "batch_matmul",
            MatMulChainModule(chain_len).eval(),
            (mm_a, mm_b),
            ["x", "y"],
            "bmn_bnk",
            f"m={m};k={k};n={n}",
            2.0 * batch * m * k * n * chain_len,
            tensor_bytes(tuple(mm_a.shape), tuple(mm_b.shape), (batch, m, n)) * chain_len,
            chain_len,
        )
    )

    # ONNX chain mode requires the output shape of one Conv to be a valid input
    # for the next Conv. Use square channel shapes here instead of reusing every
    # ResNet-style case, where some Conv layers intentionally change channels.
    cin, cout, h, w, ksize, stride, pad = (
        (256, 256, 56, 56, 3, 1, 1)
        if shape_profile == "large"
        else (64, 64, 56, 56, 3, 1, 1)
    )
    conv = ChainModule(nn.Conv2d(cin, cout, kernel_size=ksize, stride=stride, padding=pad, bias=False).eval(), chain_len).eval()
    conv_x = torch.randn(batch, cin, h, w)
    conv_out_h = (h + 2 * pad - ksize) // stride + 1
    conv_out_w = (w + 2 * pad - ksize) // stride + 1
    specs.append(
        (
            "conv2d",
            conv,
            (conv_x,),
            ["input"],
            "nchw",
            f"cin={cin};cout={cout};k={ksize};stride={stride};pad={pad}",
            2.0 * batch * cout * conv_out_h * conv_out_w * cin * ksize * ksize * chain_len,
            tensor_bytes(tuple(conv_x.shape), (batch, cout, conv_out_h, conv_out_w)) * chain_len,
            chain_len,
        )
    )

    bn_channels = 128 if shape_profile == "large" else 64
    bn_hw = 112 if shape_profile == "large" else 56
    bn = ChainModule(nn.BatchNorm2d(bn_channels).eval(), chain_len).eval()
    bn_x = torch.randn(batch, bn_channels, bn_hw, bn_hw)
    specs.append(("batchnorm2d", bn, (bn_x,), ["input"], "nchw", f"channels={bn_channels}", 0.0, tensor_bytes(tuple(bn_x.shape), tuple(bn_x.shape)) * chain_len, chain_len))

    maxpool = ChainModule(nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval(), 1).eval()
    pool_x = torch.randn(batch, bn_channels, bn_hw, bn_hw)
    specs.append(("maxpool2d", maxpool, (pool_x,), ["input"], "nchw", "k=3;stride=2;pad=1;effective_chain=1", 0.0, tensor_bytes(tuple(pool_x.shape), (batch, bn_channels, bn_hw // 2, bn_hw // 2)), 1))

    avgpool = ChainModule(nn.AvgPool2d(kernel_size=7, stride=1).eval(), 1).eval()
    avg_x = torch.randn(batch, bn_channels, bn_hw, bn_hw)
    specs.append(("avgpool2d", avgpool, (avg_x,), ["input"], "nchw", "k=7;stride=1;effective_chain=1", 0.0, tensor_bytes(tuple(avg_x.shape), (batch, bn_channels, bn_hw - 6, bn_hw - 6)), 1))

    in_features, out_features = (4096, 4096) if shape_profile == "large" else (2048, 2048)
    linear = ChainModule(nn.Linear(in_features, out_features, bias=True).eval(), chain_len).eval()
    linear_x = torch.randn(batch, in_features)
    specs.append(("linear", linear, (linear_x,), ["input"], "nc", f"in={in_features};out={out_features};bias=true", 2.0 * batch * in_features * out_features * chain_len, tensor_bytes(tuple(linear_x.shape), (batch, out_features)) * chain_len, chain_len))

    layernorm = ChainModule(nn.LayerNorm(hidden).eval(), chain_len).eval()
    ln_x = torch.randn(batch, seq_len, hidden)
    specs.append(("layernorm", layernorm, (ln_x,), ["input"], "bsh", f"hidden={hidden}", 0.0, tensor_bytes(tuple(ln_x.shape), tuple(ln_x.shape)) * chain_len, chain_len))

    emb = EmbeddingModule().eval()
    ids = torch.randint(0, 30522, (batch, seq_len), dtype=torch.long)
    specs.append(("embedding", emb, (ids,), ["input_ids"], "bs", "vocab=30522;hidden=768;effective_chain=1", 0.0, tensor_bytes(tuple(ids.shape), (batch, seq_len, 768)), 1))

    for name, module, inputs, input_names, layout, attrs, flops, bytes_moved, effective_chain_len in specs:
        path = ONNX_DIR / f"op_{name}_{shape_profile}_chain{effective_chain_len}_bs{batch}_seq{seq_len}.onnx"
        try:
            # Operator modules can contain randomly initialized parameters.
            # Re-export to keep ONNX weights aligned with the PyTorch reference
            # used by correctness validation.
            export_onnx_once(module, inputs, path, input_names, use_cache=False)
            feed = {
                n: t.numpy().astype(np.int64 if t.dtype == torch.long else np.float32)
                for n, t in zip(input_names, inputs)
            }
            profile_name = f"ort_{name}_{shape_profile}_chain{effective_chain_len}_bs{batch}_rep{repeat_id}"
            stats, provider, output_shape, session_create_ms, first_run_ms, profile_path, ort_outputs = run_onnx(
                path,
                feed,
                warmup,
                iters,
                profile_name,
                RUNTIME.onnx_providers,
            )
            graph_ops = onnx_graph_ops(path)
            with torch.no_grad():
                reference_outputs = module(*inputs)
            correctness = compare_outputs(reference_outputs, ort_outputs, rtol=correctness_rtol, atol=correctness_atol)
            rows.append(
                row(
                    op_name=name,
                    backend="onnxruntime",
                    provider=provider,
                    dtype="int64/fp32" if any(t.dtype == torch.long for t in inputs) else "fp32",
                    layout=layout,
                    input_shape=";".join(str(tuple(t.shape)) for t in inputs),
                    output_shape=output_shape,
                    graph_ops=graph_ops,
                    attributes=attrs,
                    batch_size=batch,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    chain_len=effective_chain_len,
                    session_create_ms=session_create_ms,
                    first_run_ms=first_run_ms,
                    profile_path=profile_path,
                    stats=stats,
                    flops=flops,
                    bytes_moved=bytes_moved,
                    correctness_status=str(correctness["correctness_status"]),
                    max_abs_error=correctness["max_abs_error"],
                    max_rel_error=correctness["max_rel_error"],
                    correctness_message=str(correctness["correctness_message"]),
                )
            )
        except Exception as exc:
            rows.append(
                row(
                    op_name=name,
                    backend="onnxruntime",
                    provider="",
                    dtype="fp32",
                    layout=layout,
                    input_shape=";".join(str(tuple(t.shape)) for t in inputs),
                    output_shape="",
                    attributes=attrs,
                    batch_size=batch,
                    repeat_id=repeat_id,
                    shape_profile=shape_profile,
                    chain_len=effective_chain_len,
                    stats={},
                    status="error",
                    error_message=str(exc).replace("\n", " ")[:500],
                )
            )
    return rows


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "op_name",
        "backend",
        "provider",
        "repeat_id",
        "shape_profile",
        "chain_len",
        "dtype",
        "layout",
        "input_shape",
        "output_shape",
        "graph_ops",
        "attributes",
        "batch_size",
        "session_create_ms",
        "first_run_ms",
        "latency_mean_ms",
        "latency_per_op_mean_ms",
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
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator-level benchmark for PyTorch and ONNX Runtime providers.")
    parser.add_argument("--backends", default="all", help="all,torch,onnx")
    parser.add_argument("--batches", default="1,8,16,32")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shape-profile", choices=["standard", "large"], default="standard")
    parser.add_argument("--chain-len", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "operator_results.csv"))
    parser.add_argument("--onnx-backend", default="auto", help="auto,migraphx,cuda,cpu,custom")
    parser.add_argument("--onnx-providers", default="auto", help="auto or comma list such as CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--device-label", default="auto", help="auto or a stable label such as rx9070xt_rocm/rtx3080_cuda")
    parser.add_argument("--correctness-rtol", type=float, default=1e-3)
    parser.add_argument("--correctness-atol", type=float, default=1e-4)
    args = parser.parse_args()

    global RUNTIME
    RUNTIME = detect_runtime(args.onnx_backend, args.onnx_providers, args.device_label)
    print(
        "Runtime "
        f"device_label={RUNTIME.device_label} "
        f"device_name={RUNTIME.device_name} "
        f"torch_backend={RUNTIME.torch_backend} "
        f"onnx_backend={RUNTIME.onnx_backend} "
        f"onnx_providers={','.join(RUNTIME.onnx_providers)}"
    )

    setup_env()
    backends = {x.strip() for x in args.backends.split(",")}
    run_torch = "all" in backends or "torch" in backends
    run_onnx = "all" in backends or "onnx" in backends

    rows: list[dict[str, object]] = []
    for repeat_id in range(1, args.repeat + 1):
        for batch in parse_ints(args.batches):
            if run_torch:
                print(f"Running torch ops repeat={repeat_id} batch={batch}")
                rows.extend(torch_basic_ops(batch, args.seq_len, args.warmup, args.iters, repeat_id, args.shape_profile, args.chain_len))
            if run_onnx:
                print(f"Running onnx ops repeat={repeat_id} batch={batch}")
                rows.extend(
                    onnx_basic_ops(
                        batch,
                        args.seq_len,
                        args.warmup,
                        args.iters,
                        repeat_id,
                        args.shape_profile,
                        args.chain_len,
                        args.correctness_rtol,
                        args.correctness_atol,
                    )
                )

    output = Path(args.output)
    write_rows(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
