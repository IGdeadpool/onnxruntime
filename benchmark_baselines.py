import argparse
import csv
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from datasets import Dataset, DownloadConfig, load_dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_runtime import RuntimeConfig, detect_runtime


ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/home/l/benchmarks" if Path("/home/l").exists() else str(Path.home() / "benchmarks")))
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
ONNX_DIR = ROOT / "models" / "onnx"

CIFAR10_DIR = DATA_DIR / "cifar10"
HF_DATASETS_DIR = DATA_DIR / "huggingface_datasets"
TORCH_MODEL_DIR = MODEL_DIR / "torch"
HF_MODEL_DIR = MODEL_DIR / "huggingface"
DISTILBERT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
RUNTIME: RuntimeConfig = detect_runtime()


def setup_env() -> None:
    os.environ.setdefault("TORCH_HOME", str(TORCH_MODEL_DIR))
    os.environ.setdefault("HF_HOME", str(HF_MODEL_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_MODEL_DIR / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_DIR))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)


def load_hf_dataset_cached_first(*args, **kwargs):
    kwargs.setdefault("cache_dir", str(HF_DATASETS_DIR))
    try:
        return load_dataset(*args, download_config=DownloadConfig(local_files_only=True), **kwargs)
    except Exception as local_exc:
        print(f"HuggingFace dataset cache miss, falling back to download: {local_exc}")
        return load_dataset(*args, **kwargs)


def load_sst2_validation_cached_first() -> Dataset:
    cache_root = HF_DATASETS_DIR / "glue" / "sst2"
    arrow_files = sorted(cache_root.glob("**/glue-validation.arrow"), key=lambda p: p.stat().st_mtime, reverse=True)
    if arrow_files:
        return Dataset.from_file(str(arrow_files[0]))
    return load_hf_dataset_cached_first("glue", "sst2", split="validation")


def from_pretrained_cached_first(factory, model_id: str, **kwargs):
    kwargs.setdefault("cache_dir", str(HF_MODEL_DIR))
    try:
        return factory.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception as local_exc:
        print(f"HuggingFace model cache miss, falling back to download: {model_id}: {local_exc}")
        return factory.from_pretrained(model_id, **kwargs)


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def torch_backend_name() -> str:
    return RUNTIME.torch_backend


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def correctness_metrics(reference: np.ndarray, actual: np.ndarray, rtol: float = 1e-3, atol: float = 1e-4) -> dict[str, object]:
    ref = np.asarray(reference)
    got = np.asarray(actual)
    if ref.shape != got.shape:
        return {
            "correctness_status": "shape_mismatch",
            "max_abs_error": "",
            "max_rel_error": "",
            "correctness_message": f"reference_shape={ref.shape};actual_shape={got.shape}",
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


def benchmark_callable(fn: Callable[[], None], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync_device()

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync_device()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return {
        "latency_mean_ms": float(np.mean(times_ms)),
        "latency_p50_ms": percentile(times_ms, 50),
        "latency_p95_ms": percentile(times_ms, 95),
        "latency_min_ms": float(np.min(times_ms)),
        "latency_max_ms": float(np.max(times_ms)),
    }


def get_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def load_resnet18_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    dataset = datasets.CIFAR10(root=str(CIFAR10_DIR), train=False, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    images, labels = next(iter(loader))
    return images, labels


def run_resnet18_torch(batch_size: int, warmup: int, iters: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval().to(device)
    images, labels = load_resnet18_batch(batch_size)
    images = images.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        stats = benchmark_callable(lambda: model(images), warmup, iters)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

    labels_1k = labels
    # CIFAR-10 labels do not match ImageNet classes, so this is a smoke metric
    # only. Real accuracy requires fine-tuning ResNet18 on CIFAR-10.
    pseudo_acc = float((preds[: len(labels_1k)] == labels_1k).float().mean().item())

    return {
        **stats,
        "model": "resnet18",
        "dataset": "cifar10",
        "backend": torch_backend_name() if device.type == "cuda" else "torch_cpu",
        "batch_size": batch_size,
        "seq_len": "",
        "samples_per_sec": batch_size / (stats["latency_mean_ms"] / 1000.0),
        "metric_name": "pseudo_top1",
        "metric_value": pseudo_acc,
        "gpu_mem_mb": get_gpu_mem_mb(),
        "correctness_status": "reference",
        "max_abs_error": "",
        "max_rel_error": "",
        "correctness_message": "torch_eager_reference",
    }


def export_resnet18_onnx(batch_size: int) -> Path:
    path = ONNX_DIR / f"resnet18_bs{batch_size}.onnx"
    if path.exists() and path.stat().st_size > 0:
        return path

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()
    x = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )
    return path


def run_resnet18_onnx(batch_size: int, warmup: int, iters: int, correctness_rtol: float, correctness_atol: float) -> dict[str, object]:
    import onnxruntime as ort

    path = export_resnet18_onnx(batch_size)
    images, labels = load_resnet18_batch(batch_size)
    x = images.numpy().astype(np.float32)

    sess = ort.InferenceSession(str(path), providers=RUNTIME.onnx_providers)
    active = ",".join(sess.get_providers())

    stats = benchmark_callable(lambda: sess.run(None, {"input": x}), warmup, iters)
    logits = sess.run(None, {"input": x})[0]
    ref_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    with torch.no_grad():
        ref_logits = ref_model(images).detach().numpy()
    correctness = correctness_metrics(ref_logits, logits, rtol=correctness_rtol, atol=correctness_atol)
    preds = np.argmax(logits, axis=1)
    pseudo_acc = float(np.mean(preds[: len(labels)] == labels.numpy()))

    return {
        **stats,
        "model": "resnet18",
        "dataset": "cifar10",
        "backend": f"onnxruntime:{active}",
        "batch_size": batch_size,
        "seq_len": "",
        "samples_per_sec": batch_size / (stats["latency_mean_ms"] / 1000.0),
        "metric_name": "pseudo_top1",
        "metric_value": pseudo_acc,
        "gpu_mem_mb": "",
        "correctness_status": correctness["correctness_status"],
        "max_abs_error": correctness["max_abs_error"],
        "max_rel_error": correctness["max_rel_error"],
        "correctness_message": correctness["correctness_message"],
    }


def load_distilbert_batch(batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    dataset = load_sst2_validation_cached_first()
    texts = list(dataset.select(range(batch_size))["sentence"])
    tokenizer = from_pretrained_cached_first(AutoTokenizer, DISTILBERT_MODEL_ID)
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def run_distilbert_torch(batch_size: int, seq_len: int, warmup: int, iters: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = from_pretrained_cached_first(AutoModelForSequenceClassification, DISTILBERT_MODEL_ID).eval().to(device)
    batch = {k: v.to(device) for k, v in load_distilbert_batch(batch_size, seq_len).items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        stats = benchmark_callable(lambda: model(**batch), warmup, iters)
        logits = model(**batch).logits
        preds = logits.argmax(dim=1).detach().cpu().numpy()

    return {
        **stats,
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "dataset": "glue/sst2-validation",
        "backend": torch_backend_name() if device.type == "cuda" else "torch_cpu",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "samples_per_sec": batch_size / (stats["latency_mean_ms"] / 1000.0),
        "metric_name": "pred_positive_ratio",
        "metric_value": float(np.mean(preds)),
        "gpu_mem_mb": get_gpu_mem_mb(),
        "correctness_status": "reference",
        "max_abs_error": "",
        "max_rel_error": "",
        "correctness_message": "torch_eager_reference",
    }


def export_distilbert_onnx(batch_size: int, seq_len: int) -> Path:
    path = ONNX_DIR / f"distilbert_sst2_bs{batch_size}_seq{seq_len}.onnx"
    if path.exists() and path.stat().st_size > 0:
        return path

    model = from_pretrained_cached_first(AutoModelForSequenceClassification, DISTILBERT_MODEL_ID).eval()
    batch = load_distilbert_batch(batch_size, seq_len)
    torch.onnx.export(
        model,
        (batch["input_ids"], batch["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
    )
    return path


def run_distilbert_onnx(
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    correctness_rtol: float,
    correctness_atol: float,
) -> dict[str, object]:
    import onnxruntime as ort

    path = export_distilbert_onnx(batch_size, seq_len)
    batch = load_distilbert_batch(batch_size, seq_len)
    inputs = {
        "input_ids": batch["input_ids"].numpy().astype(np.int64),
        "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
    }
    sess = ort.InferenceSession(str(path), providers=RUNTIME.onnx_providers)
    active = ",".join(sess.get_providers())

    stats = benchmark_callable(lambda: sess.run(None, inputs), warmup, iters)
    logits = sess.run(None, inputs)[0]
    ref_model = from_pretrained_cached_first(AutoModelForSequenceClassification, DISTILBERT_MODEL_ID).eval()
    with torch.no_grad():
        ref_logits = ref_model(**batch).logits.detach().numpy()
    correctness = correctness_metrics(ref_logits, logits, rtol=correctness_rtol, atol=correctness_atol)
    preds = np.argmax(logits, axis=1)

    return {
        **stats,
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "dataset": "glue/sst2-validation",
        "backend": f"onnxruntime:{active}",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "samples_per_sec": batch_size / (stats["latency_mean_ms"] / 1000.0),
        "metric_name": "pred_positive_ratio",
        "metric_value": float(np.mean(preds)),
        "gpu_mem_mb": "",
        "correctness_status": correctness["correctness_status"],
        "max_abs_error": correctness["max_abs_error"],
        "max_rel_error": correctness["max_rel_error"],
        "correctness_message": correctness["correctness_message"],
    }


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def selected(items: Iterable[str], value: str) -> bool:
    return "all" in items or value in items


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "dataset",
        "backend",
        "batch_size",
        "seq_len",
        "samples_per_sec",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_min_ms",
        "latency_max_ms",
        "metric_name",
        "metric_value",
        "gpu_mem_mb",
        "correctness_status",
        "max_abs_error",
        "max_rel_error",
        "correctness_message",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResNet18 and DistilBERT baseline benchmarks.")
    parser.add_argument("--models", default="all", help="Comma list: all,resnet18,distilbert")
    parser.add_argument("--backends", default="all", help="Comma list: all,torch,onnx")
    parser.add_argument("--resnet-batches", default="1,8,16,32,64")
    parser.add_argument("--bert-batches", default="1,4,8,16,32")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "baseline_results.csv"))
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
    models = {x.strip() for x in args.models.split(",")}
    backends = {x.strip() for x in args.backends.split(",")}

    rows: list[dict[str, object]] = []

    for batch_size in parse_csv_ints(args.resnet_batches):
        if selected(models, "resnet18") and selected(backends, "torch"):
            print(f"Running ResNet18 torch batch={batch_size}")
            rows.append(run_resnet18_torch(batch_size, args.warmup, args.iters))
        if selected(models, "resnet18") and selected(backends, "onnx"):
            print(f"Running ResNet18 onnx batch={batch_size}")
            rows.append(run_resnet18_onnx(batch_size, args.warmup, args.iters, args.correctness_rtol, args.correctness_atol))

    for batch_size in parse_csv_ints(args.bert_batches):
        if selected(models, "distilbert") and selected(backends, "torch"):
            print(f"Running DistilBERT torch batch={batch_size} seq={args.seq_len}")
            rows.append(run_distilbert_torch(batch_size, args.seq_len, args.warmup, args.iters))
        if selected(models, "distilbert") and selected(backends, "onnx"):
            print(f"Running DistilBERT onnx batch={batch_size} seq={args.seq_len}")
            rows.append(
                run_distilbert_onnx(
                    batch_size,
                    args.seq_len,
                    args.warmup,
                    args.iters,
                    args.correctness_rtol,
                    args.correctness_atol,
                )
            )

    output = Path(args.output)
    write_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
