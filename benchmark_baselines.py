import argparse
import csv
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_runtime import RuntimeConfig, detect_runtime


ROOT = Path("/home/l/benchmarks")
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


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def torch_backend_name() -> str:
    return RUNTIME.torch_backend


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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


def run_resnet18_onnx(batch_size: int, warmup: int, iters: int) -> dict[str, object]:
    import onnxruntime as ort

    path = export_resnet18_onnx(batch_size)
    images, labels = load_resnet18_batch(batch_size)
    x = images.numpy().astype(np.float32)

    sess = ort.InferenceSession(str(path), providers=RUNTIME.onnx_providers)
    active = ",".join(sess.get_providers())

    stats = benchmark_callable(lambda: sess.run(None, {"input": x}), warmup, iters)
    logits = sess.run(None, {"input": x})[0]
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
    }


def load_distilbert_batch(batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    dataset = load_dataset("glue", "sst2", split="validation", cache_dir=str(HF_DATASETS_DIR))
    texts = list(dataset.select(range(batch_size))["sentence"])
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_ID, cache_dir=str(HF_MODEL_DIR))
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
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID,
        cache_dir=str(HF_MODEL_DIR),
    ).eval().to(device)
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
    }


def export_distilbert_onnx(batch_size: int, seq_len: int) -> Path:
    path = ONNX_DIR / f"distilbert_sst2_bs{batch_size}_seq{seq_len}.onnx"
    if path.exists() and path.stat().st_size > 0:
        return path

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID,
        cache_dir=str(HF_MODEL_DIR),
    ).eval()
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


def run_distilbert_onnx(batch_size: int, seq_len: int, warmup: int, iters: int) -> dict[str, object]:
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
            rows.append(run_resnet18_onnx(batch_size, args.warmup, args.iters))

    for batch_size in parse_csv_ints(args.bert_batches):
        if selected(models, "distilbert") and selected(backends, "torch"):
            print(f"Running DistilBERT torch batch={batch_size} seq={args.seq_len}")
            rows.append(run_distilbert_torch(batch_size, args.seq_len, args.warmup, args.iters))
        if selected(models, "distilbert") and selected(backends, "onnx"):
            print(f"Running DistilBERT onnx batch={batch_size} seq={args.seq_len}")
            rows.append(run_distilbert_onnx(batch_size, args.seq_len, args.warmup, args.iters))

    output = Path(args.output)
    write_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
