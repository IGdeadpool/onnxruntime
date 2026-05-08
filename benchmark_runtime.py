import os
import platform
import subprocess
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    device_label: str
    torch_backend: str
    onnx_backend: str
    onnx_providers: list[str]
    device_name: str


def _command_output(command: list[str]) -> str:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _torch_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def _detect_gpu_vendor(device_name: str) -> str:
    lower = device_name.lower()
    if any(token in lower for token in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")):
        return "nvidia"
    if any(token in lower for token in ("amd", "radeon", "instinct")):
        return "amd"

    rocminfo = _command_output(["bash", "-lc", "rocminfo 2>/dev/null | head -120"])
    if any(token in rocminfo.lower() for token in ("amd", "radeon", "gfx")):
        return "amd"

    nvidia_smi = _command_output(["bash", "-lc", "nvidia-smi -L 2>/dev/null | head -20"])
    if "nvidia" in nvidia_smi.lower() or "gpu " in nvidia_smi.lower():
        return "nvidia"

    return "cpu"


def _available_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def detect_runtime(
    onnx_backend: str = "auto",
    onnx_providers: str = "auto",
    device_label: str = "auto",
) -> RuntimeConfig:
    device_name = _torch_device_name()
    vendor = _detect_gpu_vendor(device_name)
    available = _available_ort_providers()

    if onnx_providers and onnx_providers != "auto":
        providers = _split_csv(onnx_providers)
        backend = onnx_backend if onnx_backend != "auto" else "custom"
    else:
        backend = onnx_backend
        if backend == "auto":
            backend = "migraphx" if vendor == "amd" else "cuda" if vendor == "nvidia" else "cpu"

        if backend == "migraphx":
            preferred = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        elif backend == "cuda":
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif backend == "cpu":
            preferred = ["CPUExecutionProvider"]
        else:
            preferred = ["CPUExecutionProvider"]

        providers = [p for p in preferred if not available or p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

    if vendor == "amd":
        torch_backend = "torch_rocm"
    elif vendor == "nvidia":
        torch_backend = "torch_cuda"
    else:
        torch_backend = "torch_cpu"

    if device_label == "auto":
        safe_name = device_name.lower().replace(" ", "_").replace("/", "_") if device_name else platform.machine()
        label_backend = backend if backend != "custom" else "ort"
        device_label = f"{safe_name}_{label_backend}"

    return RuntimeConfig(
        device_label=device_label,
        torch_backend=torch_backend,
        onnx_backend=backend,
        onnx_providers=providers,
        device_name=device_name,
    )


def runtime_env() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    return {key: os.environ.get(key, "") for key in keys}


if __name__ == "__main__":
    runtime = detect_runtime()
    print(
        json.dumps(
            {
                "device_label": runtime.device_label,
                "device_name": runtime.device_name,
                "torch_backend": runtime.torch_backend,
                "onnx_backend": runtime.onnx_backend,
                "onnx_providers": runtime.onnx_providers,
                "runtime_env": runtime_env(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
