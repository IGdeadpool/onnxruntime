# Windows Native + CUDA + PyTorch + ONNX Runtime 部署教程

适用于 Windows 11 + NVIDIA RTX 3080 环境，直接在 Windows 上运行 benchmark，无需 WSL。

已验证环境：

- OS: Windows 11 Pro (10.0.26200)
- GPU: NVIDIA GeForce RTX 3080 (10 GB)
- Driver: 591.86 / CUDA 13.1
- Python: 3.14.4 (`C:\Python314\`)
- PyTorch: 2.11.0+cu126
- ONNX Runtime: 1.25.1 (CUDAExecutionProvider)

## 1. 前置条件

### 1.1 确认 GPU 驱动

```powershell
nvidia-smi
```

应能看到 RTX 3080 和 CUDA 版本号。

### 1.2 安装 Python

使用独立 Python（非 Anaconda），从 https://www.python.org/ 下载安装 Python 3.x。

本机当前 Python 路径：`C:\Python314\python.exe`

## 2. 安装 Python 依赖

```powershell
# 安装 PyTorch CUDA（使用 CUDA 12.6，驱动 13.1 向前兼容）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 安装 ONNX Runtime GPU 和其他依赖
python -m pip install onnx onnxruntime-gpu transformers datasets evaluate scikit-learn pandas requests tqdm psutil onnxscript
```

### 2.1 镜像加速（国内网络可选）

```powershell
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 3. 验证安装

```powershell
python -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"
```

期望输出：

```text
torch 2.11.0+cu126
cuda_available True
device NVIDIA GeForce RTX 3080
```

```powershell
python -c "
import onnxruntime as ort
print('ort', ort.__version__)
print('providers', ort.get_available_providers())
"
```

期望输出中应包含 `CUDAExecutionProvider`。

## 4. 准备 benchmark 目录

```powershell
mkdir C:\Users\%USERNAME%\benchmarks\scripts
mkdir C:\Users\%USERNAME%\benchmarks\data
mkdir C:\Users\%USERNAME%\benchmarks\models
mkdir C:\Users\%USERNAME%\benchmarks\outputs
mkdir C:\Users\%USERNAME%\benchmarks\runs
```

将仓库脚本复制到 benchmark 目录：

```powershell
copy C:\Users\%USERNAME%\Desktop\onnxruntime\*.py C:\Users\%USERNAME%\benchmarks\scripts\
copy C:\Users\%USERNAME%\Desktop\onnxruntime\benchmark_config.windows.jsonc C:\Users\%USERNAME%\benchmarks\scripts\benchmark_config.local.jsonc
```

## 5. 创建本地配置

配置模板 `benchmark_config.windows.jsonc` 已提供，关键参数：

```jsonc
{
  "benchmark_root": "C:/Users/User/benchmarks",
  "script_dir": "C:/Users/User/benchmarks/scripts",
  "runs_dir": "C:/Users/User/benchmarks/runs",
  "python": "C:/Python314/python.exe",
  "onnx_backend": "auto",
  "onnx_providers": "auto",
  "device_label": "rtx3080_cuda",
  ...
}
```

## 6. 验证设备检测

```powershell
cd C:\Users\%USERNAME%\benchmarks
python scripts\benchmark_runtime.py
```

期望输出：

```json
{
  "device_label": "nvidia_geforce_rtx_3080_cuda",
  "device_name": "NVIDIA GeForce RTX 3080",
  "torch_backend": "torch_cuda",
  "onnx_backend": "cuda",
  "onnx_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

## 7. 下载模型和数据集

```powershell
python scripts\download_baselines.py
```

下载内容：
- ResNet18 ImageNet 权重 (~45 MB)
- CIFAR-10 训练/测试集 (~170 MB)
- GLUE/SST-2 数据集
- DistilBERT SST-2 模型和 tokenizer

## 8. 轻量 smoke test

```powershell
python scripts\run_full_benchmark.py `
  --config scripts\benchmark_config.local.jsonc `
  --run-id rtx3080_cuda_smoke `
  --baseline-models resnet18 `
  --baseline-backends all `
  --resnet-batches 1 `
  --baseline-warmup 1 `
  --baseline-iters 3 `
  --operator-batches 1 `
  --shape-profile standard `
  --chain-len 2 `
  --repeat 1 `
  --operator-warmup 1 `
  --operator-iters 3 `
  --continue-on-error
```

检查结果：

```powershell
type runs\rtx3080_cuda_smoke\steps_status.md
type runs\rtx3080_cuda_smoke\operator_pair_summary.csv
```

## 9. 完整 benchmark

```powershell
python scripts\run_full_benchmark.py `
  --config scripts\benchmark_config.local.jsonc
```

每次运行会创建独立时间戳目录：`runs/<run_id>/`

## 10. 运行目录结构

```text
C:\Users\User\benchmarks\
├── scripts/          # Python 脚本
│   └── benchmark_config.local.jsonc
├── data/             # 数据集缓存
│   ├── cifar10/
│   └── huggingface_datasets/
├── models/           # 模型文件
│   ├── torch/
│   ├── huggingface/
│   ├── onnx/
│   └── onnx_ops/
├── outputs/
│   └── ort_profiles/ # ORT profile JSON
└── runs/
    └── <run_id>/
        ├── 00_run_config.json
        ├── metadata.json
        ├── steps_status.md
        ├── baseline_results.csv
        ├── operator_results.csv
        ├── operator_pair_summary.csv
        ├── profile_summary.csv
        ├── regression_report.csv
        ├── summary.md
        ├── logs/
        │   ├── 01_environment.log
        │   ├── 02_baseline.log
        │   └── 03_operator.log
        └── ort_profiles/
            └── *.json
```

## 11. 与 AMD ROCm 结果对比

将本机 RTX 3080 的 run 目录与 AMD ROCm 的 run 目录对比：

```text
operator_pair_summary.csv
baseline_results.csv
profile_summary.csv
metadata.json
```

注意：
- RTX 3080 的 provider 是 `CUDAExecutionProvider`
- AMD 的 provider 是 `MIGraphXExecutionProvider`
- 两者 runtime/compiler 行为不同，需结合 profile、chain_len 综合评估

## 12. ORT Profile 可视化

直接运行 `ORT_Profile_Analyzer_WinForms.exe`，导入 `C:\Users\User\benchmarks\outputs\ort_profiles\*.json` 即可。
