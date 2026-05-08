# Codex CLI 在 RTX 3080 / CUDA 机器上的部署引导

本文档用于引导另一台机器上的 Codex CLI 复用本仓库，完成 NVIDIA RTX 3080 CUDA 体系的 benchmark 部署和测试，并与当前 AMD ROCm 体系结果对比。

## 1. 给另一个 Codex 的任务说明

可以直接把下面这段交给另一台机器上的 Codex CLI：

```text
请在当前机器部署并运行 https://github.com/IGdeadpool/onnxruntime.git 中的 benchmark 自动化流程。

目标机器是 NVIDIA RTX 3080，需要使用 CUDA + PyTorch + ONNX Runtime CUDAExecutionProvider。

要求：
1. 克隆仓库。
2. 在 WSL/Linux 中创建 /home/<user>/benchmarks 目录。
3. 将仓库脚本同步到 /home/<user>/benchmarks/scripts。
4. 创建 Python 虚拟环境。
5. 安装 CUDA 版 PyTorch、torchvision、transformers、datasets、onnx、onnxruntime-gpu 等依赖。
6. 运行 benchmark_runtime.py，确认自动识别为 torch_cuda + CUDAExecutionProvider。
7. 复制 benchmark_config.example.jsonc 为 benchmark_config.local.jsonc。
8. 将 device_label 固定为 rtx3080_cuda。
9. 先运行轻量 smoke test。
10. smoke test 成功后运行完整 benchmark。
11. 把 runs/<run_id> 目录中的 summary.md、metadata.json、baseline_results.csv、operator_results.csv、operator_pair_summary.csv、profile_summary.csv 保存下来。
12. 不要覆盖已有 run 目录。
```

## 2. 克隆仓库

```bash
cd ~
git clone https://github.com/IGdeadpool/onnxruntime.git onnxruntime-benchmark
```

## 3. 准备 benchmark 目录

```bash
mkdir -p ~/benchmarks/{scripts,data,models,outputs,runs}
cp ~/onnxruntime-benchmark/*.py ~/benchmarks/scripts/
cp ~/onnxruntime-benchmark/benchmark_config.example.jsonc ~/benchmarks/scripts/
```

## 4. 创建 Python 环境

示例：

```bash
python3 -m venv ~/torch-cuda/.venv
source ~/torch-cuda/.venv/bin/activate
python -m pip install --upgrade pip
```

安装依赖时，RTX 3080 机器应使用 CUDA 版 PyTorch。具体 CUDA 版本以目标机器驱动支持为准。

示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets evaluate scikit-learn pandas requests onnx onnxscript onnxruntime-gpu
```

如果目标机器已有 CUDA 版 PyTorch 环境，可以直接使用已有虚拟环境。

## 5. 检查设备和 provider

```bash
cd ~/benchmarks
source ~/torch-cuda/.venv/bin/activate

python ~/benchmarks/scripts/benchmark_runtime.py
```

期望输出类似：

```json
{
  "device_name": "NVIDIA GeForce RTX 3080",
  "torch_backend": "torch_cuda",
  "onnx_backend": "cuda",
  "onnx_providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

如果只看到 `CPUExecutionProvider`，说明 `onnxruntime-gpu` 未安装正确，或者 CUDA/cuDNN 运行库不可用。

## 6. 配置文件

```bash
cp ~/benchmarks/scripts/benchmark_config.example.jsonc \
   ~/benchmarks/scripts/benchmark_config.local.jsonc
```

编辑：

```bash
vim ~/benchmarks/scripts/benchmark_config.local.jsonc
```

建议 RTX 3080 固定：

```jsonc
"python": "/home/<user>/torch-cuda/.venv/bin/python",
"onnx_backend": "auto",
"onnx_providers": "auto",
"device_label": "rtx3080_cuda"
```

如需强制 CUDA provider：

```jsonc
"onnx_backend": "cuda",
"onnx_providers": "CUDAExecutionProvider,CPUExecutionProvider",
"device_label": "rtx3080_cuda"
```

## 7. 轻量 smoke test

```bash
cd ~/benchmarks
source ~/torch-cuda/.venv/bin/activate

python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc \
  --run-id rtx3080_cuda_smoke \
  --baseline-models resnet18 \
  --baseline-backends all \
  --resnet-batches 1 \
  --baseline-warmup 0 \
  --baseline-iters 1 \
  --operator-batches 1 \
  --shape-profile standard \
  --chain-len 2 \
  --repeat 1 \
  --operator-warmup 0 \
  --operator-iters 1 \
  --continue-on-error
```

检查：

```bash
cat ~/benchmarks/runs/rtx3080_cuda_smoke/steps_status.md
cat ~/benchmarks/runs/rtx3080_cuda_smoke/metadata.json
head ~/benchmarks/runs/rtx3080_cuda_smoke/operator_pair_summary.csv
```

## 8. 完整 benchmark

```bash
cd ~/benchmarks
source ~/torch-cuda/.venv/bin/activate

python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc
```

每次运行会自动创建时间戳目录：

```text
~/benchmarks/runs/<run_id>/
```

重点保存：

```text
summary.md
metadata.json
baseline_results.csv
operator_results.csv
operator_pair_summary.csv
profile_summary.csv
steps_status.md
logs/
```

## 9. 与 AMD ROCm 对比

将 RTX 3080 的 run 目录和 AMD ROCm 的 run 目录放到同一台机器后，可以比较：

```text
operator_pair_summary.csv
baseline_results.csv
profile_summary.csv
metadata.json
```

建议对比键：

```text
op_name
batch_size
shape_profile
torch_chain_len
ort_chain_len
device_label
torch_backend
onnx_providers
```

注意：

```text
AMD 的 ONNX provider 是 MIGraphXExecutionProvider。
NVIDIA 的 ONNX provider 是 CUDAExecutionProvider。
两者 runtime/compiler 行为不同，不能只看单个 latency，需要结合 provider、profile、p95 和 chain_len。
```
