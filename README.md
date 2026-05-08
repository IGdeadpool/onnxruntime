# ONNX Runtime Provider Benchmark Toolkit

本仓库用于归档 ONNX Runtime provider benchmark 工具链。当前已在 WSL2 + AMD ROCm + PyTorch + ONNX Runtime MIGraphX 环境验证，同时已加入 NVIDIA CUDAExecutionProvider 自动识别支持，便于在 RTX 3080 机器上做双体系对比。

## 当前进度

已完成：

- 配置文件驱动的一键 benchmark 流程。
- 每次运行自动创建独立时间戳 run 目录，不覆盖旧结果。
- 每一步输出 `steps_status.md`、日志、CSV、profile summary 和最终 `summary.md`。
- 自动设备识别：
  - AMD ROCm -> `torch_rocm` + `MIGraphXExecutionProvider`
  - NVIDIA CUDA -> `torch_cuda` + `CUDAExecutionProvider`
  - CPU only -> `torch_cpu` + `CPUExecutionProvider`
- Torch 与 ONNX Runtime 算子 benchmark 使用同一套有效 chain 语义。
- ROCm 当前环境 smoke test 已通过，输出目录：

```text
/home/l/benchmarks/runs/rocm_auto_provider_smoke
```

待在另一台 RTX 3080 机器验证：

- CUDA 版 PyTorch 环境。
- `onnxruntime-gpu` 的 `CUDAExecutionProvider`。
- 与当前 AMD ROCm run 的跨体系 CSV 对比。

## 主要内容

- `benchmark_baselines.py`：ResNet18 + CIFAR-10、DistilBERT + SST-2 模型级 benchmark。
- `benchmark_runtime.py`：自动识别 GPU、PyTorch backend 和 ONNX Runtime provider。
- `operator_benchmark.py`：算子级 benchmark，支持 AMD ROCm/MIGraphX 与 NVIDIA CUDA/CUDAExecutionProvider。
- `analyze_operator_csv.py`：算子 CSV 汇总分析脚本。
- `download_baselines.py`：模型和数据集下载脚本。
- `OrtProfileAnalyzer.cs`：ORT profiling JSON 可视化工具源码。
- `ORT_Profile_Analyzer_WinForms.exe`：Windows 侧 ORT profile 分析工具。
- `ORT_Profile_Analyzer使用说明.md`：profile 分析工具使用说明。
- `AI芯片性能测试指导文档.md`：芯片性能测试层级、测试矩阵、自动化流程和问题清单。
- `基线Benchmark部署流程.md`：当前基线 benchmark 的部署与运行流程。
- `CODEX_CLI_CUDA部署引导.md`：另一台 RTX 3080/CUDA 机器使用 Codex CLI 部署和运行本流程的引导。
- `NEW_AI_CHIP_ONNXRUNTIME对齐指南.md`：新 AI 芯片驱动/runtime/ONNX Runtime EP 与 benchmark 对齐指南。
- `部署教程.md`、`使用文档.md`：环境部署和日常使用说明。

## 推荐运行目录

WSL 中推荐使用：

```bash
/home/l/benchmarks
```

当前 AMD ROCm Python 环境：

```bash
source ~/torch-rocm/.venv/bin/activate
```

RTX 3080 / CUDA 机器可使用自己的 CUDA 虚拟环境，例如：

```bash
source ~/torch-cuda/.venv/bin/activate
```

## 模型级 benchmark

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

python ~/benchmarks/scripts/benchmark_baselines.py \
  --output ~/benchmarks/outputs/baseline_results.csv
```

## 算子级 benchmark

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

python ~/benchmarks/scripts/operator_benchmark.py \
  --backends all \
  --batches 1,8,16 \
  --shape-profile large \
  --chain-len 10 \
  --repeat 3 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_results_v2.csv
```

ORT profiling JSON 输出目录：

```bash
~/benchmarks/outputs/ort_profiles
```

## 一键自动化 benchmark

`run_full_benchmark.py` 会从配置文件读取参数，创建独立 run 目录，并为每一步生成对应输出。

先复制一份本地配置。推荐使用 `.jsonc`，里面有注释：

```bash
cp ~/benchmarks/scripts/benchmark_config.example.jsonc ~/benchmarks/scripts/benchmark_config.local.jsonc
```

按需修改：

```bash
vim ~/benchmarks/scripts/benchmark_config.local.jsonc
```

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc
```

命令行参数可以临时覆盖配置文件，例如只跑算子级 benchmark：

```bash
python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc \
  --skip-baseline
```

每次运行会生成：

```text
~/benchmarks/runs/<run_id>/
  00_run_config.json
  metadata.json
  steps_status.md
  baseline_results.csv
  operator_results.csv
  operator_pair_summary.csv
  profile_summary.csv
  profile_summary.json
  regression_report.csv
  summary.md
  logs/
    01_environment.log
    02_baseline.log
    03_operator.log
  ort_profiles/
    *.json
```

其中 `steps_status.md` 是面向开发者阅读的步骤状态表，包含每一步状态、耗时、输出文件和失败原因。

`run_id` 留空时会自动使用时间戳目录；如果目录已存在，会自动追加 `_2`、`_3`，不会覆盖上次结果。

## 自动设备识别

脚本会自动识别当前 GPU 并配置 PyTorch/ONNX Runtime backend：

```text
AMD ROCm:
torch_backend=torch_rocm
onnx_providers=MIGraphXExecutionProvider,CPUExecutionProvider

NVIDIA CUDA:
torch_backend=torch_cuda
onnx_providers=CUDAExecutionProvider,CPUExecutionProvider

CPU only:
torch_backend=torch_cpu
onnx_providers=CPUExecutionProvider
```

单独检查当前环境识别结果：

```bash
python ~/benchmarks/scripts/benchmark_runtime.py
```

配置文件中可以覆盖自动识别：

```jsonc
"onnx_backend": "auto",
"onnx_providers": "auto",
"device_label": "auto",
"correctness_rtol": 0.001,
"correctness_atol": 0.0001
```

双体系对比时建议固定 `device_label`：

```jsonc
"device_label": "rx9070xt_rocm"
```

或：

```jsonc
"device_label": "rtx3080_cuda"
```

重新生成已有 run 的汇总，不重跑 benchmark：

```bash
python ~/benchmarks/scripts/run_full_benchmark.py \
  --summarize-run /home/l/benchmarks/runs/<run_id>
```

`operator_pair_summary.csv` 按 `op_name + batch_size + shape_profile` 配对 Torch ROCm 与 ONNX Runtime，并使用 `latency_per_op_mean_ms` 比较。新版算子脚本会让 Torch 与 ONNX 使用同一套有效 chain：可链式算子使用配置里的 `chain_len`，pool 和 embedding 使用 `effective_chain=1`。

## 正确性校验

模型级和算子级 ONNX Runtime 结果会自动和 PyTorch eager 参考输出比较，并在 CSV 中写入：

```text
correctness_status
max_abs_error
max_rel_error
correctness_message
```

`correctness_status=ok` 表示通过当前 `correctness_rtol/correctness_atol` 容差；`reference` 表示该行是 Torch 参考结果；`mismatch`、`shape_mismatch` 或 `output_count_mismatch` 需要优先看 `summary.md` 里的 `Correctness Issues`。新芯片或低精度模式可以调整容差，但要在开发记录中说明原因。

## ORT Profile 可视化

Windows 打开：

```text
ORT_Profile_Analyzer_WinForms.exe
```

导入 `~/benchmarks/outputs/ort_profiles/*.json` 后，工具会按 `op_name + batch_size` 记录并比较优化前后的变化。

从 Windows 资源管理器访问 WSL profile 目录：

```text
\\wsl.localhost\Ubuntu\home\l\benchmarks\outputs\ort_profiles
```

## 版本控制说明

仓库保存源码、文档和可执行工具。以下内容默认不纳入版本控制：

- benchmark 输出 CSV
- ORT profile JSON
- run 归档目录
- Python 缓存
- PyInstaller 构建中间产物
- 本地 profile 历史记录 `ort_profile_history.csv`

如需保留某次正式测试结果，建议放入独立 `runs/<timestamp>/` 目录，并根据需要调整 `.gitignore`。
