# ONNX Runtime MIGraphX Benchmark Toolkit

本仓库用于归档当前 WSL2 + ROCm + PyTorch + ONNX Runtime MIGraphX 环境下的基线测试脚本、算子级 benchmark、profiling 分析工具和部署文档。

## 主要内容

- `benchmark_baselines.py`：ResNet18 + CIFAR-10、DistilBERT + SST-2 模型级 benchmark。
- `operator_benchmark.py`：算子级 benchmark，支持 Torch ROCm 与 ONNX Runtime MIGraphX 对比。
- `analyze_operator_csv.py`：算子 CSV 汇总分析脚本。
- `download_baselines.py`：模型和数据集下载脚本。
- `OrtProfileAnalyzer.cs`：ORT profiling JSON 可视化工具源码。
- `ORT_Profile_Analyzer_WinForms.exe`：Windows 侧 ORT profile 分析工具。
- `ORT_Profile_Analyzer使用说明.md`：profile 分析工具使用说明。
- `AI芯片性能测试指导文档.md`：芯片性能测试层级、测试矩阵、自动化流程和问题清单。
- `基线Benchmark部署流程.md`：当前基线 benchmark 的部署与运行流程。
- `部署教程.md`、`使用文档.md`：环境部署和日常使用说明。

## 推荐运行目录

WSL 中推荐使用：

```bash
/home/l/benchmarks
```

Python 环境：

```bash
source ~/torch-rocm/.venv/bin/activate
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
