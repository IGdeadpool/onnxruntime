# ResNet18 + DistilBERT 基线 Benchmark 部署流程

本文档记录当前 WSL2 ROCm 环境中部署两个 baseline benchmark 的完整流程：

- ResNet18 + CIFAR-10
- DistilBERT + GLUE/SST-2

当前机器环境：

```text
WSL: Ubuntu 24.04.4 LTS
GPU: AMD Radeon RX 9070 XT
ROCm: 7.2
Python env: /home/l/torch-rocm/.venv
Benchmark root: /home/l/benchmarks
```

## 1. 目录规划

所有 benchmark 相关文件放在：

```text
/home/l/benchmarks
```

目录结构：

```text
/home/l/benchmarks
├── data
│   ├── cifar10
│   └── huggingface_datasets
├── models
│   ├── torch
│   └── huggingface
├── outputs
└── scripts
```

创建目录：

```bash
mkdir -p ~/benchmarks/{data,models,scripts,outputs}
```

## 2. 启动 Python 环境

使用之前已经部署好的 ROCm PyTorch 虚拟环境：

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate
```

设置缓存目录：

```bash
export TORCH_HOME=/home/l/benchmarks/models/torch
export HF_HOME=/home/l/benchmarks/models/huggingface
export TRANSFORMERS_CACHE=/home/l/benchmarks/models/huggingface/transformers
export HF_DATASETS_CACHE=/home/l/benchmarks/data/huggingface_datasets
```

如果要运行 ONNX Runtime MIGraphX，也设置：

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## 3. 安装依赖

依赖安装在 `/home/l/torch-rocm/.venv` 中：

```bash
pip install --upgrade --retries 10 --timeout 120 \
  transformers datasets evaluate scikit-learn pandas requests
```

已有核心依赖：

```text
torch 2.9.1+rocm7.2.0
torchvision 0.24.0+rocm7.2.0
onnx 1.21.0
onnxruntime-migraphx 1.23.2
```

## 4. ResNet18 + CIFAR-10 Baseline

模型：

```text
torchvision.models.resnet18
```

权重：

```text
ResNet18_Weights.DEFAULT
```

数据集：

```text
CIFAR-10
```

推荐输入规格：

```text
Input shape: [N, 3, 224, 224]
Batch sizes: 1, 8, 16, 32, 64
Precision: FP32
Metrics: latency, throughput, Top-1 accuracy
```

下载 ResNet18 权重和 CIFAR-10：

```bash
python - <<'PY'
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

root = Path("/home/l/benchmarks")
weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights).eval()
print("ResNet18 ready:", weights)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

train = datasets.CIFAR10(
    root=str(root / "data" / "cifar10"),
    train=True,
    download=True,
    transform=transform,
)

test = datasets.CIFAR10(
    root=str(root / "data" / "cifar10"),
    train=False,
    download=True,
    transform=transform,
)

print("CIFAR-10:", len(train), len(test))
PY
```

当前状态：

```text
ResNet18 权重已下载到 /home/l/benchmarks/models/torch/hub/checkpoints
CIFAR-10 还需要继续下载
```

## 5. DistilBERT + SST-2 Baseline

模型：

```text
distilbert-base-uncased-finetuned-sst-2-english
```

数据集：

```text
GLUE / SST-2
```

推荐输入规格：

```text
Max sequence length: 128
Batch sizes: 1, 4, 8, 16, 32
Precision: FP32
Metrics: latency, throughput, accuracy
```

下载 SST-2 和 DistilBERT：

```bash
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = Path("/home/l/benchmarks")
model_id = "distilbert-base-uncased-finetuned-sst-2-english"

sst2 = load_dataset(
    "glue",
    "sst2",
    cache_dir=str(root / "data" / "huggingface_datasets"),
)
print("SST-2 splits:", {k: len(v) for k, v in sst2.items()})

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=str(root / "models" / "huggingface"),
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    cache_dir=str(root / "models" / "huggingface"),
).eval()

print("DistilBERT ready:", model_id)
print("vocab size:", tokenizer.vocab_size)
PY
```

## 6. 下载脚本

已保存下载脚本：

```text
/home/l/benchmarks/scripts/download_baselines.py
```

运行方式：

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

export TORCH_HOME=/home/l/benchmarks/models/torch
export HF_HOME=/home/l/benchmarks/models/huggingface
export TRANSFORMERS_CACHE=/home/l/benchmarks/models/huggingface/transformers
export HF_DATASETS_CACHE=/home/l/benchmarks/data/huggingface_datasets

python ~/benchmarks/scripts/download_baselines.py
```

如果网络中断，可以重复运行。Torch 和 Hugging Face 都会复用缓存。

## 7. 下载状态检查

查看目录大小：

```bash
du -h -d 4 ~/benchmarks | sort -h | tail -60
```

查看 ResNet18 权重：

```bash
ls -lh ~/benchmarks/models/torch/hub/checkpoints
```

查看 CIFAR-10：

```bash
find ~/benchmarks/data/cifar10 -maxdepth 3 -type f | head
```

查看 Hugging Face 缓存：

```bash
find ~/benchmarks/models/huggingface -maxdepth 4 -type f | head
find ~/benchmarks/data/huggingface_datasets -maxdepth 4 -type f | head
```

## 8. 验证脚本

验证 ResNet18：

```bash
python - <<'PY'
import torch
from torchvision.models import ResNet18_Weights, resnet18

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval().cuda()
x = torch.randn(8, 3, 224, 224, device="cuda")
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print(y.shape, y.device)
PY
```

验证 DistilBERT：

```bash
python - <<'PY'
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/l/benchmarks/models/huggingface")
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    cache_dir="/home/l/benchmarks/models/huggingface",
).eval().cuda()

inputs = tokenizer(
    ["this movie is great", "this movie is terrible"],
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    out = model(**inputs)
torch.cuda.synchronize()
print(out.logits.shape, out.logits.device)
PY
```

## 9. 预期完成状态

完成后应具备：

```text
ResNet18 权重缓存
CIFAR-10 train/test 数据
DistilBERT SST-2 fine-tuned 模型和 tokenizer
GLUE/SST-2 train/validation/test 数据
```

这两个 baseline 后续可用于：

```text
PyTorch ROCm latency/throughput benchmark
PyTorch -> ONNX export
ONNX Runtime MIGraphX inference benchmark
CPU fallback comparison
```

## 10. 运行 Benchmark 脚本

已生成统一 benchmark 脚本：

```text
Windows:
C:\Users\57323\Desktop\runtime\benchmark_baselines.py

WSL:
/home/l/benchmarks/scripts/benchmark_baselines.py
```

进入 WSL 后，先启用环境：

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

export TORCH_HOME=/home/l/benchmarks/models/torch
export HF_HOME=/home/l/benchmarks/models/huggingface
export HF_DATASETS_CACHE=/home/l/benchmarks/data/huggingface_datasets
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

运行完整 benchmark：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models all \
  --backends all \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/baseline_results.csv
```

只运行 PyTorch GPU：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models all \
  --backends torch \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/pytorch_gpu_results.csv
```

只运行 ONNX Runtime 自动 provider：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models all \
  --backends onnx \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/onnx_provider_results.csv
```

只跑 ResNet18：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models resnet18 \
  --backends all \
  --resnet-batches 1,8,16,32,64 \
  --output ~/benchmarks/outputs/resnet18_results.csv
```

只跑 DistilBERT：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models distilbert \
  --backends all \
  --bert-batches 1,4,8,16,32 \
  --seq-len 128 \
  --output ~/benchmarks/outputs/distilbert_results.csv
```

快速 smoke test：

```bash
python ~/benchmarks/scripts/benchmark_baselines.py \
  --models resnet18 \
  --backends torch \
  --resnet-batches 1 \
  --warmup 0 \
  --iters 1 \
  --output ~/benchmarks/outputs/smoke_results.csv
```

## 11. Benchmark 参数说明

脚本参数：

```text
--models
```

选择模型。可选：

```text
all
resnet18
distilbert
```

```text
--backends
```

选择运行后端。可选：

```text
all
torch
onnx
```

其中：

```text
torch -> PyTorch ROCm
onnx  -> ONNX Runtime MIGraphXExecutionProvider + CPUExecutionProvider fallback
```

```text
--resnet-batches
```

ResNet18 batch size 列表，例如：

```text
1,8,16,32,64
```

```text
--bert-batches
```

DistilBERT batch size 列表，例如：

```text
1,4,8,16,32
```

```text
--seq-len
```

DistilBERT 输入序列长度，默认：

```text
128
```

```text
--warmup
```

正式计时前的预热轮数。预热用于排除首次 GPU 初始化、kernel 编译和缓存构建的影响。

```text
--iters
```

正式计时轮数。数值越大，统计越稳定，但运行时间越长。

```text
--output
```

CSV 输出路径。

## 12. CSV 字段说明

输出 CSV 字段：

```text
model
```

模型名称，例如 `resnet18` 或 `distilbert-base-uncased-finetuned-sst-2-english`。

```text
dataset
```

数据集名称，例如 `cifar10` 或 `glue/sst2-validation`。

```text
backend
```

运行后端，例如：

```text
torch_rocm
onnxruntime:MIGraphXExecutionProvider,CPUExecutionProvider
```

```text
batch_size
```

每次推理的样本数。

```text
seq_len
```

文本模型的序列长度。ResNet18 没有该字段，因此为空。

```text
samples_per_sec
```

吞吐量，每秒处理样本数：

```text
samples_per_sec = batch_size / (latency_mean_ms / 1000)
```

```text
latency_mean_ms
latency_p50_ms
latency_p95_ms
latency_min_ms
latency_max_ms
```

单个 batch 的推理延迟统计，单位毫秒。

```text
metric_name
metric_value
```

附带的简单验证指标。

当前脚本中：

```text
ResNet18: pseudo_top1
DistilBERT: pred_positive_ratio
```

说明：ResNet18 当前使用 ImageNet 预训练权重跑 CIFAR-10，类别不匹配，因此 `pseudo_top1` 只用于 smoke test，不代表真实 CIFAR-10 accuracy。

```text
gpu_mem_mb
```

PyTorch ROCm 后端的 PyTorch allocator 峰值显存，单位 MB。

ONNX Runtime MIGraphX 行该字段为空，因为 ONNX Runtime 的显存分配不经过 PyTorch allocator，`torch.cuda.max_memory_allocated()` 无法统计 MIGraphX 的显存。

## 13. 完整自动化测试流程建议

当前建议将测试流程固定为三步，并使用 `run_full_benchmark.py + benchmark_config.local.json` 统一调度：

```text
第一步：模型级 benchmark
第二步：算子级 benchmark
第三步：ORT profiling JSON 导入可视化工具
```

### 13.1 配置文件

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

cp ~/benchmarks/scripts/benchmark_config.example.jsonc \
   ~/benchmarks/scripts/benchmark_config.local.jsonc
```

编辑本地配置：

```bash
vim ~/benchmarks/scripts/benchmark_config.local.jsonc
```

关键配置：

```text
benchmark_root
script_dir
runs_dir
run_id
python
baseline_models
baseline_backends
resnet_batches
bert_batches
baseline_warmup
baseline_iters
operator_backends
operator_batches
shape_profile
chain_len
repeat
operator_warmup
operator_iters
compare_run
regression_threshold_pct
```

说明：

```text
配置文件支持 // 和 /* ... */ 注释，推荐使用 .jsonc。
run_id 留空时，脚本会自动使用当前时间戳创建目录。
如果目标 run 目录已存在，脚本会自动追加 _2、_3，不会覆盖上次结果。
```

设备和 provider 配置：

```text
onnx_backend=auto
onnx_providers=auto
device_label=auto
```

自动识别规则：

```text
AMD ROCm:
torch_backend=torch_rocm
ONNX Runtime providers=MIGraphXExecutionProvider,CPUExecutionProvider

NVIDIA CUDA:
torch_backend=torch_cuda
ONNX Runtime providers=CUDAExecutionProvider,CPUExecutionProvider

CPU:
torch_backend=torch_cpu
ONNX Runtime providers=CPUExecutionProvider
```

单独检查当前机器识别结果：

```bash
python ~/benchmarks/scripts/benchmark_runtime.py
```

当前 RX 9070 XT WSL2 ROCm 环境验证结果：

```text
device_name=AMD Radeon RX 9070 XT
torch_backend=torch_rocm
onnx_backend=migraphx
onnx_providers=MIGraphXExecutionProvider,CPUExecutionProvider
```

### 13.2 一键运行完整流程

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate

python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc
```

运行开始后，终端会实时输出当前步骤：

```text
[RUN] run_id=...
[RUN] run_dir=/home/l/benchmarks/runs/...
[RUN] steps_status=/home/l/benchmarks/runs/.../steps_status.md
[START] 01_environment
[OK] 01_environment finished ...
[START] 02_baseline
...
```

说明：

```text
终端会实时显示每一步开始和结束状态。
benchmark_baselines.py / operator_benchmark.py 的输出会同时显示在终端，并写入 logs/*.log。
如果长时间没有新输出，可以打开 steps_status.md 或 tail 对应 log 文件确认进度。
```

临时覆盖配置示例：

```bash
python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.local.jsonc \
  --skip-baseline
```

### 13.3 每一步输出

每次运行会创建：

```text
/home/l/benchmarks/runs/<run_id>/
```

对应输出：

```text
00_run_config.json
metadata.json
steps_status.md
baseline_results.csv
operator_results.csv
subgraph_results.csv
gpu_streams_results.csv
gpu_pinned_memory_results.csv
profile_collect_summary.txt
profile_summary.csv
profile_summary.json
operator_pair_summary.csv
regression_report.csv 或 regression_skipped.txt
summary.md
logs/
ort_profiles/
```

说明：

```text
steps_status.md:
开发者阅读的步骤状态表，记录每一步状态、耗时、输出文件、失败原因和执行命令。

metadata.json:
环境快照，包括 OS、Python、PyTorch、ONNX Runtime、ROCm/GPU 信息和关键环境变量。

数据集和模型缓存:
benchmark_baselines.py 会优先从本地缓存读取 HuggingFace 数据集和模型，不会在缓存已存在时每次访问 HuggingFace。只有本地缓存缺失时才回退到下载。CIFAR-10 仍要求本地目录已准备好。

baseline_results.csv:
模型级 benchmark 原始结果。

operator_results.csv:
算子级 benchmark 原始结果。

subgraph_results.csv:
模型子图级 benchmark 原始结果，覆盖 conv_bn_relu、resnet_basic_block、transformer_mlp、self_attention。
该层级用于判断算子组合、融合、layout transform、memory planning、attention/MLP 子图是否成为瓶颈。

gpu_streams_results.csv:
可选 GPU stream 并发专项测试，CUDA 和 ROCm 共用。默认跳过，配置 `run_gpu_aux=true` 后生成。脚本会先创建 CSV，再按 `single_stream/serial/multi_stream/copy_compute_overlap` 逐项打印进度并增量写入。

gpu_pinned_memory_results.csv:
可选 pageable/pinned H2D/D2H、copy/compute overlap、pin_memory 开销测试，CUDA 和 ROCm 共用。默认跳过，配置 `run_gpu_aux=true` 后生成。默认参数是快速诊断模式：

```jsonc
"gpu_aux_warmup": 1,
"gpu_aux_repeat": 3,
"gpu_aux_matrix_size": 256,
"gpu_aux_iters": 1,
"gpu_aux_kernels": "vectorAdd,gemm",
"gpu_aux_streams": "2,4"
```

如果要做压力测试，可手动提高到 `gpu_aux_matrix_size=1024`、`gpu_aux_iters=10`、`gpu_aux_repeat=5`，并建议单独运行，避免完整流程被长时间 stream/拷贝测试阻塞。运行期间显存不下降通常是 PyTorch/ROCm 缓存和测试进程仍在持有张量，进程结束后才会完全释放。

正确性校验字段:
baseline_results.csv、operator_results.csv 和 subgraph_results.csv 会包含 correctness_status、max_abs_error、max_rel_error、correctness_message。
ONNX Runtime 行会和 PyTorch eager 参考输出比较；Torch 行标记为 reference。

profile_collect_summary.txt:
记录从 outputs/ort_profiles 收集到本轮 run 目录的 JSON 文件。

profile_summary.csv:
从 ORT profiling JSON 提取出的 node/kernel/session 摘要。

operator_pair_summary.csv:
按 op + batch + shape + chain 对比 Torch ROCm 与 ONNX Runtime。

regression_report.csv:
配置 compare_run 后，比较当前 run 与历史 run 的性能回退。

summary.md:
本轮 benchmark 的最终人类可读报告。
```

正确性状态说明：

```text
ok:
ONNX Runtime 输出在 correctness_rtol / correctness_atol 容差内。

reference:
Torch eager 参考行，不参与正确性失败统计。

mismatch:
shape 一致但数值误差超出容差，需要结合 max_abs_error、max_rel_error 和 ORT profiling 判断。

shape_mismatch / output_count_mismatch:
输出结构不一致，通常优先检查 ONNX 导出、动态 shape、provider kernel 输出定义或 fallback 路径。
```

配置文件中的默认容差：

```jsonc
"correctness_rtol": 0.001,
"correctness_atol": 0.0001
```

FP32 基准建议先固定该容差。若新芯片使用 FP16、BF16、INT8 或近似数学库，可以放宽容差，但需要在开发文档里记录精度模式和放宽原因。

注意：

```text
算子级 ONNX 会在每次运行时按当前模块重新导出。
原因是 conv2d / linear / embedding 等算子包含随机初始化权重，复用旧 ONNX 文件会导致 ONNX 权重和 PyTorch 参考权重不一致，从而产生虚假的高错误率。
模型级 ResNet18 / DistilBERT 使用固定预训练权重，仍可复用已导出的 ONNX 文件。
```

`operator_results.csv` 的 chain 规则：

```text
Torch ROCm 和 ONNX Runtime 使用同一套有效 chain_len。

可安全链式的算子：
relu / add / gelu / softmax / batch_matmul / conv2d / batchnorm2d / linear / layernorm
两边都使用配置中的 chain_len，例如 10。

不适合简单链式的算子：
maxpool2d / avgpool2d / embedding
两边都使用 effective_chain=1，并在 attributes 中标注 effective_chain=1。
```

原因：

```text
pool 连续串联会改变空间尺寸，不再代表同一个算子 shape。
embedding 输入是 token id，输出是 fp32 embedding，不能直接把输出再送入同一个 embedding。
```

`operator_pair_summary.csv` 配对规则：

```text
按 op_name + batch_size + shape_profile 配对 Torch ROCm 与 ONNX Runtime。
正常情况下，torch_chain_len 和 ort_chain_len 应一致。
比较字段使用 latency_per_op_mean_ms，避免 ONNX chain_len=10 的总耗时直接和 Torch 单算子总耗时比较。
```

如果看到 `torch_chain_len=1` 但 `ort_chain_len=10`：

```text
说明该结果来自旧版 operator_benchmark.py。
旧版只给 ONNX Runtime 使用 chain_len，Torch 仍按单算子跑。
该结果不建议作为长期自动化对比基线，应使用新版脚本重新运行。
```

如果 `operator_pair_summary.csv` 只有表头，通常说明：

```text
operator_results.csv 没有成功生成
03_operator 步骤失败或被跳过
operator_results.csv 中缺少 torch_rocm 或 onnxruntime 其中一个 backend
同一 op + batch + shape 下没有同时成功的 Torch 与 ONNX 行
```

已有 run 目录不需要重跑 benchmark，也可以重新生成汇总：

```bash
python ~/benchmarks/scripts/run_full_benchmark.py \
  --summarize-run /home/l/benchmarks/runs/<run_id>
```

实时查看进度：

```bash
# 查看步骤状态
cat /home/l/benchmarks/runs/<run_id>/steps_status.md

# 查看 baseline 日志
tail -f /home/l/benchmarks/runs/<run_id>/logs/02_baseline.log

# 查看 operator 日志
tail -f /home/l/benchmarks/runs/<run_id>/logs/03_operator.log
```

检查是否仍在运行：

```bash
ps -ef | grep -E 'run_full_benchmark|benchmark_baselines|operator_benchmark' | grep -v grep
```

如果中断了终端但进程仍在后台运行，应先确认是否需要保留本轮测试。若确认要停止：

```bash
kill <pid>
```

不要直接删除正在写入的 run 目录。

### 13.8 当前 ROCm 自动 provider 验证记录

已在当前 WSL2 + RX 9070 XT + ROCm 环境运行轻量验证：

```bash
python ~/benchmarks/scripts/run_full_benchmark.py \
  --config ~/benchmarks/scripts/benchmark_config.example.jsonc \
  --run-id rocm_auto_provider_smoke \
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

输出目录：

```text
/home/l/benchmarks/runs/rocm_auto_provider_smoke
```

验证结果：

```text
metadata.json 中 runtime_detection 正确识别：
device_name=AMD Radeon RX 9070 XT
torch_backend=torch_rocm
onnx_backend=migraphx
onnx_providers=MIGraphXExecutionProvider,CPUExecutionProvider

baseline_results.csv 生成 torch_rocm 和 onnxruntime:MIGraphXExecutionProvider,CPUExecutionProvider 两行。
operator_pair_summary.csv 生成 12 个算子配对结果。
profile_summary.csv 正常生成。
```

### 13.4 导入 ORT profile 可视化工具

Windows 侧工具：

```text
C:\Users\57323\Desktop\runtime\ORT_Profile_Analyzer_WinForms.exe
```

使用方式：

```text
1. 打开 exe
2. Run label 填本轮测试名称，例如 before_driver_fix
3. Notes 填驱动、ROCm、ONNX Runtime、MIGraphX、固件或编译器版本
4. Import JSON 选择 /home/l/benchmarks/outputs/ort_profiles 中的 JSON
5. 优化后重复导入，Run label 填 after_xxx
6. 查看 Comparison by op + batch
```

如果要从 Windows 资源管理器访问 WSL profile 目录：

```text
\\wsl.localhost\Ubuntu\home\l\benchmarks\outputs\ort_profiles
```

### 13.5 run 归档结构

当前 `baseline_results.csv` 和 `operator_results_v2.csv` 容易被覆盖。建议后续每次运行都生成独立目录：

```text
/home/l/benchmarks/runs/2026-05-07_1534_rocm72_ort1232/
  metadata.json
  baseline_results.csv
  operator_results.csv
  ort_profiles/
  summary.md
```

`metadata.json` 建议记录：

```text
测试时间
GPU / 芯片型号
Windows driver 版本
WSL Ubuntu 版本
ROCm 版本
PyTorch 版本
ONNX Runtime 版本
MIGraphX 版本
Python 版本
运行命令
warmup / iters / repeat
batch size
shape profile
chain_len
```

### 13.6 当前流程的已知问题

```text
1. ONNX Runtime 的 gpu_mem_mb 目前为空
   因为 ORT/MIGraphX 不经过 PyTorch allocator，需要额外 telemetry。

2. tflops 对轻量算子没有明显意义
   add/relu/pool/norm 应重点看 latency、p95、bandwidth、fusion 和 launch overhead。

3. chain_len 必须按算子理解
   pool 和 embedding 当前不能简单串联 10 次，因此分析 CSV 时要同时看 chain_len。

4. provider 不能只看 session provider 列表
   必须检查 ORT profile JSON 中 Node 事件的 provider 字段。

5. session_create_ms / session_init_ms 不等于稳态推理延迟
   MIGraphX 编译初始化开销需要单独分析。

6. WSL2 结果适合开发验证，但不能完全代表最终嵌入式板端
   最终仍需要裸机 Linux 或目标系统复测。

7. 当前模型级 metric 还不是严格精度评估
   后续应增加真实 accuracy / F1 或输出误差对比。
```

### 13.7 后续最值得补齐的能力

优先级建议：

```text
P0:
自动创建 run 目录，避免覆盖旧结果。

P1:
自动生成 metadata.json，保证每轮结果可追溯。

P2:
自动扫描 profile JSON，列出 CPU fallback 和最慢 Node。

P3:
增加正确性校验，记录 max_abs_error / max_rel_error。

P4:
增加模型子图 benchmark，例如 Conv+BN+ReLU、Attention block、MLP block。

P5:
接入功耗、温度、频率、显存、带宽等 telemetry。
```
