# AI 芯片性能测试指导文档

本文档面向 AI 芯片嵌入式工程师、驱动工程师、runtime 工程师和编译器工程师，用于规划从模型级到系统级的性能测试体系。

## 1. 测试目标

AI 芯片性能测试不应只停留在端到端模型延迟。完整测试目标包括：

```text
模型能否跑通
输出是否正确
算子是否完整支持
是否存在 CPU fallback
图优化和算子融合是否生效
kernel / DMA / memory / driver 是否存在瓶颈
长时间运行是否稳定
功耗和温度是否满足产品要求
```

建议将性能测试分为四层：

```text
第一层：模型级 Benchmark
第二层：算子级 Benchmark
第三层：图级 / Runtime Profile
第四层：系统级 / Driver Telemetry
```

## 2. 第一层：模型级 Benchmark

模型级 benchmark 是最外层的端到端测试，用于判断用户实际 workload 的整体表现。

典型 baseline：

```text
CV:
ResNet18 + CIFAR-10

NLP:
DistilBERT + GLUE/SST-2
```

目标回答：

```text
模型能不能跑通
端到端 latency 是多少
吞吐是否随 batch 增长
不同 runtime/backend 的差异
是否存在明显性能断崖
是否疑似 fallback 到 CPU
```

建议输出：

```text
baseline_results.csv
```

推荐字段：

```text
model
dataset
backend
batch_size
seq_len
precision
latency_mean_ms
latency_p50_ms
latency_p95_ms
latency_min_ms
latency_max_ms
samples_per_sec
memory_mb
metric_name
metric_value
status
error_message
```

适合读者：

```text
系统工程师
应用工程师
项目负责人
性能验证人员
```

局限：

```text
模型级 CSV 只能说明症状，不能直接定位具体算子、kernel 或驱动瓶颈。
```

## 3. 第二层：算子级 Benchmark

算子级 benchmark 是 AI 芯片开发中必须具备的测试层。它用于定位具体算子支持情况和性能短板。

### 3.1 CNN 相关算子

建议覆盖：

```text
Conv2D
DepthwiseConv2D
BatchNorm
ReLU
Add / Residual Add
MaxPool
AvgPool
GlobalAveragePool
GEMM / Linear
Concat
Reshape
Transpose
```

典型 shape：

```text
NCHW / NHWC
batch = 1, 8, 16, 32, 64
input = 224x224, 112x112, 56x56, 28x28, 14x14, 7x7
channels = 3, 16, 32, 64, 128, 256, 512
kernel = 1x1, 3x3, 5x5, 7x7
stride = 1, 2
padding = same / valid
```

### 3.2 Transformer 相关算子

建议覆盖：

```text
MatMul
BatchMatMul
GEMM
Add
LayerNorm
Softmax
GELU
Embedding
Gather
Attention pattern
Transpose
Reshape
Slice
Concat
Cast
```

典型 shape：

```text
batch = 1, 4, 8, 16, 32
seq_len = 32, 64, 128, 256, 512
hidden_size = 384, 768, 1024, 2048, 4096
num_heads = 6, 8, 12, 16, 32
head_dim = 64, 80, 128
```

目标回答：

```text
哪个算子不支持
哪个算子 fallback 到 CPU
哪个算子 latency 异常
哪个 shape 出现性能断崖
是否存在 layout transform 成本过高
是否存在小 shape launch overhead 过高
是否存在大 shape memory bandwidth 瓶颈
```

建议输出：

```text
operator_results.csv
```

推荐字段：

```text
op_name
opset
backend
dtype
layout
input_shape
output_shape
attributes
batch_size
warmup
iters
repeat_id
latency_mean_ms
latency_p50_ms
latency_p95_ms
samples_per_sec
bandwidth_gb_s
flops
tflops
memory_mb
status
error_message
```

适合读者：

```text
算子库工程师
kernel 工程师
驱动工程师
编译器工程师
runtime 工程师
```

## 4. 第三层：图级 / Runtime Profile

图级 profile 用来连接模型级问题和算子级问题，是定位真实性能瓶颈的关键层。

需要观察：

```text
每个 graph node 跑在哪个 backend
哪些 node 被融合
哪些 node fallback 到 CPU
每个 node 的耗时
每个 fused segment 的耗时
图编译时间
执行时间
内存拷贝时间
layout transform 时间
host-device sync 次数
```

目标回答：

```text
端到端慢到底慢在哪些节点
算子本身慢，还是图优化差
是 kernel 慢，还是调度/拷贝/同步慢
融合是否生效
动态 shape 是否导致重复编译
fallback 是否打断整图执行
```

建议输出：

```text
runtime_profile.json
provider_placement.csv
node_latency.csv
fusion_report.txt
compile_report.txt
```

推荐字段：

```text
node_name
op_type
provider
input_shape
output_shape
dtype
duration_us
is_fused
fused_group_id
fallback_reason
layout_before
layout_after
copy_time_us
compile_time_us
```

典型问题：

```text
大量小算子未融合
CPU fallback 频繁出现
layout transform 比 compute 还慢
host-device copy 过多
动态 shape 导致重复编译
runtime 同步点过多
```

适合读者：

```text
runtime 工程师
编译器工程师
图优化工程师
驱动工程师
```

## 5. 第四层：系统级 / Driver Telemetry

系统级 telemetry 用于判断芯片、驱动、内存系统和功耗温度是否满足产品要求。

需要观察：

```text
NPU/GPU 利用率
DDR 带宽
SRAM 使用率
DMA 时间
kernel launch 时间
队列等待时间
功耗
温度
频率
thermal throttling
内存碎片
driver error
timeout / reset
```

目标回答：

```text
芯片算力有没有喂满
瓶颈是 compute 还是 memory
是否被功耗或温度限制
是否存在驱动调度问题
是否存在 DMA 或缓存一致性问题
长时间运行是否稳定
是否存在内存泄漏
```

建议输出：

```text
driver_trace.csv
power_thermal.csv
memory_bandwidth.csv
runtime_timeline.json
stability_report.txt
```

推荐字段：

```text
timestamp
model
backend
batch_size
chip_util_percent
memory_bw_gb_s
ddr_read_gb_s
ddr_write_gb_s
power_w
temperature_c
frequency_mhz
dma_time_us
queue_wait_us
kernel_time_us
host_time_us
error_count
reset_count
```

适合读者：

```text
嵌入式工程师
驱动工程师
系统软件工程师
硬件验证工程师
产品稳定性工程师
```

## 6. 测试阶段矩阵

### 6.1 Bring-up 阶段

目标：

```text
能跑
```

测试内容：

```text
单算子 smoke test
小模型 smoke test
最小 batch
固定 shape
FP32
```

关注点：

```text
crash
unsupported op
driver load failure
runtime init failure
wrong result
timeout
```

输出：

```text
bringup_status.csv
unsupported_ops.txt
error_log.txt
```

### 6.2 Correctness 阶段

目标：

```text
跑得对
```

测试内容：

```text
算子精度对齐
模型输出对齐
不同 dtype
不同 layout
不同 shape
边界输入
```

关注点：

```text
误差
NaN / Inf
dtype cast
layout 错误
量化误差
padding / stride / broadcast 语义
```

输出：

```text
correctness_report.csv
operator_accuracy.csv
model_accuracy.csv
```

### 6.3 Performance 阶段

目标：

```text
跑得快
```

测试内容：

```text
模型级 benchmark
算子级 benchmark
batch scaling
seq_len scaling
不同 precision
不同 backend 对比
```

关注点：

```text
latency
throughput
p50 / p95
fallback
fusion
memory bandwidth
kernel launch overhead
```

输出：

```text
baseline_results.csv
operator_results.csv
throughput_scaling.csv
```

### 6.4 Optimization 阶段

目标：

```text
定位瓶颈并优化
```

测试内容：

```text
node profile
kernel profile
driver trace
fusion report
layout transform report
memory copy report
```

关注点：

```text
热点 node
热点 kernel
layout transform
host-device sync
DMA 等待
队列调度
重复编译
cache miss
```

输出：

```text
node_latency.csv
provider_placement.csv
runtime_profile.json
driver_trace.csv
optimization_report.md
```

### 6.5 Production 阶段

目标：

```text
稳定可交付
```

测试内容：

```text
长稳测试
高温测试
功耗测试
并发测试
多模型切换
异常输入
内存泄漏检查
性能衰减检查
driver recovery
```

关注点：

```text
稳定性
资源泄漏
thermal throttling
driver reset
超时恢复
功耗上限
频率波动
长时间性能退化
```

输出：

```text
stability_report.txt
power_thermal.csv
soak_test.csv
memory_leak_report.txt
production_readiness_report.md
```

## 7. 推荐最小闭环

对于新 AI 芯片，建议至少建立以下闭环：

```text
模型级 CSV 发现问题方向
算子级 benchmark 定位疑似算子
runtime profile 确认 fallback / fusion / node latency
driver telemetry 判断 compute / memory / DMA / scheduling 瓶颈
优化驱动、kernel、runtime 或编译器
回到模型级 CSV 验证端到端收益
```

最小必要产物：

```text
baseline_results.csv
operator_results.csv
node_latency.csv
provider_placement.csv
driver_trace.csv
```

## 8. 当前环境可先推进的下一步

基于当前 WSL2 + ROCm + PyTorch + ONNX Runtime MIGraphX 环境，建议下一步补齐：

```text
1. operator_benchmark.py
2. operator_results.csv
3. ONNX Runtime profiling json
4. provider placement / fallback 报告
5. PyTorch profiler trace
```

优先覆盖算子：

```text
Conv2D
BatchNorm
ReLU
Add
MatMul
BatchMatMul
LayerNorm
Softmax
GELU
Embedding
```

这些测试将把当前模型级 benchmark 扩展成可指导算子库、runtime、驱动和编译器优化的诊断体系。

## 9. 当前环境的算子级 Benchmark 脚本

已生成算子级 benchmark 脚本：

```text
Windows:
C:\Users\57323\Desktop\runtime\operator_benchmark.py

WSL:
/home/l/benchmarks/scripts/operator_benchmark.py
```

输出路径：

```text
/home/l/benchmarks/outputs/operator_results.csv
```

启用环境：

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

运行完整算子 benchmark：

```bash
python ~/benchmarks/scripts/operator_benchmark.py \
  --backends all \
  --batches 1,8,16,32 \
  --seq-len 128 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_results.csv
```

只运行 PyTorch GPU 算子：

```bash
python ~/benchmarks/scripts/operator_benchmark.py \
  --backends torch \
  --batches 1,8,16,32 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_torch_gpu.csv
```

只运行 ONNX Runtime 自动 provider 算子：

```bash
python ~/benchmarks/scripts/operator_benchmark.py \
  --backends onnx \
  --batches 1,8,16,32 \
  --seq-len 128 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_onnx_provider.csv
```

快速 smoke test：

```bash
python ~/benchmarks/scripts/operator_benchmark.py \
  --backends torch \
  --batches 1 \
  --warmup 0 \
  --iters 1 \
  --output ~/benchmarks/outputs/operator_smoke.csv
```

当前覆盖的 PyTorch GPU 算子：

```text
conv2d
batchnorm2d
maxpool2d
avgpool2d
relu
add
gelu
linear
batch_matmul
layernorm
softmax
embedding
```

当前覆盖的 ONNX Runtime provider 单算子模型：

```text
relu
add
gelu
softmax
batch_matmul
conv2d
batchnorm2d
maxpool2d
avgpool2d
linear
layernorm
embedding
```

CSV 字段：

```text
op_name
backend
provider
dtype
layout
input_shape
output_shape
graph_ops
attributes
batch_size
latency_mean_ms
latency_p50_ms
latency_p95_ms
latency_min_ms
latency_max_ms
tflops
bandwidth_gb_s
gpu_mem_mb
status
error_message
```

说明：

```text
provider 字段用于判断 ONNX Runtime 是否启用了目标 Execution Provider，例如 MIGraphXExecutionProvider、CUDAExecutionProvider 或未来的 YourChipExecutionProvider。
graph_ops 字段记录导出的 ONNX graph 中实际包含的 op_type，例如 Gemm、Conv、LayerNormalization、Gather。
output_shape 字段记录实际运行输出 shape，不再使用 unknown 占位。
gpu_mem_mb 只对 PyTorch GPU 后端有效，ONNX Runtime provider 的显存通常不经过 PyTorch allocator。
ONNX Runtime 首次运行可能触发 provider 编译或初始化，正式测试应使用 warmup 排除首次编译开销。
```

## 10. 算子级 Benchmark v2 修改记录

### 10.1 问题诊断

基于第一版 `operator_results.csv`，观察到：

```text
ONNX Runtime MIGraphX 在模型级 benchmark 中明显快于 PyTorch ROCm。
但在算子级 benchmark 中，ONNX 的 latency / p95 / TFLOPS 没有明显优势，部分算子甚至慢于 PyTorch ROCm。
```

初步判断：

```text
第一版 ONNX 算子 benchmark 更接近在测 ONNX Runtime 单次 sess.run() 调用成本，而不是纯 MIGraphX kernel 性能。
```

主要原因：

```text
1. 每个 ONNX microbenchmark graph 太小，通常只有 1 个 op。
2. 每次 sess.run() 都包含 Python -> ONNX Runtime 调用开销。
3. 输入来自 CPU numpy array，可能包含 host/device copy。
4. 输出回到 CPU，可能包含 device/host copy。
5. 单算子计算量太小，runtime 调度开销占比过高。
6. 没有记录 session 创建时间和首次运行时间，无法区分编译开销、首次执行开销和稳态执行开销。
7. 没有 ORT profiling，无法进一步检查 node 级执行和 provider 行为。
```

因此，第一版 CSV 只能说明：

```text
MIGraphX provider 能加载
算子 graph 能运行
端到端 sess.run latency
```

不能直接说明：

```text
MIGraphX 纯 kernel 算力
真实硬件 TFLOPS
底层 kernel 是否优于 PyTorch ROCm
```

### 10.2 v2 修改目标

v2 的目标是把以下开销显式拆出来，并降低单次 `sess.run()` 开销对结果的影响：

```text
session 创建 / 图编译开销
首次运行开销
稳态运行 latency
chain 后的单 op 平均 latency
ORT profiling trace
repeat 稳定性
大 shape 下的计算密度
```

### 10.3 已完成修改

`operator_benchmark.py` 已增加以下参数：

```text
--repeat
```

重复整组 benchmark 的次数。CSV 增加：

```text
repeat_id
```

用于观察多轮运行波动。

```text
--chain-len
```

将 ONNX 单算子串联多次，例如：

```text
input -> Conv -> Conv -> ... -> output
```

CSV 增加：

```text
chain_len
latency_per_op_mean_ms
```

计算方式：

```text
latency_per_op_mean_ms = latency_mean_ms / chain_len
```

用于摊薄 `sess.run()` 调用开销。

```text
--shape-profile
```

可选：

```text
standard
large
```

`large` 使用更大的 Conv / MatMul / Linear / elementwise shape，让计算时间占比更高。

CSV 增加：

```text
shape_profile
```

新增计时字段：

```text
session_create_ms
first_run_ms
```

含义：

```text
session_create_ms: 创建 ONNX Runtime session 的时间，包含 MIGraphX 编译/初始化相关开销。
first_run_ms: 第一次 sess.run() 时间，常包含 runtime lazy init 和缓存建立。
latency_mean_ms: warmup 后稳态运行的平均 latency。
```

新增 profiling：

```text
SessionOptions.enable_profiling = True
```

CSV 增加：

```text
profile_path
```

profile JSON 输出目录：

```text
/home/l/benchmarks/outputs/ort_profiles/
```

### 10.4 v2 输出字段新增项

相比第一版，CSV 新增：

```text
repeat_id
shape_profile
chain_len
session_create_ms
first_run_ms
latency_per_op_mean_ms
profile_path
```

这些字段用于区分：

```text
编译慢
首次运行慢
稳态运行慢
runtime 调用开销大
kernel 本身慢
大 shape 下算力不足
repeat 间波动过大
```

### 10.5 v2 运行方式

推荐先跑 ONNX large shape + chain：

```bash
cd ~/benchmarks
source ~/torch-rocm/.venv/bin/activate
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python ~/benchmarks/scripts/operator_benchmark.py \
  --backends onnx \
  --batches 1,8,16 \
  --shape-profile large \
  --chain-len 10 \
  --repeat 3 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_onnx_migraphx_v2.csv
```

对比 PyTorch ROCm：

```bash
python ~/benchmarks/scripts/operator_benchmark.py \
  --backends torch \
  --batches 1,8,16 \
  --shape-profile large \
  --repeat 3 \
  --warmup 10 \
  --iters 50 \
  --output ~/benchmarks/outputs/operator_torch_rocm_v2.csv
```

完整运行：

```bash
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

### 10.6 v2 结果解读方法

如果：

```text
session_create_ms 很高
first_run_ms 很高
latency_per_op_mean_ms 正常
```

说明主要问题是：

```text
图编译 / session 初始化 / 首次执行开销
```

如果：

```text
chain_len 增大后 latency_per_op_mean_ms 明显下降
```

说明第一版主要受：

```text
sess.run 调用开销
Python runtime 开销
输入输出处理开销
```

影响。

如果：

```text
large shape 下 ONNX TFLOPS 仍显著低于 PyTorch ROCm
```

说明更可能是：

```text
MIGraphX 对该算子的 kernel 路径较弱
该 shape 未命中高性能实现
layout transform 或 memory copy 仍然过重
provider 内部存在 fallback 或分解
```

如果：

```text
p95 远高于 p50
repeat 间差异大
```

说明需要进一步检查：

```text
WSL 调度抖动
GPU 频率变化
runtime 缓存
后台负载
MIGraphX profile trace
```

### 10.7 注意事项

`chain_len` 适合以下算子：

```text
relu
add
gelu
softmax
batch_matmul
conv2d
batchnorm2d
linear
layernorm
```

对于 pool 类算子：

```text
maxpool2d
avgpool2d
```

连续串联会改变空间尺寸，因此当前脚本只 chain 1 次。

对于 embedding：

```text
embedding
```

输入是 int64 token id，输出是 fp32 embedding，不能简单串联相同 embedding 层，因此当前脚本不 chain。

### 10.8 当前验证状态

已完成 smoke test：

```text
--backends onnx
--batches 1
--shape-profile standard
--chain-len 2
--repeat 1
--warmup 0
--iters 1
```

验证结果：

```text
12 个 ONNX 算子全部 status=ok
profile_path 正常生成
linear 链式 shape 已修正为 2048 -> 2048
graph_ops 可显示重复 op，例如 Gemm;Gemm
```

### 10.9 Conv2D chain 修复记录

问题现象：

```text
ONNX conv2d 在 --shape-profile large --chain-len 10 下全部 status=error。
导出日志出现 ChainModule export 失败。
```

原因：

```text
第一版 v2 复用了 ResNet 风格 Conv case，其中 large profile 选到的是 128 -> 256 的 Conv。
当 chain_len > 1 时，第一层 Conv 输出 256 通道，第二层 Conv 仍然期望 128 通道输入，导致链式 graph shape 不闭合。
```

修复：

```text
ONNX conv2d chain 改为专用的方形通道配置：

standard: 64 -> 64, H/W=56, kernel=3, stride=1, pad=1
large:    256 -> 256, H/W=56, kernel=3, stride=1, pad=1
```

修复后验证：

```text
--backends onnx
--batches 1
--shape-profile large
--chain-len 10
--repeat 1
--warmup 0
--iters 1
```

结果：

```text
conv2d status=ok
graph_ops=Conv;Conv;Conv;Conv;Conv;Conv;Conv;Conv;Conv;Conv
profile_path 正常生成
```

## 11. 基线自动化测试闭环与改进建议

本节记录当前环境下已经形成的自动化测试闭环，以及后续继续向“芯片优化指导工具链”演进时需要补齐的问题。

### 11.0 当前项目状态

当前 benchmark 工具链已经从单一 AMD MIGraphX 验证，扩展为可自动识别 ONNX Runtime provider 的通用流程：

```text
AMD ROCm:
torch_backend=torch_rocm
ONNX Runtime provider=MIGraphXExecutionProvider

NVIDIA CUDA:
torch_backend=torch_cuda
ONNX Runtime provider=CUDAExecutionProvider

CPU:
torch_backend=torch_cpu
ONNX Runtime provider=CPUExecutionProvider
```

当前 AMD RX 9070 XT + WSL2 ROCm 环境已完成 smoke test：

```text
run_id=rocm_auto_provider_smoke
device_name=AMD Radeon RX 9070 XT
torch_backend=torch_rocm
onnx_backend=migraphx
onnx_providers=MIGraphXExecutionProvider,CPUExecutionProvider
```

对于新 AI 芯片，目标应是实现类似：

```text
torch_backend=torch_<your_backend> 或 native runtime backend
ONNX Runtime provider=YourChipExecutionProvider,CPUExecutionProvider
```

并使 `operator_results.csv`、`profile_summary.csv`、`operator_pair_summary.csv` 能按同一结构输出。

### 11.1 当前已有流程

当前测试链路已经覆盖：

```text
模型级 benchmark:
ResNet18 + CIFAR-10
DistilBERT + GLUE/SST-2

算子级 benchmark:
operator_benchmark.py

ONNX Runtime profiling:
SessionOptions.enable_profiling = True
profile JSON 输出到 outputs/ort_profiles

结果汇总:
baseline_results.csv
operator_results_v2.csv

profile 可视化:
ORT_Profile_Analyzer_WinForms.exe
```

当前推荐的算子级运行方式：

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

profile JSON 目录：

```text
/home/l/benchmarks/outputs/ort_profiles/
```

Windows 侧可视化工具：

```text
C:\Users\57323\Desktop\runtime\ORT_Profile_Analyzer_WinForms.exe
```

### 11.2 当前 operator_results_v2.csv 的判断

当前 `operator_results_v2.csv` 已确认：

```text
rows: 243
errors: 0
shape_profile: large
profile_path 正常生成
ORT Node provider 可在 profile JSON 中看到 MIGraphXExecutionProvider
```

初步性能判断：

```text
conv2d:
batch 越大，ORT/MIGraphX 优势越明显

batch_matmul:
ORT/MIGraphX 有稳定优势

gelu:
ORT/MIGraphX 优势明显

linear:
小 batch 有优势，大 batch 基本持平

add / relu / batchnorm2d / maxpool2d / softmax:
当前 ORT/MIGraphX 没有明显优势，部分场景明显慢于 Torch ROCm
```

这说明当前 benchmark 已经能区分不同类型瓶颈：

```text
计算密集型算子:
更适合观察 kernel/compiler 优化收益

轻量 elementwise / pool / norm:
更容易暴露 runtime 调度、图融合、内存访问和 launch overhead 问题
```

### 11.3 使用 profile 可视化工具的优化闭环

优化前后建议按如下方式记录：

```text
1. 优化前运行 benchmark
2. 将本次 ORT profile JSON 导入 ORT_Profile_Analyzer_WinForms.exe
3. Run label 填 before_xxx
4. Notes 填驱动版本、runtime 版本、firmware 版本、编译器 commit
5. 优化后重新运行同一套 benchmark
6. 再次导入 JSON
7. Run label 填 after_xxx
8. 查看 Comparison by op + batch
```

重点比较字段：

```text
model_run_mean_ms:
稳态整体推理耗时，适合做主指标

model_run_p95_ms:
尾延迟，适合观察调度抖动和异常慢路径

node_mean_ms:
图节点层面的平均耗时

kernel_mean_ms:
MIGraphX kernel 事件的平均耗时

session_init_ms:
建图、编译、初始化耗时，不应和稳态推理延迟混在一起比较

provider:
确认是否为 MIGraphXExecutionProvider，检查是否有 CPU fallback

top_node_ms / top_node_name:
定位最慢节点
```

### 11.4 整个自动化流程的建议

后续建议把当前脚本升级为“一键运行 + 一次归档”的形式。

建议目录结构：

```text
/home/l/benchmarks/runs/
  2026-05-07_1534_rocm72_migraphx1232/
    metadata.json
    baseline_results.csv
    operator_results.csv
    ort_profiles/
    logs/
    summary.md
```

每次运行必须记录 metadata：

```text
测试时间
主机 OS / WSL 版本
GPU / 芯片型号
驱动版本
ROCm 版本
PyTorch 版本
ONNX Runtime 版本
MIGraphX 版本
Python 版本
benchmark 脚本 git hash 或文件 hash
环境变量
运行命令
warmup / iters / repeat
batch size
shape profile
chain_len
```

建议新增一个总控脚本：

```text
run_full_benchmark.py
```

职责：

```text
1. 创建唯一 run 目录
2. 采集环境信息
3. 运行模型级 benchmark
4. 运行算子级 benchmark
5. 收集 ORT profile JSON
6. 生成 summary.md
7. 生成可导入可视化工具的 profile 记录
8. 检查错误项和性能回退项
```

### 11.5 当前流程存在的问题

当前流程已经能用于初步定位方向，但还有以下问题：

```text
1. CSV 输出仍然是单文件覆盖模式
   多次运行容易覆盖旧结果，建议强制写入 timestamp run 目录。

2. operator_results_v2.csv 中可能混入不同 chain_len
   例如 chain_len=1 和 chain_len=10 同时存在时，分析必须按 chain_len 分组。

3. Torch ROCm 和 ONNX Runtime 的执行图不一定完全等价
   ONNX 导出、图优化、算子分解可能让二者不是逐 kernel 对比。

4. 轻量算子的 tflops 没有实际意义
   add/relu/pool/norm 更应该看 latency、bandwidth、launch overhead 和 fusion。

5. gpu_mem_mb 当前对 ONNX Runtime 缺失
   WSL/ROCm 环境下不容易从 ORT 侧直接拿到准确显存，需要结合 rocm-smi 或外部 telemetry。

6. provider 字段不能只看 session providers
   必须用 profile JSON 的 Node provider 或 runtime 日志确认每个节点是否 fallback。

7. session_create_ms 和 first_run_ms 不能代表稳态性能
   MIGraphX 可能有编译/初始化开销，应该和 latency_mean_ms 分开看。

8. WSL2 不是最终量产环境
   WSL2 适合开发验证，但驱动调度、内存路径和功耗行为不能完全代表裸机 Linux 或嵌入式板端。

9. 当前缺少精度正确性检查
   性能优化前后必须同时比较输出误差，避免 kernel 优化引入数值问题。

10. 当前缺少功耗、温度、频率、带宽计数器
    对芯片优化来说，仅有 latency CSV 不足以区分算力瓶颈、带宽瓶颈和调度瓶颈。
```

### 11.6 建议补齐的自动化检查

建议在每次 benchmark 后自动生成以下检查结果：

```text
错误检查:
status != ok 的算子列表

fallback 检查:
profile JSON 中 provider != 目标 EP 的 Node 列表

性能回退检查:
与上一轮同 op + batch + shape + chain 的结果比较，超过阈值则标红

稳定性检查:
p95 / mean 超过阈值时标记为 jitter

初始化开销检查:
session_create_ms 或 session_init_ms 异常高时单独列出

图结构检查:
graph_ops 是否符合预期，是否出现额外 Cast / Transpose / Reshape / Expand

正确性检查:
与 PyTorch CPU 或参考实现比较 max_abs_error / max_rel_error
```

建议阈值：

```text
性能回退:
latency_mean_ms 上升超过 5% 需要关注
latency_mean_ms 上升超过 10% 标记为 regression

尾延迟:
p95 / mean > 1.3 需要关注
p95 / mean > 1.5 标记为 jitter

fallback:
任何 CPUExecutionProvider Node 都需要单独列出

正确性:
fp32 max_abs_error 建议先控制在 1e-4 到 1e-3 量级，具体按算子调整
```

### 11.7 对芯片工程师的下一步建议

如果目标是指导新 AI 芯片的驱动、runtime、compiler、kernel 优化，建议按优先级推进：

```text
第一优先级:
保证所有 baseline 模型和算子 status=ok，且无非预期 fallback。

第二优先级:
建立固定测试矩阵和 run 目录归档，保证优化前后可重复比较。

第三优先级:
把 ORT profile JSON 转换成 node/kernel 级摘要，按 op + batch 追踪趋势。

第四优先级:
引入正确性校验，防止性能优化破坏数值结果。

第五优先级:
接入芯片 telemetry，包括功耗、温度、频率、HBM/DDR 带宽、DMA、队列深度。

第六优先级:
加入模型子图 benchmark，例如 Conv+BN+ReLU、Attention block、MLP block。
```

特别建议增加模型子图 benchmark：

```text
CV 子图:
Conv2D + BatchNorm + ReLU
Conv2D + Add + ReLU
DepthwiseConv + PointwiseConv

Transformer 子图:
QKV Linear
BatchMatMul + Softmax + BatchMatMul
LayerNorm + Linear + GELU + Linear

Embedding 子图:
Embedding + LayerNorm
Embedding + Position Add
```

原因：

```text
单算子 benchmark 能定位 kernel 能力。
模型级 benchmark 能说明最终用户体验。
子图 benchmark 才最适合定位 fusion、memory reuse、layout transform、runtime 调度这些真实性能问题。
```

### 11.8 当前结论

当前流程已经可以作为“早期芯片性能 bring-up 和优化方向判断”的基础工具链。

它已经能回答：

```text
哪些模型能跑
哪些算子能跑
哪些算子在目标 EP 上有优势
哪些算子疑似受 runtime overhead 或 memory overhead 影响
是否生成 ORT profiling JSON
优化前后同 op + batch 的延迟变化
```

但如果要作为芯片团队长期使用的性能回归系统，还需要继续补齐：

```text
run 目录归档
环境 metadata
正确性校验
provider fallback 自动扫描
profile 自动摘要
子图级 benchmark
功耗/频率/带宽 telemetry
性能回退阈值和自动报告
```

## 12. 新 AI 芯片与 ONNX Runtime 对齐建议

如果要基于当前 benchmark 测试一款新的 AI 芯片，不建议让驱动直接理解 ONNX。推荐软件栈分层：

```text
ONNX Runtime
  |
YourChipExecutionProvider
  |
YourChip Runtime / Compiler
  |
User Driver / HAL
  |
Kernel Driver
  |
AI Chip
```

各层职责：

```text
ONNX Runtime:
模型加载、图优化、子图 partition、调用 Execution Provider。

YourChipExecutionProvider:
声明支持哪些 ONNX op，将 ORT 子图转换成芯片 IR，调用 runtime 编译和执行。

Runtime / Compiler:
图编译、算子 lowering、layout、memory planning、kernel 选择、profiling timestamp。

User Driver / HAL:
device open/close、memory alloc、DMA、command buffer、queue submit、sync。

Kernel Driver:
设备枚举、MMU、IRQ、reset、power、错误恢复。
```

建议 bring-up 顺序：

```text
1. 裸驱动 bring-up:
   设备枚举、寄存器访问、DMA、命令队列、IRQ、reset。

2. runtime bring-up:
   malloc/memcpy/launch/sync/profile。

3. native 单算子:
   Add、Relu、MatMul、Gemm、Conv。

4. ONNX Runtime EP 最小接入:
   YourChipExecutionProvider 能接住 Add/Relu/MatMul/Gemm。

5. 运行当前 operator_benchmark.py:
   status=ok，provider=YourChipExecutionProvider,CPUExecutionProvider。

6. 运行模型子图:
   Conv+BN+Relu、Attention block、MLP block。

7. 运行模型级 benchmark:
   ResNet18、DistilBERT。

8. 性能优化:
   使用 operator_pair_summary.csv 和 profile_summary.csv 定位 kernel/runtime/driver 瓶颈。
```

与当前 benchmark 对齐的关键输出：

```text
metadata.json:
记录 device_label、device_name、torch_backend 或 native backend、onnx_providers。

operator_results.csv:
记录每个算子的 status、provider、latency、p95、chain_len、profile_path、correctness_status、max_abs_error、max_rel_error。

subgraph_results.csv:
记录模型子图级 Conv-BN-ReLU、ResNet BasicBlock、Transformer MLP、Self-Attention 的 latency、p95、provider、profiling 和 correctness。该层级用于检查算子组合、融合、layout transform、memory planning 和 attention/MLP 路径。

gpu_streams_results.csv / gpu_pinned_memory_results.csv:
记录 GPU stream 并发、copy/compute overlap、pageable/pinned H2D/D2H、pin_memory 开销。CUDA 和 ROCm 使用同一套 PyTorch 接口输出，适合排查 runtime/driver/memory transfer 层问题。

profile_summary.csv:
记录 ORT profiling JSON 中的 Session/Node/kernel 耗时和 provider。

operator_pair_summary.csv:
按 op_name + batch_size + shape_profile 对比不同 backend 的 per-op 延迟。
```

判断问题的方式：

```text
status=error:
算子支持、shape 支持、compiler lowering 或 runtime 执行失败。

provider=CPUExecutionProvider:
EP 没接住图，或者 GetCapability/partition 失败。

session_create_ms 高:
编译耗时高，或缺少 compile cache。

first_run_ms 高:
lazy init、memory allocation、kernel load 或首次 command queue 初始化开销。

latency_mean_ms 高:
kernel、DMA、layout transform、memory planning 或 queue overhead。

p95/mean 高:
驱动调度抖动、同步机制、频率变化、内存碎片或系统负载。

correctness_status=mismatch:
先暂停性能结论，检查 ONNX 导出、dtype、layout、广播语义、近似数学库和 provider fallback。新芯片开发应先让 P0 算子 correctness_status=ok，再看 latency / p95 / profiling。
```
