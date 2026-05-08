# 新 AI 芯片接入 ONNX Runtime Benchmark 对齐指南

本文档说明如果有一款新的 AI 芯片，如何开发驱动/runtime/ONNX Runtime Execution Provider，并使用当前 benchmark 工具链做 bring-up、验收和性能优化。

## 1. 推荐软件栈

不要让驱动直接理解 ONNX。推荐分层：

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

职责划分：

```text
ONNX Runtime:
加载模型、图优化、子图 partition、调用 EP。

YourChipExecutionProvider:
声明支持的 ONNX op，将子图转换为芯片 IR，调用 runtime 编译和执行。

Runtime / Compiler:
图编译、算子 lowering、layout、memory planning、kernel 选择、profiling。

User Driver / HAL:
device open/close、memory alloc、DMA、command buffer、queue submit、sync。

Kernel Driver:
设备枚举、MMU、IRQ、reset、power、错误恢复。
```

## 2. 驱动 bring-up 顺序

建议先完成裸驱动和 runtime，再接 ONNX Runtime。

```text
P0:
设备枚举
寄存器访问
固件加载
内存申请和映射
Host -> Device DMA
Device -> Host DMA
命令队列提交
完成通知 / IRQ / polling
reset 和错误恢复

P1:
runtime malloc/free
runtime memcpy
runtime launch
runtime synchronize
runtime profiling timestamp

P2:
native Add / Relu / MatMul / Gemm / Conv

P3:
ONNX Runtime Execution Provider 最小接入
```

最早期验收：

```text
memcpy H2D/D2H 正确
vector add 正确
小 MatMul 正确
连续提交 10000 次不挂
错误命令能恢复
```

## 3. ONNX Runtime EP 对齐目标

目标是在 benchmark CSV 中看到：

```text
backend=onnxruntime
provider=YourChipExecutionProvider,CPUExecutionProvider
status=ok
```

如果看到：

```text
provider=CPUExecutionProvider
```

说明该节点没有跑在你的芯片上，而是 fallback 到 CPU。

EP 需要实现：

```text
GetCapability:
告诉 ORT 你支持哪些 op / shape / dtype。

Compile:
接收 ORT partition 后的子图，转换为芯片 IR 或 runtime graph。

Kernel Compute:
分配/绑定输入输出，提交 runtime command，等待完成。

Profiling:
暴露 node/kernel/session 级耗时，用于 profile_summary.csv。
```

## 4. 与当前 benchmark 对齐的输出

当前工具链要求每次 run 至少生成：

```text
metadata.json
steps_status.md
baseline_results.csv
operator_results.csv
operator_pair_summary.csv
profile_summary.csv
summary.md
```

对新芯片，`metadata.json` 应记录：

```text
device_label
device_name
torch_backend 或 native_backend
onnx_backend
onnx_providers
driver_version
runtime_version
compiler_version
firmware_version
```

`operator_results.csv` 重点字段：

```text
op_name
backend
provider
shape_profile
chain_len
batch_size
latency_mean_ms
latency_per_op_mean_ms
latency_p95_ms
correctness_status
max_abs_error
max_rel_error
profile_path
status
error_message
```

`profile_summary.csv` 重点字段：

```text
provider
has_cpu_fallback
session_init_ms
model_run_mean_ms
node_mean_ms
kernel_mean_ms
top_node_name
top_node_ms
```

## 5. 算子支持优先级

建议按以下顺序接入：

```text
P0:
Add
Relu
MatMul
Gemm
Conv
Reshape
Transpose
Cast
Constant
Identity

正确性要求:
每个接入到 YourChipExecutionProvider 的算子都应先做到 correctness_status=ok，再进入性能优化。若低精度内核需要放宽 correctness_rtol/correctness_atol，应在 run 的文档中记录精度模式、误差来源和业务可接受范围。

P1:
BatchNormalization
LayerNormalization
Softmax
Gelu
MaxPool
AveragePool
Gather

P2:
Conv+BN+Relu fusion
Gemm+Gelu fusion
Attention block
MLP block
```

当前 `operator_benchmark.py` 已覆盖：

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

## 6. 问题定位规则

```text
status=error:
算子支持、shape 支持、compiler lowering 或 runtime 执行失败。

provider=CPUExecutionProvider:
EP 没接住图，或 GetCapability/partition 失败。

session_create_ms / session_init_ms 高:
编译慢、cache 缺失、初始化重。

first_run_ms 高:
lazy init、memory allocation、kernel load 或首次 queue 初始化。

latency_mean_ms 高:
kernel、DMA、layout transform、memory planning 或 queue overhead。

p95 / mean 高:
驱动调度抖动、同步机制、频率变化、内存碎片或系统负载。

小算子慢:
launch overhead、fusion 缺失、runtime 调度开销。

大算子慢:
tiling、memory bandwidth、kernel 算法或 compiler 优化不足。
```

## 7. 建议里程碑

```text
阶段 1:
native driver/runtime 能稳定执行简单 command。

阶段 2:
native Add/Relu/MatMul/Gemm/Conv 正确。

阶段 3:
YourChipExecutionProvider 接入 ORT，P0 算子无 fallback。

阶段 4:
跑 operator_benchmark.py，生成 operator_results.csv 和 profile_summary.csv。

阶段 5:
跑模型子图 benchmark。

阶段 6:
跑 ResNet18 / DistilBERT。

阶段 7:
使用 operator_pair_summary.csv 和 profile_summary.csv 做性能优化闭环。
```

## 8. 与 AMD/NVIDIA 对比

当前工具链已经支持：

```text
AMD ROCm:
torch_rocm + MIGraphXExecutionProvider

NVIDIA CUDA:
torch_cuda + CUDAExecutionProvider
```

新芯片建议对齐为：

```text
YourChip:
native 或 torch_<your_backend> + YourChipExecutionProvider
```

最终比较时，不要只看单个 latency。至少同时看：

```text
device_label
provider
status
latency_per_op_mean_ms
latency_p95_ms
chain_len
has_cpu_fallback
session_init_ms
top_node_ms
```
