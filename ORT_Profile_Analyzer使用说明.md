# ORT Profile Analyzer 使用说明

## 文件

- `ORT_Profile_Analyzer_WinForms.exe`：Windows 桌面工具。
- `ort_profile_history.csv`：工具自动生成的历史记录文件，和 exe 放在同一目录。
- `OrtProfileAnalyzer.cs`：源码，后续需要改字段或逻辑时可继续编译。

## 使用流程

1. 双击 `ORT_Profile_Analyzer_WinForms.exe`。
2. 在 `Run label` 填本次测试名称，例如 `before_driver_fix`、`after_kernel_opt_v1`。
3. 在 `Notes` 填备注，例如驱动版本、库版本、芯片固件版本。
4. 点击 `Import JSON`，选择 ORT profiling 生成的 JSON 文件，可一次选择多个。
5. 上方表格展示单个 JSON 的关键数据。
6. 下方 `Comparison by op + batch` 会按 `op_name + batch_size` 自动比较最早记录和最新记录。
7. 点击 `Export Compare CSV` 可导出优化前后对比结果。

## 关键字段

- `model_run_mean_ms`：ORT 单次 `model_run` 平均耗时，适合做优化前后主对比。
- `model_run_p95_ms`：`model_run` 的 P95 延迟，用于观察抖动。
- `node_mean_ms`：Node 事件平均耗时，更接近执行图节点层面。
- `kernel_mean_ms`：名字包含 `kernel` 的 Node 事件平均耗时，通常对应 MIGraphX kernel 事件。
- `session_init_ms`：Session 初始化/编译耗时，不建议和稳态推理延迟混在一起比较。
- `provider`：Node 事件中记录的执行 Provider，例如 `MIGraphXExecutionProvider:62`。
- `top_node_ms` / `top_node_name`：最慢 Node 事件，适合定位瓶颈。

## 文件名解析规则

工具会从如下格式的文件名自动解析算子、batch、chain：

```text
ort_conv2d_large_chain10_bs16_rep3_2026-05-07_15-34-01.json
```

解析结果：

- `op_name=conv2d`
- `shape_profile=large`
- `chain_len=10`
- `batch_size=16`
- `repeat_id=3`

如果 JSON 文件名不符合该格式，仍可导入，但 `op_name/batch_size/chain_len` 会为空，无法自动按算子和 batch 聚合对比。

## 对芯片优化的使用方式

同一批 benchmark 建议保持：

- 相同算子集合
- 相同 batch size
- 相同 shape profile
- 相同 chain_len
- 相同 warmup/iters/repeat

优化前导入一次，`Run label` 写 `before_xxx`。
优化后再导入一次，`Run label` 写 `after_xxx`。

下方对比表中：

- `delta_ms < 0`：优化后更快。
- `change_pct < 0`：优化后延迟下降百分比。
- `change_pct > 0`：优化后变慢，需要检查 provider fallback、kernel、驱动或内存访问。
