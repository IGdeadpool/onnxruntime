import csv
from collections import defaultdict
from pathlib import Path


CSV_PATH = str(Path.home() / "benchmarks" / "outputs" / "operator_results.csv")


def fnum(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("rows", len(rows))
    print("columns", ",".join(rows[0].keys()) if rows else "")
    errors = [r for r in rows if r.get("status") != "ok"]
    print("errors", len(errors))
    for r in errors[:20]:
        print(
            "ERR",
            r.get("op_name"),
            r.get("backend"),
            r.get("shape_profile"),
            r.get("chain_len"),
            r.get("batch_size"),
            r.get("error_message", "")[:160],
        )

    grouped = defaultdict(dict)
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r.get("op_name"), r.get("batch_size"), r.get("shape_profile"))
        grouped[key][r.get("backend")] = r

    print("\nPAIR_SUMMARY")
    print(
        "op,batch,profile,torch_ms,onnx_per_op_ms,onnx_p95_per_op_ms,"
        "speedup_torch_over_onnx,chain_len,session_create_ms,first_run_ms,provider"
    )
    for key in sorted(grouped, key=lambda x: (x[0], int(x[1] or 0), x[2] or "")):
        d = grouped[key]
        t = d.get("torch_rocm")
        o = d.get("onnxruntime")
        if not (t and o):
            continue
        chain_len = int(o.get("chain_len") or 1)
        torch_ms = fnum(t, "latency_mean_ms")
        onnx_per_op = fnum(o, "latency_per_op_mean_ms", fnum(o, "latency_mean_ms") / chain_len)
        onnx_p95_per_op = fnum(o, "latency_p95_ms") / max(chain_len, 1)
        speedup = torch_ms / onnx_per_op if onnx_per_op else 0.0
        print(
            ",".join(
                [
                    key[0],
                    key[1],
                    key[2],
                    f"{torch_ms:.6f}",
                    f"{onnx_per_op:.6f}",
                    f"{onnx_p95_per_op:.6f}",
                    f"{speedup:.3f}",
                    str(chain_len),
                    f"{fnum(o, 'session_create_ms'):.3f}",
                    f"{fnum(o, 'first_run_ms'):.3f}",
                    o.get("provider", ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
