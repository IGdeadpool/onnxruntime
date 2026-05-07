import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = Path("/home/l/benchmarks")
DEFAULT_OUTPUTS = DEFAULT_ROOT / "outputs"
DEFAULT_RUNS = DEFAULT_ROOT / "runs"
DEFAULT_PROFILE_DIR = DEFAULT_OUTPUTS / "ort_profiles"

PROFILE_RE = re.compile(
    r"ort_(?P<op>.+?)_(?P<shape>[^_]+)_chain(?P<chain>\d+)_bs(?P<bs>\d+)_rep(?P<rep>\d+)",
    re.IGNORECASE,
)


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class StepRunner:
    def __init__(self, run_dir: Path, continue_on_error: bool):
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.continue_on_error = continue_on_error
        self.status: list[dict[str, object]] = []
        ensure_dir(self.logs_dir)

    def write_status_md(self) -> None:
        lines = [
            "# Benchmark Step Status",
            "",
            f"Updated at: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "| Step | Status | Duration | Outputs | Error |",
            "|---|---|---:|---|---|",
        ]
        for item in self.status:
            outputs = "<br>".join(f"`{output}`" for output in item.get("outputs", []))
            error = str(item.get("error", "")).replace("|", "\\|")
            lines.append(
                f"| `{item['step']}` | `{item['status']}` | {item['duration_sec']} sec | {outputs} | {error} |"
            )

        lines += [
            "",
            "## Commands",
            "",
        ]
        for item in self.status:
            command = item.get("command", [])
            if command:
                lines.append(f"### {item['step']}")
                lines.append("")
                lines.append("```bash")
                lines.append(" ".join(str(part) for part in command))
                lines.append("```")
                lines.append("")

        (self.run_dir / "steps_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def record(
        self,
        step: str,
        status: str,
        start: float,
        outputs: list[Path] | None = None,
        error: str = "",
        command: list[str] | None = None,
    ) -> None:
        item = {
            "step": step,
            "status": status,
            "started_at": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - start, 3),
            "outputs": [str(p) for p in outputs or []],
            "error": error,
            "command": command or [],
        }
        self.status.append(item)
        self.write_status_md()

    def command(self, step: str, command: list[str], outputs: list[Path]) -> bool:
        start = time.time()
        log_path = self.logs_dir / f"{step}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n\n")
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        all_outputs = [log_path] + outputs
        if proc.returncode == 0:
            self.record(step, "ok", start, all_outputs, command=command)
            return True
        error = f"exit_code={proc.returncode}; see {log_path}"
        self.record(step, "error", start, all_outputs, error=error, command=command)
        if not self.continue_on_error:
            raise RuntimeError(error)
        return False


def run_capture(command: list[str], timeout: int = 20) -> dict[str, object]:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "output": str(exc),
        }


def collect_environment(run_dir: Path, args: argparse.Namespace) -> list[Path]:
    start = time.time()
    info: dict[str, object] = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "benchmark_root": str(args.benchmark_root),
        "script_dir": str(args.script_dir),
        "args": vars(args),
        "environment": {
            key: os.environ.get(key, "")
            for key in [
                "LD_LIBRARY_PATH",
                "ROCM_PATH",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "TORCH_HOME",
                "HF_HOME",
                "HF_ENDPOINT",
                "HF_DATASETS_CACHE",
                "TRANSFORMERS_CACHE",
            ]
        },
        "commands": {},
    }
    commands = {
        "uname": ["uname", "-a"],
        "os_release": ["bash", "-lc", "cat /etc/os-release"],
        "git_head": ["git", "-C", str(args.script_dir), "rev-parse", "HEAD"],
        "git_status": ["git", "-C", str(args.script_dir), "status", "--short"],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "rocminfo": ["bash", "-lc", "rocminfo | grep -Ei 'Agent|Name:|Marketing Name:|gfx|Device Type:' | head -120"],
        "rocm_smi": ["bash", "-lc", "rocm-smi --showproductname --showdriverversion 2>&1 | head -120"],
        "torch_runtime": [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({"
                "'torch': torch.__version__, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device_count': torch.cuda.device_count(), "
                "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''"
                "}, ensure_ascii=False))"
            ),
        ],
        "onnxruntime_runtime": [
            sys.executable,
            "-c",
            (
                "import json, onnxruntime as ort; "
                "print(json.dumps({"
                "'onnxruntime': ort.__version__, "
                "'providers': ort.get_available_providers()"
                "}, ensure_ascii=False))"
            ),
        ],
    }
    for name, command in commands.items():
        info["commands"][name] = run_capture(command, timeout=60 if name == "pip_freeze" else 20)

    metadata_path = run_dir / "metadata.json"
    write_json(metadata_path, info)
    log_path = run_dir / "logs" / "01_environment.log"
    log_path.write_text(json.dumps(json_safe(info), indent=2, ensure_ascii=False), encoding="utf-8")
    return [metadata_path, log_path]


def parse_profile_name(path: Path) -> dict[str, str]:
    match = PROFILE_RE.search(path.name)
    if not match:
        return {"op_name": "", "shape_profile": "", "chain_len": "", "batch_size": "", "repeat_id": ""}
    return {
        "op_name": match.group("op"),
        "shape_profile": match.group("shape"),
        "chain_len": match.group("chain"),
        "batch_size": match.group("bs"),
        "repeat_id": match.group("rep"),
    }


def summarize_profile(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    meta = parse_profile_name(path)

    session_init: list[float] = []
    model_runs: list[float] = []
    node_durs: list[float] = []
    kernel_durs: list[float] = []
    providers: Counter[str] = Counter()
    cats: Counter[str] = Counter()
    top_node_ms = 0.0
    top_node_name = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        cats[cat] += 1
        if "dur" not in event:
            continue
        dur_ms = to_float(event.get("dur")) / 1000.0
        args = event.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if cat == "Session" and name == "session_initialization":
            session_init.append(dur_ms)
        elif cat == "Session" and name == "model_run":
            model_runs.append(dur_ms)
        elif cat == "Node":
            node_durs.append(dur_ms)
            provider = str(args.get("provider", ""))
            if provider:
                providers[provider] += 1
            if "kernel" in name.lower():
                kernel_durs.append(dur_ms)
            if dur_ms > top_node_ms:
                top_node_ms = dur_ms
                top_node_name = name

    row: dict[str, object] = {
        **meta,
        "json_file": path.name,
        "json_path": str(path),
        "event_count": len(events),
        "session_event_count": cats.get("Session", 0),
        "node_event_count": len(node_durs),
        "provider": ";".join(f"{k}:{v}" for k, v in providers.most_common()) or "unknown",
        "has_cpu_fallback": "yes" if any("CPUExecutionProvider" in k for k in providers) else "no",
        "session_init_ms": round(sum(session_init), 6),
        "first_model_run_ms": round(model_runs[0], 6) if model_runs else 0.0,
        "model_run_mean_ms": round(mean(model_runs), 6) if model_runs else 0.0,
        "model_run_p95_ms": round(pct(model_runs, 0.95), 6),
        "node_total_ms": round(sum(node_durs), 6),
        "node_mean_ms": round(mean(node_durs), 6) if node_durs else 0.0,
        "node_p95_ms": round(pct(node_durs, 0.95), 6),
        "kernel_total_ms": round(sum(kernel_durs), 6),
        "kernel_mean_ms": round(mean(kernel_durs), 6) if kernel_durs else 0.0,
        "kernel_p95_ms": round(pct(kernel_durs, 0.95), 6),
        "top_node_ms": round(top_node_ms, 6),
        "top_node_name": top_node_name,
    }
    return row


def summarize_profiles(profile_dir: Path, output_csv: Path, output_json: Path) -> list[dict[str, object]]:
    fields = [
        "op_name",
        "batch_size",
        "shape_profile",
        "chain_len",
        "repeat_id",
        "json_file",
        "event_count",
        "session_event_count",
        "node_event_count",
        "provider",
        "has_cpu_fallback",
        "session_init_ms",
        "first_model_run_ms",
        "model_run_mean_ms",
        "model_run_p95_ms",
        "node_total_ms",
        "node_mean_ms",
        "node_p95_ms",
        "kernel_total_ms",
        "kernel_mean_ms",
        "kernel_p95_ms",
        "top_node_ms",
        "top_node_name",
        "json_path",
    ]
    rows: list[dict[str, object]] = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            rows.append(summarize_profile(path))
        except Exception as exc:
            rows.append({
                "json_file": path.name,
                "json_path": str(path),
                "provider": "parse_error",
                "top_node_name": str(exc),
            })
    write_csv(output_csv, rows, fields)
    write_json(output_json, rows)
    return rows


def operator_pairs(operator_csv: Path) -> list[dict[str, object]]:
    rows = read_csv(operator_csv)
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("op_name", ""),
            row.get("batch_size", ""),
            row.get("shape_profile", ""),
            row.get("chain_len", ""),
        )
        grouped[key][row.get("backend", "")] = row

    out: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda x: (x[0], int(x[1] or 0), x[2], int(x[3] or 0))):
        item = grouped[key]
        torch_row = item.get("torch_rocm")
        ort_row = item.get("onnxruntime")
        if not torch_row or not ort_row:
            continue
        torch_ms = to_float(torch_row.get("latency_per_op_mean_ms"), to_float(torch_row.get("latency_mean_ms")))
        ort_ms = to_float(ort_row.get("latency_per_op_mean_ms"), to_float(ort_row.get("latency_mean_ms")))
        out.append({
            "op_name": key[0],
            "batch_size": key[1],
            "shape_profile": key[2],
            "chain_len": key[3],
            "torch_per_op_mean_ms": round(torch_ms, 6),
            "ort_per_op_mean_ms": round(ort_ms, 6),
            "ort_div_torch": round(ort_ms / torch_ms, 6) if torch_ms else 0.0,
            "torch_p95_ms": torch_row.get("latency_p95_ms", ""),
            "ort_p95_ms": ort_row.get("latency_p95_ms", ""),
            "provider": ort_row.get("provider", ""),
            "profile_path": ort_row.get("profile_path", ""),
        })
    return out


def summarize_operator(operator_csv: Path, output_csv: Path) -> list[dict[str, object]]:
    fields = [
        "op_name",
        "batch_size",
        "shape_profile",
        "chain_len",
        "torch_per_op_mean_ms",
        "ort_per_op_mean_ms",
        "ort_div_torch",
        "torch_p95_ms",
        "ort_p95_ms",
        "provider",
        "profile_path",
    ]
    rows = operator_pairs(operator_csv)
    write_csv(output_csv, rows, fields)
    return rows


def write_text_output(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def strip_json_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


def unique_run_dir(runs_dir: Path, run_id: str) -> tuple[str, Path]:
    base = run_id.strip() or now_label()
    candidate = runs_dir / base
    if not candidate.exists():
        return base, candidate
    idx = 2
    while True:
        next_id = f"{base}_{idx}"
        candidate = runs_dir / next_id
        if not candidate.exists():
            return next_id, candidate
        idx += 1


def compare_operator(current_csv: Path, previous_csv: Path, output_csv: Path, threshold: float) -> list[dict[str, object]]:
    current = read_csv(current_csv)
    previous = read_csv(previous_csv)

    def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
        return (
            row.get("op_name", ""),
            row.get("backend", ""),
            row.get("batch_size", ""),
            row.get("shape_profile", ""),
            row.get("chain_len", ""),
        )

    prev_ok = {key(row): row for row in previous if row.get("status") == "ok"}
    fields = [
        "op_name",
        "backend",
        "batch_size",
        "shape_profile",
        "chain_len",
        "previous_ms",
        "current_ms",
        "delta_ms",
        "change_pct",
        "status",
    ]
    rows: list[dict[str, object]] = []
    for row in current:
        if row.get("status") != "ok":
            continue
        old = prev_ok.get(key(row))
        if not old:
            continue
        old_ms = to_float(old.get("latency_per_op_mean_ms"), to_float(old.get("latency_mean_ms")))
        new_ms = to_float(row.get("latency_per_op_mean_ms"), to_float(row.get("latency_mean_ms")))
        change = ((new_ms - old_ms) / old_ms * 100.0) if old_ms else 0.0
        rows.append({
            "op_name": row.get("op_name", ""),
            "backend": row.get("backend", ""),
            "batch_size": row.get("batch_size", ""),
            "shape_profile": row.get("shape_profile", ""),
            "chain_len": row.get("chain_len", ""),
            "previous_ms": round(old_ms, 6),
            "current_ms": round(new_ms, 6),
            "delta_ms": round(new_ms - old_ms, 6),
            "change_pct": round(change, 3),
            "status": "regression" if change > threshold else "ok",
        })
    write_csv(output_csv, rows, fields)
    return rows


def copy_new_profiles(source_dir: Path, dest_dir: Path, start_time: float) -> list[Path]:
    ensure_dir(dest_dir)
    copied: list[Path] = []
    if not source_dir.exists():
        return copied
    for path in source_dir.glob("*.json"):
        if path.stat().st_mtime >= start_time - 1:
            dest = dest_dir / path.name
            shutil.copy2(path, dest)
            copied.append(dest)
    return copied


def write_profile_collect_summary(path: Path, source_dir: Path, dest_dir: Path, copied: list[Path]) -> Path:
    lines = [
        "Profile collection summary",
        "",
        f"source_dir: {source_dir}",
        f"dest_dir: {dest_dir}",
        f"copied_count: {len(copied)}",
        "",
        "copied_files:",
    ]
    for item in copied:
        lines.append(f"- {item}")
    return write_text_output(path, "\n".join(lines) + "\n")


def write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    status: list[dict[str, object]],
    baseline_csv: Path,
    operator_csv: Path,
    operator_summary: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
) -> Path:
    baseline_rows = read_csv(baseline_csv)
    operator_rows = read_csv(operator_csv)
    baseline_errors = [r for r in baseline_rows if r.get("status") != "ok"]
    operator_errors = [r for r in operator_rows if r.get("status") != "ok"]
    cpu_fallback = [r for r in profile_rows if r.get("has_cpu_fallback") == "yes"]
    regressions = [r for r in regression_rows if r.get("status") == "regression"]
    best_ort = sorted(operator_summary, key=lambda r: to_float(r.get("ort_div_torch")))[:8]
    worst_ort = sorted(operator_summary, key=lambda r: to_float(r.get("ort_div_torch")), reverse=True)[:8]

    lines = [
        f"# Benchmark Run Summary",
        "",
        f"- run_id: `{args.run_id}`",
        f"- run_dir: `{run_dir}`",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Step Outputs",
        "",
    ]
    for item in status:
        lines.append(f"- `{item['step']}`: `{item['status']}` ({item['duration_sec']} sec)")
        for output in item.get("outputs", []):
            lines.append(f"  - `{output}`")
        if item.get("error"):
            lines.append(f"  - error: `{item['error']}`")

    lines += [
        "",
        "## Result Counts",
        "",
        f"- baseline rows: `{len(baseline_rows)}`",
        f"- baseline errors: `{len(baseline_errors)}`",
        f"- operator rows: `{len(operator_rows)}`",
        f"- operator errors: `{len(operator_errors)}`",
        f"- profile json summaries: `{len(profile_rows)}`",
        f"- CPU fallback profile rows: `{len(cpu_fallback)}`",
        f"- regressions over threshold: `{len(regressions)}`",
        "",
        "## Best ORT/MIGraphX Relative Results",
        "",
        "| op | batch | chain | ORT/Torch | torch_ms | ort_ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_ort:
        lines.append(
            f"| {row.get('op_name','')} | {row.get('batch_size','')} | {row.get('chain_len','')} | "
            f"{row.get('ort_div_torch','')} | {row.get('torch_per_op_mean_ms','')} | {row.get('ort_per_op_mean_ms','')} |"
        )

    lines += [
        "",
        "## Worst ORT/MIGraphX Relative Results",
        "",
        "| op | batch | chain | ORT/Torch | torch_ms | ort_ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst_ort:
        lines.append(
            f"| {row.get('op_name','')} | {row.get('batch_size','')} | {row.get('chain_len','')} | "
            f"{row.get('ort_div_torch','')} | {row.get('torch_per_op_mean_ms','')} | {row.get('ort_per_op_mean_ms','')} |"
        )

    if regressions:
        lines += [
            "",
            "## Regressions",
            "",
            "| op | backend | batch | chain | previous_ms | current_ms | change_pct |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in regressions[:30]:
            lines.append(
                f"| {row.get('op_name','')} | {row.get('backend','')} | {row.get('batch_size','')} | "
                f"{row.get('chain_len','')} | {row.get('previous_ms','')} | {row.get('current_ms','')} | {row.get('change_pct','')} |"
            )

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full benchmark workflow with per-step outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Benchmark workflow config JSON file.")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--script-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run-id", default=now_label())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-operator", action="store_true")
    parser.add_argument("--compare-run", type=Path, default=None, help="Previous run directory or operator_results.csv")
    parser.add_argument("--regression-threshold-pct", type=float, default=10.0)

    parser.add_argument("--baseline-models", default="all")
    parser.add_argument("--baseline-backends", default="all")
    parser.add_argument("--resnet-batches", default="1,8,16,32,64")
    parser.add_argument("--bert-batches", default="1,4,8,16,32")
    parser.add_argument("--baseline-warmup", type=int, default=10)
    parser.add_argument("--baseline-iters", type=int, default=50)

    parser.add_argument("--operator-backends", default="all")
    parser.add_argument("--operator-batches", default="1,8,16")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shape-profile", choices=["standard", "large"], default="large")
    parser.add_argument("--chain-len", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=10)
    parser.add_argument("--operator-iters", type=int, default=50)
    return parser


def load_config(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(strip_json_comments(raw))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def merge_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not args.config:
        return args

    config = load_config(args.config)
    defaults = vars(parser.parse_args([]))
    merged = vars(args).copy()
    path_keys = {"benchmark_root", "script_dir", "runs_dir", "compare_run"}

    for key, value in config.items():
        if key.startswith("_"):
            continue
        if not hasattr(args, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(args, key)
        default = defaults[key]
        # Command-line values override config. If the current value still equals
        # the parser default, use the config value.
        if current == default:
            if key in path_keys and value not in ("", None):
                merged[key] = Path(str(value))
            elif key in path_keys:
                merged[key] = None if key == "compare_run" else default
            else:
                merged[key] = value

    return argparse.Namespace(**merged)


def main() -> int:
    parser = build_parser()
    args = merge_config(parser.parse_args(), parser)
    args.run_id, run_dir = unique_run_dir(args.runs_dir, str(args.run_id or ""))
    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    write_json(run_dir / "00_run_config.json", vars(args))

    runner = StepRunner(run_dir, args.continue_on_error)

    start = time.time()
    try:
        env_outputs = collect_environment(run_dir, args)
        runner.record("01_environment", "ok", start, env_outputs)
    except Exception as exc:
        runner.record("01_environment", "error", start, error=str(exc))
        if not args.continue_on_error:
            raise

    baseline_csv = run_dir / "baseline_results.csv"
    if args.skip_baseline:
        skipped = write_text_output(run_dir / "baseline_skipped.txt", "Baseline benchmark skipped by config or CLI.\n")
        runner.record("02_baseline", "skipped", time.time(), [skipped])
    else:
        baseline_script = args.script_dir / "benchmark_baselines.py"
        runner.command(
            "02_baseline",
            [
                args.python,
                str(baseline_script),
                "--models",
                args.baseline_models,
                "--backends",
                args.baseline_backends,
                "--resnet-batches",
                args.resnet_batches,
                "--bert-batches",
                args.bert_batches,
                "--seq-len",
                str(args.seq_len),
                "--warmup",
                str(args.baseline_warmup),
                "--iters",
                str(args.baseline_iters),
                "--output",
                str(baseline_csv),
            ],
            [baseline_csv],
        )

    operator_csv = run_dir / "operator_results.csv"
    profile_copy_start = time.time()
    if args.skip_operator:
        skipped = write_text_output(run_dir / "operator_skipped.txt", "Operator benchmark skipped by config or CLI.\n")
        runner.record("03_operator", "skipped", time.time(), [skipped])
    else:
        operator_script = args.script_dir / "operator_benchmark.py"
        runner.command(
            "03_operator",
            [
                args.python,
                str(operator_script),
                "--backends",
                args.operator_backends,
                "--batches",
                args.operator_batches,
                "--seq-len",
                str(args.seq_len),
                "--shape-profile",
                args.shape_profile,
                "--chain-len",
                str(args.chain_len),
                "--repeat",
                str(args.repeat),
                "--warmup",
                str(args.operator_warmup),
                "--iters",
                str(args.operator_iters),
                "--output",
                str(operator_csv),
            ],
            [operator_csv],
        )

    profile_dir = run_dir / "ort_profiles"
    source_profile_dir = args.benchmark_root / "outputs" / "ort_profiles"
    copied_profiles = copy_new_profiles(source_profile_dir, profile_dir, profile_copy_start)
    collect_summary = write_profile_collect_summary(
        run_dir / "profile_collect_summary.txt",
        source_profile_dir,
        profile_dir,
        copied_profiles,
    )
    runner.record("04_collect_profiles", "ok", time.time(), [collect_summary] + copied_profiles)

    profile_summary_csv = run_dir / "profile_summary.csv"
    profile_summary_json = run_dir / "profile_summary.json"
    start = time.time()
    profile_rows = summarize_profiles(profile_dir, profile_summary_csv, profile_summary_json)
    runner.record("05_profile_summary", "ok", start, [profile_summary_csv, profile_summary_json])

    operator_pair_csv = run_dir / "operator_pair_summary.csv"
    start = time.time()
    operator_summary = summarize_operator(operator_csv, operator_pair_csv)
    runner.record("06_operator_summary", "ok", start, [operator_pair_csv])

    regression_rows: list[dict[str, object]] = []
    regression_csv = run_dir / "regression_report.csv"
    if args.compare_run:
        previous_csv = args.compare_run
        if args.compare_run.is_dir():
            previous_csv = args.compare_run / "operator_results.csv"
        start = time.time()
        regression_rows = compare_operator(operator_csv, previous_csv, regression_csv, args.regression_threshold_pct)
        runner.record("07_regression_report", "ok", start, [regression_csv])
    else:
        skipped = write_text_output(run_dir / "regression_skipped.txt", "Regression report skipped because compare_run is not configured.\n")
        runner.record("07_regression_report", "skipped", time.time(), [skipped], error="--compare-run not provided")

    start = time.time()
    summary_path = write_summary(
        run_dir,
        args,
        runner.status,
        baseline_csv,
        operator_csv,
        operator_summary,
        profile_rows,
        regression_rows,
    )
    runner.record("08_summary", "ok", start, [summary_path])

    print(f"run_dir={run_dir}")
    print(f"summary={summary_path}")
    print(f"steps_status={run_dir / 'steps_status.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
