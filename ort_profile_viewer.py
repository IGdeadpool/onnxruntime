import csv
import json
import os
import re
import statistics
import sys
import tkinter as tk
from collections import Counter
from datetime import datetime
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "ORT Profiling Analyzer"
FIELDS = [
    "import_time",
    "run_label",
    "op_name",
    "batch_size",
    "shape_profile",
    "chain_len",
    "repeat_id",
    "provider",
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
    "node_event_count",
    "top_node_ms",
    "top_node_name",
    "json_path",
    "notes",
]


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


HISTORY_PATH = os.path.join(app_dir(), "ort_profile_history.csv")


PROFILE_RE = re.compile(
    r"ort_(?P<op>.+?)_(?P<shape>[^_]+)_chain(?P<chain>\d+)_bs(?P<bs>\d+)_rep(?P<rep>\d+)",
    re.IGNORECASE,
)


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def mean(values):
    return statistics.mean(values) if values else 0.0


def ms(us):
    return float(us) / 1000.0


def parse_name(path):
    name = os.path.basename(path)
    m = PROFILE_RE.search(name)
    if not m:
        return {
            "op_name": "",
            "batch_size": "",
            "shape_profile": "",
            "chain_len": "",
            "repeat_id": "",
        }
    return {
        "op_name": m.group("op"),
        "batch_size": m.group("bs"),
        "shape_profile": m.group("shape"),
        "chain_len": m.group("chain"),
        "repeat_id": m.group("rep"),
    }


def load_events(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("traceEvents", [])


def analyze_json(path, run_label, notes):
    events = load_events(path)
    meta = parse_name(path)

    session_init = []
    model_runs = []
    node_durs = []
    kernel_durs = []
    providers = Counter()
    top_node = (0.0, "")

    for event in events:
        dur = event.get("dur")
        if dur is None:
            continue
        dur_ms = ms(dur)
        name = event.get("name", "")
        cat = event.get("cat", "")
        args = event.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if cat == "Session" and name == "session_initialization":
            session_init.append(dur_ms)
        elif cat == "Session" and name == "model_run":
            model_runs.append(dur_ms)
        elif cat == "Node":
            node_durs.append(dur_ms)
            provider = args.get("provider", "")
            if provider:
                providers[provider] += 1
            if "kernel_time" in name.lower() or "kernel" in name.lower():
                kernel_durs.append(dur_ms)
            if dur_ms > top_node[0]:
                top_node = (dur_ms, name)

    provider = ";".join(f"{k}:{v}" for k, v in providers.most_common())
    if not provider:
        provider = "unknown"

    row = {
        "import_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_label": run_label,
        "op_name": meta["op_name"],
        "batch_size": meta["batch_size"],
        "shape_profile": meta["shape_profile"],
        "chain_len": meta["chain_len"],
        "repeat_id": meta["repeat_id"],
        "provider": provider,
        "session_init_ms": f"{sum(session_init):.6f}",
        "first_model_run_ms": f"{(model_runs[0] if model_runs else 0.0):.6f}",
        "model_run_mean_ms": f"{mean(model_runs):.6f}",
        "model_run_p95_ms": f"{percentile(model_runs, 0.95):.6f}",
        "node_total_ms": f"{sum(node_durs):.6f}",
        "node_mean_ms": f"{mean(node_durs):.6f}",
        "node_p95_ms": f"{percentile(node_durs, 0.95):.6f}",
        "kernel_total_ms": f"{sum(kernel_durs):.6f}",
        "kernel_mean_ms": f"{mean(kernel_durs):.6f}",
        "kernel_p95_ms": f"{percentile(kernel_durs, 0.95):.6f}",
        "node_event_count": str(len(node_durs)),
        "top_node_ms": f"{top_node[0]:.6f}",
        "top_node_name": top_node[1],
        "json_path": path,
        "notes": notes,
    }
    return row


def read_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_history(rows):
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x760")
        self.minsize(1040, 620)
        self.rows = read_history()
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Run label").pack(side=tk.LEFT)
        self.run_label = ttk.Entry(top, width=20)
        self.run_label.insert(0, datetime.now().strftime("run_%Y%m%d_%H%M"))
        self.run_label.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(top, text="Notes").pack(side=tk.LEFT)
        self.notes = ttk.Entry(top, width=36)
        self.notes.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Button(top, text="Import JSON", command=self.import_json).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Reload", command=self.reload_history).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Export Summary CSV", command=self.export_summary).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=3)

        filter_bar = ttk.Frame(root)
        filter_bar.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(filter_bar, text="Filter").pack(side=tk.LEFT)
        self.filter_text = ttk.Entry(filter_bar, width=42)
        self.filter_text.pack(side=tk.LEFT, padx=6)
        self.filter_text.bind("<KeyRelease>", lambda _event: self.refresh())
        self.summary_label = ttk.Label(filter_bar, text="")
        self.summary_label.pack(side=tk.RIGHT)

        columns = [
            "run_label",
            "op_name",
            "batch_size",
            "chain_len",
            "provider",
            "model_run_mean_ms",
            "model_run_p95_ms",
            "node_mean_ms",
            "kernel_mean_ms",
            "session_init_ms",
            "top_node_ms",
            "notes",
        ]
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=16)
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            width = 105
            if col in ("provider", "notes"):
                width = 190
            if col == "run_label":
                width = 150
            self.tree.column(col, width=width, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.LabelFrame(root, text="Comparison by op + batch")
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        cmp_cols = [
            "op_name",
            "batch_size",
            "baseline_label",
            "baseline_ms",
            "latest_label",
            "latest_ms",
            "delta_ms",
            "change_pct",
        ]
        self.compare_tree = ttk.Treeview(bottom, columns=cmp_cols, show="headings", height=8)
        for col in cmp_cols:
            self.compare_tree.heading(col, text=col)
            self.compare_tree.column(col, width=130, anchor=tk.W)
        self.compare_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def visible_rows(self):
        text = self.filter_text.get().strip().lower()
        if not text:
            return list(self.rows)
        return [
            row for row in self.rows
            if text in " ".join(str(row.get(f, "")) for f in FIELDS).lower()
        ]

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.visible_rows()
        for idx, row in enumerate(rows):
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=[
                    row.get("run_label", ""),
                    row.get("op_name", ""),
                    row.get("batch_size", ""),
                    row.get("chain_len", ""),
                    row.get("provider", ""),
                    row.get("model_run_mean_ms", ""),
                    row.get("model_run_p95_ms", ""),
                    row.get("node_mean_ms", ""),
                    row.get("kernel_mean_ms", ""),
                    row.get("session_init_ms", ""),
                    row.get("top_node_ms", ""),
                    row.get("notes", ""),
                ],
            )
        self.summary_label.config(text=f"records: {len(rows)} / {len(self.rows)}")
        self.refresh_compare(rows)

    def refresh_compare(self, rows):
        for item in self.compare_tree.get_children():
            self.compare_tree.delete(item)

        groups = {}
        for row in rows:
            key = (row.get("op_name", ""), row.get("batch_size", ""))
            groups.setdefault(key, []).append(row)

        for (op, bs), group in sorted(groups.items()):
            if len(group) < 2:
                continue
            group = sorted(group, key=lambda r: r.get("import_time", ""))
            baseline = group[0]
            latest = group[-1]
            b = float(baseline.get("model_run_mean_ms") or 0)
            l = float(latest.get("model_run_mean_ms") or 0)
            delta = l - b
            pct = (delta / b * 100.0) if b else 0.0
            self.compare_tree.insert(
                "",
                tk.END,
                values=[
                    op,
                    bs,
                    baseline.get("run_label", ""),
                    f"{b:.6f}",
                    latest.get("run_label", ""),
                    f"{l:.6f}",
                    f"{delta:.6f}",
                    f"{pct:.2f}%",
                ],
            )

    def import_json(self):
        paths = filedialog.askopenfilenames(
            title="Select ORT profiling JSON files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not paths:
            return
        run_label = self.run_label.get().strip() or datetime.now().strftime("run_%Y%m%d_%H%M")
        notes = self.notes.get().strip()
        added = 0
        failed = []
        for path in paths:
            try:
                self.rows.append(analyze_json(path, run_label, notes))
                added += 1
            except Exception as exc:
                failed.append(f"{os.path.basename(path)}: {exc}")
        write_history(self.rows)
        self.refresh()
        if failed:
            messagebox.showwarning("Import finished", f"Imported {added} files.\nFailed:\n" + "\n".join(failed[:8]))
        else:
            messagebox.showinfo("Import finished", f"Imported {added} files.")

    def reload_history(self):
        self.rows = read_history()
        self.refresh()

    def export_summary(self):
        path = filedialog.asksaveasfilename(
            title="Export summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        rows = []
        for item in self.compare_tree.get_children():
            rows.append(dict(zip(self.compare_tree["columns"], self.compare_tree.item(item)["values"])))
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.compare_tree["columns"]))
            writer.writeheader()
            writer.writerows(rows)
        messagebox.showinfo("Export finished", path)

    def delete_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        visible = self.visible_rows()
        remove = {id(visible[int(i)]) for i in selected}
        self.rows = [row for row in self.rows if id(row) not in remove]
        write_history(self.rows)
        self.refresh()

    def sort_by(self, col):
        def key(row):
            value = row.get(col, "")
            try:
                return float(value)
            except ValueError:
                return value
        self.rows.sort(key=key)
        write_history(self.rows)
        self.refresh()


if __name__ == "__main__":
    App().mainloop()
