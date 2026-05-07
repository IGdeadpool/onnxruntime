using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace OrtProfileAnalyzer
{
    public class ProfileRecord
    {
        public string ImportTime = "";
        public string RunLabel = "";
        public string OpName = "";
        public string BatchSize = "";
        public string ShapeProfile = "";
        public string ChainLen = "";
        public string RepeatId = "";
        public string Provider = "";
        public double SessionInitMs;
        public double FirstModelRunMs;
        public double ModelRunMeanMs;
        public double ModelRunP95Ms;
        public double NodeTotalMs;
        public double NodeMeanMs;
        public double NodeP95Ms;
        public double KernelTotalMs;
        public double KernelMeanMs;
        public double KernelP95Ms;
        public int NodeEventCount;
        public double TopNodeMs;
        public string TopNodeName = "";
        public string JsonPath = "";
        public string Notes = "";
    }

    public class MainForm : Form
    {
        static readonly string[] CsvFields = new string[] {
            "import_time", "run_label", "op_name", "batch_size", "shape_profile",
            "chain_len", "repeat_id", "provider", "session_init_ms",
            "first_model_run_ms", "model_run_mean_ms", "model_run_p95_ms",
            "node_total_ms", "node_mean_ms", "node_p95_ms", "kernel_total_ms",
            "kernel_mean_ms", "kernel_p95_ms", "node_event_count",
            "top_node_ms", "top_node_name", "json_path", "notes"
        };

        readonly string historyPath;
        readonly List<ProfileRecord> records = new List<ProfileRecord>();
        readonly DataGridView grid = new DataGridView();
        readonly DataGridView compareGrid = new DataGridView();
        readonly TextBox runLabelBox = new TextBox();
        readonly TextBox notesBox = new TextBox();
        readonly TextBox filterBox = new TextBox();
        readonly Label statusLabel = new Label();

        public MainForm()
        {
            Text = "ORT Profiling Analyzer";
            Width = 1320;
            Height = 820;
            MinimumSize = new Size(1060, 650);

            historyPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ort_profile_history.csv");
            BuildUi();
            LoadHistory();
            RefreshTables();
        }

        void BuildUi()
        {
            var root = new TableLayoutPanel();
            root.Dock = DockStyle.Fill;
            root.Padding = new Padding(10);
            root.RowCount = 4;
            root.ColumnCount = 1;
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 62));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 38));
            Controls.Add(root);

            var top = new FlowLayoutPanel();
            top.Dock = DockStyle.Fill;
            top.FlowDirection = FlowDirection.LeftToRight;
            top.WrapContents = false;
            root.Controls.Add(top, 0, 0);

            runLabelBox.Width = 170;
            runLabelBox.Text = DateTime.Now.ToString("run_yyyyMMdd_HHmm");
            notesBox.Width = 300;
            top.Controls.Add(new Label { Text = "Run label", Width = 68, TextAlign = ContentAlignment.MiddleLeft });
            top.Controls.Add(runLabelBox);
            top.Controls.Add(new Label { Text = "Notes", Width = 44, TextAlign = ContentAlignment.MiddleLeft });
            top.Controls.Add(notesBox);

            AddButton(top, "Import JSON", ImportJson);
            AddButton(top, "Reload", delegate { LoadHistory(); RefreshTables(); });
            AddButton(top, "Export Compare CSV", ExportCompareCsv);
            AddButton(top, "Delete Selected", DeleteSelected);

            var filterPanel = new FlowLayoutPanel();
            filterPanel.Dock = DockStyle.Fill;
            filterPanel.FlowDirection = FlowDirection.LeftToRight;
            filterPanel.WrapContents = false;
            root.Controls.Add(filterPanel, 0, 1);
            filterPanel.Controls.Add(new Label { Text = "Filter", Width = 42, TextAlign = ContentAlignment.MiddleLeft });
            filterBox.Width = 360;
            filterBox.TextChanged += delegate { RefreshTables(); };
            filterPanel.Controls.Add(filterBox);
            statusLabel.AutoSize = true;
            statusLabel.Padding = new Padding(20, 7, 0, 0);
            filterPanel.Controls.Add(statusLabel);

            ConfigureGrid(grid);
            grid.Columns.Add("run_label", "run_label");
            grid.Columns.Add("op_name", "op_name");
            grid.Columns.Add("batch_size", "batch_size");
            grid.Columns.Add("chain_len", "chain_len");
            grid.Columns.Add("provider", "provider");
            grid.Columns.Add("model_run_mean_ms", "model_run_mean_ms");
            grid.Columns.Add("model_run_p95_ms", "model_run_p95_ms");
            grid.Columns.Add("node_mean_ms", "node_mean_ms");
            grid.Columns.Add("kernel_mean_ms", "kernel_mean_ms");
            grid.Columns.Add("session_init_ms", "session_init_ms");
            grid.Columns.Add("top_node_ms", "top_node_ms");
            grid.Columns.Add("notes", "notes");
            grid.Columns["provider"].Width = 230;
            grid.Columns["notes"].Width = 220;
            root.Controls.Add(grid, 0, 2);

            var group = new GroupBox();
            group.Text = "Comparison by op + batch";
            group.Dock = DockStyle.Fill;
            root.Controls.Add(group, 0, 3);
            ConfigureGrid(compareGrid);
            compareGrid.Dock = DockStyle.Fill;
            compareGrid.Columns.Add("op_name", "op_name");
            compareGrid.Columns.Add("batch_size", "batch_size");
            compareGrid.Columns.Add("baseline_label", "baseline_label");
            compareGrid.Columns.Add("baseline_ms", "baseline_ms");
            compareGrid.Columns.Add("latest_label", "latest_label");
            compareGrid.Columns.Add("latest_ms", "latest_ms");
            compareGrid.Columns.Add("delta_ms", "delta_ms");
            compareGrid.Columns.Add("change_pct", "change_pct");
            group.Controls.Add(compareGrid);
        }

        void AddButton(FlowLayoutPanel panel, string text, EventHandler handler)
        {
            var button = new Button();
            button.Text = text;
            button.Width = 130;
            button.Height = 28;
            button.Click += handler;
            panel.Controls.Add(button);
        }

        void ConfigureGrid(DataGridView view)
        {
            view.Dock = DockStyle.Fill;
            view.AllowUserToAddRows = false;
            view.AllowUserToDeleteRows = false;
            view.ReadOnly = true;
            view.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            view.MultiSelect = true;
            view.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
            view.RowHeadersVisible = false;
        }

        void ImportJson(object sender, EventArgs e)
        {
            var dialog = new OpenFileDialog();
            dialog.Title = "Select ORT profiling JSON files";
            dialog.Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*";
            dialog.Multiselect = true;
            if (dialog.ShowDialog(this) != DialogResult.OK) return;

            int ok = 0;
            var errors = new List<string>();
            foreach (string path in dialog.FileNames)
            {
                try
                {
                    records.Add(AnalyzeJson(path, runLabelBox.Text.Trim(), notesBox.Text.Trim()));
                    ok++;
                }
                catch (Exception ex)
                {
                    errors.Add(Path.GetFileName(path) + ": " + ex.Message);
                }
            }
            SaveHistory();
            RefreshTables();
            if (errors.Count > 0)
                MessageBox.Show(this, "Imported " + ok + " files.\nFailed:\n" + string.Join("\n", errors.Take(8).ToArray()), "Import finished");
            else
                MessageBox.Show(this, "Imported " + ok + " files.", "Import finished");
        }

        ProfileRecord AnalyzeJson(string path, string runLabel, string notes)
        {
            string text = File.ReadAllText(path, Encoding.UTF8);
            var serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = int.MaxValue;
            object parsed = serializer.DeserializeObject(text);
            object[] events = ToEventArray(parsed);

            var rec = ParseFileName(path);
            rec.ImportTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            rec.RunLabel = string.IsNullOrWhiteSpace(runLabel) ? DateTime.Now.ToString("run_yyyyMMdd_HHmm") : runLabel;
            rec.JsonPath = path;
            rec.Notes = notes;

            var sessionInit = new List<double>();
            var modelRuns = new List<double>();
            var nodeDurs = new List<double>();
            var kernelDurs = new List<double>();
            var providers = new Dictionary<string, int>();

            foreach (object obj in events)
            {
                var ev = obj as Dictionary<string, object>;
                if (ev == null || !ev.ContainsKey("dur")) continue;

                double durMs = Convert.ToDouble(ev["dur"], CultureInfo.InvariantCulture) / 1000.0;
                string name = GetString(ev, "name");
                string cat = GetString(ev, "cat");
                var args = ev.ContainsKey("args") ? ev["args"] as Dictionary<string, object> : null;

                if (cat == "Session" && name == "session_initialization")
                    sessionInit.Add(durMs);
                else if (cat == "Session" && name == "model_run")
                    modelRuns.Add(durMs);
                else if (cat == "Node")
                {
                    nodeDurs.Add(durMs);
                    string provider = args == null ? "" : GetString(args, "provider");
                    if (!string.IsNullOrEmpty(provider))
                    {
                        if (!providers.ContainsKey(provider)) providers[provider] = 0;
                        providers[provider]++;
                    }
                    if (name.ToLowerInvariant().Contains("kernel"))
                        kernelDurs.Add(durMs);
                    if (durMs > rec.TopNodeMs)
                    {
                        rec.TopNodeMs = durMs;
                        rec.TopNodeName = name;
                    }
                }
            }

            rec.Provider = providers.Count == 0
                ? "unknown"
                : string.Join(";", providers.OrderByDescending(kv => kv.Value).Select(kv => kv.Key + ":" + kv.Value).ToArray());
            rec.SessionInitMs = sessionInit.Sum();
            rec.FirstModelRunMs = modelRuns.Count > 0 ? modelRuns[0] : 0;
            rec.ModelRunMeanMs = Mean(modelRuns);
            rec.ModelRunP95Ms = Percentile(modelRuns, 0.95);
            rec.NodeTotalMs = nodeDurs.Sum();
            rec.NodeMeanMs = Mean(nodeDurs);
            rec.NodeP95Ms = Percentile(nodeDurs, 0.95);
            rec.KernelTotalMs = kernelDurs.Sum();
            rec.KernelMeanMs = Mean(kernelDurs);
            rec.KernelP95Ms = Percentile(kernelDurs, 0.95);
            rec.NodeEventCount = nodeDurs.Count;
            return rec;
        }

        object[] ToEventArray(object parsed)
        {
            object[] arr = parsed as object[];
            if (arr != null) return arr;
            var dict = parsed as Dictionary<string, object>;
            if (dict != null && dict.ContainsKey("traceEvents"))
            {
                arr = dict["traceEvents"] as object[];
                if (arr != null) return arr;
            }
            throw new InvalidDataException("JSON is not an ORT trace event array.");
        }

        ProfileRecord ParseFileName(string path)
        {
            var rec = new ProfileRecord();
            string name = Path.GetFileNameWithoutExtension(path);
            var m = Regex.Match(name, @"ort_(.+?)_([^_]+)_chain(\d+)_bs(\d+)_rep(\d+)", RegexOptions.IgnoreCase);
            if (m.Success)
            {
                rec.OpName = m.Groups[1].Value;
                rec.ShapeProfile = m.Groups[2].Value;
                rec.ChainLen = m.Groups[3].Value;
                rec.BatchSize = m.Groups[4].Value;
                rec.RepeatId = m.Groups[5].Value;
            }
            return rec;
        }

        void RefreshTables()
        {
            string f = filterBox.Text.Trim().ToLowerInvariant();
            var visible = records.Where(r => string.IsNullOrEmpty(f) || RecordText(r).ToLowerInvariant().Contains(f)).ToList();

            grid.Rows.Clear();
            foreach (var r in visible)
            {
                int idx = grid.Rows.Add(r.RunLabel, r.OpName, r.BatchSize, r.ChainLen, r.Provider,
                    F(r.ModelRunMeanMs), F(r.ModelRunP95Ms), F(r.NodeMeanMs), F(r.KernelMeanMs),
                    F(r.SessionInitMs), F(r.TopNodeMs), r.Notes);
                grid.Rows[idx].Tag = r;
            }
            statusLabel.Text = "records: " + visible.Count + " / " + records.Count + "    history: " + historyPath;
            RefreshCompare(visible);
        }

        void RefreshCompare(List<ProfileRecord> visible)
        {
            compareGrid.Rows.Clear();
            var groups = visible.GroupBy(r => (r.OpName ?? "") + "\t" + (r.BatchSize ?? ""))
                .OrderBy(g => g.Key);
            foreach (var g in groups)
            {
                var list = g.OrderBy(r => r.ImportTime).ToList();
                if (list.Count < 2) continue;
                var baseline = list.First();
                var latest = list.Last();
                double delta = latest.ModelRunMeanMs - baseline.ModelRunMeanMs;
                double pct = baseline.ModelRunMeanMs == 0 ? 0 : delta / baseline.ModelRunMeanMs * 100.0;
                compareGrid.Rows.Add(latest.OpName, latest.BatchSize, baseline.RunLabel, F(baseline.ModelRunMeanMs),
                    latest.RunLabel, F(latest.ModelRunMeanMs), F(delta), pct.ToString("0.00", CultureInfo.InvariantCulture) + "%");
            }
        }

        void DeleteSelected(object sender, EventArgs e)
        {
            var toDelete = new HashSet<ProfileRecord>();
            foreach (DataGridViewRow row in grid.SelectedRows)
            {
                var rec = row.Tag as ProfileRecord;
                if (rec != null) toDelete.Add(rec);
            }
            if (toDelete.Count == 0) return;
            foreach (var rec in toDelete) records.Remove(rec);
            SaveHistory();
            RefreshTables();
        }

        void ExportCompareCsv(object sender, EventArgs e)
        {
            var dialog = new SaveFileDialog();
            dialog.Title = "Export comparison CSV";
            dialog.Filter = "CSV files (*.csv)|*.csv";
            dialog.FileName = "ort_profile_compare.csv";
            if (dialog.ShowDialog(this) != DialogResult.OK) return;

            using (var w = new StreamWriter(dialog.FileName, false, new UTF8Encoding(true)))
            {
                var names = compareGrid.Columns.Cast<DataGridViewColumn>().Select(c => c.Name).ToArray();
                w.WriteLine(string.Join(",", names.Select(CsvEscape).ToArray()));
                foreach (DataGridViewRow row in compareGrid.Rows)
                {
                    var vals = names.Select((n, i) => CsvEscape(Convert.ToString(row.Cells[i].Value))).ToArray();
                    w.WriteLine(string.Join(",", vals));
                }
            }
            MessageBox.Show(this, dialog.FileName, "Export finished");
        }

        void LoadHistory()
        {
            records.Clear();
            if (!File.Exists(historyPath)) return;
            var lines = File.ReadAllLines(historyPath, Encoding.UTF8);
            if (lines.Length < 2) return;
            for (int i = 1; i < lines.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(lines[i])) continue;
                var vals = CsvSplit(lines[i]);
                var d = new Dictionary<string, string>();
                for (int j = 0; j < CsvFields.Length && j < vals.Count; j++) d[CsvFields[j]] = vals[j];
                records.Add(FromDict(d));
            }
        }

        void SaveHistory()
        {
            using (var w = new StreamWriter(historyPath, false, new UTF8Encoding(true)))
            {
                w.WriteLine(string.Join(",", CsvFields.Select(CsvEscape).ToArray()));
                foreach (var r in records)
                {
                    w.WriteLine(string.Join(",", ToFields(r).Select(CsvEscape).ToArray()));
                }
            }
        }

        ProfileRecord FromDict(Dictionary<string, string> d)
        {
            var r = new ProfileRecord();
            r.ImportTime = D(d, "import_time");
            r.RunLabel = D(d, "run_label");
            r.OpName = D(d, "op_name");
            r.BatchSize = D(d, "batch_size");
            r.ShapeProfile = D(d, "shape_profile");
            r.ChainLen = D(d, "chain_len");
            r.RepeatId = D(d, "repeat_id");
            r.Provider = D(d, "provider");
            r.SessionInitMs = P(d, "session_init_ms");
            r.FirstModelRunMs = P(d, "first_model_run_ms");
            r.ModelRunMeanMs = P(d, "model_run_mean_ms");
            r.ModelRunP95Ms = P(d, "model_run_p95_ms");
            r.NodeTotalMs = P(d, "node_total_ms");
            r.NodeMeanMs = P(d, "node_mean_ms");
            r.NodeP95Ms = P(d, "node_p95_ms");
            r.KernelTotalMs = P(d, "kernel_total_ms");
            r.KernelMeanMs = P(d, "kernel_mean_ms");
            r.KernelP95Ms = P(d, "kernel_p95_ms");
            int n;
            int.TryParse(D(d, "node_event_count"), out n);
            r.NodeEventCount = n;
            r.TopNodeMs = P(d, "top_node_ms");
            r.TopNodeName = D(d, "top_node_name");
            r.JsonPath = D(d, "json_path");
            r.Notes = D(d, "notes");
            return r;
        }

        string[] ToFields(ProfileRecord r)
        {
            return new string[] {
                r.ImportTime, r.RunLabel, r.OpName, r.BatchSize, r.ShapeProfile,
                r.ChainLen, r.RepeatId, r.Provider, F(r.SessionInitMs),
                F(r.FirstModelRunMs), F(r.ModelRunMeanMs), F(r.ModelRunP95Ms),
                F(r.NodeTotalMs), F(r.NodeMeanMs), F(r.NodeP95Ms), F(r.KernelTotalMs),
                F(r.KernelMeanMs), F(r.KernelP95Ms), r.NodeEventCount.ToString(),
                F(r.TopNodeMs), r.TopNodeName, r.JsonPath, r.Notes
            };
        }

        static string RecordText(ProfileRecord r)
        {
            return string.Join(" ", new string[] { r.RunLabel, r.OpName, r.BatchSize, r.ChainLen, r.Provider, r.Notes, r.JsonPath });
        }

        static string GetString(Dictionary<string, object> d, string key)
        {
            if (!d.ContainsKey(key) || d[key] == null) return "";
            return Convert.ToString(d[key], CultureInfo.InvariantCulture);
        }

        static double Mean(List<double> values)
        {
            return values.Count == 0 ? 0 : values.Average();
        }

        static double Percentile(List<double> values, double pct)
        {
            if (values.Count == 0) return 0;
            var sorted = values.OrderBy(x => x).ToList();
            if (sorted.Count == 1) return sorted[0];
            double pos = (sorted.Count - 1) * pct;
            int lo = (int)Math.Floor(pos);
            int hi = Math.Min(lo + 1, sorted.Count - 1);
            double frac = pos - lo;
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        }

        static string F(double v)
        {
            return v.ToString("0.000000", CultureInfo.InvariantCulture);
        }

        static string D(Dictionary<string, string> d, string key)
        {
            return d.ContainsKey(key) ? d[key] : "";
        }

        static double P(Dictionary<string, string> d, string key)
        {
            double v;
            return double.TryParse(D(d, key), NumberStyles.Float, CultureInfo.InvariantCulture, out v) ? v : 0;
        }

        static string CsvEscape(string s)
        {
            if (s == null) s = "";
            if (s.Contains("\"") || s.Contains(",") || s.Contains("\n") || s.Contains("\r"))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        static List<string> CsvSplit(string line)
        {
            var vals = new List<string>();
            var sb = new StringBuilder();
            bool q = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (q)
                {
                    if (c == '"' && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else if (c == '"') q = false;
                    else sb.Append(c);
                }
                else
                {
                    if (c == '"') q = true;
                    else if (c == ',')
                    {
                        vals.Add(sb.ToString());
                        sb.Length = 0;
                    }
                    else sb.Append(c);
                }
            }
            vals.Add(sb.ToString());
            return vals;
        }

        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }
}
