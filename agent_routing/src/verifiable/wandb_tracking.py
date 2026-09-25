"""Optional W&B metrics and explicitly enabled text tables with stage identities.

Enable explicitly with MARGENT_WANDB_MODE=online or offline. Local experiment
records remain authoritative. MARGENT_WANDB_TEXT=1 enables output tables;
model weights are never uploaded by this module.
"""
from __future__ import annotations

from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


def tracking_mode():
    mode = os.environ.get("MARGENT_WANDB_MODE", "disabled").lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("MARGENT_WANDB_MODE must be online, offline, or disabled")
    return mode


def text_tracking_enabled():
    return os.environ.get("MARGENT_WANDB_TEXT", "0").lower() in {"1", "true", "yes"}


def scalar_metrics(values, prefix=""):
    """Flatten finite scalars only. Text, examples, labels and lists stay local."""
    out = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            out[name] = value
    return out


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def _config(value):
    """Do not send authentication fields if future experiment configs add them."""
    if not isinstance(value, dict):
        return value
    return {k: "[redacted]" if re.search(r"api.?key|secret|password|credential|access.?token", k, re.I)
            else _config(v) if isinstance(v, dict)
            else [_config(x) for x in v] if isinstance(v, list) else v
            for k, v in value.items()}


class WandbTracker:
    def __init__(self, root, stage, attempt):
        self.run = None
        self.root = Path(root).resolve()
        self.stage, self.attempt = stage, attempt
        self.mode = tracking_mode()
        self.text_enabled = text_tracking_enabled()
        self.tables, self.table_counts, self.pending_tables = {}, {}, set()
        self.last_table_flush = None

    def start(self):
        if self.mode == "disabled":
            return
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("W&B requested but not installed; install requirements-math.txt") from exc
        self.sdk = wandb
        if self.text_enabled:
            self.table_max_rows = int(os.environ.get("MARGENT_WANDB_TABLE_MAX_ROWS", "10000"))
            self.table_max_chars = int(os.environ.get("MARGENT_WANDB_TABLE_MAX_CHARS", "20000"))
            if self.table_max_rows < 1 or self.table_max_chars < 256:
                raise ValueError("W&B table limits require MAX_ROWS >= 1 and MAX_CHARS >= 256")
        project = os.environ.get("WANDB_PROJECT", "margent-math-rsi")
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            raise ValueError("Set WANDB_ENTITY to your W&B username or team before enabling tracking")
        # A loop owns the group; its subprocess stages discover the same manifest.
        experiment = next((p for p in (self.root, *self.root.parents)
                           if any((p / name).exists() for name in ("loop.json", "benchmark_run.json"))), self.root)
        manifest = next((experiment / name for name in ("loop.json", "benchmark_run.json", "run.json", "training_run.json")
                         if (experiment / name).exists()), None)
        metadata = _read(manifest) if manifest else {}
        group_file = experiment / "wandb_experiment.json"
        if group_file.exists():
            group = _read(group_file)
            if (group["project"], group["entity"]) != (project, entity):
                raise ValueError("W&B project/entity changed for this experiment; use the original settings")
        else:
            group = {"project": project, "entity": entity,
                     "group": experiment.name + "-" + uuid.uuid4().hex[:8]}
            _write(group_file, group)
        identity_file = self.root / "wandb_run.json"
        identity = _read(identity_file) if identity_file.exists() else {
            **group, "stage": self.stage, "id": uuid.uuid4().hex[:12]}
        if any(identity[k] != v for k, v in {**group, "stage": self.stage}.items()):
            raise ValueError("W&B stage identity changed; use the original experiment directory")
        _write(identity_file, identity)
        # Offline W&B cannot resume. Give each segment its own ID, retaining a
        # common logical_stage_id/group so later sync does not overwrite data.
        run_id = identity["id"] if self.mode == "online" else identity["id"] + "-" + self.attempt[:8]
        relative = self.root.relative_to(experiment).as_posix()
        name = group["group"] + "/" + (relative if relative != "." else self.stage)
        match = re.search(r"(?:^|/)round_(\d+)(?:/|$)", relative)
        config = _config(metadata.get("config", {}))
        config.update(arm=metadata.get("arm", "standalone"), stage=self.stage,
                      stage_path=relative, round=int(match[1]) if match else 0,
                      logical_stage_id=identity["id"])
        self.run = wandb.init(project=project, entity=entity, group=group["group"],
            id=run_id, name=name, job_type=self.stage, config=config, mode=self.mode,
            **({"resume": "allow"} if self.mode == "online" else {}), dir=str(self.root),
            settings=wandb.Settings(init_timeout=60, console="off", disable_git=True, save_code=False))
        self.run.define_metric("trainer_step")
        self.run.define_metric("train/*", step_metric="trainer_step")
        self.run.define_metric("grpo/*", step_metric="trainer_step")
        self.run.define_metric("diagnostic_step")
        for pattern in ("eval/*", "internalization/*", "delegation/*"):
            self.run.define_metric(pattern, step_metric="diagnostic_step")
        self.run.summary.update({"attempt": self.attempt, "status": "running"})
        self.run.summary.update({"debug/text_logging": self.text_enabled})
        url = self.run.url if self.mode == "online" else None
        link = {**identity, "active_id": run_id, "mode": self.mode, "url": url,
                "attempt": self.attempt}
        _write(self.root / "wandb_link.json", link)
        print(f"[wandb] {name}: {url or 'offline records in ' + str(self.root / 'wandb')}", flush=True)

    def log(self, values):
        if self.run:
            clean = scalar_metrics(values)
            if clean:
                # W&B owns its monotonic history step; trainer_step can restart
                # from the last saved checkpoint without dropping new records.
                self.run.log(clean)

    def log_text(self, kind, record):
        if self.run is None or not self.text_enabled:
            return
        from .debug_records import (GENERATION_COLUMNS, QUESTION_COLUMNS, ROLLOUT_COLUMNS,
                                    generation_values, question_values, table_row)
        if kind == "generation":
            columns, values = GENERATION_COLUMNS, generation_values(record)
        elif kind == "question":
            columns, values = QUESTION_COLUMNS, question_values(record, record.get("source", "live"))
        elif kind == "rollout":
            columns, values = ROLLOUT_COLUMNS, record
        else:
            raise ValueError(f"Unknown text table: {kind}")
        # SDK incremental tables do not resume their in-memory cursor. Separate
        # attempts keep earlier rows visible instead of replacing them on resume.
        key = f"debug/{kind}s_{self.attempt}"
        count = self.table_counts.get(key, 0)
        if count >= self.table_max_rows:
            self.run.summary[key + "_omitted_rows"] = self.run.summary.get(key + "_omitted_rows", 0) + 1
            if count == self.table_max_rows:
                print(f"[wandb] {key} reached its display row limit; full records remain local", flush=True)
            self.table_counts[key] = count + 1
            return
        if key not in self.tables:
            self.tables[key] = self.sdk.Table(columns=columns, log_mode="INCREMENTAL")
            self.run.summary["debug/table_keys"] = list(self.tables)
        self.tables[key].add_data(*table_row(columns, values, self.table_max_chars))
        self.table_counts[key] = count + 1
        self.pending_tables.add(key)
        self.flush_tables(force=bool(record.get("truncated") or record.get("error")))

    def flush_tables(self, force=False):
        if not self.pending_tables or (not force and self.last_table_flush is not None
                                       and time.monotonic() - self.last_table_flush < 30):
            return
        keys = sorted(self.pending_tables)
        self.run.log({key: self.tables[key] for key in keys})
        self.pending_tables.difference_update(keys)
        self.last_table_flush = time.monotonic()

    def finish(self, status):
        if self.run:
            try:
                self.flush_tables(force=True)
            finally:
                try:
                    self.run.summary.update({"status": status})
                finally:
                    self.run.finish(exit_code=0 if status == "completed" else 1)
