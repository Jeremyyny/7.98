"""Optional scalar-only W&B tracking with persistent stage identities.

Enable explicitly with MARGENT_WANDB_MODE=online or offline. Local experiment
records remain authoritative; this module never uploads trajectories or weights.
"""
from __future__ import annotations

from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
import uuid


def tracking_mode():
    mode = os.environ.get("MARGENT_WANDB_MODE", "disabled").lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("MARGENT_WANDB_MODE must be online, offline, or disabled")
    return mode


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

    def start(self):
        if self.mode == "disabled":
            return
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("W&B requested but not installed; install requirements-math.txt") from exc
        project = os.environ.get("WANDB_PROJECT", "margent-math-rsi")
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            raise ValueError("Set WANDB_ENTITY to your W&B username or team before enabling tracking")
        # A loop owns the group; its subprocess stages discover the same manifest.
        experiment = next((p for p in (self.root, *self.root.parents)
                           if (p / "loop.json").exists()), self.root)
        manifest = next((experiment / name for name in ("loop.json", "run.json", "training_run.json")
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
        self.run.define_metric("diagnostic_step")
        for pattern in ("eval/*", "internalization/*", "delegation/*"):
            self.run.define_metric(pattern, step_metric="diagnostic_step")
        self.run.summary.update({"attempt": self.attempt, "status": "running"})
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

    def finish(self, status):
        if self.run:
            self.run.summary.update(status=status)
            self.run.finish(exit_code=0 if status == "completed" else 1)
