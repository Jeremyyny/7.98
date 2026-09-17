from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from ..utils.io import read_jsonl, write_json
from .data import prepare
from .experiment import compare
from .runner import load_config, run_data, run_loop


def doctor(config, output):
    import torch
    import transformers
    from trl import GRPOConfig, GRPOTrainer
    from .answers import correct
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run training preflight inside the RunPod GPU container")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This first math runner uses one training process; do not launch it with torchrun")
    if "environment_factory" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise RuntimeError("TRL environment_factory unavailable")
    required = {"max_tool_calling_iterations", "chat_template_kwargs", "mask_truncated_completions"}
    if required - set(inspect.signature(GRPOConfig).parameters):
        raise RuntimeError("Installed GRPOConfig is incompatible")
    assert correct(r"FINAL_ANSWER: \boxed{\frac{1}{2}}", "0.5")
    assert not correct("My working contains 42", "42")
    model_cfg = transformers.AutoConfig.from_pretrained(config["base_model"])
    if model_cfg.model_type == "qwen3_5" and not hasattr(transformers, "Qwen3_5ForCausalLM"):
        raise RuntimeError("Transformers lacks Qwen3.5 text-only support")
    from .backend import configure_tokenizer, render
    tok = configure_tokenizer(transformers.AutoTokenizer.from_pretrained(config["base_model"]))
    from .protocol import SYSTEM, tool_schemas
    render(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Compute 1+1."}], tool_schemas())
    # TRL needs a response parser in addition to a generation chat template.
    from trl.chat_template_utils import is_chat_template_prefix_preserving
    if not is_chat_template_prefix_preserving(tok):
        raise RuntimeError("Math chat template failed prefix-preservation check")
    parsed = tok.parse_response('<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call><|im_end|>')
    if parsed["tool_calls"][0]["function"]["name"] != "reasoner_tool":
        raise RuntimeError("Native tool parsing failed")
    result = {"packages": {name: importlib.metadata.version(name) for name in
              ("torch", "transformers", "trl", "peft", "datasets", "math-verify")},
              "gpu": torch.cuda.get_device_name(0),
              "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
              "model_type": model_cfg.model_type,
              "note": "API and template preflight only; run the GPU smoke test before the main experiment"}
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description="MARGENT free-response math / iterative self-training")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--train-size", type=int, default=1024)
    prep.add_argument("--dev-size", type=int, default=256)
    prep.add_argument("--scan-limit", type=int, default=30000)
    prep.add_argument("--seed", type=int, default=42)
    for name in ("numina", "aime2026", "beyondaime"):
        prep.add_argument("--local-" + name, help="Optional local source JSONL, mainly for offline tests")
    check = sub.add_parser("doctor")
    check.add_argument("--config", required=True)
    check.add_argument("--out", default="environment_report.json")
    for name in ("collect", "diagnose", "evaluate", "sft", "rl"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--data", required=True)
        sp.add_argument("--checkpoint")
        sp.add_argument("--out", required=True)
        if name in ("collect", "diagnose", "evaluate"):
            sp.add_argument("--resume", action="store_true")
            sp.add_argument("--limit", type=int, default=0)
            sp.add_argument("--selection", choices=("counterfactual", "success"), default="counterfactual")
    loop = sub.add_parser("loop")
    loop.add_argument("--config", required=True)
    loop.add_argument("--data-dir", required=True)
    loop.add_argument("--out", required=True)
    loop.add_argument("--arm", choices=("dynamic_rl", "dynamic_sft", "static_rl", "success_rl"), default="dynamic_rl")
    loop.add_argument("--rounds", type=int, default=2)
    loop.add_argument("--checkpoint")
    loop.add_argument("--resume", action="store_true")
    loop.add_argument("--dry-run", action="store_true")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--before", required=True)
    cmp.add_argument("--after", required=True)
    cmp.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        sources = {name: getattr(args, "local_" + name) for name in ("numina", "aime2026", "beyondaime")
                   if getattr(args, "local_" + name)}
        result = prepare(args.data_dir, args.train_size, args.dev_size, args.seed, args.scan_limit, sources)
    elif args.command == "compare":
        result = compare(read_jsonl(args.before), read_jsonl(args.after))
        write_json(args.out, result)
    elif args.command == "loop":
        result = run_loop(args.config, args.data_dir, args.out, args.arm, args.rounds,
                          args.checkpoint, args.resume, args.dry_run)
        if args.dry_run:
            return
    else:
        cfg = load_config(args.config)
        if args.command == "doctor":
            result = doctor(cfg, args.out)
        else:
            checkpoint = args.checkpoint or cfg["base_model"]
            if args.command in ("sft", "rl"):
                from .training import train_sft, train_rl
                fn = train_sft if args.command == "sft" else train_rl
                result = fn(cfg, checkpoint, args.data, args.out)
            else:
                result = run_data(cfg, args.data, checkpoint, args.out, args.command,
                                  args.resume, args.limit, args.selection)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
