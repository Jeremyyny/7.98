# MARGENT：自由回答数学与两轮迭代实验

本实现把已有 MARGENT 的同状态反事实比较、最短成功分支选择，扩展到自由回答数学。
入口为 `python -m src.verifiable`，所有命令在 `agent_routing/` 下运行。
旧的 MCQ 入口、实验配置和数值不受影响。新模块复用 `StandardRow`、IO 和原有的最短成功路径选择函数。

## 实验范围与 benchmark

| 用途 | 数据 | 说明 |
|---|---|---|
| 训练 | AI-MO/NuminaMath-1.5 的有效、非证明题子集 | 不把原始解答放进模型输入；使用题目和答案进行独立采样及评分 |
| 诊断/开发 | 同一来源中固定、与训练不相交的题 | 每个检查点测独立解题、实际委派、有限搜索覆盖和自我续写 |
| 外部测试 1 | MathArena/aime_2026，30 题 | 上游叫 `train`，本实现强制标记为 `test` |
| 外部测试 2 | ByteDance-Seed/BeyondAIME，100 题 | 整数答案、较难的竞赛数学；是否太难应由独立开发池的 pilot 判断 |

两项数学测试支持“超出多选题、适用于最终结果可验证的任务”。**它们不证明每一步自然语言推导可验证，也不能证明适用于所有可验证任务。**
如果论文要明确提出“中间步骤可验证”，下一项建议 miniF2F/Lean：每个 tactic 都交给 Lean 检查、保存可回放的证明状态。
这是不同的环境适配工作，本次没有把一个未经测试的 Lean 执行器混入数学训练。

数据准备会记录 HF 数据集 commit SHA、文件哈希、过滤数量，先去重再切分，并排除与两个外部测试的规范化文本重复。
这只做 Unicode/空白/大小写规范化的精确去重，不能排除改写题或模型预训练污染。
题目要求图片时不会悄悄删除图片；官方测试集有丢行会报错。

## 新协议和旧实验的关系

1. Manager 先独立生成完整解题草稿，得到 `D_t`；这一份草稿是各分支共同的起点。
2. 同一状态分别直接提交、调用不同 advisor 后修正；每个 advisor 最多一次，深度可配置为 1–3。
3. 在有界空间内完整搜索，保存所有结果。深度 2 为 3+6 个委派分支；没有因提前成功而缩短后续检查点的搜索预算。
4. 根据最短成功路径导出调用决策及完整解答。额外导出“只给原题→完整成功解答”的蒸馏样本，用于检验能力内化；可设置 `distill_solutions=false` 做对照。
5. 用 LoRA 继续 SFT/GRPO；第二轮从上轮 checkpoint 继续，而非重置为初始模型。

终止格式为 `FINAL_ANSWER: \boxed{...}`，评分只解析唯一的最后一行。
不会把推导里的最后一个数字误当作答案。整数/普通小数用精确有理数比较，符号表达式使用 Math-Verify。
这里的 verifier advisor 是会犯错的模型；答案评分器不会向它或 manager 泄露标签。

采集、SFT、RL、独立基线和策略评估统一使用仓库内的固定 Qwen ChatML/JSON 工具模板（关闭隐藏 thinking，要求显式推导）。
这是新数学实验的固定执行协议，不是 Qwen3.5 默认的 XML 工具模板，不能将其数值直接与旧 MCQ scaffold 的数值作因果比较。
RL 每个阶段开始时用当时的 SFT checkpoint 重新生成草稿；该阶段内草稿固定，GRPO 更新后续委派和修正。跨轮再生成新草稿和反事实数据。

默认 advisor 服务是一个**冻结的基础模型，采用三个数学角色提示**，不需要外部付费 API，也不会自动使用旧医学 LoRA。
如已有数学 advisor adapters，可用兼容的模型服务提供三个别名，并在配置的 `advisor_models` 中映射。第一版保持整个循环的 advisor 权重不变。

## RunPod 环境

建议先用带 CUDA 的 PyTorch 镜像、Python 3.11/3.12。9B 正式实验可从两张 80GB GPU 的配置试跑：GPU 0 放冻结 advisor，GPU 1 放 manager/LoRA 训练。
这是保守的起跑配置，并非已实测的显存保证。单张 80GB 可尝试让两个进程共卡，但需减小长度/批次并检查 OOM；本入口不支持多卡分布式 manager 训练，不要用 `torchrun` 包装。
0.6B smoke test 可共用一张 GPU。

代码、虚拟环境、数据和 checkpoints 放在 `/workspace` 挂载卷，建议预留至少 100GB。网络卷可跨 Pod 保留；普通 volume disk 随 Pod 删除而消失。

```bash
cd /workspace
git clone --branch codex/math-rsi-runpod https://github.com/Jeremyyny/7.98.git
cd /workspace/7.98/agent_routing
bash scripts/runpod_math_setup.sh
source /workspace/margent-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export MARGENT_RUN_ROOT=/workspace/margent-runs
```

依赖与原 MCQ 环境分开。Transformers/TRL/PEFT/Math-Verify 已固定版本；脚本另保存实际安装锁文件和 GPU 信息。
GPU 训练依赖 Pod 镜像中的 CUDA PyTorch，不要在 CPU 镜像里期待本脚本自动配置驱动。

## 先做 GPU smoke test

终端 A（保持运行）：

```bash
cd /workspace/7.98/agent_routing
source /workspace/margent-venv/bin/activate
export HF_HOME=/workspace/hf-cache
CUDA_VISIBLE_DEVICES=0 python -m src.verifiable.serve --model Qwen/Qwen3-0.6B
```

终端 B：

```bash
cd /workspace/7.98/agent_routing
source /workspace/margent-venv/bin/activate
export HF_HOME=/workspace/hf-cache
# 两张卡时使用 1；只有一张卡时改成 0。
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_smoke.sh
```

这用 8 道训练算术题、4 道诊断算术题、每阶段 2 个训练步，跑两个完整周期。
只验证训练流程，不作为论文实验。正常完成后应生成 `/workspace/margent-runs/smoke-loop/loop_report.json`。
若没有有效成功轨迹，先检查输出格式和截断情况，不要直接启动正式长训练。

## 正式数据和小规模 pilot

先停止 smoke advisor（终端 A 的 Ctrl-C），换成正式且全程冻结的 advisor：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.verifiable.serve --model Qwen/Qwen3.5-9B --max-context 24576
```

终端 B：

```bash
python -m src.verifiable prepare --data-dir /workspace/margent-data \
  --train-size 1024 --dev-size 256 --seed 42

CUDA_VISIBLE_DEVICES=1 python -m src.verifiable doctor \
  --config configs/math_rsi.json --out /workspace/margent-runs/environment.json

CUDA_VISIBLE_DEVICES=1 python -m src.verifiable collect \
  --config configs/math_rsi.json --data /workspace/margent-data/train.jsonl \
  --out /workspace/margent-runs/pilot --limit 32 --resume
```

先看 `pilot/summary.json` 和完整 `records.jsonl`：是否有非零的独立成功和协作救援；格式是否正确；是否频繁截断。
2048 tokens 是起跑预算，不代表足够解决竞赛数学。若需要增加，联动增加 `max_context`、`max_seq_len`、`rl_max_completion_length`，再以冻结后的统一配置启动所有正式对照。
advisor 截断或 HTTP 错误会中断采集，避免把系统错误当成“此分支数学上失败”。manager 截断明确记为无效完成。
SFT 过长样本会整条丢弃并计数，不会静默截去答案；若丢弃多，应先调整长度。

## 两轮闭环与对照

```bash
# 先显示将执行的命令，不加载模型、不开始训练
python -m src.verifiable loop --config configs/math_rsi.json \
  --data-dir /workspace/margent-data --out /workspace/margent-runs/dynamic_rl \
  --arm dynamic_rl --rounds 2 --dry-run

CUDA_VISIBLE_DEVICES=1 python -m src.verifiable loop --config configs/math_rsi.json \
  --data-dir /workspace/margent-data --out /workspace/margent-runs/dynamic_rl \
  --arm dynamic_rl --rounds 2 --resume
```

对照只需换 `--arm` 和不同 `--out`，其他设置保持一致：

| arm | 行为 |
|---|---|
| `dynamic_rl` | 每轮新反事实数据→SFT→GRPO |
| `dynamic_sft` | 每轮新反事实数据→SFT，无 RL |
| `static_rl` | 初始反事实数据固定，每轮继续 SFT→GRPO |
| `success_rl` | 同样采集分支，但随机选择成功分支，而非最短成功分支 |

`success_rl` 是在**同一已采集分支池**上的选择对照，不是廉价的独立普通 self-training。
`static_rl` 会省去后续反事实采集，而 `dynamic_sft` 少了 RL；默认脚本不自动制造“总算力相同”。
论文应使用计入采集、advisor、SFT 和 RL 的成本曲线，或者另行锁定相同总预算。
保持相同步数也不意味着相同训练 token 数。保存的 token/时间记录用于成本分析，不是 FLOPs 的精确估计。

每阶段用独立子进程运行以释放显存。已完成阶段可跳过，采集按完整题目恢复；训练每 10 步保存最近两个 Trainer checkpoint 并恢复优化器/RNG。
首次 checkpoint 前中断的训练会从该阶段开头重跑；重试所浪费的计算量不要从论文成本中隐去。
改了模型、配置、数据或轮数，请使用新的输出目录。`--resume` 不会混用不同实验。

## 最终测试和结果

循环只看 `dev.jsonl`，不会自动用 AIME/BeyondAIME 选 checkpoint。固定实验选择后显式运行：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.verifiable evaluate \
  --config configs/math_rsi.json --data /workspace/margent-data/aime2026.jsonl \
  --checkpoint /workspace/margent-runs/dynamic_rl/round_2/rl \
  --out /workspace/margent-runs/final_aime2026 --resume

CUDA_VISIBLE_DEVICES=1 python -m src.verifiable evaluate \
  --config configs/math_rsi.json --data /workspace/margent-data/beyondaime.jsonl \
  --checkpoint /workspace/margent-runs/dynamic_rl/round_2/rl \
  --out /workspace/margent-runs/final_beyondaime --resume
```

用同样命令、不同输出目录评估预先指定的初始模型和各个对照 checkpoint。
`evaluate` 只运行独立作答及部署策略，不在最终测试中采集训练目标；有限搜索机制分析在 `diagnose`/dev 中完成。

主要产物：

- `initial_dev/summary.json`、`round_*/sft_dev/summary.json`、`round_*/rl_dev/summary.json`：D/P/实测分支覆盖、自我续写、平均调用数和 token 计数。
- `round_*/collection/records.jsonl`：未截短的根草稿、每条分支和调用历史；`sft.jsonl` 为实际训练目标。
- `round_*/sft/sft_data_report.json`：完整训练样本的保留/丢弃数量和每 epoch token 数。
- `round_*/rl/rollouts.jsonl`：终止正确性、调用和 advisor 用量；`training_metrics.json` 及 Trainer 日志用于观察 reward/梯度与零方差组。
- `loop_report.json`：各检查点相对初始模型的新解题、退步，以及“原先需要帮助、现在独立解决”的题目 ID。

coverage 是给定工具池、搜索深度、生成预算和解码设置下的**实测覆盖率**，不是真实能力上限。
训练题上的救援不代表泛化；能力内化要看独立诊断题和外部测试。AIME 的 30 题分数较粗，应同时报告题级变化与不确定性。

## 本地验证与限制

```bash
python -m pytest -q
CUDA_VISIBLE_DEVICES='' MARGENT_CPU_INTEGRATION=1 python -m pytest -q tests/test_math_training_integration.py
```

逻辑测试覆盖严格评分、gold 隔离、数据去重/切分、完整轨迹、同根分支、策略不读答案、恢复检查和两轮权重延续。
可选 CPU 集成测试创建微型随机模型，实际运行 SFT、GRPO、保存及重新加载 LoRA，不下载 9B 权重。
这不替代 RunPod 上的 Qwen3.5 CUDA、长上下文、工具调用和性能验收。

本地测试包含上述 CPU 集成测试及流式读取提前结束的进程退出检查；实际下载并完整保留 AIME 2026 的 30 题和 BeyondAIME 的 100 题，
另用 NuminaMath 小样本验证过滤与切分。数据准备通过 HF 的文件映射逐批读取 Parquet、关闭 Arrow 读取线程并显式关闭迭代器，规避上游 [流式读取退出卡住的问题](https://github.com/apache/arrow/issues/50482)。

参考：[Math-Verify](https://github.com/huggingface/Math-Verify)、[NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)、[AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026)、[BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME)、[TRL 0.29 GRPO](https://huggingface.co/docs/trl/v0.29.0/grpo_trainer)、[RunPod storage](https://docs.runpod.io/pods/storage/types)。
