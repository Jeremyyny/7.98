# RunPod 数学补充实验：从安装到论文报告（协议 v2）

本指南对应 `codex/math-paper-protocol-v2` 分支。合并后可使用 main；所有正式运行必须固定同一个提交。
只运行两轮 SFT。主对照为 `dynamic_sft` 与 `success_sft`；时间允许再加 `static_sft`。

## 1. 创建 Pod

一个 Pod，2 张 A100 80GB：GPU 0 跑冻结 subagent，GPU 1 跑 Manager 的采集、训练和评估。
这不是双卡分布式训练，PCIe 与 SXM 都可用。选有 CUDA PyTorch 的镜像，建议 PyTorch 2.8 或更新的兼容版本，Python 3.11/3.12。
建议至少 64 GB 主机内存、150 GB 持久存储；实际空间随模型缓存和 checkpoint 数量增加。
若希望删除 Pod 后保留文件，在创建时挂载 network volume。
普通 `/workspace` volume 随 Pod 删除而删除，不能把它当永久备份。
参见 [RunPod 存储说明](https://docs.runpod.io/pods/storage/types)。

## 2. 安装和固定代码

在 Pod 终端运行：

```bash
cd /workspace
git clone --branch codex/math-paper-protocol-v2 --single-branch https://github.com/Jeremyyny/7.98.git margent
cd /workspace/margent/agent_routing
export MARGENT_RUN_ROOT=/workspace/margent-runs-v2
export MARGENT_DATA=/workspace/margent-data-v2
export HF_HOME=/workspace/hf-cache
bash scripts/runpod_math_setup.sh
source /workspace/margent-venv/bin/activate
git rev-parse HEAD > "$MARGENT_RUN_ROOT/code_commit.txt"
```

安装脚本保留镜像中的 CUDA PyTorch，安装固定的 Transformers/TRL/PEFT 和答案校验依赖，运行 CPU 回归测试并保存环境版本。
此后不要在现有运行目录中 `git pull`、换 prompt 或改配置。

## 3. 配置 W&B

```bash
export MARGENT_WANDB_MODE=online
export WANDB_PROJECT=margent-math-rsi
export WANDB_ENTITY=你的WandB用户名或团队名
# 需要在自己的 W&B 项目里查看题目、正确答案、输入消息和模型原文时开启：
export MARGENT_WANDB_TEXT=1
wandb login
python -m src.verifiable wandb-check --out "$MARGENT_RUN_ROOT/wandb-check"
```

`wandb login` 在终端交互输入 API key，不要把 key 写进 Git。离线记录可用 `MARGENT_WANDB_MODE=offline`，完成后 `wandb sync` 对应目录；完全不用则设为 `disabled`。
每个实验有一个 group，各训练/采集阶段有单独 run。`wandb_link.json` 保存链接。
默认只上传指标和配置；`MARGENT_WANDB_TEXT=1` 额外上传逐题原文表格，模型权重仍不上传。
在 `initial_dev`、`collection`、`sft_dev` 或外部评估的 stage run 中查看：

| W&B Table 前缀 | 内容 |
| --- | --- |
| `debug/questions_` | 已完成题目的题目、正确答案、独立回答、正确/有效/截断标记、分支成功数、策略回答与判定、自我修改回答、token 和生成耗时合计 |
| `debug/generations_` | 每次 Manager/advisor 生成的输入消息、回答、角色、当前阶段、委托序列、截断/错误、token 上限、实际 token、耗时与生成速度 |

表格名后缀是本次 attempt ID。恢复同一在线 run 时使用新后缀，旧表仍留在该 run 中。
表格使用增量上传：首条立即提交，之后约每 30–40 秒、每题结束或退出时提交；advisor 截断时立即提交。
`status=running` 的心跳不等于生成取得进展；用生成表的实际输出、token 和秒数判断。
`tokens_per_second` 包含单次调用的提示处理等时间，是观测吞吐，不能当作纯解码速度；缓存命中时不虚报实际生成 token。
每张表每个 attempt 默认最多 10000 行、每个文本字段最多 20000 字符；可用
`MARGENT_WANDB_TABLE_MAX_ROWS` / `MARGENT_WANDB_TABLE_MAX_CHARS` 调整。
被裁剪的字段在 `clipped_fields` 标出，超出行数在 run summary 的 `*_omitted_rows` 计数。
完整的逐次生成（包括截断的 advisor 输出）先写入 stage 目录的 `generations.jsonl`，与上传开关无关。
上传故障保留本地文件并继续原有实验流程；advisor 截断仍按原规则中止，不会被当成正确训练数据。

### 在 W&B 查看之前已经保存的结果

这个命令只读取旧结果，不加载模型、不运行 GPU、不修改旧实验。旧版本没有记录的失败回答无法补回。
`--run-dir` 指向具体 stage（例如 `initial_dev`），`--out` 必须是实验 loop 外的全新目录。

```bash
export MARGENT_WANDB_MODE=online
export MARGENT_WANDB_TEXT=1
python -m src.verifiable wandb-upload-records \
  --run-dir /workspace/margent-runs-v2/smoke-loop/initial_dev \
  --out "/workspace/margent-runs-v2/wandb-text-review-$(date +%Y%m%d-%H%M%S)"
```

打开命令打印的新 review run 链接，查看上述两类表。旧版 `records.jsonl` 的 `costs` 可补传回答和耗时，
未保存的输入消息、阶段和序列保留为空；`source=legacy_costs` 明确标识这些恢复的行。
新版本的 `generations.jsonl` 则可补传没有完成整道题就失败的输出。
review 配置记录原运行配置、原 harness 标识及源文件 SHA256，不重新计算标签或覆盖原 run。

### 更新代码后的运行目录

这次更新改变了 harness 源码标识。补传旧结果不受影响；继续计算时需要重启 advisor，使两端运行同一版代码，
并使用新的 loop/pilot 输出目录，不能用新代码对旧 loop 执行 `--resume`。
模型缓存和准备好的数据可以继续使用。例：在新终端中激活环境、设置 HF/W&B 环境变量，停止旧 advisor 后：

```bash
export MARGENT_RUN_ROOT=/workspace/margent-runs-v2-tables
mkdir -p "$MARGENT_RUN_ROOT"
cp -n /workspace/margent-runs-v2/frozen_math.json "$MARGENT_RUN_ROOT/frozen_math.json"
cp -n /workspace/margent-runs-v2/frozen_smoke_9b.json "$MARGENT_RUN_ROOT/frozen_smoke_9b.json"
git rev-parse HEAD > "$MARGENT_RUN_ROOT/code_commit.txt"
export MARGENT_SMOKE_CONFIG="$MARGENT_RUN_ROOT/frozen_smoke_9b.json"
```

按下文第 5 步重新启动 advisor，再跑 smoke。启用表格本身不会解决此前的截断问题；
即使仍失败，新版本也会保留该次 advisor 原文供排查。调整 token 预算或 prompt 后仍须使用新的运行目录。

## 4. 准备数据与固定模型版本

7 天预算先用 128 条 train、64 条 dev 做有对照的小实验。测试集保持完整；不要为了缩短时间抽取测试题。
吞吐允许时，可以在正式运行前另建目录增加训练/验证规模。

```bash
python -m src.verifiable prepare --data-dir "$MARGENT_DATA" \
  --train-size 128 --dev-size 64 --scan-limit 30000 --seed 42
python -m src.verifiable freeze-config --config configs/math_rsi.json \
  --out "$MARGENT_RUN_ROOT/frozen_math.json"
```

只准备一次数据，所有组和种子共用它。训练 seed 变化不应重新划分题目。
配置默认最多两次调用、答案上限 4096 token、subagent 上限 2048 token、每轮 50 个 SFT optimizer steps。
这些值是试跑起点，显存和截断率必须由下一步实测。

## 5. 在 GPU 0 启动冻结 subagent

打开第二个终端或 tmux 会话；保持它运行。所有终端先进入相同目录并激活环境。

```bash
cd /workspace/margent/agent_routing
source /workspace/margent-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export MARGENT_RUN_ROOT=/workspace/margent-runs-v2
MARGENT_REVISION=$(python -c 'import json,os; print(json.load(open(os.environ["MARGENT_RUN_ROOT"]+"/frozen_math.json"))["base_model_revision"])')
CUDA_VISIBLE_DEVICES=0 python -m src.verifiable.serve \
  --model Qwen/Qwen3.5-9B --revision "$MARGENT_REVISION" \
  --max-context 32768 --port 8000
```

看到 `Frozen advisor ready` 后继续。Extractor、Reasoner、Verifier 是同一冻结模型的三个角色 prompt；这是当前实验设定。
服务器只监听本机，不需要暴露公网端口。

## 6. 在 GPU 1 做预检查、真实训练冒烟测试和数学 pilot

回到 Manager 终端，保留第 2/3 步的环境变量：

```bash
export CUDA_VISIBLE_DEVICES=1
python -m src.verifiable doctor --config "$MARGENT_RUN_ROOT/frozen_math.json" \
  --out "$MARGENT_RUN_ROOT/environment.json"
```

先用正式 9B 模型和正式 prompt 跑 2 步真实训练，测试 LoRA、存储、续训与跨进程执行。算术冒烟题不计入论文：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['MARGENT_RUN_ROOT'])
cfg = json.loads((root / 'frozen_math.json').read_text())
cfg.update(sft_max_steps=2, sft_accumulation=1)
path = root / 'frozen_smoke_9b.json'
if path.exists() and json.loads(path.read_text()) != cfg:
    raise ValueError('已有 smoke 配置不同，请使用新目录')
path.write_text(json.dumps(cfg, indent=2))
PY
export MARGENT_SMOKE_CONFIG="$MARGENT_RUN_ROOT/frozen_smoke_9b.json"
bash scripts/runpod_math_smoke.sh
```

接着在真实数学题上测格式、截断、成功轨迹数量和时间：

```bash
python -m src.verifiable collect --config "$MARGENT_RUN_ROOT/frozen_math.json" \
  --data "$MARGENT_DATA/train.jsonl" --out "$MARGENT_RUN_ROOT/pilot-collect" --limit 8 --resume
python -m src.verifiable diagnose --config "$MARGENT_RUN_ROOT/frozen_math.json" \
  --data "$MARGENT_DATA/dev.jsonl" --out "$MARGENT_RUN_ROOT/pilot-dev" --limit 8 --resume
python -m src.verifiable pilot-cost --collect-dir "$MARGENT_RUN_ROOT/pilot-collect" \
  --diagnose-dir "$MARGENT_RUN_ROOT/pilot-dev" --train-size 128 --dev-size 64 \
  --arms 2 --seeds 2 --rounds 2
```

检查 `summary.json`、`records.jsonl` 和 `sft.jsonl`：是否有成功解法、被求助救回的题，是否频繁截断或输出非法动作，成功解法是否自洽且能脱离工具上下文独立阅读。
`pilot-cost` 只估计采集和 dev 诊断时间，不包括训练、加载、外部测试和失败重试。
若这部分已接近 7 天预算，不要直接启动整套实验。

需要调整 token 上限/学习率/数据量时，在正式运行前做，用新配置与新运行目录重新试跑；所有正式组使用同一最终配置。
不要看 AIME/BeyondAIME 分数后再调参。尚未运行 GPU 的环境不能保证 9B 显存或速度。

## 7. 正式运行：两组 × 两个种子 × 两轮

建议在 tmux 中执行，避免浏览器断开影响前台任务。先固定主对照种子 `42 43`。
下列脚本默认只完成学习循环，暂不打开外部测试分数：

```bash
for MARGENT_SEED in 42 43; do
  bash scripts/runpod_math_experiment.sh dynamic_sft "$MARGENT_SEED"
  bash scripts/runpod_math_experiment.sh success_sft "$MARGENT_SEED"
done
```

`dynamic_sft` 是每轮重新采集的最短成功轨迹；`success_sft` 是每轮重新采集的随机成功轨迹。
两组都包含完整解法蒸馏，并使用相同训练更新预算。它们不具有严格相等的实际 token/FLOPs。
时间足够时，可在看外部测试前预先决定增加第三个 seed 44，或运行静态数据对照：

```bash
bash scripts/runpod_math_experiment.sh static_sft 42
bash scripts/runpod_math_experiment.sh static_sft 43
```

不要只保留效果最好的 seed。训练曲线、格式失败或负向结果都应保留。

## 8. 锁定初始/最终模型，运行完整外部测试

所有设计选择固定后：

```bash
for MARGENT_SEED in 42 43; do
  for MARGENT_ARM in dynamic_sft success_sft; do
    python -m src.verifiable evaluate-suite \
      --run-dir "$MARGENT_RUN_ROOT/${MARGENT_ARM}_s${MARGENT_SEED}" \
      --data-dir "$MARGENT_DATA"
  done
done
```

如果运行了静态对照，也用相同命令评估 `static_sft_s42` 和 `static_sft_s43`。
这里没有测试集分支搜索，只评估初始和最终 checkpoint 的独立解题与实际策略。

## 9. 生成论文表格并检查完整性

```bash
python -m src.verifiable paper-check \
  --runs "$MARGENT_RUN_ROOT/dynamic_sft_s42" "$MARGENT_RUN_ROOT/success_sft_s42" \
         "$MARGENT_RUN_ROOT/dynamic_sft_s43" "$MARGENT_RUN_ROOT/success_sft_s43" \
  --out "$MARGENT_RUN_ROOT/paper-report" --min-seeds 2
```

主要查看：

| 文件 | 用途 |
| --- | --- |
| `paper_readiness.json` | 配置、种子、数据来源、阶段和外部测试是否完整 |
| `paper_main.csv/.tex` | 初始/最终独立准确率、策略准确率、调用数 |
| `arm_comparisons.csv`、`arm_summary.csv` | 方法组相对对照组的逐种子配对差值与跨种子变化 |
| `delegation_behavior.csv` | 当前独立/可救回/未解决各组的调用和成功情况 |
| `paper_mechanism.csv/.tex`、`paired_questions.jsonl` | 内化比例、同一批已学会题目的前后调用和退步 |
| `costs.csv`、`report_manifest.json` | 实际 token/时间、失败记录和统计解释边界 |
| `fig1`–`fig4`、`fig6` 的 PDF/PNG | 学习曲线、内化、外部测试、成本和条件求助图 |

检查通过不代表结果为正、统计显著或论文一定录用。最终答案校验也不是中间步骤验证。

## 10. 监控、暂停与恢复

```bash
python -m src.verifiable status --run-dir "$MARGENT_RUN_ROOT/dynamic_sft_s42"
tail -f "$MARGENT_RUN_ROOT/dynamic_sft_s42/logs/dynamic_sft_s42_round_1_sft.log"
```

实际日志文件名由阶段路径生成；也可直接查看 `logs/`，不要仅凭 heartbeat 判断训练有进展。
逐阶段 `status.json` 显示进度；`errors.log` 保存异常；W&B 显示 loss、token 用量和各 checkpoint 指标。

恢复：重启同版本冻结 subagent，激活相同环境，然后重新执行原来的 `runpod_math_experiment.sh` 命令。
保持原配置、数据目录、输出目录、代码和 W&B project/entity。已有完成阶段跳过，采集按题恢复，训练从 optimizer checkpoint 恢复。
不要用协议 v1 的旧训练目录继续 v2。关闭或删除 Pod 前备份代码提交、环境、配置、数据 manifest、逐题记录、报告与 adapter 权重。
