# MARGENT RSI：24 小时 pilot 启动说明

代码已合并到本仓库的 `main` 分支，包含符合 immutable-COMMIT 协议的 GRPO 与三组两轮 pilot。优先通过 Git 获取 `main`，无需上传 ZIP。

原有 `python -m src.verifiable loop` 仍然是 SFT-only；本次必须使用 `python -m src.verifiable.rsi` 或下面的新脚本。不要把新结果混入原来的 `/workspace/margent-runs-restart-20260925`。

若首次 smoke 报 `Unclosed tool call` / 全组非法动作，先按[协议恢复说明](RSI_PROTOCOL_RECOVERY.md)复用已有 SFT checkpoint 做只读检查。新语法约束需显式使用 `math_rsi_actions.json`，原配置不会偷偷改变。

## 1. 获取代码并检查

在现有 RunPod 仓库中执行。只复制代码框内的命令，不要把 Markdown 反引号一起贴进终端。

```bash
cd /workspace/margent-restart-20260925
git switch main
git pull --ff-only origin main
cd agent_routing
source /workspace/margent-venv/bin/activate
python -m pip check
python -m pytest -q tests/test_rsi.py
```

若 Git 提示本地修改或分叉，保留修改，不要 reset/强推。可直接克隆到一个新目录后，将后续命令中的路径换成新的 `agent_routing` 路径：

```bash
cd /workspace
git clone --branch main https://github.com/Jeremyyny/7.98.git margent-rsi-code
cd /workspace/margent-rsi-code/agent_routing
```

测试使用本地随机小模型，不下载 Qwen 权重。在有 CUDA 的机器上，小模型测试可能使用 CUDA。

已有环境应包含：torch 2.8.0、transformers 5.3.0、TRL 0.29.0、PEFT 0.18.1、datasets 4.x、math-verify 0.8.0、W&B 0.30.0。脚本不会重装 torch，也不会下载第二份 Qwen 缓存。缺少 pytest 时在已有 venv 安装 pytest；其他依赖以 `requirements-math.txt` 为准。

## 2. 先运行真实 GPU smoke，再启动完整 pilot

小模型单元测试不能替代 A100/Qwen3.5-9B 端到端验证。首次启动先运行：

```bash
tmux new-session -d -s rsi-smoke -c /workspace/margent-restart-20260925/agent_routing '/workspace/margent-venv/bin/python -u scripts/runpod_rsi_smoke.py --out /workspace/margent-rsi-smoke-01 > /workspace/margent-rsi-smoke-01.log 2>&1'
tail -n 40 /workspace/margent-rsi-smoke-01.log
```

该测试固定选取原 NuminaMath 划分的 2 train / 1 dev，使用真实 9B 模型：
W&B check → 自有 advisor → doctor → 采集 → 1 步 SFT → 动作格式预检 → 1 步 GRPO → 重载评估 → 1 步后续 SFT。
后续 SFT 重用这两道题的首轮目标，只检查 GRPO adapter 能接着训练；它不验证第二轮重新采集或完整三组实验。
保留 pilot 的长度、深度、4 条 GRPO 采样等设置，仅缩小数据、步数和 SFT 梯度累积。
Manager 默认 GPU 1，发现已有计算进程就拒绝启动；advisor 使用 GPU 0 上自有的 8002 服务，不停止其他服务。
测试最多 60 分钟，退出时关闭自己启动的子进程及 advisor；不会关闭 Pod 或停止其他服务计费。
输出目录必须全新；修复失败后用 `--out` 指定新目录，不能覆盖证据。固定端口也避免同一 smoke 的重复启动。

`smoke_report.json` 的 `completed` / `execution_passed` 只说明流程和 checkpoint 重载通过。
非法采样比例和缺少奖励差异记为警告，不再把已完成的流程判失败；奖励仍按原规则计分。
已有运行可用 `bash scripts/review_rsi_smoke.sh` 上传汇总和逐条输出，无需重跑训练。
`grpo_learning_signal_observed` 需要混合奖励、非零梯度和实际 adapter 变化，可能为 false。
真实题上没有可用 SFT 目标、非法格式、截断、OOM 或超时都会保留日志并停止，不换题或伪造奖励使测试通过。
本测试不验证中断后的 optimizer 恢复，不触碰 AIME/BeyondAIME，也不配置或发送邮件告警。
W&B 默认 online、文本记录开启，项目 `yuningyangaillm/MATH_rsi`。

**先检查 smoke 报告和 W&B 页面，再执行下面的完整 pilot。**

## 3. 终端 A：启动冻结 advisor

先在原 advisor 终端按 Ctrl+C 停止你之前启动的旧服务，然后在新代码目录启动本次服务。新代码有新的 harness 哈希，旧服务不能用于新实验。

```bash
cd /workspace/margent-restart-20260925/agent_routing
bash scripts/runpod_rsi_pilot.sh advisor
```

保持这个终端打开。默认 GPU 0、端口 8001、Qwen3.5-9B pinned revision、32768 context。看到服务 ready 后再执行下一步。另开终端可检查：

```bash
curl -fsS http://127.0.0.1:8001/health
```

默认缓存还是 `/workspace/hf-cache`，默认 venv 还是 `/workspace/margent-venv`。

## 4. 终端 B：预览然后启动

```bash
cd /workspace/margent-restart-20260925/agent_routing
bash scripts/runpod_rsi_pilot.sh plan
```

这会从旧的 128/64 数据中按题目哈希抽取 16 train / 16 dev，写入新的 `/workspace/margent-rsi-pilot-data-v1`，并打印执行计划。不会训练，不碰 AIME2026 / BeyondAIME。

计划应含有 `collect → sft → assess → grpo → assess`，第二轮的 `sft --checkpoint` 指向本组第一轮的 `grpo`。确认后用下面一条启动；不需要再次请助手批准。

```bash
cd /workspace/margent-restart-20260925/agent_routing
bash scripts/runpod_rsi_pilot.sh run
```

默认 Manager GPU 1，结果 `/workspace/margent-rsi-pilot-v1`。三组 dynamic / static / success，各两轮。控制器最多运行 24 小时；重启同一路径不会重置时限。它只限制这次实验的子进程，**不会自动停止 Pod 计费，也不会停止另一个终端的 advisor 服务**。

在网络断开也需要继续跑时，把终端 B 放在已有 tmux 会话中；不要同时开两个相同输出目录的 run。启动器只把每个阶段的完整输出写入日志，终端不会持续刷每条生成。

W&B 默认沿用 `yuningyangaillm/MATH_rsi`，online 模式；登录沿用现有 venv 配置。如需重登，使用：

```bash
/workspace/margent-venv/bin/python -m wandb login
```

不把 API key 放进文档、配置或实验日志。

## 5. 看进度与结果

```bash
cd /workspace/margent-restart-20260925/agent_routing
bash scripts/runpod_rsi_pilot.sh report
```

关键文件：

- `/workspace/margent-rsi-pilot-v1/pilot_report.json`：已完成/计划阶段、两种正确率、调用数、标签变化、混合奖励组、配对探索性区间。
- `/workspace/margent-rsi-pilot-v1/pilot_timeline.csv`：每个 SFT / GRPO 后的对照表。
- `/workspace/margent-rsi-pilot-v1/logs/`：每个子阶段日志。
- `/workspace/margent-rsi-pilot-v1/initial_gate.json`：首轮 rescue / commit 题数。
- `dynamic/round_1/grpo/step-*/step.json`：真实梯度、reward、advantage、合法率、训练 token 数。
- `dynamic/round_1/grpo/step-*/rollouts.json`：可审计轨迹与实际 token ID；advisor 文本只作上下文。

训练开始前，initial_dev 和 initial_collection 本来就可能耗时。CPU setup/replay 成功只是环境准备；只有出现 GRPO 的 `step.json`、非零梯度和至少一个 mixed-reward group，才有 outcome 学习信号的证据。

## 6. 停止条件与恢复

| 情况 | 怎么处理 |
|---|---|
| advisor 截断、网络失败、身份不匹配 | 停止，保留日志；修复环境后恢复。若改了生成设置，换新配置和新输出目录。 |
| 首轮少于 2 rescue 或 2 commit | 这是信息量不足的 pilot，不继续花训练预算；保留记录，再决定扩大 train 子集。不能挑掉“不好看的题”重跑同一 ID。 |
| all-zero / all-one reward group | 单个同奖组不是程序异常，但 outcome 优势为零。整个 GRPO 阶段没有混合奖励组时，本 pilot 停止；不要用失败重采样隐蔽地改变训练分布。 |
| Manager invalid / truncated 过多 | 优先诊断格式和长度，而不是解读准确率。改变长度后新建运行。 |
| 中断但 24 小时时限尚未过 | 在相同代码、配置、数据、advisor 下重新执行同一条 run。SFT 用 Trainer checkpoint；GRPO 用已原子提交的 optimizer+adapter。 |
| 24 小时耗尽 | 自动停止训练进程；以 incomplete 报告，不比较未完成组的“最终结果”。扩大预算属于下一次实验。 |

不同模型版本、prompt、数据或采样配置不允许接着旧目录跑。时间上限是最大运行时长，不保证三组一定完成。

## 7. 实验解释

Manager 的 greedy evaluation temperature=0 与 RL temperature=0.8 是两个字段。采集时用稳定状态做配对反事实；训练时用同题 4 条随机路线探索。advisor 设置独立且在三组间固定。

当前 pilot 只评估 16 dev 题，一题就是 6.25 个百分点。它用于检查闭环、梯度、预算和趋势；不能据此证明普适 RSI。正式论文还需要更大样本、多种子、独立 test 和 additional baselines，见[文献与实验设计](RSI_RESEARCH_DESIGN.md)。

本地已运行真实小模型的 LoRA 更新、采样概率一致性、断点恢复及下一轮 SFT 测试。A100/Qwen3.5-9B 的显存、吞吐和效果尚未验证，不把 CPU 测试当成远程 GPU 成功运行。
