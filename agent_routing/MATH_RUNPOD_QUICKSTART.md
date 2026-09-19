# 数学 RSI：从 RunPod 到 W&B 的分步教程

本教程对应 `codex/math-wandb-runpod` 分支。合并后也可使用 `main`。
目标是先确认环境、W&B 和短训练都正常，再启动正式两轮实验。
代码测试与 W&B 离线检查不能替代真实 Qwen3.5-9B GPU 实跑；显存与速度需要由下面的试跑确认。

## 1. 创建 Pod，打开终端

在 RunPod 的 Pods 中创建带 CUDA/PyTorch 的 GPU Pod，使用 Python 3.11 或 3.12。
首次正式 9B 试跑可从 **2 张 80GB GPU** 开始：GPU 0 运行冻结 Advisor，GPU 1 运行 Manager/训练。
这是保守的起跑配置，尚不是测量过的显存保证。当前代码只支持一张训练卡，不使用 `torchrun`；两张卡不是分布式训练。
只跑 0.6B smoke test 时可以共用一张卡，将后文训练命令中的 `CUDA_VISIBLE_DEVICES=1` 改成 `0`。

把数据、模型缓存、虚拟环境和 checkpoint 放在 `/workspace`，建议至少预留 100GB 并观察剩余空间。
需要跨 Pod 保留时，在创建 Pod 时挂载 network volume；普通 volume disk 随 Pod 删除而消失。
启动后从 Connect 打开终端或使用 SSH。建议用两个 tmux 会话保留长时间任务；镜像没有 tmux 时可安装它。
[RunPod 连接说明](https://docs.runpod.io/pods/connect-to-a-pod) · [存储说明](https://docs.runpod.io/pods/storage/types)

## 2. 下载代码、安装环境

下面按新 Pod、尚未存在 `/workspace/7.98` 编写：

```bash
cd /workspace
git clone --branch codex/math-wandb-runpod --single-branch https://github.com/Jeremyyny/7.98.git
cd /workspace/7.98/agent_routing
bash scripts/runpod_math_setup.sh
source /workspace/margent-venv/bin/activate
```

如果已有仓库，先保留本地改动，再 fetch/switch 到此分支；不要覆盖已有实验输出。
setup 会检查 CUDA、安装数学依赖与 W&B、检查依赖冲突、运行逻辑测试并保存环境信息。
出现错误先处理错误；末尾显示 `Ready. Activate with: ...` 表示这一阶段完成，不表示 GPU 训练已经通过。

## 3. 登录 W&B，保存终端配置

在激活环境后的终端运行：

```bash
wandb login
```

按提示在终端输入自己的 W&B API key。下面配置文件不保存 key。
把 `REPLACE_WITH_YOUR_WANDB_ENTITY` 换成 W&B 用户名或团队名，不是邮箱；该账号需能在这个 entity 下创建项目。

```bash
cat > /workspace/margent-env.sh <<'EOF'
cd /workspace/7.98/agent_routing
source /workspace/margent-venv/bin/activate
export HF_HOME=/workspace/hf-cache
export MARGENT_RUN_ROOT=/workspace/margent-runs
export MARGENT_DATA=/workspace/margent-data
export MARGENT_WANDB_MODE=online
export WANDB_PROJECT=margent-math-rsi
export WANDB_ENTITY=REPLACE_WITH_YOUR_WANDB_ENTITY
EOF
source /workspace/margent-env.sh
python -m src.verifiable wandb-check --out /workspace/margent-runs/wandb-check
```

检查成功会打印 W&B run 链接，网页显示 `check/logging_check=1`，最后状态为完成。
这个 run 仅验证日志连接，没有训练模型。失败时先检查登录、entity 和网络，不要先加载 9B 模型。
以后每个新终端、tmux 窗口或 Pod 重启后都先 `source /workspace/margent-env.sh`；凭据没有保留时重新 `wandb login`。

## 4. 两个终端完成 0.6B GPU smoke test

终端 A（Advisor；保持运行）：

```bash
tmux new -s math-advisor
source /workspace/margent-env.sh
CUDA_VISIBLE_DEVICES=0 python -m src.verifiable.serve --model Qwen/Qwen3-0.6B
```

等服务就绪后，打开终端 B（训练）：

```bash
tmux new -s math-train
source /workspace/margent-env.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_smoke.sh
```

两个会话在同一个 Pod；Advisor 默认端口 8000，不需要暴露成公共服务。
两轮完成后应生成 `/workspace/margent-runs/smoke-loop/loop_report.json`。
W&B 中会出现该实验的 group，以及 collection、SFT、RL 和诊断 run。
这是少量算术题与极短训练，只证明流程可执行，不是论文结果。

tmux 中按 Ctrl+B，然后 D 可退出会话界面，进程继续运行；回到训练会话用 `tmux attach -t math-train`。

## 5. 切换到 9B，先跑一个短周期

回到终端 A，Ctrl+C 停掉 0.6B Advisor，然后启动正式冻结 Advisor：

```bash
source /workspace/margent-env.sh
CUDA_VISIBLE_DEVICES=0 python -m src.verifiable.serve --model Qwen/Qwen3.5-9B --max-context 24576
```

终端 B 准备独立的 pilot 目录（32 道训练题、8 道开发题），并把 SFT/GRPO 各限制为 2 步：

```bash
source /workspace/margent-env.sh
python -m src.verifiable prepare --data-dir /workspace/margent-pilot-data \
  --train-size 32 --dev-size 8 --seed 42
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/math_rsi.json').read_text())
cfg.update(sft_max_steps=2, rl_max_steps=2)
out = Path('/workspace/margent-runs/pilot_config.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2))
PY
CUDA_VISIBLE_DEVICES=1 python -m src.verifiable doctor \
  --config /workspace/margent-runs/pilot_config.json \
  --out /workspace/margent-runs/pilot_environment.json
CUDA_VISIBLE_DEVICES=1 python -m src.verifiable loop \
  --config /workspace/margent-runs/pilot_config.json \
  --data-dir /workspace/margent-pilot-data \
  --out /workspace/margent-runs/pilot_9b --arm dynamic_rl --rounds 1 --resume
```

pilot 会实际经过 9B 采集、SFT、GRPO 和开发评估，能检查训练显存与工具调用。它不运行外部测试，也不用于论文主结果。
即使训练只设 2 步，采集和生成草稿仍然需要时间。观察 GPU 显存、每题耗时、输出截断率，以及是否有成功训练样本。
当前 2048-token 解题预算是起点，不保证够解竞赛题；若频繁截断，应调整 `configs/math_rsi.json` 的相关长度预算并重新生成 pilot 配置。
改了配置后使用新的 pilot 输出目录，例如 `pilot_9b_v2`；不能在原目录混用配置。

## 6. 正式跑一组、一个 seed

pilot 通过、固定正式配置后，准备正式数据。默认 1024 道训练题、256 道开发题，另保留 AIME 2026 和 BeyondAIME 外部测试：

```bash
python -m src.verifiable prepare --data-dir /workspace/margent-data \
  --train-size 1024 --dev-size 256 --seed 42
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_experiment.sh dynamic_rl 42
```

这会从基础模型开始，依次运行环境检查、初始开发评估、两轮采集/SFT/GRPO及开发评估，最后评估初始和末轮模型在两项外部测试上的表现。
它不接着 pilot checkpoint 训练，也不按外部测试分数选择 checkpoint。
输出在 `/workspace/margent-runs/dynamic_rl_s42`；Advisor 整个过程中继续在终端 A 运行，模型与 adapter 身份不变。

第一组完成并检查结果后，再顺序运行其他对照，避免多个训练进程抢同一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_experiment.sh dynamic_sft 42
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_experiment.sh static_rl 42
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_experiment.sh success_rl 42
```

需要多 seed 时把命令末尾改成 43、44，但沿用同一份正式数据，不重新切分。
先根据 pilot 和第一组实测耗时决定预算，不要假设四组成本相同。

## 7. 怎么看 W&B 和本地状态

打开 W&B 的 `margent-math-rsi` project，按 group 查看一次实验，用 `config.arm`、`config.seed`、`config.round`、job type 筛选。

| 要看什么 | 在哪里看 |
|---|---|
| SFT 是否收敛、RL 奖励是否变化 | SFT/RL run：`train/loss`、`train/reward` 等，横轴 `trainer_step` |
| 独立解题与协作能力随轮次变化 | loop run：`eval/independent_accuracy`、`eval/policy_accuracy`，横轴 `diagnostic_step` |
| 是否少调用、是否内化 | loop run：`eval/mean_calls`、`internalization/rescued_now_independent_rate` |
| 外部测试 | evaluate_suite run：`test/initial/*`、`test/final/*` |
| 是否 OOM 或卡住 | `gpu/*`、`progress/*`，以及下面的 status 命令与本地错误日志 |
| token 与请求耗时 | 每阶段 `usage/*`；完整跨阶段成本用最终报表 |

开发诊断轴：0=初始模型，1=第1轮SFT，2=第1轮RL，3=第2轮SFT，4=第2轮RL；`dynamic_sft` 只有0/1/2。
正确率和内化率是0到1的比例。Tracker 不会编造 Trainer 没有返回的指标。
完整解答、答案标签和权重仍保存在本地；W&B 只同步配置和标量。

在第三个终端可执行（不加载模型）：

```bash
source /workspace/margent-env.sh
python -m src.verifiable status --run-dir /workspace/margent-runs/dynamic_rl_s42
```

有心跳只说明进程仍响应，不代表训练在改善。错误详情在各阶段 `errors.log`，子进程输出在实验根目录 `logs/`。

## 8. 中断恢复、离线模式与导出

重新连接后，恢复相同 Advisor、相同配置、相同输出目录，再运行原命令即可：

```bash
source /workspace/margent-env.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/runpod_math_experiment.sh dynamic_rl 42
```

已完成阶段会跳过，未完成训练从最近保存的 checkpoint 继续；首次 checkpoint 前中断的部分会重跑。
在线 W&B 使用保存的 stage run ID 继续记录。保留整个实验目录，包括 `wandb_experiment.json`、`wandb_run.json` 和 checkpoint。
不要并行运行同一个实验目录。W&B 曲线恢复不等于模型恢复，两者分别由 run ID 与 Trainer checkpoint 控制。

没有网络时可先 `export MARGENT_WANDB_MODE=offline`，仍保留 `WANDB_ENTITY` 与 `WANDB_PROJECT`。
离线重启会产生不同 segment，不能假定它们已经合成一条曲线。网络恢复后登录，再同步每个离线目录：

```bash
find /workspace/margent-runs -type d -name 'offline-run-*' -print0 | \
  while IFS= read -r -d '' wb_dir; do wandb sync "$wb_dir"; done
```

离线同步只能上传实际记录下来的数据；缺失的训练历史不会补造。关闭 W&B 用 `export MARGENT_WANDB_MODE=disabled`。
W&B 初始化出错会直接报错；运行中的同步异常会提示并保留本地日志，查看 `status.json` 中的 `wandb_status`。

一组完成后即可导出论文表格与图（完成几组就填写几组目录）：

```bash
python -m src.verifiable report \
  --runs /workspace/margent-runs/dynamic_rl_s42 \
  --out /workspace/margent-paper
```

关键文件是 `paper_main.csv/.tex`、`paper_mechanism.csv/.tex`、`costs.csv` 和 `fig*.pdf/.png`。
保留运行目录与数据清单，检查结果后再停止或删除 Pod；W&B 不包含模型权重备份。
完整指标定义和实验限制见 [MATH_RUNPOD.md](MATH_RUNPOD.md)。
