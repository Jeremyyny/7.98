# MARGENT：从一次性选择性委派到递归学习

检索截止：2026-09-25。最近六个月按**首次公开日期 2026-03-25—2026-09-25**计算。本文是一份文献审查和预先实验设计，尚无新的 GPU 实验结果。

## 1. 建议增加的核心贡献

**检验 MARGENT 是否能在反复自生成经验、SFT、GRPO 的过程中，随 Manager 能力变化重新校准委派价值，并把有用的外部帮助转化为独立能力。**

建议标题：**Marginal-Value Recalibration for Iterative Agent Self-Improvement**。

你的原稿《MARGENT: Measuring the Marginal Value of Delegation in Agentic Systems》已经给出了三块基础：

- §3.2：在同一个存储候选答案上，比较 COMMIT 与“子 agent 建议＋Manager 修订”的正确性差值。
- §3.3：一次性采集后训练，原稿明确没有在 SFT 后刷新反事实数据。这正是本扩展要改变的地方。
- Appendix D / Table 10：两个 outcome-only GRPO continuation 的调用数逼近上限（2.98、3.00）。原稿明确这些运行缺失准确率记录，所以只能支持“调用行为退化”，不能支持“准确率下降”。

原来的 candidate→policy 改善是**同一训练后 Manager 的推理时委派收益**。新实验要额外测量 checkpoint→checkpoint 的独立能力增长；二者不能混作一个提升数字。

### RSI 的命名边界

固定训练程序下，更新后的模型生成下一轮经验，再更新自身，可以作为**有界、参数层面的迭代自改进**来研究。它有递归反馈，但不能仅凭两轮 SFT→GRPO 就宣称实现了自主研究、改进学习算法或递归自加速。正文宜写“supports an iterative agent self-improvement loop”；若使用 RSI，先给出这个受限定义。

## 2. 最近六个月值得优先读的工作

“有名”需要与“刚发表、很相关”区分。这一批 7—9 月论文多数仍是预印本，尚不能依据稳定引用量宣称领域共识。本清单按与实验的相关性、团队和公开实现筛选，**不是引用榜单，也不是穷尽检索**。公开仓库的关注量只作辅助背景，不能替代研究质量判断。

| 工作 / 首次公开 | 他们更新什么、怎么做 | 对 MARGENT 的直接启发 |
|---|---|---|
| **SESA: Self-Play Meets Skill Evolution**，2026-07-31 | Challenger 出题，Solver 用搜索和技能记忆解题；失败被蒸馏为技能，反馈到后续自博弈和 GRPO。 | 最直接的方法参考：关闭技能库仍然保留多少增益？对应我们的“关闭子 agent 后独立正确率”。[论文](https://arxiv.org/abs/2607.29468) / [作者代码](https://github.com/Zenghuang-Fu/SESA-Self-Evolving-Search-Agents) |
| **RSIBench-Data**，2026-07-28 | Agent 根据训练与验证反馈调整数据策略；固定目标模型、训练服务和评分设施。每次候选从固定 base 做 LoRA SFT，并非持续在上一候选权重上接着训练。 | 把数据选择与训练设施隔离；同时报告末轮、验证集选出的最好 checkpoint、预算和退化。[论文](https://arxiv.org/abs/2607.25886) / [作者代码](https://github.com/evolvent-ai/RSIBench-Data) |
| **SEAGym**，2026-06-16 | 在 Terminal-Bench 2.0、HLE 上统一评估 ACE、TF-GRPO、AHE 等 harness 更新；保存训练、验证、ID/OOD、回放与成本记录。 | 不能只画一条训练准确率上升曲线；必须检验跨题迁移、遗忘与成本。其 TF-GRPO 是 harness 侧方法，不能与本实验的参数 GRPO 混同。[论文](https://arxiv.org/abs/2606.17546) |
| **PAST-Bench**，2026-08-04 | 在 fresh-session 任务序列里，匹配条件地开/关保留经验；同时检查经验保存、检索、更新的证据。 | 最终涨分之外，还需显示“旧失败→新经验→新策略”的机制链条。[论文](https://arxiv.org/abs/2608.04003) / [作者代码](https://github.com/Gen-Verse/PAST-Bench) |
| **EvoAgentBench**，2026-07-06 | 从执行轨迹抽取可迁移的能力，建立能力支持的 train/test 任务；检验经验内容、路由与使用。 | 看未训练题上的能力迁移；把“有帮助的经验存在”与“Manager 会使用”分开。[论文](https://arxiv.org/abs/2607.05202) |
| **Metaⁿ**，2026-08-25 | 固定元操作 Ω，读取已有 solver 栈的代码与执行轨迹，递归生成更高层策略及工具；与 flat refinement、去递归版本比较。 | 必须有去递归/不刷新对照，不能用多跑几轮替代机制证明。[论文](https://arxiv.org/abs/2608.24735) / [作者代码](https://github.com/minnesotanlp/meta-n) |
| **AIDE²: Recursive self-improvement of AI research agents**，2026-09-22 | 修改 research agent 自身代码，按固定预算评估候选并保留改进，再评估完全未参与选择的任务。 | 采用固定预算和独立的最终测试；区分选择集进步与泛化。刚发表三天，应称“最新工作”，不能称“公认经典”。[论文](https://arxiv.org/abs/2609.26457) |
| **Agentic Harness Engineering (AHE)**，2026-04-28 | 让 agent 编辑可回滚的 harness 组件，提炼历史轨迹证据，每次修改先预测效果，再用下一轮任务结果检验；冻结改进后的 harness 做跨任务/模型迁移。 | 每个修改都要有可证伪的作用假设，并做组件消融；不能仅用总分归因。[论文](https://arxiv.org/abs/2604.25850) |
| **SEA-Eval**，2026-04-10 | 连续任务流里同时考察成功率与 token 消耗、演化稳定性。 | 调用更少不自动意味着更高效，准确率更高也可能只是花费更多推理预算。[论文](https://arxiv.org/abs/2604.08988) |

两个补充阅读：

- **Active-GRPO**（2026-07-01）：分子优化里按当前策略相对参考的表现，调整模仿/强化并更新参考。它提供“能力改变后参考价值也应改变”的相邻思路；任务不是 agent RSI，不能当同任务 baseline。[论文](https://arxiv.org/abs/2607.00531)
- **Self-Improvements in Modern Agentic Systems: A Survey**（2026-07-14）：区分参数、提示、记忆、工具和控制逻辑等更新对象，适合 related work 的分类框架。[论文](https://arxiv.org/abs/2607.13104)

截至本次查询，作者仓库 RSIBench-Data 为 164 stars，Metaⁿ 为 32，PAST-Bench 为 29，SESA 为 18。它们是值得跟进的相关新作；这些数字不足以支持“都是最著名 RSI 论文”的说法。

### 窗口外但必须了解的参照

- **Hyperagents**，2026-03-19，比严格窗口早六天。把 task agent 与可自修改的 meta agent 放在同一可编辑程序中，考察改进过程本身的变化及迁移；作者包括 Jenny Zhang、Jeff Clune、Jakob Foerster 等。它是强 RSI 主张的重要参照，单列而不改动日期。[原论文](https://arxiv.org/abs/2603.19461)
- **Agent0**，2025-11-20。课程生成者与执行者通过工具任务共同演化；在 2026 RSI Workshop 有 oral 展示，但首次论文并不属于最近六个月。它说明“如何更新训练任务”是可研究的闭环组件。[原论文](https://arxiv.org/abs/2511.16043) / [workshop 官方日程](https://recursive-workshop.github.io/)
- **Tool-R0**，2026-02-24。Generator / Solver 自博弈，从零外部训练题构建工具学习过程。我们的 Numina 有答案题池与它的数据假设不同，不能宣称 zero-data。[原论文](https://arxiv.org/abs/2602.21320)
- **GEPA**，2025-07-25，ICLR 2026 Oral。通过轨迹反思、提示变异与 Pareto 选择优化 LLM 系统，是较成熟的相邻基线参照；它更新提示而非参数，不是 SFT/GRPO 交替的证据。[原论文](https://arxiv.org/abs/2507.19457)

### 五篇精读的实际实验做法

**SESA**：同一个训练后的 Solver 分别开启和关闭技能库，再对照无技能库的自博弈基线，从而分离参数内化与推理时检索收益。它还消融初始记忆、难度塑形、失败蒸馏。我们借鉴这种拆分，但冻结 advisor，不同时演化 advisor，以免无法归因。其训练是 GRPO，不能引用它证明所有 RSI 都必须交替 SFT/GRPO。[方法与实验](https://arxiv.org/html/2607.29468v1)

**RSIBench-Data**：让研究 agent 调整数据策略，训练服务从固定 base 产生候选；选择反馈和正式测试分离。论文报告多次迭代可能在出现最好候选后继续退化。我们因此锁定两轮最后 checkpoint 为 pilot 终点，完整保留 SFT 后与 GRPO 后结果；正式实验另外用 dev 选 best，并在独立 test 上比较。[方法 §3](https://arxiv.org/html/2607.25886v1)

**SEAGym**：将任务执行与演化更新分离，保存不可变快照，并分别评估验证、迁移、旧任务回放和成本。对我们而言，训练题上的 rescued→direct 只是机制诊断，held-out 的 direct accuracy 才能支持能力泛化。[方法 §3](https://arxiv.org/html/2606.17546v1)

**Metaⁿ**：不是把同一训练脚本重复执行，而是让后续元层看到前面层的代码与轨迹；去掉递归条件、控制计算预算，检查增益来源。我们的对应消融是固定首轮反事实数据；它只消融经验刷新，不消融 GRPO 自身的 on-policy 更新。[方法 §2 / 消融 §3.4](https://arxiv.org/html/2608.24735v1)

**AIDE²**：保留在选择任务上胜出的 harness，再用没参与选择的四组 benchmark 验证泛化；固定每任务预算。其内层 agent 改进明显，但论文也承认，改进后的 agent 作为外层自改进者是否更强，证据尚不能明确区分于强基线。我们也应避免把 solver 变强说成“改进者自加速”。[实验 §3](https://arxiv.org/html/2609.26457v1)

## 3. 预注册假设与实验分组

令第 t 轮 Manager 为 M_t，冻结 advisors 为 A。训练题的当前候选状态为 s，定义：

Δ_t(s,a) = C(revise(M_t, s, A_a(s))) − C(commit(s))。

C 使用独立答案校验器；advisor 的 Verdict 不是 reward。每个 Δ 是固定解码种子下的配对结果，不能当成对采样分布期望的无偏估计。正式实验应扩展多生成种子。

要检验三个假设：

1. **H1：自适应价值。** Manager 改变后，部分题从需帮助变成能独立解，部分路由标签发生变化；刷新数据优于继续使用旧标签。
2. **H2：能力内化。** 独立解题正确率在 held-out 题上提升，而不只是允许更多调用后分数提升。
3. **H3：选择性保持。** 达到可比的最终正确率时，不必要委派较少，尤其是在当前已经独立做对的题上；同时不牺牲本来需要帮助的题。

| Pilot 组 | 每轮 SFT 数据 | GRPO | 用来回答什么 |
|---|---|---|---|
| dynamic | 用当前 Manager 重新生成反事实树，选 COMMIT 或最短成功路线 | 当前 SFT checkpoint 上的 on-policy GRPO | 完整方案 |
| static | 每轮都重用 M₀ 的首轮 MARGENT 标签和轨迹 | 完全相同 | 新鲜经验是否有用？ |
| success | 当前 Manager 的同一搜索空间中随机选成功路线，不按最短路线优选 | 完全相同 | 是否只是成功样本蒸馏？ |

三组从同一个 base 开始，使用同一训练题集合、advisor、种子、深度、SFT 步数、梯度累积和 RL rollout 数。首轮共享同一棵反事实树。dynamic 与 static 首轮应接近一致，是实现 sanity check。

为了避免 static 少花采集成本造成不透明，第二轮 static 同样采集一棵 **shadow tree**，用来计量成本和标签变化；训练仍只读取首轮标签。应同时报告“实际执行成本”和“部署 static 时可省掉 shadow 采集的成本”。该设计匹配采集机会与优化更新数，**不是等 token 或等 FLOPs**。

success 继承现有采集器的可解题筛选和 commit/rescue 配比，再在保留题内随机选成功路线。因此它是“匹配覆盖/配比后的普通成功路线对照”，不是无限制的所有成功轨迹蒸馏。

正式论文还需要的对照（不塞进首个 24 小时 pilot）：

- 相同 warm-start 后只继续 outcome-GRPO：定位每轮 SFT 刷新的作用。
- 只做动态 SFT：说明交替 RL 是否必要，而不是仅仅更多训练。
- 去掉 question-only solution distillation：区分路由学习与直接解题监督的贡献。
- 相同状态下 self-revision、相同预算的更多独立采样：排除额外推理计算本身的贡献。
- 至少三个训练种子、三轮以上、独立最终 test，以及第二个模型或 agent 任务。

## 4. 24 小时 pilot 的实际规模

硬件：已有两张 A100 80GB。GPU 0 冻结 advisor；GPU 1 跑 Manager 的采集、SFT 和 GRPO。三组顺序跑，避免训练进程抢显存。

- 从既有、带哈希 manifest 的 Numina 分割里，按题目哈希选择 **16 train / 16 dev**；不按答题结果挑题。
- 两轮、三组、一个训练种子。每阶段 SFT 8 optimizer steps，accumulation 2；GRPO 8 groups/steps，每题 4 条采样。
- 最大委派深度 2。当前数学采集器穷举最多 3+6 条分支，**不同于原稿 BFS 成功层即停**；三组一致，必须披露。
- Manager 独立答案和修订预算 2048 tokens，决策 128；advisor 2048；总上下文/训练序列上限 32768。截断会被记录，advisor 截断直接报错；不能把截断后缺答案当可靠数学负例。
- 评测 / 反事实标注 Manager temperature=0。GRPO Manager temperature=0.8，full-softmax，无 top-k / top-p 裁切或重复惩罚；采样和 loss 使用相同分布。
- advisor 固定为 temperature=0.7、top_p=0.8、top_k=20、presence_penalty=1.5、seed=42。这沿用已有 replay 的可运行候选设置；单次 replay 成功不意味着已证明稳定。三组固定一致，不因答案正确与否重试。
- advisor 是同一个冻结 Qwen3.5-9B 的三个角色提示，**不是原稿的三个角色专用 LoRA**。本 pilot 是数学迁移设置，不能说完全复现原稿专家配置。
- 首轮至少需要 2 个 rescue 和 2 个 direct-correct/COMMIT 题，否则停止并报告样本不足。改变题数、长度或配置后换新运行目录，保留失败记录。
- 每个 GRPO 阶段至少应出现 1 个混合奖励组；若全阶段同奖，则停止后续预算消耗并报告缺乏 outcome 学习信号。
- 控制器启动后最多运行 24 小时，到时终止当前子进程并保留已完成检查点。**这是时间上限，不是三组必定完成的保证，也不会自动关闭 RunPod 计费或 advisor 服务。**

原来的 128 train / 64 dev 数据不重建、不删除；只是抽取新的小子集。AIME2026 和 BeyondAIME 在 pilot 中保持封存。

### 闭环

M₀ → 首轮反事实采集 → SFT₁ → GRPO₁ → 用 M₁ 重新采集 → SFT₂ → GRPO₂。

SFT₂ 必须加载 GRPO₁ adapter；不能从 base 重启。每个 SFT 和 GRPO 后都在同一 dev 集评测 direct 与 policy 两种模式。pilot 没有 test-based checkpoint selection。

### GRPO 的具体定义

一个 group 共享当前模型贪心生成的初始候选，采样的是 Manager 决策与修订。奖励为协议有效且最终答案正确的 0/1；不加调用惩罚。初始候选是条件状态，不对它做 RL loss；SFT 中的独立 solution 蒸馏训练独立解题。

每组优势使用 population standard deviation 加 1e-4；相同奖励组优势全零。保存实际生成 token IDs（包含 EOS），按精确的 prompt/response 边界计算 log-prob。工具回复仅作上下文。损失是逐 token clipped surrogate 加 KL k3 近似，先按每条轨迹的 Manager token 数归一化，再在组内平均。每组做一次 optimizer update，clip=0.2，beta=0.01，lr=1e-6，AdamW weight_decay=0，grad clip=1。

KL reference 是**该 GRPO 阶段入口的 SFT adapter**，同一冻结 base 上挂第二份小 adapter，不额外加载第二个 9B。下一轮会重新定义 reference；这不是全程固定初始模型的 KL。

这是保留标准 GRPO 目标的简洁单 GPU 实现，不是调用旧 TRL environment。旧入口继续关闭，避免把旧协议误当成新实验。pilot 优先审计性，未做 vLLM、并行 rollout 或吞吐优化。

## 5. 应输出什么，怎样判断值得扩大

每个阶段至少报告：

| 指标 | 含义 |
|---|---|
| independent_accuracy | 完全不调用 advisor 的正确率；检验参数内化 |
| policy_accuracy | 使用同一 checkpoint 的实际路由策略最终正确率 |
| policy − independent | 该 checkpoint 的推理时委派收益，不能替代学习增益 |
| mean_calls | 平均调用数；要和正确率一起看 |
| currently_independent_call_rate | 已独立做对仍然调用的比例 |
| correct→wrong / wrong→correct | 委派的破坏和救援分别计数 |
| independent_new / independent_regressed | 对齐同一 dev 题后的新学会与遗忘 |
| preferred_label_changed_n | 训练题的最优路线标签变化；仅是机制诊断 |
| mixed_reward_groups | GRPO 真正有非零 outcome 优势的组数 |
| invalid/truncated | 格式失败、重复调用、超长等；不能隐藏 |
| 实际 tokens / stage wall time | 搜索、采样、训练成本分别记录 |

值得继续的条件：完整跑通两轮；存在非零梯度和混合奖励组；主要结果不是由截断/非法格式支配；刷新标签确有变化；held-out direct 或 policy 出现可解释的正向趋势，同时调用成本没有无节制增长。

这些条件是 **go/no-go 诊断**，不是统计显著性或论文接受门槛。16 个 dev 题中一题就是 6.25 个百分点；一两题波动不能支持“大幅提升”。如果三组都没变化、RL 全同奖、或动态组不优于 static，应报告“不足以支持假设”，优先检查信号、任务难度和预算。

正式实验：按题配对 bootstrap 置信区间，多 seed 同时报告均值与离散程度；test 只用于预定初始/最终及 dev-selected-best checkpoint 的锁定评估。AIME 只有 30 题，应单独报告计数，不把反复调参后的 AIME 当未见测试。

建议图：①每个 SFT/GRPO checkpoint 的 independent / policy 双曲线；②准确率—实际 token 成本；③同题 rescued→direct 与 direct→wrong 迁移；④当前可独立解题的调用率；⑤更新标签与 static 旧标签的差异。不要只画最终最高分。

## 6. 可以怎样写进论文

当前可写的方法性表述：

> We extend MARGENT to a bounded iterative agent-learning loop in which each updated Manager generates the counterfactual experience used by the next supervised update. Alternating these updates with outcome-only GRPO lets us test whether refreshing marginal delegation values improves independent problem solving while preserving selective delegation. Frozen advisors, matched training schedules, static-data controls, and successful-trajectory controls isolate the contribution of marginal-value recalibration.

这段只描述设计，没有声称结果。结果出来后，只有 H1/H2/H3 分别得到对应证据，才补相应结论。若只是调用下降而准确率下降，不能写“更高效的 RSI”；若只有训练题直接正确率上升，不能写泛化内化；若只增加 best checkpoint 而末轮退化，要同时报告。

## 7. 代码交付范围

基于 `Jeremyyny/7.98` 的 `codex/math-wandb-debug-tables` 分支、基准 commit `c008c45a23a9751486b737d78af2a7558be55862` 扩展。新入口是 `python -m src.verifiable.rsi`，不复用旧 SFT-only loop。

- `rsi_grpo.py`：真实多轮策略梯度、token masking、阶段 SFT reference、optimizer/adapter 原子恢复。
- `rsi.py`：子集验证、三组两轮计划、时间上限、按阶段隔离执行、结果和标签漂移报告。
- `math_rsi_pilot.json`：已冻结模型 revision、采样和训练配置。
- `runpod_rsi_pilot.sh`：使用已有 venv/cache/data 的 advisor、plan、run、report 四个入口。
- `tests/test_rsi.py`：真实小模型梯度、恢复一致性、下一轮 SFT、采样分布与 scoring 一致性、协议、数据和计划测试。

附带的代码与本地 CPU 测试不是 A100 实验结果。远程 GPU 吞吐、显存峰值、advisor 稳定性和科研假设仍需 pilot 验证。具体启动命令见 [RunPod 启动说明](RSI_RUNPOD.md)。
