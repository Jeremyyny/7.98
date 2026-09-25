# MARGENT RSI pilot 代码验证

日期：2026-09-25。

基准仓库：`https://github.com/Jeremyyny/7.98`。
基准分支：`codex/math-wandb-debug-tables`。
基准提交：`c008c45a23a9751486b737d78af2a7558be55862`，交付前重新查询 GitHub 分支 head，仍为这个提交。

本记录对应 RSI pilot 代码引入时的本地验证。运行方式见 [RunPod 启动说明](RSI_RUNPOD.md)。

## 已完成

- 完整测试：**116 passed，11 subtests passed，24.08 秒**。
- 其中新增 RSI 测试 10 项；启用了原有 opt-in CPU integration tests，未跳过它们。
- 真正执行随机小 Qwen 模型和 PEFT LoRA 的前向、反向、optimizer 更新与 adapter 保存。
- 对齐训练时 log-prob 与真实生成器的 temperature-softmax 概率。
- 检查奖励组归一化、clipping 与 KL、响应 token 位置及 prefix masking。
- 检查 COMMIT 不可变、无效决策不能获得正确奖励、禁止重复调用。
- 模拟中断并恢复 optimizer + adapter；本地 CPU 下与不间断训练权重完全相同。
- 把 GRPO 输出作为下一轮 SFT checkpoint，验证参数延续而非重新从 base 开始。
- 检查三组两轮计划、static 复用旧数据、train/dev 数据完整性、报告按题配对与退化计数。
- 原有 W&B 离线序列化测试、报告测试和 Qwen3.5 文本权重加载测试通过。
- Shell 语法与 git diff 空白检查通过。

测试环境：macOS / Python 3.12；torch 2.8.0、transformers 5.3.0、TRL 0.29.0、PEFT 0.18.1、datasets 4.8.5、math-verify 0.8.0、W&B 0.30.0。

完整测试启用了 `MARGENT_CPU_INTEGRATION=1`。W&B 测试使用离线模式和本机套接字；未发送科研结果至在线项目。

## 尚未验证

- RunPod 上 Qwen3.5-9B 的真实 A100 80GB 显存峰值和 throughput。
- 两轮三组能否全部在 24 小时上限内完成。
- 新配置下 advisor 是否在真实数学题上持续遵守长度和回答格式。
- 所提出的能力内化、动态标签刷新、选择性委派假设是否获得支持。

随机小模型与 scripted reward 测试仅验证实现。它们不是数学能力实验，不能进入论文主结果表。
