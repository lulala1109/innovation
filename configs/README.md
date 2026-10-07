# 配置状态索引

## 当前 7B RQ1 主线

- 权威只读 catalog：`stage1_rq1_qwen7b_short_sure_here_is_run01_frozen.json`
- 正式交接文档：`../docs/RQ1创新点1_7B正式交接文档_2026-10-02.md`
- 状态：18/18 阶段 complete，深度验证 `VALID`，28 层、3584 维

恢复和 RQ2 bundle 绑定必须使用上述 7B frozen config。

## 3B 历史归档

- `stage1_rq1_current_frozen.json`：名称中的 `current` 是旧 3B 时期遗留；当前仅作历史只读 catalog。
- `stage1_rq1_v2_TEMPLATE.json`：旧 3B 模板，不是当前 7B RQ1/RQ2 的默认配置。

不要重命名、覆盖或原地修改历史 frozen config；文档和命令直接引用当前 7B frozen config，
以免破坏已有 provenance。

## RQ2

- `rq2_candidate_layer_preregistration_qwen7b_v1.json`：T0.2 原始机器预注册记录；作为 v1 历史依据保留，只适用于当前 28 层 7B frozen bundle，不得原地覆盖。
- `stage2_rq2_TEMPLATE.json`：schema v3 的 7B RQ2 配置模板；通过 `preregistration.path/sha256` 绑定上述记录，复制到新文件、填写正式 manifest 后使用。
- `advbench_rq2_clean_7b.json`：AdvBench 7B clean baseline 筛选配置，不是正式 RQ2 role manifest。
- 455 条保守候选集由 `python -m data.build_rq2_conservative_pool build` 生成，并用同模块的 `check` 只读验收；它明确跳过人工语义与音频内容复核，仍不是正式 role manifest。
- T0.6 划分由 `python -m data.build_rq2_split build` 按 seed 42 确定性生成，`check` 只读验收；冻结计数为
  `rq2_dev=20`、`rq2_causal_test=40`、`reserve=395`，产物在 `dataset/processed/rq2/advbench_split_v1/`。
- T0.7 正式 source manifest 由 `python -m data.build_rq2_manifest build` 生成，`check` 只读验收；正式输入是
  `dataset/processed/rq2/advbench_manifest_v1/rq2_manifest.jsonl`，共60条并固定 `target_text="Sure, here is"`。
- `stage2_rq2_qwen7b_advbench_sure_here_is_run01.json`：T0.8 正式可执行配置，指向上述 manifest，使用
  `trajectory.mode="generate"`；已通过 schema v3、预注册、`plan` 与首次 `validate`，尚未启动任何实验阶段。

### 层角色与剂量修订 v2

- `rq2_candidate_layer_preregistration_qwen7b_v2.json`：继承并绑定 v1 父记录 SHA，冻结本次层与剂量规则。
- `stage2_rq2_qwen7b_advbench_sure_here_is_v2_run01.json`：schema v4，新输出目录为 `outputs/stage2_rq2/qwen7b_advbench_sure_here_is_v2_run01`；后续 v2 工作使用此配置，不覆盖 v1 run。
- `stage2_rq2_qwen7b_advbench_smoke01.json` + 同名前缀 `_manifest.jsonl`：独立 smoke run；按冻结 dev `split_rank` 取前两条，执行层 `[19,24]`。源 manifest SHA 和 v2 预注册 SHA 均绑定；代码只允许执行至 `mechanism_analyze`，不能进入 protocol lock/formal/event/report。
- 候选层 `[19,24,26,27]`，邻层 `[18,20,23,25]`，远距层 `[2,12]`；远距层不再称为深度匹配对照。
- 主 restoration / suppression 剂量均为 `1.0`；`0.5` 仅作 dev pilot 敏感性分析，`1.5` 不进入 v2。
- v2 已通过轻量配置/预注册一致性检查，未运行测试或实验。代码现已加入 pilot 稳定性门禁、formal 每比较有效 N≥20、预定 F1–F5 BH family 及三种事件各自的 E1 同人口配对 offset family；运行时需在正式 dev pilot 后生成绑定 40 个 causal-test pair 和 family 槽位的 `protocol_lock.json`。目前没有真实 protocol lock 或 formal 结果。
- v2 配置中沿用的 `formal.minimum_pairs=20` 只是生成前的低限核查，不代表正式人口可少于 40；`protocol_lock` 会另外要求冻结的 40 个不同 causal-test pair，分析阶段再逐比较核验有效 N≥20。
- 详细状态见 [v2 修订协议](../docs/RQ2_7B_v2_实验协议修订_2026-10-04.md)。

smoke 配置已存在；需要执行时先用 `python experiments/run_stage2_rq2.py --config configs/stage2_rq2_qwen7b_advbench_smoke01.json plan` 只读查看阶段，再按 v2 协议执行真实 smoke。本轮未启动 GPU/Judge。\n\nv2 预注册只读验收：

```bash
python experiments/preregister_rq2_layers.py \
  --record configs/rq2_candidate_layer_preregistration_qwen7b_v2.json \
  --document docs/RQ2候选层与剂量预注册_Qwen2.5-Omni-7B_v2_2026-10-04.md check
```

v1 预注册只读验收（默认仍为 v1）：

```bash
python experiments/preregister_rq2_layers.py check
```

若模型、RQ1 bundle 或任一冻结输入变化，必须生成新版本记录并更新配置引用，不能覆盖 v1。

