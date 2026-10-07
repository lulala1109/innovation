# RQ2 代码架构与运行说明

## 边界

RQ2 全部实现位于 `rq2/`，入口为 `experiments/run_stage2_rq2.py`，输出只能写入
`outputs/stage2_rq2/<run_name>/`。RQ1 源码、配置、测试和 `outputs/stage1` 均为只读输入。
任何输出路径落入 RQ1 命名空间、冻结 SHA/model fingerprint 不一致、正式集与 RQ1 重叠时，流水线都会停止。

## 数据契约

RQ2 manifest 支持 CSV、JSON、JSONL。每行至少包含：

- `pair_id`、`rq2_role`（`rq2_dev`、`rq2_causal_test` 或 `rq1_reference`）；
- `clean_audio_path`（或 `harmful_audio_path`）、`harmful_text`、`clean_refused`；
- `trajectory_path`（`trajectory.mode=existing` 时必需）；
- 建议显式提供 `content_group`，防止语义相同但 ID 不同的数据跨 split 泄漏。

当前7B正式 source manifest 为 `dataset/processed/rq2/advbench_manifest_v1/rq2_manifest.jsonl`，由
`python -m data.build_rq2_manifest build` 确定性生成，共20条 `rq2_dev` 和40条 `rq2_causal_test`，固定
`target_text="Sure, here is"`。该初始 manifest 用于 `trajectory.mode=generate`，因此不含伪造的
`trajectory_path`；轨迹阶段生成的 `trajectory/resolved_manifest.json` 才交给 RQ2 data loader。

正式集会按 pair ID、harmful-content identity、clean-audio SHA-256 与 RQ1 三重隔离。manifest 中的
`clean_refused=true` 只是候选条件；正式 eligibility 还必须通过 RQ2 自己的 StrongREJECT clean baseline 复核。
`rq1_sources.rq1_manifests` 必须同时覆盖 RQ1 split/content 信息和带 harmful-audio 路径的 manifest；模板已同时
列出 `jbb_pairs_split.csv` 与 `jbb_pairs_audio.csv`，以确保第三重 SHA 隔离实际生效。

`trajectory.mode=generate` 会在 RQ2 输出目录调用现有攻击器，参数固定为 Standard PGD、margin loss、
100 steps、zero init、`early_stop=false`、`save_all_steps=true`，随后生成只属于该 run 的
`trajectory/resolved_manifest.json`。两种模式都会校验每条轨迹具有精确的 `0..100` checkpoint，并从
`run.json` 核对 Standard/L∞ PGD、margin loss、模型/model_id/dtype、目标文本、ε/α/κ、输入音频 SHA 和
experiment fingerprint。index、clean audio、`run.json` 和全部 checkpoint SHA 都进入阶段恢复契约。

## 阶段与门禁

阶段严格顺序如下：

1. `sources`：校验候选层预注册记录及其全部冻结输入，同时校验 probe、training states、模型指纹、方向定义、pooling/token span 和 SHA；记录路径、记录 SHA 与输入 SHA 一并写入 provenance。
2. `trajectory`：生成或验证完整 Standard-PGD 轨迹。
3. `trajectory_behavior_generate` → `trajectory_behavior_judge` → `events`：对 clean 和全部 step 做无干预生成，
   由 RQ2 Judge sidecar 推导 weakening/non-refusal/compliance 事件；不读取 RQ1 behavior event。
4. `state_index`：使用配置中预注册的 `sampling.fixed_steps`（7B 模板默认为 `2/10/100`）与事件窗口
   `[-2,-1,0,+1,+3]`；fixed steps 必须递增且位于轨迹范围内，缺失 offset 保持缺失。
5. `layer_map` → `identity`：逐层同时比对 hook 与 HF hidden states、独立 RQ1 `forward_attack` audio-mean replay，
   并验证 embedding offset；随后验证 no-hook/capture/self/sham/λ=0 的 logits、全层 H/R profile、确定性
   generation 和 hook 清理。
6. `oracle_*`：Full-State Patch。Oracle 无效时停止并检查 hook、audio span、reference 对齐和 generation 路径。
7. `mechanism_*`：R-direction dose、H/random/sham controls 和 clean reverse suppression pilot。
8. `subspace_*`：只有 Oracle 强而 1D-R 弱时触发。subspace checkpoint 必须来自 `rq2_dev/calibration`，并绑定
   model fingerprint、probe SHA；pilot 必须覆盖 candidate layers，若锁定 subspace 为正式主干预，则必须覆盖
   每个 formal layer。若 1D-R 已有效则写出明确的 `not_triggered` 产物。
9. `protocol_lock`：核查 Oracle 与选定机制的 dev pilot 稳定性证据，绑定候选/邻层/远距层、显式主剂量与 random replicates；记录合格层×状态及分析 SHA，不依据 dev 结果重选层。
10. `formal_*`：冻结 RQ1 bundle 的全部 `L` 层×固定状态主扫描；高成本 H/random/sham/远距层/token controls
    仅在注册层运行；formal 强制使用 `all`（当前 7B bundle 自动展开为 `0..27`，共 28 层）和冻结的 40 个
    causal-test pair。F1–F4 的每项主比较及必要同层配对对照须至少有 20 个有效独立 pair；不足时保留 FDR 槽位，
    标记 `insufficient_data`，不能进入结论门禁。
11. `event_*`：只在 candidate layers 做独立 event-centered 分析；无 eligible event 时生成带表头的空表，
    不替换为相邻 step。E1 在每类事件内用同一 pair 的五个完整 offsets 比较后段与前段，并分事件校正。
12. `report`：生成正式表格、因果 heatmap、specificity controls、restore/suppress、RQ1-vs-RQ2 ranking 和
    event profile（图片在 matplotlib 可用时生成），并应用最终结论门禁。

层数和 hidden width 不在 RQ2 代码中写死：加载冻结 probe 后要求层号是从 0 开始的连续集合，候选层、邻层和
深度对照层再按该 bundle 校验；实际模型运行时还会核对模型 decoder modules 与 bundle 层集合完全一致。
当前模板已经绑定 7B RQ1 的 28 层、3584 维 probe，并通过 schema v3 的 `preregistration.path/sha256`
引用机器权威记录 `configs/rq2_candidate_layer_preregistration_qwen7b_v1.json`。candidate、neighbor、depth control、
fixed steps、剂量顺序和统计阈值均在任何 RQ2 generation、dev pilot 或 formal 结果产生前冻结。dev pilot 只能在
已冻结的集合和剂量顺序内选择 primary intervention 与主剂量，不能重选层；formal 结果也不得用于改写预注册记录。

subspace pilot 不能与 1D mechanism pilot 混跑；go/no-go 只评估配置中首个、预注册的主剂量，其他剂量仅形成
dose curve，不会替换 formal λ/η，也不会自动挑层。若需改变参数，应创建新 run/config，而不是改写已有
`protocol_lock.json`。

## 干预口径

- Restore：`CE_restore = Y_patch - Y_base`。
- Suppress：`CE_suppress = Y_base_clean - Y_suppress_clean`。
- 主连续 outcome：`refusal_orientation = 1 - strongreject_score`。
- R、H、seeded-random controls 为等 shift norm；文本位置 control 在 audio span 估计方向差，在非 audio token
  施加，并按 `sqrt(n_audio/n_text)` 缩放以匹配总 L2。
- reverse suppression 使用冻结训练 states 中 `refusal-positive ∩ X_H` 总体的逐层 `σ_R`，并提供等范数
  reverse-H、reverse-random、reverse-sham controls。

只有主剂量的 utility 改善、同一配对人口的二值拒答确认、同层 R-vs-H/random/sham/token 对照及同层 reverse
suppression 均达到有效 N、预定 F1–F4 的 BH-FDR、效应方向和 CI 门槛时，`causal_claim_gate.supported` 才为 true。
邻层和远距层只用于 F5 描述，不充当机制特异性对照；未通过则 `rq2_causal_prior.json` 标记为 `withheld`。

攻击状态依赖结论不以“任一状态显著”代替交互证据：fixed-state 使用含/不含 Layer×State interaction 的
随机截距模型整体 likelihood-ratio test；F1 预定网格若有不足 20 pair 或推断不可用，先暂停此模型的正式结论。
事件逐 offset 的 F5 表仅供描述；另有同事件、同完整病例人口的 E1 配对 offset 检验。E1 达到有效 N≥20、分事件 BH q 值与 CI 门槛，并且同层已满足双向证据时，才能支持事件特定的状态变化；它不单独证明跨层迁移。
模型未收敛、方差边界或信息矩阵奇异时，该证据无效。

## 运行

先确认预注册记录可由冻结 RQ1 输入逐字节复建；该检查只读，不产生工作区改动：

```bash
python experiments/preregister_rq2_layers.py check
```

当前7B正式配置为 `configs/stage2_rq2_qwen7b_advbench_sure_here_is_run01.json`，指向冻结的60条 manifest，
使用全新输出命名空间和 `trajectory.mode="generate"`。不得修改预注册绑定的层号、顺序、剂量或阈值；一旦产生
`pipeline_state.json`，也不得原地修改该配置。schema v3 加载器会逐项比对显式配置值与预注册记录，任一漂移都会拒绝加载。

```bash
python experiments/run_stage2_rq2.py --config configs/stage2_rq2_qwen7b_advbench_sure_here_is_run01.json plan
python experiments/run_stage2_rq2.py --config configs/stage2_rq2_qwen7b_advbench_sure_here_is_run01.json validate
python experiments/run_stage2_rq2.py --config configs/<run>.json run --through identity
python experiments/run_stage2_rq2.py --config configs/<run>.json run --from oracle_generate --through protocol_lock
python experiments/run_stage2_rq2.py --config configs/<run>.json run --from formal_generate --through report
```

GPU generation 与 API Judge 应分阶段执行。每个阶段完成后，`pipeline_state.json` 记录该阶段全部产物及关键只读输入
SHA；`sources` 还会登记预注册记录及其 frozen config、population index、两个人口统计文件的路径与 SHA。执行任意后续
阶段前会复核所有前置阶段。任一产物、预注册记录或其输入、RQ1 source、manifest、clean audio、trajectory index、
`run.json` 或 checkpoint 被改写，resume 都会拒绝继续。GPU trial 使用 append-only 私有 `commits.jsonl` 作为事务点；
恢复时会丢弃截断尾行，再投影为 response 与无正文 trial sidecar。Judge label 同时绑定 response SHA 和非密钥
Judge config fingerprint。unknown 保持 unknown，retryable unknown 可恢复重试。

## 主要产物

```text
outputs/stage2_rq2/<run_name>/
  provenance/rq1_sources.json
  provenance/layer_map.json
  provenance/reference_statistics.json
  trajectory/
  trajectory_behavior/
  state_index/behavior_events.json
  state_index/index.json
  identity_tests.json
  oracle_pilot/
  mechanism_pilot/
  subspace_pilot/
  protocol_lock.json
  formal/
  event/
  rq2_causal_prior.json
  rq2_summary.json
  rq2_report.md
```

模型正文只存在于 `responses.jsonl` 和同目录私有恢复日志 `commits.jsonl`；分析 CSV/JSON、Judge label 和报告不保存
response、reasoning 或 Judge raw output。归档或共享分析结果时不应包含这两个正文 sidecar。

## 验收

独立 smoke 已有 `configs/stage2_rq2_qwen7b_advbench_smoke01.json` 和 2 条 dev pair 的同名前缀 manifest。
执行层限定 `[19,24]`，只允许运行到 `mechanism_analyze`；`protocol_lock`、formal、event 与最终 report 会被拒绝。
本轮仅做静态配置检查，真实 GPU/Judge smoke 尚未执行。先只读查看阶段：

```bash
python experiments/run_stage2_rq2.py --config configs/stage2_rq2_qwen7b_advbench_smoke01.json plan
```

随后可用同一配置 `run --through identity`，再分段检查 Oracle 与机制通路。smoke 的 2 条结果只供工程诊断，不进入正式统计。
