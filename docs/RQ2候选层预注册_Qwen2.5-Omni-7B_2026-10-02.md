# RQ2 候选层预注册：Qwen2.5-Omni-7B

- 日期：2026-10-02
- 权威机器记录：`configs/rq2_candidate_layer_preregistration_qwen7b_v1.json`
- 权威记录 SHA-256：`79b69084067787bd7f1eef80772902ef1df8c4ac1f5895d7922c0c818ab3fcfa`
- 适用范围：冻结的 28 层 Qwen2.5-Omni-7B RQ1 bundle → RQ2 pilot

本文只解释机器记录；若本文与 JSON 不一致，以 JSON 及其 SHA 为准。

## 选择规则

主排序指标是 `R_probe.slope_mean`。在 `all` 和 `baseline_refused` 中分别保留
`slope_mean < 0`、95% CI 上界 `< 0`、FDR `q <= 0.05` 的层，再按
`slope_mean → q → layer` 升序取 top-5，最后取两套 top-5 的交集。

- `all` top-5：`[24, 19, 27, 26, 25]`
- `baseline_refused` top-5：`[27, 19, 24, 17, 26]`
- 跨人口全层排序 Spearman：`0.970990695129`
- candidate：`[19, 24, 26, 27]`
- neighbor：`[18, 20, 23, 25]`，由 candidate 的 `±1` 机械生成并排除 candidate
- depth control：`[2, 12]`，是前段/中段设计锚点，不按效果排序

候选层还必须在两个人口的 `R_direction` 上同时满足负斜率、CI 不跨零和 FDR 门槛。
`H_probe`、`H_probe_minus_R_probe`、phase profile 和 profile reproducibility 只作诊断，
不进入候选层筛选。

## 冻结协议

- fixed steps：`[2, 10, 100]`；event offsets：`[-2, -1, 0, 1, 3]`
- restoration doses：`[1.0, 0.5, 1.5]`；suppression doses：`[1.0, 0.5, 1.5]`
- random replicates：`3`；seed：`42`
- refusal weakening：`0.2`
- Oracle / mechanism 门槛：`0.05` / `0.03`
- bootstrap / confidence / FDR：`2000` / `0.95` / `0.05`
- formal：全部 `0..27` 层，至少 `20` 个 causal-test pair

## 输入身份

- `configs/stage1_rq1_qwen7b_short_sure_here_is_run01_frozen.json`：`8b51073c8ae846c1f8900ee502050619631f36a78d748eb4495daf90865af45c`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/population_index.json`：`686e32d9a09db9a35732c413554bdc93b68b60debd56c15714290b2b8a3d2695`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/layer_slopes.csv`：`219280ae8b75f9073c4a3416ab36c13d906d80e9aa886bbd9f4bb0982c4ce92b`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/phase_profiles.csv`：`8aa351d551e130a7bdd7b170fdaa123c4a70b6b68d952ab6c4d2ac4dce02867d`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/profile_reproducibility.csv`：`4e994b545d2242e2f78fa77047f810f7c27f893b03be9cadaf265d88a1becf7a`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/layer_slopes.csv`：`59a690bdc0f21f990244e9e1d3831d4b5d36499ff359bd2200794e4b7460b913`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/phase_profiles.csv`：`850d4c42d55c469cbee4321618a3bf8796d609f0146716b389e459f267d3e5ce`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/profile_reproducibility.csv`：`3fe0605a396d71a66e8378442bbbc6cf4844eddd0821bdd9ec720f1b1362aad0`

## 解释边界

这些层只是由 RQ1 观察性结果形成的 RQ2 pilot 优先级，不是已发现的“因果关键层”。
任何 RQ2 dev 或 formal 结果都不得用于重选候选层、相邻层或深度对照层。dev pilot
只能按预注册规则决定 primary intervention/主剂量是否进入后续 protocol lock；formal
始终扫描冻结 bundle 的全部 `0..27` 层。
