# RQ2 层角色与剂量预注册 v2：Qwen2.5-Omni-7B

- 日期：2026-10-04
- 机器记录：`configs/rq2_candidate_layer_preregistration_qwen7b_v2.json`
- 机器记录 SHA-256：`5c3a99b7347d400a48c117b44040d0f12f515b4cec66de304cc506a271f89a41`
- v1 父记录：`configs/rq2_candidate_layer_preregistration_qwen7b_v1.json`
- v1 父记录 SHA-256：`79b69084067787bd7f1eef80772902ef1df8c4ac1f5895d7922c0c818ab3fcfa`

## 层角色

- 候选层：`[19, 24, 26, 27]`，继承 v1 的冻结 RQ1 选择结果。
- 邻层：`[18, 20, 23, 25]`，用于局部轮廓，不预设为弱机制层。
- 远距层：`[2, 12]`，用于前段/中段参照，不是深度匹配对照。
- 正式主扫描：全部 `0..27` 层。

机制门禁使用同层、同 pair、同状态、同剂量的 R-vs-H/random/sham/token-position 对照。
邻层/远距层比较另表报告，不作为机制特异性通过条件。邻层比较包含相邻候选层，
并保留其 candidate/neighbor 角色；不得用“胜过任意一个远距层”宣称排除了深度效应。

## 剂量与分析口径

- restoration 主剂量：`1.0`。
- suppression 主剂量：`1.0`。
- 敏感性剂量：`[0.5]`，用于 pilot 剂量响应，独立报告。
- v2 不执行 `1.5`；`0` 用于等价性/sham。
- restoration 的单位是 paired R 投影差；suppression 的单位是冻结的 refusal_sigma。
- 对照按 dose 精确配对，random replicate 在同一 pair/剂量内求均值。
- 主结论、主热图和层排序只使用主剂量；未生成匹配对照的敏感性条件不作机制确认。

## 版本与范围

本修订继承并验证 v1 的输入 SHA 和候选层，不修改 v1 文件，不使用 RQ2 结果重选层或剂量。
新配置应绑定本记录 SHA 并使用独立输出目录。本记录只冻结本次层角色与剂量修订，
不代表整份 v2 协议中的 pilot、FDR family、有效 N 等后续修订已全部完成。

## 输入身份

- `configs/stage1_rq1_qwen7b_short_sure_here_is_run01_frozen.json`：`8b51073c8ae846c1f8900ee502050619631f36a78d748eb4495daf90865af45c`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/population_index.json`：`686e32d9a09db9a35732c413554bdc93b68b60debd56c15714290b2b8a3d2695`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/layer_slopes.csv`：`219280ae8b75f9073c4a3416ab36c13d906d80e9aa886bbd9f4bb0982c4ce92b`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/phase_profiles.csv`：`8aa351d551e130a7bdd7b170fdaa123c4a70b6b68d952ab6c4d2ac4dce02867d`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/all/profile_reproducibility.csv`：`4e994b545d2242e2f78fa77047f810f7c27f893b03be9cadaf265d88a1becf7a`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/layer_slopes.csv`：`59a690bdc0f21f990244e9e1d3831d4b5d36499ff359bd2200794e4b7460b913`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/phase_profiles.csv`：`850d4c42d55c469cbee4321618a3bf8796d609f0146716b389e459f267d3e5ce`
- `outputs/stage1_v2/qwen7b_short_sure_here_is_run01/rq1/baseline_refused/profile_reproducibility.csv`：`3fe0605a396d71a66e8378442bbbc6cf4844eddd0821bdd9ec720f1b1362aad0`
- `configs/rq2_candidate_layer_preregistration_qwen7b_v1.json`：`79b69084067787bd7f1eef80772902ef1df8c4ac1f5895d7922c0c818ab3fcfa`
