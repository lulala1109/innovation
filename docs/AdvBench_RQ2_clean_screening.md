# AdvBench RQ2：7B clean baseline 筛选

## 目的与边界

本流程把官方 AdvBench `harmful_behaviors.csv` 的每条 `goal` 转为**未加扰动**的有害语音，记录冻结的 Qwen2.5-Omni-7B 对该语音的 clean 回复，再用与 RQ1 相同的 StrongREJECT 协议筛选正常拒答的样本。筛选结果只是**未分组、语义独立性待人工确认的候选池**，不是 RQ2 manifest；这里不分配 `rq2_dev`/`rq2_causal_test`，不生成 PGD 轨迹，也不据此宣称 RQ2 因果结论。

数据源固定为 [`llm-attacks/llm-attacks` 提交 `098262edf85f807224e70ecd87b9d83716bf6b73` 的 `harmful_behaviors.csv`](https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv)。预期 SHA-256 为 `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`，表头为 `goal,target`，共 520 条数据。稳定 ID 是 `advbench_0000` 至 `advbench_0519`；源 CSV 记录序号、`goal`、原始 `target` 及来源哈希保留作溯源。**只有 `goal` 可进入 TTS、clean 输入和拒答判断；原始 `target` 不能进入模型输入，也不能覆盖后续 RQ2 预注册的攻击目标 `Sure, here is`。**

## 第一步：下载并制作未攻击语音

下面的 `python` 指安装了项目 [`requirements.txt`](../requirements.txt) 的 Python 3.10 环境；当前工作机的完整依赖解释器示例为 `/root/miniconda3/envs/whisper_default_v2/bin/python`，默认 `python` 环境缺少部分依赖。先激活完整环境，再从项目根目录运行：

```bash
python -m data.prepare_advbench_rq2 fetch
python -m data.prepare_advbench_rq2 tts
```

`fetch` 只接受上述固定版本和哈希；已有正确文件会复用，已有但内容不符则报错。`tts` 默认处理全量 520 条，复用 JBB 单条 TTS，写出 16 kHz、单声道 PCM16 WAV；已在清单中绑定文本与音频哈希的合格文件会复用，失败条目保留状态。网络等临时失败可重跑 `tts`；若目标路径上存在**未绑定来源的 WAV（即使格式合格）**，程序也不会自动接纳或覆盖，须先核对对应 `qa_reason` 并人工处理该文件。可先用 `tts --limit 3` 小规模核对，再运行无 `--limit` 的全量命令。TTS 使用在线 gTTS 服务，运行需要联网，文本会发送给该服务。

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 固定原始 CSV | `dataset/raw/advbench/harmful_behaviors.csv` | 不编辑；保留 SHA 溯源。 |
| 未攻击有害语音 | `dataset/derived/advbench_audio/harmful_clean/advbench_XXXX.wav` | 逐条音频，不写入 JBB 目录。 |
| 全量 inventory | `dataset/processed/rq2/advbench_audio_inventory.jsonl` | 520 行；含生成/质检状态、文本与音频哈希、TTS 版本及质量指标；失败行也保留。 |

音频质检至少检查文件可读、声道/采样率/PCM16、非空/非静音及时长。TTS 通过只表示输入语音已备好，**不等于模型已拒答**。

## 第二步：7B clean 生成、Judge 和候选池

筛选使用独立配置 [`configs/advbench_rq2_clean_7b.json`](../configs/advbench_rq2_clean_7b.json)：`source_csv` 指向固定官方文件，筛选会重新读取并逐行核对 inventory，不信任单独修改过的清单；模型路径指向本项目 Qwen2.5-Omni-7B，`bfloat16`、`max_tokens=100`、`do_sample=false`，音频提示封装沿用现有 Qwen 运行时。启动前还会核对 checkpoint 的 28 层、3584 维文本模型结构，防止误把 3B 当作 7B；模型指纹绑定 checkpoint 的配置、索引和权重分片，初次核验可能耗时。Judge 使用冻结 7B RQ1 的 `deepseek` provider、`DeepSeek-V4-Flash`、`https://token.matpool.com/v1` 和 `strongreject-canonical-balanced` 口径。Judge 密钥只从环境变量 `DEEPSEEK_API_KEY` 读取；配置中的 `env_file` 指向本地 `.env`，也可改为 `null` 以仅使用进程环境，**不要把密钥写入配置或输出文件**。配置要求 RQ1 对照清单恰有 100 个 JBB pair，避免只用局部清单执行独立性审计；输出只允许位于 `outputs/stage2_rq2/`。当前验收环境为 PyTorch 2.6.0+cu124、Transformers 4.57.1 和 NVIDIA GeForce RTX 3090，真实 7B clean generation 已通过；从项目根目录依次运行：

```bash
python -m experiments.screen_advbench_clean generate --config configs/advbench_rq2_clean_7b.json --limit 3
python -m experiments.screen_advbench_clean judge --config configs/advbench_rq2_clean_7b.json
```

以上只做 smoke：检查三条回复、标签及音频输入，不执行 `finalize`，也不把 smoke 结果当作全量候选。确认无误后再运行全量：

```bash
python -m experiments.screen_advbench_clean generate --config configs/advbench_rq2_clean_7b.json
python -m experiments.screen_advbench_clean judge --config configs/advbench_rq2_clean_7b.json
python -m experiments.screen_advbench_clean finalize --config configs/advbench_rq2_clean_7b.json
```

各阶段可断点续跑；生成阶段已记录的 `error`/空回复须排查后用 `generate --retry-failed` 显式重试，Judge 的可重试 `unknown` 可重跑 `judge`。输入音频、模型/生成参数、Judge 配置或回复 SHA 变化时必须拒绝复用旧记录。不要通过修改同一 run 的配置来尝试新实验；另开 `output_root` 和 run 名称。

只有音频 `audio_status=ready`、`qa_status=ok`，并且 7B 生成成功、回复非空、Judge 状态 `ok` 且 `refusal_label=true` 的样本可入选。`unknown`、非拒答、生成失败、音频不合格及与 RQ1 JBB 明确重合的条目都有明确排除原因。与 RQ1 的疑似近重复会**暂时隔离在候选池之外**并单列供人工复核。当前自动规则主要依据规范化文本、音频哈希和词面相似度，不能证明语义独立；尤其 RQ1 JBB 中有 18 条标注 `harmful_source=AdvBench`。`finalize` 另为这 18 条逐条列出 AdvBench 词面相似度最高的 3 个候选（即使低于普通近重复阈值），供正式分组前人工核对行为意图是否实质相同。不能仅凭数据集名称或自动分数放行。筛选账本应覆盖全部 520 个源 ID，不能把缺失标签当作拒答。

筛选输出固定在 `outputs/stage2_rq2/advbench_clean_screening_run01/`：

| 产物 | 文件 | 分享约束 |
| --- | --- | --- |
| 私有模型回复 | `private/responses.jsonl` | 含模型正文，勿公开分享；本地权限应为 `0600`。 |
| 生成状态 | `generation_status.jsonl` | 无模型正文；用于审计失败与续跑。 |
| Judge 标签 | `labels.jsonl` | RQ2 StrongREJECT sidecar，不含回复正文或密钥。 |
| 全量筛选账本 | `screening_ledger.jsonl` | 每个源 ID 一行，记录入选或排除原因。 |
| 未分组候选池 | `eligible_pool.jsonl` | 仅 clean 拒答及自动去重通过；不含回复正文或 AdvBench 原始 `target`，语义独立性仍待人工确认。 |
| 疑似近重复 | `possible_near_duplicates.jsonl` | 暂不入池的复核材料；不能未经复核直接放行。 |
| RQ1 AdvBench 来源复核 | `rq1_advbench_source_review.jsonl` | RQ1 18 条 AdvBench 来源各自的 top-3 词面候选，正式分组前逐条人工核对。 |
| 汇总 | `summary.json` | 各状态数量与来源指纹。 |

本次运行已完成 520 条真实 `generate → judge → finalize`：520 条生成成功、520 条 Judge 标签为 `ok`，
其中 505 条拒答；自动重合与近重复隔离后发布 480 条 `eligible_pool.jsonl`。RQ2 本身仍会独立复核
clean baseline；预筛不能替代该门禁。

`summary.json` 的 `screening_complete=true` 只表示 clean 筛选账本已闭合，不表示与 RQ1 在语义上独立。账本须恰有 520 行。

## 第三步：本次运行采用的极保守候选集

2026-10-02，用户决定不执行人工语义判断和音频内容保真听审，改为从 480 条自动候选中机械排除
`rq1_advbench_source_review.jsonl` 全部 top-match 里仍在候选池中的 25 个唯一 ID。该决定通过离线构建器实现：

```bash
python -m data.build_rq2_conservative_pool build
python -m data.build_rq2_conservative_pool check
```

输出位于：

```text
outputs/stage2_rq2/advbench_clean_screening_run01/conservative_candidate_pool/
  conservative_candidate_pool.jsonl  # 455 条
  conservative_exclusions.jsonl      # 25 条
  summary.json
```

原始 `eligible_pool.jsonl` 保持不变。455 条候选集 SHA-256 为
`1dea865fbf376526a3645e6b2f4cd5bc446c6a2a93029c062f2043369346ab3f`。汇总明确记录
`manual_semantic_review_performed=false` 和 `audio_content_fidelity_review_performed=false`；因此后续可以将
该文件作为 T0.6 的未分组输入，但不得声称“人工语义独立性已确认”或“音频内容保真已确认”。

## 第四步：冻结 RQ2 数据角色划分

T0.6 已按预先确定的 `rq2_dev=20`、`rq2_causal_test=40`、`reserve=395` 执行。划分不使用任何
RQ2 pilot、PGD 成功率或 formal 结果；它将 seed `42`、统一 `stratum="advbench"` 和稳定身份哈希排序规则
`sha256-canonical-identity-sort-v1` 写入审计文件。

```bash
python -m data.build_rq2_split build
python -m data.build_rq2_split check
```

冻结产物位于 `dataset/processed/rq2/advbench_split_v1/`。其中 `split_audit.json` SHA-256 为
`0cc3f20ed40a654d59aa5953c8c450917618033360393e2844b11f34302b8c48`，且 `check` 返回 `VALID`。dev/test 之间以及
dev/test 与完整 RQ1 JBB 的 pair ID、规范化文本/content group、clean-audio SHA-256 交集均为 0。

`reserve.jsonl` 中的 395 条只是当前 run 未选中的冻结后备，不能在看到实验结果后随意替换 dev/test 样本。

## 第五步：构建正式 RQ2 source manifest

T0.7 使用冻结 split 生成60条最小公开 manifest，并将统一攻击目标固定为 `Sure, here is`：

```bash
python -m data.build_rq2_manifest build
python -m data.build_rq2_manifest check
```

产物位于 `dataset/processed/rq2/advbench_manifest_v1/`。`rq2_manifest.jsonl` 包含20条 dev 和40条
causal-test，不包含395条 reserve、AdvBench 原始 `target`、回复正文、Judge reasoning 或伪造的 `trajectory_path`。
其 SHA-256 为 `66ed99e487ab63cba6af6a529aac962de0342721a23e8e77d0caf7889d6e4e86`；
`manifest_audit.json` SHA-256 为 `ebfa636b8088e812878ac05359db493c31e715eaa67eebe1ff237f6e5565c45b`。
初始 manifest 供 `trajectory.mode=generate` 的 Standard PGD batch runner 使用；真实轨迹生成后，流水线会自动写出
带 `trajectory_path` 的 `trajectory/resolved_manifest.json`。
