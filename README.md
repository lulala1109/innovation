# Safety-State-Aware Waveform PGD

本项目是“优化时安全状态动力学（Optimization-Time Safety-State Dynamics）”、
“动态安全瓶颈（Dynamic Safety Bottleneck）”和“安全状态感知的层自适应 PGD”
的研究实现。当前主线只在**波形空间**优化扰动，并将输出行为、逐层安全状态和
优化轨迹分开保存，便于做公平消融、断点恢复和机制验证。

本目录不包含模型权重、数据集、实验结果、`.env` 或 API 密钥。

## 当前研究边界

- **Qwen 2.5 Omni 是安全状态主方法。** `QwenModel.forward_attack()` 提供可微
  WAV→官方对齐 MEL→音频 embedding→语言模型 hidden states 的完整路径。
- `models/optional/phi.py` 和 `models/optional/voxtral.py` 是迁移后的可选归档，
  不在默认 model factory 中，也没有经过当前主线验证。
- 本代码面向经授权的模型安全研究与防御评测。运行者负责数据、模型和评测服务
  的访问权限。

## 迁移后的结构

```text
attacks/
  base.py                       # 波形扰动初始化、L∞ 投影、裁剪、SNR
  pgd.py                        # Standard waveform PGD baseline
  layer_adaptive_pgd.py         # 五种层级方法，共用 canonical PGD 更新
core/
  activations.py                # hidden-state 选择、token pooling、hook、patching
  artifacts.py                  # 原子化 .pt 轨迹、JSON 索引和断点恢复
  audio.py                      # 加载、重采样、TTS helper、保存
  safety_state.py               # 独立 H/R probe、gap、权重和动态瓶颈
  reproducibility.py            # seed、确定性配置和运行环境信息
  judge.py / llm_backend.py     # 外部行为评测后端
data/
  datasets.py                   # 本地表格与可选 Hugging Face 数据加载
  sampling.py                   # FPC、分层/均衡抽样、排除重叠
  build_safety_pairs.py         # X_B / X_H / X_J manifest 和 pair_id
docs/
  *.docx                        # 创新点说明与具体实验方案
experiments/
  train_safety_probes.py        # 独立 H/R layerwise probe 训练
  batch_safety_attack.py        # 可恢复的通用批量攻击入口
  analyze_safety_dynamics.py    # Layer×Step 状态表、瓶颈路径和 Go/No-Go
  activation_patching.py        # 关键层与随机层对照的 activation patching
evaluation/
  behavior.py                   # StrongREJECT、Llama Guard、JailbreakEval
  evaluate_safety_runs.py       # 通用 run.json 评测和分组汇总
  perceptual.py                 # L∞/L2/RMS/SNR；可选 PESQ/STOI
models/
  qwen.py                       # 主方法模型
  optional/                     # Phi/Voxtral 归档，不进入默认 factory
tests/                           # 不下载模型的离线单元与接口测试
```

## 六种方法

六种方法使用相同的波形预算、步长、步数、初始化、target loss、token span 和
生成检查策略。层级方法的区别只在层权重如何产生。

| `--method` | 层权重/目标 | 额外必需输入 |
| --- | --- | --- |
| `standard` | 仅 output target loss，不读取 hidden states | 无 |
| `fixed` | 一个预先指定层的 one-hot 权重 | probe、X_H reference、`--fixed-layer` |
| `uniform` | 所选层等权 | probe、X_H reference |
| `static_topk` | 预先独立选出的固定 top-k 层等权 | probe、X_H reference、`--static-topk-layers` |
| `gradient_adaptive` | 每步按各层状态目标对波形扰动的梯度范数分配权重 | probe、X_H reference |
| `safety_state_adaptive` | 每步按 refusal degradation gap 的 softmax 分配权重 | probe、X_H reference |

`core/safety_state.py` 始终把 harmfulness 状态 H 和 refusal 状态 R 当作两个独立
量；不会隐式相加或混成一个分数。当前层自适应优化以 R 为状态目标，H 可独立
记录和分析。对层 `l`、优化状态 `t`，refusal degradation 定义为：

```text
G_l^t = R_l(X_H) - R_l(X_adv^t)
w_l^t = softmax(G_l^t / temperature)
```

最大 `G_l^t` 对应当前动态安全瓶颈。`gradient_adaptive` 的权重只来自梯度范数；
仍要求 X_H reference，是为了记录可横向比较的 gap 和 bottleneck，不会用 gap
来产生该方法的权重。

## 公平的 waveform L∞ 威胁模型

所有主线方法都直接优化 16 kHz 浮点波形。输入必须位于 `[-1, 1]`，每次更新后
同时执行 L∞ 投影和有效音频范围裁剪：

```text
x_adv = clip(x + delta, -1, 1)
||delta||∞ <= eps
delta <- Project_L∞(delta - alpha * sign(grad_delta L))
```

这里的 `eps` 是波形振幅预算，不是 MEL、embedding、SNR 或感知距离预算。公平
对比至少要固定以下项目：

- `eps`、`alpha`、`steps`、`init-mode`、`seed`；
- `loss-type`、`kappa`、target text、模型 checkpoint 和 dtype；
- 输入音频、重采样流程、所选层、token span 和 pooling；
- probe 训练集、probe checkpoint 和每个 `pair_id` 自己的 X_H reference；
- checkpoint 状态编号和生成检查间隔。

做固定步数消融时建议加 `--no-early-stop`，否则成功样本的实际更新次数不同。
`static_topk` 的层必须只从独立的训练/验证数据选出，不能观察当前测试 case 后再选。
PESQ、STOI、SNR 等是额外报告指标，不能替代 L∞ 约束；保存 WAV 后也应重新核验
量化误差下的实际预算。

## X_B / X_H / X_J 数据 manifest

规范格式是一行一个语义 pair 的宽表。`pair_id` 由 source、stratum、benign_text
和 harmful_text 内容稳定生成，用于阻止跨样本拼接和 reference 泄漏。

| 字段 | 含义 |
| --- | --- |
| `pair_id` | 稳定且唯一的语义配对 ID |
| `source` | 数据来源 |
| `stratum` | 危害类别/分层标签 |
| `benign_text` | X_B：配对的良性输入 |
| `harmful_text` | X_H/X_J：同一有害意图 |
| `clean_response` | X_H 的干净响应 |
| `clean_refused` | X_H 是否拒绝 |
| `jailbreak_response` | X_J 的越狱响应 |
| `jailbreak_success` | X_J 是否成功 |
| `clean_audio_path` | X_H 干净波形；批量攻击从这里加载输入 |
| `jailbreak_audio_path` | X_J 波形，用于配对分析/复现 |

manifest 可以保留额外列；批处理支持逐行 `case_id`、`target_text` 和
`reference_refusal_path`。构建并严格验证完整 triplet：

```bash
python -m data.build_safety_pairs \
  --input data/raw/safety_pairs.csv \
  --output data/manifests/safety_pairs.csv \
  --state-dir data/manifests/states \
  --require-complete \
  --require-state-triplets
```

该命令同时可写出 `x_b.csv`、`x_h.csv` 和 `x_j.csv` 视图。已有 manifest 可只
验证，不重写 ID：

```bash
python -m data.build_safety_pairs \
  --input data/manifests/safety_pairs.csv \
  --validate-only \
  --require-state-triplets
```

本地 CSV、TSV、JSON/JSONL 和 Parquet 是首选入口。Hugging Face 数据只在显式
使用 `--hf-dataset` 时加载；Parquet 还需要环境中安装兼容的 `pyarrow`。

## Probe 与 X_H reference 必须显式提供

`fixed`、`uniform`、`static_topk`、`gradient_adaptive` 和
`safety_state_adaptive` 都会在启动前检查以下两个文件；批处理器**不会自动训练
probe、不会猜 reference，也不会用当前攻击样本拟合它们**。

Probe checkpoint 是 `torch.save` 的映射，格式为：

```text
{
  "state_dict": <DualSafetyStateScorer.state_dict()>,
  "hidden_size": <int>
}
```

不同层维度不同时可用 `hidden_sizes: {layer: width}` 代替 `hidden_size`。加载器只
接受 state dict，不接受任意序列化 Python module。

Reference checkpoint 必须包含 `reference_refusal`（也兼容 `refusal` 或
`refusal_scores`）。单 case 可以直接给逐层张量/映射；多 case 必须按 `pair_id`
隔离，例如：

```text
{
  "reference_refusal": {
    "by_pair_id": {
      "pair_...": {layer: refusal_score, ...}
    }
  }
}
```

manifest 中的 `reference_refusal_path` 会覆盖全局 reference。这样每个 X_J/攻击
轨迹只能和同一 `pair_id` 的 X_H refusal 状态比较。

## 训练独立 H/R Probes

攻击时不会临时拟合 probe。先在与测试 case 隔离的 hidden-state 数据上运行：

```bash
python -m experiments.train_safety_probes \
  --input artifacts/probes/qwen_probe_training.pt \
  --output artifacts/probes/qwen_safety_state.pt \
  --validation-fraction 0.2 \
  --epochs 100 --learning-rate 0.01 --seed 42
```

输入 `.pt` 使用受限的 `weights_only=True` 加载，schema 为：

```text
{
  "hidden_states": {layer: Tensor[N, D], ...},
  "harmfulness_labels": Tensor[N],  # 0/1
  "refusal_labels": Tensor[N],      # 0/1
  "pair_ids": ["pair_...", ...]
}
```

训练/验证按完整 `pair_id` 分组划分，同一个 pair 不会跨分区。H 与 R 使用不同的
linear probe、optimizer、BCE-with-logits 历史和指标；输出 checkpoint 可直接传给
`--probe-checkpoint`。训练 payload 应由受控的 X_B/X_H/X_J forward 和
`core.activations.collect_hidden_states()` 生成，不能包含测试 case。

## 运行单样本实验

单样本入口只负责写一行 manifest，然后调用与批量实验完全相同的引擎：

```bash
python attack.py \
  --wav /path/to/input.wav \
  --output-dir runs/single_standard \
  --harmful-text "<对应的有害文本>" \
  --target-text "<目标文本>" \
  --method standard --model qwen-3b \
  --model-id /path/to/Qwen2.5-Omni-3B
```

因此它与批量入口共享预算、checkpoint、`run.json`、恢复指纹和失败处理，不再维护
旧的单样本 Two-Stage/RL 分支。

## 运行批量实验

建议使用项目现有 Python 3.10 环境；也可以在自己的隔离环境中安装：

```bash
pip install -r requirements.txt
```

默认 factory 仅支持 `qwen-3b` 和 `qwen-7b`。
`--model-id` 可传本地 checkpoint 路径或 Hugging Face ID，避免在代码里硬编码
权重位置。

Standard PGD：

```bash
python -m experiments.batch_safety_attack \
  --manifest data/manifests/safety_pairs.csv \
  --output-dir runs/qwen_standard \
  --method standard \
  --model qwen-3b \
  --model-id /path/to/Qwen2.5-Omni-3B \
  --target-text "<统一目标文本>" \
  --eps 0.1 --alpha 0.005 --steps 100 \
  --checkpoint-steps 0,25,50,75,100 \
  --seed 42 --no-early-stop
```

如果 manifest 每行已有 `target_text`，可省略全局 `--target-text`。安全状态自适应
方法：

```bash
python -m experiments.batch_safety_attack \
  --manifest data/manifests/safety_pairs.csv \
  --output-dir runs/qwen_safety_state_adaptive \
  --method safety_state_adaptive \
  --model qwen-3b \
  --model-id /path/to/Qwen2.5-Omni-3B \
  --probe-checkpoint artifacts/probes/qwen_safety_state.pt \
  --reference-refusal artifacts/references/x_h_refusal.pt \
  --layers 8,12,16,20,24 \
  --state-loss-weight 1.0 --temperature 1.0 \
  --token-span audio --pooling mean \
  --eps 0.1 --alpha 0.005 --steps 100 \
  --checkpoint-steps 0,25,50,75,100 \
  --seed 42 --no-early-stop
```

其他层基线只替换 `--method`，并补充其固定选择参数：

```bash
# 固定层
--method fixed --fixed-layer 16

# 独立数据上预选的固定层集合
--method static_topk --static-topk-layers 12,16,20
```

批处理逐 case 原子写入结果；已有、schema 完整且 `experiment_fingerprint` 与当前配置一致的 `run.json` 才会在重跑时跳过，
失败 case 写 `error.json`，批次进度写 `summary.json`。这使中断后的重跑不会改变
后续 case 的逐行 seed。

## 分析安全状态动力学

层方法的轨迹可以直接生成 Layer×Step 长表、动态瓶颈路径和 Go/No-Go 判断：

```bash
python -m experiments.analyze_safety_dynamics \
  --trajectory runs/qwen_safety_state_adaptive/<case-dir> \
  --output-json analyses/case_dynamics.json \
  --output-csv analyses/case_layer_step.csv \
  --min-unique-bottlenecks 2 \
  --min-bottleneck-switches 1 \
  --require-all-states
```

输入可以是 case 目录、`trajectory/` 目录或 `trajectory/index.json`。输出 JSON
包含 `bottleneck_path`、`go_no_go.dynamic_migration`、切换次数、退化范围和静态
critical window；CSV 是逐 step×layer 的 refusal、harmfulness、weight 和 gap。

## Activation patching

通用协调器不猜模型层名或 forward continuation，需要项目侧提供
`module:callable` factory。该 callable 接收 CLI namespace，并返回：`model`、
`layer_modules`、`source_activations`（同 pair 的 X_H refusal activation）、
`forward_fn` 和 `score_fn`。

```bash
python -m experiments.activation_patching \
  --factory my_experiment.patch_context:build_context \
  --critical-layers 12,16,20 \
  --random-control-count 3 \
  --seed 42 \
  --token-start 0 --token-end 128 \
  --output analyses/activation_patching.json
```

每次 trial 只 patch 一个层，并报告关键层与不重叠随机层的 score delta 和聚合差；
随机对照由 seed 固定。source activation、目标 forward 和行为评分必须属于同一
语义 pair，避免把配对错误解释成因果效应。

## 行为评测

评测器只读取通用 `run.json`，不解释旧 Stage-1/Stage-2 字段。StrongREJECT 的
provider、model、base URL 和密钥通过 CLI/环境变量显式配置，密钥不会写进结果。

```bash
python -m evaluation.evaluate_safety_runs \
  runs/qwen_safety_state_adaptive \
  --runs 3 \
  --strongreject-provider "$STRONGREJECT_PROVIDER" \
  --strongreject-model "$STRONGREJECT_MODEL" \
  --output runs/qwen_safety_state_adaptive/evaluation/eval_results.json
```

可选增加 `--enable-llamaguard` 或 `--enable-jailbreakeval`。汇总同时按 method、
stratum 和 method×stratum 输出；provider 错误或无效 judge JSON 会明确失败，不会
被静默替换成零分。

波形质量指标可通过 `evaluation.perceptual` 的
`compute_perturbation_metrics()`、`compute_pesq()` 和 `compute_stoi()` 计算；PESQ/STOI
只有在显式调用时才要求各自的第三方包。

## 通用产物 schema

一个批次的核心布局为：

```text
runs/<batch>/
  summary.json
  <case-dir>/
    run.json
    adversarial.wav
    trajectory/
      index.json
      step_000000.pt
      step_000025.pt
      ...
    error.json                 # 仅失败时存在
```

`run.json` 顶层严格限定为以下字段：

```json
{
  "case_id": "case_000001",
  "pair_id": "pair_...",
  "method": "safety_state_adaptive",
  "model": "qwen-3b",
  "stratum": "category",
  "harmful_text": "...",
  "adversarial_response": "...",
  "attack_success": false,
  "budget": {
    "experiment_fingerprint": "<sha256>",
    "experiment_config": {"schema_version": 1},
    "norm": "linf",
    "eps": 0.1,
    "alpha": 0.005,
    "steps": 100,
    "loss_type": "margin",
    "kappa": 5.0,
    "init_mode": "zero",
    "seed": 42,
    "determinism": {},
    "checkpoint_steps": [0, 25, 50, 75, 100]
  },
  "artifacts": {
    "trajectory": "trajectory/index.json",
    "adversarial_audio": "adversarial.wav",
    "input_audio": "/absolute/path/to/input.wav"
  }
}
```

轨迹遵循 `safety-state-trajectory` version 1：

- `trajectory/step_XXXXXX.pt` 只保存 tensor，默认至少包含 `delta` 和
  `adversarial_wav`；自定义 snapshot 可以添加 hidden/refusal/harmfulness tensor。
- `trajectory/index.json` 的 `checkpoints[]` 保存 step、相对路径、保存时间、tensor
  的 shape/dtype 索引，以及可 JSON 序列化的标量元数据。
- 层方法的元数据包含 loss、target/state loss、L∞、RMS、SNR、逐层
  `refusal_scores`、`safety_gaps`、`layer_weights` 和当前 bottleneck。
- `events.jsonl` 可用于额外的标量/文本事件。任何 tensor 都会被拒绝写入 JSON。
- 状态 0 表示初始化后的扰动、尚未完成更新；状态 N 表示恰好完成 N 次更新。
  无 early stop 时，N 步攻击记录 N+1 个对齐状态。

checkpoint 先写同目录临时文件，再原子 rename；即使进程在索引更新前中断，完整
的孤立 `.pt` 仍能由 `TrajectoryArtifactStore` 扫描并恢复。

## 验证

完整离线测试不会下载模型权重：

```bash
/root/miniconda3/envs/whisper_default_v2/bin/python \
  -m unittest discover -s tests -v
```

静态编译与 CLI smoke test：

```bash
/root/miniconda3/envs/whisper_default_v2/bin/python \
  -m compileall attacks core data evaluation experiments models

python attack.py --help
python -m data.build_safety_pairs --help
python -m experiments.train_safety_probes --help
python -m experiments.batch_safety_attack --help
python -m experiments.analyze_safety_dynamics --help
python -m experiments.activation_patching --help
python -m evaluation.evaluate_safety_runs --help
```

在正式报告结果前，还应检查：不同 method 的预算/seed 是否一致、`pair_id` 是否
唯一、probe 数据是否与测试集隔离、每个 case 是否使用自己的 X_H reference、
checkpoint 是否覆盖预定状态，以及保存后的波形是否仍满足 L∞ 预算。
