# RQ2 window Judge network recovery

This separate entry addresses the final unreceipted `URLError` in the existing
exploration ledger. It only continues `window_judge`, `window_analyze`, and
`report`. It does not run GPU generation or change frozen source files/configs.

The original uncertain attempt, reservation, receipts, caches and ledger remain
unchanged. A new ledger under `network_recovery/attempt_000234/scoring/` carries
all earlier costs and HTTP attempts, including the full uncertain reservation.
This is conservative accounting, not confirmation of the provider's charge.
Retrying may pay for the same input again if the original request was processed.
The original cumulative budget and three-attempt limit per input still apply.
The scoring request, rubric, contract fingerprint and existing labels are unchanged.
The recovery audit binds the original scoring files and new recovery code.

On the scoring server, use the existing `whisper_default_v2` environment and the
project root. Make sure these two new files are present on that server:

- `rq2/exploration_network_resume.py`
- `experiments/rq2_exploration_network_resume.py`

```bash
rq2_config=configs/stage2_rq2_qwen7b_advbench_post_oracle_explore_v1_run01.json

# Offline validation only: no network, writes, GPU or scoring charges.
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 inspect

# Offline validation plus credential-free HTTPS GET /models, no completion call.
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 check

# Explicitly accept the uncertain original charge and a bounded retry.
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 run \
  --accept-uncertain-charge-and-retry --acknowledge-cost --accept-exploratory

# Read the combined budget and progress through the recovery entry.
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 status
```

`check` verifies network reachability and that an API key is loaded; it does not
verify that the key is valid or that the provider account has credit. A 401, 403,
404 or 405 from the credential-free probe only confirms that HTTPS responded.
Run `check` on the actual scoring server; another host may have different DNS,
proxy, certificates or firewall settings.

If the terminal has a broken proxy, add `--direct` to both commands:

```bash
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 check --direct
python experiments/rq2_exploration_network_resume.py --config "$rq2_config" --attempt 234 run --direct \
  --accept-uncertain-charge-and-retry --acknowledge-cost --accept-exploratory
```

This adds only the configured API host to the current process's proxy bypass
list, after loading `.env`. It applies to both the probe and actual scoring and
does not change the parent shell, `.env`, or frozen scoring transport. The
`[network]` line shows the effective route and whether a key is loaded without
printing credentials. Direct connectivity still depends on that server's network.

If another scoring request fails with an uncertain outcome, this entry stops and
does not automatically authorize another recovery. Preserve its new ledger and
inspect the failure. Do not delete either ledger or relabel uncertain attempts
as settled. Use this recovery entry for subsequent progress/budget queries; the
original entry deliberately still sees the original unresolved request.
