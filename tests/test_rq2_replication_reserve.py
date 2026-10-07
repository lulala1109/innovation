"""Small CPU-only checks; no repository experiments or behavior data."""
import unittest

from data.audit_rq2_replication_reserve import (
    identities, normalized_text_sha, overlap, protected_path, select_rows, usage_tokens,
)


class ReservePreparationTests(unittest.TestCase):
    def test_original_order_and_prior_use_exclusion(self):
        rows = [{"pair_id": f"advbench_{i:04}", "split_rank": i} for i in range(61, 91)]
        selected = select_rows(list(reversed(rows)), {"advbench_0062"})
        self.assertEqual([r["split_rank"] for r in selected], [61, *range(63, 82)])

    def test_no_adaptive_expansion_or_borrowing(self):
        with self.assertRaises(ValueError):
            select_rows([], set())
        with self.assertRaises(ValueError):
            select_rows([], set(), count=21)

    def test_path_hash_and_text_usage_are_recognized(self):
        digest = "a" * 64
        lookup = {"advbench_0001": {"advbench_0001"}, digest: {"advbench_0002"},
                  normalized_text_sha("Example instruction"): {"advbench_0003"}}
        record = {"checkpoint": "/attacks/advbench_0001_abcd/index.json",
                  "sha256": digest, "harmful_text": "  EXAMPLE   instruction  "}
        self.assertEqual(usage_tokens(record, lookup),
                         {"advbench_0001", "advbench_0002", "advbench_0003"})

    def test_exact_content_overlap_detected_across_different_ids(self):
        a = {"pair_id": "a", "harmful_text": "Example instruction", "content_group": "g1",
             "clean_audio_sha256": "a" * 64}
        b = {**a, "pair_id": "b", "harmful_text": " EXAMPLE instruction ", "content_group": "g2"}
        result = overlap(identities([a]), identities([b]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["pair_ids"], 0)
        self.assertEqual(result["content_groups"], 1)
        self.assertEqual(result["audio_sha256"], 1)

    def test_protected_behavior_paths_not_read(self):
        for path in ("outputs/formal/run/labels.jsonl", "outputs/rq2_causal_test/responses.jsonl",
                     "outputs/causal-test/index.json"):
            self.assertTrue(protected_path(path))
        self.assertFalse(protected_path("outputs/event_dev/run/trials.jsonl"))


if __name__ == "__main__":
    unittest.main()
