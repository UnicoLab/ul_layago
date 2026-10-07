import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import io
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_training_data import (ADJUDICATION_PROTOCOL, PROTOCOL,
    adjudication_binding, canonical, permute_choices, prepare, read_jsonl,
    write_output)

ROOT = Path(__file__).resolve().parents[1]


def row(n="one", split="train", kind="choice"):
    value = {"id": f"row-{n}", "instance": "R", "task": "r_evidence_support",
             "task_type": kind, "input": {"question": f"Question {n}",
             "evidence": [{"id": "local-source", "text": f"Original record {n}."}]},
             "options": [{"key": "supports", "text": "Supported"},
                         {"key": "contradicts", "text": "Contradicted"},
                         {"key": "insufficient", "text": "Not established"}],
             "label": 1, "label_source": "template", "split": split,
             "lang": "en", "domain_id": "shared-training-domain",
             "story_id": f"story-{n}", "generator_version": "test-fixture/1",
             "seed": 13, "meta": {"provenance_note": "Keep this unchanged"}}
    if kind == "score":
        value.update(options=["low", "middle", "high"], label=2,
                     task="r_rank")
    elif kind == "yes_no":
        value.update(options=["false criterion", "true criterion"], label=True,
                     task="r_yes_no")
    if split == "heldout_domain":
        value["domain_id"] = "sealed-domain"
    if split == "heldout_language":
        value["lang"] = "fr"
    return value


def adjudicate(value, owner="test-owner"):
    value = copy.deepcopy(value)
    value["meta"]["replay_adjudication"] = {
        "protocol": ADJUDICATION_PROTOCOL, "status": "adjudicated",
        "owner_id": owner, "reviewer_id": "human-reviewer",
        "record_id": f"review-{value['id']}",
        "binding_sha256": adjudication_binding(value)}
    return value


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self._private = tempfile.TemporaryDirectory()
        self.access_log = Path(self._private.name)/"protected-access.jsonl"

    def tearDown(self):
        self._private.cleanup()

    def test_choice_hard_soft_targets_follow_keys_not_positions(self):
        original = row()
        original["label_dist"] = [.2, .7, .1]
        original["provenance"] = ["a"*64]
        before = copy.deepcopy(original)
        options = {o["key"]: o for o in original["options"]}
        target = dict(zip(options, original["label_dist"]))
        variants = permute_choices(original, 41, 8)
        self.assertEqual(len(variants), 5)  # all nonidentity 3! orders
        self.assertEqual(len({tuple(o["key"] for o in r["options"]) for r in variants}), 5)
        self.assertEqual(len({r["id"] for r in variants}), 5)
        for variant in variants:
            keys = [o["key"] for o in variant["options"]]
            self.assertEqual(keys[variant["label"]], "contradicts")
            self.assertEqual(dict(zip(keys, variant["label_dist"])), target)
            self.assertEqual({o["key"]: o for o in variant["options"]}, options)
            for key in ("input", "provenance", "story_id", "label_source",
                        "generator_version", "lang", "domain_id"):
                self.assertEqual(variant[key], original[key])
            self.assertEqual(variant["seed"], 41)
            self.assertEqual(variant["meta"]["training_preparation"]["original_seed"], 13)
        self.assertEqual(original, before)
        self.assertEqual(variants, permute_choices(original, 41, 8))

    def test_score_boolean_and_every_protected_split_remain_untouched(self):
        originals = [row("choice"), row("score", kind="score"), row("bool", kind="yes_no"),
                     row("dev", "dev"), row("domain", "heldout_domain"),
                     row("lang", "heldout_language"), row("locked", "locked_test")]
        before = copy.deepcopy(originals)
        result, report = prepare(originals, seed=44, permutations=3, protected_access_log=self.access_log)
        self.assertEqual(result[:len(originals)], before)
        self.assertEqual(report["choice_variants"], 3)
        self.assertEqual(report["protected_rows_unchanged"], 4)
        self.assertEqual(report["per_example_choice_tasks"], ["r_evidence_support"])
        self.assertEqual(originals, before)
        self.assertTrue(all(r["split"] == "train" and r["task_type"] == "choice"
                            for r in result[len(originals):]))

    def test_seed_reproducibility_and_bounds(self):
        original = row()
        self.assertEqual(prepare([original], seed=3), prepare([original], seed=3))
        self.assertNotEqual(permute_choices(original, 3, 1)[0]["id"],
                            permute_choices(original, 4, 1)[0]["id"])
        for kwargs in [{"seed": True}, {"seed": -1}, {"seed": 2**63},
                       {"permutations": 9}, {"permutations": -1},
                       {"permutations": True}, {"max_rows": 0},
                       {"max_rows": 1000001}, {"replay_fraction": .6},
                       {"replay_fraction": math.nan}, {"replay_fraction": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare([original], **kwargs)
        with self.assertRaisesRegex(ValueError, "exceed max_rows"):
            prepare([original], permutations=3, max_rows=3)

    def test_two_options_have_only_one_unique_variant(self):
        original = row()
        original["options"] = original["options"][:2]
        variants = permute_choices(original, 3, 8)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["label"], 0)

    def test_string_options_move_by_key_and_identity_collision_fails(self):
        original = row()
        original["options"] = ["a", "b", "c"]
        variants = permute_choices(original, 3, 5)
        self.assertTrue(all(r["options"][r["label"]] == "b" for r in variants))
        collided = row("different")
        collided["id"] = variants[0]["id"]
        with self.assertRaisesRegex(ValueError, "collision"):
            prepare([original, collided], seed=3, permutations=5)

    def test_bad_keys_hard_labels_soft_labels_fail_before_output(self):
        changes = [
            lambda r: r.update(label=True),
            lambda r: r.update(label=3),
            lambda r: r.update(label_dist=[.7,.2,.1]),
            lambda r: r.update(label_dist=[.2,.7,.2]),
            lambda r: r.update(label_dist=[True,0,0]),
            lambda r: r.update(label_dist=[.1,math.nan,.9]),
            lambda r: r.update(label_dist=[.3,.7]),
            lambda r: r["options"][0].update(key="contradicts"),
        ]
        for change in changes:
            original = row()
            change(original)
            with self.subTest(original=original), self.assertRaises(ValueError):
                prepare([original])
        original = row(kind="yes_no")
        original["label"] = 1
        with self.assertRaises(ValueError):
            prepare([original])

    def test_replay_requires_owned_label_bound_adjudication(self):
        original = row("new")
        replay = adjudicate(row("old"))
        result, report = prepare([original], permutations=0, replay=[replay],
                                  replay_owner="test-owner", replay_fraction=.5)
        self.assertEqual(report["replay_rows"], 1)
        added = result[-1]
        self.assertNotEqual(added["id"], replay["id"])
        self.assertEqual(added["story_id"], replay["story_id"])
        self.assertEqual(added["input"], replay["input"])
        self.assertEqual(added["meta"]["replay_adjudication"],
                         replay["meta"]["replay_adjudication"])
        self.assertEqual(adjudication_binding(added), adjudication_binding(replay))
        for change in [
            lambda r: r["meta"].pop("replay_adjudication"),
            lambda r: r["meta"]["replay_adjudication"].update(status="model_agrees"),
            lambda r: r["meta"]["replay_adjudication"].update(reviewer_id=""),
            lambda r: r["meta"]["replay_adjudication"].update(owner_id="foreign"),
            lambda r: r.update(label=0),
            lambda r: r["input"].update(question="changed after review"),
        ]:
            invalid = copy.deepcopy(replay)
            change(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare([original], replay=[invalid], replay_owner="test-owner")
        with self.assertRaises(ValueError):
            prepare([original], replay=[replay])

    def test_all_supplied_replay_rejected_if_protected_even_outside_budget(self):
        baseline = [row("new"), row("held", "dev"), row("locked", "locked_test")]
        for split in ("dev", "locked_test", "heldout_domain", "heldout_language"):
            bad = adjudicate(row(f"bad-{split}", split))
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "only.*train"):
                prepare(baseline, replay=[bad], replay_owner="test-owner",
                        replay_fraction=0, protected_access_log=self.access_log)
        bad = row("other")
        bad["story_id"] = baseline[1]["story_id"]
        with self.assertRaisesRegex(ValueError, "protected"):
            prepare(baseline, replay=[adjudicate(bad)], replay_owner="test-owner",
                    replay_fraction=0, protected_access_log=self.access_log)

    def test_changed_ids_do_not_hide_source_text_or_hash_leaks(self):
        baseline = [row("new"), row("held", "dev")]
        bad = row("old")
        bad["input"]["evidence"][0].update(
            id="another-local-id",
            text=" \n"+baseline[1]["input"]["evidence"][0]["text"]+"\t ")
        with self.assertRaisesRegex(ValueError, "protected"):
            prepare(baseline, replay=[adjudicate(bad)], replay_owner="test-owner")
        baseline[1]["meta"]["sources"] = [{"document_id": "original-a", "sha256": "a"*64}]
        bad = row("old")
        bad["meta"]["sources"] = [{"document_id": "changed-id", "sha256": "a"*64}]
        with self.assertRaisesRegex(ValueError, "protected"):
            prepare(baseline, replay=[adjudicate(bad)], replay_owner="test-owner")

    def test_baseline_leaks_and_heldout_domain_language_are_not_repaired(self):
        train = row("train")
        evaluation = row("different", "dev")
        evaluation["story_id"] = train["story_id"]
        with self.assertRaises(ValueError):
            prepare([train, evaluation])
        train = row("train")
        evaluation = row("eval", "dev")
        evaluation["input"] = copy.deepcopy(train["input"])
        with self.assertRaises(ValueError):
            prepare([train, evaluation])
        for protected in [row("domain", "heldout_domain"), row("lang", "heldout_language")]:
            bad = row("replay")
            bad["domain_id"], bad["lang"] = protected["domain_id"], protected["lang"]
            with self.subTest(protected=protected), self.assertRaisesRegex(ValueError, "held-out"):
                prepare([row("train"), protected], replay=[adjudicate(bad)],
                        replay_owner="test-owner")

    def test_choice_descriptions_alone_are_not_source_leaks(self):
        baseline = [row("new"), row("dev", "dev")]
        replay = adjudicate(row("old"))
        result, report = prepare(baseline, permutations=0, replay=[replay],
                                  replay_owner="test-owner", replay_fraction=.5)
        self.assertEqual(report["replay_rows"], 1)
        self.assertEqual(result[-1]["options"], baseline[0]["options"])

    def test_replay_fraction_max_rows_and_whole_families(self):
        baseline = [row(f"new-{i}") for i in range(8)]
        replay = [adjudicate(row(f"old-{i}")) for i in range(6)]
        # Two rows are one family; it must not be chopped to fill a single slot.
        replay[1]["story_id"] = replay[0]["story_id"]
        replay[1] = adjudicate(replay[1])
        result, report = prepare(baseline, permutations=0, replay=replay,
                                  replay_owner="test-owner", replay_fraction=.2,
                                  max_rows=9, seed=5)
        self.assertLessEqual(report["replay_fraction_actual"], .2)
        self.assertLessEqual(len(result), 9)
        selected = [r["meta"]["training_preparation"]["original_id"]
                    for r in result[len(baseline):]]
        self.assertEqual(replay[0]["id"] in selected, replay[1]["id"] in selected)
        self.assertEqual((result, report), prepare(baseline, permutations=0,
                         replay=replay, replay_owner="test-owner",
                         replay_fraction=.2, max_rows=9, seed=5))
        self.assertTrue(report["replay_families_skipped"])

    def test_replay_source_linked_families_are_selected_together(self):
        baseline = [row(f"train-{i}") for i in range(4)]
        a, b = row("a"), row("b")
        a["meta"]["document_id"] = b["meta"]["document_id"] = "same-original"
        result, report = prepare(baseline, permutations=0,
                                replay=[adjudicate(a), adjudicate(b)],
                                replay_owner="test-owner", replay_fraction=.2)
        self.assertEqual(report["replay_cap_rows"], 1)
        self.assertEqual(report["replay_rows"], 0)
        self.assertEqual(len(result), 4)

    def test_cli_preserves_byte_prefix_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            baseline, out = root/"baseline.jsonl", root/"prepared.jsonl"
            rows = [row("train"), row("dev", "dev"), row("locked", "locked_test")]
            raw = b"\n"+b"\n".join(json.dumps(r, ensure_ascii=True,
                                 indent=None, separators=(", ", ": ")).encode()
                                 for r in rows)+b"\n\n"
            baseline.write_bytes(raw)
            cmd = [sys.executable, str(ROOT/"scripts/prepare_training_data.py"),
                   "--dataset", str(baseline), "--out", str(out), "--seed", "7",
                   "--protected-access-log", str(root/"access.jsonl")]
            process = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertEqual(baseline.read_bytes(), raw)
            self.assertTrue(out.read_bytes().startswith(raw))
            self.assertEqual(report["output_sha256"],
                             hashlib.sha256(out.read_bytes()).hexdigest())
            kept = out.read_bytes()
            process = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(out.read_bytes(), kept)
            self.assertEqual(baseline.read_bytes(), raw)

    def test_json_duplicates_nonfinite_unicode_and_output_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = root/"input.jsonl"
            for raw in [b'{"id":"a","id":"b"}\n', b'{"id":NaN}\n',
                        json.dumps(dict(row(), id="\ud800")).encode()]:
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    read_jsonl(path, 100)
            out = root/"bounded.jsonl"
            with patch("prepare_training_data.MAX_OUTPUT_BYTES", 4):
                with self.assertRaises(ValueError):
                    write_output(out, [b"too long"], [])
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob(".bounded*")), [])

    def test_permutation_collision_fallback_preserves_keys(self):
        import prepare_training_data as module
        original_digest = module.digest
        def collided_rank(value):
            if isinstance(value, list) and len(value) == 5 and value[0] == PROTOCOL:
                return "same-rank"
            return original_digest(value)
        with patch("prepare_training_data.digest", side_effect=collided_rank):
            variants = permute_choices(row(), 42, 8)
        self.assertEqual(len(variants), 5)
        self.assertEqual(len({tuple(o["key"] for o in v["options"]) for v in variants}), 5)
        self.assertTrue(all(v["options"][v["label"]]["key"] == "contradicts" for v in variants))

    def test_replay_catalog_change_requires_options_even_without_augmentation(self):
        baseline, old = row("baseline"), row("replay")
        old["options"] = list(reversed(old["options"]))
        old["label"] = 1
        prepared, report = prepare([baseline], permutations=0,
                                  replay=[adjudicate(old)], replay_owner="test-owner",
                                  replay_fraction=.5)
        self.assertEqual(len(prepared), 2)
        self.assertEqual(report["per_example_choice_tasks"], ["r_evidence_support"])
        old = row("replay")
        # String and object choices with the same keys/empty texts are equivalent in Go.
        baseline["options"] = ["a", "b", "c"]
        old["options"] = [{"key": k} for k in ["a", "b", "c"]]
        _, report = prepare([baseline], permutations=0, replay=[adjudicate(old)],
                            replay_owner="test-owner", replay_fraction=.5)
        self.assertEqual(report["per_example_choice_tasks"], [])

    def test_locked_default_refuses_before_logical_validation_or_hash(self):
        import prepare_training_data as module
        locked = {"split": "locked_test", "label": "poison"}
        with patch("prepare_training_data.validate_row", side_effect=AssertionError("logical label read")), patch("prepare_training_data.digest", side_effect=AssertionError("source hash")):
            with self.assertRaisesRegex(ValueError, "protected_access_log"):
                prepare([locked])
            path = Path(self._private.name)/"locked.jsonl"
            path.write_text(json.dumps(locked))
            with self.assertRaisesRegex(ValueError, "protected-access-log"):
                read_jsonl(path, 10)
        self.assertFalse(self.access_log.exists())

    def test_explicit_access_is_durable_before_read_hash_and_failed_validation(self):
        import prepare_training_data as module
        path = Path(self._private.name)/"locked.jsonl"
        raw = json.dumps(dict(row("locked", "locked_test"), label="invalid")).encode()
        path.write_bytes(raw)
        original_read = Path.read_bytes
        def checked_read(value):
            if value == path:
                events = [json.loads(line) for line in original_read(self.access_log).splitlines()]
                self.assertEqual(events[-1]["event"], "authorized_read_started")
                self.assertEqual(events[-1]["scope"], "jsonl_file_read")
            return original_read(value)
        original_validate = module.validate_row
        def checked_validate(value):
            events = [json.loads(line) for line in original_read(self.access_log).splitlines()]
            self.assertEqual(events[-1]["event"], "source_snapshot_hashed")
            self.assertEqual(events[-1]["snapshots"][0]["sha256"], hashlib.sha256(raw).hexdigest())
            return original_validate(value)
        with patch.object(Path, "read_bytes", checked_read), patch("prepare_training_data.validate_row", side_effect=checked_validate):
            with self.assertRaisesRegex(ValueError, "hard label"):
                read_jsonl(path, 10, protected_access_log=self.access_log)
        logged = self.access_log.read_text()
        self.assertNotIn("Original record", logged)
        self.assertNotIn("invalid", logged)
        self.assertEqual(path.read_bytes(), raw)

    def test_log_alias_symlink_unrelated_file_and_fake_api_authorization_refused(self):
        original = row("locked", "locked_test")
        for fake in [True, {}, {"authorized": True}]:
            with self.subTest(fake=fake), self.assertRaises(ValueError):
                prepare([original], protected_access_log=fake)
        path = Path(self._private.name)/"locked.jsonl"
        raw = json.dumps(original).encode()
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "alias"):
            read_jsonl(path, 10, protected_access_log=path)
        self.assertEqual(path.read_bytes(), raw)
        target = Path(self._private.name)/"unrelated"
        target.write_text("keep this unchanged")
        with self.assertRaisesRegex(ValueError, "not a protected-access log"):
            prepare([original], protected_access_log=target)
        self.assertEqual(target.read_text(), "keep this unchanged")
        link = Path(self._private.name)/"linked-log"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare([original], protected_access_log=link)
        self.assertEqual(target.read_text(), "keep this unchanged")

    def test_api_log_covers_here_only_and_does_not_change_prepared_bytes(self):
        original = row("locked", "locked_test")
        before = copy.deepcopy(original)
        output, report = prepare([original], protected_access_log=self.access_log)
        self.assertEqual(output, [before])
        self.assertFalse(report["caller_prior_source_access_established"])
        events = [json.loads(line) for line in self.access_log.read_text().splitlines()]
        self.assertEqual([e["event"] for e in events], ["authorized_read_started", "source_snapshot_hashed"])
        self.assertEqual(events[0]["request_id"], events[1]["request_id"])
        self.assertEqual(events[1]["scope"], "in_memory_processing_only")
        self.assertTrue(all(e["gold_payload_in_log"] is False for e in events))
        self.assertNotIn("Original record", self.access_log.read_text())

    def test_preallocation_byte_budget_refuses_before_extra_deepcopy(self):
        import prepare_training_data as module
        original = row()
        first = permute_choices(original, 42, 1)[0]
        budget = len(canonical(original))+1+len(canonical(first))+1
        cloned_variants = []
        deepcopy = copy.deepcopy
        def measured(value, *args, **kwargs):
            if isinstance(value, dict) and "::perm-" in value.get("id", ""):
                cloned_variants.append(value["id"])
            return deepcopy(value, *args, **kwargs)
        with patch("prepare_training_data.MAX_OUTPUT_BYTES", budget), patch("prepare_training_data.copy.deepcopy", side_effect=measured):
            with self.assertRaisesRegex(ValueError, "output byte budget"):
                prepare([original], seed=42, permutations=5)
        self.assertEqual(len(cloned_variants), 1)

    def test_replay_byte_budget_refuses_before_accumulating_extra_copy(self):
        baseline = [row(f"base-{i}") for i in range(4)]
        replay = [adjudicate(row(f"old-{i}")) for i in range(3)]
        complete, _ = prepare(baseline, permutations=0, replay=replay,
                              replay_owner="test-owner", replay_fraction=.5,
                              seed=42)
        first = complete[len(baseline)]
        budget = sum(len(canonical(r))+1 for r in baseline)+len(canonical(first))+1
        deepcopy = copy.deepcopy
        cloned = []
        def measured(value, *args, **kwargs):
            if isinstance(value, dict) and "::replay-" in value.get("id", ""):
                cloned.append(value["id"])
            return deepcopy(value, *args, **kwargs)
        with patch("prepare_training_data.MAX_OUTPUT_BYTES", budget), patch("prepare_training_data.copy.deepcopy", side_effect=measured):
            with self.assertRaisesRegex(ValueError, "output byte budget"):
                prepare(baseline, permutations=0, replay=replay,
                        replay_owner="test-owner", replay_fraction=.5, seed=42)
        self.assertEqual(len(cloned), 1)

    def test_combined_input_byte_cap_precedes_any_file_read(self):
        import prepare_training_data as module
        root = Path(self._private.name)
        one, two = root/"one.jsonl", root/"two.jsonl"
        one.write_bytes(canonical(row("one")))
        two.write_bytes(canonical(row("two")))
        budget = max(one.stat().st_size, two.stat().st_size)+1
        with patch("prepare_training_data.MAX_FILE_BYTES", budget), patch("prepare_training_data.read_jsonl", side_effect=AssertionError("file read before combined cap")), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                module.main(["--dataset", str(one), "--dataset", str(two),
                             "--out", str(root/"out.jsonl")])
        self.assertFalse((root/"out.jsonl").exists())

    def test_go_validate_prepare_and_dynamic_serving_catalog(self):
        if shutil.which("go") is None:
            self.skipTest("Go integration requires the package Go toolchain")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = row()
            original["label_dist"] = [.1,.8,.1]
            prepared, _ = prepare([original], permutations=3, seed=42)
            dataset = root/"prepared.jsonl"
            dataset.write_bytes(b"".join(canonical(r)+b"\n" for r in prepared))
            source = root/"check.go"
            source.write_text(GO_CHECK)
            process = subprocess.run(["go", "run", str(source), str(dataset),
                                      str(ROOT), str(root/"artifact")],
                                     cwd=ROOT, text=True, capture_output=True,
                                     timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertEqual(report["examples"], 4)
            self.assertEqual(report["options"], "per_example")
            self.assertTrue(report["missing_options_rejected"])
            self.assertTrue(report["explicit_options_allowed"])
            self.assertTrue(report["hard_and_soft_key_bindings"])
            self.assertTrue(report["bad_soft_target_rejected"])
            self.assertTrue(report["split_leak_rejected"])


GO_CHECK = r'''
package main
import ("encoding/json";"fmt";"os";"path/filepath";"context";"github.com/UnicoLab/ul_layago/train";"github.com/UnicoLab/ul_layago/laya")
func main(){
 dataset,root,out:=os.Args[1],os.Args[2],os.Args[3]
 ds,err:=train.LoadDataset([]string{dataset},"R");if err!=nil{panic(err)}
 trunk,err:=train.OpenTrunk(train.TrunkSpec{Dir:filepath.Join(root,"testdata/parity/checkpoint"),MaxLen:1024,HeadMaxLen:512});if err!=nil{panic(err)}
 prepared,err:=train.Prepare(ds,trunk,train.PrepareOptions{});if err!=nil{panic(err)}
 _=context.Background()
 task:=prepared.Tasks["r_evidence_support"];if task.Options!="per_example"||string(task.FixedOptions)!="null"{panic("fixed options falsely retained")}
 bound:=true
 for _,ex:=range ds.Examples{if ex.GoldKey()!="contradicts"{bound=false};for i,o:=range ex.Options{want:=map[string]float64{"supports":.1,"contradicts":.8,"insufficient":.1}[o.Key];if ex.Target()[i]!=want{bound=false}}}
 if err=os.Mkdir(out,0700);err!=nil{panic(err)}
 raw,err:=os.ReadFile(filepath.Join(root,"testdata/artifact/tiny-q/float32/manifest.json"));if err!=nil{panic(err)}
 var m map[string]any;if err=json.Unmarshal(raw,&m);err!=nil{panic(err)};m["instance"]="R";m["tasks"]=prepared.Tasks;m["act"].(map[string]any)["thresholds"]=map[string]any{};m["calibration"].(map[string]any)["temperatures"]=map[string]any{};raw,err=json.Marshal(m);if err!=nil{panic(err)}
 if err=os.WriteFile(filepath.Join(out,"manifest.json"),raw,0600);err!=nil{panic(err)}
 weights,err:=os.ReadFile(filepath.Join(root,"testdata/artifact/tiny-q/float32/model.safetensors"));if err!=nil{panic(err)};if err=os.WriteFile(filepath.Join(out,"model.safetensors"),weights,0600);err!=nil{panic(err)}
 art,err:=laya.LoadArtifact(out,filepath.Join(root,"testdata/parity/checkpoint"),laya.ArtifactOptions{});if err!=nil{panic(err)}
 _,missing:=art.Question(laya.ArtifactQuestion{ID:"missing",Task:task.Name})
 var options []laya.Option;for _,o:=range ds.Examples[0].Options{options=append(options,laya.Option{Key:o.Key,Description:o.Text})}
 _,explicit:=art.Question(laya.ArtifactQuestion{ID:"explicit",Task:task.Name,Options:options})
 bytes,err:=os.ReadFile(dataset);if err!=nil{panic(err)};var rows []map[string]any;for _,line:=range splitLines(bytes){var r map[string]any;if json.Unmarshal(line,&r)==nil{rows=append(rows,r)}}
 rows[0]["label_dist"]=[]float64{.9,.05,.05};bad,_:=json.Marshal(rows[0]);badPath:=filepath.Join(filepath.Dir(dataset),"bad.jsonl");os.WriteFile(badPath,bad,0600);_,badError:=train.LoadDataset([]string{badPath},"R")
 rows[0]["label_dist"]=[]float64{.1,.8,.1};rows[1]["split"]="dev";delete(rows[1],"label_dist")
 leakPath:=filepath.Join(filepath.Dir(dataset),"leak.jsonl");f,_:=os.Create(leakPath);for _,r:=range rows{b,_:=json.Marshal(r);f.Write(append(b,'\n'))};f.Close()
 leakDS,leak:=train.LoadDataset([]string{leakPath},"R");if leak==nil{_,leak=train.Prepare(leakDS,trunk,train.PrepareOptions{})}
 json.NewEncoder(os.Stdout).Encode(map[string]any{"examples":len(ds.Examples),"options":task.Options,"missing_options_rejected":missing!=nil,"explicit_options_allowed":explicit==nil,"hard_and_soft_key_bindings":bound,"bad_soft_target_rejected":badError!=nil,"split_leak_rejected":leak!=nil})
 fmt.Fprint(os.Stderr,"")
}
func splitLines(data []byte)[][]byte{var out [][]byte;start:=0;for i,b:=range data{if b=='\n'{if i>start{out=append(out,data[start:i])};start=i+1}};if start<len(data){out=append(out,data[start:])};return out}
'''


if __name__ == "__main__":
    unittest.main()
