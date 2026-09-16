import csv
import hashlib
import importlib
import json

gpqa = importlib.import_module("nemo_skills.dataset.gpqa.prepare")
ifeval = importlib.import_module("nemo_skills.dataset.ifeval.prepare")


def test_gpqa_source_revision_and_official_split_contracts_are_pinned():
    assert gpqa.SOURCE_REPOSITORY == "Idavidrein/gpqa"
    assert gpqa.SOURCE_REVISION == "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
    assert gpqa.SOURCE_FILES == {
        "extended": {
            "filename": "gpqa_extended.csv",
            "rows": 546,
            "sha256": "0926ee24949d02ed6748eb75a2611546c34479e30ddc42efd01d6f1681aaa48a",
        },
        "main": {
            "filename": "gpqa_main.csv",
            "rows": 448,
            "sha256": "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3",
        },
        "diamond": {
            "filename": "gpqa_diamond.csv",
            "rows": 198,
            "sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
        },
    }


def test_gpqa_source_validator_rejects_hash_or_row_drift(tmp_path, monkeypatch):
    path = tmp_path / "gpqa_test.csv"
    with path.open("wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Question"])
        writer.writeheader()
        writer.writerow({"Question": "What?"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        gpqa.SOURCE_FILES,
        "test",
        {"filename": path.name, "rows": 1, "sha256": digest},
    )
    assert gpqa._validate_source_csv(path, "test")["rows"] == 1
    gpqa.SOURCE_FILES["test"]["rows"] = 2
    try:
        gpqa._validate_source_csv(path, "test")
    except RuntimeError as error:
        assert "source contract changed" in str(error)
    else:
        raise AssertionError("row-count drift was accepted")


def test_ifeval_source_revision_and_official_contract_are_pinned():
    assert ifeval.SOURCE_REPOSITORY == "google-research/google-research"
    assert ifeval.SOURCE_REVISION == "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
    assert ifeval.SOURCE_PATH == "instruction_following_eval/data/input_data.jsonl"
    assert ifeval.SOURCE_SHA256 == "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
    assert ifeval.EXPECTED_ROWS == 541
    assert ifeval.EXPECTED_INSTRUCTION_INSTANCES == 834
    assert ifeval.EXPECTED_INSTRUCTION_TYPES == 25


def test_ifeval_source_validator_rejects_hash_or_inventory_drift(tmp_path, monkeypatch):
    path = tmp_path / "input_data.jsonl"
    rows = [
        {
            "key": 1,
            "prompt": "Write one word.",
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{}],
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(ifeval, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(ifeval, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(ifeval, "EXPECTED_INSTRUCTION_INSTANCES", 1)
    monkeypatch.setattr(ifeval, "EXPECTED_INSTRUCTION_TYPES", 1)
    assert ifeval._validate_source(path) == rows
    monkeypatch.setattr(ifeval, "EXPECTED_INSTRUCTION_INSTANCES", 2)
    try:
        ifeval._validate_source(path)
    except RuntimeError as error:
        assert "source contract changed" in str(error)
    else:
        raise AssertionError("instruction inventory drift was accepted")
