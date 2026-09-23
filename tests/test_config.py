"""Test configuration module integrity"""
from nemesys_gnn4id.config import (
    ATTACK_TYPES, LABEL_DICT, GNN_DEFAULTS, NEMESYS_DEFAULTS,
    FLOW_FEATURE_NAMES, TRAIN_DEFAULTS, EVAL_DEFAULTS, from_env,
)


class TestAttackTypes:
    def test_eight_classes(self):
        assert len(ATTACK_TYPES) == 8

    def test_has_benign(self):
        assert ATTACK_TYPES[0]["name_en"] == "Benign"

    def test_labels_match_attack_types(self):
        assert len(LABEL_DICT) == 8
        for name, idx in LABEL_DICT.items():
            assert ATTACK_TYPES[idx]["name_en"] == name


class TestFlowFeatures:
    def test_feature_count(self):
        assert len(FLOW_FEATURE_NAMES) == 82

    def test_contains_proto_onehot(self):
        assert "proto_6" in FLOW_FEATURE_NAMES  # TCP
        assert "proto_17" in FLOW_FEATURE_NAMES  # UDP


class TestDefaults:
    def test_gnn_defaults(self):
        assert "hidden_size" in GNN_DEFAULTS
        assert GNN_DEFAULTS["hidden_size"] == 64

    def test_nemesys_defaults(self):
        assert "sigma" in NEMESYS_DEFAULTS
        assert NEMESYS_DEFAULTS["sigma"] == 0.6

    def test_train_defaults(self):
        assert "batch_size" in TRAIN_DEFAULTS
        assert TRAIN_DEFAULTS["epochs"] == 100

    def test_eval_defaults(self):
        assert "proto_threshold" in EVAL_DEFAULTS


class TestFromEnv:
    def test_unknown_key_returns_default(self):
        result = from_env("nonexistent_key", default=42)
        assert result == 42
