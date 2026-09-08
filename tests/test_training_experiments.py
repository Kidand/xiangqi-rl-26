"""Regression tests for independent continuation, target diagnostics, and CSV upgrades."""
import csv
import hashlib
import math

import numpy as np
import pytest
import torch

from rl.config import Config, ModelConfig
from rl.logger import CSV_COLUMNS, TrainLogger
from rl.model import create_model, save_checkpoint
from rl.replay_buffer import ReplayBuffer
from rl.targets import value_target_metadata
from rl.train import _train_phase, _check_checkpoint_contract, run_training


class NullLogger:
    def train_step(self, **kwargs):
        self.step = kwargs


class DistributionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((8100,), -1000.0))
        with torch.no_grad():
            self.logits[:3] = torch.tensor([.5, .25, .25]).log()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, states):
        self.seen_states = states.detach().clone()
        return self.logits.expand(len(states), -1), self.value.expand(len(states))


def test_true_kl_does_not_subtract_prediction_entropy():
    cfg = Config()
    cfg.train.batch_size = 1
    cfg.train.lr = cfg.train.lr_min = 0.0
    cfg.train.total_iterations = 1
    model = DistributionNet()
    buffer = ReplayBuffer(1, (31, 10, 9))
    buffer.add_game(np.zeros((1, 31, 10, 9), np.uint8),
                    [(np.array([0, 1]), np.array([.5, .5], np.float32))], np.zeros(1))
    logger = NullLogger()
    result = _train_phase(model, torch.optim.SGD(model.parameters(), lr=0),
                          buffer, cfg, 0, "cpu", logger, 1)
    # CE==H(p) while KL is strictly positive: catches the earlier diagnostic mistake.
    assert result["policy_loss"] == pytest.approx(result["entropy"], abs=1e-6)
    assert result["target_entropy"] == pytest.approx(math.log(2), abs=1e-6)
    assert result["policy_kl"] == pytest.approx(.5 * math.log(2), abs=1e-6)
    assert logger.step["policy_kl"] == result["policy_kl"]


def test_training_mirrors_state_and_target_together_without_changing_value():
    cfg = Config()
    cfg.train.batch_size = 1
    cfg.train.mirror_prob = 1
    cfg.train.lr = cfg.train.lr_min = 0
    model = DistributionNet()
    buffer = ReplayBuffer(1, (31, 10, 9))
    states = np.zeros((1, 31, 10, 9), np.uint8)
    states[0, 0, 0, 0] = 1
    buffer.add_game(states, [(np.array([0]), np.array([1.0]))], np.array([.6]))
    result = _train_phase(model, torch.optim.SGD(model.parameters(), lr=0),
                          buffer, cfg, 0, "cpu", NullLogger(), 1)
    assert model.seen_states[0, 0, 0, 8].item() == 1
    assert model.seen_states[0, 0, 0, 0].item() == 0
    assert result["policy_loss"] == pytest.approx(1000, abs=.001)  # action 0 -> 8*90+8
    assert result["value_loss"] == pytest.approx(.36, abs=1e-6)
    assert buffer._states[0, 0, 0, 0] == 1


def test_csv_upgrade_preserves_old_values_and_blanks_new_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    old_columns = CSV_COLUMNS[:-4]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({"iter": 1498, "policy_loss": "0.995776", "entropy": "0.995868"})
    with TrainLogger(str(tmp_path), tensorboard=False) as logger:
        logger.iteration_summary({"iter": 1499, "target_entropy": .8, "policy_kl": .1})
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == 2
    assert rows[0]["policy_loss"] == "0.995776"
    assert rows[0]["target_entropy"] == ""
    assert rows[1]["policy_kl"] == "0.1"


def test_unknown_csv_schema_rejected_without_rewriting(tmp_path):
    path = tmp_path / "metrics.csv"
    original = b"iter,unknown\n1,2\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="表头"):
        TrainLogger(str(tmp_path), tensorboard=False)
    assert path.read_bytes() == original


def experiment_config(tmp_path):
    cfg = Config()
    cfg.model = ModelConfig(blocks=1, filters=4)
    cfg.train.device = "cpu"
    cfg.train.buffer_window = 8
    cfg.train.total_iterations = 0
    cfg.train.checkpoint_dir = str(tmp_path / "run" / "ckpts")
    cfg.log.log_dir = str(tmp_path / "run" / "logs")
    cfg.log.records_dir = str(tmp_path / "run" / "records")
    cfg.log.tensorboard = False
    return cfg


def test_weight_only_init_preserves_source_and_resets_lineage(tmp_path):
    cfg = experiment_config(tmp_path)
    source = tmp_path / "0716.pt"
    model = create_model(cfg.model)
    save_checkpoint(model, cfg.model, source, iteration=1498)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    run_training(cfg, init_from=str(source))
    dest = torch.load(tmp_path / "run/ckpts/best.pt", weights_only=False)
    assert dest["iteration"] == -1
    assert dest["optimizer_state"] is None
    assert dest["meta"]["init_from"]["iteration"] == 1498
    assert dest["meta"]["init_from"]["sha256"] == before
    assert dest["meta"]["value_target"] == value_target_metadata(cfg)
    assert dest["meta"]["config"]["model"] == dest["model_config"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(dest["model_state"][name], value, rtol=0, atol=0)
    # A repeated fresh start must fail before overwriting its results.
    with pytest.raises(ValueError, match="必须为空"):
        run_training(cfg, init_from=str(source))


def test_init_architecture_mismatch_leaves_output_absent(tmp_path):
    cfg = experiment_config(tmp_path)
    source = tmp_path / "other.pt"
    other = ModelConfig(blocks=2, filters=4)
    save_checkpoint(create_model(other), other, source)
    with pytest.raises(ValueError, match="结构不匹配"):
        run_training(cfg, init_from=str(source))
    assert not (tmp_path / "run").exists()


def test_resume_contract_cannot_override_known_mismatch():
    cfg = Config()
    expected = value_target_metadata(cfg)
    saved = dict(expected, draw_penalty=-.2)
    ckpt = {"model_config": cfg.to_dict()["model"], "meta": {"value_target": saved}}
    with pytest.raises(ValueError, match="契约"):
        _check_checkpoint_contract(ckpt, cfg, expected, allow_legacy=True)


def test_zero_sum_cannot_accept_unknown_legacy_buffer(tmp_path):
    cfg = experiment_config(tmp_path)
    with pytest.raises(ValueError, match="新 buffer"):
        run_training(cfg, resume=True, allow_legacy_buffer=True)


def test_legacy_resume_does_not_certify_unknown_buffer(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import rl.train as train
    cfg = experiment_config(tmp_path)
    cfg.selfplay.draw_penalty = -.2
    cfg.train.total_iterations = 2
    cfg.train.min_buffer_to_train = 100
    cfg.train.async_buffer_save = "sync"
    ckpts = Path(cfg.train.checkpoint_dir)
    model = create_model(cfg.model)
    save_checkpoint(model, cfg.model, ckpts / "iter_0000.pt", iteration=0)
    save_checkpoint(model, cfg.model, ckpts / "best.pt", iteration=0)
    ReplayBuffer(8, (31, 10, 9)).save(ckpts / "buffer")

    def selfplay(*args):
        return dict(games=1, new_samples=0, red_win=0, black_win=0, draw=1,
                    red_winrate=0., black_winrate=0., draw_rate=1., avg_plies=2.,
                    resign_rate=0., resign_fp_rate=0.)
    def arena(*args):
        return dict(games=2, wins=0, draws=2, losses=0, score=.5, score_se=0.,
                    red_wins=0, red_draws=1, red_losses=0, red_score=.5,
                    black_wins=0, black_draws=1, black_losses=0, black_score=.5,
                    distinct_openings=1)
    monkeypatch.setattr(train, "_selfplay_phase", selfplay)
    monkeypatch.setattr(train, "arena", arena)
    run_training(cfg, resume=True, allow_legacy_buffer=True)
    meta = json.loads((ckpts / "buffer/meta.json").read_text())
    assert meta.get("target_metadata") is None
    saved = torch.load(ckpts / "iter_0001.pt", weights_only=False)
    assert saved["meta"]["legacy_metadata_accepted"] is True
    # Use a known best contract so the buffer guard itself is exercised.
    save_checkpoint(model, cfg.model, ckpts / "best.pt", iteration=1, meta=saved["meta"])
    with pytest.raises(ValueError, match="target_metadata"):
        run_training(cfg, resume=True)
    cfg.train.total_iterations = 3
    run_training(cfg, resume=True, allow_legacy_buffer=True)
    assert json.loads((ckpts / "buffer/meta.json").read_text()).get("target_metadata") is None
    assert torch.load(ckpts / "iter_0002.pt", weights_only=False)["meta"]["legacy_metadata_accepted"]


def test_presets_are_isolated_single_variable_groups():
    from pathlib import Path
    configs = {name: Config.from_yaml(Path(__file__).parents[1] / "configs/experiments" / f"{name}.yaml")
               for name in ("control", "mirror", "zero", "zero_mirror")}
    def core(c):
        d = c.to_dict()
        for k in ("checkpoint_dir", "mirror_prob"):
            d["train"].pop(k)
        for k in ("log_dir", "records_dir"):
            d["log"].pop(k)
        d["selfplay"].pop("draw_penalty")
        return d
    for c in configs.values():
        assert c.model == ModelConfig(blocks=15, filters=192, se=True, history_steps=2)
        assert c.selfplay.value_blend_iters == 0
        assert core(c) == core(configs["control"])
    assert len({c.train.checkpoint_dir for c in configs.values()}) == 4
    assert configs["mirror"].train.mirror_prob == .5
    assert configs["zero"].selfplay.draw_penalty == 0
