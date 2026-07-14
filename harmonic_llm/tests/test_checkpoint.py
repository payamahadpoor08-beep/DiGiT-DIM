"""
Tests for checkpoint save/load and training resume.
"""
import os
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

import harmonic_llm as hl
from harmonic_llm.training import (
    ByteTokenizer, LanguageModelDataset, TrainConfig, train,
    save_checkpoint, load_checkpoint, _config_to_dict,
)


def _tiny_setup():
    model = hl.build_model(hl.ModelConfig.tiny())
    tok = ByteTokenizer()
    ds = LanguageModelDataset("hello world " * 60, tok, seq_len=16)
    return model, ds


# ---------------------------------------------------------------------------
# Metadata normalisation (no model needed)
# ---------------------------------------------------------------------------
class TestConfigToDict:
    def test_model_config(self):
        d = _config_to_dict(hl.ModelConfig.tiny())
        assert isinstance(d, dict) and d["dim"] == 128

    def test_plain_dict_and_none(self):
        assert _config_to_dict({"a": 1}) == {"a": 1}
        assert _config_to_dict(None) is None


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
class TestCheckpointIO:
    def test_roundtrip_metadata_and_weights(self, tmp_path):
        model, _ = _tiny_setup()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = save_checkpoint(str(tmp_path / "c.pt"), model, optimizer=opt,
                               step=7, epoch=2, config=hl.ModelConfig.tiny(),
                               tokenizer_meta={"kind": "byte", "vocab": 259})
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")      # atomic: no temp left behind

        ck = load_checkpoint(path)
        assert ck["step"] == 7 and ck["epoch"] == 2
        assert ck["config"]["dim"] == 128
        assert ck["tokenizer_meta"]["vocab"] == 259
        assert ck["optimizer"] is not None
        sd = model.state_dict()
        for k, v in ck["model"].items():
            assert torch.equal(v, sd[k])

    def test_load_restores_perturbed_weights(self, tmp_path):
        model, _ = _tiny_setup()
        path = save_checkpoint(str(tmp_path / "c.pt"), model, step=1)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        model.load_state_dict(load_checkpoint(path)["model"])
        for k, v in model.state_dict().items():
            assert torch.equal(v, saved[k])

    def test_creates_missing_dir(self, tmp_path):
        model, _ = _tiny_setup()
        nested = tmp_path / "a" / "b" / "c.pt"
        save_checkpoint(str(nested), model, step=0)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
class TestResume:
    def test_final_checkpoint_written(self, tmp_path):
        model, ds = _tiny_setup()
        cfg = TrainConfig(epochs=1, batch_size=2, log_every=1000, eval_every=1000,
                          checkpoint_dir=str(tmp_path))
        r1 = train(model, ds, ds, cfg, config=hl.ModelConfig.tiny())
        final = tmp_path / "final.pt"
        assert final.exists()
        ck = load_checkpoint(str(final))
        assert ck["step"] == r1.total_steps and ck["epoch"] == 1

    def test_periodic_checkpoints(self, tmp_path):
        model, ds = _tiny_setup()
        cfg = TrainConfig(epochs=1, batch_size=2, log_every=1000, eval_every=1000,
                          checkpoint_dir=str(tmp_path), save_every=2)
        train(model, ds, ds, cfg)
        step_ckpts = list(tmp_path.glob("step_*.pt"))
        assert len(step_ckpts) >= 1

    def test_resume_continues_step_count(self, tmp_path):
        model, ds = _tiny_setup()
        cfg = TrainConfig(epochs=1, batch_size=2, log_every=1000, eval_every=1000,
                          checkpoint_dir=str(tmp_path))
        r1 = train(model, ds, ds, cfg, config=hl.ModelConfig.tiny())
        steps_per_epoch = r1.total_steps
        final = str(tmp_path / "final.pt")

        model2 = hl.build_model(hl.ModelConfig.tiny())
        cfg2 = TrainConfig(epochs=2, batch_size=2, log_every=1000, eval_every=1000,
                           resume_from=final)
        r2 = train(model2, ds, ds, cfg2)
        # Restored at end of epoch 0 (step=steps_per_epoch), ran epoch 1 only.
        assert r2.total_steps == 2 * steps_per_epoch
