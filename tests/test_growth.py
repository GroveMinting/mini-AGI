"""Pool growth/prune during serve, and the save-reload it requires."""

import tempfile
import types
import unittest

import train as T
from minagi.create import create
from minagi.growth import PoolKeeper
from minagi.live import LiveLearner
from minagi.pool import AutoGrow
from minagi.recur import load_any


class _Pool:
    def __init__(self, prune_gone=2):
        self.n = 3
        self.prune_gone = prune_gone
        self.pruned = []

    def prune(self, step, survival=None):
        self.pruned.append((step, survival))
        return self.prune_gone

    def n_experts(self):
        return self.n


class _Grower:
    def __init__(self, grew=3):
        self.grew = grew
        self.calls = []

    def step(self, val, pool, step):
        self.calls.append((val, step))
        pool.n += self.grew
        return {"grew": self.grew, "reason": "test"}


class _Learner:
    def __init__(self, steps=10):
        self.steps = steps


def _model(pool):
    return types.SimpleNamespace(pool=pool)


class PoolKeeperTest(unittest.TestCase):
    def test_cadence_and_model_step(self):
        pool = _Pool()
        grower = _Grower()
        keeper = PoolKeeper(_model(pool), _Learner(10), grower,
                            survival=100, base_step=50, every=2)
        self.assertIsNone(keeper.after_check(1.0))
        out = keeper.after_check(1.0)
        self.assertEqual(out["step"], 60)
        self.assertEqual(out["pruned"], 2)
        self.assertEqual(out["grew"], 3)
        self.assertEqual(pool.pruned, [(60, 100)])
        self.assertEqual(grower.calls, [(1.0, 60)])
        self.assertEqual(len(keeper.events), 1)

    def test_no_change_returns_none(self):
        keeper = PoolKeeper(_model(_Pool(prune_gone=0)), _Learner(),
                            _Grower(grew=0), survival=10, every=1)
        self.assertIsNone(keeper.after_check(1.0))

    def test_prune_can_be_disabled(self):
        pool = _Pool()
        keeper = PoolKeeper(_model(pool), _Learner(), _Grower(), every=1,
                            prune=False)
        keeper.after_check(1.0)
        self.assertEqual(pool.pruned, [])

    def test_zero_cadence_never_runs(self):
        keeper = PoolKeeper(_model(_Pool()), _Learner(), _Grower(), every=0)
        self.assertIsNone(keeper.after_check(1.0))

    def test_rebind_follows_a_new_learner(self):
        keeper = PoolKeeper(_model(_Pool()), _Learner(10), _Grower(),
                            base_step=50, every=1)
        fresh_pool = _Pool()
        keeper.rebind(_model(fresh_pool), _Learner(0), base_step=70)
        self.assertIs(keeper.pool, fresh_pool)
        self.assertEqual(keeper.model_step(), 70)


class GrowthSaveReloadTest(unittest.TestCase):
    def test_grown_expert_is_written_and_reloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/weights"
            create(path, verbose=False, experts=3, resident=1, d_ff=8,
                   depth=1, d_model=8, trunk_d_ff=16, n_head=1, block=16,
                   max_steps=1, top_k=1)
            model, _cfg, _pool, man = T.build_paged(path, "cpu", resident=1,
                                                    ram_capacity=2)
            learner = LiveLearner(model, chunk=8, context=16,
                                  weights_dir=path, manifest=man, save_every=0)
            pool = model.pool
            pool.born.fill_(0.0)
            pool.trial = 0
            grower = AutoGrow(grow_k=1, max_experts=10, dying_frac_max=1.0,
                              keep_ratio_min=0.0, mem_frac_max=1.0,
                              max_disk_gb=1.0)
            keeper = PoolKeeper(model, learner, grower, survival=1,
                                base_step=100, every=1, prune=False)
            before = pool.n_experts()
            out = keeper.after_check(5.0)
            self.assertEqual(out["grew"], 1)
            learner.manifest["step"] = keeper.model_step()
            learner.save()
            fresh, _ = load_any(path, "cpu", read_only=False)
            self.assertEqual(fresh.pool.n_experts(), before + 1)
            with open(f"{path}/manifest.json") as f:
                self.assertEqual(__import__("json").load(f)["n_experts"],
                                 before + 1)

    def test_disk_cap_refuses_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/weights"
            create(path, verbose=False, experts=3, resident=1, d_ff=8,
                   depth=1, d_model=8, trunk_d_ff=16, n_head=1, block=16,
                   max_steps=1, top_k=1)
            model, _cfg, _pool, man = T.build_paged(path, "cpu", resident=1,
                                                    ram_capacity=2)
            learner = LiveLearner(model, chunk=8, context=16,
                                  weights_dir=path, manifest=man, save_every=0)
            pool = model.pool
            pool.born.fill_(0.0)
            pool.trial = 0
            grower = AutoGrow(grow_k=1, max_experts=10, dying_frac_max=1.0,
                              keep_ratio_min=0.0, mem_frac_max=1.0,
                              max_disk_gb=1e-9)
            keeper = PoolKeeper(model, learner, grower, survival=1,
                                base_step=100, every=1, prune=False)
            self.assertIsNone(keeper.after_check(5.0))


if __name__ == "__main__":
    unittest.main()
