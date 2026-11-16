from __future__ import annotations

from typing import Any
import unittest

import numpy as np
import pandas as pd

from sustaingym.algorithms.data_center import (
    ASAP, DependencyCriticality, ForecastWindow, job_centrality, signal_entropy)
from sustaingym.envs.data_center import (
    DataCenterEnv, DependencyJobGenerator, Job, JobGenerator,
    SyntheticJobGenerator)


class FixedJobs(JobGenerator):
    """Returns the same hand-written jobs every episode."""
    def __init__(self, jobs: list[dict[str, Any]]):
        super().__init__(max_slack_hours=1, workload_rule='mix', mix={'W4': 1.0})
        self.spec = jobs

    def generate(self, day: pd.Timestamp, num_gpus: int,
                 rng: np.random.Generator) -> list[Job]:
        return [Job(job_id=i, workload='W4', **kw) for i, kw in enumerate(self.spec)]


def small_env(**kwargs: Any) -> DataCenterEnv:
    kwargs.setdefault('num_gpus', 16)
    kwargs.setdefault('date_range', ('2020-06-01', '2020-06-10'))
    return DataCenterEnv(**kwargs)


def job_fields(jobs: list[Job]) -> list[tuple]:
    return [(j.job_id, j.arrival, j.duration, j.num_gpus, j.workload, j.deadline)
            for j in jobs]


class TestDependencyJobGenerator(unittest.TestCase):
    def test_base_jobs_are_unchanged(self) -> None:
        base = SyntheticJobGenerator()
        wrapped = DependencyJobGenerator(SyntheticJobGenerator(), dep_prob=0.5)
        day = pd.Timestamp('2020-06-02')
        a = base.generate(day, 16, np.random.default_rng(5))
        b = wrapped.generate(day, 16, np.random.default_rng(5))
        self.assertEqual(job_fields(a), job_fields(b))
        self.assertTrue(any(j.parents for j in b))
        self.assertTrue(any(j.tier for j in b))

    def test_parents_are_earlier_and_feasible(self) -> None:
        gen = DependencyJobGenerator(SyntheticJobGenerator(), dep_prob=0.6)
        jobs = gen.generate(pd.Timestamp('2020-06-02'), 16, np.random.default_rng(1))
        window = gen.window_steps
        for child in jobs:
            self.assertLessEqual(len(child.parents), 2)
            for p in child.parents:
                parent = jobs[p]
                self.assertLess(p, child.job_id)
                self.assertLessEqual(child.arrival - parent.arrival, window)
                self.assertLessEqual(parent.arrival + parent.duration, child.latest_start)

    def test_off_switch_matches_base_episode(self) -> None:
        plain = small_env()
        wrapped = small_env(job_generator=DependencyJobGenerator(
            SyntheticJobGenerator(), dep_prob=0.0, critical_frac=0.0))
        for policy in (ASAP, ForecastWindow):
            for safe in (False, True):
                a = policy(plain, deadline_safe=safe).run_episode(seed=4)
                b = policy(wrapped, deadline_safe=safe).run_episode(seed=4)
                self.assertEqual(a, b)
                self.assertEqual(plain.schedule(), wrapped.schedule())

    def test_rejects_bad_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            DependencyJobGenerator(SyntheticJobGenerator(), dep_prob=1.5)


class TestDependenciesInEnv(unittest.TestCase):
    def test_child_waits_for_parent(self) -> None:
        jobs = [dict(arrival=0, duration=12, num_gpus=2, deadline=60),
                dict(arrival=0, duration=6, num_gpus=2, deadline=60, parents=(0,))]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        ASAP(env).run_episode(seed=0)
        parent, child = env.jobs
        self.assertEqual(parent.start, 0)
        self.assertEqual(child.start, parent.finish)
        summary = env.episode_summary()
        self.assertEqual(summary['misses'], 0)
        self.assertEqual(summary['finished'], 2)

    def test_blocked_child_misses_are_counted(self) -> None:
        # the parent runs past the child's deadline, so the child misses
        # while it is still waiting for the parent
        jobs = [dict(arrival=0, duration=24, num_gpus=2, deadline=30),
                dict(arrival=0, duration=6, num_gpus=2, deadline=12, parents=(0,), tier=1)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        summary = ASAP(env).run_episode(seed=0)
        self.assertEqual(summary['misses'], 1)
        self.assertEqual(summary['dependent_misses'], 1)
        self.assertEqual(summary['critical_misses'], 1)
        self.assertTrue(env.jobs[1].missed)

    def test_rejects_parents_that_are_not_earlier(self) -> None:
        jobs = [dict(arrival=0, duration=6, num_gpus=1, deadline=30, parents=(1,)),
                dict(arrival=0, duration=6, num_gpus=1, deadline=30)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        with self.assertRaises(ValueError):
            env.reset(seed=0)


class TestDependencyCriticality(unittest.TestCase):
    def test_centrality_is_zero_without_edges(self) -> None:
        jobs = [Job(job_id=i, arrival=0, duration=1, num_gpus=1, workload='W4',
                    deadline=10) for i in range(4)]
        self.assertTrue(np.array_equal(job_centrality(jobs), np.zeros(4)))

    def test_shared_parent_ranks_highest(self) -> None:
        jobs = [Job(job_id=i, arrival=0, duration=1, num_gpus=1, workload='W4',
                    deadline=10, parents=(0,) if i in (1, 2, 3) else ())
                for i in range(5)]
        c = job_centrality(jobs)
        self.assertEqual(int(np.argmax(c)), 0)
        self.assertAlmostEqual(float(c.max()), 1.0)
        self.assertAlmostEqual(float(c.min()), 0.0)

    def test_signal_entropy_range(self) -> None:
        self.assertEqual(signal_entropy(np.full(20, 0.3)), 0.0)
        self.assertEqual(signal_entropy(np.array([0.3])), 0.0)
        spread = signal_entropy(np.linspace(0.0, 0.999, 100))
        self.assertAlmostEqual(spread, 1.0, places=6)

    def test_scores_are_in_range(self) -> None:
        env = small_env(job_generator=DependencyJobGenerator(SyntheticJobGenerator()))
        policy = DependencyCriticality(env, deadline_safe=True)
        env.reset(seed=2)
        for _ in range(60):
            for job in env.queue_window_jobs():
                s = policy.score(job, job.latest_start)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)
            env.step(policy.get_action(np.zeros(1)))

    def test_full_score_starts_at_once(self) -> None:
        jobs = [dict(arrival=0, duration=6, num_gpus=1, deadline=200, tier=1)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        DependencyCriticality(env, weights=(0.0, 0.0, 1.0)).run_episode(seed=0)
        self.assertEqual(env.jobs[0].start, 0)

    def test_zero_score_matches_forecast_window(self) -> None:
        gen = DependencyJobGenerator(SyntheticJobGenerator(), dep_prob=0.0,
                                     critical_frac=0.0)
        env = small_env(job_generator=gen)
        for safe in (False, True):
            a = ForecastWindow(env, deadline_safe=safe).run_episode(seed=6)
            sa = env.schedule()
            b = DependencyCriticality(env, weights=(0.0, 0.0, 1.0),
                                      deadline_safe=safe).run_episode(seed=6)
            self.assertEqual(a, b)
            self.assertEqual(sa, env.schedule())

    def test_same_seed_same_schedule(self) -> None:
        env = small_env(job_generator=DependencyJobGenerator(SyntheticJobGenerator()))
        policy = DependencyCriticality(env, deadline_safe=True, reserve_frac=0.05)
        a = policy.run_episode(seed=3)
        sa = env.schedule()
        b = policy.run_episode(seed=3)
        self.assertEqual(a, b)
        self.assertEqual(sa, env.schedule())

    def test_rejects_bad_weights(self) -> None:
        env = small_env()
        with self.assertRaises(ValueError):
            DependencyCriticality(env, weights=(0.6, 0.5, 0.2))


if __name__ == '__main__':
    unittest.main()
