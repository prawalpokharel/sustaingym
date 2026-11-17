from __future__ import annotations

from typing import Any
import unittest

import numpy as np
import pandas as pd

from sustaingym.algorithms.data_center import (
    ASAP, CarbonThreshold, ForecastWindow, OracleWindow)
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


# Jobs without parents or tiers, and the schedule each baseline made for them
# at the commit that added DataCenterEnv, before dependencies existed.
PINNED_JOBS = [dict(arrival=a, duration=d, num_gpus=g, deadline=dl) for a, d, g, dl in [
    (0, 24, 4, 200), (0, 12, 2, 60), (6, 36, 4, 300), (10, 6, 1, 40),
    (20, 48, 8, 400), (30, 12, 2, 120), (40, 6, 2, 80), (50, 24, 4, 250)]]
PINNED_STARTS = {
    (ASAP, False): [0, 0, 16, 10, 74, 30, 40, 50],
    (CarbonThreshold, False): [83, 48, 107, 34, 144, 83, 74, 95],
    (ForecastWindow, True): [82, 47, 106, 34, 142, 106, 74, 82],
    (OracleWindow, False): [129, 36, 183, 34, 81, 129, 41, 191],
    (OracleWindow, True): [86, 36, 97, 34, 352, 85, 41, 110],
}


class TestUnchangedWithoutDependencies(unittest.TestCase):
    def test_baselines_match_pinned_schedules(self) -> None:
        env = small_env(job_generator=FixedJobs(PINNED_JOBS), num_gpus=8)
        for (policy, safe), starts in PINNED_STARTS.items():
            with self.subTest(policy=policy.__name__, safe=safe):
                policy(env, deadline_safe=safe).run_episode(seed=0, day='2020-06-03')
                self.assertEqual(env.schedule(), list(enumerate(starts)))

    def test_zero_probabilities_match_plain_env(self) -> None:
        plain = small_env()
        wrapped = small_env(job_generator=DependencyJobGenerator(
            SyntheticJobGenerator(), dep_prob=0.0, critical_frac=0.0))
        for policy in (ASAP, ForecastWindow):
            for safe in (False, True):
                a = policy(plain, deadline_safe=safe).run_episode(seed=4)
                b = policy(wrapped, deadline_safe=safe).run_episode(seed=4)
                self.assertEqual(a, b)
                self.assertEqual(plain.schedule(), wrapped.schedule())


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
        for child in jobs:
            self.assertLessEqual(len(child.parents), 2)
            for p in child.parents:
                parent = jobs[p]
                self.assertLess(p, child.job_id)
                self.assertLessEqual(child.arrival - parent.arrival, gen.window_steps)
                self.assertLessEqual(parent.arrival + parent.duration, child.latest_start)

    def test_rejects_bad_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            DependencyJobGenerator(SyntheticJobGenerator(), dep_prob=1.5)


class TestDependenciesInEnv(unittest.TestCase):
    def test_child_waits_for_parent(self) -> None:
        jobs = [dict(arrival=0, duration=12, num_gpus=2, deadline=60),
                dict(arrival=0, duration=6, num_gpus=2, deadline=60, parents=(0,))]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        summary = ASAP(env).run_episode(seed=0)
        parent, child = env.jobs
        self.assertEqual(parent.start, 0)
        self.assertEqual(child.start, parent.finish)
        self.assertEqual(summary['misses'], 0)
        self.assertEqual(summary['finished'], 2)

    def test_child_waits_for_both_parents(self) -> None:
        jobs = [dict(arrival=0, duration=6, num_gpus=2, deadline=60),
                dict(arrival=0, duration=18, num_gpus=2, deadline=60),
                dict(arrival=0, duration=6, num_gpus=2, deadline=60, parents=(0, 1))]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        ASAP(env).run_episode(seed=0)
        self.assertEqual(env.jobs[2].start, env.jobs[1].finish)
        self.assertGreater(env.jobs[2].start, env.jobs[0].finish)

    def test_blocked_job_is_not_in_the_window(self) -> None:
        jobs = [dict(arrival=0, duration=12, num_gpus=2, deadline=60),
                dict(arrival=0, duration=6, num_gpus=2, deadline=60, parents=(0,))]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=8)
        env.reset(seed=0)
        self.assertEqual([j.job_id for j in env.queue_window_jobs()], [0])
        self.assertEqual([j.job_id for j in env.blocked], [1])

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


if __name__ == '__main__':
    unittest.main()
