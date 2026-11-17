"""Dependency-criticality policy for DataCenterEnv.

The policy holds a job for a lower-carbon start only as long as the job's
criticality allows. Criticality is the Centrality-Entropy Index (CEI) from
P. Pokharel, "Federated Deployment and Empirical Evaluation of the
Centrality-Entropy Index Framework for Enterprise Cloud Governance in
Multi-Tier Distributed Environments", FTC 2026: a weighted sum of the job's
centrality in the pipeline graph, the entropy of the carbon signal over the
steps where it could start, and its declared tier.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sustaingym.envs.data_center import DataCenterEnv, Job
from sustaingym.envs.data_center.env import FORECAST_STEPS
from .baselines import DataCenterPolicy, best_offset


def job_centrality(jobs: list[Job]) -> np.ndarray:
    """Composite centrality of each job in the episode's dependency graph.

    The graph has an edge from each job to each of its parents ("depends
    on"), so jobs that many others wait for rank high. The score is
    0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness,
    min-max scaled over the episode's jobs. Every job scores 0 when no job
    has parents.

    Args:
        jobs: the episode's jobs, with ``job_id`` equal to list position

    Returns:
        array of centralities in [0, 1], indexed by job id
    """
    if not any(j.parents for j in jobs):
        return np.zeros(len(jobs))
    import networkx as nx  # only needed for jobs with dependencies

    g = nx.DiGraph()
    g.add_nodes_from(range(len(jobs)))
    g.add_edges_from((j.job_id, p) for j in jobs for p in j.parents)
    pr = nx.pagerank(g, alpha=0.85)
    bw = nx.betweenness_centrality(g)
    dg = nx.degree_centrality(g)
    cl = nx.closeness_centrality(g)
    comp = np.array([0.35 * pr[n] + 0.30 * bw[n] + 0.20 * dg[n] + 0.15 * cl[n]
                     for n in range(len(jobs))])
    lo, hi = comp.min(), comp.max()
    if hi - lo < 1e-12:
        return np.zeros(len(jobs))
    return np.round((comp - lo) / (hi - lo), 9)


def signal_entropy(series: np.ndarray, bins: int = 10, top: float = 1.0) -> float:
    """Shannon entropy of ``series`` over ``bins`` equal bins on [0, top],
    divided by log(bins) so it is in [0, 1]. Values are clipped to the range.
    """
    if len(series) < 2:
        return 0.0
    hist, _ = np.histogram(np.clip(series, 0.0, top), bins=bins, range=(0.0, top))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))


class DependencyCriticality(DataCenterPolicy):
    """Holds each job at most ``(1 - score) * slack`` steps, then starts it at
    the lowest-carbon offset in the forecast within that limit, as
    `ForecastWindow` does.

    ``score = alpha * C + beta * H + gamma * R`` (the CEI), where C is the
    job's centrality (`job_centrality`), H is the entropy of the current and
    forecast MOER over the steps where the job could start
    (`signal_entropy`), and R is the job's tier, clipped to [0, 1]. Slack is
    the gap to the job's latest start, or to its planned start with
    ``deadline_safe``. A score of 1 means the job starts as soon as it can.

    Args:
        env: environment
        weights: ``(alpha, beta, gamma)``. The default ``(0.40, 0.35, 0.25)``
            is the CEI reference setting. Weights must be non-negative and
            sum to at most 1.
        **kwargs: ``deadline_safe`` and ``reserve_frac``, see `DataCenterPolicy`

    Attributes:
        weights: tuple[float, float, float], ``(alpha, beta, gamma)``
    """
    name = 'criticality'

    def __init__(self, env: DataCenterEnv,
                 weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
                 **kwargs: Any):
        if min(weights) < 0 or sum(weights) > 1 + 1e-9:
            raise ValueError('weights must be non-negative and sum to at most 1')
        super().__init__(env, **kwargs)
        self.weights = weights
        self._jobs_ref: list[Job] | None = None
        self._centrality = np.zeros(0)

    def centrality(self, job: Job) -> float:
        """The job's centrality, computed once per episode."""
        if self._jobs_ref is not self.env.jobs:
            self._jobs_ref = self.env.jobs
            self._centrality = job_centrality(self.env.jobs)
        return float(self._centrality[job.job_id])

    def score(self, job: Job, latest_start: int) -> float:
        """The job's criticality index at the current step, in [0, 1]."""
        alpha, beta, gamma = self.weights
        env = self.env
        c = self.centrality(job) if alpha > 0 else 0.0
        h = 0.0
        if beta > 0:
            n = min(max(latest_start - env.t, 0), FORECAST_STEPS) + 1
            series = np.concatenate([[env.moer_now()], env.moer_forecast()])[:n]
            h = signal_entropy(series)
        r = min(max(float(job.tier), 0.0), 1.0)
        # rounded to 9 places so tiny float differences between machines
        # rarely change the hold limit
        return round(alpha * c + beta * h + gamma * r, 9)

    def start_now(self, job: Job, latest_start: int) -> bool:
        slack = latest_start - self.env.t
        if slack <= 0:
            return True
        hold = int(np.floor((1.0 - self.score(job, latest_start)) * slack))
        if hold <= 0:
            return True
        series = np.concatenate([[self.env.moer_now()], self.env.moer_forecast()])
        return best_offset(series, job.duration, min(hold, FORECAST_STEPS)) == 0
