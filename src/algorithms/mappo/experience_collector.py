"""Collect soccer joint actions without hiding resets or multiplying rewards."""

import torch

from algorithms.mappo.buffer import RolloutBuffer
from algorithms.mappo.networks import actions_to_env
from soccer.encoding import encode_team


class ExperienceCollector:
    def __init__(self, env, teams, training_teams=(0,), seed=0):
        self.env, self.teams = env, teams
        self.training_teams = tuple(training_teams)
        if not self.training_teams or len(set(self.training_teams)) != len(self.training_teams) or not set(self.training_teams) <= {0, 1}:
            raise ValueError('Choose distinct training teams from 0 and 1')
        self.observations, _ = env.reset(seed=seed)
        self.random_actions = env.action_space
        self.random_actions.seed(seed + 10000)
        self.episode_returns = {0: 0.0, 1: 0.0}
        self.episodes = 0

    def encode(self, team):
        device = next(self.teams[team].actor.parameters()).device
        ids, obs, state, active = encode_team(self.observations, team, self.env.width,
                                             self.env.height, 3, self.env.max_steps)
        taker = self.observations[ids[0]]['free_kick_taker']
        if taker != -1:
            active &= [player_id == taker for player_id in ids]
        return ids, torch.as_tensor(obs, device=device), torch.as_tensor(state, device=device), torch.as_tensor(active, device=device)

    @torch.no_grad()
    def collect(self, n_steps):
        if n_steps < 1:
            raise ValueError('Rollout length must be positive')
        buffers = {team: RolloutBuffer() for team in self.training_teams}
        completed = []
        for _ in range(n_steps):
            actions = self.random_actions.sample()
            samples = {}
            for team in self.training_teams:
                ids, obs, state, active = self.encode(team)
                pair = self.teams[team]
                distribution = pair.actor(obs, active)
                chosen = distribution.sample()
                samples[team] = (obs, state, chosen, distribution.log_prob(chosen), pair.critic(state), active)
                actions.update(actions_to_env(chosen, ids))
            next_obs, _, terminations, truncations, _ = self.env.step(actions)
            done = all(terminations.values()) or all(truncations.values())
            for team, (obs, state, chosen, log_probs, value, active) in samples.items():
                buffers[team].add(obs, state, chosen, log_probs, value,
                                  self.env.last_team_rewards[team], done, active)
            for team in (0, 1):
                self.episode_returns[team] += self.env.last_team_rewards[team]
            if done:
                self.episodes += 1
                completed.append(dict(episode=self.episodes, score=self.env.team_scores.copy(),
                                      winner=self.env.winner, result=self.env.match_result,
                                      transitions=self.env.step_count, returns=self.episode_returns.copy()))
                self.episode_returns = {0: 0.0, 1: 0.0}
                self.observations, _ = self.env.reset()
            else:
                self.observations = next_obs
        bootstrap = {team: self.teams[team].critic(self.encode(team)[2]) for team in self.training_teams}
        return buffers, bootstrap, completed
