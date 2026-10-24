import itertools
import math
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ranking.exact_policy import expected_ndcg
    from ranking.policy import (action_log_probs, clipped_task, critic_features, make_rollout, objective,
                                permutation_rewards, policy_logits, sample_permutations)


def config(kind='grpo', mode='dense'):
    return {'objective_kind': kind, 'policy_mode': mode, 'samples_per_group': 4,
        'temperature': 1., 'gamma': 1., 'gae_lambda': 1., 'clip_epsilon': .2,
        'kl_coefficient': .02, 'sft_coefficient': .1, 'dpo_beta': .15}


@unittest.skipUnless(torch is not None, "policy tests require PyTorch")
class PolicyTests(unittest.TestCase):
    def test_pl_step_normalization_and_enumeration(self):
        g = torch.tensor([.3, -.2, .8], dtype=torch.float64, requires_grad=True)
        perms = torch.tensor(list(itertools.permutations(range(3))))
        lp = action_log_probs(g, perms)
        self.assertTrue(torch.allclose(lp[:, -1], torch.zeros(6, dtype=lp.dtype)))
        for k, perm in enumerate(perms):
            for t in range(3):
                wanted = g[perm[t]] - torch.logsumexp(g[perm[t:]], 0)
                self.assertAlmostEqual(float(lp[k, t]), float(wanted), places=13)
        probability = lp.sum(-1).exp()
        self.assertAlmostEqual(float(probability.sum()), 1., places=13)
        enumerated = (probability * permutation_rewards(perms, 1)).sum()
        self.assertTrue(torch.allclose(enumerated, expected_ndcg(g, 1), atol=1e-13, rtol=0))
        self.assertTrue(torch.autograd.gradcheck(lambda x: expected_ndcg(x, 1), (g,)))

    def test_sampling_is_pl_and_without_replacement(self):
        g = torch.tensor([.3, -.2, .8], dtype=torch.float64)
        perm = sample_permutations(g, 20000, torch.Generator().manual_seed(6062))
        self.assertTrue(torch.equal(perm.sort(-1).values, torch.arange(3).expand_as(perm)))
        empirical = torch.bincount(perm[:, 0], minlength=3).double() / len(perm)
        self.assertTrue(torch.allclose(empirical, g.softmax(0), atol=.015, rtol=0))

    def test_sum_steps_matches_exact_gradient_at_old_policy(self):
        # Equal logits give equally likely permutations; no trajectory reweighting needed.
        g = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        perm = torch.tensor(list(itertools.permutations(range(3))))
        reward = permutation_rewards(perm, 0)
        old = action_log_probs(g.detach(), perm)
        rollout = {'permutations': perm, 'old_logp': old,
                   'advantages': (reward[:, None] - .5).expand_as(old)}
        loss, _ = clipped_task(g, rollout)
        wanted = -expected_ndcg(g, 0)
        self.assertTrue(torch.allclose(torch.autograd.grad(loss, g)[0],
                                     torch.autograd.grad(wanted, g)[0], atol=1e-13, rtol=0))

    def test_signed_clipping_and_old_detach(self):
        for advantage, ratio, expected in [(1., 1.5, -1.2), (-1., .5, .8),
                                           (1., .5, -.5), (-1., 1.5, 1.5)]:
            g = torch.tensor([0., 0.], dtype=torch.float64, requires_grad=True)
            old = torch.tensor([[math.log(.5 / ratio), 0.]], dtype=torch.float64, requires_grad=True)
            adv = torch.tensor([[advantage, 0.]], dtype=torch.float64, requires_grad=True)
            roll = {'permutations': torch.tensor([[0, 1]]), 'old_logp': old, 'advantages': adv}
            loss, _ = clipped_task(g, roll)
            self.assertAlmostEqual(float(loss), expected, places=13)
            loss.backward()
            self.assertIsNone(old.grad)
            self.assertIsNone(adv.grad)
            if (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < .8):
                self.assertTrue(torch.equal(g.grad, torch.zeros_like(g)))
            else:
                self.assertGreater(float(g.grad.norm()), 0.)

    def test_modes_dpo_initialization_and_prior_cancellation(self):
        delta = torch.tensor([[.2, -.2], [.5, -.1], [-.2, .7]], dtype=torch.float64, requires_grad=True)
        score = delta[:, 0] - delta[:, 1]
        prior1, prior2 = torch.tensor([1., 2., 3.]), torch.tensor([10., -8., .3])
        self.assertTrue(torch.equal(policy_logits(score, prior1, 'dense'), policy_logits(score, prior2, 'dense')))
        cfg = config('candidate_dpo')
        loss, info = objective(delta, score.detach(), prior1, 0, 1, .4, cfg, ['a', 'b', 'c'])
        self.assertAlmostEqual(info['task_loss'], math.log(2), places=13)
        self.assertAlmostEqual(info['kl'], 0., places=13)
        cfg['policy_mode'] = 'fused'
        reference = score.detach() + torch.tensor([.1, -.2, .4])
        _, info1 = objective(delta, reference, prior1, 0, 1, .4, cfg, ['a', 'b', 'c'])
        _, info2 = objective(delta, reference, prior2, 0, 1, .4, cfg, ['a', 'b', 'c'])
        self.assertAlmostEqual(info1['task_loss'], info2['task_loss'], places=13)
        loss.backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_grpo_population_std_and_zero_variance(self):
        g = torch.tensor([.2, -.3, .1], dtype=torch.float64)
        roll = make_rollout(g, 0, config(), torch.Generator().manual_seed(7))
        r = roll['reward']
        wanted = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
        self.assertTrue(torch.equal(roll['advantages'][:, 0], wanted))
        zero = make_rollout(torch.tensor([100., -100.]), 0, config(), torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(zero['advantages'], torch.zeros_like(zero['advantages'])))

    def test_critic_no_label_input_and_actor_gradient_isolation(self):
        g = torch.tensor([.2, -.3, .1], dtype=torch.float64, requires_grad=True)
        critic = torch.nn.Sequential(torch.nn.Linear(7, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        roll = make_rollout(g, 0, config('ppo'), torch.Generator().manual_seed(7), critic)
        self.assertTrue(torch.equal(roll['advantages'], roll['reward'][:, None] - roll['old_values']))
        self.assertTrue(all(not x.requires_grad for x in roll.values()))
        loss, _ = clipped_task(g, roll)
        loss.backward()
        self.assertTrue(all(param.grad is None for param in critic.parameters()))
        before = g.grad.clone()
        pred = critic(roll['critic_features'][:, :-1]).squeeze(-1)
        target = roll['reward'][:, None].expand_as(pred).float()
        torch.nn.functional.mse_loss(pred, target).backward()
        self.assertTrue(torch.equal(before, g.grad))
        perm = roll['permutations']
        self.assertTrue(torch.equal(critic_features(g, perm), roll['critic_features']))

    def test_remaining_mask_and_strict_cuda_sampling_backward(self):
        permutations = torch.tensor([[3, 0, 2, 1], [1, 2, 0, 3]])
        scores = torch.tensor([.3, -.1, .8, -.2], dtype=torch.float64)
        features = critic_features(scores, permutations)
        for k, permutation in enumerate(permutations.tolist()):
            for step in range(4):
                wanted = torch.tensor([float(i in permutation[step:]) for i in range(4)])
                self.assertTrue(torch.equal(features[k, step, 4:8], wanted))
        if not torch.cuda.is_available():
            return
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for mode in ['dense', 'fused']:
                for kind in ['ppo', 'grpo']:
                    delta = torch.stack((torch.linspace(-.5, .8, 10, device='cuda'),
                                         torch.linspace(.4, -.2, 10, device='cuda').sin()), -1).requires_grad_()
                    prior = torch.linspace(.1, .9, 10, device='cuda')
                    score = delta[:, 0] - delta[:, 1]
                    cfg = config(kind, mode)
                    critic = (torch.nn.Sequential(torch.nn.Linear(21, 8), torch.nn.Tanh(),
                              torch.nn.Linear(8, 1)).cuda() if kind == 'ppo' else None)
                    rollout = make_rollout(policy_logits(score, prior, mode), 2, cfg,
                                           torch.Generator(device='cuda').manual_seed(7), critic)
                    cpu_features = critic_features(rollout['old_logits'], rollout['permutations'])
                    self.assertTrue(torch.equal(cpu_features, rollout['critic_features']))
                    loss, _ = objective(delta, score.detach(), prior, 2, 3, .4, cfg,
                                        [str(i) for i in range(10)], rollout)
                    loss.backward()
                    self.assertTrue(torch.isfinite(delta.grad).all())
                    self.assertGreater(float(delta.grad.norm()), 0.)
                    if critic is not None:
                        self.assertTrue(all(p.grad is None for p in critic.parameters()))
                        prediction = critic(rollout['critic_features'].cuda()[:, :-1]).squeeze(-1)
                        target = rollout['reward'].cuda()[:, None].expand_as(prediction).float()
                        torch.nn.functional.mse_loss(prediction, target).backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in critic.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_saved_old_rollout_and_sampling_rng_restore(self):
        gen = torch.Generator().manual_seed(6062)
        g = torch.tensor([.2, -.3, .1], dtype=torch.float64)
        roll = make_rollout(g, 0, config(), gen)
        state = {'old_rollout_window': {'start': 0, 'end': 16, 'actor_step': 0, 'rollouts': [roll]},
                 'sampling_rng': gen.get_state()}
        expected = sample_permutations(g, 4, gen)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            torch.save(state, path)
            loaded = torch.load(path, weights_only=False)
        gen.set_state(loaded['sampling_rng'])
        self.assertTrue(torch.equal(sample_permutations(g, 4, gen), expected))
        for key in roll:
            self.assertTrue(torch.equal(loaded['old_rollout_window']['rollouts'][0][key], roll[key]))


if __name__ == '__main__':
    unittest.main()
