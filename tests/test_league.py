"""The league trainer: returns follow the side that acted, the learner trains
only on its own sides, and a few iterations run end to end."""
import pathlib
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import train_league as tl  # noqa: E402


def test_a_p2_win_is_good_for_p2_and_bad_for_p1():
    T = 3
    rew = torch.zeros(T, 1, 2)
    rew[2, 0] = torch.tensor([-1.0, 1.0])  # p2 wins on the last step
    done = torch.tensor([[False], [False], [True]])
    trunc = torch.zeros(T, 1, dtype=torch.bool)
    adv, ret = tl.gae(rew, torch.zeros(T, 1, 2), done, trunc, torch.zeros(1, 2), 0.99, 0.95)
    assert (adv[:, 0, 1] > 0).all() and (adv[:, 0, 0] < 0).all()
    # A game end cuts the stream: nothing flows back across it.
    rew2 = torch.zeros(T, 1, 2)
    rew2[2, 0] = torch.tensor([1.0, -1.0])
    done2 = torch.tensor([[True], [False], [True]])
    adv2, _ = tl.gae(rew2, torch.zeros(T, 1, 2), done2, trunc, torch.zeros(1, 2), 0.99, 0.95)
    assert adv2[0].abs().sum() == 0


def test_the_learner_plays_both_sides_only_against_itself():
    opp = torch.tensor([tl.SELF, 0, tl.BASE + 2])
    side = torch.tensor([1, 1, 0])
    assert tl.learner_mask(opp, side).tolist() == [[True, True], [False, True], [True, False]]


def test_a_few_iterations_run(tmp_path):
    out = tmp_path / 'league.npz'
    tl.main(['--device', 'cpu', '--no-compile', '--arch', 'mlp', '--batch', '16', '--steps', '8', '--iters', '3', '--snapshot-every', '1',
             '--minibatch', '64', '--save', str(out), '--log-every', '1', '--eval-every', '0'])
    assert out.exists()


def test_v2_with_evaluation_runs(tmp_path):
    out = tmp_path / 'v2.npz'
    tl.main(['--device', 'cpu', '--no-compile', '--arch', 'v2', '--hidden', '64', '--batch', '16', '--steps', '8', '--iters', '2',
             '--minibatch', '64', '--eval-every', '1', '--eval-games', '8', '--eval-ref', 'nonexistent',
             '--save', str(out)])
    curve = (tmp_path / 'v2_curve.jsonl').read_text().splitlines()
    assert len(curve) == 2 and '"maxdamage"' in curve[0]
    assert net_load_type(out) == 'PolicyV2'
    # A warm start keeps the checkpoint's architecture: the default --arch tf
    # and --opp-action do not apply to it.
    tl.main(['--device', 'cpu', '--no-compile', '--init', str(out), '--batch', '16', '--steps', '8', '--iters', '1',
             '--minibatch', '64', '--eval-every', '0'])


def net_load_type(path):
    from advsim import net as net_
    return type(net_.load(path, 'cpu')).__name__


def test_foe_action_labels_name_the_move_or_the_switch_target():
    from advsim import net as net_
    from conftest import make_env
    env = make_env(4, seed=5)
    _, mask = env.observe()
    legal = net_.legal_bits(mask)
    acts = torch.tensor([[1, 5]] * 4, dtype=torch.int32)  # p1 uses its second move, p2 switches to slot 1
    label = tl.foe_action(env, acts, legal).numpy()
    w = env.words()
    for b in range(4):
        assert label[b, 1] == w['moves'][b, 0, w['active'][b, 0], 1]  # p2 predicts p1's move
        assert label[b, 0] == net_.ids.N_MOVES + w['species'][b, 1, 1]  # p1 predicts p2's switch
    passed = tl.foe_action(env, torch.tensor([[10, 10]] * 4, dtype=torch.int32), legal)
    assert (passed == -1).all()


def test_record_and_groups_match_the_per_opponent_loops():
    gen = torch.Generator().manual_seed(0)
    league = tl.League(8, [0.5, 0.25, 0.25], 3, 'cpu', gen)
    league.pool = [{}, {}]
    league.opp = torch.tensor([tl.SELF, 0, 1, 1, tl.BASE + 2, tl.BASE + 2, tl.BASE + 4, 0])
    finished = torch.tensor([True, True, True, True, True, False, True, False])
    reward = torch.tensor([1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0, -1.0])
    base, snap = league.base_wr.clone(), league.snap_wr.clone()
    league.record(finished, reward)
    score = (reward + 1) / 2
    for wr, code, i in ((base, tl.BASE + 2, 2), (base, tl.BASE + 4, 4), (snap, 0, 0), (snap, 1, 1)):
        sel = finished & (league.opp == code)
        wr[i] += 0.05 * (score[sel].mean() - wr[i])
    assert torch.allclose(league.base_wr, base) and torch.allclose(league.snap_wr, snap)
    groups = {k: s.tolist() for k, s in league.groups()}
    assert groups == {0: [1, 7], 1: [2, 3], 3: [4, 5, 6]}  # snapshots 0 and 1, then the baselines as group 3
    league.draw(torch.tensor([False] * 7 + [True]))
    assert league.opp[:7].tolist() == [tl.SELF, 0, 1, 1, tl.BASE + 2, tl.BASE + 2, tl.BASE + 4]


def test_the_transformer_trains_and_loads(tmp_path):
    from advsim import net as net_
    out = tmp_path / 'tf.npz'
    tl.main(['--device', 'cpu', '--no-compile', '--arch', 'tf', '--token-dim', '32', '--tf-layers', '2', '--tf-heads', '2',
             '--opp-action', '0.1', '--batch', '16', '--steps', '8', '--iters', '2', '--minibatch', '64',
             '--eval-every', '1', '--eval-games', '8', '--eval-ref', 'nonexistent', '--save', str(out)])
    assert net_.load(out, 'cpu').react


def test_micro_batches_take_the_same_step(tmp_path):
    from advsim import fileio
    import numpy as np
    common = ['--device', 'cpu', '--no-compile', '--arch', 'v2', '--hidden', '64', '--batch', '16', '--steps', '8', '--iters', '1',
              '--minibatch', '64', '--epochs', '1', '--eval-every', '0', '--no-amp']
    tl.main(common + ['--save', str(tmp_path / 'whole.npz')])
    tl.main(common + ['--micro', '16', '--save', str(tmp_path / 'micro.npz')])
    whole, micro = fileio.read_npz(tmp_path / 'whole.npz'), fileio.read_npz(tmp_path / 'micro.npz')
    moved = [k for k in whole if k != 'arch' and not np.allclose(whole[k], micro[k], atol=1e-6)]
    assert not moved, moved


def test_damage_inputs_train(tmp_path):
    from advsim import net as net_
    out = tmp_path / 'tfd.npz'
    tl.main(['--device', 'cpu', '--no-compile', '--arch', 'tf', '--damage', '--token-dim', '32', '--tf-layers', '2', '--tf-heads', '2',
             '--tf-compact', '--batch', '16', '--steps', '8', '--iters', '2', '--minibatch', '64',
             '--eval-every', '0', '--save', str(out)])
    assert net_.load(out, 'cpu').damage


def test_the_defaults_are_the_recipe(tmp_path):
    """A bare run trains the network that works: compact PolicyTF, width 96, 3
    layers, reacting, with damage inputs."""
    from advsim import net as net_
    out = tmp_path / 'default.npz'
    tl.main(['--device', 'cpu', '--no-compile', '--batch', '16', '--steps', '8', '--iters', '1', '--minibatch', '64',
             '--eval-every', '0', '--save', str(out)])
    assert net_.load(out, 'cpu').config == (96, 3, 4, 1, 1, 1)
