"""The league trainer's auxiliary target: the foe's next action, which a
reacting PolicyTF predicts. Labels come from the state before the step, as the
observation did."""
from __future__ import annotations

import torch
import torch.nn as nn
import warp as wp

from advsim import net as net_
from advsim.engine import mask as mask_
from advsim.env import Gen3Env


def _arrays(env: Gen3Env, names) -> dict:
    return {k: wp.to_torch(env.arrays[k]).long() for k in names}


def _foe_moves(a: dict, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """The foe active's move slots, its copy's when transformed."""
    act = a['active'][b, f]
    xf = (a['vflags'][b, f] & mask_.VF_TRANSFORMED) != 0
    return torch.where(xf[..., None], a['xf_moves'][b, f], a['moves'][b, f, act])


def foe_action(env: Gen3Env, acts: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """[B, 2] what each side's opponent chose this step, as one class: the
    move's id, or N_MOVES + the species it switched to; -1 when the opponent
    had no choice or passed. Action codes: 0-3 a move slot, 4-9 a party slot."""
    a = _arrays(env, ('active', 'moves', 'vflags', 'xf_moves', 'species'))
    B = acts.shape[0]
    b = torch.arange(B, device=acts.device)[:, None].expand(-1, 2)
    f = torch.tensor([1, 0], device=acts.device)[None].expand(B, -1)
    code = acts.long()[b, f]
    move = _foe_moves(a, b, f).gather(-1, code.clamp(max=3)[..., None]).squeeze(-1)
    species = a['species'][b, f].gather(-1, (code - 4).clamp(0, 5)[..., None]).squeeze(-1)
    label = torch.where(code < 4, move, net_.ids.N_MOVES + species)
    chose = (legal[b, f].sum(-1) > 1) & (code < 10) & ((code >= 4) | (move > 0))
    return torch.where(chose, label, torch.full_like(label, -1))


def foe_action_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the rows with a label; 0 when none has one."""
    total = nn.functional.cross_entropy(logits, label, ignore_index=-1, reduction='sum')
    return total / (label >= 0).sum().clamp(min=1)
