"""One player's info-state, built from what its Showdown client receives: the
protocol stream (public lines, plus the private half of its own side's split
lines) and the request JSON. `observation.py` turns it into obs_layout's
vector, which must equal the engine's `obs` kernel at every decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

WEATHER = {'SunnyDay': 1, 'RainDance': 2, 'Sandstorm': 3, 'none': 0}
FORMES = {'Castform-Sunny': 'Fire', 'Castform-Rainy': 'Water', 'Castform': 'Normal'}
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')
# What Baton Pass carries across, as the engine's keep_slot does.
PASSED = ('boosts', 'substitute', 'confusion', 'leech_seed', 'perish', 'trapped')


# Lines only the residual phase prints: once one shows, it has begun.
RESIDUAL = re.compile(r'\[upkeep\]|\[from\] (?:item: Leftovers|psn|brn|Sandstorm|Leech Seed)|\|-start\|[^|]*\|perish')


def to_id(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


@dataclass
class Mon:
    name: str
    species: str = ''
    level: int = 0
    hp: int = 0
    maxhp: int = 0            # 100 for a foe: its HP comes as a percentage
    status: str = ''
    toxic_stage: int = 0
    fainted: bool = False
    ability: str = ''         # the one in play, a Trace copy included
    base_ability: str = ''    # what a copy reverts to
    item: str = ''
    ability_known: bool = False
    item_known: bool = False
    moves: set = field(default_factory=set)   # a foe's revealed moves
    transformed: bool = False
    req_moves: list = field(default_factory=list)  # own: the request's, in slot order
    own_moves: list = field(default_factory=list)  # own: its set, from a request while not a copy
    hp_type: str = ''                                # own: from the request's move id
    stats: dict = field(default_factory=dict)        # own: atk, def, spa, spd, spe from the request


@dataclass
class Side:
    mons: dict = field(default_factory=dict)  # name -> Mon
    order: list = field(default_factory=list) # own: fixed party order; foe: first seen
    active: str = ''
    boosts: dict = field(default_factory=dict)
    types: tuple = ()
    vol: dict = field(default_factory=dict)   # visible volatiles
    spikes: int = 0
    wish: int = 0
    last_move: str = ''
    choice_move: str = ''
    pp: list | None = None                    # own active, from the request
    pending_choice: str = ''
    trapper: str = ''                         # Mean Look holds while this one is out
    released: bool = False                    # a charge fired; it ends at the residual


class InfoState:
    def __init__(self, player: str, species_types) -> None:
        self.me = player                        # 'p1' or 'p2'
        self.foe = 'p2' if player == 'p1' else 'p1'
        self.sides = {'p1': Side(), 'p2': Side()}
        self.species_types = species_types      # species id -> (type, type) names
        self.turn = 0
        self.weather = 0
        self.weather_turns = 0
        self.pos = 0
        self.request = None
        self.ended = False
        self.in_residual = False
        self.movers: dict[str, str] = {}      # this turn: side -> its own action's move, in the order they went
        self.last_movers: dict[str, str] = {}  # the turn before
        self.mid_turn = False                   # something has happened since the last `turn` line

    def species_id(self, name: str) -> str:
        """The engine keeps base formes (Deoxys-Attack is deoxys, Castform-Rainy
        is castform); a hyphen that is part of the name (Ho-Oh) stays."""
        base = to_id(name.split('-')[0])
        return base if base in self.species_types else to_id(name)

    # ---- the stream

    def feed(self, log: list[str], upto: int) -> None:
        """Consume log lines up to `upto`, keeping this player's half of each split."""
        while self.pos < upto:
            line = log[self.pos]
            self.pos += 1
            if line.startswith('|split|'):
                mine = line.split('|')[2] == self.me
                private, public = log[self.pos], log[self.pos + 1]
                self.pos += 2
                self.line(private if mine else public)
                continue
            self.line(line)

    def mon(self, ident: str) -> tuple[Side | None, Mon | None]:
        m = re.match(r'^(p[12])[a-z]?: (.*)$', ident.strip())
        if not m:
            return None, None
        side = self.sides[m.group(1)]
        if m.group(2) not in side.mons:
            side.mons[m.group(2)] = Mon(m.group(2))
            if m.group(1) == self.foe:
                side.order.append(m.group(2))
        return side, side.mons[m.group(2)]

    def condition(self, mon: Mon, text: str) -> None:
        parts = text.split()
        if parts[0] == '0' or 'fnt' in parts:
            mon.hp, mon.fainted, mon.status = 0, True, ''
            return
        hp, _, maxhp = parts[0].partition('/')
        mon.hp, mon.maxhp = int(hp), int(maxhp)
        mon.status = parts[1] if len(parts) > 1 else ''

    def line(self, line: str) -> None:
        part = line.split('|')
        if len(part) < 2:
            return
        tag = part[1]
        if tag not in ('', 't:', 'turn'):
            self.mid_turn = True
        of = re.search(r'\[of\] ([^|]+)', line)
        if tag in ('move', 'switch', 'drag', 'faint', 'upkeep', 'turn'):
            self.settle_choice()
        if RESIDUAL.search(line):
            self.in_residual = True
        if tag == 'turn':
            self.turn = int(part[2])
            self.in_residual = False
            self.last_movers, self.movers, self.mid_turn = self.movers, {}, False
        elif tag in ('switch', 'drag'):
            self.switch(part)
        elif tag == 'move':
            self.move(part, line)
        elif tag in ('-damage', '-heal', '-sethp'):
            side, mon = self.mon(part[2])
            if mon:
                self.condition(mon, part[3])
                if '[from] move: Wish' in line:
                    side.wish = 0  # the heal is the Wish coming true
                if tag == '-damage' and '[from] confusion' in line:
                    # Hitting itself stops the move like a `cant` does.
                    side.vol.pop('charging', None)
                    side.vol.pop('destiny_bond', None)
                if tag == '-damage' and '[from] psn' in line and mon.status == 'tox':
                    mon.toxic_stage += 1
        elif tag == 'faint':
            side, mon = self.mon(part[2])
            mon.hp, mon.fainted, mon.status = 0, True, ''
            self.revert_trace(mon)
            if mon.transformed and mon.own_moves:
                mon.req_moves = list(mon.own_moves)  # the copy reverts as it faints
            mon.transformed = False
            # The slot clears with the faint: boosts, volatiles, the lock, the
            # last move used, and the types go back to the species'.
            side.boosts, side.vol, side.choice_move, side.last_move = {}, {}, '', ''
            side.types = self.species_types[mon.species]
            other = self.sides['p2' if side is self.sides['p1'] else 'p1']
            if other.trapper == mon.name:
                other.vol.pop('trapped', None)  # the trap goes with its source
                other.trapper = ''
        elif tag in ('win', 'tie'):
            self.ended = True
            if self.in_residual:
                # Wish's slot (order 7) comes before anything that prints in
                # the residual, so a Wish due this turn is spent, heal or not.
                for side in self.sides.values():
                    side.wish = max(side.wish - 1, 0)
        elif tag == '-status':
            _, mon = self.mon(part[2])
            mon.status = part[3]
            mon.toxic_stage = 0
        elif tag == '-curestatus':
            _, mon = self.mon(part[2])
            mon.status, mon.toxic_stage = '', 0
        elif tag == '-cureteam':
            # Heal Bell and Aromatherapy cure the bench without a line apiece.
            side, _ = self.mon(part[2])
            for m in side.mons.values():
                m.status, m.toxic_stage = '', 0
        elif tag in ('-boost', '-unboost'):
            side, _ = self.mon(part[2])
            sign = 1 if tag == '-boost' else -1
            side.boosts[part[3]] = max(-6, min(6, side.boosts.get(part[3], 0) + sign * int(part[4])))
        elif tag == '-setboost':
            side, _ = self.mon(part[2])
            side.boosts[part[3]] = int(part[4])
        elif tag == '-clearallboost':
            for side in self.sides.values():
                side.boosts = {}
        elif tag == '-clearnegativeboost':
            side, _ = self.mon(part[2])
            side.boosts = {k: v for k, v in side.boosts.items() if v > 0}
        elif tag == '-weather':
            self.weather = WEATHER.get(part[2], 0)
            if '[upkeep]' in line:
                self.weather_turns = max(self.weather_turns - 1, 0)
            else:
                self.weather_turns = 0 if (self.weather == 0 or '[from] ability:' in line) else 5
        elif tag == '-sidestart' and 'Spikes' in line:
            self.sides[part[2][:2]].spikes += 1
        elif tag == '-sideend' and 'Spikes' in line:
            self.sides[part[2][:2]].spikes = 0
        elif tag == 'upkeep':
            for side in self.sides.values():
                side.wish = max(side.wish - 1, 0)
                if side.released:
                    side.vol.pop('charging', None)
                    side.released = False
        elif tag in ('-start', '-end', '-activate', '-singlemove', '-prepare', '-mustrecharge',
                     '-transform', '-formechange', 'cant', '-anim'):
            self.volatile(tag, part, line)
        self.reveal(tag, part, line, of.group(1) if of else None)

    def switch(self, part: list[str]) -> None:
        side, mon = self.mon(part[2])
        details = part[3].split(', ')
        mon.species = self.species_id(details[0])
        mon.level = next((int(d[1:]) for d in details[1:] if d.startswith('L')), 100)
        self.condition(mon, part[4])
        leaving = side.mons.get(side.active)
        if leaving and leaving is not mon:
            self.revert_trace(leaving)
            leaving.transformed = False
        passing = side.last_move == 'batonpass' and part[1] == 'switch'
        kept = {k: side.vol[k] for k in PASSED if k in side.vol} if passing else {}
        side.boosts = side.boosts if passing else {}
        side.vol = kept
        side.active = part[2].split(': ', 1)[1]
        side.types = self.species_types[mon.species]
        other = self.sides['p2' if side is self.sides['p1'] else 'p1']
        if passing and leaving and other.trapper == leaving.name:
            other.trapper = side.active  # Baton Pass hands the trap's source on
        if other.trapper and other.trapper != side.active:
            other.vol.pop('trapped', None)  # the trap ends when its source leaves
            other.trapper = ''
        side.last_move = ''
        side.choice_move = side.pending_choice = ''
        mon.toxic_stage = 0

    def move(self, part: list[str], line: str) -> None:
        side, mon = self.mon(part[2])
        move = to_id(part[3])
        side.vol.pop('destiny_bond', None)
        if '[still]' not in line and 'charging' in side.vol:
            # The beam fires; the charge volatile stays until the residual.
            side.released = True
        if not mon.transformed and move != 'struggle':
            mon.moves.add(move)
        if move == 'wish' and not side.wish:
            side.wish = 2  # it lands two residual phases on
        # A side's own action: a chase, a call or a reflection names its source; a charged move's
        # release or a rampage names `lockedmove`, and is still the action.
        if '[from]' not in line or '[from] lockedmove' in line:
            self.movers.setdefault(part[2][:2], move)
        # A called move (Sleep Talk's) names its caller in [from]; a Pursuit
        # chasing a switch names itself, and is a use of Pursuit like any other.
        if '[from]' not in line or '[from] Pursuit' in line:
            side.last_move = move
            # The Choice lock names the first move used, if the band is still
            # held once the move is over (Trick can hand it away); that is
            # settled at the next request.
            if side is self.sides[self.me] and not side.choice_move and move != 'struggle':
                side.pending_choice = move

    def volatile(self, tag: str, part: list[str], line: str) -> None:
        side, mon = self.mon(part[2])
        what = part[3] if len(part) > 3 else ''
        key = {'Substitute': 'substitute', 'confusion': 'confusion', 'move: Leech Seed': 'leech_seed',
               'Leech Seed': 'leech_seed', 'Encore': 'encore', 'Attract': 'attract', 'move: Yawn': 'yawn',
               'ability: Flash Fire': 'flash_fire'}.get(what)
        if tag == '-start' and key:
            side.vol[key] = 1
        elif tag == '-end' and key:
            side.vol.pop(key, None)
        elif tag == '-start' and what.startswith('perish'):
            side.vol['perish'] = int(what[6:])
        elif tag == '-start' and what == 'typechange':
            side.types = (part[4], part[4])
        elif tag == '-activate' and what.startswith('move: Wrap'):
            side.vol['partial_trap'] = 1
        elif tag == '-end' and '[partiallytrapped]' in line:
            side.vol.pop('partial_trap', None)
        elif tag == '-activate' and what == 'trapped':
            side.vol['trapped'] = 1
            side.trapper = self.sides[self.foe if side is self.sides[self.me] else self.me].active
        elif tag == '-singlemove' and what == 'Destiny Bond':
            side.vol['destiny_bond'] = 1
        elif tag == '-prepare':
            side.vol['charging'] = 1
        elif tag == '-mustrecharge':
            side.vol['recharge'] = 1
        elif tag == 'cant':
            # A move stopped before it starts (MoveAborted) drops a Solar Beam
            # charge and a standing Destiny Bond with it.
            side.vol.pop('charging', None)
            side.vol.pop('destiny_bond', None)
            if what == 'recharge':
                side.vol.pop('recharge', None)
        elif tag == '-anim':
            side.vol.pop('charging', None)  # in the sun the beam fires the turn it charges
        elif tag == '-transform':
            other, _ = self.mon(part[3])
            mon.transformed = True
            side.vol['transformed'] = 1
            side.types = other.types
            side.boosts = dict(other.boosts)
        elif tag == '-formechange':
            side.types = (FORMES.get(what, 'Normal'),) * 2

    def settle_choice(self) -> None:
        """The lock takes the move just used if the band is held when that
        move is over: the next action is the first point that is certain."""
        side = self.sides[self.me]
        holding = side.active in side.mons and side.mons[side.active].item == 'choiceband'
        if side.pending_choice and holding and not side.choice_move:
            side.choice_move = side.pending_choice
        if not holding:
            side.choice_move = ''  # the lock is the band's; without it there is none
        side.pending_choice = ''

    def revert_trace(self, mon: Mon) -> None:
        if mon.base_ability:
            mon.ability, mon.base_ability = mon.base_ability, ''

    def reveal(self, tag: str, part: list[str], line: str, of: str | None) -> None:
        """Ability and item knowledge, by the rules export_state.js uses."""
        if tag in ('-item', '-enditem'):
            _, mon = self.mon(part[2])
            mon.item = '' if tag == '-enditem' else to_id(part[3])
            mon.item_known = True
        elif '[from] item:' in line:
            # A tag names what the holder had when it acted: news only when
            # nothing is known yet. A berry's boost comes after its -enditem.
            _, mon = self.mon(part[2])
            if not mon.item_known:
                mon.item = to_id(re.search(r'\[from\] item: ([^|]+)', line).group(1))
                mon.item_known = True
        names = re.findall(r'(?:\[from\] )?ability: ([^|]+)', line)
        if tag == '-ability':
            _, mon = self.mon(part[2])
            if mon.transformed:
                return  # the copy's ability, not its own
            if '[from] ability: Trace' in line:
                mon.base_ability, mon.ability = 'trace', to_id(part[3])
                _, traced = self.mon(of)
                traced.ability, traced.ability_known = to_id(part[3]), True
            else:
                mon.ability = to_id(part[3])
            mon.ability_known = True
        elif names:
            holder = of if of and tag != '-heal' else part[2]
            _, mon = self.mon(holder)
            if mon and not mon.transformed:
                mon.ability, mon.ability_known = to_id(names[0]), True

    # ---- the request: the player's own side, exactly

    def take_request(self, request: dict | None) -> None:
        if not request or 'side' not in request or self.ended:
            self.sides[self.me].pp = None  # a finished battle's request is the last one, stale
            return
        self.request = request
        side = self.sides[self.me]
        for i, p in enumerate(request['side']['pokemon']):
            name = p['ident'].split(': ', 1)[1]
            mon = side.mons.setdefault(name, Mon(name))
            if name not in side.order:
                side.order.append(name)
            details = p['details'].split(', ')
            mon.species = self.species_id(details[0])
            mon.level = next((int(d[1:]) for d in details[1:] if d.startswith('L')), 100)
            self.condition(mon, p['condition'])
            mon.item = p['item']
            mon.stats = p.get('stats', {})
            if not mon.base_ability:
                mon.ability = p['baseAbility']
            # The request spells some moves with their power or type attached:
            # return102, hiddenpowerice.
            mon.req_moves = ['hiddenpower' if m.startswith('hiddenpower') else re.sub(r'\d+$', '', m)
                             for m in p['moves']]
            mon.hp_type = next((m[11:] for m in p['moves'] if m.startswith('hiddenpower') and len(m) > 11), '')
            if not mon.transformed:
                mon.own_moves = list(mon.req_moves)
            if p.get('active'):
                side.active = name
        self.settle_choice()
        active = (request.get('active') or [None])[0]
        # PP shows only when the request lists every move: a lock, a recharge
        # or Struggle lists one, with nothing to read.
        side.pp = None
        if active and not self.ended and len(active['moves']) == len(side.mons[side.active].req_moves):
            side.pp = [m.get('pp', 0) for m in active['moves']]
