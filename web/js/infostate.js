// One player's info-state from its Showdown client's view: a line-for-line
// port of advsim/live/infostate.py, which is the spec. Parity:
// tests/test_web.py compares this converter's vectors with the Python one.

export const WEATHER = { SunnyDay: 1, RainDance: 2, Sandstorm: 3, none: 0 };
const FORMES = { 'Castform-Sunny': 'Fire', 'Castform-Rainy': 'Water', Castform: 'Normal' };
export const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const PASSED = ['boosts', 'substitute', 'confusion', 'leech_seed', 'perish', 'trapped'];
const RESIDUAL = /\[upkeep\]|\[from\] (?:item: Leftovers|psn|brn|Sandstorm|Leech Seed)|\|-start\|[^|]*\|perish/;
const VOLATILES = { Substitute: 'substitute', confusion: 'confusion', 'move: Leech Seed': 'leech_seed',
  'Leech Seed': 'leech_seed', Encore: 'encore', Attract: 'attract', 'move: Yawn': 'yawn', 'ability: Flash Fire': 'flash_fire' };

const toId = (name) => name.toLowerCase().replace(/[^a-z0-9]/g, '');
// A Pokemon's name from its protocol ident: 'p1a: Mr. Mime' -> 'Mr. Mime'.
export const nameOf = (ident) => ident.split(': ').slice(1).join(': ');

const newMon = (name) => ({ name, species: '', level: 0, hp: 0, maxhp: 0, status: '', toxic_stage: 0, fainted: false,
  ability: '', base_ability: '', item: '', ability_known: false, item_known: false, moves: new Set(),
  transformed: false, req_moves: [], own_moves: [], hp_type: '', stats: {} });
const newSide = () => ({ mons: new Map(), order: [], active: '', boosts: {}, types: [], vol: {}, spikes: 0, wish: 0,
  last_move: '', choice_move: '', pp: null, pending_choice: '', trapper: '', released: false });
const levelOf = (details) => { const d = details.slice(1).find((x) => x.startsWith('L')); return d ? parseInt(d.slice(1), 10) : 100; };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class InfoState {
  constructor(player, speciesTypes) {
    this.me = player;
    this.foe = player === 'p1' ? 'p2' : 'p1';
    this.sides = { p1: newSide(), p2: newSide() };
    this.speciesTypes = speciesTypes;  // species id -> [type, type] names
    this.turn = 0; this.weather = 0; this.weather_turns = 0; this.pos = 0;
    this.request = null; this.ended = false; this.in_residual = false;
    // Version 4's move order: this turn's side -> its own action's move, in the order they went; the
    // turn before's; and whether anything has happened since the last `turn` line.
    this.movers = new Map(); this.last_movers = new Map(); this.mid_turn = false;
  }

  speciesId(name) {
    const base = toId(name.split('-')[0]);
    return this.speciesTypes[base] ? base : toId(name);
  }

  other(side) { return side === this.sides.p1 ? this.sides.p2 : this.sides.p1; }

  // ---- the stream

  feed(log, upto) {
    while (this.pos < upto) {
      const line = log[this.pos++];
      if (line.startsWith('|split|')) {
        const mine = line.split('|')[2] === this.me;
        const priv = log[this.pos], pub = log[this.pos + 1];
        this.pos += 2;
        this.line(mine ? priv : pub);
        continue;
      }
      this.line(line);
    }
  }

  mon(ident) {
    const m = /^(p[12])[a-z]?: (.*)$/.exec((ident || '').trim());
    if (!m) return [null, null];
    const side = this.sides[m[1]];
    if (!side.mons.has(m[2])) {
      side.mons.set(m[2], newMon(m[2]));
      if (m[1] === this.foe) side.order.push(m[2]);
    }
    return [side, side.mons.get(m[2])];
  }

  condition(mon, text) {
    const parts = text.split(/\s+/).filter(Boolean);
    if (parts[0] === '0' || parts.includes('fnt')) { mon.hp = 0; mon.fainted = true; mon.status = ''; return; }
    const [hp, maxhp] = parts[0].split('/');
    mon.hp = parseInt(hp, 10); mon.maxhp = parseInt(maxhp, 10);
    mon.status = parts.length > 1 ? parts[1] : '';
  }

  line(line) {
    const part = line.split('|');
    if (part.length < 2) return;
    const tag = part[1];
    if (!['', 't:', 'turn'].includes(tag)) this.mid_turn = true;
    const of = /\[of\] ([^|]+)/.exec(line);
    if (['move', 'switch', 'drag', 'faint', 'upkeep', 'turn'].includes(tag)) this.settleChoice();
    if (RESIDUAL.test(line)) this.in_residual = true;
    if (tag === 'turn') {
      this.turn = parseInt(part[2], 10);
      this.in_residual = false;
      [this.last_movers, this.movers, this.mid_turn] = [this.movers, new Map(), false];
    } else if (tag === 'switch' || tag === 'drag') {
      this.switchIn(part);
    } else if (tag === 'move') {
      this.move(part, line);
    } else if (tag === '-damage' || tag === '-heal' || tag === '-sethp') {
      const [side, mon] = this.mon(part[2]);
      if (mon) {
        this.condition(mon, part[3]);
        if (line.includes('[from] move: Wish')) side.wish = 0;
        if (tag === '-damage' && line.includes('[from] confusion')) { delete side.vol.charging; delete side.vol.destiny_bond; }
        if (tag === '-damage' && line.includes('[from] psn') && mon.status === 'tox') mon.toxic_stage += 1;
      }
    } else if (tag === 'faint') {
      const [side, mon] = this.mon(part[2]);
      mon.hp = 0; mon.fainted = true; mon.status = '';
      this.revertTrace(mon);
      if (mon.transformed && mon.own_moves.length) mon.req_moves = [...mon.own_moves];
      mon.transformed = false;
      // The slot clears with the faint: boosts, volatiles, the lock, the last move used, the types.
      side.boosts = {}; side.vol = {}; side.choice_move = ''; side.last_move = '';
      side.types = this.speciesTypes[mon.species];
      const other = this.other(side);
      if (other.trapper === mon.name) { delete other.vol.trapped; other.trapper = ''; }
    } else if (tag === 'win' || tag === 'tie') {
      this.ended = true;
      if (this.in_residual) for (const side of Object.values(this.sides)) side.wish = Math.max(side.wish - 1, 0);
    } else if (tag === '-status') {
      const [, mon] = this.mon(part[2]);
      mon.status = part[3]; mon.toxic_stage = 0;
    } else if (tag === '-curestatus') {
      const [, mon] = this.mon(part[2]);
      mon.status = ''; mon.toxic_stage = 0;
    } else if (tag === '-cureteam') {
      const [side] = this.mon(part[2]);
      for (const m of side.mons.values()) { m.status = ''; m.toxic_stage = 0; }
    } else if (tag === '-boost' || tag === '-unboost') {
      const [side] = this.mon(part[2]);
      const sign = tag === '-boost' ? 1 : -1;
      side.boosts[part[3]] = clamp((side.boosts[part[3]] || 0) + sign * parseInt(part[4], 10), -6, 6);
    } else if (tag === '-setboost') {
      const [side] = this.mon(part[2]);
      side.boosts[part[3]] = parseInt(part[4], 10);
    } else if (tag === '-clearallboost') {
      for (const side of Object.values(this.sides)) side.boosts = {};
    } else if (tag === '-clearnegativeboost') {
      const [side] = this.mon(part[2]);
      side.boosts = Object.fromEntries(Object.entries(side.boosts).filter(([, v]) => v > 0));
    } else if (tag === '-weather') {
      this.weather = WEATHER[part[2]] || 0;
      if (line.includes('[upkeep]')) this.weather_turns = Math.max(this.weather_turns - 1, 0);
      else this.weather_turns = (this.weather === 0 || line.includes('[from] ability:')) ? 0 : 5;
    } else if (tag === '-sidestart' && line.includes('Spikes')) {
      this.sides[part[2].slice(0, 2)].spikes += 1;
    } else if (tag === '-sideend' && line.includes('Spikes')) {
      this.sides[part[2].slice(0, 2)].spikes = 0;
    } else if (tag === 'upkeep') {
      for (const side of Object.values(this.sides)) {
        side.wish = Math.max(side.wish - 1, 0);
        if (side.released) { delete side.vol.charging; side.released = false; }
      }
    } else if (['-start', '-end', '-activate', '-singlemove', '-prepare', '-mustrecharge',
      '-transform', '-formechange', 'cant', '-anim'].includes(tag)) {
      this.volatile(tag, part, line);
    }
    this.reveal(tag, part, line, of ? of[1] : null);
  }

  switchIn(part) {
    const [side, mon] = this.mon(part[2]);
    const details = part[3].split(', ');
    mon.species = this.speciesId(details[0]);
    mon.level = levelOf(details);
    this.condition(mon, part[4]);
    const leaving = side.mons.get(side.active);
    if (leaving && leaving !== mon) { this.revertTrace(leaving); leaving.transformed = false; }
    const passing = side.last_move === 'batonpass' && part[1] === 'switch';
    const kept = {};
    if (passing) for (const k of PASSED) if (k in side.vol) kept[k] = side.vol[k];
    if (!passing) side.boosts = {};
    side.vol = kept;
    side.active = nameOf(part[2]);
    side.types = this.speciesTypes[mon.species];
    const other = this.other(side);
    if (passing && leaving && other.trapper === leaving.name) other.trapper = side.active;
    if (other.trapper && other.trapper !== side.active) { delete other.vol.trapped; other.trapper = ''; }
    side.last_move = '';
    side.choice_move = side.pending_choice = '';
    mon.toxic_stage = 0;
  }

  move(part, line) {
    const [side, mon] = this.mon(part[2]);
    const move = toId(part[3]);
    delete side.vol.destiny_bond;
    if (!line.includes('[still]') && 'charging' in side.vol) side.released = true;
    if (!mon.transformed && move !== 'struggle') mon.moves.add(move);
    if (move === 'wish' && !side.wish) side.wish = 2;
    // A side's own action: a chase, a call or a reflection names its source; a charged move's
    // release or a rampage names `lockedmove`, and is still the action.
    if ((!line.includes('[from]') || line.includes('[from] lockedmove')) && !this.movers.has(part[2].slice(0, 2))) {
      this.movers.set(part[2].slice(0, 2), move);
    }
    // A called move (Sleep Talk's) names its caller in [from]; a Pursuit chasing a switch names
    // itself, and is a use of Pursuit like any other.
    if (!line.includes('[from]') || line.includes('[from] Pursuit')) {
      side.last_move = move;
      if (side === this.sides[this.me] && !side.choice_move && move !== 'struggle') side.pending_choice = move;
    }
  }

  volatile(tag, part, line) {
    const [side, mon] = this.mon(part[2]);
    const what = part.length > 3 ? part[3] : '';
    const key = VOLATILES[what];
    if (tag === '-start' && key) side.vol[key] = 1;
    else if (tag === '-end' && key) delete side.vol[key];
    else if (tag === '-start' && what.startsWith('perish')) side.vol.perish = parseInt(what.slice(6), 10);
    else if (tag === '-start' && what === 'typechange') side.types = [part[4], part[4]];
    else if (tag === '-activate' && what.startsWith('move: Wrap')) side.vol.partial_trap = 1;
    else if (tag === '-end' && line.includes('[partiallytrapped]')) delete side.vol.partial_trap;
    else if (tag === '-activate' && what === 'trapped') {
      side.vol.trapped = 1;
      side.trapper = this.sides[side === this.sides[this.me] ? this.foe : this.me].active;
    } else if (tag === '-singlemove' && what === 'Destiny Bond') side.vol.destiny_bond = 1;
    else if (tag === '-prepare') side.vol.charging = 1;
    else if (tag === '-mustrecharge') side.vol.recharge = 1;
    else if (tag === 'cant') {
      delete side.vol.charging; delete side.vol.destiny_bond;
      if (what === 'recharge') delete side.vol.recharge;
    } else if (tag === '-anim') delete side.vol.charging;
    else if (tag === '-transform') {
      const [other] = this.mon(part[3]);
      mon.transformed = true;
      side.vol.transformed = 1;
      side.types = other.types;
      side.boosts = { ...other.boosts };
    } else if (tag === '-formechange') {
      const t = FORMES[what] || 'Normal';
      side.types = [t, t];
    }
  }

  settleChoice() {
    const side = this.sides[this.me];
    const holding = side.mons.has(side.active) && side.mons.get(side.active).item === 'choiceband';
    if (side.pending_choice && holding && !side.choice_move) side.choice_move = side.pending_choice;
    if (!holding) side.choice_move = '';
    side.pending_choice = '';
  }

  revertTrace(mon) {
    if (mon.base_ability) { mon.ability = mon.base_ability; mon.base_ability = ''; }
  }

  reveal(tag, part, line, of) {
    if (tag === '-item' || tag === '-enditem') {
      const [, mon] = this.mon(part[2]);
      mon.item = tag === '-enditem' ? '' : toId(part[3]);
      mon.item_known = true;
    } else if (line.includes('[from] item:')) {
      const [, mon] = this.mon(part[2]);
      if (!mon.item_known) {
        mon.item = toId(/\[from\] item: ([^|]+)/.exec(line)[1]);
        mon.item_known = true;
      }
    }
    const names = [...line.matchAll(/(?:\[from\] )?ability: ([^|]+)/g)].map((m) => m[1]);
    if (tag === '-ability') {
      const [, mon] = this.mon(part[2]);
      if (mon.transformed) return;
      if (line.includes('[from] ability: Trace')) {
        mon.base_ability = 'trace'; mon.ability = toId(part[3]);
        const [, traced] = this.mon(of);
        traced.ability = toId(part[3]); traced.ability_known = true;
      } else {
        mon.ability = toId(part[3]);
      }
      mon.ability_known = true;
    } else if (names.length) {
      const holder = of && tag !== '-heal' ? of : part[2];
      const [, mon] = this.mon(holder);
      if (mon && !mon.transformed) { mon.ability = toId(names[0]); mon.ability_known = true; }
    }
  }

  // ---- the request: the player's own side, exactly

  takeRequest(request) {
    if (!request || !request.side || this.ended) { this.sides[this.me].pp = null; return; }
    this.request = request;
    const side = this.sides[this.me];
    for (const p of request.side.pokemon) {
      const name = nameOf(p.ident);
      if (!side.mons.has(name)) side.mons.set(name, newMon(name));
      const mon = side.mons.get(name);
      if (!side.order.includes(name)) side.order.push(name);
      const details = p.details.split(', ');
      mon.species = this.speciesId(details[0]);
      mon.level = levelOf(details);
      this.condition(mon, p.condition);
      mon.item = p.item;
      mon.stats = p.stats || {};
      if (!mon.base_ability) mon.ability = p.baseAbility;
      mon.req_moves = p.moves.map((m) => (m.startsWith('hiddenpower') ? 'hiddenpower' : m.replace(/\d+$/, '')));
      const hp = p.moves.find((m) => m.startsWith('hiddenpower') && m.length > 11);
      mon.hp_type = hp ? hp.slice(11) : '';
      if (!mon.transformed) mon.own_moves = [...mon.req_moves];
      if (p.active) side.active = name;
    }
    this.settleChoice();
    const active = (request.active || [null])[0];
    side.pp = null;
    if (active && !this.ended && active.moves.length === side.mons.get(side.active).req_moves.length) {
      side.pp = active.moves.map((m) => m.pp || 0);
    }
  }
}
