// obs_layout's vector from an InfoState: a port of advsim/live/observation.py.
// `vocab` is web/data/vocab.json (tools/export_web.py).
import { BOOSTS } from './infostate.js';

const IMMUNE = -8, PASS = 10, FORCED = 11;

export class Vocab {
  constructor(v) {
    this.v = v;
    this.L = v.layout;
    this.index = {};
    for (const t of ['species', 'moves', 'abilities', 'items', 'conditions', 'types']) {
      this.index[t] = new Map(v[t].map((n, i) => [n, i]));
    }
    this.speciesTypes = {};
    v.species.forEach((n, i) => { if (n) this.speciesTypes[n] = [v.types[v.species_type1[i]], v.types[v.species_type2[i]]]; });
  }

  id(table, name) { return name ? (this.index[table].get(name) || 0) : 0; }
  typeId(name) { return this.index.types.get(name) || 0; }
  effectiveness(t, d1, d2) {
    const c = this.v.type_chart;
    const m = c[t][d1] + (d2 !== d1 ? c[t][d2] : 0);
    return m < -6 ? IMMUNE : Math.max(-6, Math.min(6, m));
  }
}

export function hpPercent(hp, maxhp) {
  if (hp <= 0 || maxhp <= 0) return 0;
  const pct = Math.floor((100 * hp + maxhp - 1) / maxhp);
  return pct === 100 && hp < maxhp ? 99 : pct;
}

const sortedMoveIds = (mon, v) => [...mon.moves].map((m) => v.id('moves', m)).sort((a, b) => a - b).slice(0, 4);

export function observe(st, v) {
  const L = v.L;
  const out = new Int16Array(L.OBS_DIM);
  const me = st.sides[st.me], foe = st.sides[st.foe];
  const M = L.MON.length;

  const putMon = (token, mon, own, active, pp) => {
    const row = { present: 1, species: v.id('species', mon.species), level: mon.level,
      hp_pct: own ? hpPercent(mon.hp, mon.maxhp) : mon.hp,
      status: mon.fainted ? 0 : v.id('conditions', mon.status),
      toxic_stage: mon.status === 'tox' ? mon.toxic_stage : 0,
      fainted: +mon.fainted, active: +active };
    if (own) {
      Object.assign(row, { hp: mon.hp, maxhp: mon.maxhp, ability: v.id('abilities', mon.ability), ability_known: 1,
        item: v.id('items', mon.item), item_known: 1 });
      mon.req_moves.slice(0, 4).forEach((m, i) => {
        row[`move${i}`] = v.id('moves', m);
        if (active && pp) row[`pp${i}`] = pp[i];
      });
    } else {
      if (mon.ability_known) Object.assign(row, { ability: v.id('abilities', mon.ability), ability_known: 1 });
      if (mon.item_known) Object.assign(row, { item: v.id('items', mon.item), item_known: 1 });
      sortedMoveIds(mon, v).forEach((m, i) => { row[`move${i}`] = m; });
    }
    for (const [name, value] of Object.entries(row)) out[token * M + L.MON.indexOf(name)] = value;
  };

  me.order.forEach((name, i) => putMon(i, me.mons.get(name), true, name === me.active, me.pp));
  const seen = [...foe.mons.entries()].filter(([, m]) => m.species)
    .map(([n, m]) => [v.id('species', m.species), n])
    .sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
  seen.forEach(([, name], rank) => putMon(6 + rank, foe.mons.get(name), false, name === foe.active, null));

  [me, foe].forEach((side, token) => {
    const base = L.ACTIVE_BASE + token * L.ACTIVE.length;
    const row = {};
    for (const k of BOOSTS) row[`boost_${k === 'accuracy' || k === 'evasion' ? k.slice(0, 3) : k}`] = side.boosts[k] || 0;
    const [t0, t1] = side.types && side.types.length ? side.types : ['???', '???'];
    Object.assign(row, { type0: v.typeId(t0), type1: v.typeId(t1) }, side.vol);
    if (token === 0 && side.choice_move) row.choice_move = v.id('moves', side.choice_move);
    for (const [name, value] of Object.entries(row)) out[base + L.ACTIVE.indexOf(name)] = value;
  });

  const field = { weather: st.weather, weather_turns: st.weather_turns, turn: Math.min(st.turn, 1000),
    spikes_own: me.spikes, spikes_foe: foe.spikes, wish_own: +(me.wish > 0), wish_foe: +(foe.wish > 0) };
  for (const [name, value] of Object.entries(field)) out[L.FIELD_BASE + L.FIELD.indexOf(name)] = value;

  [[me, foe], [foe, me]].forEach(([side, other], token) => {
    const mon = side.mons.get(side.active);
    if (!mon || !other.types || !other.types.length) return;
    const moves = token === 0 ? mon.req_moves.slice(0, 4).map((m) => [v.id('moves', m), mon.hp_type])
      : sortedMoveIds(mon, v).map((m) => [m, null]);
    const d1 = v.typeId(other.types[0]), d2 = v.typeId(other.types[1]);
    const own = new Set(side.types.map((t) => v.typeId(t)));
    moves.forEach(([m, hpType], i) => {
      const fromMon = v.v.move_type_from_mon[m];
      if (m === 0 || v.v.move_category[m] === 2 || (fromMon && hpType === null)) return;
      const t = fromMon ? v.typeId(hpType.charAt(0).toUpperCase() + hpType.slice(1)) : v.v.move_type[m];
      out[L.MATCHUP_BASE + token * 8 + 2 * i] = v.effectiveness(t, d1, d2);
      out[L.MATCHUP_BASE + token * 8 + 2 * i + 1] = +own.has(t);
    });
  });

  const mask = legalMask(st);
  for (let a = 0; a < 12; a++) out[L.MASK_BASE + a] = (mask >> a) & 1;
  // Version 2: the move each active last used; version 3: each own Pokemon's stats, party order.
  out[L.HISTORY_BASE] = v.id('moves', me.last_move);
  out[L.HISTORY_BASE + 1] = v.id('moves', foe.last_move);
  me.order.forEach((name, i) => L.STATS.forEach((stat, k) => { out[L.STATS_BASE + 5 * i + k] = me.mons.get(name).stats[stat] || 0; }));
  // Version 4: who moved first, in the turn in progress once anything has happened in it (the last
  // one until then), and only when both used a move of the same priority: the order tells Speed.
  const movers = st.mid_turn ? st.movers : st.last_movers;
  if (movers.size === 2 && new Set([...movers.values()].map((m) => v.v.move_priority[v.id('moves', m)])).size === 1) {
    out[L.ORDER_BASE] = [...movers.keys()][0] === st.me ? 1 : -1;
  }
  return out;
}

export function legalMask(st) {
  const req = st.request;
  if (!req || req.wait || st.ended) return 1 << PASS;
  const me = st.sides[st.me];
  let switches = 0;
  me.order.forEach((name, i) => { if (!me.mons.get(name).fainted && name !== me.active) switches |= 1 << (4 + i); });
  if (req.forceSwitch) return switches || 1 << PASS;
  const active = req.active[0];
  const trapped = active.trapped || false;
  const moves = active.moves;
  // One move offered is a forced action (a lock or Struggle) unless it is a one-move set's own move.
  const lone = moves.length === 1 && (moves[0].id === 'struggle' || me.mons.get(me.active).req_moves.length > 1);
  if (lone && !moves[0].disabled) {
    return (1 << FORCED) | (trapped ? 0 : switches);
  }
  let bits = 0;
  moves.forEach((m, i) => { if (!m.disabled && (m.pp === undefined ? 1 : m.pp) > 0) bits |= 1 << i; });
  return (bits || 1 << FORCED) | (trapped ? 0 : switches);
}
