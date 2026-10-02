// advsim/damage.py's network inputs for one observation, in plain JS: for each move that matters
// this turn, the least and most one use would take off its target as a fraction of its max HP.
// The own active's four moves against the foe active, the foe active's revealed moves against the
// own active, and each own Pokemon's most dealt to the foe active and most taken from its revealed
// moves. Everything comes from the observation, so nothing hidden leaks in; the foe's stats are
// estimated from species and level as Random Battle builds them. `D`: tools/export_web.py's
// `damage` (damage.tables plus the ids the formula names). A port, held to the Python by
// tests/test_web.py through the networks that read it.

const stage = (s) => (s >= 0 ? (2 + s) / 2 : 2 / (2 - s));

// Per Pokemon token: what the formula needs about it.
function tokens(obs, L, D) {
  const M = L.MON.length, A = L.ACTIVE.length, mc = (n) => L.MON.indexOf(n), ac = (n) => L.ACTIVE.indexOf(n);
  const mon = (t, c) => obs[L.MON_BASE + t * M + c];
  const act = (side, n) => obs[L.ACTIVE_BASE + side * A + ac(n)];
  return [...Array(12).keys()].map((t) => {
    const own = t < 6, side = own ? 0 : 1, species = mon(t, mc('species')), level = mon(t, mc('level'));
    const active = mon(t, mc('active')) * mon(t, mc('present')) !== 0;
    const base = D.base[species];
    const est = base.map((b, k) => Math.floor((2 * b + 52) * level / 100) + (k === 0 ? level + 10 : 5));
    const stats = own ? [est[0], ...[0, 1, 2, 3, 4].map((k) => obs[L.STATS_BASE + 5 * t + k])] : est;
    const maxhp = own ? mon(t, mc('maxhp')) : est[0];
    const hpfrac = own ? mon(t, mc('hp')) / Math.max(maxhp, 1) : mon(t, mc('hp_pct')) / 100;
    const types = active ? [act(side, 'type0'), act(side, 'type1')] : [D.type1[species], D.type2[species]];
    const boosted = ['boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe'];
    const boosts = [0, ...boosted.map((b) => (active ? act(side, b) : 0))];
    const known = (w) => (mon(t, mc(`${w}_known`)) > 0 ? mon(t, mc(w)) : 0);
    return { level, stats: stats.map((v, k) => v * stage(boosts[k])), maxhp, hpfrac, types, status: mon(t, mc('status')),
      ability: known('ability'), item: known('item'), flashFire: active && act(side, 'flash_fire') > 0,
      moves: [0, 1, 2, 3].map((j) => mon(t, mc('move0') + j)), active };
  });
}

// [least, most] fraction of the defender's max HP one use of `move` takes, attacker `a`, defender `d`.
function hit(a, d, move, weather, D) {
  const I = D.ids;
  const hp = move === I.MOVE_HIDDENPOWER;
  const t = hp ? I.TYPE_NORMAL : D.mtype[move];  // its type is the holder's secret: neutral, no STAB
  const special = D.special[t], physical = !special;
  const atk = special ? a.stats[3] : a.stats[1];
  let dfs = special ? d.stats[4] : d.stats[2];
  const ab = a.ability, statused = D.statused.includes(a.status);
  let boost = 1;
  if (physical && (ab === I.ABILITY_HUGEPOWER || ab === I.ABILITY_PUREPOWER)) boost *= 2;
  if (physical && ab === I.ABILITY_HUSTLE) boost *= 1.5;
  if (physical && ab === I.ABILITY_GUTS && statused) boost *= 1.5;
  if (physical && a.item === I.ITEM_CHOICEBAND) boost *= 1.5;
  if (d.ability === I.ABILITY_THICKFAT && (t === I.TYPE_FIRE || t === I.TYPE_ICE)) boost *= 0.5;
  if (D.selfdestruct[move]) dfs /= 2;
  let power = D.power[move];
  if (move === I.MOVE_RETURN) power = 102;
  if (hp) power = 70;
  if (move === I.MOVE_FLAIL || move === I.MOVE_REVERSAL) {
    const x = Math.floor(48 * a.hpfrac);
    power = D.flail_power[D.flail_at.filter((b) => b <= x).length];
  }
  if (move === I.MOVE_FACADE && statused) power = 140;
  // Gen 3's order: the base, burn, weather, Flash Fire, +2; then the roll, STAB and type.
  let base = Math.floor(Math.floor(Math.floor(2 * a.level / 5 + 2) * power * atk * boost / Math.max(dfs, 1)) / 50);
  if (physical && a.status === I.COND_BRN && ab !== I.ABILITY_GUTS) base = Math.floor(base / 2);
  let wx = 1;
  if (weather === I.WEATHER_RAIN) wx = t === I.TYPE_WATER ? 1.5 : t === I.TYPE_FIRE ? 0.5 : 1;
  if (weather === I.WEATHER_SUN) wx = t === I.TYPE_FIRE ? 1.5 : t === I.TYPE_WATER ? 0.5 : 1;
  base = Math.floor(base * wx);
  if (a.flashFire && t === I.TYPE_FIRE) base = Math.floor(base * 1.5);
  base += 2;
  const e1 = D.chart[t][d.types[0]], e2 = d.types[1] !== d.types[0] ? D.chart[t][d.types[1]] : 0;
  const da = d.ability;
  let immune = e1 < -1 || e2 < -1 || (da === I.ABILITY_LEVITATE && t === I.TYPE_GROUND)
    || (da === I.ABILITY_FLASHFIRE && t === I.TYPE_FIRE) || (da === I.ABILITY_WATERABSORB && t === I.TYPE_WATER)
    || (da === I.ABILITY_VOLTABSORB && t === I.TYPE_ELECTRIC);
  const eff = hp ? 1 : 2 ** (e1 + e2);
  immune = immune || (da === I.ABILITY_WONDERGUARD && eff <= 1 && !hp);
  const stab = !hp && (t === a.types[0] || t === a.types[1]) ? 1.5 : 1;
  let hi = Math.floor(Math.floor(base * stab) * eff);
  let lo = Math.floor(Math.floor(Math.floor(base * 85 / 100) * stab) * eff);
  if (D.least[move] > 0) lo *= D.least[move];
  if (D.most[move] > 0) hi *= D.most[move];
  const fixed = D.fixed[move];
  if (fixed === 255) { lo = a.level; hi = a.level; } else if (fixed > 0) { lo = fixed; hi = fixed; }  // Seismic Toss, Night Shade, ...
  const maxhp = Math.max(d.maxhp, 1);
  lo /= maxhp; hi /= maxhp;
  if (D.ohko[move]) { lo = 1; hi = 1; }
  if (move === 0 || D.cat[move] === I.CATEGORY_STATUS || immune) return [0, 0];
  const clamp = (x) => Math.min(Math.max(x, 0), 3);
  return [clamp(lo), clamp(hi)];
}

// { ownMoves: [4][2], foeMoves: [4][2], party: [6][2] }, as damage.features.
export function damageFeatures(obs, L, D) {
  const tok = tokens(obs, L, D);
  const weather = obs[L.FIELD_BASE + L.FIELD.indexOf('weather')];
  const first = (lo) => { for (let t = lo; t < lo + 6; t++) if (tok[t].active) return t; return lo; };  // argmax: the first, or 0
  const own = first(0), foe = first(6);
  const ranges = (att, dfn) => tok[att].moves.map((m) => hit(tok[att], tok[dfn], m, weather, D));
  const most = (att, dfn) => Math.max(...ranges(att, dfn).map((r) => r[1]));
  return { ownMoves: ranges(own, foe), foeMoves: ranges(foe, own),
    party: [0, 1, 2, 3, 4, 5].map((i) => [most(i, foe), most(foe, i)]) };
}
