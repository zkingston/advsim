// The Emerald cartridge AI's choice of replacement, as a policy for the sweep.
//
// Source: the write-up at docs.google.com/document/d/13E61Jj4KwhIy3ZKgLjPY-_uWKWklfBR_zUlhZHUZqik.
// Section 1 scores each benched Pokemon by how the foe's types hit it, and takes
// the best one that has a super-effective damaging move. Section 2, if nothing
// qualifies, takes the Pokemon with the move the AI thinks does most damage,
// with the cartridge's integer rounding and its byte-wide best-damage store.
//
// This is a policy, never engine semantics: it reads Showdown's state and
// draws nothing, so it cannot move the battle's own PRNG.
'use strict';
const { Dex } = require('pokemon-showdown');

const gen3 = Dex.mod('gen3');

// The cartridge's type table in its own order, which is what makes the
// rounding come out as it does, and the moves it gives power 1, which
// Section 2 skips. The table stops before the two rows after it in the
// cartridge (Normal and Fighting on Ghost), since the AI's loops never reach
// them. The engine builds its copy from the same file.
const { type_order: TABLE, nonstandard } = require('./emerald.json');
const NONSTANDARD = new Set(nonstandard);
const PHYSICAL = new Set(['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel']);

// One attacking type against a defender, row by row, flooring at each step.
function applyType(value, atk, defTypes) {
  for (const [a, d, mult] of TABLE) {
    if (a === atk && defTypes.includes(d)) value = Math.floor(value * mult / 10);
  }
  return value;
}

// Section 1. `foe` is {types, ability}; each candidate is {types, moves: [{type, category}]},
// or null where the party slot cannot come in. Returns a party index or -1.
function section1(foe, party) {
  // A single type is stored twice, so it hits twice.
  const attackers = foe.types.length === 1 ? [foe.types[0], foe.types[0]] : foe.types;
  const scores = party.map((mon) => mon && attackers.reduce((v, t) => applyType(v, t, mon.types), 10));
  const superEffective = (move) => move.category !== 'Status' &&
    !(move.type === 'Ground' && foe.ability === 'levitate') && applyType(10, move.type, foe.types) > 10;
  const left = new Set(party.keys());
  for (;;) {
    let best = -1;
    for (const i of left) if (party[i] && scores[i] > (best < 0 ? 0 : scores[best])) best = i;
    if (best < 0) return -1;
    if (party[best].moves.some(superEffective)) return best;
    left.delete(best);
  }
}

// Section 2. `base` is the last move's damage from the Pokemon going out;
// `stab` are its types. Returns a party index or -1.
function section2(foe, party, base, stab) {
  let bestDmg = 0, best = -1;
  party.forEach((mon, i) => {
    if (!mon) return;
    for (const move of mon.moves) {
      if (NONSTANDARD.has(move.id)) continue;
      let dmg = base;
      if (stab.includes(move.type)) dmg = Math.floor(dmg * 15 / 10);
      if (!(move.type === 'Ground' && foe.ability === 'levitate')) dmg = applyType(dmg, move.type, foe.types);
      if (dmg > bestDmg) { bestDmg = dmg % 256; best = i; }
    }
  });
  return best;
}

const stage = (stat, boost) => boost >= 0 ? Math.floor(stat * (2 + boost) / 2) : Math.floor(stat * 2 / (2 - boost));

// The cartridge's CalculateBaseDamage for the last move used, as the Pokemon
// going out would deal it: max roll, no crit, no STAB or effectiveness.
// ponytail: Thick Fat, Explosion's halved Defense, Mud/Water Sport and the
// dynamic power of variable moves are left out; add them if the policy's
// Section 2 picks ever need to match a cartridge trace.
function baseDamage(battle, attacker, target, move) {
  // A fainted Pokemon's status reads 'fnt', which the engine stores as none.
  const status = (p) => (p.status === 'fnt' ? '' : p.status);
  if (!move || move.category === 'Status') return 3;
  const physical = PHYSICAL.has(move.type);
  const ability = attacker.ability, item = attacker.item, species = attacker.species.id;
  let power = NONSTANDARD.has(move.id) ? 1 : move.basePower;
  let atk = attacker.storedStats[physical ? 'atk' : 'spa'];
  let def = target.storedStats[physical ? 'def' : 'spd'];
  if (ability === 'hugepower' || ability === 'purepower') atk *= 2;
  if (item === 'choiceband' && physical) atk = Math.floor(atk * 15 / 10);
  if ((item === 'silkscarf' && move.type === 'Normal') || (item === 'twistedspoon' && move.type === 'Psychic')) {
    atk = Math.floor(atk * 110 / 100);
  }
  if (item === 'souldew' && ['latios', 'latias'].includes(species) && !physical) atk = Math.floor(atk * 15 / 10);
  if (item === 'lightball' && species === 'pikachu' && !physical) atk *= 2;
  if (item === 'thickclub' && ['cubone', 'marowak'].includes(species) && physical) atk *= 2;
  if (target.item === 'souldew' && ['latios', 'latias'].includes(target.species.id) && !physical) {
    def = Math.floor(def * 15 / 10);
  }
  if (ability === 'hustle' && physical) atk = Math.floor(atk * 15 / 10);
  if (ability === 'guts' && status(attacker) && physical) atk = Math.floor(atk * 15 / 10);
  if (target.ability === 'marvelscale' && status(target) && physical) def = Math.floor(def * 15 / 10);
  const pinch = { overgrow: 'Grass', blaze: 'Fire', torrent: 'Water', swarm: 'Bug' }[ability];
  if (pinch === move.type && attacker.hp <= Math.floor(attacker.maxhp / 3)) power = Math.floor(power * 15 / 10);
  atk = stage(atk, attacker.boosts[physical ? 'atk' : 'spa']);
  def = stage(def, target.boosts[physical ? 'def' : 'spd']);
  let damage = Math.floor(Math.floor(atk * power * (Math.floor(2 * attacker.level / 5) + 2) / def) / 50);
  const screen = target.side.sideConditions[physical ? 'reflect' : 'lightscreen'];
  if (physical && status(attacker) === 'brn' && ability !== 'guts') damage = Math.floor(damage / 2);
  if (screen) damage = Math.floor(damage / 2);
  const weather = battle.field.effectiveWeather();
  if (!physical && weather === 'raindance') {
    if (move.type === 'Water') damage = Math.floor(damage * 15 / 10);
    if (move.type === 'Fire') damage = Math.floor(damage / 2);
  }
  if (!physical && weather === 'sunnyday') {
    if (move.type === 'Fire') damage = Math.floor(damage * 15 / 10);
    if (move.type === 'Water') damage = Math.floor(damage / 2);
  }
  return Math.max(damage, 1) + 2;
}

const moveInfo = (id) => {
  const m = gen3.moves.get(id);
  return { id: m.id, type: m.type, category: m.category };
};

// The replacement for `side`, as an index into side.pokemon. `order` maps each
// Pokemon to its original party slot, which is the order the cartridge walks.
function chooseReplacement(battle, side, order) {
  const outgoing = side.active[0];
  const foeMon = side.foe.active[0];
  const foe = { types: foeMon.getTypes(), ability: foeMon.ability };
  const party = [];
  for (const p of side.pokemon) {
    party[order.get(p)] = p.fainted || p.isActive ? null
      : { types: p.baseSpecies.types, moves: p.moveSlots.map((ms) => moveInfo(ms.id)) };
  }
  let pick = section1(foe, party);
  if (pick < 0) {
    const last = battle.lastMove && gen3.moves.get(battle.lastMove.id);
    pick = section2(foe, party, baseDamage(battle, outgoing, foeMon, last), outgoing.getTypes());
  }
  // Immune to everything: the next Pokemon in the party.
  if (pick < 0) pick = party.findIndex((mon) => mon);
  return side.pokemon.findIndex((p) => order.get(p) === pick);
}

// Where the table disagrees with Showdown's gen 3 chart, apart from the two
// rows the AI never reaches. Should always be empty.
function tableMismatches() {
  const types = gen3.types.names().filter((t) => t !== '???');
  const out = [];
  for (const atk of types) {
    for (const def of types) {
      const row = TABLE.find(([a, d]) => a === atk && d === def);
      const cartridge = row ? row[2] : 10;
      const showdown = !gen3.getImmunity(atk, def) ? 0 : 10 * 2 ** gen3.getEffectiveness(atk, def);
      const ghostRow = def === 'Ghost' && (atk === 'Normal' || atk === 'Fighting');
      if (cartridge !== showdown && !ghostRow) out.push(`${atk}>${def} ${cartridge} vs ${showdown}`);
    }
  }
  return out;
}

module.exports = { section1, section2, chooseReplacement, tableMismatches, moveInfo };
