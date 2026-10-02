// The max-damage baseline's move choice, shared by the sweep's policy mix
// (play.js) and the web page's Emerald-style opponent (bundled by bundle.js).
'use strict';

const SPECIAL = new Set(['Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Dark', 'Psychic', 'Dragon']);

// A rough damage estimate from dex lookups alone: power, STAB, effectiveness
// and the attack-to-defence ratio. getDamage would draw from the battle's PRNG.
function estimate(battle, attacker, target, moveId) {
  const move = battle.dex.moves.get(moveId);
  if (move.category === 'Status') return 0;
  const type = move.id === 'hiddenpower' ? attacker.hpType : move.type;
  const types = target.getTypes();
  if (!battle.dex.getImmunity(type, types)) return 0;
  const power = move.basePower || (move.damage === 'level' ? attacker.level : move.damage || 0);
  const [atk, def] = SPECIAL.has(type) ? ['spa', 'spd'] : ['atk', 'def'];
  const stab = attacker.getTypes().includes(type) ? 1.5 : 1;
  return power * stab * 2 ** battle.dex.getEffectiveness(type, types) * attacker.storedStats[atk] / target.storedStats[def];
}

// The legal action code (0-3 a move slot, 11 the forced action) that hits
// hardest; the first on a tie. The forced action is all there is when it is legal.
function maxDamageMove(battle, side, moves) {
  const active = side.active[0];
  let best = moves[0], bestDmg = -1;
  for (const m of moves) {
    const dmg = m < 4 ? estimate(battle, active, side.foe.active[0], active.moveSlots[m].id) : 0;
    if (dmg > bestDmg) { best = m; bestDmg = dmg; }
  }
  return best;
}

module.exports = { estimate, maxDamageMove };
