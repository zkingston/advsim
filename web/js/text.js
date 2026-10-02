// Protocol lines as battle text, from one player's side of the split lines.
// Covers what gen 3 singles prints; anything else is left out.
import { nameOf } from './infostate.js';

const STATUS = { brn: 'was burned', par: 'was paralyzed', psn: 'was poisoned', tox: 'was badly poisoned',
  slp: 'fell asleep', frz: 'was frozen solid' };
const CURED = { slp: 'woke up!', frz: 'thawed out!', par: 'was cured of paralysis!', brn: 'was cured of its burn!',
  psn: 'was cured of its poison!', tox: 'was cured of its poison!' };
const WEATHER = { SunnyDay: 'The sunlight is strong.', RainDance: 'Rain is falling.', Sandstorm: 'The sandstorm rages.',
  none: 'The weather cleared.' };
// Status ids spelled out, where the log names one as a cause or a party ball's tooltip shows it.
export const STATUS_NAMES = { brn: 'burn', par: 'paralysis', psn: 'poison', tox: 'bad poison', slp: 'sleep', frz: 'freeze' };
const CAUSES = { ...STATUS_NAMES, tox: 'poison', Recoil: 'recoil' };
// A volatile starting and ending, as the games word it (the Pokemon's name goes first).
const STARTED = { confusion: 'became confused!', Substitute: 'put in a substitute!', 'Leech Seed': 'was seeded!',
  Attract: 'fell in love!', Encore: 'received an encore!', Yawn: 'grew drowsy!', 'Flash Fire': 'powered up its Fire-type moves!' };
const ENDED = { confusion: 'snapped out of its confusion!', Substitute: 'lost its substitute!', 'Leech Seed': 'was freed from Leech Seed!',
  Attract: 'got over its infatuation.', Encore: 'is no longer under Encore.' };
const TRAPS = new Set(['Wrap', 'Bind', 'Fire Spin', 'Whirlpool', 'Sand Tomb', 'Clamp']);
const STATS = { atk: 'Attack', def: 'Defense', spa: 'Sp. Atk', spd: 'Sp. Def', spe: 'Speed', accuracy: 'accuracy', evasion: 'evasiveness' };

// Why a Pokemon did not act: the log's sentence and the banner's label.
export const CANT = {
  par: ['is paralyzed! It can\'t move!', 'Fully paralyzed!'], slp: ['is fast asleep.', 'Fast asleep'],
  frz: ['is frozen solid!', 'Frozen solid'], flinch: ['flinched and couldn\'t move!', 'Flinched!'],
  recharge: ['must recharge!', 'Must recharge'], 'ability: Truant': ['is loafing around!', 'Loafing around'],
  Attract: ['is immobilized by love!', 'Immobilized by love'], 'Focus Punch': ['lost its focus and couldn\'t move!', 'Lost its focus'],
  Disable: ['can\'t use its disabled move!', 'Disabled'], nopp: ['has no PP left for that move!', 'No PP left'],
  'move: Taunt': ['can\'t use that move after the taunt!', 'Taunted'],
};

export function lines(log, upto, me) {
  const out = [];
  for (let i = 0; i < upto; i++) {
    let line = log[i];
    if (line.startsWith('|split|')) {
      line = line.split('|')[2] === me ? log[i + 1] : log[i + 2];
      i += 2;
    }
    const t = text(line, me);
    if (t) out.push(t);
  }
  return out;
}

export function text(line, me) {
  const p = line.split('|');
  const who = (id) => (id.slice(0, 2) === me ? nameOf(id) : `The foe's ${nameOf(id)}`);
  const from = /\[from\] (?:item: |ability: |move: )?([^|]+)/.exec(line);
  const src = from ? ` (${CAUSES[from[1]] || from[1]})` : '';
  const effect = (x) => x.replace(/^(move|ability): /, '');
  switch (p[1]) {
    case 'turn': return { turn: +p[2] };
    case 'switch': case 'drag': return { text: `${p[2].slice(0, 2) === me ? 'Go' : 'The foe sent out'} ${p[3].split(',')[0]}!` };
    case 'move': return { text: `${who(p[2])} used ${p[3]}!` };
    case '-damage': return { text: `${who(p[2])} is at ${hp(p[3])}${src}.`, dim: true };
    case '-heal': return { text: `${who(p[2])} restored HP to ${hp(p[3])}${src}.`, dim: true };
    case 'faint': return { text: `${who(p[2])} fainted!`, strong: true };
    case '-status': return { text: `${who(p[2])} ${STATUS[p[3]] || p[3]}${src}.` };
    case '-curestatus': return { text: `${who(p[2])} ${CURED[p[3]] || `was cured of its ${p[3]}!`}${src}` };
    case '-boost': case '-unboost': {
      const n = +p[4];
      const how = n === 0 ? "won't go any " + (p[1] === '-boost' ? 'higher' : 'lower') : `${p[1] === '-boost' ? 'rose' : 'fell'}${n > 1 ? ' sharply' : ''}`;
      return { text: `${who(p[2])}'s ${STATS[p[3]] || p[3]} ${how}!` };
    }
    case '-supereffective': return { text: "It's super effective!" };
    case '-resisted': return { text: "It's not very effective..." };
    case '-immune': return { text: `It doesn't affect ${who(p[2]).replace(/^The /, 'the ')}.` };
    case '-crit': return { text: 'A critical hit!' };
    case '-miss': return { text: `${who(p[2])}'s attack missed!` };
    case '-fail': return { text: 'But it failed!' };
    case 'cant': return { text: `${who(p[2])} ${CANT[p[3]] ? CANT[p[3]][0] : `can't move (${p[3]}).`}`, strong: p[3] === 'par' };
    case '-weather': return line.includes('[upkeep]') ? null : { text: WEATHER[p[2]] || p[2] };
    case '-start': {
      const e = effect(p[3]);
      if (/^perish\d$/.test(e)) return { text: `${who(p[2])}'s perish count fell to ${e.slice(6)}.` };
      if (e === 'typechange') return { text: `${who(p[2])} became the ${p[4]} type${src}.` };
      return { text: `${who(p[2])} ${STARTED[e] || `is affected by ${e}.`}`, dim: true };
    }
    case '-end': {
      const e = effect(p[3]);
      if (e === 'Yawn') return null;  // the sleep that follows says it
      return { text: `${who(p[2])} ${ENDED[e] || (TRAPS.has(e) ? `was freed from ${e}!` : `is no longer affected by ${e}.`)}`, dim: true };
    }
    case '-sidestart': return { text: `${p[3].replace('move: ', '')} on ${p[2].slice(0, 2) === me ? 'your' : "the foe's"} side.` };
    case '-item': return { text: `${who(p[2])} has ${p[3]}${src}.` };
    case '-enditem': return { text: `${who(p[2])}'s ${p[3]} is gone${src}.`, dim: true };
    case '-ability': return { text: `${who(p[2])}'s ${p[3]}${src}.` };
    case '-transform': return { text: `${who(p[2])} transformed!` };
    // What a status move did, or why it did nothing (Protect, Sleep Clause), and the like.
    case '-activate': return activated(p, who, line);
    case '-singleturn': {
      const e = effect(p[3]);
      if (e === 'Protect') return { text: `${who(p[2])} protected itself!` };
      if (e === 'Endure') return { text: `${who(p[2])} braced itself!` };
      return null;
    }
    case '-singlemove': return effect(p[3]) === 'Destiny Bond' ? { text: `${who(p[2])} is trying to take its foe down with it!` } : null;
    case '-message': return { text: p[2], strong: true };
    case '-hint': return { text: p[2], dim: true };
    case '-cureteam': return { text: 'A soothing aroma wafted through the area!' };
    case '-clearallboost': return { text: 'All stat changes were eliminated!' };
    case '-sethp': return line.includes('[silent]') ? null : { text: 'The battlers shared their pain!' };
    case '-setboost': return effect(line.match(/\[from\] move: ([^|]+)/)?.[1] || '') === 'Belly Drum'
      ? { text: `${who(p[2])} cut its own HP and maximized its Attack!` } : null;
    case '-fieldactivate': return effect(p[2]) === 'Perish Song' ? { text: 'All Pokémon hearing the song will faint in three turns!' } : null;
    case '-sideend': return { text: `The ${p[3].replace('move: ', '')} disappeared from around ${p[2].slice(0, 2) === me ? 'your' : "the foe's"} team${src}!` };
    case '-prepare': return { text: `${who(p[2])} is charging ${p[3]}!` };
    case '-hitcount': return { text: `Hit ${p[3]} time${p[3] === '1' ? '' : 's'}!` };
    case 'win': return { text: `${p[2]} won!`, strong: true };
    case 'tie': return { text: 'The battle is a tie.', strong: true };
    default: return null;
  }
}

// -activate: an effect doing its work (the target named first, `[of]` the other side).
function activated(p, who, line) {
  const e = (p[3] || '').replace(/^(move|ability): /, '');
  const of = /\[of\] ([^|]+)/.exec(line)?.[1];
  switch (e) {
    case 'Protect': return { text: `${who(p[2])} protected itself!` };
    case 'Endure': return { text: `${who(p[2])} endured the hit!` };
    case 'Substitute': return { text: `The substitute took damage for ${who(p[2]).replace(/^The /, 'the ')}!`, dim: true };
    case 'confusion': return { text: `${who(p[2])} is confused!` };
    case 'trapped': return { text: `${who(p[2])} can no longer escape!` };
    case 'Attract': return { text: `${who(p[2])} is in love${of ? ` with ${who(of).replace(/^The /, 'the ')}` : ''}!` };
    case 'Destiny Bond': return { text: `${who(p[2])} took its attacker down with it!` };
    case 'Heal Bell': return { text: 'A bell chimed!' };
    case 'Trick': return { text: `${who(p[2])} switched items${of ? ` with ${who(of).replace(/^The /, 'the ')}` : ''}!` };
    case 'Shed Skin': return { text: `${who(p[2])}'s Shed Skin activated!`, dim: true };
    case 'Wrap': case 'Bind': case 'Fire Spin': case 'Whirlpool': case 'Sand Tomb': case 'Clamp':
      return { text: `${who(p[2])} is hurt by ${e}!`, dim: true };
    default: return null;
  }
}

const hp = (cond) => {
  const [v] = cond.split(' ');
  if (v === '0') return '0%';
  const [a, b] = v.split('/').map(Number);
  return b === 100 ? `${a}%` : `${a}/${b}`;
};
