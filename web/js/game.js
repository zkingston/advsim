// One battle: Showdown's simulator driven decision by decision, each side's InfoState, the
// views (log cursor and requests) at every decision, and the choices that got there, which
// the replay link records.
import { InfoState } from './infostate.js';
import { Vocab } from './observation.js';
import { command } from './command.js';
import { OPPONENTS } from './settings.js';
import { json } from './fetch.js';

const { Battle, Teams } = window.Showdown;
export const vocab = new Vocab(await json('data/vocab.json'));
const seedString = (n) => `sodium,${n.toString(16).padStart(64, '0')}`;
const copy = (x) => JSON.parse(JSON.stringify(x));

export class Game {
  // `players`: [p1, p2], each 'human' or an OPPONENTS key.
  constructor(n, players) {
    this.n = n;
    this.players = players;
    const gen = Teams.getGenerator('gen3randombattle', seedString(n));
    this.battle = new Battle({ formatid: 'gen3randombattle', seed: seedString(n) });
    const label = (who) => (who === 'human' ? 'You' : OPPONENTS[who][0]);
    this.battle.setPlayer('p1', { name: label(players[0]), team: gen.getTeam() });
    this.battle.setPlayer('p2', { name: players[1] === players[0] ? `${label(players[1])} (2)` : label(players[1]), team: gen.getTeam() });
    this.states = ['p1', 'p2'].map((p) => new InfoState(p, vocab.speciesTypes));
    this.views = [this.view()];
    this.evals = [null];          // per view, [p1 bot's, p2 bot's] own win estimate from the decision before it, or null
    this.choices = [];            // per decision, each side's Showdown choice (null: none)
    this.thoughts = [];           // per decision, each side's MCTS summary (thinking.js), or null
    this.picked = [null, null];
    this.notice = '';
  }

  // A battle rebuilt from its seed and recorded choices. The bots' sampling is not the
  // battle's PRNG, so this is exact.
  static replay(n, players, choices) {
    const g = new Game(n, players);
    // Both sides see every request, as in play: an InfoState takes its party order from the first.
    const catchUp = () => { g.state(0); g.state(1); };
    catchUp();
    choices.forEach((pair, i) => {
      pair.forEach((c, p) => { if (c && !g.battle.sides[p].choose(c)) throw new Error(`replay: ${c} refused at decision ${i}`); });
      g.picked = pair;
      g.commit();
      catchUp();
    });
    return g;
  }

  // The battle as it stood at decision k, to play on from there; checked against this log.
  branch(k) {
    const g = Game.replay(this.n, this.players, this.choices.slice(0, k));
    g.evals = this.evals.slice(0, k + 1);
    g.thoughts = this.thoughts.slice(0, k);
    const upTo = (game) => game.battle.log.slice(0, game.views[k].cursor).filter((l) => !l.startsWith('|t:|')).join('\n');
    if (upTo(g) !== upTo(this)) throw new Error(`branch: the replay diverged before decision ${k}`);
    return g;
  }

  view() {
    return { cursor: this.battle.log.length, requests: this.battle.sides.map((s) => copy(s.activeRequest || null)) };
  }

  get last() { return this.views.length - 1; }

  // Player p's own info-state at the latest decision; choose() reads what this last took.
  state(p) {
    const st = this.states[p], v = this.views.at(-1);
    st.feed(this.battle.log, v.cursor);
    st.takeRequest(v.requests[p]);
    return st;
  }

  needs(p) { const r = this.views.at(-1).requests[p]; return !this.battle.ended && r && !r.wait; }

  // p1's InfoState at view k, replayed view by view: the own party's order is the first
  // request's, which the action codes index, and later requests list it as switches left it.
  infoAt(k) {
    const st = new InfoState('p1', vocab.speciesTypes);
    for (const u of this.views.slice(0, k + 1)) {
      st.feed(this.battle.log, u.cursor);
      st.takeRequest(u.requests[0]);
    }
    return st;
  }

  // The latest request at or before view k that shows p1's team: a finished battle's is null.
  teamAt(k) { return this.views.slice(0, k + 1).map((u) => u.requests[0]).filter((r) => r && r.side).at(-1); }

  // False when Showdown refuses the choice for something the request hid (Arena Trap, Magnet
  // Pull and Shadow Tag trap with only `maybeTrapped`): the refusal updates the request, and
  // the player chooses again from it.
  choose(p, code) {
    const v = this.views.at(-1), side = this.battle.sides[p];
    const c = command(code, this.states[p], v.requests[p]);
    if (!c || side.choose(c)) { this.picked[p] = c; return true; }
    const updated = copy(side.activeRequest);
    if (JSON.stringify(updated) === JSON.stringify(v.requests[p])) throw new Error(`invalid choice ${c} (code ${code}) for ${side.id}: ${side.choice.error}`);
    v.requests[p] = updated;
    this.notice = side.choice.error;
    return false;
  }

  commit(values = [null, null], thoughts = [null, null]) {
    this.choices.push(this.picked);
    this.thoughts.push(thoughts);
    this.picked = [null, null];
    this.battle.commitChoices();
    this.views.push(this.view());
    this.evals.push(values);
    this.notice = '';
  }
}

// ---- replay links: #r/<h|w>/<battle>/<opponent>/<choices>, two characters per decision (p1,
// p2): 1-4 a move, a-f a switch to that request slot, '-' no choice. Kept in the address bar
// as the battle goes, so reloading or sharing the link rebuilds the same game.

const SLOTS = 'abcdef';
const encodeChoice = (c) => (c == null ? '-' : c.startsWith('move ') ? c.slice(5) : SLOTS[+c.slice(7) - 1]);
const decodeChoice = (ch) => (ch === '-' ? null : /[1-4]/.test(ch) ? `move ${ch}` : `switch ${SLOTS.indexOf(ch) + 1}`);

export const saveLink = (game) => history.replaceState(null, '', `#r/${game.players[0] === 'human' ? 'h' : 'w'}/${game.n}/${game.players[1]}/`
  + game.choices.map((pair) => pair.map(encodeChoice).join('')).join(''));

// A replay link's parts, or null: { watch, n, opponent, choices }.
export function readLink(hash) {
  const m = new RegExp(`^#r/([hw])/(\\d+)/(${Object.keys(OPPONENTS).join('|')})/((?:[1-4a-f-]{2})*)$`).exec(hash);
  if (!m) return null;
  return { watch: m[1] === 'w', n: +m[2], opponent: m[3], choices: (m[4].match(/../g) || []).map((pair) => [...pair].map(decodeChoice)) };
}
