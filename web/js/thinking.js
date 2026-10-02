// The Bot panel: what the MCTS thought at a decision, from its root statistics (search.js's
// decide): its own options with how often it plays each and its win chance if it does, what
// it expects the other side to do, and its win chance for each pairing.
//
// The other side's options are grouped as the search's labels have them (mcts/labels.js):
// moves by type ("Water attack", "Status move"), switches by the Pokemon named if the bot has
// seen it and by type otherwise, since a move slot is a different move in different worlds.
//
// The bot's own options name its set, so while a human plays, a move or Pokemon of the bot's
// that the human has not seen yet is shown without its name (drawThinking; the name never
// reaches the page).
import { $, esc, speciesName, typeBadges } from './dex.js';

const A = 12;
const pct = (x) => `${Math.round(100 * x)}%`;

// A decision's summary, made as the bot decides (players.js). `st`: its InfoState, `d`: decide's,
// `stats`: the workers' root statistics, with their tallies by the other side's labels (search.js).
//
// What the other side does is shown two ways: the network's prediction ("likely"), and the
// root equilibrium's ("worst case"), which is the reply that hurts the bot most, not a
// forecast: it can put 99% on a switch the network gives 9%.
export function summarize(game, p, st, d, stats) {
  const req = game.views.at(-1).requests[p], moves = req.active?.[0]?.moves || [];
  const me = st.sides[st.me];
  // Each option's name, and what naming it gives away: a move of its active Pokemon, or a Pokemon.
  const move = (m) => ({ label: m?.move.replace(/ \d+$/, ''), reveal: m && { mon: me.active, move: m.id } });  // the request says Return 102
  const mine = (a) => (a < 4 ? move(moves[a]) : a === 11 ? move(moves[0]) : a === 10 ? { label: 'Wait' }
    : { label: `Switch to ${speciesName(me.mons.get(me.order[a - 4]).species)}`, reveal: { mon: me.order[a - 4] } });
  // The bot's codes against the other side's: cells are [p1 code][p2 code], values p1's.
  const at = (b, o) => (p === 0 ? b * A + o : o * A + b);
  const n = (b, o) => d.cellN[at(b, o)];
  const w = (b, o) => (p === 0 ? 1 : -1) * d.cellW[at(b, o)];
  let total = 0, sum = 0;
  for (let c = 0; c < A * A; c++) { total += d.cellN[c]; sum += d.cellW[c]; }
  const mean = (p === 0 ? 1 : -1) * sum / Math.max(total, 1);
  const q = (b, o) => (n(b, o) ? w(b, o) / n(b, o) : mean);
  const opp = d.strategy[1 - p];
  const win = (v) => (v + 1) / 2;
  const group = (label) => ({ label, n: new Float64Array(A), w: new Float64Array(A), codes: new Float64Array(A), likely: 0, prob: 0 });

  // The workers' tallies by label: root visits and value per bot code, and how often each of the
  // other side's codes carried the label, which splits the equilibrium's weight on a code.
  const groups = new Map();
  let worlds = 0;
  for (const x of stats) {
    worlds += x.likelyWorlds;
    for (const [label, k] of Object.entries(x.keyed)) {
      const g = groups.get(label) || group(label);
      for (let a = 0; a < A; a++) { g.n[a] += k.n[a]; g.w[a] += k.w[a]; g.codes[a] += k.codes[a]; }
      groups.set(label, g);
    }
    for (const [label, mass] of Object.entries(x.likely)) {
      if (!groups.has(label)) groups.set(label, group(label));
      groups.get(label).likely += mass;
    }
  }
  const carried = new Float64Array(A);
  for (const g of groups.values()) for (let o = 0; o < A; o++) carried[o] += g.codes[o];
  for (const g of groups.values()) {
    g.likely /= Math.max(worlds, 1);
    for (let o = 0; o < A; o++) if (carried[o]) g.prob += opp[o] * g.codes[o] / carried[o];
  }
  // The top six replies keep their own column (five and 'Something else' when there are more), so
  // the table fits a phone without scrolling; a guessed bench splits into many small types.
  const ranked = [...groups.values()].filter((g) => g.likely >= 0.005 || g.prob >= 0.005 || g.n.some(Boolean))
    .sort((x, y) => y.likely + y.prob - x.likely - x.prob);
  const cols = ranked.length > 6 ? ranked.slice(0, 5) : ranked;
  if (ranked.length > 6) {
    const rest = group('Something else');
    for (const g of ranked.slice(5)) {
      for (let a = 0; a < A; a++) { rest.n[a] += g.n[a]; rest.w[a] += g.w[a]; }
      rest.likely += g.likely;
      rest.prob += g.prob;
    }
    cols.push(rest);
  }
  // Its options: the ones it plays at all or spent a real share of the search on.
  const rows = [];
  let skipped = 0;
  for (let b = 0; b < A; b++) {
    const visits = [...Array(A).keys()].reduce((s, o) => s + n(b, o), 0);
    if (d.play[b] < 0.01 && visits < 0.05 * total) { if (visits) skipped++; continue; }
    rows.push({ ...mine(b), play: d.play[b], visits,
      win: win([...Array(A).keys()].reduce((s, o) => s + opp[o] * q(b, o), 0)),
      cells: cols.map((g) => ({ n: g.n[b], win: g.n[b] ? win((p === 0 ? 1 : -1) * g.w[b] / g.n[b]) : null })) });
  }
  rows.sort((x, y) => y.play - x.play);
  return { side: p, turn: st.turn, descents: d.visits, win: win(p === 0 ? d.value : -d.value), played: mine(d.action), skipped,
    rows, cols: cols.map(({ label, likely, prob }) => ({ label, likely, prob })) };
}

// A reply or option as shown: types as the team panel's badges, a switch as ⇆. `compact` (the
// table's heads) drops the words: an attack is its type, an unseen Pokemon a '?'.
function rich(label, compact) {
  const badge = (t) => typeBadges([t], true);
  let m;
  if ((m = /^(\w+) attack$/.exec(label))) return compact ? badge(m[1]) : `${badge(m[1])} attack`;
  if ((m = /^Switch to an? (\w+) type it hasn't seen$/.exec(label))) return compact ? `⇆ <span class="nowrap">${badge(m[1])}?</span>` : `⇆ unseen ${badge(m[1])}`;
  if (compact && (m = /^Hidden move (\d+)$/.exec(label))) return `Hidden ${m[1]}`;
  if (label === 'Switch to an unseen Pokémon') return compact ? '⇆ ?' : '⇆ unseen Pokémon';
  if ((m = /^Switch to (.+)$/.exec(label))) return `⇆ ${esc(m[1])}`;
  if (label === 'Status move') return 'Status';
  if (label === 'Something else') return 'Other';
  return esc(label);
}

// A win chance's colour: orange below even, blue above, the panel's own at even; never so
// strong that the number on it stops reading.
const heat = (x) => (x == null ? 'transparent'
  : `color-mix(in srgb, ${x >= 0.5 ? 'var(--good)' : 'var(--bad)'} ${Math.round(Math.min(1, Math.abs(x - 0.5) * 5) * 80)}%, var(--panel))`);

// `shown(reveal)`: whether the viewer may see that name. Hidden moves are numbered so rows stay apart.
function section(t, name, you, shown) {
  const hidden = new Map();
  const label = (o) => {
    if (shown(o.reveal)) return o.label;
    if (!o.reveal.move) return 'Switch to an unseen Pokémon';
    const key = `${o.reveal.mon}|${o.reveal.move}`;
    if (!hidden.has(key)) hidden.set(key, `Hidden move ${hidden.size + 1}`);
    return hidden.get(key);
  };
  const bar = (x) => `<span class="meter"><span class="bar"><i style="width:${(100 * x).toFixed(1)}%"></i></span>${pct(x)}</span>`;
  return `<section class="thought">
    <h3>${esc(name)} · turn ${t.turn}</h3>
    <p>${Math.round(t.descents)} simulations. It played <b>${rich(label(t.played), false)}</b> and gave itself <b>${pct(t.win)}</b> to win.</p>
    <h4>Its options</h4>
    <table class="opts"><thead><tr><th></th><th>Plays</th><th>Wins if played</th></tr></thead><tbody>
      ${t.rows.map((r) => `<tr><td>${rich(label(r), false)}</td><td>${bar(r.play)}</td><td>${pct(r.win)}</td></tr>`).join('')}
    </tbody></table>
    ${t.skipped ? `<p class="note">${t.skipped} more it barely tried.</p>` : ''}
    <h4>What it expected ${you}</h4>
    <table class="opts"><thead><tr><th></th><th>Likely</th><th>Worst case</th></tr></thead><tbody>
      ${t.cols.map((c) => `<tr><td>${rich(c.label, false)}</td><td>${bar(c.likely)}</td><td>${bar(c.prob)}</td></tr>`).join('')}
    </tbody></table>
    <h4>Its win chance for each pairing</h4>
    <table class="matrix"><thead><tr><th></th>${t.cols.map((c) => `<th title="${esc(c.label)}">${rich(c.label, true)}</th>`).join('')}</tr></thead><tbody>
      ${t.rows.map((r) => `<tr><th title="${esc(label(r))}">${rich(label(r), true)}</th>${r.cells.map((c) => `<td style="background:${heat(c.win)}" title="${c.n} simulations">${c.win == null ? '–' : pct(c.win)}</td>`).join('')}</tr>`).join('')}
    </tbody></table>
  </section>`;
}

// The panel at view k: the decision made there, or at the latest view (not decided yet) the one before it.
export function drawThinking(game, k) {
  const j = k < game.last ? k : game.last - 1, t = game.thoughts[j] || [];
  const human = game.players[0] === 'human';
  // Playing, the bot's names are the human's to see only once the battle has shown them (by now,
  // even if after decision j); watching, or once it is over, everything is.
  const foe = human && !game.battle.ended ? game.infoAt(game.last).sides.p2.mons : null;
  const shown = (r) => !foe || !r || (r.move ? !!foe.get(r.mon)?.moves.has(r.move) : !!foe.get(r.mon)?.species);
  const parts = [0, 1].filter((p) => t[p]).map((p) => section(t[p], human ? 'MCTS' : `MCTS (p${p + 1})`, human ? 'from you' : 'from the other side', shown));
  $('botpanel').innerHTML = parts.join('') || `<p class="empty">${game.players.includes('mcts')
    ? 'The MCTS shows its reasoning here once it has made a move. Moves replayed from a link were not searched.'
    : 'Choose MCTS as the opponent in Settings to see its reasoning here after each move.'}</p>`;
}
