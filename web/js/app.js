// The page: Showdown's simulator (dist/showdown.js) runs the battle in the browser, p1's
// InfoState (exactly what the protocol shows p1) is drawn, and each turn plays back line by
// line. game.js is the battle, players.js the bots, scene.js and panels.js the drawing,
// settings.js the Settings panel and the tabs.
import { Game, readLink, saveLink } from './game.js';
import { bot, assist, probsOf } from './players.js';
import { settings, OPPONENTS, setOpponent, setTab, toBattle, phone } from './settings.js';
import { $, setText, speciesName } from './dex.js';
import { drawField, clearField, animate, recall, say, showResult, setTerrain } from './scene.js';
import { drawLog, drawControls, drawTeam, drawSpark } from './panels.js';
import { lines, text } from './text.js';
import { drawThinking } from './thinking.js';
import { json } from './fetch.js';

setText(await json('data/text.json'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fresh = () => Math.floor(Math.random() * 2 ** 32);
// Draws the view the slider is on, through its own handler (the pending choice, the scrub buttons).
const showSlider = (k = null) => {
  if (k != null) $('scrub').value = k;
  $('scrub').dispatchEvent(new Event('input'));
};

// View k, whole. `onPick`: at the last view, the human's pending choice. Watching, the assist's
// percentages for p1's options come once the network has scored the view: kept per view, and the
// view drawn again through the slider (as it was drawn) if it is still the one shown.
let rendered = 0;
function render(game, k, onPick, probs = null) {
  const st = game.infoAt(k), last = k === game.last, req = game.views[k].requests[0], seq = ++rendered;
  const watching = game.players[0] !== 'human';
  if (watching && settings.assist && req) {
    game.assists ??= [];
    probs = game.assists[k] ?? null;
    if (!probs) {
      probsOf(st).then((p) => {
        game.assists[k] = p;
        if (seq === rendered && !animating && current === game && +$('scrub').value === k) showSlider();
      });
    }
  }
  drawField(st, last && game.notice);
  drawLog(game.battle.log, game.views[k].cursor);
  drawTeam(game.teamAt(k), game.players[0] === 'human' ? 'Your team' : `${OPPONENTS[game.players[0]][0]}'s team (p1)`);
  const names = game.players.map((w, p) => (w === 'human' ? 'You' : `${OPPONENTS[w][0]}${game.players[0] === game.players[1] ? ` (p${p + 1})` : ''}`));
  drawSpark(game.evals, k, showSlider, names);
  drawControls(st, req, last ? onPick : null, probs, watching);
  drawThinking(game, k);
  if (last && onPick && req) {
    if (req.forceSwitch && phone.matches) setTab('team');  // the replacement is picked from the team
    const me = st.sides.p1;
    say(req.forceSwitch ? 'Choose a Pokémon to send out!' : `What will ${speciesName(me.mons.get(me.active).species)} do?`);
  } else {
    say(lines(game.battle.log, game.views[k].cursor, 'p1').filter((l) => l.text).at(-1)?.text || '');
  }
}

// View k - 1 to view k, one protocol line at a time (a split line and its two halves count as one).
const SPEEDS = { normal: 1, fast: 0.4, off: 0 };
let animating = false;  // a turn is playing back: nothing redraws the whole view over it
async function playTurn(game, k) {
  const factor = SPEEDS[settings.speed] ?? 1;
  if (!factor) return;
  animating = true;
  try { await playLines(game, k, factor); } finally { animating = false; }
}
async function playLines(game, k, factor) {
  const log = game.battle.log;
  const st = game.infoAt(k - 1);
  if (game.players[0] === 'human') drawControls(st, null, null);  // watching, the read-only moves stay up
  while (st.pos < game.views[k].cursor && current === game) {
    const at = st.pos, split = log[at].startsWith('|split|');
    const line = split ? log[at + 1 + (log[at].split('|')[2] === 'p1' ? 0 : 1)] : log[at];
    const out = recall(line) * factor;  // the Pokemon going out, before the switch is drawn
    if (out) await sleep(out);
    if (current !== game) return;  // a new battle started meanwhile: this one must not draw over it
    st.feed(log, split ? at + 3 : at + 1);
    drawField(st, '');
    drawLog(log, st.pos);
    const t = text(line, 'p1');
    if (t && t.text) say(t.text);
    const hold = animate(line) * factor;
    if (hold) await sleep(hold);
    await whilePaused();
  }
}

// ---- the loop

let current = null;       // the game on screen; a loop whose game is no longer it stops
// Watching, the Pause button holds the loop at the next event or turn; the slider still works.
let paused = false, unpause = null;
async function whilePaused() { while (paused) await new Promise((r) => { unpause = r; }); }
function setPaused(on) {
  paused = on;
  $('pause').textContent = on ? 'Resume' : 'Pause';
  if (!on) unpause?.();
}
$('pause').onclick = () => {
  setPaused(!paused);
  if (!paused) showSlider($('scrub').max);  // resuming goes back to the latest decision from wherever the slider was
};
let redrawPrompt = null;  // redraws the human's pending choice (the assist toggle, the scrubber)
// The assist toggle redraws a pending choice, or, watching, the view the slider is on.
settings.onAssist = () => (redrawPrompt ? redrawPrompt() : showSlider());

// A new battle, or `from`: a replayed or branched one (Game.replay, Game.branch), played on
// from its last view. `delay`: the pause between turns when watching.
async function start(n, players, delay, from = null) {
  const game = from || new Game(n, players);
  current = game;
  redrawPrompt = null;
  setPaused(false);
  $('pause').hidden = players[0] === 'human';
  saveLink(game);
  showResult(null);
  setTerrain(n);
  clearField();
  $('seed').textContent = `Battle #${n}`;
  const scrub = $('scrub');
  const show = (k, pick, probs) => { scrub.max = game.last; scrub.value = k; render(game, k, pick, probs); };
  // The end screen, over the last view only: scrubbing back to look at the battle hides it.
  let ending = null;
  const showEnding = (k) => {
    showResult(k === game.last && ending ? ending.text : null, ending?.won);
    if (k === game.last && ending) $('again').onclick = ending.again;
  };
  // Scrubbing shows a past view; the last one, or the button, brings back a pending choice.
  scrub.oninput = () => {
    const k = +scrub.value, last = game.last;
    showEnding(k);
    if (k === last && redrawPrompt) return redrawPrompt();
    render(game, k, null);
    if (k === last) return;
    const button = (label, title, onclick) => {
      const b = document.createElement('button');
      b.textContent = label; b.title = title; b.onclick = onclick;
      $('controls').append(b);
    };
    if (redrawPrompt) button('Back to the current turn', 'Return to the decision waiting for you', () => { scrub.value = last; redrawPrompt(); });
    button('Play from here', 'Go back to this decision and play on from it', () => { toBattle(); start(n, players, delay, game.branch(k)); });
  };
  show(game.last);
  // Bots start on a decision as soon as it exists, so a search runs while the last turn
  // animates and while the human picks; choices are simultaneous, so nothing is lost.
  const think = () => [0, 1].map((p) => {
    if (game.players[p] === 'human' || !game.needs(p)) return null;
    const b = bot(game, p);
    b.then(() => { b.done = true; }, () => {});
    return b;
  });
  let bots = think();
  while (!game.battle.ended && current === game) {
    await whilePaused();
    if (current !== game) return;
    const values = [null, null];
    const thoughts = [null, null];
    for (const p of [0, 1]) {
      if (!game.needs(p)) continue;
      for (let chosen = false; !chosen;) {
        let code;
        if (game.players[p] === 'human') {
          game.state(p);
          code = await new Promise((resolve) => {
            redrawPrompt = async () => {
              const probs = settings.assist ? await assist(game, p) : null;
              if (current === game) show(game.last, resolve, probs);  // not over a battle started meanwhile
            };
            redrawPrompt();
          });
          redrawPrompt = null;
          if (current !== game) return;
          if (document.body.dataset.tab === 'team') toBattle();  // a switch was picked there
        } else {
          bots[p] ??= bot(game, p);  // a refused choice asks again with the updated request
          if (game.players[p] === 'mcts' && !bots[p].done) say(`${OPPONENTS.mcts[0]} is thinking…`);
          const b = await bots[p];
          bots[p] = null;
          if (current !== game) return;
          code = b.code;
          thoughts[p] = b.thought || null;
          values[p] = b.value ?? null;
        }
        chosen = game.choose(p, code);
      }
    }
    game.commit(values, thoughts);
    saveLink(game);
    scrub.max = game.last;
    scrub.value = game.last;
    if (!game.battle.ended) bots = think();
    await playTurn(game, game.last);
    if (current !== game) return;
    show(game.last, null);
    if (delay) await sleep(delay);
  }
  if (current !== game) return;
  $('pause').hidden = true;
  show(game.last, null);
  const winner = game.battle.winner, human = players[0] === 'human';
  ending = human ? { text: winner === 'You' ? 'You won!' : winner ? `${winner} won…` : "It's a tie!", won: winner === 'You' }
    : { text: winner ? `${winner} won!` : "It's a tie!", won: true };
  ending.again = () => (human ? play() : watch());
  showEnding(game.last);
}

const play = (n = fresh()) => start(n, ['human', settings.opponent], 0);
const watch = (n = fresh()) => start(n, [settings.opponent, settings.opponent], 300);
$('new').onclick = () => { toBattle(); play(); };
$('watch').onclick = () => { toBattle(); watch(); };
$('loading').hidden = true;

// #r/... replays a link (and plays on if it is not over); #<n> plays battle n; #watch/<n>
// watches the chosen opponent play itself in it.
const link = readLink(location.hash);
const [, watching, digits] = /^#(watch\/)?(\d*)$/.exec(location.hash) || [];
if (link) {
  setOpponent(link.opponent);
  const players = link.watch ? [link.opponent, link.opponent] : ['human', link.opponent];
  let game;
  try {
    game = Game.replay(link.n, players, link.choices);
  } catch {  // not this battle's choices: play it fresh, and say so until the first move
    game = new Game(link.n, players);
    game.notice = `That replay link does not fit battle #${link.n}; it starts fresh`;
  }
  start(link.n, players, link.watch ? 300 : 0, game);
} else {
  (watching ? watch : play)(digits ? +digits : fresh());
}
