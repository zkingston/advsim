// The Settings panel and the page's chrome: what is kept in the browser, the opponent and its
// parameters, the tabs (the record panel's three, a phone's five), the replay link, About.
import { $, esc } from './dex.js';

// localStorage, which can be blocked (a private window): reads fall back, writes are dropped.
const store = {
  get(key) { try { return localStorage.getItem(`advsim.${key}`); } catch { return null; } },
  set(key, value) {
    try { if (value == null) localStorage.removeItem(`advsim.${key}`); else localStorage.setItem(`advsim.${key}`, value); } catch { /* blocked */ }
  },
};

// The players the page offers, in the menu's order; the first is the default.
export const OPPONENTS = {
  mcts: ['MCTS', 'The PPO network guiding a Monte Carlo tree search over worlds sampled from what it has seen of your team.'],
  ppo: ['PPO', 'The PPO network, sampling at the chosen temperature: 0 always plays its top choice, higher is more varied.'],
  emerald: ['Emerald-style AI', 'The strongest-looking move every turn, and the Emerald cartridge\'s own logic for who comes in after a faint. It never switches otherwise.'],
  random: ['Random', 'Any legal choice, uniformly.'],
};

// The engine's best search: 512 descents (64 iterations of 8 lanes) to depth 6, prior weight
// 0.25, actions under 2% of the best prior pruned, the final move mixed with the policy at
// temperature 0.25 (search-64-pnet-t25-mix-g25-prune02 in the bracket, docs/RESULTS.md).
const MCTS_DEFAULTS = { descents: 512, seconds: 3, depth: 6, gamma: 0.25, prune: 0.02, final: 'mix',
  workers: Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) };
const storedMcts = () => { try { return JSON.parse(store.get('mcts') || '{}'); } catch { return {}; } };

// The live settings. `onAssist` is app.js's: it redraws a pending choice when assist flips.
export const settings = {
  assist: store.get('assist') === '1',  // off unless the viewer turned it on
  speed: store.get('speed') || 'normal',
  mcts: { ...MCTS_DEFAULTS, ...storedMcts() },
  onAssist: null,
  get opponent() { return $('opponent').value; },
  get temperature() { return +$('temp').value; },
};

$('assist').checked = settings.assist;
$('assist').onchange = () => {
  settings.assist = $('assist').checked;
  store.set('assist', settings.assist ? '1' : '0');
  settings.onAssist?.();
};
$('speed').value = settings.speed;
$('speed').onchange = () => { settings.speed = $('speed').value; store.set('speed', settings.speed); };
$('temp').oninput = () => { $('tempval').textContent = settings.temperature.toFixed(2); };

$('opponent').innerHTML = Object.entries(OPPONENTS).map(([v, [label, about]]) => `<option value="${v}" title="${esc(about)}">${esc(label)}</option>`).join('');
export function setOpponent(name) {
  if (name) $('opponent').value = name;
  $('templabel').hidden = !['ppo', 'mcts'].includes(settings.opponent);
  $('mctsopts').hidden = settings.opponent !== 'mcts';
}
$('opponent').onchange = () => setOpponent();
setOpponent();

// The MCTS inputs: each is named after its setting, and clamped to its own min and max.
const mctsInputs = document.querySelectorAll('#mctsopts [name]');
const showMcts = () => { for (const input of mctsInputs) input.value = settings.mcts[input.name]; };
for (const input of mctsInputs) {
  input.onchange = () => {
    const x = input.type === 'number' ? Math.min(Math.max(+input.value, +input.min), +input.max) : input.value;
    settings.mcts = { ...settings.mcts, [input.name]: x };
    input.value = x;
    store.set('mcts', JSON.stringify(settings.mcts));
  };
}
$('mctsreset').onclick = () => { settings.mcts = { ...MCTS_DEFAULTS }; showMcts(); store.set('mcts', null); };
showMcts();

// ---- tabs

// The record panel's: the battle log, the bot's reasoning (thinking.js), or the settings.
const RECORD = { 'tab-log': 'logpanel', 'tab-bot': 'botpanel', 'tab-settings': 'settings' };
for (const tab of Object.keys(RECORD)) {
  $(tab).onclick = () => {
    for (const [t, panel] of Object.entries(RECORD)) {
      $(t).setAttribute('aria-selected', String(t === tab));
      $(panel).hidden = t !== tab;
    }
    $('replay').hidden = tab === 'tab-settings';
  };
}

// Phones: no page scroll. The views are tabs (Battle, Team, Log, Bot, Settings) over one screen,
// and the header's controls sit in Settings. The media query is css/phone.css's.
export const phone = matchMedia('(max-width: 900px), (orientation: landscape) and (max-height: 500px)');
export function setTab(name) {
  document.body.dataset.tab = name;
  for (const b of document.querySelectorAll('.apptabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  if (['log', 'bot', 'settings'].includes(name)) $(`tab-${name}`).click();
}
// On a phone, the battle's own tab: after starting one, or picking a switch from the team.
export const toBattle = () => { if (phone.matches) setTab('battle'); };
for (const b of document.querySelectorAll('.apptabs button')) b.onclick = () => setTab(b.dataset.tab);
const bar = document.querySelector('.bar');
const placeBar = () => (phone.matches ? $('settings').prepend(bar) : document.querySelector('header').append(bar));
placeBar();
phone.addEventListener('change', placeBar);
setTab('battle');

// ---- the replay link (the address bar's, which app.js keeps current), and About

$('copylink').onclick = async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    $('copylink').textContent = 'Copied!';
  } catch {
    $('linkbox').hidden = false;  // no clipboard access: show the link to copy by hand
    $('linkbox').value = location.href;
    $('linkbox').select();
  }
  setTimeout(() => { $('copylink').textContent = 'Copy replay link'; }, 1500);
};
$('aboutbtn').onclick = () => $('about').showModal();
$('about').onclick = (e) => { if (e.target === $('about')) $('about').close(); };  // the backdrop
