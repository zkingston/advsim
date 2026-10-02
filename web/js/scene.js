// The battle scene, drawn from p1's InfoState: the two active Pokemon, updated in place so HP
// bars and faints run as CSS transitions, the weather and terrain, the dialog box, the end
// screen, and each protocol line's animation.
import { CANT, STATUS_NAMES } from './text.js';
import { $, esc, gen3, sprite, speciesName, hpClass, moveType, typeBadges, moveTip, abilityTip, itemTip, tipped } from './dex.js';

// The scene is laid out on a fixed stage and scaled to fit, keeping its shape on any screen.
// Its design size is css/scene.css's (--sw, --sh): offsetWidth/Height, which the transform does not change.
{
  const stage = document.querySelector('.stage'), scene = stage.parentElement;
  new ResizeObserver(() => {
    const f = Math.min(scene.clientWidth / stage.offsetWidth, scene.clientHeight / stage.offsetHeight);
    stage.style.transform = `scale(${f})`;
    stage.dataset.scale = f;
  }).observe(scene);
}

// Gen 3 sprites leave empty rows under the Pokemon, a different number each: measure them once
// per image and move the sprite down by that much, so its feet are on the box's bottom edge
// (the foe's platform, the scene's edge).
const gaps = new Map();
function ground(img) {
  const apply = () => { img.style.translate = `0 ${100 * gaps.get(img.src)}%`; };
  if (gaps.has(img.src)) return apply();
  img.style.translate = '';
  img.onload = () => {
    const w = img.naturalWidth, h = img.naturalHeight, c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const alpha = ctx.getImageData(0, 0, w, h).data;
    let lowest = h - 1;
    find: for (; lowest > 0; lowest--) for (let x = 0; x < w; x++) if (alpha[4 * (lowest * w + x) + 3]) break find;
    gaps.set(img.src, (h - 1 - lowest) / h);
    apply();
  };
}

// ---- the active Pokemon

const CARD = `<div class="card"><div class="spritebox"><img class="sprite" alt=""><img class="doll" alt="Substitute"><div class="spikes"></div></div>
  <div class="info"><div class="name"></div><div class="hprow"><div class="hpbar"><div class="fill"></div></div></div>
  <div class="hptext"></div><div class="held"></div><div class="chips"></div><div class="known"></div><div class="party"></div></div></div>`;

function drawSide(el, side, own) {
  if (!el.firstChild) el.innerHTML = CARD;
  const q = (c) => el.querySelector(c);
  const cells = side.order.map((n) => {
    const m = side.mons.get(n);
    const label = `${speciesName(m.species)}${m.fainted ? ' (fainted)' : m.status ? ` (${STATUS_NAMES[m.status] || m.status})` : ''}`;
    return `<span class="ball ${m.fainted ? 'fnt' : m.status ? 'st' : ''}" title="${esc(label)}" aria-label="${esc(label)}"></span>`;
  });
  while (cells.length < 6) cells.push('<span class="ball unknown" title="not seen yet" aria-label="not seen yet"></span>');
  q('.party').innerHTML = cells.join('');
  const mon = side.mons.get(side.active);
  q('.card').style.visibility = mon ? '' : 'hidden';
  if (!mon) return;
  const img = q('.sprite');
  if (el.dataset.active !== side.active) {
    el.dataset.active = side.active;
    img.style.visibility = '';
    img.onerror = () => { img.style.visibility = 'hidden'; };
    img.src = sprite(mon.species, own);
    ground(img);
    img.classList.remove('recall');
    replay(img, 'enter');
  }
  img.classList.toggle('gone', mon.fainted);
  // A Substitute stands in for the Pokemon, as in the games; Spikes lie on that side's platform.
  const doll = q('.doll');
  if (!doll.src) doll.src = sprite('_substitute', own);
  q('.spritebox').classList.toggle('subbed', !!side.vol.substitute && !mon.fainted);
  q('.spikes').innerHTML = '<i></i>'.repeat(3 * side.spikes);
  const pct = own ? Math.ceil((100 * mon.hp) / Math.max(mon.maxhp, 1)) : mon.hp;
  q('.name').innerHTML = `${esc(speciesName(mon.species))} <span class="lv">L${mon.level}</span> ${typeBadges(side.types)}`
    + (mon.status ? ` <span class="status ${esc(mon.status)}">${esc(mon.status)}</span>` : '');
  const fill = q('.fill');
  fill.style.width = `${pct}%`;
  fill.className = `fill ${hpClass(pct)}`;
  q('.hptext').textContent = own ? `${mon.hp}/${mon.maxhp}` : `${pct}%`;
  // The foe's item only once revealed; a used berry leaves none.
  q('.held').innerHTML = !own && !mon.item_known ? 'item: ?'
    : mon.item ? `item: ${tipped(itemTip(mon.item), gen3.items.get(mon.item).name)}` : 'item: none';
  q('.chips').innerHTML = [
    ...Object.entries(side.boosts).filter(([, v]) => v).map(([k, v]) => `<span class="chip ${v > 0 ? 'up' : 'down'}">${esc(k)} ${v > 0 ? '+' : ''}${v}</span>`),
    ...Object.keys(side.vol).map((k) => `<span class="chip">${esc(k.replace('_', ' '))}</span>`),
  ].join('');
  q('.known').innerHTML = own ? '' : [
    mon.ability_known && `ability: ${tipped(abilityTip(mon.ability), gen3.abilities.get(mon.ability).name)}`,
    mon.moves.size && `moves: ${[...mon.moves].map((m) => tipped(moveTip(m, moveType(m, '')), gen3.moves.get(m).name)).join(', ')}`,
  ].filter(Boolean).join(' · ');
}

export function drawField(st, notice) {
  $('weather').className = ['', 'wx-sun', 'wx-rain', 'wx-sand'][st.weather] || '';
  drawSide($('foe'), st.sides.p2, false);
  drawSide($('own'), st.sides.p1, true);
  $('fieldinfo').textContent = [`Turn ${st.turn}`, ['', 'Sun', 'Rain', 'Sandstorm'][st.weather],
    st.sides.p1.spikes && `Spikes (yours) ${st.sides.p1.spikes}`, st.sides.p2.spikes && `Spikes (foe) ${st.sides.p2.spikes}`,
    notice].filter(Boolean).join(' · ');
}

// A new battle's empty scene: the next draw sends both Pokemon in.
export function clearField() {
  for (const id of ['own', 'foe']) { $(id).innerHTML = ''; delete $(id).dataset.active; }
}

// The scene's terrain, fixed per battle: sky top and bottom, ground, platform and its rim.
const TERRAINS = [
  ['grass', '#f8f8d0', '#d8f0b8', '#c0e098', '#b8d890', '#88b068'],
  ['desert', '#f8f0d0', '#f0e0b0', '#e8d098', '#e0c080', '#b89858'],
  ['cave', '#6a5a4c', '#8a7864', '#8a7a68', '#a89880', '#6a5a48'],
  ['water', '#e0f4fc', '#b0dcf4', '#88c0e8', '#b0dcf4', '#5890c0'],
  ['snow', '#f0f6fc', '#dce8f4', '#dce8f0', '#f8fbff', '#a0b4c8'],
  ['gym', '#ece4f4', '#d8cce8', '#c8b8d8', '#dccdea', '#9880b0'],
  ['night', '#28304c', '#405070', '#486848', '#5a7a50', '#34523a'],
];
export function setTerrain(n) {
  const [, sky1, sky2, ground, plat, rim] = TERRAINS[n % TERRAINS.length];
  document.querySelector('.scene').style.cssText = `--sky1:${sky1};--sky2:${sky2};--ground:${ground};--plat:${plat};--rim:${rim}`;
}

// The dialog box under the scene: the event being played, or the prompt.
export const say = (text) => { $('say').textContent = text; };

// The end screen over the scene; null hides it.
export function showResult(text, won) {
  const el = $('result');
  el.hidden = !text;
  if (!text) return;
  el.className = won ? 'won' : 'lost';
  el.innerHTML = `<div>${esc(text)}</div><button id="again">Battle again</button>`;
  replay(el, 'pop');
}

// ---- events: one protocol line's animation, and how long to hold it (ms at normal speed)

// A one-off animation's class comes off when it ends (or is cancelled by a hidden tab), or
// showing the tab again would replay it. Recall's end state holds until the next sprite.
const KEYFRAMES = { show: 'banner', pop: 'result' };
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth;  // restart the animation
  el.classList.add(cls);
  if (cls === 'recall') return;
  const name = KEYFRAMES[cls] || cls;
  const anim = el.getAnimations().find((a) => a.animationName === name);
  // Unless a later replay restarted it.
  const done = () => { if (!el.getAnimations().some((a) => a.animationName === name && a.playState === 'running')) el.classList.remove(cls); };
  if (anim) anim.finished.then(done, done);  // a hidden tab cancels it, which rejects
  else el.classList.remove(cls);  // not running at all: hidden already
}

const HOLD = { move: 750, '-damage': 650, '-heal': 550, faint: 750, switch: 1000, drag: 1000, '-status': 450,
  '-boost': 400, '-unboost': 400, '-weather': 450, '-start': 400, '-end': 300, '-sidestart': 450, '-item': 400,
  '-enditem': 400, '-ability': 400, '-crit': 400, '-supereffective': 350, '-resisted': 350, '-immune': 400,
  '-miss': 400, '-fail': 600, cant: 800, '-transform': 500, '-activate': 500, win: 500,
  // Long enough to read what a status move did, or why it did nothing (hints stay in the log).
  '-singleturn': 500, '-singlemove': 500, '-message': 900, '-cureteam': 500, '-clearallboost': 500,
  '-sethp': 500, '-setboost': 600, '-fieldactivate': 600, '-sideend': 500, '-prepare': 600, '-hitcount': 400 };

// The banner over a Pokemon's sprite, kept inside the scene.
function banner(text, img) {
  const el = $('banner');
  el.textContent = text;
  // Screen rects are scaled; the banner is placed in the stage's own 600x380 units.
  const stage = el.parentElement, f = +stage.dataset.scale || 1;
  const s = stage.getBoundingClientRect(), r = img.getBoundingClientRect();
  el.style.left = '0px';
  const half = el.offsetWidth / 2;
  const x = Math.min(Math.max((r.left - s.left + r.width / 2) / f, half + 6), stage.offsetWidth - half - 6);
  el.style.left = `${x}px`;
  el.style.top = `${Math.max((r.top - s.top + r.height * 0.25) / f, 24)}px`;
  replay(el, 'show');
}

// Before a switch is drawn: the Pokemon going out flashes red and shrinks into its ball.
// Returns how long that takes (ms at normal speed); 0 when nobody is out to recall (a
// replacement after a faint, or the first send-out).
export function recall(line) {
  const p = line.split('|');
  if (p[1] !== 'switch' && p[1] !== 'drag') return 0;
  const side = $(p[2].startsWith('p1') ? 'own' : 'foe'), img = side.querySelector('.sprite');
  if (!side.dataset.active || !img || img.classList.contains('gone') || img.style.visibility === 'hidden') return 0;
  img.className = 'sprite';  // drop earlier animations, which would otherwise win over this one
  replay(img, 'recall');
  return 600;
}

export function animate(line) {
  const p = line.split('|'), tag = p[1];
  if (tag === '-weather' && line.includes('[upkeep]')) return 0;
  const img = /^p[12]/.test(p[2] || '') ? $(p[2].startsWith('p1') ? 'own' : 'foe').querySelector('.sprite') : null;
  if (!img) return HOLD[tag] || 0;
  if (tag === 'move') {
    replay(img, p[2].startsWith('p1') ? 'lunge-own' : 'lunge-foe');
    banner(p[3], img);
  } else if (tag === 'cant') {
    replay(img, p[3] === 'par' ? 'para' : 'flash');
    banner(CANT[p[3]] ? CANT[p[3]][1] : "Can't move", img);
  } else if (tag === 'switch' || tag === 'drag') {
    // A ball drops in, spins and bursts; the sprite's own 'enter' grows out of the flash.
    const ball = document.createElement('span');
    ball.className = 'throwball';
    img.parentElement.append(ball);
    const gone = () => ball.remove(), anim = ball.getAnimations()[0];
    if (anim) anim.finished.then(gone, gone);  // ended, or cancelled by a hidden tab
    else gone();
  } else if (tag === '-damage') {
    replay(img, line.includes('[from]') ? 'flash' : 'hit');
  } else if (tag === '-heal' || tag === '-boost') {
    replay(img, 'glow');
  } else if (tag === '-unboost' || tag === '-status') {
    replay(img, 'flash');
  }
  return HOLD[tag] || 0;
}
