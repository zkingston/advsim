// Everything around the scene: the move buttons, p1's team (whose cards are the switches),
// the battle log, the win-estimate sparkline, and the one floating tooltip.
import { legalMask } from './observation.js';
import { nameOf } from './infostate.js';
import { lines } from './text.js';
import { $, esc, gen3, sprite, hpClass, typeColor, moveId, moveType, catIcon, typeBadges, moveText, moveTip, abilityTip, itemTip, tipped } from './dex.js';

// Hover-only affordances (tooltips, lifts) only where there is a real hover: a tap on a touch
// screen would leave them stuck.
const canHover = matchMedia('(hover: hover) and (pointer: fine)');
const markHover = () => document.documentElement.classList.toggle('hover', canHover.matches);
markHover();
canHover.addEventListener('change', markHover);

// One floating tooltip for every [data-tip], placed in the page so no panel clips it.
const tipEl = document.createElement('div');
tipEl.id = 'tip';
tipEl.hidden = true;
document.body.append(tipEl);
document.addEventListener('mouseover', (e) => {
  if (!canHover.matches) return;
  const el = e.target.closest?.('[data-tip]');
  if (!el) { tipEl.hidden = true; return; }
  tipEl.textContent = el.dataset.tip;
  tipEl.hidden = false;
  const r = el.getBoundingClientRect(), t = tipEl.getBoundingClientRect();
  const left = Math.min(Math.max(8, r.left), innerWidth - t.width - 8);
  const top = r.top - t.height - 6 >= 8 ? r.top - t.height - 6 : r.bottom + 6;
  tipEl.style.left = `${left + scrollX}px`;
  tipEl.style.top = `${top + scrollY}px`;
});
document.addEventListener('mouseout', (e) => { if (!e.relatedTarget) tipEl.hidden = true; });

export function drawLog(log, cursor) {
  $('log').innerHTML = lines(log, cursor, 'p1').map((l) => (l.turn ? `<h3>Turn ${l.turn}</h3>`
    : `<p class="${l.dim ? 'dim' : ''}${l.strong ? ' strong' : ''}">${esc(l.text)}</p>`)).join('');
  $('log').scrollTop = $('log').scrollHeight;
}

// The bots' win estimates over the battle: each one's value head per decision as a probability
// of its own win (both when watching, one line per bot), a dashed even-odds line, a dot at
// decision k. Hovering reads a decision off; clicking jumps the slider there (onJump).
// `evals[i]`: [p1 bot's, p2 bot's] at view i, null where none; `names`: the two players.
export function drawSpark(evals, k, onJump, names) {
  const box = $('spark');
  const series = [0, 1].map((p) => ({ p, pts: evals.map((e, i) => [i, e?.[p]]).filter(([, v]) => v != null) }))
    .filter((x) => x.pts.length);
  box.hidden = !series.length;
  if (!series.length) return;
  const n = Math.max(evals.length - 1, 1), X = (i) => (100 * i) / n, Y = (v) => 100 - 50 * (v + 1);
  const pct = (v) => `${Math.round(50 * (v + 1))}%`;
  const at = (pts, i) => pts.filter(([j]) => j <= i).at(-1) || pts[0];
  const one = series.length === 1;
  const key = (x, i) => (one ? `<b>${pct(at(x.pts, i)[1])}</b>` : `<span class="key s${x.p}"><i></i>${esc(names[x.p])} <b>${pct(at(x.pts, i)[1])}</b></span>`);
  const head = (i, where = '') => (one ? `<span>Bot's win estimate</span><span>${key(series[0], i)}${where}</span>`
    : `<span class="keys">${series.map((x) => key(x, i)).join('')}</span><span>${where}</span>`);
  box.innerHTML = `<div class="spark-head">${head(k)}</div>
    <div class="spark-plot"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" x2="100" y1="50" y2="50" class="even"/>${series.map((x) => `<polyline class="s${x.p}" points="${x.pts.map(([i, v]) => `${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join(' ')}"/>`).join('')}</svg>
      ${series.map((x) => { const [i, v] = at(x.pts, k); return `<span class="mark s${x.p}" style="left:${X(i)}%;top:${Y(v)}%"></span>`; }).join('')}</div>`;
  const plot = box.querySelector('.spark-plot'), headEl = box.querySelector('.spark-head');
  const nearest = (e) => {
    const r = plot.getBoundingClientRect();
    return Math.min(n, Math.max(0, Math.round(((e.clientX - r.left) / r.width) * n)));
  };
  plot.onmousemove = (e) => { const i = nearest(e); headEl.innerHTML = head(i, ` at decision ${i}`); };
  plot.onmouseleave = () => { headEl.innerHTML = head(k); };
  plot.onclick = (e) => onJump(nearest(e));
}

// ---- the human's choices

// A damaging move against the foe's current types, as Showdown's chart has it: '×2', '×½',
// '×0' and so on, or '' for neutral, status moves and no foe. Fixed-damage moves (Seismic
// Toss, Night Shade) only have the immunity.
function effect(id, type, foe) {
  const move = gen3.moves.get(id);
  const types = [...new Set(foe.types || [])].filter((t) => t !== '???');  // mono types come as [t, t]
  if (!types.length || move.category === 'Status') return '';
  if (!gen3.getImmunity(type, types)) return '×0';
  if (move.damage || move.ohko) return '';
  return { 2: '×4', 1: '×2', [-1]: '×½', [-2]: '×¼' }[gen3.getEffectiveness(type, types)] || '';
}

const aiTab = (p) => `<span class="ai">${Math.round(100 * p)}%</span>`;

// `assist`: the network's probability for each of the 12 action codes, or null. Moves are
// buttons under the dialog; switches are the team panel's cards, clickable while that switch
// is legal. Without onPick both go inert. `readonly` (watching the bots): the moves are shown
// for what they are, and nothing is picked.
export function drawControls(st, req, onPick, assist = null, readonly = false) {
  const box = $('controls');
  box.innerHTML = '';
  for (const card of document.querySelectorAll('.mate')) {
    card.classList.remove('can-switch', 'ai-top');
    card.onclick = card.onkeydown = null;
    for (const a of ['tabindex', 'role', 'title']) card.removeAttribute(a);
    card.querySelector('.ai')?.remove();
  }
  if (!req || (!onPick && !readonly)) return;
  const mask = legalMask(st), me = st.sides.p1;
  const top = assist ? assist.indexOf(Math.max(...assist)) : -1;
  const button = (code, html, cls, style = '', tipText = '') => {
    const b = document.createElement('button');
    b.innerHTML = html; b.className = cls; b.style.cssText = style;
    if (tipText) b.dataset.tip = tipText;
    b.disabled = !(mask & (1 << code));
    if (assist && !b.disabled) {
      b.insertAdjacentHTML('beforeend', aiTab(assist[code]));
      if (code === top) b.classList.add('ai-top');
    }
    if (readonly) { b.classList.add('readonly'); b.tabIndex = -1; } else b.onclick = () => { box.innerHTML = ''; onPick(code); };
    box.append(b);
  };
  if (mask & (1 << 11) && !readonly) button(11, 'Continue', 'move');
  else if (req.active) {
    req.active[0].moves.forEach((m, i) => {
      const type = moveType(m.id, me.mons.get(me.active).hp_type);
      const eff = effect(m.id, type, st.sides.p2);
      const kind = eff === '×0' ? 'none' : eff === '×2' || eff === '×4' ? 'good' : 'bad';
      const tag = eff ? `<span class="eff">${eff}</span>` : '';
      button(i, `${esc(gen3.moves.get(m.id).name)}${tag}<small>${catIcon(m.id, type)} ${esc(type)} · ${m.pp ?? '-'}/${m.maxpp ?? '-'}</small>`, `move${eff ? ` eff-${kind}` : ''}`,
        `--type:${typeColor(type)}`, moveText(m.id, type));
    });
  }
  if (req.forceSwitch && !readonly) box.innerHTML = '<p class="pick">Choose a Pokémon from your team below.</p>';
  for (const card of document.querySelectorAll('.mate')) {
    const code = 4 + me.order.indexOf(card.dataset.name);
    if (code < 4 || !(mask & (1 << code))) continue;
    if (readonly) {  // watching: the assist's share, and nothing to click
      if (assist) {
        card.insertAdjacentHTML('beforeend', aiTab(assist[code]));
        if (code === top) card.classList.add('ai-top');
      }
      continue;
    }
    const pick = () => { drawControls(st, null, null); onPick(code); };
    card.classList.add('can-switch');
    card.setAttribute('role', 'button');
    card.tabIndex = 0;
    card.title = 'Switch in';
    card.onclick = pick;
    card.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
    if (assist) {
      card.insertAdjacentHTML('beforeend', aiTab(assist[code]));
      if (code === top) card.classList.add('ai-top');
    }
  }
}

// ---- p1's team, from its latest request

// Shrink a card whose moves overflow (only a phone's fixed-height team does): moves two by
// two, then item and ability on one line. Measured, since the space depends on the screen.
function fitTeam() {
  for (const card of document.querySelectorAll('.mate')) {
    card.classList.remove('tight', 'tighter');
    // The last move's bottom against the list's: scrollHeight does not count the clipped rows here.
    const moves = card.querySelector('.moves');
    const fits = () => !moves.lastElementChild || moves.lastElementChild.getBoundingClientRect().bottom <= moves.getBoundingClientRect().bottom + 1;
    // A narrow card tries tighter lines before two columns, whose names it can barely show.
    for (const cls of card.clientWidth < 170 ? ['tighter', 'tight'] : ['tight', 'tighter']) {
      if (fits()) break;
      card.classList.add(cls);
    }
  }
}
new ResizeObserver(fitTeam).observe($('teamgrid'));

// Where a stat's bar is full: near the 99th percentile of the pool's non-HP stats
// (median ~175, p99 263-366); Shuckle's 506 defenses fill it.
const STAT_SCALE = 350;

function mateCard(p) {
  const [species, ...details] = p.details.split(', ');
  const level = (details.find((d) => /^L\d+$/.test(d)) || 'L100').slice(1);
  const [hpText, status] = p.condition.split(' ');
  const [hp, maxhp] = hpText.split('/').map(Number);
  const pct = status === 'fnt' ? 0 : Math.ceil((100 * hp) / (maxhp || 1));
  const hpType = (p.moves.find((m) => /^hiddenpower[a-z]+/.test(m)) || '').replace(/^hiddenpower([a-z]+)\d*$/, '$1');
  const moves = p.moves.map((m) => {
    const id = moveId(m), type = moveType(id, hpType);
    return `<li class="tipped" ${moveTip(id, type)}>${catIcon(id, type)}<span class="dot" style="background:${typeColor(type)}"></span>${esc(gen3.moves.get(id).name)}${id === 'hiddenpower' ? ` <small>${esc(type)}</small>` : ''}</li>`;
  }).join('');
  // Each stat over a bar of its size against STAT_SCALE.
  const stats = Object.entries(p.stats).map(([k, v]) => `<div class="stat"><i style="height:${Math.min(100, (100 * v) / STAT_SCALE).toFixed(0)}%"></i>`
    + `<b>${v}</b><span>${k.toUpperCase()}</span></div>`).join('');
  const item = p.item ? tipped(itemTip(p.item), gen3.items.get(p.item).name) : 'no item';
  const ailment = status && status !== 'fnt' ? `<span class="status ${esc(status)}">${esc(status)}</span>` : '';
  return `<div class="mate ${p.active ? 'active' : ''} ${status === 'fnt' ? 'fnt' : ''}" data-name="${esc(nameOf(p.ident))}">
      <div class="mate-head">
        <img alt="" src="${sprite(gen3.species.get(species).id, false)}" onerror="this.style.visibility='hidden'">
        <div><div class="name">${esc(species)}</div>
          <div class="types"><span class="lv">L${level}</span>${typeBadges(gen3.species.get(species).types, true)}</div>
          <div class="hprow"><div class="hpbar"><div class="fill ${hpClass(pct)}" style="width:${pct}%"></div></div>${ailment}
            <span class="hptext">${status === 'fnt' ? 'fainted' : `${hp}/${maxhp}`}</span></div>
          <div class="meta"><span>${item}</span><span class="sep"> · </span><span>${tipped(abilityTip(p.baseAbility), gen3.abilities.get(p.baseAbility).name)}</span></div></div>
      </div>
      <ul class="moves">${moves}</ul>
      <div class="stats">${stats}</div></div>`;
}

export function drawTeam(req, label) {
  $('teamlabel').textContent = label;
  if (!req || !req.side) return;
  $('teamgrid').innerHTML = req.side.pokemon.map(mateCard).join('');
  fitTeam();
}
