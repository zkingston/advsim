// Gen 3 data as the page shows it: names, types and their colours, move categories, and the
// tooltip text (web/data/text.json, showdown/web_text.js). Page only: the MCTS worker has no
// window, so nothing it imports may import this.
export const gen3 = window.Showdown.Dex.mod('gen3');
export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

const TYPE_COLORS = { Normal: '#9a9a78', Fire: '#e0702a', Water: '#5b83e6', Electric: '#d9b411', Grass: '#5fa83a',
  Ice: '#6cbcbc', Fighting: '#b0302a', Poison: '#94409a', Ground: '#c9a24e', Flying: '#8f7ae0', Psychic: '#e5507f',
  Bug: '#98a51c', Rock: '#aa9636', Ghost: '#6a5594', Dragon: '#6a3cf0', Dark: '#6a5446', Steel: '#a5a5c0' };
export const typeColor = (t) => TYPE_COLORS[t] || '#888';
export const sprite = (species, back) => `sprites/gen3${back ? '-back' : ''}/${species}.png`;  // showdown/fetch_sprites.js
export const speciesName = (id) => gen3.species.get(id).name || id;
export const hpClass = (pct) => (pct > 50 ? 'hi' : pct > 20 ? 'mid' : 'lo');

// Request move ids carry extras: return102, hiddenpowerghost70.
export const moveId = (m) => (m.startsWith('hiddenpower') ? 'hiddenpower' : m.replace(/\d+$/, ''));
const cap = (s) => s.replace(/^./, (c) => c.toUpperCase());
export const moveType = (id, hpType) => (id === 'hiddenpower' ? cap(hpType || 'normal') : gen3.moves.get(id).type);
// Gen 3's category: Status, or Physical or Special by the move's type.
const SPECIAL = new Set(['Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Dark', 'Psychic', 'Dragon']);
const category = (id, type) => (gen3.moves.get(id).category === 'Status' ? 'Status' : SPECIAL.has(type) ? 'Special' : 'Physical');
export const catIcon = (id, type) => { const c = category(id, type); return `<img class="cat" src="sprites/categories/${c}.png" alt="${c}" title="${c}">`; };

// `short`: three letters (unique across the 17 types), the full name on hover.
export const typeBadges = (types, short) => [...new Set(types || [])].filter((t) => t && t !== '???')
  .map((t) => `<span class="type" style="background:${typeColor(t)}"${short ? ` title="${esc(t)}"` : ''}>${esc(short ? t.slice(0, 3).toUpperCase() : t)}</span>`).join('');

// ---- tooltips: [data-tip] text, which panels.js shows on hover

let TEXT = { moves: {}, abilities: {}, items: {} };
export const setText = (t) => { TEXT = t; };
const tip = (text) => `data-tip="${esc(text)}"`;
export function moveText(id, type) {
  const m = TEXT.moves[id] || {}, cat = category(id, type);
  const facts = [`${type} · ${cat}`, cat !== 'Status' && `power ${m.power || 'varies'}`,
    `accuracy ${m.accuracy ? `${m.accuracy}%` : '—'}`, m.priority && `priority ${m.priority > 0 ? '+' : ''}${m.priority}`];
  return `${gen3.moves.get(id).name}: ${facts.filter(Boolean).join(', ')}\n${m.desc || ''}`;
}
export const moveTip = (id, type) => tip(moveText(id, type));
export const abilityTip = (id) => tip(`${gen3.abilities.get(id).name}: ${TEXT.abilities[id] || ''}`);
export const itemTip = (id) => tip(`${gen3.items.get(id).name}: ${TEXT.items[id] || ''}`);
// A name with its tooltip.
export const tipped = (tipAttr, label) => `<span class="tipped" ${tipAttr}>${esc(label)}</span>`;
