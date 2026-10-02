import { nameOf } from './infostate.js';

// An action code as a Showdown choice: 0-3 a move slot, 4-9 a switch to that Pokemon in
// the player's first-request order (InfoState.order), 10 pass (null), 11 the forced move.
// Showdown counts switches in its current party array, which a switch reorders.
export function command(code, st, request) {
  if (code === 10) return null;
  if (code === 11) return 'move 1';
  if (code < 4) return `move ${code + 1}`;
  const name = st.sides[st.me].order[code - 4];
  return `switch ${request.side.pokemon.findIndex((p) => nameOf(p.ident) === name) + 1}`;
}
