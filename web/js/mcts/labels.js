// What the other side's action codes are in one world, for the page's Bot panel: its moves by
// type and category ("Water attack", "Status move"), and its switches by the Pokemon coming in,
// named if the searcher has seen it and by its type otherwise ("Switch to a Water type it hasn't seen").
// A code is a move slot of a sampled set, so slot 3 is a different move in different worlds;
// labels are what the search's root statistics can be tallied by instead.

function moveLabel(world, mon, id) {
  const move = world.dex.moves.get(id);
  if (move.category === 'Status') return 'Status move';
  const type = id === 'hiddenpower' ? mon.hpType : move.type;
  return `${type} attack`;  // gen 3's type decides physical or special, so it says both
}

// Side `o` of `world` at the root: its codes' labels (undefined where a code means nothing).
// `st`: that side's InfoState, whose order the switch codes index; `view`: the searcher's.
export function rootLabels(world, st, o, view) {
  const side = world.sides[o], moves = side.activeRequest?.active?.[0]?.moves || [];
  const seen = view.sides[side.id].mons, out = [];
  for (let a = 0; a < 12; a++) {
    if (a === 10) out[a] = 'Wait';
    else if (a < 4 || a === 11) {
      const m = moves[a === 11 ? 0 : a];
      if (m) out[a] = moveLabel(world, side.active[0], m.id);
    } else {
      const q = side.pokemon.find((x) => x.name === st.sides[st.me].order[a - 4]);
      const type = q?.species.types[0], an = /^[AEIOU]/.test(type) ? 'an' : 'a';
      if (q) out[a] = seen.get(q.name)?.species ? `Switch to ${q.species.name}` : `Switch to ${an} ${type} type it hasn't seen`;
    }
  }
  return out;
}
