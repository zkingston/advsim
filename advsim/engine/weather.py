"""Weather as the Pokemon on the field see it: Cloud Nine and Air Lock switch it
off, and Castform's Forecast changes forme to match."""
import warp as wp

from advsim.engine import mask as mask_
from advsim.engine import mon
from advsim.engine.effects import ability_fx
from advsim.engine._generated import ids
from advsim.engine._generated.state import State
from advsim.engine.dex import Dex


@wp.func
def effective_weather(s: State, dex: Dex, b: int) -> int:
    """Cloud Nine and Air Lock switch the weather off for everyone while they
    are in. One that has gone down no longer counts: `suppressingWeather` skips
    a fainted Pokemon, and its End event has already said so."""
    for side in range(2):
        slot = int(s.active[b, side])
        if int(s.hp[b, side, slot]) > 0 and \
                ability_fx.suppresses_weather(dex, int(s.ability[b, side, slot])):
            return 0
    return int(s.weather[b])


@wp.func
def forecast(s: State, dex: Dex, b: int, side: int):
    """Castform takes the weather's type, and Normal when there is none it can
    feel. It reads the effective weather, so Air Lock turns it back."""
    slot = int(s.active[b, side])
    if not ability_fx.has(dex, int(s.ability[b, side, slot]), ids.ABILITYFAM_FORECAST):
        return
    if int(s.species[b, side, slot]) != ids.SPECIES_CASTFORM:
        return  # it is a forme change, so a traced Forecast changes nothing
    if (int(s.vflags[b, side]) & mask_.VF_TRANSFORMED) != 0:
        return
    kind = effective_weather(s, dex, b)
    become = ids.TYPE_NORMAL
    if kind == ids.WEATHER_SUN:
        become = ids.TYPE_FIRE
    elif kind == ids.WEATHER_RAIN:
        become = ids.TYPE_WATER
    if int(s.types[b, side, 0]) != become:
        # A real forme change goes through setSpecies, which writes the raw
        # Speed back into the cache: a paralysed Castform sorts at full Speed
        # until the next updateSpeed.
        s.cached_spe[b, side, slot] = s.stats[b, side, slot, 4]
        mon.reveal(s, b, side, slot, mon.REVEAL_ABILITY)  # `-formechange ... [from] ability: Forecast`
    s.types[b, side, 0] = wp.uint8(become)
    s.types[b, side, 1] = wp.uint8(become)
