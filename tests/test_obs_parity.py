"""M5's exit half: the live converter matches the obs kernel on played games."""
import pytest
from advsim import oracle
from advsim.live.infostate import InfoState
from advsim.live.observation import FORCED, Vocab, legal_mask
from advsim.live.parity import compare
from advsim.replay import replay


@pytest.mark.oracle
def test_the_converter_matches_the_kernel():
    cases = oracle.turn_cases(20, seed=77, turns=60, policy='mix', protocol=True)
    bad, examples = compare(cases, replay)
    assert not bad, f'{dict(bad)}; first: {examples[:3]}'


@pytest.mark.oracle
def test_a_one_move_set_out_of_pp_is_forced_to_struggle():
    """Unown with its Hidden Power spent: the request offers only Struggle,
    which is the forced action, not move slot 0 (seed 94001, battle 180)."""
    mon = lambda name, moves, active: {
        'ident': f'p2: {name}', 'details': f'{name}, L84', 'condition': '258/258 par', 'active': active,
        'item': '', 'baseAbility': 'levitate', 'moves': moves, 'stats': {}}
    st = InfoState('p2', Vocab().species_types)
    st.take_request({'active': [{'moves': [{'move': 'Struggle', 'id': 'struggle', 'target': 'randomNormal',
                                            'disabled': False}]}],
                     'side': {'pokemon': [mon('Unown', ['hiddenpowerrock'], True),
                                          mon('Snorlax', ['bodyslam'], False)]}})
    st.sides['p2'].active = 'Unown'
    assert legal_mask(st) == (1 << FORCED) | (1 << 5)
