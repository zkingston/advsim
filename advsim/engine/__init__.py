"""The battle engine: pure Warp, no NumPy, no I/O.

`kernels` and `kernels_search` hold every kernel; `layout` defines the battle
state, `obs_layout` the observation, `mask` the action codes and flags. The
other modules are `@wp.func`s those kernels inline. `advsim.env` (and
`advsim.search`) allocate the arrays and launch the kernels; use those rather
than this package directly.
"""
