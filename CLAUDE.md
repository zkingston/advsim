# advsim: GPU engine for Showdown [Gen 3] Random Battle

NVIDIA Warp battle engine, bit-exact with `pokemon-showdown` 0.11.11, for RL training and test-time search.

- Rules and workflow: `CONTRIBUTING.md`. They are hard rules.
- Design, the source of truth: `docs/SPEC.md` (~110 KB). Read the sections a task touches; do not @-import it.
- What has been measured, and how: `docs/RESULTS.md`. Add new results there.
- Setup and commands: `README.md`.

## Working here

1. Propose a short plan (files, tests, exit criterion, minutes). Wait for approval before writing code.
2. Small commits, one concern each. Run the tests before committing.
3. If the code has to contradict the SPEC: stop, propose the change, and after approval update `docs/SPEC.md` in the same commit.
4. After an engine change, sweep a new seed range against Showdown before calling it done.
5. Report tersely: status, completed items, estimates in minutes, one concrete next step. Report errors flatly.

## Process hygiene

Never `pkill -f` or `pgrep -f` a pattern that appears anywhere in your own shell's command line: pkill kills the shell running it, and an `until ! pgrep -f ...` wait loop waits on itself forever. Build the pattern from variables (`E=elo; pgrep -f "${E}.py"`) and keep the literal name out of the rest of the command.
