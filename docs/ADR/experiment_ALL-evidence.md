# ADR-030: Evidence is never regenerated, and every figure names itself

**Issue**: Two failure modes waste the most time in this project: re-running a sweep that already produced
its data, and hunting for the run behind a figure. Both were experienced. A run of the model takes 23
seconds on MNIST and 167 seconds on MNIST-1D, so a sweep is hours, and a figure without its run directory is
not evidence.

**Decision**:
1. A finished run is never repeated. Each run writes one JSON as soon as it ends, and a sweep skips a run
   whose file exists, so a sweep can be stopped and restarted.
2. Nothing is retrained to draw a figure. A figure reads JSON that exists.
3. Every figure is named `experiment_<experiment id>_<dataset>_<content>.png`. The prefix is the document
   that shows it.
4. Every figure states, in its own footer, its file name, the run directory behind its numbers, and how many
   runs make one curve.
5. A figure is written by a script. No figure is made by hand.

**Status**: Decided. Items 1 and 2 from 2026-11-14, items 3 to 5 on 2026-11-16, commit `be568b1`.

**Group**: tooling, experiment protocol.

**Assumptions**: A run is deterministic given its seed, its data order and its configuration, so a stored
JSON is as good as the run.

**Constraints**: A figure and its document must be committed together, so the plot rule of `.gitignore`
excepts `docs/experiment_*.png`.

**Positions**:
1. Report means only, and re-run when a figure is needed.
2. Store per-run JSON, read it back, and name the figure after the experiment.

**Argument**: The block experiment drew its figure after every arm, so a figure existed while the sweep was
still running. That is only possible when the figure is a function of stored files. The naming rule came from
a real problem: `block_mnist.png` did not say which experiment it belonged to.

**Implications**: `out/` holds the JSON of every run and is not committed. `docs/` holds one figure per
experiment and dataset, and the figure is committed. A mean without its seeds cannot be checked, so the JSON
is the record and the figure is the view.

**Related decisions**: ADR-020, ADR-021, ADR-022.

**Related requirements**: A number that a reader can recompute from the repository.

**Related artifacts**: `.gitignore`, `python/curves.py`, `python/block_sweep.py`, `python/l1_report.py`,
`docs/README.md`, `docs/experiment_block_mnist_curves.png`.

**Related principles**: The repository holds the evidence, not a picture of it.

**Notes**: Restartable sweeps are also why a killed run costs at most one run, and why a sweep can be stopped
with `pkill -f "[b]lock_sweep"` without losing finished work.
