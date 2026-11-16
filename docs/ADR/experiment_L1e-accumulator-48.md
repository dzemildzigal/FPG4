# ADR-014: The accumulator is 48 bits with one point per layer at min(y)

**Issue**: Block sums have different scales, so they can only be added after their binary points align. The
alignment loses bits. The width of the register that holds the sum, and the place of its binary point,
decide how many bits are lost and how large the register is. Two earlier rules were tried and withdrawn.

**Decision**: The layer accumulator is 48 bits, two's complement, and it has one binary point for the whole
layer, placed at the smallest block scale of the layer, `min(y)`. There is one point, not one point per
term. For the experiments the point is fixed at `24.24`. The width is not raised at run time.

**Status**: Decided, 2026-11-11, commit `6240628`. Specified in `spec.md` v0.13, commit `228254b`.

**Group**: datapath.

**Assumptions**: The weight and activation scales of one layer span a bounded number of octaves, and the
largest block sum is bounded by the payload range.

**Constraints**: The smallest scale of a layer comes from the 49 or 256 scale codes of the layer, so the
width of the span is a function of the block count and not of the data.

**Positions**:
1. 32 bits. Rejected: defeated once the span between the smallest and largest block scale is wide enough.
2. Place the point at the mean of `min(y)` and `max(y)`. Rejected: it splits the headroom between two
   directions and leaves the sum of the worst case too small.
3. Widen the accumulator at run time when a block is large. Rejected: the register path becomes variable,
   and a variable path cannot be pipelined at a fixed latency.
4. 48 bits, one point per layer at `min(y)`.

**Argument**: The width follows from two terms. The growth of one block sum is 8 bits, because `S` is 13
bits and the largest term needs 19 magnitude bits. The distance between the smallest and the largest block
scale is 20 octaves at 256 blocks per layer and 22 at 49 blocks. 8 plus 20 plus headroom lands on 48. The
measured headroom is 1.41 times on MNIST and 1.32 times on MNIST-1D.

**Implications**: One lane costs 260 to 300 LUT6 and 48 flip-flops for the accumulator. Four lanes cost
1,040 to 1,200 LUT6. The shifter that aligns a term to the point costs four times the adder, so the
shifter is the lever on the cost of the accumulator.

**Related decisions**: ADR-003, ADR-012, ADR-013, ADR-015.

**Related requirements**: No run-time reconfiguration of a datapath width.

**Related artifacts**: `docs/spec.md` section 5.7, `docs/datapath.md` section 9, `python/width_probe.py`,
`python/datapath_probe.py`.

**Related principles**: One width, one point, no run-time surprises.

**Notes**: Two earlier rules were written and withdrawn in one week: a rule on the block count `nb`
(commit `54ea76e`, withdrawn) and a rule that the width must hold every input. `python/width_probe.py`
measured the real headroom and the rule came back as a constant.
