You are writing a `generate_input()` function for ARC-AGI-2 task `{task_id}`.

## Your inputs

1. **Spec** (human-reviewed):
```yaml
{spec_yaml}
```

2. **Solver code**: Read `{solver_path}` — contains the `main(grid)` function. Your `generate_input()` will be appended to this file.

3. **Eval I/O data**: Read `{eval_path}` — reference examples for the task to understand unspecified details if necessary, though this should seldom be needed.

4. **Utility functions** from `arc_utils` — **use these instead of reimplementing**:

**Placement** (use these for placing sprites on grids):
- `paste(dst, patch, r0, c0, *, transparent=None, mask=None, overwrite=True, clip=False)` — place patch at (r0,c0)
- `random_free_location(canvas, patch, *, transparent, border=0, padding=0, padding_connectivity=8) -> (r,c)` — random non-overlapping position; raises ValueError if none
- `first_free_location(canvas, patch, *, transparent, border=0, padding=0, padding_connectivity=8) -> (r,c)|None`

**Connected components**:
- `find_components(grid, *, bg=None, connectivity=4, monochromatic=True, return_format="dict") -> list[dict]` — each dict has keys: color, coords, bbox, area
- `components_mask(mask, connectivity=4) -> list[np.ndarray]` — list of (N,2) coord arrays from bool mask
- `label_mask(mask, connectivity=4) -> (labeled_array, num_components)`

**Colors & background**:
- `get_bg(grid) -> int` — most common color
- `colors_present(grid, *, exclude=None) -> list[int]`

**Bounding box** (`BBox = (r0, c0, r1, c1)` inclusive):
- `bbox(mask) -> BBox|None`, `bbox_coords(coords) -> BBox`, `crop_bbox(grid, bbox) -> np.ndarray`

**Geometry**:
- `draw_line(grid, r, c, *, end_r=None, end_c=None, length=None, direction=None, color, stop_at=None, clip=False) -> (r,c)`
- `flood_fill(grid, r, c, color, *, connectivity=4)` — in-place
- `is_contiguous(bitmask, background=0, connectivity=4) -> bool`

Full source: `{arc_utils_path}`

## Pre-check: validate solver on eval data

Before writing any code, verify that the solver works correctly on ALL eval examples:

```bash
cat > /tmp/pre_check.py << 'EOF'
import json, numpy as np
from solvers.arc2 import load_pid
mod = load_pid('{task_id}')
data = json.load(open('{eval_path}'))
for split in ('train', 'test'):
    for i, ex in enumerate(data[split]):
        inp = np.array(ex['input'], dtype=int)
        expected = np.array(ex['output'], dtype=int)
        actual = mod.main(inp)
        ok = np.array_equal(actual, expected)
        print(f'{{split}}[{{i}}]: {{"OK" if ok else "FAIL"}}')
        if not ok:
            raise SystemExit(f'Solver fails on {{split}}[{{i}}] — aborting')
print('All eval examples pass.')
EOF
PYTHONPATH=/home/xd/projects/BARC /home/xd/miniconda3/envs/tune/bin/python /tmp/pre_check.py
```

**If any example fails, STOP immediately. Do not attempt to fix the solver or other files. Report the failure and exit.**

## Your task

Write a `generate_input() -> np.ndarray` function that follows the `generation_strategy` in the spec.

## Requirements

1. **Follow the spec faithfully**. The spec's `generation_strategy` describes the approach — implement it, don't redesign it.
2. **Constructive, not rejection-based**: Generation logic must directly construct objects that satisfy constraints, not randomly generate candidates and filter. For example, if a sprite needs a 2-pixel contact edge, grow it from those 2 pixels — don't generate random sprites and hope one matches. The only acceptable rejection loop is the outer `_gen_attempt` retry + final `main(inp) == out` oracle check. Any inner loop with > 50% rejection rate means the approach is wrong — restructure to construct the desired output directly.
3. **Output-first means output-first in code structure**: When the generation_strategy in spec says "output-first" approach, your code must literally follow the generation_strategy steps to build the full output first (typically the more constrained structure), then derive the input (the easier thing) from it by undoing the solver's transformation described in `main()`. Finally, validate with `np.array_equal(main(input), output)`. You are doing it wrong if your code places input objects first and hopes the solver produces a good output by random trial, which often has near-zero acceptance rate.
4. **Read the solver code** to understand exact constraints, tie-break logic, and edge cases if necessary.
5. **Use `arc_utils` helpers** where possible.
6. **Code structure**: split into `_gen_attempt() -> np.ndarray | None` (single attempt, returns grid or None on failure) and `generate_input()` (retry loop, max 300 attempts). This keeps generation logic flat and readable.configuration knobs that control difficulty or scenario type (e.g. grid size, object count, reachable-vs-stuck, transformation variant) must be sampled ONCE in `generate_input()` and passed as parameters to `_gen_attempt()`. Do NOT re-sample them inside `_gen_attempt()` — when each retry re-rolls the configuration, easy configurations succeed disproportionately often and dominate the output distribution, while hard configurations are almost never produced.
7. **Do not modify any existing code or comments** in the solver file. Append your function at the end.
8. **Spec compliance check**: Before testing, review EVERY property and constraint in the spec (both `spec:` and `generation_strategy:` sections) line by line. For each one, identify the exact code location that implements it. If any spec item has no corresponding code, implement it before proceeding to testing. Pay special attention to:
   - Conditional transformation rules (if-then branches in the spec)
   - Constraints that prevent solver ambiguity (e.g. uniqueness of matching)
   - Features that the oracle check `main(input)==output` cannot detect as missing (e.g. a transformation variant that is valid but never generated)

## Testing

After writing the function, test it. **Important**: always write test/audit scripts to a temp file and run it, never use `python -c "..."` with multi-line code (it triggers sandbox permission prompts that block execution).

```bash
cat > /tmp/test_gen.py << 'EOF'
import numpy as np, time
from solvers.arc2 import load_pid
mod = load_pid('{task_id}')
main, generate_input = mod.main, mod.generate_input
ok, fail = 0, 0
t0 = time.time()
for i in range(200):
    try:
        inp = generate_input()
        out = main(inp)
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 5: print(f'Trial {{i}}: {{e}}')
elapsed = time.time() - t0
print(f'Success: {{ok}}/200, Time: {{elapsed:.1f}}s ({{elapsed/200*1000:.0f}}ms/call)')
EOF
PYTHONPATH=/home/xd/projects/BARC /home/xd/miniconda3/envs/tune/bin/python /tmp/test_gen.py
```

If success rate < 90%, diagnose the failures and fix the code. Re-test until ≥ 90%.

### Quality audit

After achieving ≥ 90% success rate, audit 100 generated samples against the spec:

1. **Range constraints** (e.g. `count: 2-5`, `size: 3-7 pixels`): measure the actual distribution in your 100 samples. Compute the **Jensen-Shannon distance** (`scipy.spatial.distance.jensenshannon(p, q, base=2)`) between the generated distribution and a **uniform distribution** over the spec range. **Flag** any property with JS distance > 0.5. The goal is **diversity** — generated data should explore the full spec range, not cluster at one value.
2. **Absolute constraints** (e.g. `count: 3`, `single connected component`, `no overlap`): extract ALL non-`[observed]` absolute constraints from the spec, verify that **100% of samples** satisfy each one. Flag any violation. These are hard requirements — even a single violation means the generator is buggy.
3. **Visual self-check (MUST NOT skip)**: save a PNG comparing 4-5 generated input-output pairs along with all eval examples (use matplotlib imshow with colors `['#000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAA','#F012BE','#FF851B','#7FDBFF','#870C25']` for values 0-9, eval pairs on top rows, generated pairs on bottom). Read the image and inspect shapes, positions, spatial relationships etc. - generated data should look like it belongs in the same task as eval data. This catches issues that programmatic checks miss. Revise code to reduce significant visual discrepancies as much as you can and report any remaining.
4. Print a summary table (property, type, result, flag status) as a **self-check report**.
5. **Transformation variant coverage**: If the spec describes multiple transformation paths (e.g. "rotation only" vs "flip + color swap", or "horizontal" vs "vertical"), measure what fraction of generated samples exercise each path. Flag any path with 0% coverage — this usually means the generator code is missing that branch entirely. Each variant should appear in at least 10% of samples unless the spec marks it as rare.

### Handling audit failures

When a property is flagged (JS > 0.5), determine which case applies:

- **Code defect**: the code fails to cover a reasonable spec range → fix the code.
- **Spec error**: the spec range contradicts the task's structural constraints (e.g. symmetry excludes even dimensions) → correct or delete the spec entry.
- **Spec over-constraint**: the constraint is unrelated to the task's core logic, and satisfying it would add disproportionate code complexity → correct or delete the spec entry.

Spec entries follow Occam's razor: keep only constraints necessary for solver correctness or core-logic coverage. A constraint that cannot justify its existence should be deleted, not patched.

**`[observed]` constraints**: Spec entries marked `[observed]` are patterns seen in eval data but not required by the solver. Satisfy them if low-cost (e.g. passing a `symmetry` parameter to an existing helper), but do not write significant extra code (> ~50 lines) to enforce them. They should never be audited — exclude them from JS distance checks.

After modifying the spec: output the spec diff with rationale, re-audit against the updated spec, and iterate until all properties pass.

## Common tips / pitfalls

- If the solver infers colors by frequency (e.g., bg = most common), ensure generated grids maintain the same frequency ordering.
- Avoid scenes where multiple valid solutions exist — the solver's heuristic may pick a different one than your expected output.
- If the solver assumes every object participates in the transformation, verify all generated objects are reachable/active.

## Output

Append `generate_input()` (and any helper functions it needs) to `{solver_path}`.
