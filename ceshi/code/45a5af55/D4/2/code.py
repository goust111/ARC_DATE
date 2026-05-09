# Variant of 45a5af55
# Dimension: D4 — 中心填充方式 (transform)
# Change: 中心为实心正方形（用最底层色条颜色完全填充）... → 中心用最底层和次底层色条颜色交替条纹填充

from __future__ import annotations

import numpy as np

"""
Design analysis for the D4 variant

2) Reference task
- source_task is None, so there is no extra solver to inspect. The original 45a5af55
  solver is the full baseline.

3a) Concrete behavior and trace
- Input remains a stack of full-width horizontal stripes. Reading bottom -> top gives
  runs [(c0, h0), (c1, h1), (c2, h2), ...].
- Original behavior:
  - center = h0 x h0 solid block of color c0
  - wrap that with a c1 frame of thickness h1
  - keep wrapping outward with higher runs
- Variant behavior:
  - center is still h0 x h0, but its interior is filled with horizontal 1-row stripes
    alternating between c0 and c1, starting with c0 on the top row
  - the second-bottom run is still used again as the first outer frame of thickness h1
- Small trace:
  - runs = [(7, 3), (2, 2), (4, 1)]
  - original center:
      7 7 7
      7 7 7
      7 7 7
  - variant center:
      7 7 7
      2 2 2
      7 7 7
  - then color 2 still wraps this as the first 2-thick frame, and color 4 wraps that
    as the outer 1-thick frame.
- So the only semantic change is the innermost fill rule; frame geometry and run parsing
  stay the same.

3b) Complexity calibration
- The rule is intentionally narrow: only two colors participate in the new center fill,
  and the stripe thickness inside the center is fixed to 1 row. This keeps inference
  crisp instead of introducing a second height-dependent stripe rule.
- To avoid the change disappearing visually, generator forces h0 >= 2 so both c0 and c1
  are visible inside the center. Colors are sampled without replacement so c1's reuse is
  easy to notice instead of being muddied by repeated outer colors.

3c) Implicit assumption audit against the original main()
- Parsing runs bottom->top still works unchanged because the input format is unchanged.
- `c0, h0 = runs[0]` used to imply "center is solid c0"; that assumption no longer holds.
  Only the center size still comes from h0.
- `cur = np.full((h0, h0), c0)` is the line that depended on the old solid-fill property.
  It must be replaced with an explicit striped-center builder that also reads runs[1].
- The wrapping loop over `runs[1:]` remains valid. Although c1 now appears inside the
  center too, the frame construction still works because each step paints a new outer
  square and pastes the previous result into the middle; it never overwrites the interior.
- New assumption introduced by the variant: there must be at least two runs if the center
  is to reference both c0 and c1. The generator guarantees this, and main() checks it.

3d) Placement legality
- Input generation is still trivial stacking: full-width rectangles, no overlap, no
  out-of-bounds placement.
- The output size formula stays `h0 + 2 * sum(h_i for i >= 1)`, so the original <= 30
  cap remains valid.
- Because the input is still only uniform full-width stripes, the solver can parse it
  unambiguously.

3e) Visual observability
- Human solvers should be able to compare at least two examples and notice that:
  1. the innermost square is no longer solid,
  2. its alternating rows use exactly the bottom and second-bottom stripe colors, and
  3. the second-bottom color is reused both inside the center and as the first frame.
- Enforcing h0 >= 2 is essential; otherwise a 1x1 center would collapse back to the
  original appearance and hide the D4 change.

3f) Visual cleanliness
- The variant adds detail only inside the center square. It does not add extra objects,
  offsets, or overlaps, so the nested-frame picture remains clean and easy to parse.

3g) Color-change inferability
- Only two colors change role at the center, and those same colors already have strong
  structural cues in the input: they are the bottom two stripes.
- Using unique run colors avoids accidental ambiguity where the same color might reappear
  elsewhere and weaken the "reuse of the second-bottom color" signal.

8) Variant-behavior validation target
- A useful metric is whether the new solver differs from the original solver on the same
  generated input. For this variant that should happen whenever h0 >= 2 and c0 != c1,
  because the center contains at least one row recolored to c1. The generator enforces
  exactly those conditions.

9) Audit of new assumptions in main()
- `_parse_runs` assumes every stripe row is horizontally uniform, as in the original task.
  The generator constructs rows that satisfy this exactly.
- `_build_output` assumes `len(runs) >= 2` because c1 is needed for the alternating center.
  The generator samples `n_runs` from 2..6.
- The alternating fill assumes row-level stripes of thickness 1. This is a deliberate,
  fixed rule, not inferred from any additional latent parameter; generator does not need
  extra support beyond ensuring the center is tall enough to display it.
"""


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width stripe runs from bottom to top as (color, height)."""
    g = np.array(grid, dtype=int)
    h_total, _w_total = g.shape

    runs: list[tuple[int, int]] = []
    r = int(h_total - 1)
    while r >= 0:
        color = int(g[r, 0])
        height = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == color).all()):
            height += 1
            rr -= 1
        runs.append((color, height))
        r = rr
    return runs


def _build_center(c0: int, h0: int, c1: int) -> np.ndarray:
    """
    Build the new center square.

    The center keeps side length h0, but alternates 1-row horizontal stripes between
    c0 and c1, starting with c0 on the top row.
    """
    center = np.full((h0, h0), int(c0), dtype=int)
    center[1::2, :] = int(c1)
    return center


def _build_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Construct the nested-square output from bottom->top runs."""
    if len(runs) < 2:
        raise ValueError("D4 variant requires at least two horizontal stripes")

    c0, h0 = runs[0]
    c1, _h1 = runs[1]
    cur = _build_center(int(c0), int(h0), int(c1))

    for color, height in runs[1:]:
        height = int(height)
        new_n = int(cur.shape[0] + 2 * height)
        nxt = np.full((new_n, new_n), int(color), dtype=int)
        nxt[height : height + cur.shape[0], height : height + cur.shape[1]] = cur
        cur = nxt

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant D4:
    - The bottom stripe no longer becomes a solid center square.
    - Instead, the center square of side length h0 is filled with alternating horizontal
      1-row stripes using the bottom color c0 and the second-bottom color c1.
    - The second-bottom stripe still also forms the first outer frame of thickness h1.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    return _build_output(runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 7))
    widths = int(np.random.randint(6, 15))

    # h0 >= 2 guarantees the striped center visibly differs from the original solid center.
    heights = [int(np.random.randint(2, 5))]
    heights.extend(int(np.random.randint(1, 5)) for _ in range(n_runs - 1))

    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    # Unique colors make the c1 reuse in the center easier to infer visually.
    colors = [int(x) for x in np.random.choice(np.arange(1, 10), size=n_runs, replace=False)]
    runs = list(zip(colors, heights))

    stripes = [
        np.full((height, widths), color, dtype=int)
        for color, height in reversed(runs)
    ]
    candidate = np.vstack(stripes)

    # Validate the sampled instance against the solver to keep generation robust.
    expected = _build_output(runs)
    solved = main(candidate)
    if solved.shape[0] > 30 or solved.shape[1] > 30:
        return None
    if not np.array_equal(solved, expected):
        return None
    return candidate


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
