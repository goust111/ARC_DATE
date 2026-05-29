# Combo variant of 7666fa5d (index 1)
# combo_spec: D1/0+D12/0
# Name: 颜色编码配对与多色线段系统

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
Original task:
- The input has several short anti-diagonal segments on distinct k=r+c lines.
- For each main diagonal b=r-c, the solver collects the active segments, sorts
  them by k, and fills every geometric gap between consecutive segments with a
  single fixed red color.

Combo variant:
- Keep the same diagonal geometry and the same sweep along b=r-c.
- Change the "always fill every consecutive gap" rule into a color-coded rule:
  a gap fills only when the two boundary segment colors are either identical or
  one of two specific partner pairs.
- The fill color is also color-coded:
  same-color pair (x,x) -> x
  mixed pair {3,4} -> 3
  mixed pair {6,7} -> 6
  any other adjacent color pair -> no fill

Small trace on one b-line:
- Suppose the active segments in k-order have colors [3,3,4,6,7].
- Original logic would fill all four gaps.
- Combo logic fills:
  - 3|3 with color 3
  - 3|4 with color 3
  - 4|6 is blocked and stays background
  - 6|7 with color 6
- So geometry still decides where a gap could exist, but color decides whether
  the gap actually fills and which color it receives.

b) Complexity calibration
-------------------------
- Use exactly 5 anti-diagonal lines per sample. That gives 4 adjacent gaps,
  enough room to show all required evidence in one puzzle.
- Restrict the line palette to {3,4,6,7}. This is small enough to infer from a
  few examples, but still supports repeated colors plus mixed partner families.
- Every sampled color order is chosen to contain:
  - at least one same-color adjacent pair,
  - both mixed partner families {3,4} and {6,7},
  - at least one geometrically valid but color-blocked adjacent pair.
- This prevents degeneration into the original task or into a pure multicolor
  propagation task with no negative examples.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `line = max(non_bg, key=...)`
  Old assumption: all segments share one color. Broken by the combo because
  multiple line colors are meaningful. Fix: extract components per non-bg color.
- `fill = 2`
  Old assumption: every filled gap uses one fixed color. Broken because the fill
  now depends on the boundary-color combination. Fix: compute fill per pair.
- `components_mask(g == line)`
  Old assumption: a single color mask recovers every segment. Broken because the
  control signal is distributed across several colors. Fix: scan every color.
- `fill every adjacent k-pair`
  Old assumption: geometry alone defines matching. Broken because some adjacent
  geometric gaps must stay blank. Fix: test each adjacent pair against the color
  rule before painting.
- `out[r, c] = fill`
  Old assumption: there is no pair-specific output ambiguity. Broken because the
  same geometry can produce different fill colors. Fix: route through a pair
  lookup that handles both same-color and partner-pair cases.

d) Placement legality
---------------------
- All segments still lie on single anti-diagonals and are generated from valid
  b-ranges, so every painted cell stays inside the 16x16 grid.
- Distinct k values prevent cell overlap between different segments.
- Consecutive segments are sampled with positive b-overlap so every adjacent
  color relation is visible on at least some main diagonals.
- The generator rejects layouts unless the same-color gap, both mixed-family
  gaps, and the blocked geometric gap all occupy multiple cells.

e) Visual observability
-----------------------
- The input already exposes the control code: the five diagonal segments are
  visibly multi-colored, with one repeated color and two partner families.
- The output shows three filled parallelograms and one skipped gap, so a human
  can see that color matching overrides pure geometric adjacency.
- The fill colors repeat visible boundary colors rather than introducing an
  arbitrary hidden legend, which keeps the mapping learnable from 2-4 examples.

f) Visual cleanliness
---------------------
- The scene remains only line segments plus filled bands; no extra markers or
  legend objects are added.
- Limiting the palette to four saturated colors avoids clutter.
- Using a single blocked gap per chain keeps the contrast with the original task
  obvious without making the grid noisy.

g) Change inferability
----------------------
- The rule is structured, not arbitrary:
  same colors match themselves, and two recurring mixed families also match.
- Because every sample includes positive same-color evidence, positive mixed
  evidence, and negative blocked evidence, the solver can infer both "which
  pairs count" and "what color each valid pair produces".
- The generator also measures how many cells would have been filled by the
  original geometry-only rule but are blocked here; this quantifies that the
  color logic is genuinely active.

New solver assumptions introduced by the combo
----------------------------------------------
- Every meaningful non-background 8-connected component in the input is a single
  anti-diagonal segment with step-2 continuity in b-space.
- The input contains no pre-existing filled regions; otherwise a fill region in
  the same color could merge with a line segment and stop looking like a pure
  single-k component.
- The generator guarantees the sampled color order belongs to the supported
  template family, so the solver's fixed partner-pair table is sufficient.
"""

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


Segment = tuple[int, int, int, int]  # (k, b_min, b_max, color)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


H = W = 16
BG = 8
LINE_COLORS = (3, 4, 6, 7)
MIXED_PAIR_TO_FILL = {
    frozenset((3, 4)): 3,
    frozenset((6, 7)): 6,
}
COLOR_ORDER_TEMPLATES = (
    (3, 3, 4, 6, 7),
    (4, 3, 3, 6, 7),
    (6, 7, 3, 3, 4),
    (6, 7, 4, 4, 3),
    (3, 4, 6, 7, 7),
    (7, 7, 6, 3, 4),
)

K_LO, K_HI = 4, 26
K_GAPS = (4, 4, 4, 4)
SEG_LEN_MIN, SEG_LEN_MAX = 4, 9
OVERLAP_MIN_B = 6
SEG_RETRY = 80
GEN_RETRY = 500

MIN_FILL_CELLS = 12
MIN_BLOCKED_CELLS = 4
MIN_GAP_CELLS_PER_ACTIVE_PAIR = 4


def _pair_fill_color(color_a: int, color_b: int) -> int | None:
    if color_a == color_b:
        return int(color_a)
    return MIXED_PAIR_TO_FILL.get(frozenset((int(color_a), int(color_b))))


def _segment_info(pts: list[tuple[int, int]], color: int) -> Segment | None:
    if not pts:
        return None
    ks = {r + c for r, c in pts}
    if len(ks) != 1:
        return None
    bs = sorted(r - c for r, c in pts)
    if any(bs[i] + 2 != bs[i + 1] for i in range(len(bs) - 1)):
        return None
    return int(next(iter(ks))), int(bs[0]), int(bs[-1]), int(color)


def _extract_segments(grid: np.ndarray, bg: int) -> list[Segment]:
    segs: list[Segment] = []
    for color in sorted(set(map(int, grid.ravel()))):
        if color == bg:
            continue
        for pts in _cc8(grid == color):
            info = _segment_info(pts, color)
            if info is not None:
                segs.append(info)
    segs.sort(key=lambda s: (s[0], s[1], s[3]))
    return segs


def _paint_from_segments(
    grid: np.ndarray,
    segs: list[Segment],
    *,
    color_sensitive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    mask = np.zeros_like(out, dtype=bool)
    h, w = out.shape
    bg = int(_bg_color(out))

    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda s: s[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left, right in zip(active, active[1:]):
            if color_sensitive:
                fill_color = _pair_fill_color(left[3], right[3])
                if fill_color is None:
                    continue
            else:
                fill_color = 1
            k_left, k_right = int(left[0]), int(right[0])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = int(fill_color)
                    mask[r, c] = True
    return out, mask


def _overlap_b_lines(left: Segment, right: Segment) -> int:
    overlap_lo = max(int(left[1]), int(right[1]))
    overlap_hi = min(int(left[2]), int(right[2]))
    if overlap_lo > overlap_hi:
        return 0
    return (overlap_hi - overlap_lo) // 2 + 1


def _gap_cell_count(left: Segment, right: Segment) -> int:
    width = max(0, (int(right[0]) - int(left[0])) // 2 - 1)
    return width * _overlap_b_lines(left, right)


def _pair_kind(left: Segment, right: Segment) -> str:
    if left[3] == right[3]:
        return "same"
    pair = frozenset((int(left[3]), int(right[3])))
    if pair == frozenset((3, 4)):
        return "mixed_34"
    if pair == frozenset((6, 7)):
        return "mixed_67"
    return "blocked"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only those diagonal gaps whose boundary colors satisfy the color-coded
    pairing rule, and color each filled band according to that pair.

    Input structure:
    - The background is the dominant color.
    - Multiple non-background colors form short 8-connected line segments, each
      confined to a single anti-diagonal k=r+c.
    - Some adjacent segment pairs repeat a color, while others form visible
      partner-color families.

    Rule:
    - For each main diagonal b=r-c, collect the anti-diagonal segments whose
      b-range covers that line and sort them by k.
    - Examine each consecutive pair in that k-order.
    - Fill the background cells strictly between them only if their colors are
      identical or belong to one of the supported mixed pairs {3,4} or {6,7}.
      Same-color pairs fill with that shared color; mixed pairs fill with the
      family color 3 or 6 respectively.

    Key insight:
    Geometry still tells you where a gap could be, but color now decides both
    whether the gap is active and what color the parallelogram band should take.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    segs = _extract_segments(g, bg)
    assert segs, "grid must contain anti-diagonal line segments"
    out, _mask = _paint_from_segments(g, segs, color_sensitive=True)
    return out


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    total_span = sum(K_GAPS)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None
    k0 = random.choice(starts)
    ks = [k0]
    for gap in K_GAPS:
        ks.append(ks[-1] + gap)
    return ks if ks[-1] <= K_HI else None


def _random_seg(
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        step = 2 * (length - 1)
        if b_lo + step > b_hi:
            continue
        candidates = list(range(b_lo, b_hi - step + 1, 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                start
                for start in candidates
                if start + step >= constrain_lo + OVERLAP_MIN_B
                and start <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        start = random.choice(candidates)
        return start, start + step
    return None


def _assign_colors(ks: list[int]) -> dict[int, int]:
    order = random.choice(COLOR_ORDER_TEMPLATES)
    assert len(order) == len(ks)
    return {k: int(color) for k, color in zip(ks, order)}


def _pair_metrics(segs: list[Segment]) -> dict[str, int]:
    ordered = sorted(segs, key=lambda s: s[0])
    metrics = {"same": 0, "mixed_34": 0, "mixed_67": 0, "blocked": 0}
    for left, right in zip(ordered, ordered[1:]):
        metrics[_pair_kind(left, right)] += _gap_cell_count(left, right)
    return metrics


def _gen_attempt() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    color_by_k = _assign_colors(ks)
    segs: list[Segment] = []

    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], color_by_k[ks[0]]))

    for k in ks[1:]:
        prev = segs[-1]
        nxt = _random_seg(k, prev[1], prev[2])
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1], color_by_k[k]))

    metrics = _pair_metrics(segs)
    if metrics["same"] < MIN_GAP_CELLS_PER_ACTIVE_PAIR:
        return None
    if metrics["mixed_34"] < MIN_GAP_CELLS_PER_ACTIVE_PAIR:
        return None
    if metrics["mixed_67"] < MIN_GAP_CELLS_PER_ACTIVE_PAIR:
        return None
    if metrics["blocked"] < MIN_BLOCKED_CELLS:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max, color in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(color)

    output, combo_mask = _paint_from_segments(grid, segs, color_sensitive=True)
    _geometry_output, geometry_mask = _paint_from_segments(grid, segs, color_sensitive=False)

    if int(np.sum(combo_mask)) < MIN_FILL_CELLS:
        return None
    if int(np.sum(geometry_mask & (~combo_mask))) < MIN_BLOCKED_CELLS:
        return None

    fill_colors = {int(output[r, c]) for r, c in np.argwhere(combo_mask)}
    if len(fill_colors) < 2:
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
