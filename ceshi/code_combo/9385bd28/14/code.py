# Combo variant of 9385bd28 (index 14)
# combo_spec: D1/2+D10/0+D17/1
# Name: 大小编码匹配+颜色环映射+单条目

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The old task had several legend rows, each explicitly giving `(marker, fill)`, and each marker
#      color owned at most one object family. This combo removes both assumptions.
#    - The new legend is one single bottom-left object: a 3-cell vertical color cycle. Reading bottom
#      to top gives `[c0, c1, c2]`. Only `c0` is used as a scene marker color; `c1` and `c2` are not
#      standalone marker families, they only extend the ring `c0 -> c1 -> c2 -> c0`.
#    - Outside the legend, every non-legend object uses that single marker color `c0` and is a
#      4-corner frame: all four bbox corners contain inward-facing L markers, three are small
#      (`arm=2`, 3 pixels) and exactly one is larger. The large arm length is either 3, 4, or 5.
#    - The large corner still selects the marker diagonal exactly like the D1 size-coded frame idea:
#      if the large corner is on TL/BR, the fill goes on the opposite TR/BL diagonal; if it is on
#      TR/BL, the fill goes on TL/BR.
#    - The new part is that the large corner size also controls the fill color by ring distance.
#      Concrete trace:
#        legend column bottom->top = [2, 7, 4]
#        object A has large arm 3  => advance 1 step on the ring => fill 7
#        object B has large arm 4  => advance 2 steps           => fill 4
#        object C has large arm 5  => advance 3 steps, wrap     => fill 2
#      So one visible legend object generates three effective fill outcomes.
#    - Compared with the original task, color is no longer enough to decide either grouping or fill:
#      grouping is by reconstructing several same-colored frames, and fill depends on the size bucket
#      of the unique oversized corner.
#
# b) Complexity calibration
#    - The ring is fixed to exactly 3 colors. That is enough to make the cyclic rule visible while
#      keeping the examples learnable.
#    - Large-corner arm lengths are restricted to exactly 3 values: 3, 4, 5. This gives a small,
#      repeated size vocabulary instead of many arbitrary numeric cases.
#    - Every generated sample contains one object from each size bucket, so the color-cycle effect is
#      present inside a single grid rather than only across many samples.
#    - There are no 2x2 blocks and no distractor colors here. They were orthogonal in the source
#      task, but in this fusion they would add clutter without teaching the new multi-step color rule.
#
# c) Implicit assumptions review against the original solver
#    - Original `_find_legend` expected a 2-column `(marker, fill)` stack. That fails here because the
#      legend is a single-column cycle token, so legend parsing must be rewritten.
#    - Original `marker_pts = all pixels of one color` implied one bbox per color. That breaks twice:
#      there is only one marker color in the whole scene, and each full frame is split into four
#      disconnected corner components.
#    - D10-style "pair corner pieces back together" is also insufficient, because each object now has
#      four components, not two. The solver must partition components into valid quartets.
#    - Original diagonal inference used anchor presence on opposite corners. That would tie here,
#      because all four corners are populated. The unique large corner becomes the diagonal selector.
#    - Original fill lookup came directly from the legend row. That no longer holds; the solver must
#      derive a step count from the large-corner size and then walk that many steps around the legend
#      cycle.
#
# d) Placement legality
#    - Objects are generated as non-overlapping bbox frames and are kept at least one cell apart, so
#      corner components stay disconnected and visual grouping stays readable.
#    - Minimum bbox size depends on the large corner size. For arm length `L`, the bbox needs both
#      dimensions at least `L + 3`; the extra gap cell prevents same-row / same-column corner arms
#      from becoming 4-connected.
#    - The generator stores oracle metadata for every placed object and rejects any sample where the
#      solver reconstruction disagrees with that oracle output.
#
# e) Visual observability
#    - A human can directly see three things: there is only one scene marker color, there are several
#      same-colored rectangles, and those rectangles contain visibly different large-corner sizes.
#    - The legend is also visibly different from the source task: it is one vertical 3-color strip
#      rather than multiple `(marker, fill)` rows.
#    - Because the sample always includes size buckets 3/4/5, the outputs show that progressively
#      larger oversized corners map to successive colors in the visible legend cycle, including wrap.
#
# f) Visual cleanliness
#    - Only one object family color appears in the input outside the legend, which makes the size cue
#      the only varying object feature.
#    - Bboxes do not overlap or touch, so the filled diamonds remain separable and the quartet search
#      is not fighting clutter introduced by the generator.
#    - A one-column blank buffer is kept to the right of the legend strip so it is not misread as the
#      original 2-column table.
#
# g) Color-change inferability
#    - The legend exposes exactly the three colors participating in the cycle, and all three are used
#      by the outputs because one object from each size bucket is always present.
#    - No hidden sort by color id is used. The only color-order source is the visible bottom-to-top
#      order of the legend strip, and the only object-index source is the visible large-corner size.
#
# Post-change solver assumptions audited
# - `_find_legend()` assumes the legend is the unique bottom-left single-column non-bg run of length 3;
#   `generate_input()` guarantees that with a forbidden mask and by reserving a blank buffer column.
# - `_extract_objects()` assumes every non-legend marker component is an L-corner occupying exactly one
#   corner of some bbox, and that objects can be partitioned into quartets with sizes `[3,3,3,5/7/9]`;
#   the generator constructs only those frames.
# - `_object_fill()` assumes the unique large component size is one of 5, 7, 9, corresponding to large
#   arm lengths 3, 4, 5. The generator uses exactly those arm lengths.
# - `main()` protects all marker pixels globally rather than object-by-object. Because bboxes never
#   overlap or touch, this cannot suppress a legitimate fill region that belongs to another object.


LEGEND_LEN = 3
SMALL_ARM = 2
LARGE_ARMS = (3, 4, 5)
LARGE_COMPONENT_SIZES = tuple(2 * arm - 1 for arm in LARGE_ARMS)
CORNERS = ("TL", "TR", "BL", "BR")

_L_DIRS = {
    "TL": ((0, 1), (1, 0)),
    "TR": ((0, -1), (1, 0)),
    "BL": ((0, 1), (-1, 0)),
    "BR": ((0, -1), (-1, 0)),
}


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    """
    Legend is a single bottom-left vertical color cycle.
    Reading bottom->top gives exactly three colors [c0, c1, c2].
    Returns (legend_col, entries) ordered bottom->top as (row, color).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None, "legend must exist at bottom-left"

    rows: list[tuple[int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        rows.append((r, int(g[r, c])))

    assert len(rows) == LEGEND_LEN, f"legend must have exactly {LEGEND_LEN} colors"
    colors = [color for _, color in rows]
    assert len(set(colors)) == LEGEND_LEN, "legend cycle colors must be distinct"
    return c, rows


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """Cells inside [r0..r1]x[c0..c1] on a shortest 4-neigh path between corners a and b."""
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    remaining = set(map(tuple, np.argwhere(mask)))
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    comps.sort(key=_bbox_of_pts)
    return comps


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_pts(shape: tuple[int, int], pts: set[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r, c in pts:
        m[r, c] = True
    return m


def _component_info(pts: set[tuple[int, int]]) -> dict[str, object]:
    rmin, cmin, rmax, cmax = _bbox_of_pts(pts)
    return {
        "pts": pts,
        "size": len(pts),
        "rmin": rmin,
        "cmin": cmin,
        "rmax": rmax,
        "cmax": cmax,
    }


def _component_corner(
    info: dict[str, object],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> str | None:
    ir0 = int(info["rmin"])
    ir1 = int(info["rmax"])
    ic0 = int(info["cmin"])
    ic1 = int(info["cmax"])
    matches: list[str] = []
    if ir0 == rmin and ic0 == cmin:
        matches.append("TL")
    if ir0 == rmin and ic1 == cmax:
        matches.append("TR")
    if ir1 == rmax and ic0 == cmin:
        matches.append("BL")
    if ir1 == rmax and ic1 == cmax:
        matches.append("BR")
    if len(matches) != 1:
        return None
    return matches[0]


def _valid_corner_component(info: dict[str, object]) -> bool:
    size = int(info["size"])
    rspan = int(info["rmax"]) - int(info["rmin"])
    cspan = int(info["cmax"]) - int(info["cmin"])
    if size == 3:
        return rspan == 1 and cspan == 1
    if size in LARGE_COMPONENT_SIZES:
        arm = (size + 1) // 2
        return rspan == arm - 1 and cspan == arm - 1
    return False


def _decode_quartet(comps: list[dict[str, object]]) -> dict[str, object] | None:
    if len(comps) != 4:
        return None
    if not all(_valid_corner_component(info) for info in comps):
        return None

    all_pts: set[tuple[int, int]] = set()
    for info in comps:
        all_pts |= set(info["pts"])
    rmin, cmin, rmax, cmax = _bbox_of_pts(all_pts)
    if rmin == rmax or cmin == cmax:
        return None

    by_corner: dict[str, dict[str, object]] = {}
    for info in comps:
        corner = _component_corner(info, rmin, rmax, cmin, cmax)
        if corner is None or corner in by_corner:
            return None
        by_corner[corner] = info
    if set(by_corner) != set(CORNERS):
        return None

    sizes = sorted(int(info["size"]) for info in comps)
    if sizes[:3] != [3, 3, 3] or sizes[3] not in LARGE_COMPONENT_SIZES:
        return None

    large_corner = next(corner for corner, info in by_corner.items() if int(info["size"]) > 3)
    large_size = int(by_corner[large_corner]["size"])
    large_arm = (large_size + 1) // 2

    if large_corner in {"TL", "BR"}:
        fill_diag = ((rmin, cmax), (rmax, cmin))
    else:
        fill_diag = ((rmin, cmin), (rmax, cmax))

    return {
        "bbox": (rmin, cmin, rmax, cmax),
        "pts": all_pts,
        "large_corner": large_corner,
        "large_arm": large_arm,
        "fill_diag": fill_diag,
    }


def _extract_objects(shape: tuple[int, int], marker_mask: np.ndarray) -> list[dict[str, object]]:
    comps_pts = _connected_components(marker_mask)
    infos = [_component_info(pts) for pts in comps_pts]
    assert infos, "single-entry legend must drive at least one marker object"

    n = len(infos)
    assert n % 4 == 0, "frame objects require corner components in groups of four"

    candidate_masks: list[int] = []
    decoded_by_mask: dict[int, dict[str, object]] = {}
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    decoded = _decode_quartet([infos[a], infos[b], infos[c], infos[d]])
                    if decoded is not None:
                        candidate_masks.append(mask)
                        decoded_by_mask[mask] = decoded

    assert candidate_masks, "no valid frame quartet found"
    cand_by_first: dict[int, list[int]] = {i: [] for i in range(n)}
    for mask in candidate_masks:
        for i in range(n):
            if mask & (1 << i):
                cand_by_first[i].append(mask)

    full = (1 << n) - 1
    memo: dict[int, list[int] | None] = {}

    def solve(used: int) -> list[int] | None:
        if used == full:
            return []
        if used in memo:
            return memo[used]
        first = next(i for i in range(n) if not (used & (1 << i)))
        for mask in cand_by_first[first]:
            if mask & used:
                continue
            tail = solve(used | mask)
            if tail is not None:
                memo[used] = [mask] + tail
                return memo[used]
        memo[used] = None
        return None

    solution = solve(0)
    assert solution is not None, "components must partition into valid size-coded frames"
    objects = [decoded_by_mask[mask] for mask in solution]
    objects.sort(key=lambda obj: tuple(obj["bbox"]))
    return objects


def _object_fill(cycle_colors: list[int], obj: dict[str, object]) -> int:
    step = int(obj["large_arm"]) - SMALL_ARM
    return int(cycle_colors[step % len(cycle_colors)])


def _paint_objects(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    marker_mask: np.ndarray,
    cycle_colors: list[int],
    objects: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    protect = legend_mask | marker_mask
    for obj in objects:
        rmin, cmin, rmax, cmax = obj["bbox"]
        a, b = obj["fill_diag"]
        fill = _object_fill(cycle_colors, obj)
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = fill
    return out


def _corner_l_cells(anchor_r: int, anchor_c: int, corner: str, arm_len: int) -> list[tuple[int, int]]:
    cells = {(anchor_r, anchor_c)}
    (dr1, dc1), (dr2, dc2) = _L_DIRS[corner]
    for k in range(1, arm_len):
        cells.add((anchor_r + dr1 * k, anchor_c + dc1 * k))
        cells.add((anchor_r + dr2 * k, anchor_c + dc2 * k))
    return list(cells)


def _frame_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    large_corner: str,
    large_arm: int,
) -> list[tuple[int, int]]:
    specs = {
        "TL": ((rmin, cmin), large_arm if large_corner == "TL" else SMALL_ARM),
        "TR": ((rmin, cmax), large_arm if large_corner == "TR" else SMALL_ARM),
        "BL": ((rmax, cmin), large_arm if large_corner == "BL" else SMALL_ARM),
        "BR": ((rmax, cmax), large_arm if large_corner == "BR" else SMALL_ARM),
    }
    cells: set[tuple[int, int]] = set()
    for corner, ((r, c), arm) in specs.items():
        cells.update(_corner_l_cells(r, c, corner, arm))
    return list(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several same-colored size-coded frames using one cyclic legend token.

    Input structure:
    - The bottom-left legend is a single vertical 3-color strip. Reading bottom-to-top yields a
      visible color cycle `[c0, c1, c2]`.
    - Only the bottom legend color `c0` appears elsewhere as marker pixels.
    - Every non-legend object is a disconnected 4-corner frame of color `c0`: three corners are small
      L markers and one corner is a larger L whose arm length is 3, 4, or 5.

    Rule:
    - Reconstruct the same-colored corner components into full 4-corner frames.
    - The corner position of the unique oversized L selects which diagonal is the marker diagonal, so
      the opposite diagonal's diamond is the region to fill.
    - The oversized L length determines how far to advance along the legend cycle: arm 3 means one
      step (`c0 -> c1`), arm 4 means two steps (`c0 -> c2`), arm 5 means three steps and wraps back
      to `c0`.
    - Fill each object's diamond with that cycle-derived color while preserving the legend cells and
      all marker pixels.

    Key insight: The single legend object no longer gives one fixed fill color. Its vertical color
    order defines a ring, and the visible size of each frame's unique large corner chooses which ring
    step to apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend_c, legend_rows = _find_legend(g, bg)
    cycle_colors = [color for _, color in legend_rows]
    marker = cycle_colors[0]

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _ in legend_rows:
        legend_mask[r, legend_c] = True

    marker_mask = (g == marker) & (~legend_mask)
    objects = _extract_objects(g.shape, marker_mask)
    return _paint_objects(g, legend_mask, marker_mask, cycle_colors, objects)


# --------------- generator ---------------

# Solver assumptions guaranteed by the generator:
# 1. The legend is exactly one 3-color vertical strip at the bottom-left with a blank buffer column.
# 2. Only the bottom legend color is used for non-legend marker objects.
# 3. Every sample contains one frame with large arm 3, one with large arm 4, and one with large arm 5.
# 4. Every frame is a 4-corner object with component sizes `[3,3,3,5/7/9]` and isolated bbox spacing.
# 5. The solver output must match an oracle rendering computed from placement metadata.

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (18, 22)
GRID_W_RANGE = (18, 22)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ROW_OFFSET_CHOICES = (0, 1)
LEGEND_BUFFER = 1

# ── CFG: objects under one legend entry ─────────────────────────────
EXTRA_OBJECT_PROB = 0.0

# ── CFG: frame bbox shape ───────────────────────────────────────────
BBOX_H_MAX = 11
BBOX_W_MAX = 11
BBOX_EXTRA_SPAN = 1

# ── CFG: retry budgets ──────────────────────────────────────────────
FRAME_RETRIES = 220
GEN_RETRIES = 400


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = H - 1 - random.choice(LEGEND_ROW_OFFSET_CHOICES)
    if legend_bot - (LEGEND_LEN - 1) < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < LEGEND_LEN:
        return None
    colors = random.sample(pool, LEGEND_LEN)

    grid = np.full((H, W), bg, dtype=int)
    legend_mask = np.zeros((H, W), dtype=bool)
    legend_rows: list[tuple[int, int]] = []
    for i, color in enumerate(colors):
        r = legend_bot - i
        grid[r, legend_c] = color
        legend_mask[r, legend_c] = True
        legend_rows.append((r, color))
    return grid, legend_mask, legend_c, legend_bot, colors, legend_rows


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int]]) -> np.ndarray:
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    keep_clear_upto = min(W, legend_c + 1 + LEGEND_BUFFER)
    forbidden[: legend_bot + 1, :keep_clear_upto] = True
    for r, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _try_place_frame(
    H: int,
    W: int,
    color: int,
    large_arm: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    min_dim = large_arm + 3
    h_hi = min(BBOX_H_MAX, H)
    w_hi = min(BBOX_W_MAX, W)
    if h_hi < min_dim or w_hi < min_dim:
        return False

    for _ in range(retries):
        bh = random.randint(min_dim, min(h_hi, min_dim + BBOX_EXTRA_SPAN))
        bw = random.randint(min_dim, min(w_hi, min_dim + BBOX_EXTRA_SPAN))
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        large_corner = random.choice(list(CORNERS))
        cells = _frame_cells(rmin, cmin, rmax, cmax, large_corner, large_arm)
        bbox = (rmin, cmin, rmax, cmax)
        if can_place(cells, bbox):
            do_place(cells, color, bbox, large_corner, large_arm)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_mask, legend_c, legend_bot, cycle_colors, legend_rows = legend_result
    marker = cycle_colors[0]

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    used_boxes: list[tuple[int, int, int, int]] = []
    objects: list[dict[str, object]] = []

    def can_place(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
        cell_set = set(cells)
        if len(cell_set) != len(cells):
            return False
        if any(_boxes_touch_or_overlap(bbox, other) for other in used_boxes):
            return False
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(
        cells: list[tuple[int, int]],
        color: int,
        bbox: tuple[int, int, int, int],
        large_corner: str,
        large_arm: int,
    ) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        used_boxes.append(bbox)
        rmin, cmin, rmax, cmax = bbox
        if large_corner in {"TL", "BR"}:
            fill_diag = ((rmin, cmax), (rmax, cmin))
        else:
            fill_diag = ((rmin, cmin), (rmax, cmax))
        objects.append(
            {
                "bbox": bbox,
                "large_corner": large_corner,
                "large_arm": large_arm,
                "fill_diag": fill_diag,
                "pts": set(cells),
            }
        )

    large_arms = list(LARGE_ARMS)
    if random.random() < EXTRA_OBJECT_PROB:
        large_arms.append(random.choice(LARGE_ARMS))
    random.shuffle(large_arms)

    for arm in large_arms:
        if not _try_place_frame(H, W, marker, arm, FRAME_RETRIES, can_place, do_place):
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    oracle = _paint_objects(grid, legend_mask, (grid == marker) & (~legend_mask), cycle_colors, objects)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, oracle):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
