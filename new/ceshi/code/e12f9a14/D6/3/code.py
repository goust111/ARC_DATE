# Variant of e12f9a14
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线延伸到网格边界或碰到另一条射线的路径时终止... → 射线填充整条对角线直到两侧边界

import itertools

import numpy as np

from arc_utils import get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# a) Concrete behavior:
#    The original task treated each gap as the start of a single moving beam
#    with collision rules. This variant removes beam dynamics entirely. Every
#    generated gap is a corner gap on the 4x4 frame, and that corner determines
#    one unique tangent diagonal that stays outside the frame. The output paints
#    the whole boundary-to-boundary diagonal through that corner gap in both
#    opposite directions at once.
#
#    Small trace:
#    - Original-style corner gap at frame NW corner (r, c): one beam would run
#      away from the frame in direction (-1, -1).
#    - Variant gap at the same corner: paint the full tangent diagonal
#      {(r, c), (r-1, c+1), (r-2, c+2), ...} U {(r+1, c-1), (r+2, c-2), ...}.
#    - Difference: no moving head, no stopping on contact, no turn/merge logic;
#      the diagonal is a static complete line centered on the gap.
#
# b) Complexity calibration:
#    Full diagonals are visually denser than the original rays. To avoid noisy
#    outputs, the generator limits itself to corner gaps and usually chooses
#    only 1-2 gaps per frame, but it still varies block count, colors, frame
#    placement, and diagonal family so the rule is not trivialized.
#
# c) Hidden assumptions review:
#    - _find_blocks_2x2 / _infer_frame_color / _frame_gaps_for_block only depend
#      on framed 2x2 blocks, so they remain valid.
#    - All of the original beam-state machinery depended on the old invariant
#      "a ray has a single forward head and evolves step-by-step". That entire
#      assumption is invalid here, so the dynamic simulation is removed.
#    - New solver assumption: every structural gap used by this variant is a
#      corner gap, giving exactly one tangent diagonal outside the frame.
#    - New solver assumption: these diagonals only traverse background cells and
#      are pairwise disjoint. The generator validates both conditions.
#
# d) Placement legality:
#    Frames are still 4x4 shells around unique 2x2 blocks and keep the original
#    frame-bbox separation. For each candidate corner gap, the generator checks:
#    - both diagonal branches have enough runway to the border;
#    - the whole tangent line avoids every frame/block cell in the obstacle map;
#    - the full line does not overlap any previously selected diagonal.
#
# e) Visual observability:
#    The transformation is readable from a few examples because every corner
#    hole becomes a complete straight diagonal crossing the board. The rule does
#    not rely on color ids or processing order; colors simply identify which
#    block owns which full diagonal.
#
# f) Visual cleanliness:
#    Overlapping diagonals of different colors would create order-dependent
#    artifacts, so the generator forbids them. This keeps outputs as clean sets
#    of long 1-pixel diagonals radiating around the small framed objects.
#
# g) Color inference:
#    No color remapping is introduced. Each diagonal inherits its block color,
#    exactly as in the original task, so the only concept the solver must learn
#    is the new full-line geometry.
# ---------------------------------------------------------------------------


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    """Find the unique non-bg color on the 4x4 border around a 2x2 block."""
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))

    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        ok = True
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))

    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _gap_signs(br: int, bc: int, gr: int, gc: int) -> tuple[int, int]:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    return dr, dc


def _tangent_dirs(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    assert dr != 0 and dc != 0, "this variant expects corner gaps"
    return ((dr, -dc), (-dr, dc))


def _trace_from(cell: tuple[int, int], d: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = cell
    out: list[tuple[int, int]] = []
    while True:
        r += d[0]
        c += d[1]
        if not (0 <= r < h and 0 <= c < w):
            return out
        out.append((r, c))


def _full_tangent_line(
    cell: tuple[int, int], dr: int, dc: int, shape: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    dirs = _tangent_dirs(dr, dc)
    a = _trace_from(cell, dirs[0], shape)
    b = _trace_from(cell, dirs[1], shape)
    cells = tuple([cell, *a, *b])
    return cells, (len(a), len(b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint complete diagonals through framed corner gaps instead of simulating beams.

    Input structure:
    - The background is the most frequent color, with one shared frame color forming 4x4 borders around 2x2 solid blocks.
    - Each generated frame gap is a corner hole, so every gap has one unambiguous diagonal that stays outside the frame.

    Rule:
    - Read each corner gap as a seed of the block's color.
    - Paint the gap cell itself, then extend along the two opposite directions of the tangent diagonal through that gap.
    - Continue both branches independently until they leave the grid; no collision, turning, or truncation logic is used.

    Key insight: the old dynamic ray simulation is no longer relevant; every corner opening directly encodes one full boundary-to-boundary diagonal line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    painted: dict[tuple[int, int], int] = {}

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    for color, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected at least one structural gap per frame"
        for gr, gc in sorted(gaps):
            dr, dc = _gap_signs(br, bc, gr, gc)
            assert dr != 0 and dc != 0, "variant generator must use corner gaps only"
            line_cells, _branch_lengths = _full_tangent_line((gr, gc), dr, dc, (h, w))
            for idx, cell in enumerate(line_cells):
                if idx > 0:
                    assert int(g[cell[0], cell[1]]) == bg, "tangent diagonal must stay in background"
                owner = painted.get(cell)
                assert owner is None or owner == color, "different full diagonals must not overlap"
                painted[cell] = color
                out[cell[0], cell[1]] = color

    return out


# --------------- generator (o5x) ---------------

# ---- CFG: tunable generation parameters ----
BLOCK_COUNT_RANGE = (1, 6)              # uniform draw [lo, hi)
GRID_SIZE_MAX = 31                      # exclusive upper bound for square grid side
GRID_SIZE_MIN_BY_BLOCKS = {1: 8, 2: 14, 5: 20}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_BRANCH_STEPS = 2
MIN_LINE_PIXELS_PER_GAP = 5
GEN_RETRIES = 400
GAP_COUNT_CHOICES = [1, 2]
GAP_COUNT_WEIGHTS = [0.55, 0.45]

_CORNER_GAPS = ((0, 0), (0, 3), (3, 0), (3, 3))


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _line_cells_for_gap(
    n: int, r0: int, c0: int, gap: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    gr, gc = gap
    dr = -1 if gr == 0 else 1
    dc = -1 if gc == 0 else 1
    return _full_tangent_line((r0 + gr, c0 + gc), dr, dc, (n, n))


def _gap_data_for_block(
    obstacles: np.ndarray, bg: int, n: int, r0: int, c0: int
) -> dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    data: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = {}
    for gap in _CORNER_GAPS:
        line_cells, branch_lengths = _line_cells_for_gap(n, r0, c0, gap)
        if min(branch_lengths) < MIN_BRANCH_STEPS:
            continue
        if any(int(obstacles[r, c]) != bg for r, c in line_cells[1:]):
            continue
        data[gap] = (line_cells, branch_lengths)
    return data


def _candidate_gap_combos(
    gap_data: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], tuple[int, int]]],
    occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    combos: list[tuple[tuple[int, int], ...]] = []
    gaps = sorted(gap_data)
    for gap_count in GAP_COUNT_CHOICES:
        for combo in itertools.combinations(gaps, gap_count):
            union: set[tuple[int, int]] = set()
            ok = True
            for gap in combo:
                line_cells = gap_data[gap][0]
                line_set = set(line_cells)
                if union & line_set:
                    ok = False
                    break
                union |= line_set
            if not ok:
                continue
            if union & occupied:
                continue
            combos.append(combo)
    return combos


def _choose_gap_combo(
    combos: list[tuple[tuple[int, int], ...]], rng: np.random.Generator
) -> tuple[tuple[int, int], ...] | None:
    if not combos:
        return None
    weights = np.array(
        [GAP_COUNT_WEIGHTS[GAP_COUNT_CHOICES.index(len(combo))] for combo in combos], dtype=float
    )
    weights /= weights.sum()
    return combos[int(rng.choice(len(combos), p=weights))]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    order = list(range(block_count))
    rng.shuffle(order)
    chosen_gaps: list[tuple[tuple[int, int], ...] | None] = [None] * block_count
    occupied: set[tuple[int, int]] = set()

    for idx in order:
        r0, c0 = positions[idx]
        gap_data = _gap_data_for_block(obstacles, bg, n, r0, c0)
        combos = _candidate_gap_combos(gap_data, occupied)
        combo = _choose_gap_combo(combos, rng)
        if combo is None:
            return None
        chosen_gaps[idx] = combo
        for gap in combo:
            occupied.update(gap_data[gap][0])

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        assert gaps is not None
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_LINE_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D6 input")
