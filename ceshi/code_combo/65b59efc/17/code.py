# Combo variant of 65b59efc (index 17)
# combo_spec: D6/1+D7/3
# Name: 互补模板OR合并

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Suppose N=4 and K=3. The top band shows three template masks that partition the 4x4 tile:
#    template A might own the left strip, B the top-right corner, C the bottom-right corner.
#    In the original task, slot (1,2) could be claimed by at most one template, so the output
#    block at row-block 1 / col-block 2 would contain only one recolored mask on black.
#    In this combo, multiple middle-band columns may place markers at the same local slot (1,2).
#    If A and C both mark that slot, their recolored masks are OR-combined pixelwise, producing
#    a two-color partial mosaic. If A, B, and C all mark that slot, the complementary masks fill
#    the whole 4x4 block with no background left.
# b) Complexity calibration:
#    The generator always includes repeated multi-template overlaps, at least one full-stack slot,
#    and usually "all except one" slots. This keeps the new rule visible while avoiding an
#    overcomplicated subset language. Template and palette colors are chosen as disjoint sets so
#    the input coding colors and output paint colors stay visually separable.
# c) Hidden assumptions audited:
#    - Original `slot_owner` exclusivity is invalid now and must be removed.
#    - Original template density assumptions are no longer physically compatible with K=3..4
#      complementary coverage, so the generator switches from dense random sprites to connected
#      partition regions.
#    - The solver now assumes the top masks are pairwise disjoint and their union covers the full
#      NxN tile. The generator explicitly constructs and validates that partition.
#    - Middle-band tile columns still stay color-uniform, preserving the original template-id
#      decoding mechanism.
# d) Placement legality:
#    Each marker is still written inside the N x N middle band of one tile column, separators are
#    untouched, and output blocks are addressed only by legal local coordinates. Multiple markers
#    may target the same output block, but they come from different tile columns and therefore do
#    not overlap in the input grid.
# e) Observability:
#    Humans can see the complementary partition directly in the top band, then notice repeated
#    same-slot markers across several columns, and finally see partial/full mosaics in the output.
#    A full-stack slot yields a fully filled block, making the complementary OR behavior explicit.
# f) Visual cleanliness:
#    Connected partition pieces avoid noisy speckled templates. OR never causes same-pixel color
#    fights because the active cells of different templates are disjoint by construction.
# g) Inferability of color changes:
#    Palette colors are aligned by bottom-band columns exactly as in the base task. Because each
#    template also appears in singleton slots, the mapping from a top-band partition piece to its
#    output color remains recoverable before the mixed slots need to be explained.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend in which several complementary templates may be stacked into the same output block.

    Input structure:
    - The input is split into K tile columns of width N, separated by 1-column dashed dividers.
    - Rows 0..N-1 contain K template masks; together those masks form a complementary partition of one N x N tile.
    - Rows N+1..2N contain marker pixels; different tile columns may place markers at the same local slot.
    - Rows 2N+2..bottom contain K palette swatches, one recolor for each template column.
    - Rows N and 2N+1 are dashed separator rows, matching the vertical dashed separators.

    Rule:
    - A marker's color still identifies which template it refers to, and its local coordinates (lr, lc) still pick
      one N x N block inside the output canvas.
    - When several markers from different tile columns land on the same (lr, lc), stamp every referenced template
      into that output block and combine them by pixelwise OR of activity.
    - Because the top-band masks are complementary, each active pixel belongs to exactly one template, so the merged
      block becomes a clean multi-color mosaic; if all K templates appear, the block is completely filled.

    Key insight:
    Read the top band as K disjoint pieces of a single partitioned tile, then accumulate all template markers that
    share an output slot. The block is not owned by one template anymore; it is the union of every complementary
    piece whose marker points to that slot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if (g[:, c] == sep).sum() >= H // 2]
    sep_rows = [r for r in range(H) if (g[r, :] == sep).sum() >= W // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    # Read the complementary template partition from the top band.
    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    cover = np.zeros((n, n), dtype=bool)
    for st in starts:
        sub = g[0:n, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        mask = sub == col
        assert mask.any()
        assert not np.any(cover & mask), "templates must be pairwise disjoint"
        cover |= mask
        tcols.append(col)
        tmasks.append(mask)
    assert cover.all(), "template masks must cover the entire NxN tile"
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    # Read palette colors from the bottom band.
    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:H, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    # Collect every template marker that points at each output slot.
    mid_r0 = n + 1
    slot_templates: dict[tuple[int, int], set[int]] = {}
    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                slot_templates.setdefault((lr, lc), set()).add(j)

    out = np.full((n * n, n * n), bg, dtype=int)
    for (lr, lc), js in slot_templates.items():
        tile = np.full((n, n), bg, dtype=int)
        seen = np.zeros((n, n), dtype=bool)
        for j in sorted(js):
            assert not np.any(seen & tmasks[j]), "complementary masks should not overlap"
            tile[tmasks[j]] = pcols[j]
            seen |= tmasks[j]
        out[lr * n:(lr + 1) * n, lc * n:(lc + 1) * n] = tile

    return out


# --------------- generator ---------------

# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- CFG: grid dimensions --
_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

# -- CFG: complementary partition --
_MIN_REGION_CELLS = 2
_MAX_REGION_FRACTION = 0.45
_PARTITION_TRIES = 200

# -- CFG: slot coverage / overlap --
_MIN_SLOT_COVERAGE = 0.75
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 300


def _dash_cell(pos: int, n: int) -> bool:
    """True iff position `pos` receives a separator pip for tile size `n`."""
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    """Count dashed separator pips inside a span of `length` cells."""
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    """Map (tile-column index, local column) to an absolute input-grid column."""
    return tile_col * (n + 1) + local_c


def _neighbors4(r: int, c: int, n: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < n:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < n:
        yield r, c + 1


def _grow_complementary_templates(n: int, k: int) -> list[np.ndarray]:
    """Generate K connected, pairwise-disjoint masks whose union is the full N x N tile."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    total = n * n
    max_region = math.ceil(total * _MAX_REGION_FRACTION)

    for _ in range(_PARTITION_TRIES):
        seeds = random.sample(cells, k)
        owner = np.full((n, n), -1, dtype=int)
        frontiers = [set() for _ in range(k)]
        sizes = [0] * k

        for j, (r, c) in enumerate(seeds):
            owner[r, c] = j
            sizes[j] = 1
        for j, (r, c) in enumerate(seeds):
            for nr, nc in _neighbors4(r, c, n):
                if owner[nr, nc] == -1:
                    frontiers[j].add((nr, nc))

        while np.any(owner == -1):
            candidates = [j for j in range(k) if frontiers[j]]
            if not candidates:
                break

            smallest = min(sizes[j] for j in candidates)
            if random.random() < 0.8:
                choices = [j for j in candidates if sizes[j] == smallest]
            else:
                choices = candidates
            j = random.choice(choices)

            frontier_cells = list(frontiers[j])
            frontier_cells.sort(
                key=lambda rc: (
                    -sum(owner[nr, nc] == -1 for nr, nc in _neighbors4(rc[0], rc[1], n)),
                    random.random(),
                )
            )
            r, c = frontier_cells[0]
            owner[r, c] = j
            sizes[j] += 1
            for q in range(k):
                frontiers[q].discard((r, c))
            for nr, nc in _neighbors4(r, c, n):
                if owner[nr, nc] == -1:
                    frontiers[j].add((nr, nc))

        if np.any(owner == -1):
            continue
        if min(sizes) < _MIN_REGION_CELLS:
            continue
        if max(sizes) > max_region:
            continue

        masks = [(owner == j) for j in range(k)]
        if all(mask.any() for mask in masks):
            return masks

    raise RuntimeError("failed to build complementary templates")


def _sample_slot_subsets(n: int, k: int) -> list[tuple[int, int, tuple[int, ...]]]:
    """Choose output slots and the template subsets that should OR together at each slot."""
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    groups: dict[tuple[int, int], tuple[int, ...]] = {}

    # Singleton evidence: lets a human recover each partition piece individually.
    for j in range(k):
        groups[all_slots.pop()] = (j,)

    # Full-stack evidence: guarantees a completely filled mosaic block.
    groups[all_slots.pop()] = tuple(range(k))

    # Missing-one evidence: shows the complementary relationship directly.
    for miss in range(k):
        if not all_slots:
            break
        groups[all_slots.pop()] = tuple(j for j in range(k) if j != miss)

    target = random.randint(max(len(groups), math.ceil(_MIN_SLOT_COVERAGE * n * n)), n * n)
    while len(groups) < target and all_slots:
        size_choices = [s for s in range(2, k + 1)]
        size_weights = [4 if s == k else 3 if s == k - 1 else 2 for s in size_choices]
        size = random.choices(size_choices, weights=size_weights)[0]
        subset = tuple(sorted(random.sample(range(k), size)))
        groups[all_slots.pop()] = subset

    items = [(rr, cc, js) for (rr, cc), js in groups.items()]
    random.shuffle(items)
    return items


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    slot_subsets: list[tuple[int, int, tuple[int, ...]]],
) -> np.ndarray:
    """Render the OR-merged output grid from slot subsets."""
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for rr, cc, js in slot_subsets:
        tile = np.full((n, n), _BG, dtype=int)
        for j in js:
            tile[templates[j]] = pcols[j]
        paste(expected, tile, rr * n, cc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    """Draw the dashed horizontal and vertical separators in-place."""
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    color_pool = random.sample(_AVAILABLE_COLORS, 2 * k)
    tcols = color_pool[:k]
    pcols = color_pool[k:]

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    try:
        templates = _grow_complementary_templates(n, k)
    except RuntimeError:
        return None
    slot_subsets = _sample_slot_subsets(n, k)
    expected = _build_expected(n, templates, pcols, slot_subsets)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    # Top band: complementary template masks.
    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    # Middle band: each template owns one tile column, but several columns may mark the same slot.
    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for rr, cc, js in slot_subsets:
        for j in js:
            grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    # Bottom band: palette swatches.
    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0:c0 + size] = col

    try:
        solved = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
