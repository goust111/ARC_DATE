# Combo variant of 65b59efc (index 30)
# combo_spec: D15/0+D18/1
# Name: 稀疏Voronoi填充

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_sprite


# Design analysis for the fused variant:
# - Concrete behavior trace:
#   In the original task, if N=3 and only three markers land at slots (0,0), (0,2), (2,1),
#   only those three 3x3 output blocks are stamped and the other six blocks stay background.
#   Here the same sparse markers act as Voronoi seeds instead: every empty slot chooses the
#   nearest marked slot by Manhattan distance, so all nine output blocks get stamped.
# - Complexity calibration:
#   The middle band now uses only 20-40% of slots, but each template still appears at least
#   once and palette colors are disjoint from template colors. This keeps the examples sparse
#   without becoming ambiguous, and the filled output visibly differs from the original task.
# - Hidden-assumption audit:
#   The old solver assumed "no marker means blank output". That assumption is removed; we must
#   first collect all marked slots, then propagate ownership to every slot. We also avoid
#   nearest-neighbor ties in the generator so the Voronoi partition stays human-readable.
# - Placement legality and observability:
#   Markers still occupy unique local slots, stay inside the N x N middle band, and each middle
#   tile column contains only one marker color. Distinct template/palette sets and large numbers
#   of propagated blocks make the new rule visible from a few examples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a sparse stamp map: read template masks, sparse marker seeds, and palette swatches,
    then expand the seeds into a fully tiled Voronoi-style output.

    Input structure:
    - The grid is split into K tile columns of width N, with dashed separator columns between them.
    - Rows 0..N-1 contain K template masks, one template per tile column.
    - Rows N+1..2N contain sparse marker pixels; only about 20-40% of the local N x N slots are used.
    - Rows below the second dashed separator contain K palette swatches aligned with the template columns.
    - Dashed separator rows sit at row N and row 2N+1.

    Rule:
    - A marker's color chooses which template to use, and its local slot (lr, lc) names one output block.
    - Marked slots are stamped directly; every unmarked output slot inherits the template of the marked slot
      with smallest Manhattan distance, using uppermost/leftmost seed only as a deterministic tie-break.
    - Each chosen template is recolored with its aligned palette color before being stamped.

    Key insight:
    - The middle band no longer lists every output block. It gives only sparse seeds, and the real output
      comes from propagating those seed identities across the full N x N block lattice by nearest distance.
    """
    bg, _sep, n, _starts, tmasks, pcols, seeds = _decode_scene(grid)
    assert seeds, "at least one marker seed is required"
    placements, _unique = _voronoi_assignments(n, seeds)
    return _render_output(bg, n, tmasks, pcols, placements)


# --------------- generator ---------------

# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- CFG: grid dimensions --
_N_RANGE = (3, 5)
_K_VALUES = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

# -- CFG: template generation --
_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_TEMPLATE_RETRIES = 80

# -- CFG: marker placement --
_MIN_SLOT_COVERAGE = 0.20
_MAX_SLOT_COVERAGE = 0.40
_PLACEMENT_RETRIES = 150

# -- CFG: palette swatch --
_SWATCH_DOUBLE_PROB = 0.35

# -- CFG: retry --
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _sample_template(n: int, color: int) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[color],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=_BG,
    )
    return sprite == color


def _sample_templates(n: int, tcols: list[int]) -> list[np.ndarray]:
    for _ in range(_TEMPLATE_RETRIES):
        masks = [_sample_template(n, col) for col in tcols]
        if len({mask.tobytes() for mask in masks}) == len(masks):
            return masks
    return [_sample_template(n, col) for col in tcols]


def _voronoi_assignments(
    n: int, seeds: list[tuple[int, int, int]]
) -> tuple[list[tuple[int, int, int]], bool]:
    placements: list[tuple[int, int, int]] = []
    unique_nearest = True
    for rr in range(n):
        for cc in range(n):
            best_seed = None
            best_key = None
            best_dist = None
            dist_tie = False
            for j, sr, sc in seeds:
                dist = abs(rr - sr) + abs(cc - sc)
                key = (dist, sr, sc, j)
                if best_key is None or key < best_key:
                    best_seed = (j, sr, sc)
                    best_key = key
                    best_dist = dist
                    dist_tie = False
                elif dist == best_dist:
                    dist_tie = True
            assert best_seed is not None
            if dist_tie:
                unique_nearest = False
            placements.append((best_seed[0], rr, cc))
    return placements, unique_nearest


def _render_output(
    bg: int,
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], bg)
        paste(out, tile, rr * n, cc * n)
    return out


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _decode_scene(
    grid: np.ndarray,
) -> tuple[int, int, int, list[int], list[np.ndarray], list[int], list[tuple[int, int, int]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if (g[:, c] == sep).sum() >= h // 2]
    sep_rows = [r for r in range(h) if (g[r, :] == sep).sum() >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    mid_r0 = n + 1
    slot_owner: dict[tuple[int, int], int] = {}
    seeds: list[tuple[int, int, int]] = []
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
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                seeds.append((j, lr, lc))

    return bg, sep, n, starts, tmasks, pcols, seeds


def variant_metrics(grid: np.ndarray) -> dict[str, float | int | bool]:
    bg, _sep, n, _starts, tmasks, pcols, seeds = _decode_scene(grid)
    dense_placements, unique_nearest = _voronoi_assignments(n, seeds)
    sparse_out = _render_output(bg, n, tmasks, pcols, seeds)
    dense_out = _render_output(bg, n, tmasks, pcols, dense_placements)
    return {
        "n": n,
        "seed_count": len(seeds),
        "slot_count": n * n,
        "coverage": len(seeds) / float(n * n),
        "propagated_slots": n * n - len(seeds),
        "unique_nearest": unique_nearest,
        "voronoi_added_pixels": int(np.count_nonzero(dense_out != sparse_out)),
    }


def _sample_seed_layout(
    n: int, k: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    max_marks = int(np.floor(_MAX_SLOT_COVERAGE * n * n))
    if min_marks > max_marks:
        return None

    for _ in range(_PLACEMENT_RETRIES):
        mark_count = random.randint(min_marks, max_marks)
        used_slots = random.sample(all_slots, mark_count)
        labels = list(range(k))
        labels.extend(random.randrange(k) for _ in range(mark_count - k))
        random.shuffle(labels)
        seeds = [(j, rr, cc) for (rr, cc), j in zip(used_slots, labels)]
        placements, unique_nearest = _voronoi_assignments(n, seeds)
        if not unique_nearest:
            continue

        region_sizes = Counter(j for j, _rr, _cc in placements)
        seed_sizes = Counter(j for j, _rr, _cc in seeds)
        grown_templates = sum(region_sizes[j] > seed_sizes[j] for j in range(k))
        if grown_templates < 2:
            continue
        return seeds, placements
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    feasible_k = [k for k in _K_VALUES if k <= int(np.floor(_MAX_SLOT_COVERAGE * n * n))]
    if not feasible_k:
        return None
    k = random.choice(feasible_k)

    tcols = random.sample(_AVAILABLE_COLORS, k)
    remaining = [c for c in _AVAILABLE_COLORS if c not in tcols]
    pcols = random.sample(remaining, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = _sample_templates(n, tcols)
    sampled = _sample_seed_layout(n, k)
    if sampled is None:
        return None
    seeds, placements = sampled
    expected = _render_output(_BG, n, templates, pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in seeds:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        metrics = variant_metrics(grid)
        if (
            np.array_equal(main(grid), expected)
            and metrics["coverage"] >= _MIN_SLOT_COVERAGE
            and metrics["coverage"] <= _MAX_SLOT_COVERAGE
            and metrics["propagated_slots"] > 0
            and metrics["voronoi_added_pixels"] > 0
            and metrics["unique_nearest"]
        ):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-Voronoi 65b59efc input")
