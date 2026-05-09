# Combo variant of 65b59efc (index 5)
# combo_spec: D3/2+D19/1
# Name: 双重间接颜色映射
from __future__ import annotations

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original task used marker color -> template index directly, then palette column j -> template j.
#    Here the chain is two-step and indirect: marker color matches a palette swatch color, and that swatch's
#    physical column must then be reversed to recover the template index. Example with K=3:
#    templates left->right = [T0,T1,T2], logical palette per template = [A,B,C], but bottom swatches left->right
#    are [C,B,A]. A marker colored A selects T0, even though A appears under the rightmost palette column.
#    The marker's local (lr,lc) still chooses the output block position exactly as before.
# b) Complexity calibration:
#    Palette colors are sampled distinct and disjoint from template colors, so the new marker coding is visible.
#    Template masks are forced distinct, so reversing palette/template correspondence changes observable shapes.
# c) Broken assumptions from the original solver:
#    - `tcol_to_j[v]` no longer works because marker colors are palette colors, not template colors.
#    - `pcols[j]` is no longer the recolor for template j in physical order; the correct logical palette is `pcols[::-1]`.
#    - Any solver path that assumes "same color space" for templates and markers will fail on this combo.
# d) Placement legality:
#    Slots remain unique by (lr,lc); each template is assigned to one tile column; swatches stay inside their
#    N-wide palette cells; dashed separators and optional trailing padding are unchanged.
# e) Visual observability:
#    Humans can match marker colors to the bottom swatches because those colors never appear in the top templates.
#    They can then compare output shapes with top templates and discover that the matched swatch maps to the
#    opposite-side template order.
# f) Visual cleanliness:
#    The layout still uses the original three clean bands. We avoid extra overlays and keep one uniform marker
#    color per middle tile column, so the indirect mapping is visible without clutter.
# g) Anti-degeneracy guard:
#    Generator guarantees every template appears at least once, palette colors are distinct, and template masks are
#    pairwise different. That makes the reversed palette/template relation materially affect the produced output.

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.arc_utils import paste, random_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled legend where marker colors point to palette swatches, then reverse that palette order to recover templates.

    Input structure:
    - The input is split into K tile columns of width N with dashed separator columns between them.
    - Top band (rows 0..N-1): K template masks, one template per tile column, each drawn in its own template color.
    - Middle band (rows N+1..2N): sparse marker pixels; each tile column uses a single marker color, but that color comes from the palette band rather than the template band.
    - Bottom band (rows 2N+2..): K palette swatches; read physically from left to right, they correspond to templates in reverse order.

    Rule:
    - A marker at local coordinates (lr, lc) stamps one N x N block into the output at block position (lr, lc).
    - To decide which template to stamp, first match the marker color to a palette swatch color, then map that swatch's reversed palette position back to its template.
    - Recolor the recovered template with that matched palette color before placing it into the output block.

    Key insight: The marker does not name the template directly. It names a palette entry, and the palette itself must be read right-to-left against the top templates.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if (g[:, c] == sep).sum() >= H // 2]
    sep_rows = [r for r in range(H) if (g[r, :] == sep).sum() >= W // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    # Top band: extract templates in physical left-to-right order.
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

    # Bottom band: physical left-to-right swatches must be reversed to align with templates.
    pal_r0 = sep_rows[-1] + 1
    physical_pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:H, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        physical_pcols.append(vals.pop())
    assert len(set(physical_pcols)) == k

    logical_pcols = physical_pcols[::-1]
    marker_to_template = {col: j for j, col in enumerate(logical_pcols)}

    # Middle band: marker color identifies the logical palette entry, not the template color.
    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        tile_marker = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_marker is None or v == tile_marker, "tile column must have uniform marker color"
                assert v in marker_to_template, "marker color must match a palette swatch color"
                tile_marker = v
                j = marker_to_template[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], logical_pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 300
_MAX_TEMPLATE_RESAMPLES = 80


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


def _sample_distinct_templates(n: int, tcols: list[int]) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    seen = set()
    for col in tcols:
        placed = False
        for _ in range(_MAX_TEMPLATE_RESAMPLES):
            mask = _sample_template(n, col)
            key = tuple(mask.reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            templates.append(mask)
            placed = True
            break
        if not placed:
            return None
    return templates


def _build_expected(
    n: int, templates: list[np.ndarray], logical_pcols: list[int], placements: list[tuple[int, int, int]]
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], logical_pcols[j], _BG)
        paste(expected, tile, rr * n, cc * n)
    return expected


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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    palette_pool = [c for c in _AVAILABLE_COLORS if c not in tcols]
    logical_pcols = random.sample(palette_pool, k)
    physical_pcols = logical_pcols[::-1]

    templates = _sample_distinct_templates(n, tcols)
    if templates is None:
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, logical_pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = logical_pcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for physical_idx, col in enumerate(physical_pcols):
        start = physical_idx * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 65b59efc input")
