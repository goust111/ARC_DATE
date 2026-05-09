# Combo variant of 65b59efc (index 21)
# combo_spec: D2/3+D15/2
# Name: 蛇形映射区域分组
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


# Design analysis:
# a) Concrete behavior trace:
#    - Original 65b59efc:
#      1. Read K templates from the top band and K palette colors from the bottom band.
#      2. Read middle-band markers; marker color chooses the template.
#      3. A marker at local tile coordinate (lr, lc) stamps directly into output block (lr, lc).
#    - Fused behavior:
#      1. Template/color reading stays the same, so the visible objects remain interpretable.
#      2. The changed step is the slot lookup: local coordinates are first linearized in ordinary
#         row-major order, but output blocks are indexed in diagonal snake order.
#      3. Example for N=3:
#         row-major local indices are
#         0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2) 6:(2,0) 7:(2,1) 8:(2,2)
#         diagonal-snake output indices are
#         0:(0,0) 1:(0,1) 2:(1,0) 3:(2,0) 4:(1,1) 5:(0,2) 6:(1,2) 7:(2,1) 8:(2,2)
#         so a marker at local (0,2) now lands at output block (1,0), not (0,2).
#      4. The second fused change is meta-structure: each template's markers are confined to one
#         contiguous row-band or one contiguous column-band. The grouping does not replace the
#         color/template cue; it is a repeated spatial clue that helps a human notice that marker
#         positions are being "read" as indexed slots rather than stamped literally.
# b) Complexity calibration:
#    - Bands are chosen from one axis only per sample and form a clean partition of the N x N slot
#      grid. This makes the grouping visually obvious instead of noisy.
#    - We enforce K <= N so every template can own a non-empty band; this avoids degenerate tiny
#      regions for N=3, K=4.
#    - Every template receives at least two markers when possible, so the grouping signal repeats.
# c) Original assumptions broken and replaced:
#    - `output block == (lr, lc)` is false. The solver must convert `(lr, lc)` to row-major index
#      `lr*N + lc`, then map that index through the snake ordering.
#    - Random marker scattering is false. The generator now guarantees per-template row/column
#      bands, and the solver may safely assume each tile column still has one template color and at
#      least one marker so the color-to-template permutation remains observable.
# d) Placement legality:
#    - Band partitions are disjoint, so marker slots remain unique across templates.
#    - Templates and palettes stay inside their original reserved bands; separators still occupy
#      only the dashed divider rows/columns.
#    - The output still places at most one stamp per block because every local slot has one owner.
# e) Visual observability:
#    - The grouped bands are large contiguous slices, easy to spot across 2-4 samples.
#    - Because the snake order differs from row-major on most indices, the input/output relation is
#      visibly different from the base task even when colors and templates are read correctly.
# f) Visual cleanliness:
#    - The layout remains the original three-band tiled scaffold; only the placement logic and
#      marker distribution change.
#    - Using full row/column bands avoids tangled overlaps or tiny special-case symbols.
# g) Non-degeneracy targets:
#    - Generator-side oracle rejects any sample whose fused output matches the old direct mapping.
#    - Post-run diagnostics can therefore measure how often `main(grid) != _solve_direct(grid)` and
#      how small each template's row/column span is inside the marker band.


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
_GROUP_AXES = ("row", "col")
_NEXT_GROUP_AXIS = 0


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _snake_coords(n: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for diag in range(2 * n - 1):
        r0 = max(0, diag - (n - 1))
        r1 = min(n - 1, diag)
        if diag % 2 == 0:
            rows = range(r1, r0 - 1, -1)
        else:
            rows = range(r0, r1 + 1)
        for r in rows:
            coords.append((r, diag - r))
    return coords


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


def _parse_problem(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[int], list[int], list[np.ndarray], list[int], int]:
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

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    mid_r0 = n + 1
    return g, bg, sep, n, starts, tcols, tmasks, pcols, mid_r0


def _collect_markers(
    g: np.ndarray,
    bg: int,
    sep: int,
    n: int,
    starts: list[int],
    tcol_to_j: dict[int, int],
    mid_r0: int,
) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for st in starts:
        tile_color = None
        tile_markers = 0
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                tile_markers += 1
                markers.append((tcol_to_j[v], lr, lc))
        assert tile_markers > 0, "each template must remain visible in the marker band"
    return markers


def _render_from_markers(
    n: int,
    bg: int,
    tmasks: list[np.ndarray],
    pcols: list[int],
    markers: list[tuple[int, int, int]],
    *,
    use_snake: bool,
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    stamps = [np.where(mask, pcols[j], bg) for j, mask in enumerate(tmasks)]
    snake = _snake_coords(n)

    for j, lr, lc in markers:
        rr, cc = snake[lr * n + lc] if use_snake else (lr, lc)
        assert (rr, cc) not in slot_owner or slot_owner[(rr, cc)] == j
        slot_owner[(rr, cc)] = j
        paste(out, stamps[j], rr * n, cc * n)
    return out


def _solve_direct(grid: np.ndarray) -> np.ndarray:
    g, bg, sep, n, starts, tcols, tmasks, pcols, mid_r0 = _parse_problem(grid)
    tcol_to_j = {col: j for j, col in enumerate(tcols)}
    markers = _collect_markers(g, bg, sep, n, starts, tcol_to_j, mid_r0)
    return _render_from_markers(n, bg, tmasks, pcols, markers, use_snake=False)


def _extract_marker_sets(grid: np.ndarray) -> tuple[int, dict[int, list[tuple[int, int]]]]:
    g, bg, sep, n, starts, tcols, _tmasks, _pcols, mid_r0 = _parse_problem(grid)
    tcol_to_j = {col: j for j, col in enumerate(tcols)}
    markers = _collect_markers(g, bg, sep, n, starts, tcol_to_j, mid_r0)
    by_template = {j: [] for j in range(len(tcols))}
    for j, lr, lc in markers:
        by_template[j].append((lr, lc))
    return n, by_template


def grouping_span_ratios(grid: np.ndarray) -> list[float]:
    n, by_template = _extract_marker_sets(grid)
    ratios: list[float] = []
    for coords in by_template.values():
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        row_span = max(rows) - min(rows) + 1
        col_span = max(cols) - min(cols) + 1
        ratios.append(min(row_span, col_span) / n)
    return ratios


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a grouped stamp map: read template masks, palette colors, and template-colored markers,
    then render the recolored templates into an N x N block canvas using a diagonal snake index.

    Input structure:
    - The grid is split into K tile columns of width N with dashed separator columns between them.
    - Rows 0..N-1 hold the K template masks, one monochrome template per tile column.
    - Rows N+1..2N hold marker pixels; each tile column still uses exactly one marker color, but
      the occupied cells are concentrated into one row-band or one column-band for that template.
    - Rows 2N+2.. contain K palette swatches aligned with the template columns.
    - Dashed separator rows still appear at rows N and 2N+1.

    Rule:
    - Marker color identifies which template/palette pair to use, exactly as in the base task.
    - The changed part is placement: local marker coordinates are read in ordinary row-major order,
      but output blocks are numbered by diagonal snake order. If local index `i = lr*N + lc`, stamp
      the template into the `i`-th snake slot instead of block `(lr, lc)`.

    Key insight: The marker colors still tell you "which stamp", but the marker positions no longer
    tell you the output block literally; they must be re-indexed through the diagonal snake order.
    The per-template row/column grouping in the middle band is a visual clue, not the direct key.
    """
    g, bg, sep, n, starts, tcols, tmasks, pcols, mid_r0 = _parse_problem(grid)
    tcol_to_j = {col: j for j, col in enumerate(tcols)}
    markers = _collect_markers(g, bg, sep, n, starts, tcol_to_j, mid_r0)
    return _render_from_markers(n, bg, tmasks, pcols, markers, use_snake=True)


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    return _render_from_markers(n, _BG, templates, pcols, placements, use_snake=True)


def _build_expected_direct(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    return _render_from_markers(n, _BG, templates, pcols, placements, use_snake=False)


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


def _partition_axis(n: int, k: int) -> list[tuple[int, int]]:
    cuts = sorted(random.sample(range(1, n), k - 1))
    bounds = [0] + cuts + [n]
    return list(zip(bounds[:-1], bounds[1:]))


def _band_slots(n: int, axis: str, band: tuple[int, int]) -> list[tuple[int, int]]:
    lo, hi = band
    if axis == "row":
        return [(r, c) for r in range(lo, hi) for c in range(n)]
    return [(r, c) for c in range(lo, hi) for r in range(n)]


def _sample_grouped_placements(n: int, k: int, axis: str) -> list[tuple[int, int, int]]:
    bands = _partition_axis(n, k)
    random.shuffle(bands)
    candidates = [_band_slots(n, axis, band) for band in bands]

    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)

    counts = [min(2, len(cands)) for cands in candidates]
    assert sum(counts) <= mark_count

    remaining = mark_count - sum(counts)
    while remaining > 0:
        choices = [j for j, cands in enumerate(candidates) if counts[j] < len(cands)]
        j = random.choice(choices)
        counts[j] += 1
        remaining -= 1

    placements: list[tuple[int, int, int]] = []
    for j, cands in enumerate(candidates):
        chosen = random.sample(cands, counts[j])
        placements.extend((j, rr, cc) for rr, cc in chosen)
    random.shuffle(placements)
    return placements


def _next_group_axis() -> str:
    global _NEXT_GROUP_AXIS
    axis = _GROUP_AXES[_NEXT_GROUP_AXIS % len(_GROUP_AXES)]
    _NEXT_GROUP_AXIS += 1
    return axis


def _gen_attempt(axis: str) -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(_K_RANGE[0], min(_K_RANGE[1], n))
    colors = random.sample(_AVAILABLE_COLORS, 2 * k)
    tcols = colors[:k]
    pcols = colors[k:]

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = [_sample_template(n, col) for col in tcols]
    placements = _sample_grouped_placements(n, k, axis)

    expected = _build_expected(n, templates, pcols, placements)
    direct_expected = _build_expected_direct(n, templates, pcols, placements)
    if np.array_equal(expected, direct_expected):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        solved = main(grid)
        if np.array_equal(solved, expected) and not np.array_equal(solved, _solve_direct(grid)):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    axis = _next_group_axis()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(axis)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
