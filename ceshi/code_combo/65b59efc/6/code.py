# Combo variant of 65b59efc (index 6)
# combo_spec: D4/0+D10/0
# Name: 实线网格化输出

from __future__ import annotations

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Original input: K template columns of width N are separated by 1-cell dashed lines, and the output is a
#      plain (N*N) x (N*N) stamp board with no visible boundaries between blocks.
#    - Fused input: the same three-band layout is kept, but every separator row and separator column is now a
#      solid line of the separator color. Example with N=3, K=3: columns 3 and 7 are fully separator-colored,
#      and rows 3 and 7 are fully separator-colored across the core width. The middle-band markers are still
#      read by local coordinates inside each tile column.
#    - Fused output: each logical destination block still has size N x N, but adjacent blocks are split by
#      1-pixel solid separator lines. For N=3 the output is 11 x 11 instead of 9 x 9, with separator rows and
#      columns at indices 3 and 7. A marker at local slot (1, 2) now stamps into rows 4..6 and cols 8..10.
#    - Difference: D10 changes how the tiled legend is visually partitioned in the input; D4 changes how the
#      decoded stamp board is laid out in the output. The marker/template/palette logic itself is unchanged.
#
# b) Complexity calibration
#    - The variant is not simplified into mere cosmetic noise: the input visibly uses continuous grid lines,
#      and the output size changes from N*N to N*N+N-1 along each axis.
#    - The generator still uses 3-4 templates and 60%-100% slot coverage, so multiple block placements repeat
#      the same structural cue and make the inserted output grid easy to spot from a few examples.
#    - Template masks are sampled distinctly to avoid ambiguous demonstrations where two template columns would
#      look identical and reduce the value of the marker-color lookup.
#
# c) Implicit-assumption audit against the original solver
#    - `bg = most_common_color` still works only if the generator keeps background dominant despite solid
#      separators becoming denser. The generator therefore caps template density and allows trailing bg padding.
#    - `sep = most_common_non_bg_color` still holds because solid separator lines contribute far more pixels than
#      any single template color, marker color, or palette swatch color. The generator enforces that.
#    - `sep_cols[0] == N` still holds because the first vertical separator remains immediately after the first
#      N-wide tile column.
#    - The old output write `out[lr*N:(lr+1)*N, lc*N:(lc+1)*N]` is no longer valid. The new solver must offset
#      each block start by the number of inserted separator lines before it.
#
# d) Placement legality
#    - Input legality is unchanged structurally: templates stay inside the top N rows, markers stay inside the
#      middle N rows, swatches stay inside the palette band, and separator lines occupy only the dedicated row
#      and column slots between tile columns / bands.
#    - Output legality requires explicit coordinate conversion. For block index rr, the top-left output row is
#      `rr * (N + 1)`; similarly for columns. This guarantees no stamp overlaps any separator line or exceeds
#      the enlarged `(N*N+N-1) x (N*N+N-1)` canvas.
#
# e) Visual observability
#    - Humans can immediately see the input is partitioned by continuous grid bars rather than broken dashes.
#    - The output difference is also directly visible because repeated N x N stamps are separated by a uniform
#      separator-colored lattice. This is much easier to infer from examples than a hidden arithmetic rule.
#
# f) Visual cleanliness
#    - Only one separator style exists in each sample: solid in the input and solid in the output.
#    - The extra output lines live only between blocks, so they do not obscure template interiors.
#    - Palette swatches remain tiny centered marks, preventing the lower band from becoming cluttered.
#
# g) Generator guarantees for main()
#    - Background remains the dominant color, preserving the solver's bg/sep inference path.
#    - Every template color is unique, every sampled template mask is distinct, and every template receives at
#      least one marker placement, so the decoder sees all intended correspondences.
#    - The output renderer always inserts separator rows/cols at every block boundary, ensuring D4 is present
#      in every sample rather than appearing only incidentally.

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
    Decode a solid-gridded stamp legend, then rebuild the stamp board with solid grid lines between output blocks.

    Input structure:
    - The input is arranged as K tile columns of width N, with 1-cell solid separator columns between them.
    - Rows 0..N-1 contain the K template masks, one monochrome template per tile column.
    - Row N is a solid separator row; rows N+1..2N hold marker pixels; row 2N+1 is another solid separator row.
    - Rows below the second separator contain palette swatches aligned with the template columns.

    Rule:
    - A marker's color identifies which top template to use, and its local coordinates (lr, lc) inside its tile
      column identify which logical output block to fill.
    - The chosen template is recolored by the palette swatch from the same tile column as that template.
    - The output is an N-by-N array of N-by-N stamp blocks, but every adjacent pair of blocks is split by a
      1-pixel separator line in the separator color.

    Key insight:
    - The legend decoding is the same template/marker/palette lookup as the base task; the new step is to map
      each logical block index onto a larger gridded canvas whose separator rows and columns must remain intact.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if int((g[:, c] == sep).sum()) >= H // 2]
    sep_rows = [r for r in range(H) if int((g[r, :] == sep).sum()) >= W // 2]
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
        sub = g[pal_r0:H, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    out = _blank_gridded_output(n, bg, sep)
    mid_r0 = n + 1
    slot_owner: dict[tuple[int, int], int] = {}

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
                tile = np.where(tmasks[j], pcols[j], bg)
                _paste_output_block(out, tile, lr, lc, n)

    return out


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.32, 0.56)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MAX_TEMPLATE_RESAMPLES = 100

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 400


def _solid_sep_positions(n: int) -> list[int]:
    return [i * (n + 1) - 1 for i in range(1, n)]


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _template_cell_bounds(n: int) -> tuple[int, int]:
    min_fg = max(2, (n * n) // 4)
    max_fg = (n * n + 1) // 2
    return min_fg, max_fg


def _sample_template(n: int, color: int) -> np.ndarray:
    min_fg, max_fg = _template_cell_bounds(n)
    for _ in range(_MAX_TEMPLATE_RESAMPLES):
        try:
            sprite = random_sprite(
                n,
                n,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[color],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=_BG,
            )
        except AssertionError:
            # `random_sprite` can rarely fail its own contiguity assertion; treat that as a bad draw.
            continue
        mask = sprite == color
        cells = int(mask.sum())
        if min_fg <= cells <= max_fg:
            return mask
    raise RuntimeError("failed to sample a template within the density bounds")


def _sample_distinct_templates(n: int, tcols: list[int]) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for col in tcols:
        placed = False
        for _ in range(_MAX_TEMPLATE_RESAMPLES):
            mask = _sample_template(n, col)
            key = tuple(int(v) for v in mask.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            templates.append(mask)
            placed = True
            break
        if not placed:
            return None
    return templates


def _blank_gridded_output(n: int, bg: int, sep: int) -> np.ndarray:
    size = n * n + (n - 1)
    out = np.full((size, size), bg, dtype=int)
    for idx in _solid_sep_positions(n):
        out[idx, :] = sep
        out[:, idx] = sep
    return out


def _paste_output_block(out: np.ndarray, tile: np.ndarray, rr: int, cc: int, n: int) -> None:
    r0 = rr * (n + 1)
    c0 = cc * (n + 1)
    paste(out, tile, r0, c0)


def _build_expected(
    n: int, templates: list[np.ndarray], pcols: list[int], placements: list[tuple[int, int, int]]
) -> np.ndarray:
    expected = _blank_gridded_output(n, _BG, _SEP)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        _paste_output_block(expected, tile, rr, cc, n)
    return expected


def _draw_solid_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        grid[r, :core_w] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        grid[:core_h, c] = _SEP


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)

    templates = _sample_distinct_templates(n, tcols)
    if templates is None:
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    h = core_h + random.randint(0, _MAX_ROW_PAD)
    w = core_w + random.randint(0, _MAX_COL_PAD)

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_solid_separators(grid, n, k, core_h, core_w)

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
    except Exception:
        return None

    if np.array_equal(solved, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 65b59efc input")
