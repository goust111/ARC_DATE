# Combo variant of 65b59efc (index 11)
# combo_spec: D4/2+D11/0
# Name: 水平布局矩形输出
#
# Design analysis:
# a) Concrete behavior and trace:
#    - Base task: K template tiles were arranged across the top, K marker tiles across the middle,
#      and K palette tiles across the bottom. A marker at local (lr, lc) inside an N x N marker tile
#      stamped an N x N template into output block (lr, lc), so the output always had N block rows and
#      N block columns.
#    - Fused task: the legend is rotated into three left-to-right bands:
#      template band | marker band | palette band. The K template/marker/palette records are stacked as
#      horizontal strips separated by dashed rows. Each strip has height R, the marker band has width C,
#      and the template itself still has size N x N in the left band.
#    - Small trace: if N=3, R=5, C=4 and a blue marker appears at local (4, 1) inside the middle band
#      of one strip, the solver now stamps the blue strip's template into output rows 12..14 and cols
#      3..5. The same input idea in the base task could never target row-slot 4 because the middle band
#      height there was tied to N=3.
# b) Complexity calibration:
#    - Only one new degree of freedom is introduced per axis: strip height R controls output block rows,
#      marker-band width C controls output block columns, and we enforce R != C so the rectangle is
#      visible in every sample.
#    - Template size N stays small and readable, so the puzzle remains "decode legend then stamp" rather
#      than becoming a general packing problem.
# c) Hidden assumption review:
#    - Original code assumed `sep_rows` meant band boundaries; now they mean strip boundaries because the
#      three functional regions are split by columns, not by rows.
#    - Original `N = first separator column` still works for the left template band width, but `N` no
#      longer determines the middle band's height or width. The solver must separately infer:
#      * template size N from the first separator column,
#      * output block-row count R from the first separator row,
#      * output block-column count C from the distance between the two separator columns.
#    - Original output allocation `np.full((N*N, N*N), ...)` is invalid. The fused solver must allocate
#      `(R*N, C*N)` and index blocks with `(lr, lc)` over `R x C`.
# d) Placement legality:
#    - The generator uses strips of exact height R, two separator columns, and one marker rectangle of
#      size R x C per strip, so no region overlaps any other region.
#    - Marker slots are globally unique across templates, preventing two templates from competing for one
#      output block. Each used template gets at least one marker.
# e) Visual observability:
#    - Horizontal parsing is explicit because every sample has exactly two separator columns but K-1
#      separator rows.
#    - Rectangular output is explicit because every sample enforces R != C, so the output aspect ratio
#      cannot collapse back to the square base behavior.
# f) Visual cleanliness:
#    - Each strip carries exactly one template on the left, one monochrome marker field in the middle,
#      and one small palette swatch on the right. The blank padding under shorter templates is confined
#      to the template band and does not mix with markers.
# g) Color inferability:
#    - Marker colors remain identical to template colors, and palette colors are sampled disjointly from
#      template colors, so the recoloring cue stays simple and visible.
# h) Generator-side new assumptions:
#    - `R >= N`, so the N x N template always fits inside the left strip area.
#    - There are exactly two separator columns, making `(N, C)` uniquely recoverable from geometry.
#    - Separator rows/cols use a simple even-position dash pattern, so frequency-based `sep` detection
#      remains reliable despite the rectangular marker band.

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


def _find_separator_geometry(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    for color, _cnt in counts.most_common():
        color = int(color)
        if color == bg:
            continue
        sep_cols = [c for c in range(w) if int((g[:, c] == color).sum()) >= (h + 1) // 2]
        sep_rows = [r for r in range(h) if int((g[r, :] == color).sum()) >= (w + 1) // 2]
        if len(sep_cols) == 2 and sep_rows:
            return color, sep_cols, sep_rows
    raise AssertionError("unable to identify separator color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a horizontal stamp legend: read template strips, marker strips, and palette strips, then stamp a rectangle.

    Input structure:
    - The grid is split left-to-right into three bands separated by dashed separator columns:
      template band | marker band | palette band.
    - The K records are stacked top-to-bottom as horizontal strips separated by dashed separator rows.
    - In each strip, the left band contains one N x N template mask, the middle band is an R x C marker field,
      and the right band contains one palette swatch for that template.

    Rule:
    - A marker's color still selects which template strip it refers to, and each physical strip is still uniform
      in marker color when it contains any markers.
    - A marker's local coordinates (lr, lc) inside the R x C middle band choose the output block at row lr and
      column lc.
    - The selected template is recolored with that strip's palette color and stamped into an output grid of
      R block rows and C block columns, so the output size is (R*N) x (C*N).

    Key insight: The template size N, the output block-row count R, and the output block-column count C must be
    inferred from different geometric cues: the first separator column, the separator-row spacing, and the width
    of the middle band between the two separator columns.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep, sep_cols, sep_rows = _find_separator_geometry(g, bg)

    n = int(sep_cols[0])
    row_slots = int(sep_rows[0])
    col_slots = int(sep_cols[1] - sep_cols[0] - 1)
    assert n > 0 and row_slots >= n and col_slots > 0

    starts = [0] + [r + 1 for r in sep_rows]
    k = len(starts)

    marker_c0 = n + 1
    palette_c0 = sep_cols[1] + 1
    assert palette_c0 + n <= w

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for r0 in starts:
        sub = g[r0 : r0 + n, 0:n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = int(vals.pop())
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pcols: list[int] = []
    for r0 in starts:
        sub = g[r0 : r0 + row_slots, palette_c0 : palette_c0 + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(int(vals.pop()))

    out = np.full((row_slots * n, col_slots * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for r0 in starts:
        tile_color = None
        sub = g[r0 : r0 + row_slots, marker_c0 : marker_c0 + col_slots]
        for lr in range(row_slots):
            for lc in range(col_slots):
                v = int(sub[lr, lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "marker strip must stay monochrome"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator (combo D4/2 + D11/0) ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
# Keep row_slots strictly larger than template size so every sample visibly
# demonstrates that the output row-block count is inferred independently of N.
_ROW_EXTRA_CHOICES = [1, 2]
_COL_DELTA_CHOICES = [-1, 1, 2]

_DENSITY_RANGE = (0.55, 0.9)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.55
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500


def _dash_cell(pos: int) -> bool:
    return pos % 2 == 0


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


def _sample_palette_colors(k: int, used: set[int]) -> list[int]:
    return random.sample([c for c in _AVAILABLE_COLORS if c not in used], k)


def _build_expected(
    n: int,
    row_slots: int,
    col_slots: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((row_slots * n, col_slots * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        paste(out, tile, rr * n, cc * n)
    return out


def _draw_separators(grid: np.ndarray, n: int, row_slots: int, col_slots: int, k: int) -> None:
    h, w = grid.shape
    for i in range(k - 1):
        r = (i + 1) * row_slots + i
        for c in range(w):
            if _dash_cell(c):
                grid[r, c] = _SEP
    for c in (n, n + 1 + col_slots):
        for r in range(h):
            if _dash_cell(r):
                grid[r, c] = _SEP


def _choose_dimensions() -> tuple[int, int, int]:
    n = random.randint(*_N_RANGE)
    row_slots = n + random.choice(_ROW_EXTRA_CHOICES)
    col_choices = [max(2, n + delta) for delta in _COL_DELTA_CHOICES]
    col_choices = sorted({c for c in col_choices if c != row_slots})
    if not col_choices:
        col_choices = [c for c in range(2, n + 3) if c != row_slots]
    col_slots = random.choice(col_choices)
    return n, row_slots, col_slots


def _gen_attempt() -> np.ndarray | None:
    n, row_slots, col_slots = _choose_dimensions()
    k = random.randint(*_K_RANGE)

    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = _sample_palette_colors(k, set(tcols))
    templates = [_sample_template(n, col) for col in tcols]

    all_slots = [(rr, cc) for rr in range(row_slots) for cc in range(col_slots)]
    random.shuffle(all_slots)
    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * len(all_slots))))
    mark_count = random.randint(min_marks, len(all_slots))
    used_slots = all_slots[:mark_count]

    owners = list(range(k))
    while len(owners) < mark_count:
        owners.append(random.randrange(k))
    random.shuffle(owners)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, owners)]
    expected = _build_expected(n, row_slots, col_slots, templates, pcols, placements)

    h = k * row_slots + (k - 1)
    w = 2 * n + col_slots + 2
    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, row_slots, col_slots, k)

    row_assign = list(range(k))
    random.shuffle(row_assign)

    for j, mask in enumerate(templates):
        r0 = row_assign[j] * (row_slots + 1)
        paste(grid, np.where(mask, tcols[j], _BG), r0, 0)

    for j, rr, cc in placements:
        r0 = row_assign[j] * (row_slots + 1)
        grid[r0 + rr, n + 1 + cc] = tcols[j]

    swatch_c0 = n + col_slots + 2
    for j, col in enumerate(pcols):
        r0 = row_assign[j] * (row_slots + 1)
        swatch_r = r0 + row_slots - 1
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB and n >= 2 else 1
        c0 = swatch_c0 + max(0, (n - size) // 2)
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
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
