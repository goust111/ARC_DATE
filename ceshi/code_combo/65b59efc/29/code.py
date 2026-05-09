# Combo variant of 65b59efc (index 29)
# combo_spec: D4/2+D13/2
# Name: 全矩形维度

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_sprite


# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original: one scalar N controlled template height, template width, marker-band height,
#    marker-band width, and output tile counts, so a marker at local (lr, lc) stamped an N x N
#    template into an N x N output lattice of blocks.
#    New: template height/width are (TH, TW), but the marker lattice is (SR, SC). Example:
#    TH=3, TW=2, SR=4, SC=5. The top band shows 3x2 templates inside 5-wide tile columns;
#    the middle band is 4 rows tall. A marker at local (2, 4) stamps a recolored 3x2 template
#    into output block row 2, col 4, so the output becomes (4*3) x (5*2) rather than square.
# b) Complexity calibration:
#    The variant must be visibly different from the seed, so generation forbids TH=TW, SR=SC,
#    SR=TH, and SC=TW. This keeps the rectangular freedoms active without adding too many modes.
# c) Hidden assumptions broken from the seed:
#    - `N = sep_cols[0]` is no longer valid: the separator spacing gives SC, not TW.
#    - `out.shape == (N*N, N*N)` is no longer valid: it is `(SR*TH, SC*TW)`.
#    - Iterating the middle band over `range(N)` in both axes is invalid: rows use SR, cols use SC.
#    - The top-band subgrid width is no longer the template width. We infer TW from the occupied
#      extent of the template inside each SC-wide tile column.
# d) Placement legality:
#    Templates are left-aligned inside each SC-wide tile column, markers stay inside the SR x SC
#    middle band, palette swatches stay inside their tile columns, and all pastes use exact bounds.
# e) Visual observability:
#    The repeated blank gutter to the right of every top-band template makes SC > TW visible.
#    The distance between the two separator rows makes SR distinct from TH, and the output aspect
#    ratio changes accordingly.
# f) Visual cleanliness:
#    Only one template color appears in each middle-band tile column, different templates use
#    disjoint local slots, and separator rows/cols remain sparse dashed guides, so bands stay legible.
# g) Learnability:
#    Colors still serve only as template identities and palette recolors. The new behavior is driven
#    by repeated spatial structure, not by opaque color ordering or hidden numeric properties.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a rectangular stamp map: read template masks, marker slots, and palette colors, then render a block mosaic.

    Input structure:
    - Grid split into K tile columns of width SC, separated by 1-col dashed separator columns.
    - Top band (rows 0..TH-1): K templates of size TH x TW, left-aligned inside the SC-wide tile columns.
    - Middle band (rows TH+1..TH+SR): marker pixels whose color selects a template; their local coordinates are in an SR x SC slot lattice.
    - Bottom band (rows TH+SR+2..): one palette swatch per tile column.
    - Two dashed separator rows sit between the three bands.

    Rule:
    - A marker at local position (lr, lc) within its tile column stamps one recolored template into output block (lr, lc).
    - Marker color chooses which template to stamp, while the palette band provides that template's output color.
    - Because the template is TH x TW and the slot lattice is SR x SC, the output size is (SR*TH) x (SC*TW).

    Key insight: The separator spacing reveals the slot-grid dimensions, but the occupied extent of the top-band shapes reveals the narrower template width; marker locality and marker color must be decoded separately.
    """
    g = np.array(grid, dtype=int)
    (
        bg,
        _sep,
        template_h,
        template_w,
        slot_rows,
        slot_cols,
        starts,
        tcols,
        tmasks,
        pcols,
        mid_r0,
    ) = _parse_grid(g)

    tcol_to_j = {col: j for j, col in enumerate(tcols)}
    out = np.full((slot_rows * template_h, slot_cols * template_w), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        tile_color = None
        for lr in range(slot_rows):
            for lc in range(slot_cols):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg:
                    continue
                assert v in tcol_to_j, "marker color must match one of the template colors"
                assert tile_color is None or v == tile_color, "each tile column must use one marker color"
                tile_color = v
                j = tcol_to_j[v]
                owner = slot_owner.get((lr, lc))
                assert owner is None or owner == j, "different templates must occupy disjoint slots"
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], pcols[j], bg)
                paste(out, tile, lr * template_h, lc * template_w)
        assert tile_color is not None, "each tile column must contain at least one marker"

    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_TEMPLATE_H_RANGE = (3, 5)
_TEMPLATE_W_RANGE = (2, 4)
_SLOT_ROW_RANGE = (2, 5)
_SLOT_COL_EXTRA = [1, 2]
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.58
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 400

_VALID_DIMENSIONS = [
    (template_h, template_w, slot_rows, slot_cols)
    for template_h in range(_TEMPLATE_H_RANGE[0], _TEMPLATE_H_RANGE[1] + 1)
    for template_w in range(_TEMPLATE_W_RANGE[0], _TEMPLATE_W_RANGE[1] + 1)
    if template_h != template_w
    for slot_cols in (template_w + extra for extra in _SLOT_COL_EXTRA)
    for slot_rows in range(_SLOT_ROW_RANGE[0], _SLOT_ROW_RANGE[1] + 1)
    if slot_rows != template_h
    and slot_rows != slot_cols
    and slot_rows * template_h != slot_cols * template_w
]


def _dash_cell(pos: int) -> bool:
    return pos % 2 == 0


def _raw_col(tile_col: int, local_c: int, slot_cols: int) -> int:
    return tile_col * (slot_cols + 1) + local_c


def _spans_full_extent(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return (
        int(coords[:, 0].min()) == 0
        and int(coords[:, 0].max()) == mask.shape[0] - 1
        and int(coords[:, 1].min()) == 0
        and int(coords[:, 1].max()) == mask.shape[1] - 1
    )


def _sample_template(template_h: int, template_w: int, color: int) -> np.ndarray:
    for _ in range(25):
        try:
            sprite = random_sprite(
                template_h,
                template_w,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[color],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=_BG,
            )
            mask = sprite == color
            if _spans_full_extent(mask):
                return mask
        except RecursionError:
            continue

    mask = np.zeros((template_h, template_w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    mask[0, -1] = True
    return mask


def _build_expected(
    template_h: int,
    template_w: int,
    slot_rows: int,
    slot_cols: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((slot_rows * template_h, slot_cols * template_w), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        paste(expected, tile, rr * template_h, cc * template_w)
    return expected


def _draw_separators(
    grid: np.ndarray,
    template_h: int,
    slot_rows: int,
    slot_cols: int,
    k: int,
    core_h: int,
    core_w: int,
) -> None:
    for r in (template_h, template_h + slot_rows + 1):
        for c in range(core_w):
            if _dash_cell(c):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = slot_cols + i * (slot_cols + 1)
        for r in range(core_h):
            if _dash_cell(r):
                grid[r, c] = _SEP


def _choose_dimensions() -> tuple[int, int, int, int]:
    return random.choice(_VALID_DIMENSIONS)


def _parse_grid(grid: np.ndarray) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    list[int],
    list[int],
    list[np.ndarray],
    list[int],
    int,
]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    row_sep_counts = (g == sep).sum(axis=1)
    col_sep_counts = (g == sep).sum(axis=0)
    sep_rows = [r for r, cnt in enumerate(row_sep_counts) if cnt >= max(4, w // 3)]
    sep_cols = [c for c, cnt in enumerate(col_sep_counts) if cnt >= max(3, h // 3)]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 1

    template_h = sep_rows[0]
    slot_rows = sep_rows[1] - sep_rows[0] - 1
    slot_cols = sep_cols[0]
    assert template_h > 0 and slot_rows > 0 and slot_cols > 0
    assert all(sep_rows[i + 1] > sep_rows[i] for i in range(len(sep_rows) - 1))
    assert all(sep_cols[i + 1] - sep_cols[i] == slot_cols + 1 for i in range(len(sep_cols) - 1))

    starts = [0] + [c + 1 for c in sep_cols]
    mid_r0 = template_h + 1
    pal_r0 = sep_rows[1] + 1

    tcols: list[int] = []
    tmasks_full: list[np.ndarray] = []
    widths: list[int] = []
    for st in starts:
        sub = g[0:template_h, st:st + slot_cols]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        fg = sub == col
        coords = np.argwhere(fg)
        assert coords.size > 0
        widths.append(int(coords[:, 1].max()) + 1)
        tcols.append(col)
        tmasks_full.append(fg)

    assert len(set(tcols)) == len(tcols)
    template_w = widths[0]
    assert template_w > 0 and all(width == template_w for width in widths)
    tmasks = [mask[:, :template_w] for mask in tmasks_full]

    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st:st + slot_cols]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    return bg, sep, template_h, template_w, slot_rows, slot_cols, starts, tcols, tmasks, pcols, mid_r0


def _choose_used_slots(slot_rows: int, slot_cols: int, mark_count: int) -> list[tuple[int, int]]:
    all_slots = [(rr, cc) for rr in range(slot_rows) for cc in range(slot_cols)]
    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()

    rows = list(range(slot_rows))
    cols = list(range(slot_cols))
    random.shuffle(rows)
    random.shuffle(cols)

    if slot_rows <= slot_cols:
        for idx, cc in enumerate(cols):
            slot = (rows[idx % slot_rows], cc)
            chosen.append(slot)
            chosen_set.add(slot)
    else:
        for idx, rr in enumerate(rows):
            slot = (rr, cols[idx % slot_cols])
            chosen.append(slot)
            chosen_set.add(slot)

    remaining = [slot for slot in all_slots if slot not in chosen_set]
    random.shuffle(remaining)
    need = mark_count - len(chosen)
    assert need >= 0
    chosen.extend(remaining[:need])
    random.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    template_h, template_w, slot_rows, slot_cols = _choose_dimensions()
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = template_h + 1 + slot_rows + 1 + pal_h
    core_w = k * slot_cols + (k - 1)
    h = core_h + random.randint(0, _MAX_ROW_PAD)
    w = core_w + random.randint(0, _MAX_COL_PAD)

    templates = [_sample_template(template_h, template_w, col) for col in tcols]

    min_marks = max(k, slot_rows, slot_cols, int(np.ceil(_MIN_SLOT_COVERAGE * slot_rows * slot_cols)))
    mark_count = random.randint(min_marks, slot_rows * slot_cols)
    used_slots = _choose_used_slots(slot_rows, slot_cols, mark_count)
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(template_h, template_w, slot_rows, slot_cols, templates, pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, template_h, slot_rows, slot_cols, k, core_h, core_w)

    for j, mask in enumerate(templates):
        patch = np.where(mask, tcols[j], _BG)
        paste(grid, patch, 0, j * (slot_cols + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[template_h + 1 + rr, _raw_col(tile_assign[j], cc, slot_cols)] = tcols[j]

    swatch_r = template_h + slot_rows + 1 + pal_h
    for j, col in enumerate(pcols):
        start = j * (slot_cols + 1)
        size = 2 if slot_cols >= 2 and random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (slot_cols - size) // 2)
        grid[swatch_r - 1, c0:c0 + size] = col

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
