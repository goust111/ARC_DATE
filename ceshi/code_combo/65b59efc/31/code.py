# Combo variant of 65b59efc (index 31)
# combo_spec: D17/0+D19/3
# Name: 动态色随机调色板

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_sprite


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    Original: background was always black, separators were always gray, and the j-th palette tile
#    recolored the j-th template directly by column order.
#    Fused: each example chooses a new background color and a new separator color, so the solver
#    must recover them by frequency before any band parsing. The bottom band is no longer aligned:
#    each tile contains a 2-pixel palette swatch plus one adjacent 1-pixel label whose color is the
#    template color it belongs to. Example with N=3, K=3:
#    - top columns show template colors [red, yellow, teal]
#    - bottom columns might show [(label teal -> swatch orange), (label red -> swatch blue),
#      (label yellow -> swatch maroon)] in that shuffled order
#    - a middle-band red marker at local slot (2,1) still stamps the red template, but it must now
#      be recolored blue because the nearby label identifies which shuffled swatch belongs to red.
# b) Complexity calibration:
#    The palette encoding uses only two visible multiplicities per tile: one label pixel and one
#    2-pixel swatch. This is enough structure for humans to decode, while keeping the legend simple.
#    Template colors and palette colors are sampled as disjoint sets, so the label-vs-swatch split
#    is visually obvious and never collapses into a single color run.
# c) Hidden-assumption audit against the seed solver:
#    - `_BG = 0` and `_SEP = 5` are invalid; `main()` must infer both colors first.
#    - `pcols[j]` can no longer be read from the j-th bottom tile; the bottom band must be decoded
#      as label->swatch pairs, then remapped back onto template order.
#    - Marker decoding still relies on template colors from the top band, so those colors must stay
#      unique and distinct from bg/sep.
#    - Because the old aligned-palette rule would silently work on identity orderings, the generator
#      enforces a derangement of the palette tiles and guarantees every template is used at least once.
# d) Placement legality:
#    All separators stay on the original dashed lattice, markers occupy distinct local slots, each
#    middle tile column carries exactly one template color, and each bottom tile uses exactly three
#    non-bg/non-sep cells arranged as [label][swatch][swatch] within the tile width.
# e) Visual observability:
#    The non-black background and non-gray separators make D17 visible immediately. D19 is also
#    visible because palette hints are shuffled yet locally annotated, so a solver must read the
#    little label instead of relying on column order.
# f) Visual cleanliness:
#    The legend remains three clean horizontal bands with sparse separators. In the palette band,
#    each tile has only one tiny 3-cell cue, avoiding clutter while preserving repeated evidence.
# g) Learnability:
#    The new color behavior is supported by explicit local tags, not by hidden numeric ordering.
#    Across 2-4 examples, humans can match label colors to templates and see that column order no
#    longer predicts recoloring.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a shuffled stamp legend: recover the dynamic background/separator colors, read the templates,
    interpret labeled palette swatches, and render the stamped output mosaic.

    Input structure:
    - The grid is split into K tile columns of width N, with dashed separator columns between them.
    - Rows 0..N-1 contain K template masks, one per tile column; each template uses its own template color.
    - Rows N+1..2N contain marker pixels whose local slot picks an output block and whose color picks a template.
    - Rows below the second dashed separator contain K palette hints in shuffled order; each hint is a
      2-pixel swatch with a neighboring 1-pixel label whose color names the matching template.
    - The background color and separator color are not fixed and must be inferred from the grid itself.

    Rule:
    - First identify the current example's background and separator colors, then parse the tiled layout.
    - Marker color still selects the template, and marker local coordinates still choose the N x N output block.
    - A template's output color is no longer tied to the same column: it comes from the shuffled palette hint
      whose small label pixel matches that template's color.

    Key insight: The top and middle bands still speak in template colors, but the bottom band no longer uses
    column position as the mapping signal; the tiny adjacent label is the only reliable bridge from template
    identity to palette color after the dynamic bg/sep colors are stripped away.
    """
    bg, sep, n, starts, tcols, tmasks, pcols, slot_owner = _decode_scene(grid)
    out = np.full((n * n, n * n), bg, dtype=int)
    for (rr, cc), j in slot_owner.items():
        tile = np.where(tmasks[j], pcols[j], bg)
        paste(out, tile, rr * n, cc * n)
    return out


# --------------- generator ---------------

_ALL_COLORS = list(range(10))
_NON_STANDARD_BGSEP = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_TEMPLATE_RETRIES = 30

_MIN_SLOT_COVERAGE = 0.60
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _sample_bg_sep() -> tuple[int, int]:
    bg, sep = random.sample(_NON_STANDARD_BGSEP, 2)
    return bg, sep


def _sample_template(n: int, color: int, bg: int) -> np.ndarray:
    for _ in range(_TEMPLATE_RETRIES):
        try:
            sprite = random_sprite(
                n,
                n,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[color],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=bg,
            )
        except RecursionError:
            continue
        mask = sprite == color
        if mask.any():
            return mask

    mask = np.zeros((n, n), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    mask[0, -1] = True
    return mask


def _build_expected(
    bg: int,
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], bg)
        paste(expected, tile, rr * n, cc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int, sep: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = sep
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = sep


def _sample_palette_derangement(k: int) -> list[int]:
    base = list(range(k))
    while True:
        perm = base[:]
        random.shuffle(perm)
        if all(perm[i] != i for i in range(k)):
            return perm


def _decode_scene(
    grid: np.ndarray,
) -> tuple[int, int, int, list[int], list[int], list[np.ndarray], list[int], dict[tuple[int, int], int]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
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
    label_to_pcol: dict[int, int] = {}
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        pix_counts = Counter(int(v) for v in sub.flat if int(v) not in {bg, sep})
        assert len(pix_counts) == 2, "each palette tile must contain one label color and one swatch color"
        by_mult = sorted((cnt, col) for col, cnt in pix_counts.items())
        assert [cnt for cnt, _col in by_mult] == [1, 2], "palette tile must use 1 label pixel + 2 swatch pixels"
        label_col = by_mult[0][1]
        swatch_col = by_mult[1][1]
        assert label_col in tcol_to_j, "palette label must match a template color"
        assert label_col not in label_to_pcol, "palette labels must be unique"
        label_to_pcol[label_col] = swatch_col
    assert len(label_to_pcol) == k
    pcols = [label_to_pcol[tcol] for tcol in tcols]

    mid_r0 = n + 1
    slot_owner: dict[tuple[int, int], int] = {}
    used_tile_colors: set[int] = set()
    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "each middle tile column must keep one marker color"
                tile_color = v
                assert v in tcol_to_j, "marker color must identify a template"
                j = tcol_to_j[v]
                owner = slot_owner.get((lr, lc))
                assert owner is None or owner == j, "different templates must not share a local slot"
                slot_owner[(lr, lc)] = j
        assert tile_color is not None, "each template column must appear in the marker band"
        assert tile_color not in used_tile_colors, "marker tile colors must form a permutation of templates"
        used_tile_colors.add(tile_color)
    assert len(used_tile_colors) == k
    return bg, sep, n, starts, tcols, tmasks, pcols, slot_owner


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    bg, sep = _sample_bg_sep()

    remaining = [c for c in _ALL_COLORS if c not in {bg, sep}]
    tcols = random.sample(remaining, k)
    palette_pool = [c for c in remaining if c not in set(tcols)]
    pcols = random.sample(palette_pool, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = [_sample_template(n, col, bg) for col in tcols]

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = list(range(k)) + [random.randrange(k) for _ in range(mark_count - k)]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(bg, n, templates, pcols, placements)

    grid = np.full((h, w), bg, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w, sep)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], bg), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    pal_owner = _sample_palette_derangement(k)
    swatch_r = 2 * n + 2 + pal_h - 1
    c0 = max(0, (n - 3) // 2)
    for physical_col, template_j in enumerate(pal_owner):
        start = physical_col * (n + 1) + c0
        grid[swatch_r, start] = tcols[template_j]
        grid[swatch_r, start + 1 : start + 3] = pcols[template_j]

    counts = Counter(map(int, grid.flatten()))
    ordered = counts.most_common()
    if len(ordered) < 2 or int(ordered[0][0]) != bg or int(ordered[1][0]) != sep:
        return None

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
