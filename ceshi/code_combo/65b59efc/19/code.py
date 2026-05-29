# Combo variant of 65b59efc (index 19)
# combo_spec: D11/1+D19/1
# Name: 双重反转布局
from __future__ import annotations

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Original task bands are `templates -> markers -> palette`, and palette column j recolors template j.
#      Example with K=3 and template order [T0, T1, T2]: the bottom swatches left->right [A, B, C] mean
#      output colors [A, B, C] for [T0, T1, T2].
#    - This combo flips the vertical legend and reverses the palette correspondence at the same time:
#      the input becomes `palette -> markers -> templates`, and the top swatches left->right [C, B, A]
#      now recolor bottom templates [T0, T1, T2].
#    - A marker of template color T0 at local slot (1, 2) still stamps into output block (1, 2), but its
#      color now comes from the top-right swatch instead of the top-left one. The vertical band inversion
#      changes where each object family is read; the horizontal palette inversion changes which swatch
#      belongs to which template after the bands have been located.
# b) Complexity calibration
#    - The layout change is visually obvious because the top band becomes sparse palette dots while the
#      bottom band becomes dense template masks. This avoids a barely-noticeable meta-only mutation.
#    - The reverse palette mapping is made observable by forcing template masks to be pairwise distinct,
#      palette colors to be distinct, and every template to be used at least once. Then reading swatches
#      left->right versus right->left changes visible output blocks.
# c) Implicit-assumption audit against the original solver
#    - Original `g[0:N, ...]` as templates is invalid: row 0 now belongs to palette, not template.
#    - Original `pal_r0 = sep_rows[-1] + 1` is invalid: palette is above the first separator, not below
#      the last separator.
#    - Original `pcols[j]` as the recolor for template j is invalid: physical palette order is reversed,
#      so template j must use `physical_pcols[k - 1 - j]`.
#    - Original `mid_r0 = N + 1` is invalid when `pal_h != N`; the marker band must be anchored from the
#      separator rows instead of hard-coded.
# d) Placement legality
#    - The generator keeps the same tile columns and dashed separator columns, so all objects stay inside
#      their own N-wide cells and never overlap separators.
#    - Top swatches live only inside the palette band, markers only inside the middle band, and templates
#      only inside the bottom N rows after the second separator.
#    - Slot ownership is unchanged: different templates never claim the same local (lr, lc).
# e) Visual observability
#    - Humans can see the band-role inversion directly because dense shapes moved to the bottom and tiny
#      swatches moved to the top.
#    - They can infer the right-to-left palette matching because output blocks of the leftmost and rightmost
#      templates use top-band colors from the opposite side, and those templates repeat across many slots.
# f) Visual cleanliness
#    - The scene still has exactly three horizontal bands plus separators, so no new clutter is introduced.
#    - Palette marks remain tiny and centered, keeping the top band legible; templates remain monochrome
#      connected masks, keeping the bottom band easy to compare against the output.
# g) New solver assumptions that the generator must guarantee
#    - The first separator row sits immediately below the palette band, and the second immediately below the
#      marker band; generator fixes those positions from `pal_h`.
#    - Every template cell contains exactly one non-bg/non-sep color, every palette cell contains exactly
#      one swatch color, and palette colors are unique; generator enforces all three.
#    - The reversed palette mapping must matter, so the generator rejects any sample where reversed-palette
#      rendering equals same-order rendering on the very same input.

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
    Decode a vertically flipped stamp legend whose palette must also be matched in reverse column order.

    Input structure:
    - The grid is split into K tile columns of width N, with dashed separator columns between tile columns.
    - The top band (above the first separator row) contains K palette swatches, one sparse color sample per tile column.
    - The middle band (between the two separator rows) contains marker pixels; marker color still identifies a template, and local marker coordinates still identify an output block.
    - The bottom band (below the second separator row) contains the K template masks, one monochrome template per tile column.

    Rule:
    - Read the bottom templates from left to right to recover template indices.
    - Read the top palette swatches from left to right, but assign them to templates from right to left, so template j uses the swatch found in physical palette column K-1-j.
    - For every marker, use its color to pick the template, use its local (lr, lc) inside the marker band to pick the output block, and stamp that template recolored with its reversed-match palette color.

    Key insight: Solving now requires two separate inversions: first identify that the legend bands have been flipped vertically, then undo the palette-to-template correspondence by reading the palette in the opposite horizontal direction from the templates.
    """
    decoded = _decode_grid(grid)
    n = decoded["n"]
    bg = decoded["bg"]
    logical_pcols = decoded["physical_pcols"][::-1]

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in decoded["starts"]:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(decoded["grid"][decoded["mid_r0"] + lr, st + lc])
                if v == bg or v == decoded["sep"]:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = decoded["tcol_to_j"][v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                tile = np.where(decoded["tmasks"][j], logical_pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [1, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MAX_TEMPLATE_RESAMPLES = 100

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
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


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    logical_pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], logical_pcols[j], _BG)
        paste(expected, tile, rr * n, cc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, pal_h: int, core_h: int, core_w: int) -> None:
    for r in (pal_h, pal_h + n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _decode_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and len(sep_rows) >= 2

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    pal_h = sep_rows[0]
    mid_r0 = sep_rows[0] + 1
    template_r0 = sep_rows[-1] + 1

    assert sep_rows[-1] - sep_rows[0] - 1 == n
    assert template_r0 + n <= h
    assert pal_h >= 2

    physical_pcols: list[int] = []
    for st in starts:
        sub = g[0:pal_h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        physical_pcols.append(vals.pop())
    assert len(set(physical_pcols)) == k

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[template_r0 : template_r0 + n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k

    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "n": n,
        "k": k,
        "starts": starts,
        "pal_h": pal_h,
        "mid_r0": mid_r0,
        "template_r0": template_r0,
        "physical_pcols": physical_pcols,
        "tcols": tcols,
        "tmasks": tmasks,
        "tcol_to_j": {col: j for j, col in enumerate(tcols)},
    }


def _render_from_decoded(decoded: dict, *, reversed_palette: bool) -> np.ndarray:
    n = decoded["n"]
    bg = decoded["bg"]
    pcols = decoded["physical_pcols"][::-1] if reversed_palette else decoded["physical_pcols"]
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in decoded["starts"]:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(decoded["grid"][decoded["mid_r0"] + lr, st + lc])
                if v == bg or v == decoded["sep"]:
                    continue
                assert tile_color is None or v == tile_color
                tile_color = v
                j = decoded["tcol_to_j"][v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                tile = np.where(decoded["tmasks"][j], pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    decoded = _decode_grid(grid)
    n = decoded["n"]
    combo_out = _render_from_decoded(decoded, reversed_palette=True)
    naive_out = _render_from_decoded(decoded, reversed_palette=False)

    changed_blocks = 0
    for rr in range(n):
        for cc in range(n):
            r0 = rr * n
            c0 = cc * n
            if not np.array_equal(combo_out[r0 : r0 + n, c0 : c0 + n], naive_out[r0 : r0 + n, c0 : c0 + n]):
                changed_blocks += 1

    top_band_nonbg = int(np.sum((decoded["grid"][0 : decoded["pal_h"], :] != decoded["bg"]) & (decoded["grid"][0 : decoded["pal_h"], :] != decoded["sep"])))
    bottom_band_nonbg = int(
        np.sum(
            (decoded["grid"][decoded["template_r0"] : decoded["template_r0"] + n, :] != decoded["bg"])
            & (decoded["grid"][decoded["template_r0"] : decoded["template_r0"] + n, :] != decoded["sep"])
        )
    )
    return {
        "changed_blocks_vs_same_order": changed_blocks,
        "top_band_nonbg": top_band_nonbg,
        "bottom_band_nonbg": bottom_band_nonbg,
        "layout_density_gap": bottom_band_nonbg - top_band_nonbg,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    # Keep template/marker colors disjoint from palette colors so the recoloring rule
    # and the reversed palette correspondence are visually easier to infer.
    color_pool = random.sample(_AVAILABLE_COLORS, 2 * k)
    tcols = color_pool[:k]
    physical_pcols = color_pool[k:]
    logical_pcols = physical_pcols[::-1]

    templates = _sample_distinct_templates(n, tcols)
    if templates is None:
        return None

    core_h = pal_h + 2 * n + 2
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(max(k, min_marks), n * n)
    used_slots = all_slots[:mark_count]

    assign_pool = list(range(k))
    while len(assign_pool) < mark_count:
        assign_pool.append(random.randrange(k))
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, logical_pcols, placements)
    naive_same_order = _build_expected(n, templates, physical_pcols, placements)
    if np.array_equal(expected, naive_same_order):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, pal_h, core_h, core_w)

    template_r0 = pal_h + n + 2
    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), template_r0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    mid_r0 = pal_h + 1
    for j, rr, cc in placements:
        grid[mid_r0 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    swatch_r = pal_h - 1
    for j, col in enumerate(physical_pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if not np.array_equal(main(grid), expected):
            return None
        metrics = _behavior_metrics(grid)
        if metrics["changed_blocks_vs_same_order"] <= 0:
            return None
        if metrics["layout_density_gap"] <= 0:
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
