# Combo variant of 65b59efc (index 14)
# combo_spec: D3/2+D5/0+D19/0
# Name: 三重间接色映射
from __future__ import annotations

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task:
#      1. Top band column j contains one single-color binary template Tj.
#      2. A middle-band marker pixel uses that same template color, so marker color directly reveals j.
#      3. Bottom band column j contains one palette color Pj.
#      4. Output block at local marker slot (lr, lc) is `mask(Tj) recolored to Pj`.
#    - Fused task here:
#      1. Top band still contains single-color binary templates, each with its own template color tj.
#      2. Bottom band no longer aligns palette-to-template by physical column. Instead each palette entry is a
#         two-pixel pair inside some tile column: one pixel is a template color tj (the key), the other is a
#         palette color pj (the marker code). Matching by shared template color recovers `pj -> Tj`.
#      3. Middle-band markers use palette colors pj, not template colors tj.
#      4. Output ignores pj for recoloring and stamps Tj in its original template color tj.
#      Example with K=3:
#        top templates left->right use colors [2, 7, 8]
#        bottom palette pairs left->right are [(8,4), (2,6), (7,1)]
#        so palette->template mapping is {4->template color 8, 6->template color 2, 1->template color 7}
#        A marker colored 6 therefore chooses the template whose top-band color is 2, and the output block keeps
#        color 2 rather than repainting with 6.
# b) Complexity calibration
#    - Palette colors are sampled distinct and disjoint from template colors, so marker colors visibly belong to
#      the bottom legend rather than the top templates.
#    - Template masks are forced pairwise distinct and every template appears at least once, so the color-match
#      mapping affects observable output shapes instead of collapsing to accidental equivalence.
#    - The palette band uses one compact pair per logical template. This adds one indirection layer without
#      introducing too many free variables for humans to infer from 2-4 examples.
# c) Implicit-assumption audit against the original main()
#    - Original `j = tcol_to_j[v]` is invalid: marker color v is now a palette color, not a template color.
#    - Original `pcols[j]`-by-column lookup is invalid: palette entries are assigned to arbitrary physical tile
#      columns and must be matched back by the adjacent template-color key pixel.
#    - Original `tile = np.where(tmasks[j], pcols[j], bg)` is invalid: D5/0 requires `tile = template itself`,
#      so palette colors should not appear in the output.
#    - Original "bottom subgrid has exactly one non-bg non-sep color" assumption is invalid. Each palette cell now
#      intentionally contains exactly two colors: one template-color key and one palette-color code.
# d) Placement legality
#    - Each palette pair is placed inside one N-wide tile column on a dedicated palette row; it never crosses a
#      separator column or overlaps another pair.
#    - Marker placements still use unique local slots `(lr, lc)` across templates, so output blocks never have
#      conflicting owners.
#    - Optional trailing padding remains background-only, preserving the separator-based parsing logic.
# e) Visual observability
#    - Humans can see that middle-band marker colors do not occur in the top templates but do occur in the bottom
#      palette pairs, forcing the first indirection.
#    - The repeated appearance of one top template color inside each palette pair exposes the second indirection:
#      that color is the key linking the pair back to the template.
#    - The output then preserves the original template colors instead of adopting marker/palette colors, making the
#      D5/0 contribution directly visible.
# f) Visual cleanliness
#    - The layout stays in the original three clean bands with the same separator geometry.
#    - Each palette entry uses only two adjacent pixels, so the added metadata is legible and localized.
#    - One marker color per middle tile column is preserved, avoiding clutter in the marker band.
# g) New assumptions introduced by main(), and generator support
#    - main() assumes each template has a unique template color; generator keeps template colors distinct.
#    - main() assumes each palette pair contains exactly one template color and one palette color; generator draws
#      exactly one such pair for each logical template.
#    - main() assumes palette colors are unique so `marker color -> palette entry` is unambiguous; generator
#      enforces distinct palette colors.
#    - main() assumes every logical template is represented by at least one marker somewhere; generator ensures
#      coverage so the triple-indirection behavior actually matters in the sample.

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.arc_utils import paste, random_sprite


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
_MAX_RETRIES = 500
_MAX_TEMPLATE_RESAMPLES = 100


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled legend where markers point to palette colors, palette entries point back to templates, and stamps keep their original template colors.

    Input structure:
    - The grid is divided into K tile columns of width N, separated by 1-column dashed separators, with two dashed separator rows splitting the three bands.
    - The top band (rows 0..N-1) contains K single-color binary templates; each template is drawn in its own template color.
    - The middle band (rows N+1..2N) contains sparse marker pixels. Each physical tile column uses one marker color, but that color is a palette code rather than a template color.
    - The bottom band (rows 2N+2..) contains K palette entries. Each entry is a two-pixel pair placed inside some tile column: one pixel is a template color key and the other is the palette color that markers will use.

    Rule:
    - A marker's local coordinates (lr, lc) inside its tile column choose the destination N x N block in the output.
    - To decide which template to stamp, first match the marker color to a palette entry, then use the template-color key inside that same palette pair to identify the corresponding top-band template.
    - Stamp that template into the chosen output block without recoloring it: active cells keep the template's original color and inactive cells stay background.

    Key insight: the output template is recovered by a three-step indirection chain `marker color -> palette pair -> template color key -> top template`, and the palette color itself never appears in the output.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols if c + 1 + n <= w]
    k = len(starts)
    assert k >= 2

    # Read top templates and remember their identifying template colors.
    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        tcol = vals.pop()
        mask = sub == tcol
        assert mask.any()
        tcols.append(tcol)
        tmasks.append(mask)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    # Decode palette entries by color matching: each pair contains one template key color and one palette code.
    pal_r0 = sep_rows[-1] + 1
    palette_rows = [r for r in range(pal_r0, h) if any(int(v) not in {bg, sep} for v in g[r, :])]
    assert len(palette_rows) == 1
    pair_row = palette_rows[0]

    marker_to_j: dict[int, int] = {}
    seen_templates: set[int] = set()
    for st in starts:
        vals = [int(v) for v in g[pair_row, st : st + n] if int(v) not in {bg, sep}]
        uniq = list(dict.fromkeys(vals))
        assert len(uniq) == 2
        template_keys = [v for v in uniq if v in tcol_to_j]
        palette_codes = [v for v in uniq if v not in tcol_to_j]
        assert len(template_keys) == 1 and len(palette_codes) == 1
        key = template_keys[0]
        code = palette_codes[0]
        j = tcol_to_j[key]
        assert code not in marker_to_j
        marker_to_j[code] = j
        seen_templates.add(j)
    assert len(marker_to_j) == k and len(seen_templates) == k

    # Read markers; marker color names a palette entry, local position names the output block.
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
                assert v in marker_to_j, "marker color must match one palette code"
                tile_marker = v
                j = marker_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates cannot share an output slot"
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], tcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


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
        ok = False
        for _ in range(_MAX_TEMPLATE_RESAMPLES):
            mask = _sample_template(n, col)
            key = tuple(mask.reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            templates.append(mask)
            ok = True
            break
        if not ok:
            return None
    return templates


def _render_original_like(
    n: int,
    templates: list[np.ndarray],
    palette_codes: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        paste(out, np.where(templates[j], palette_codes[j], _BG), rr * n, cc * n)
    return out


def _render_combo_expected(
    n: int,
    templates: list[np.ndarray],
    template_colors: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        paste(out, np.where(templates[j], template_colors[j], _BG), rr * n, cc * n)
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


def _pair_cols(n: int) -> tuple[int, int]:
    left = max(0, n // 2 - 1)
    right = min(n - 1, left + 1)
    if left == right:
        left = max(0, right - 1)
    return left, right


def _draw_pair(grid: np.ndarray, row: int, st: int, n: int, key: int, code: int, key_left: bool) -> None:
    c0, c1 = _pair_cols(n)
    if key_left:
        grid[row, st + c0] = key
        grid[row, st + c1] = code
    else:
        grid[row, st + c0] = code
        grid[row, st + c1] = key


def _build_marker_to_j(template_colors: list[int], palette_codes: list[int], pair_slots: list[int]) -> dict[int, int]:
    tcol_to_j = {col: j for j, col in enumerate(template_colors)}
    marker_to_j: dict[int, int] = {}
    for j, slot in enumerate(pair_slots):
        marker_to_j[palette_codes[j]] = tcol_to_j[template_colors[j]]
        assert marker_to_j[palette_codes[j]] == j
        assert 0 <= slot < len(template_colors)
    return marker_to_j


def _build_wrong_direct_mapping(palette_codes: list[int], pair_slots: list[int]) -> dict[int, int]:
    wrong: dict[int, int] = {}
    for physical_col, code in zip(pair_slots, palette_codes):
        wrong[code] = physical_col
    return wrong


def _sample_layout() -> tuple[int, int, list[int], list[int]] | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    template_colors = random.sample(_AVAILABLE_COLORS, k)
    palette_pool = [c for c in _AVAILABLE_COLORS if c not in template_colors]
    if len(palette_pool) < k:
        return None
    palette_codes = random.sample(palette_pool, k)
    return n, k, template_colors, palette_codes


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    n, k, template_colors, palette_codes = layout

    templates = _sample_distinct_templates(n, template_colors)
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
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k + 1))[:mark_count]
    random.shuffle(assign_pool)
    for j in range(k):
        if j not in assign_pool:
            assign_pool[random.randrange(mark_count)] = j
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _render_combo_expected(n, templates, template_colors, placements)
    old_like = _render_original_like(n, templates, palette_codes, placements)
    if np.array_equal(expected, old_like):
        return None

    pair_slots = list(range(k))
    random.shuffle(pair_slots)
    if pair_slots == list(range(k)):
        return None
    wrong_direct = _build_wrong_direct_mapping(palette_codes, pair_slots)
    if all(wrong_direct[palette_codes[j]] == j for j in range(k)):
        return None

    marker_to_j = _build_marker_to_j(template_colors, palette_codes, pair_slots)
    if any(marker_to_j[palette_codes[j]] != j for j in range(k)):
        return None

    # Require visible D19 effect: at least one marker must decode differently under naive physical-column alignment.
    if not any(wrong_direct[palette_codes[j]] != j for j, _, _ in placements):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, template_colors[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = palette_codes[j]

    pair_row = 2 * n + 2 + pal_h - 1
    key_left_flags = [random.choice([False, True]) for _ in range(k)]
    if all(key_left_flags) or not any(key_left_flags):
        idx = random.randrange(k)
        key_left_flags[idx] = not key_left_flags[idx]
    for j in range(k):
        slot = pair_slots[j]
        st = slot * (n + 1)
        _draw_pair(grid, pair_row, st, n, template_colors[j], palette_codes[j], key_left_flags[j])

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    # Degeneracy checks:
    # 1) combo must differ from the original recolor rule on this exact input
    # 2) at least one used marker must fail under the wrong "palette column == template column" shortcut
    wrong_out = np.full((n * n, n * n), _BG, dtype=int)
    used_wrong = False
    for j, rr, cc in placements:
        guess_j = wrong_direct[palette_codes[j]]
        if guess_j != j:
            used_wrong = True
        paste(wrong_out, np.where(templates[guess_j], template_colors[guess_j], _BG), rr * n, cc * n)
    if not used_wrong or np.array_equal(expected, wrong_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
