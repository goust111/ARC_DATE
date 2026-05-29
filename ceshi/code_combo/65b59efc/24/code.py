# Combo variant of 65b59efc (index 24)
# combo_spec: D1/4+D14/2
# Name: 变K频率消歧
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers"):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from arc_utils import paste, random_sprite


# Design analysis for the combo:
# a) Concrete behavior trace:
#    Original task: template color uniquely identifies the template, so reading one
#    marker pixel color is enough to choose a mask.
#    Combo task: exactly one repeated-color template pair is injected. Example with
#    K=4, template colors [2, 7, 7, 9]: the two orange-ish columns in the top band
#    are distinct masks, and their palette swatches still stay column-aligned. In the
#    middle band there are also two tile columns whose markers are color 7, but one
#    tile column has more 7 markers than the other. The denser one maps to the first
#    top-band 7-template (left one), the sparser one maps to the second. The local
#    marker coordinate still selects the output block exactly as in the original task.
# b) Complexity calibration:
#    The repeated-color ambiguity must be visible but not chaotic, so generation uses
#    exactly one duplicated template-color pair per sample, with a strict marker-count
#    gap and different masks for that pair. K varies across samples via _K_CHOICES,
#    but stays inside the original 3-4 template regime so the only D14 change is
#    "K differs between examples", not "K grows beyond the base task scale".
# c) Broken original assumptions:
#    - `len(set(tcols)) == K` is no longer valid.
#    - `tcol_to_j[v]` no longer works because one color may refer to two templates.
#    - Marker decoding now depends on per-tile-column counts, so each marker column
#      must contain at least one marker and repeated-color columns must have distinct
#      frequencies. The solver asserts these conditions; the generator guarantees them.
# d) Placement legality:
#    Marker slots remain globally disjoint, tile columns remain 1-per-template, and
#    all pasted templates stay inside an (N*N)x(N*N) output grid. Palette swatches
#    are still confined to their own template columns.
# e) Visual observability:
#    The repeated-color pair is obvious in the top band because two template columns
#    share the same color, and the corresponding marker columns show visibly different
#    densities. Humans can compare counts without relying on hidden numeric ordering.
# f) Visual cleanliness:
#    Only one ambiguity group is introduced, separators and band layout are unchanged,
#    and duplicate-color templates are forced to have distinct shapes so the grid does
#    not collapse into indistinguishable monochrome blocks.
# g) Inferability of the change:
#    The rule uses only visible structure: left-to-right order inside the repeated
#    template-color group plus the larger-vs-smaller marker frequency. No ranking by
#    raw color id or other non-visual metadata is used.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where K varies by example and repeated template colors may require frequency tie-breaks.

    Input structure:
    - The grid is split into K template-width tile columns separated by 1-cell dashed separators; K is inferred from the layout.
    - Top band (rows 0..N-1): K monochrome template masks, but one visible color may appear in two template columns.
    - Middle band (rows N+1..2N): marker pixels, with one template assigned to each tile column; repeated-color marker columns are disambiguated by how many markers they contain.
    - Bottom band (rows 2N+2..): K palette swatches aligned with template columns, one recolor per template.
    - Dashed separator rows sit at row N and row 2N+1, with matching dashed vertical separators between tile columns.

    Rule:
    - A marker's local coordinate (lr, lc) within its tile column still selects the output block at block position (lr, lc).
    - If a marker color appears for only one template, decode it directly as before.
    - If two template columns share the same color, rank the marker tile columns of that color by marker count: the more frequent column uses the first such template (left to right), the less frequent one uses the second.
    - Stamp the chosen template recolored by its aligned palette swatch into the corresponding N x N block of the output.

    Key insight: Color still narrows the choice to a template group, but repeated colors are resolved by visible marker frequency, while the marker's local position still encodes the output block location.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = int(sep_cols[0])
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    template_colors: list[int] = []
    template_masks: list[np.ndarray] = []
    templates_by_color: dict[int, list[int]] = defaultdict(list)
    for j, st in enumerate(starts):
        sub = g[0:n, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = int(next(iter(vals)))
        template_colors.append(col)
        template_masks.append(sub == col)
        templates_by_color[col].append(j)

    pal_r0 = int(sep_rows[-1] + 1)
    palette_colors: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        palette_colors.append(int(next(iter(vals))))

    marker_cols_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    mid_r0 = n + 1
    for tile_idx, st in enumerate(starts):
        marker_color = None
        slots: list[tuple[int, int]] = []
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert marker_color is None or v == marker_color, "marker tile column must be monochrome"
                marker_color = v
                slots.append((lr, lc))
        assert marker_color is not None, "every template needs at least one marker"
        marker_cols_by_color[int(marker_color)].append(
            {"tile_idx": tile_idx, "count": len(slots), "slots": slots}
        )

    tile_to_template: dict[int, int] = {}
    for color, template_ids in templates_by_color.items():
        marker_infos = marker_cols_by_color.get(color)
        assert marker_infos is not None and len(marker_infos) == len(template_ids)
        if len(template_ids) > 1:
            counts_here = [int(info["count"]) for info in marker_infos]
            assert len(set(counts_here)) == len(counts_here), "repeated-color columns need distinct marker frequencies"
            marker_infos = sorted(marker_infos, key=lambda info: (-int(info["count"]), int(info["tile_idx"])))
        for j, info in zip(template_ids, marker_infos):
            tile_to_template[int(info["tile_idx"])] = int(j)

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    rendered_tiles = [np.where(template_masks[j], palette_colors[j], bg) for j in range(k)]

    for tile_idx, st in enumerate(starts):
        j = tile_to_template[tile_idx]
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                paste(out, rendered_tiles[j], lr * n, lc * n)

    return out


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_CHOICES = [3, 4]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.65
_MIN_DUP_COUNT_GAP = 2

_SWATCH_DOUBLE_PROB = 0.35

_MAX_RETRIES = 400
_k_cycle_index = 0


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


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], palette_colors[j], _BG)
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


def _sample_template_colors(k: int) -> tuple[list[int], tuple[int, int]]:
    dup_left = random.randint(0, k - 2)
    dup_right = dup_left + 1
    unique_pool = random.sample(_AVAILABLE_COLORS, k - 1)
    dup_color = unique_pool[0]
    other_colors = unique_pool[1:]
    colors: list[int] = []
    other_idx = 0
    for j in range(k):
        if j in (dup_left, dup_right):
            colors.append(dup_color)
        else:
            colors.append(other_colors[other_idx])
            other_idx += 1
    return colors, (dup_left, dup_right)


def _next_k() -> int:
    global _k_cycle_index
    k = _K_CHOICES[_k_cycle_index % len(_K_CHOICES)]
    _k_cycle_index += 1
    return k


def _sample_templates(n: int, template_colors: list[int], dup_pair: tuple[int, int]) -> list[np.ndarray]:
    templates = [_sample_template(n, col) for col in template_colors]
    a, b = dup_pair
    for _ in range(80):
        if not np.array_equal(templates[a], templates[b]):
            return templates
        templates[b] = _sample_template(n, template_colors[b])
    raise RuntimeError("failed to separate duplicate-color template shapes")


def _sample_counts(k: int, n: int, dup_pair: tuple[int, int]) -> list[int]:
    total_slots = n * n
    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * total_slots)))
    a, b = dup_pair
    for _ in range(300):
        total = random.randint(min_marks, total_slots)
        cuts = sorted(random.sample(range(1, total), k - 1))
        counts = []
        prev = 0
        for cut in cuts + [total]:
            counts.append(cut - prev)
            prev = cut
        if counts[a] < counts[b]:
            counts[a], counts[b] = counts[b], counts[a]
        if counts[a] - counts[b] >= _MIN_DUP_COUNT_GAP:
            return counts
    raise RuntimeError("failed to sample distinct duplicate-color marker counts")


def _sample_placements(n: int, counts: list[int]) -> list[tuple[int, int, int]]:
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    used_slots = all_slots[:sum(counts)]
    placements: list[tuple[int, int, int]] = []
    offset = 0
    for j, cnt in enumerate(counts):
        for rr, cc in used_slots[offset:offset + cnt]:
            placements.append((j, rr, cc))
        offset += cnt
    random.shuffle(placements)
    return placements


def _gen_attempt(k: int) -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    template_colors, dup_pair = _sample_template_colors(k)
    palette_colors = random.sample(_AVAILABLE_COLORS, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    try:
        templates = _sample_templates(n, template_colors, dup_pair)
        counts = _sample_counts(k, n, dup_pair)
    except RuntimeError:
        return None

    placements = _sample_placements(n, counts)
    expected = _build_expected(n, templates, palette_colors, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, template_colors[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = template_colors[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(palette_colors):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0:c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    k = _next_k()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(k)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
