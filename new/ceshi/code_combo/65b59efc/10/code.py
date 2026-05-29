# Combo variant of 65b59efc (index 10)
# combo_spec: D1/3+D9/0
# Name: 跨列计数排序匹配
#
# Design analysis:
# - Concrete example: with N=4 and K=3, suppose the three top templates have areas 6, 9, 12.
#   The middle band may contain marker colors red/green/yellow with global counts 3, 4, 5, but those
#   pixels are scattered across all three tile columns rather than one color per column. The solver must
#   sort template areas (6<9<12), sort marker-group counts (3<4<5), pair those orders, then stamp each
#   marker at the output block given by its local (lr, lc). In the base task, the same example would have
#   used direct color equality and each template would have occupied exactly one marker column.
# - Complexity calibration: the generator enforces distinct template areas and distinct marker counts, so
#   the new matching cue is visible and non-ambiguous. Counts are moderate consecutive integers instead of
#   arbitrary large values, making the global ranking human-countable from a few examples.
# - Broken original assumptions:
#   1. `marker color == template color` no longer holds. Marker colors are sampled from colors not used
#      by the top templates.
#   2. `each tile column is monochrome` no longer holds. A physical column may carry several marker colors.
#   3. `template identity can be inferred per column` no longer holds. The solver must pool counts across
#      the entire middle band before decoding any marker.
# - Legality guarantees in the generator:
#   1. Local output slots `(lr, lc)` are globally unique, so different templates never fight over one
#      output block.
#   2. Marker pixels occupy valid in-bounds cells in the middle band and palette/template bands stay
#      disjoint, so no overlaps are introduced.
#   3. Each marker color appears in at least two physical columns, and at least two physical columns are
#      mixed, ensuring D9/0 is actually observable rather than accidental noise.
#   4. Palette colors are a deranged permutation of the template colors, so recoloring remains visible
#      while the palette band never reuses marker-group colors.
# - New implicit assumptions of `main()` and how the generator satisfies them:
#   1. Template areas are pairwise distinct: enforced with a minimum gap check when sampling masks.
#   2. Marker-group counts are pairwise distinct: enforced by constructing consecutive count profiles.
#   3. Exactly K marker colors appear in the middle band: enforced by allocating one global count group
#      per template.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a mixed-column stamp map: read templates, globally ranked marker groups, and palette colors.

    Input structure:
    - The grid still has K template columns of width N, separated by dashed 1-cell dividers.
    - Top band (rows 0..N-1): K monochrome template masks, one per tile column.
    - Middle band (rows N+1..2N): marker pixels of K arbitrary colors; the same tile column may mix them.
    - Bottom band (rows 2N+2..): one palette swatch per template column, giving that template's output color.

    Rule:
    - A marker's local coordinates (lr, lc) inside its physical tile column still choose the output N x N block.
    - Marker color no longer names the template directly. Instead, group all markers by color across the whole
      middle band, sort those groups by pixel count from fewest to most, and match them to the templates sorted
      by template area from smallest to largest.
    - After recovering that mapping, stamp the matched template recolored by its aligned palette swatch.

    Key insight: Tile columns are no longer identity buckets; only local position gives destination, while
    template choice is recovered from a global count-order correspondence between marker colors and template sizes.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = int(sep_cols[0])
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    tmasks: list[np.ndarray] = []
    template_areas: list[int] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        mask = sub == col
        tmasks.append(mask)
        template_areas.append(int(mask.sum()))
    assert len(set(template_areas)) == k
    template_order = sorted(range(k), key=lambda j: (template_areas[j], j))

    pal_r0 = int(sep_rows[-1] + 1)
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(int(vals.pop()))

    mid_r0 = n + 1
    records: list[tuple[int, int, int]] = []
    for st in starts:
        sub = g[mid_r0 : mid_r0 + n, st : st + n]
        for lr in range(n):
            for lc in range(n):
                v = int(sub[lr, lc])
                if v == bg or v == sep:
                    continue
                records.append((v, lr, lc))

    assert records
    marker_counts = Counter(v for v, _, _ in records)
    assert len(marker_counts) == k
    assert len(set(marker_counts.values())) == k
    color_order = [col for col, _ in sorted(marker_counts.items(), key=lambda item: (item[1], item[0]))]
    color_to_j = {col: j for col, j in zip(color_order, template_order)}

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    for v, lr, lc in records:
        j = color_to_j[v]
        assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
        slot_owner[(lr, lc)] = j
        tile = np.where(tmasks[j], pcols[j], bg)
        out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator (combo D1/3 + D9/0) ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_K_CHOICES = [3, 4]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.48, 0.88)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_AREA_GAP_MIN = 2

_MIN_MIXED_COLUMNS = 2
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500


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


def _sample_templates(n: int, tcols: list[int]) -> list[np.ndarray] | None:
    for _ in range(120):
        masks = [_sample_template(n, col) for col in tcols]
        areas = sorted(int(mask.sum()) for mask in masks)
        if len(set(areas)) != len(areas):
            continue
        if any(area <= 0 for area in areas):
            continue
        if min(b - a for a, b in zip(areas, areas[1:])) < _AREA_GAP_MIN:
            continue
        return masks
    return None


def _sample_palette_colors(tcols: list[int]) -> list[int]:
    """Return a derangement of template colors for the palette band."""
    while True:
        pcols = random.sample(tcols, len(tcols))
        if all(src != dst for src, dst in zip(tcols, pcols)):
            return pcols


def _count_profile(n: int, k: int) -> list[int]:
    cap = n * n
    min_total = int(np.ceil(0.55 * cap))
    max_total = int(np.floor(0.9 * cap))
    tail = k * (k - 1) // 2
    lo = max(2, int(np.ceil((min_total - tail) / k)))
    hi = int(np.floor((max_total - tail) / k))
    assert lo <= hi
    start = random.randint(lo, hi)
    return [start + i for i in range(k)]


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc, _phys in placements:
        tile = np.where(templates[j], pcols[j], _BG)
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


def _assign_physical_columns(
    k: int,
    marker_color_by_j: dict[int, int],
    slots_by_j: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int, int, int]] | None:
    placements: list[tuple[int, int, int, int]] = []
    cols_by_j: dict[int, set[int]] = defaultdict(set)
    groups_by_phys: dict[int, set[int]] = defaultdict(set)

    for j, slots in slots_by_j.items():
        spread = random.randint(2, min(k, len(slots)))
        chosen_cols = random.sample(range(k), spread)
        random.shuffle(slots)
        for idx, (rr, cc) in enumerate(slots):
            phys = chosen_cols[idx % spread]
            placements.append((j, rr, cc, phys))
            cols_by_j[j].add(phys)
            groups_by_phys[phys].add(marker_color_by_j[j])

    mixed_cols = sum(1 for cols in groups_by_phys.values() if len(cols) >= 2)
    if mixed_cols < _MIN_MIXED_COLUMNS:
        return None
    if any(len(cols) < 2 for cols in cols_by_j.values()):
        return None

    random.shuffle(placements)
    return placements


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])
    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    n = int(sep_cols[0])
    starts = [0] + [c + 1 for c in sep_cols]
    mid_r0 = n + 1

    groups_by_phys: dict[int, set[int]] = defaultdict(set)
    marker_counts: Counter[int] = Counter()
    for phys, st in enumerate(starts):
        sub = g[mid_r0 : mid_r0 + n, st : st + n]
        for rr in range(n):
            for cc in range(n):
                v = int(sub[rr, cc])
                if v == bg or v == sep:
                    continue
                groups_by_phys[phys].add(v)
                marker_counts[v] += 1

    mixed_columns = sum(1 for cols in groups_by_phys.values() if len(cols) >= 2)
    return {
        "mixed_columns": mixed_columns,
        "distinct_count_levels": len(set(marker_counts.values())),
        "marker_colors": len(marker_counts),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_N_CHOICES)
    k = random.choice(_K_CHOICES)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    marker_colors = random.sample([c for c in _AVAILABLE_COLORS if c not in tcols], k)
    pcols = _sample_palette_colors(tcols)

    templates = _sample_templates(n, tcols)
    if templates is None:
        return None

    template_order = sorted(range(k), key=lambda j: (int(templates[j].sum()), j))
    count_profile = _count_profile(n, k)

    marker_plan = list(zip(random.sample(marker_colors, k), count_profile, template_order))
    marker_color_by_j: dict[int, int] = {}
    slots_by_j: dict[int, list[tuple[int, int]]] = defaultdict(list)

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    need = sum(count_profile)
    if need > len(all_slots):
        return None
    cursor = 0
    for color, cnt, j in marker_plan:
        marker_color_by_j[j] = color
        slots_by_j[j] = all_slots[cursor : cursor + cnt]
        cursor += cnt

    placements = _assign_physical_columns(k, marker_color_by_j, slots_by_j)
    if placements is None:
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    expected = _build_expected(n, templates, pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    for j, rr, cc, phys in placements:
        grid[n + 1 + rr, _raw_col(phys, cc, n)] = marker_color_by_j[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    metrics = _fusion_metrics(grid)
    if metrics["mixed_columns"] < _MIN_MIXED_COLUMNS:
        return None
    if metrics["marker_colors"] != k or metrics["distinct_count_levels"] != k:
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
