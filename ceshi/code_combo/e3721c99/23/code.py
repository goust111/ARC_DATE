# Combo variant of e3721c99 (index 23)
# combo_spec: D1/0+D7/1
# Name: 形状轮廓匹配+逐精灵打包输出

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    The divider and legend panel remain visible in the input, but the legend now
#    encodes outer silhouettes rather than hole counts. A grey sprite matches a
#    legend when both shapes become identical after filling enclosed cavities and
#    cropping away translation. The output then discards the whole scene layout:
#    only matched sprites survive, recolored and tightly cropped, then packed into
#    compact rows following their original top-to-bottom / left-to-right order.
#    Small trace: if a red legend is a T silhouette with one cavity and a grey T
#    sprite has two cavities, the old task would miss it because 2 != 1. Here it
#    still turns red because both fill to the same T. Instead of staying in place,
#    that red sprite becomes one cropped patch inside the packed answer grid.
# b) Complexity calibration:
#    A fixed template library keeps silhouettes visually distinct and avoids noisy
#    amorphous contour comparisons. Each sample uses 2-4 legend shapes and 8-11
#    total sprites, with repeated matches per silhouette, so both the contour rule
#    and the packed-output rule have multiple visible supports.
# c) Hidden assumptions from the original main():
#    - The original legend key "hole_count -> color" is invalid once different
#      legend silhouettes may share the same number of holes.
#    - The original per-sprite scalar hole count throws away the geometry needed
#      for shape matching; we must compare filled silhouettes instead.
#    - The original full-scene output copy is invalid because legend, divider,
#      background, and distractors no longer belong to the answer.
#    - Once output is object-level, row grouping must depend on visible bbox
#      positions rather than any hidden attribute such as color id.
# d) Placement legality:
#    Legends stay inside a blocked border panel and sprites stay in the workspace
#    with at least one-cell separation. Matched sprites are biased into 2-3 row
#    bands so the packed ordering is legible and reproducible from visible rows.
# e) Visual observability:
#    The generator deliberately gives each legend at least one matching grey sprite
#    whose hole count differs from the legend's own hole count whenever possible.
#    Across examples, a human can see that equal outer contour matters more than
#    equal cavity count, and that only those matched sprites reappear in the output.
# f) Visual cleanliness:
#    Template silhouettes are thick, compact, and separated by background gaps.
#    The output is a clean collage with one-cell gaps and no outer border, so the
#    solver never has to parse overlapping marks or mixed scene layers.
# g) Color-change inferability:
#    Each legend color is tied to one repeated silhouette. The packing order uses
#    only source-row and source-column positions, not raw color numbers, so color
#    change remains grounded in visible shape evidence.
# h) Quantitative anti-degeneration target:
#    We reject generated samples when a hole-count-based packed solver would produce
#    the same answer. Samples are accepted only if shape matching changes the result
#    directly, or if duplicated legend hole counts make the old rule impossible.
# i) New solver assumptions and generator guarantees:
#    - Solver assumes one non-border-touching legend component per legend color;
#      generator assigns distinct colors and keeps legends off the outer border.
#    - Solver assumes no rotation or reflection during matching; generator places
#      sprites in the same orientation as their legend template.
#    - Solver assumes packed rows can be recovered from bbox gaps; generator places
#      matched sprites in separated horizontal bands to make those row groups stable.


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed background components in a cropped boolean mask."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only the background components fully enclosed by the mask bbox."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    out = mask.copy()
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            pts = []
            touches = False
            while st:
                r, c = st.pop()
                pts.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                for r, c in pts:
                    out[r, c] = True
    return out


def _component_bbox_mask(pts: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    """Return bbox coordinates plus a tight boolean mask for one component."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return int(r0), int(c0), int(r1), int(c1), sub


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Canonicalize a component by its hole-filled silhouette."""
    filled = _fill_internal_holes(mask)
    return tuple(tuple(int(v) for v in row) for row in filled.astype(int))


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[tuple[tuple[int, ...], ...], int]:
    """Map each legend silhouette key to its color."""
    h, w = g.shape
    mapping: dict[tuple[tuple[int, ...], ...], int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        key = _shape_key(_component_bbox_mask(candidates[0])[4])
        assert key not in mapping, "legend silhouettes must map to exactly one color"
        mapping[key] = int(c)
    return mapping


def _legend_hole_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int] | None:
    """
    Old-task helper for anti-degeneration checks.
    Returns None when duplicate legend hole counts make the original rule invalid.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        if len(candidates) != 1:
            return None
        sub = _component_bbox_mask(candidates[0])[4]
        hc = _hole_count(sub)
        if hc in mapping:
            return None
        mapping[hc] = int(c)
    return mapping


def _matched_sprite_items_by_shape(
    g: np.ndarray,
    legend: dict[tuple[tuple[int, ...], ...], int],
    grey: int,
    bg: int,
) -> list[dict]:
    """Extract matched grey sprites using the fused silhouette rule."""
    items: list[dict] = []
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1, sub = _component_bbox_mask(pts)
        key = _shape_key(sub)
        if key not in legend:
            continue
        color = int(legend[key])
        patch = np.where(sub, color, bg).astype(int)
        items.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "patch": patch,
            }
        )
    return items


def _matched_sprite_items_by_holes(
    g: np.ndarray,
    legend: dict[int, int],
    grey: int,
    bg: int,
) -> list[dict]:
    """Old-task helper: extract matched sprites using hole counts only."""
    items: list[dict] = []
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1, sub = _component_bbox_mask(pts)
        hc = _hole_count(sub)
        if hc not in legend:
            continue
        color = int(legend[hc])
        patch = np.where(sub, color, bg).astype(int)
        items.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "patch": patch,
            }
        )
    return items


def _group_rows(items: list[dict]) -> list[list[dict]]:
    """Group packed-output items by visible row bands in the source scene."""
    ordered = sorted(items, key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    rows: list[list[dict]] = []
    current: list[dict] = []
    current_bottom: int | None = None
    for item in ordered:
        if current and item["r0"] > int(current_bottom) + 1:
            rows.append(sorted(current, key=lambda x: (x["c0"], x["r0"])))
            current = [item]
            current_bottom = int(item["r1"])
        else:
            current.append(item)
            current_bottom = int(item["r1"]) if current_bottom is None else max(int(current_bottom), int(item["r1"]))
    if current:
        rows.append(sorted(current, key=lambda x: (x["c0"], x["r0"])))
    return rows


def _pack_rows(rows: list[list[dict]], bg: int) -> np.ndarray:
    """Pack tight object patches with one-cell gaps and no outer border."""
    assert rows, "packed output needs at least one matched row"
    row_heights = [max(int(item["patch"].shape[0]) for item in row) for row in rows]
    row_widths = [
        sum(int(item["patch"].shape[1]) for item in row) + (len(row) - 1)
        for row in rows
    ]
    out_h = sum(row_heights) + len(rows) - 1
    out_w = max(row_widths)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    rr = 0
    for row, row_h in zip(rows, row_heights):
        cc = 0
        for item in row:
            patch = item["patch"]
            paste(out, patch, rr, cc, transparent=bg)
            cc += int(patch.shape[1]) + 1
        rr += int(row_h) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor silhouette-matching grey sprites and output only a packed collage of those sprites.

    Input structure:
    - The dominant color is background, with grey (5) sprites scattered in the workspace.
    - Colored legend components sit in a border-adjacent panel behind a divider.
    - Legend shapes may have different hole counts from the grey sprites that should match them.

    Rule:
    - Treat each legend component as one outer silhouette, obtained after filling its enclosed holes.
    - For every grey sprite, fill its enclosed holes, compare the resulting silhouette to the legends,
      and keep only exact matches up to translation.
    - Recolor each kept sprite to the matched legend color, crop it tightly, then pack the kept
      sprites row by row using their original top-to-bottom and left-to-right layout. Discard the
      legend panel, divider, background, and all non-matching grey sprites.

    Key insight:
    - The puzzle combines two non-local changes at once: color comes from geometric contour
      matching rather than hole counting, and the answer is a compact object collage rather than
      the original full scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    items = _matched_sprite_items_by_shape(g, legend, grey, bg)
    assert items, "grid must contain at least one grey sprite matching a legend silhouette"
    rows = _group_rows(items)
    return _pack_rows(rows, bg)


# --------------- generator (contour-first templates + packed-row output) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
GEN_RETRIES = 300

FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3
PANEL_PADDING = 2
PANEL_EXTRA_THICKNESS = 2

MAX_VARIANT_HOLES = 2
MIN_HOLE_SEP = 2

MATCHED_ROW_COUNT_RANGE = (2, 3)
MATCHED_ROW_JITTER = 2
MATCHED_ROW_GAP = 2

_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_STRUCT8 = generate_binary_structure(2, 2)


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


OUTER_TEMPLATES: dict[str, np.ndarray] = {
    "l_left": _mask_from_rows([
        "11110",
        "11110",
        "11110",
        "11110",
        "11111",
        "11111",
    ]),
    "l_right": _mask_from_rows([
        "01111",
        "01111",
        "01111",
        "01111",
        "11111",
        "11111",
    ]),
    "t_cap": _mask_from_rows([
        "111111",
        "111111",
        "001100",
        "001100",
        "001100",
        "001100",
    ]),
    "plus": _mask_from_rows([
        "011110",
        "011110",
        "111111",
        "111111",
        "011110",
        "011110",
    ]),
    "step_left": _mask_from_rows([
        "111100",
        "111100",
        "111111",
        "001111",
        "001111",
        "001111",
    ]),
    "step_right": _mask_from_rows([
        "001111",
        "001111",
        "111111",
        "111100",
        "111100",
        "111100",
    ]),
    "c_shape": _mask_from_rows([
        "111111",
        "111111",
        "110000",
        "110000",
        "111111",
        "111111",
    ]),
    "p_shape": _mask_from_rows([
        "11110",
        "11111",
        "11111",
        "11110",
        "11100",
        "11100",
    ]),
}


def _template_safe_sites(mask: np.ndarray) -> list[tuple[int, int]]:
    core = binary_erosion(mask, structure=_STRUCT4, border_value=0)
    return [tuple(map(int, rc)) for rc in np.argwhere(core)]


def _seed_groups(sites: list[tuple[int, int]], count: int) -> list[tuple[tuple[int, int], ...]]:
    valid = []
    for comb in itertools.combinations(sites, count):
        ok = True
        for (r1, c1), (r2, c2) in itertools.combinations(comb, 2):
            if max(abs(r1 - r2), abs(c1 - c2)) < MIN_HOLE_SEP:
                ok = False
                break
        if ok:
            valid.append(tuple(sorted(comb)))
    return valid


def _enumerate_variants(outer: np.ndarray) -> dict[int, list[np.ndarray]]:
    variants: dict[int, list[np.ndarray]] = {0: [outer.copy()]}
    sites = _template_safe_sites(outer)
    for hole_count in range(1, MAX_VARIANT_HOLES + 1):
        cur = []
        for seeds in _seed_groups(sites, hole_count):
            mask = outer.copy()
            for r, c in seeds:
                mask[r, c] = False
            if not is_contiguous(mask, background=0, connectivity=4):
                continue
            if _hole_count(mask) != hole_count:
                continue
            if not np.array_equal(_fill_internal_holes(mask), outer):
                continue
            cur.append(mask)
        if cur:
            variants[hole_count] = cur
    return variants


TEMPLATE_VARIANTS = {name: _enumerate_variants(mask) for name, mask in OUTER_TEMPLATES.items()}

for _name, _outer in OUTER_TEMPLATES.items():
    assert is_contiguous(_outer, background=0, connectivity=4)
    assert np.array_equal(_fill_internal_holes(_outer), _outer)
    assert len(TEMPLATE_VARIANTS[_name]) >= 2, f"template {_name} must support at least two hole variants"


def _rand_between(lo: int, hi: int) -> int:
    lo = int(lo)
    hi = int(hi)
    assert lo <= hi
    return lo if lo == hi else random.randint(lo, hi)


def _render_template(name: str, color: int, hole_count: int) -> np.ndarray:
    mask = random.choice(TEMPLATE_VARIANTS[name][hole_count])
    return np.where(mask, int(color), BG).astype(int)


def _template_options(name: str) -> list[int]:
    return sorted(TEMPLATE_VARIANTS[name].keys())


def _legend_patch_size(legend: dict) -> tuple[int, int]:
    return legend["patch"].shape


def _line_span(legends: list[dict], arrangement: str) -> int:
    idx = 1 if arrangement == "horizontal" else 0
    return sum(_legend_patch_size(legend)[idx] for legend in legends) + len(legends) - 1


def _cross_span(legends: list[dict], arrangement: str) -> int:
    idx = 0 if arrangement == "horizontal" else 1
    return max(_legend_patch_size(legend)[idx] for legend in legends)


def _choose_legends(arrangement: str, max_span: int) -> list[dict]:
    del arrangement
    viable = [name for name in OUTER_TEMPLATES if len(_template_options(name)) >= 2]
    for _ in range(200):
        count = random.randint(*LEGEND_COUNT_RANGE)
        names = random.sample(viable, count)
        random.shuffle(names)
        colors = random.sample(LEGEND_COLORS, count)
        legends = []
        for name, color in zip(names, colors):
            opts = _template_options(name)
            legends.append(
                {
                    "template": name,
                    "color": int(color),
                    "legend_holes": random.choice(opts),
                }
            )

        if count >= 2 and random.random() < 0.7:
            i, j = random.sample(range(count), 2)
            shared = sorted(set(_template_options(names[i])) & set(_template_options(names[j])))
            if shared:
                legends[j]["legend_holes"] = random.choice(shared)

        for legend in legends:
            legend["patch"] = _render_template(
                legend["template"], legend["color"], legend["legend_holes"]
            )

        if _line_span(legends, "horizontal") <= max_span and _line_span(legends, "vertical") <= max_span:
            return legends
    raise RuntimeError("failed to choose legend set")


def _place_legend_line(grid: np.ndarray, legends: list[dict], arrangement: str, r: int, c: int) -> None:
    rr, cc = int(r), int(c)
    for legend in legends:
        patch = legend["patch"]
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends, "horizontal")
    cross = _cross_span(legends, "horizontal")
    panel_h = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = _rand_between(1, panel_h - cross - 1)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = _rand_between(div_r + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends, "vertical")
    cross = _cross_span(legends, "vertical")
    panel_w = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = _rand_between(1, panel_w - cross - 1)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = _rand_between(div_c + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str,
    panel_h: int,
    panel_w: int,
    cross_dim: int,
    along_dim: int,
) -> tuple[int, int, int, int, int, int]:
    n = GRID_SIZE
    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim
    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends, arrangement)
    cross = _cross_span(legends, arrangement)
    along = total + PANEL_PADDING + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + PANEL_PADDING)))
    cross_panel = cross + PANEL_PADDING + random.randint(0, PANEL_EXTRA_THICKNESS)
    if arrangement == "horizontal":
        panel_h, panel_w = cross_panel, along
        cross_dim, along_dim = cross, total
    else:
        panel_h, panel_w = along, cross_panel
        cross_dim, along_dim = total, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, cross_dim, along_dim
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = _rand_between(r_min, r_max)
    c0 = _rand_between(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_plan(legends: list[dict]) -> list[dict]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, legend in enumerate(legends):
        opts = _template_options(legend["template"])
        different = [h for h in opts if h != legend["legend_holes"]]
        first_holes = random.choice(different) if different else legend["legend_holes"]
        plan.append(
            {
                "template": legend["template"],
                "sprite_holes": first_holes,
                "color": legend["color"],
            }
        )
        for _ in range(extras[i]):
            plan.append(
                {
                    "template": legend["template"],
                    "sprite_holes": random.choice(opts),
                    "color": legend["color"],
                }
            )

    legend_templates = {legend["template"] for legend in legends}
    distractor_templates = [name for name in OUTER_TEMPLATES if name not in legend_templates]
    for _ in range(n_distractors):
        name = random.choice(distractor_templates)
        plan.append(
            {
                "template": name,
                "sprite_holes": random.choice(_template_options(name)),
                "color": None,
            }
        )
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    sprite: np.ndarray,
    row_min: int | None = None,
    row_max: int | None = None,
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    r_lo = 0 if row_min is None else max(0, int(row_min))
    r_hi = n - h if row_max is None else min(n - h, int(row_max))
    if r_hi < r_lo:
        return None
    candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _assign_matched_rows(items: list[np.ndarray], row_count: int) -> list[list[np.ndarray]]:
    ordered = items[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda sprite: (sprite.shape[1], sprite.shape[0], sprite.size), reverse=True)
    rows: list[list[np.ndarray]] = [[] for _ in range(row_count)]
    widths = [0] * row_count
    heights = [0] * row_count
    for idx, sprite in enumerate(ordered):
        if idx < row_count:
            target = idx
        else:
            target = min(range(row_count), key=lambda k: (widths[k], heights[k], len(rows[k])))
        rows[target].append(sprite)
        widths[target] += int(sprite.shape[1]) + (1 if len(rows[target]) > 1 else 0)
        heights[target] = max(heights[target], int(sprite.shape[0]))
    return rows


def _row_tops(row_heights: list[int]) -> list[int] | None:
    need = sum(int(h) + MATCHED_ROW_JITTER for h in row_heights) + MATCHED_ROW_GAP * (len(row_heights) - 1)
    if need > GRID_SIZE:
        return None
    slack = GRID_SIZE - need
    extras = [0] * (len(row_heights) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1
    tops = []
    cur = extras[0]
    for i, h in enumerate(row_heights):
        tops.append(int(cur))
        cur += int(h) + MATCHED_ROW_JITTER
        if i < len(row_heights) - 1:
            cur += MATCHED_ROW_GAP + extras[i + 1]
    return tops


def _place_matched_sprites(
    grid: np.ndarray,
    blocked: np.ndarray,
    occupied: np.ndarray,
    sprites: list[np.ndarray],
) -> bool:
    min_rows = min(MATCHED_ROW_COUNT_RANGE[0], len(sprites))
    max_rows = min(MATCHED_ROW_COUNT_RANGE[1], len(sprites))
    options = list(range(max_rows, min_rows - 1, -1))
    random.shuffle(options)

    for row_count in options:
        rows = _assign_matched_rows(sprites, row_count)
        row_heights = [max(int(sprite.shape[0]) for sprite in row) for row in rows]
        tops = _row_tops(row_heights)
        if tops is None:
            continue

        trial_grid = grid.copy()
        trial_occ = occupied.copy()
        ok = True
        for top, row in zip(tops, rows):
            band_max = top + MATCHED_ROW_JITTER
            row_items = row[:]
            random.shuffle(row_items)
            row_items.sort(key=lambda sprite: sprite.size, reverse=True)
            for sprite in row_items:
                loc = _random_sprite_location(
                    blocked, trial_occ, sprite, row_min=top, row_max=band_max
                )
                if loc is None:
                    ok = False
                    break
                r, c = loc
                paste(trial_grid, sprite, r, c, transparent=BG)
                paste(trial_occ, sprite, r, c, transparent=BG)
            if not ok:
                break
        if ok:
            grid[:, :] = trial_grid
            occupied[:, :] = trial_occ
            return True
    return False


def _place_sprites(grid: np.ndarray, blocked: np.ndarray, legends: list[dict]) -> bool:
    matched: list[np.ndarray] = []
    distractors: list[np.ndarray] = []
    for item in _sprite_plan(legends):
        sprite = _render_template(item["template"], GREY, item["sprite_holes"])
        if item["color"] is None:
            distractors.append(sprite)
        else:
            matched.append(sprite)

    occupied = np.zeros_like(grid)
    if not _place_matched_sprites(grid, blocked, occupied, matched):
        return False

    distractors.sort(key=lambda sprite: sprite.size, reverse=True)
    for sprite in distractors:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
    return True


def _old_style_packed_output(grid: np.ndarray) -> np.ndarray | None:
    """Packed answer under hole-count matching only, used to reject degeneracy."""
    hole_legend = _legend_hole_mapping(grid, BG, GREY)
    if hole_legend is None:
        return None
    items = _matched_sprite_items_by_holes(grid, hole_legend, GREY, BG)
    if not items:
        return np.zeros((0, 0), dtype=int)
    return _pack_rows(_group_rows(items), BG)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Expose fused-behavior indicators for validation scripts."""
    shape_out = main(grid)
    hole_legend = _legend_hole_mapping(grid, BG, GREY)
    duplicate_holes = int(hole_legend is None)
    if hole_legend is None:
        same_as_hole_rule = 0
        old_kept = -1
    else:
        old_out = _old_style_packed_output(grid)
        same_as_hole_rule = int(old_out is not None and np.array_equal(shape_out, old_out))
        old_items = _matched_sprite_items_by_holes(grid, hole_legend, GREY, BG)
        old_kept = len(old_items)

    shape_legend = _legend_mapping(grid, BG, GREY)
    shape_items = _matched_sprite_items_by_shape(grid, shape_legend, GREY, BG)
    rows = _group_rows(shape_items)
    return {
        "duplicate_legend_holes": duplicate_holes,
        "same_as_hole_rule": same_as_hole_rule,
        "shape_kept": len(shape_items),
        "hole_kept": old_kept,
        "packed_rows": len(rows),
        "out_h": int(shape_out.shape[0]),
        "out_w": int(shape_out.shape[1]),
    }


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    if not _place_sprites(grid, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    out = main(grid)
    if out.shape == grid.shape:
        return None
    if out.shape[0] > GRID_SIZE or out.shape[1] > GRID_SIZE:
        return None

    shape_legend = _legend_mapping(grid, BG, GREY)
    items = _matched_sprite_items_by_shape(grid, shape_legend, GREY, BG)
    rows = _group_rows(items)
    if len(rows) < 2:
        return None
    if not np.array_equal(out, _pack_rows(rows, BG)):
        return None

    old_out = _old_style_packed_output(grid)
    if old_out is not None and np.array_equal(out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
