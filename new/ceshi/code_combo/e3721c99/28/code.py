# Combo variant of e3721c99 (index 28)
# combo_spec: D8/1+D10/2
# Name: 嵌套洞图例+面积加权洞计数

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete appearance and behavior trace
#    - The original task matched grey sprites to colored legends by ordinary hole count.
#      Here the legend itself changes shape and the counting rule changes at the same time.
#    - Each legend is now a connected nested object made of 1-3 chambers. Every chamber has:
#        1. one outer cavity (always area >= 2, so it always counts), and
#        2. one inner cavity inside an attached same-color island.
#      The inner cavity is either:
#        - tiny: area 1, so it is visible but ignored by the new rule, or
#        - large: area 2, so it is visible and counted.
#    - Small trace:
#        legend signature [tiny, tiny]     -> raw holes = 4, weighted holes = 2
#        legend signature [tiny, large]    -> raw holes = 4, weighted holes = 3
#        grey sprite with [big, big, tiny] -> raw holes = 3, weighted holes = 2
#      Old logic would map by raw counts (4 or 3). New logic maps by weighted counts (2 or 3).
#      The difference is visible because the ignored cavities are still present in the input.
#
# b) Complexity calibration
#    - Legend values are limited to 1-4 effective holes. This keeps the mapping small enough to infer
#      from 2-4 examples while still allowing multiple distinct nested templates.
#    - Every legend contains at least one tiny inner cavity, so the D8 filter is not optional noise.
#    - At least one legend per sample also contains a counted inner cavity (area 2), forcing the solver
#      to notice that some inner cavities matter and some do not.
#    - Grey sprites are smaller than legends and use flat perforations rather than nested islands, so the
#      two roles stay visually distinct.
#
# c) Implicit-assumption audit
#    - Original _hole_count assumed "every enclosed background component contributes 1". That is invalid:
#      single-cell holes must now be ignored.
#    - The D10 single-variant solver used a special legend-only top-level-hole count. That is not the rule
#      here: nested holes still count if their area is at least 2, so both legends and sprites must use the
#      same area-threshold counter.
#    - Original legend detection tolerated 3x3 legend glyphs. These larger nested legends need a stricter
#      size filter so divider fragments cannot be misread as legends.
#
# d) Placement legality
#    - Legend colors exclude the divider color, so no legend can merge with the border divider.
#    - Legends stay inside a dedicated panel and never touch the outer grid border.
#    - Grey sprites are placed only in the workspace, with 8-neighborhood padding between sprites and a
#      1-cell moat from blocked regions.
#
# e) Visual observability
#    - A human can see nested inner cavities inside the legends and can also see that some cavities are
#      1 pixel while others are 2 pixels.
#    - Because recoloring groups sprites by the number of "non-tiny" holes, the examples expose the new
#      rule directly: visible holes are not all equal anymore.
#
# f) Visual cleanliness
#    - Legend panels use at most two rows or two columns with fixed gaps.
#    - Grey sprites remain single connected objects, and their protrusions are added only from the exterior,
#      so the silhouettes vary without creating clutter inside the hole structure.
#
# g) Color-change inferability
#    - Colors are arbitrary labels; the transferable structure is the effective hole count after applying the
#      area >= 2 threshold. Repeated matched sprites per legend color give support for that mapping.
#
# h) New main() assumptions and generator guarantees
#    - main() assumes each non-bg non-grey interior color corresponds to exactly one legend component.
#      generate_input() creates exactly one such component per legend color.
#    - main() assumes weighted hole counting is done on each component's tight bbox using 4-connectivity.
#      generate_input() builds both legend and sprite templates to respect that measurement.
#    - main() assumes at least one legend exists and all matched sprites use legend counts that are unique.
#      generate_input() samples distinct effective legend values and validates the final grid by replaying main().


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]
MIN_COUNTED_HOLE_AREA = 2

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 3)
LEGEND_VALUES = (1, 2, 3, 4)
LEGEND_SIGNATURES: dict[int, list[tuple[int, ...]]] = {
    1: [(0,)],
    2: [(0, 0)],
    3: [(0, 1), (0, 0, 0)],
    4: [(0, 0, 1)],
}
FULL_PANEL_THICKNESS = 17
L_PANEL_WIDE = (17, 22)   # (height, width) for horizontal legend packing
L_PANEL_TALL = (22, 17)   # (height, width) for vertical legend packing

# -- sprites --
TOTAL_SPRITES_RANGE = (6, 8)
MAX_DISTRACTORS = 4
MAX_EXTRA_PER_LEGEND = 2
SPRITE_EFFECTIVE_VALUES = (0, 1, 2, 3, 4)
SPRITE_MAX_H = 18
SPRITE_MAX_W = 18

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


# ---------------- topology helpers ----------------

def _hole_areas(mask: np.ndarray) -> list[int]:
    """Areas of enclosed background components in a binary mask, using 4-connectivity."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    areas: list[int] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            area = 0
            touches = False
            while st:
                r, c = st.pop()
                area += 1
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                areas.append(area)
    return areas


def _raw_hole_count(mask: np.ndarray) -> int:
    """Ordinary hole count: every enclosed background component contributes 1."""
    return len(_hole_areas(mask))


def _weighted_hole_count(mask: np.ndarray) -> int:
    """Count only enclosed background components whose area is at least 2."""
    return sum(area >= MIN_COUNTED_HOLE_AREA for area in _hole_areas(mask))


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    """Background cells connected to the border of the mask array."""
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    st = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                st.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                st.append((r, c))
    while st:
        r, c = st.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and not ext[rr, cc]:
                ext[rr, cc] = True
                st.append((rr, cc))
    return ext


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg,grey) color and map weighted-hole-count -> color.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        hc = _weighted_hole_count(sub)
        assert hc not in mapping, f"legend weighted hole count {hc} must map to one color"
        mapping[hc] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching their area-weighted hole count to nested legend shapes.

    Input structure:
    - The most common color is the background, grey (5) blobs sit in the workspace, and a divider splits
      off a legend panel near one edge or corner.
    - Each legend is a large colored component with nested cavities: every chamber has an outer cavity plus
      an inner cavity inside an attached island, and some of those inner cavities are only 1 cell large.

    Rule:
    - For every legend and every grey sprite, count enclosed background components inside its own bbox using
      4-connectivity, but only keep holes whose area is at least 2.
    - Each legend defines a mapping from that weighted hole count to its color.
    - Recolor each grey sprite to the color of the legend with the same weighted count; erase grey sprites
      whose weighted counts are missing from the legend.

    Key insight:
    - The visible cavities are not all worth the same amount. Tiny 1-cell holes must be ignored even when
      they are fully enclosed and nested inside another cavity-bearing structure.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "grid must contain nested legend components with weighted hole counts"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _weighted_hole_count(sub)
        newc = int(legend[hc]) if hc in legend else bg
        for r, c in pts:
            out[r, c] = newc

    return out


# ---------------- legend helpers ----------------

def _tiny_nested_chamber(color: int) -> np.ndarray:
    """7x7 chamber: raw holes = 2, weighted holes = 1."""
    patch = np.zeros((7, 7), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1, 3] = color
    patch[2:5, 2:5] = color
    patch[3, 3] = BG
    return patch


def _large_nested_chamber(color: int) -> np.ndarray:
    """7x8 chamber: raw holes = 2, weighted holes = 2."""
    patch = np.zeros((7, 8), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1, 4] = color
    patch[2:5, 2:6] = color
    patch[3, 3:5] = BG
    return patch


def _legend_patch(signature: tuple[int, ...], color: int, arrangement: str) -> np.ndarray:
    """Build a connected nested legend component from tiny/large inner-hole chambers."""
    chambers = [_large_nested_chamber(color) if flag else _tiny_nested_chamber(color) for flag in signature]
    width = sum(ch.shape[1] for ch in chambers) - (len(chambers) - 1)
    patch = np.zeros((7, width), dtype=int)
    c0 = 0
    for chamber in chambers:
        w = chamber.shape[1]
        patch[:, c0 : c0 + w] = np.maximum(patch[:, c0 : c0 + w], chamber)
        c0 += w - 1
    if arrangement == "vertical":
        patch = patch.T
    mask = patch == color
    assert _raw_hole_count(mask) == 2 * len(signature)
    assert _weighted_hole_count(mask) == len(signature) + sum(signature)
    return patch


def _choose_legends(count: int) -> list[tuple[int, tuple[int, ...], int]]:
    """Sample distinct effective values and concrete nested signatures for the legends."""
    while True:
        values = random.sample(list(LEGEND_VALUES), count)
        if any(v >= 3 for v in values):
            break
    colors = random.sample(LEGEND_COLORS, count)
    random.shuffle(values)
    legends = [(value, random.choice(LEGEND_SIGNATURES[value]), color) for value, color in zip(values, colors)]
    if not any(sum(signature) > 0 for _, signature, _ in legends):
        idx = next(i for i, (value, _, _) in enumerate(legends) if value >= 3)
        value, _, color = legends[idx]
        options = [signature for signature in LEGEND_SIGNATURES[value] if sum(signature) > 0]
        legends[idx] = (value, random.choice(options), color)
    return legends


def _place_horizontal_rows(
    grid: np.ndarray, legends: list[tuple[int, tuple[int, ...], int]], panel_r0: int, panel_c0: int, panel_w: int
) -> None:
    entries = [
        (value, color, _legend_patch(signature, color, "horizontal"))
        for value, signature, color in legends
    ]
    row_a = panel_r0 + 1
    row_b = panel_r0 + 9
    if len(entries) == 2:
        total_w = entries[0][2].shape[1] + 1 + entries[1][2].shape[1]
        if total_w <= panel_w - 2 and random.random() < 0.5:
            row = random.choice([row_a, row_b])
            order = entries[:]
            random.shuffle(order)
            start = panel_c0 + random.randint(1, panel_w - 1 - total_w)
            paste(grid, order[0][2], row, start)
            paste(grid, order[1][2], row, start + order[0][2].shape[1] + 1)
            return
        rows = [row_a, row_b]
        random.shuffle(rows)
        for row, entry in zip(rows, entries):
            patch = entry[2]
            start = panel_c0 + random.randint(1, panel_w - 1 - patch.shape[1])
            paste(grid, patch, row, start)
        return

    entries.sort(key=lambda item: item[2].shape[1], reverse=True)
    single_row, double_row = random.choice([(row_a, row_b), (row_b, row_a)])
    single = entries[0][2]
    start = panel_c0 + random.randint(1, panel_w - 1 - single.shape[1])
    paste(grid, single, single_row, start)
    pair = entries[1:]
    random.shuffle(pair)
    total_w = pair[0][2].shape[1] + 1 + pair[1][2].shape[1]
    start = panel_c0 + random.randint(1, panel_w - 1 - total_w)
    paste(grid, pair[0][2], double_row, start)
    paste(grid, pair[1][2], double_row, start + pair[0][2].shape[1] + 1)


def _place_vertical_cols(
    grid: np.ndarray, legends: list[tuple[int, tuple[int, ...], int]], panel_r0: int, panel_c0: int, panel_h: int
) -> None:
    entries = [
        (value, color, _legend_patch(signature, color, "vertical"))
        for value, signature, color in legends
    ]
    col_a = panel_c0 + 1
    col_b = panel_c0 + 9
    if len(entries) == 2:
        total_h = entries[0][2].shape[0] + 1 + entries[1][2].shape[0]
        if total_h <= panel_h - 2 and random.random() < 0.5:
            col = random.choice([col_a, col_b])
            order = entries[:]
            random.shuffle(order)
            start = panel_r0 + random.randint(1, panel_h - 1 - total_h)
            paste(grid, order[0][2], start, col)
            paste(grid, order[1][2], start + order[0][2].shape[0] + 1, col)
            return
        cols = [col_a, col_b]
        random.shuffle(cols)
        for col, entry in zip(cols, entries):
            patch = entry[2]
            start = panel_r0 + random.randint(1, panel_h - 1 - patch.shape[0])
            paste(grid, patch, start, col)
        return

    entries.sort(key=lambda item: item[2].shape[0], reverse=True)
    single_col, double_col = random.choice([(col_a, col_b), (col_b, col_a)])
    single = entries[0][2]
    start = panel_r0 + random.randint(1, panel_h - 1 - single.shape[0])
    paste(grid, single, start, single_col)
    pair = entries[1:]
    random.shuffle(pair)
    total_h = pair[0][2].shape[0] + 1 + pair[1][2].shape[0]
    start = panel_r0 + random.randint(1, panel_h - 1 - total_h)
    paste(grid, pair[0][2], start, double_col)
    paste(grid, pair[1][2], start + pair[0][2].shape[0] + 1, double_col)


# ---------------- layout helpers ----------------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends(random.randint(*LEGEND_COUNT_RANGE))
    panel_h = FULL_PANEL_THICKNESS
    top = random.choice([True, False])
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        _place_horizontal_rows(grid, legends, 0, 0, GRID_SIZE)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        _place_horizontal_rows(grid, legends, div_r + 1, 0, GRID_SIZE)
    return [(value, color) for value, _, color in legends]


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends(random.randint(*LEGEND_COUNT_RANGE))
    panel_w = FULL_PANEL_THICKNESS
    left = random.choice([True, False])
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        _place_vertical_cols(grid, legends, 0, 0, GRID_SIZE)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        _place_vertical_cols(grid, legends, 0, div_c + 1, GRID_SIZE)
    return [(value, color) for value, _, color in legends]


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(2)
    if arrangement == "horizontal":
        panel_h, panel_w = L_PANEL_WIDE
    else:
        panel_h, panel_w = L_PANEL_TALL
    corner = random.choice(["tl", "tr", "bl", "br"])
    n = GRID_SIZE

    if corner == "tl":
        div_r, div_c = panel_h, panel_w
        grid[div_r, : div_c + 1] = DIV
        grid[: div_r + 1, div_c] = DIV
        blocked[:div_r, :div_c] = 9
        blocked[div_r, : div_c + 1] = 9
        blocked[: div_r + 1, div_c] = 9
        panel_r0, panel_c0 = 0, 0
    elif corner == "tr":
        div_r, div_c = panel_h, n - 1 - panel_w
        grid[div_r, div_c:] = DIV
        grid[: div_r + 1, div_c] = DIV
        blocked[:div_r, div_c + 1 :] = 9
        blocked[div_r, div_c:] = 9
        blocked[: div_r + 1, div_c] = 9
        panel_r0, panel_c0 = 0, div_c + 1
    elif corner == "bl":
        div_r, div_c = n - 1 - panel_h, panel_w
        grid[div_r, : div_c + 1] = DIV
        grid[div_r:, div_c] = DIV
        blocked[div_r + 1 :, :div_c] = 9
        blocked[div_r, : div_c + 1] = 9
        blocked[div_r:, div_c] = 9
        panel_r0, panel_c0 = div_r + 1, 0
    else:
        div_r, div_c = n - 1 - panel_h, n - 1 - panel_w
        grid[div_r, div_c:] = DIV
        grid[div_r:, div_c] = DIV
        blocked[div_r + 1 :, div_c + 1 :] = 9
        blocked[div_r, div_c:] = 9
        blocked[div_r:, div_c] = 9
        panel_r0, panel_c0 = div_r + 1, div_c + 1

    if arrangement == "horizontal":
        _place_horizontal_rows(grid, legends, panel_r0, panel_c0, panel_w)
    else:
        _place_vertical_cols(grid, legends, panel_r0, panel_c0, panel_h)
    return [(value, color) for value, _, color in legends]


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choices(["horizontal", "vertical", "l"], weights=[3, 3, 1])[0]
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


# ---------------- sprite helpers ----------------

def _tiny_flat_chamber() -> np.ndarray:
    return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)


def _large_flat_chamber() -> np.ndarray:
    return np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)


def _sprite_signature(target_effective: int, require_micro: bool) -> tuple[int, ...]:
    max_micro = max(0, 5 - target_effective)
    if target_effective == 0:
        min_micro = 1
    else:
        min_micro = 1 if require_micro else 0
    micro = random.randint(min_micro, min(2, max_micro))
    flags = [1] * target_effective + [0] * micro
    random.shuffle(flags)
    return tuple(flags)


def _sprite_core(signature: tuple[int, ...], arrangement: str) -> np.ndarray:
    if not signature:
        mask = np.ones((3, 4), dtype=bool)
    else:
        chambers = [_large_flat_chamber() if flag else _tiny_flat_chamber() for flag in signature]
        width = sum(ch.shape[1] for ch in chambers) - (len(chambers) - 1)
        mask = np.zeros((3, width), dtype=bool)
        c0 = 0
        for chamber in chambers:
            w = chamber.shape[1]
            mask[:, c0 : c0 + w] |= chamber
            c0 += w - 1
    if arrangement == "vertical":
        mask = mask.T
    if random.random() < 0.5:
        mask = mask[::-1, :]
    if random.random() < 0.5:
        mask = mask[:, ::-1]
    return mask


def _add_protrusions(mask: np.ndarray) -> np.ndarray:
    """Add a few exterior nubs without altering internal holes."""
    out = np.pad(mask, 1)
    for _ in range(random.randint(0, 3)):
        ext = _exterior_background(out)
        candidates = []
        h, w = out.shape
        for r in range(h):
            for c in range(w):
                if not ext[r, c]:
                    continue
                if any(0 <= rr < h and 0 <= cc < w and out[rr, cc] for rr, cc in neighbors4(r, c)):
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = random.choice(candidates)
        out[r, c] = True
    return _trim_mask(out)


def _make_sprite(target_effective: int, require_micro: bool = False) -> np.ndarray | None:
    for _ in range(20):
        signature = _sprite_signature(target_effective, require_micro)
        arrangement = random.choice(["horizontal", "vertical"])
        mask = _sprite_core(signature, arrangement)
        mask = _add_protrusions(mask)
        if _weighted_hole_count(mask) != target_effective:
            continue
        if require_micro and _raw_hole_count(mask) == target_effective:
            continue
        h, w = mask.shape
        if h > SPRITE_MAX_H or w > SPRITE_MAX_W:
            continue
        return np.where(mask, GREY, BG).astype(int)
    return None


# ---------------- sprite planning & placement ----------------

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None, bool]]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d
        for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, value in enumerate(extras) if value < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None, bool]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color, False))

    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in SPRITE_EFFECTIVE_VALUES if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None, False))

    idx = random.randrange(len(plan))
    value, color, _ = plan[idx]
    plan[idx] = (value, color, True)
    random.shuffle(plan)
    return plan


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
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


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    items = []
    micro_seen = False
    for hole_count, color, require_micro in _sprite_plan(legends):
        sprite = _make_sprite(hole_count, require_micro=require_micro)
        if sprite is None:
            return False
        micro_seen |= _raw_hole_count(sprite == GREY) > _weighted_hole_count(sprite == GREY)
        items.append((sprite.size, sprite, color))
    if not micro_seen:
        return False
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


# ---------------- top-level generation ----------------

def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None

    legend_colors = {color for _, color in legends}
    legend_weighted = []
    legend_raw = []
    for color in legend_colors:
        pts = components_mask(grid == color, connectivity=4)
        if len(pts) != 1:
            return None
        r0, c0, r1, c1 = bbox_coords(pts[0])
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        legend_weighted.append(_weighted_hole_count(sub))
        legend_raw.append(_raw_hole_count(sub))
    if not any(raw > weighted for raw, weighted in zip(legend_raw, legend_weighted)):
        return None

    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
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
