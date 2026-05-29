# Combo variant of e3721c99 (index 31)
# combo_spec: D1/1+D3/2+D5/1
# Name: 面积匹配+面积编码图例+纯色块图例
#
# Design analysis
# a) Concrete form and behavior trace
#    - Original task:
#      colored legend shapes carried topology directly. A solver counted holes on
#      each legend shape, then counted holes on every grey sprite, and recolored
#      equal-hole sprites to the legend color.
#    - Fused task here:
#      the legend becomes a dedicated strip separated by a full divider row. Each
#      legend entry is just one solid colored rectangle. Its pixel area is not the
#      target value itself: `legend_area = LEGEND_SCALE * match_value`.
#      Grey sprites are also no longer judged by holes. Their own pixel area is the
#      match value. Example: a 3x5 legend block has area 15, so with scale 3 it
#      encodes value 5; every grey sprite with exactly 5 grey pixels becomes that
#      block's color.
#    - The behavioral change is therefore bilateral:
#      legend side: holes -> solid-block area code
#      sprite side: holes -> sprite pixel area
#      The same abstract number is now hidden behind two different area readings.
# b) Complexity calibration
#    - Match values are limited to 2..6. That gives 5 categories: enough room for
#      2-4 legend entries plus distractor areas, but still small enough that the
#      solid legend rectangles fit cleanly in one strip.
#    - Legend blocks use a fixed scale factor 3 and appear in either 3xN or Nx3
#      orientation. This keeps the arithmetic simple while preventing width alone
#      from being a perfect cue in every sample.
#    - Grey sprites are small connected polyominoes drawn from a per-area library.
#      For areas >= 3 the library contains multiple shapes, so equal-area sprites
#      can look different and shape identity is visibly not the rule.
# c) Implicit assumption audit for `main()`
#    - The old legend parser assumed every legend color has a unique non-border
#      component whose hole count is meaningful. That is false now: all legend
#      entries are hole-free solid rectangles, so hole-count parsing would collapse.
#    - The new parser assumes:
#        * there is exactly one full divider row in color DIV,
#        * all legend entries lie on exactly one side of that divider,
#        * each legend color appears as one solid rectangular component,
#        * each such rectangle has area divisible by LEGEND_SCALE.
#      The generator enforces those invariants.
#    - Sprite processing now assumes only that grey components are 4-connected and
#      should be classified by `len(component)` rather than by topology. The
#      generator uses hole-free sprites, but `main()` does not rely on hole-freeness.
# d) Placement legality
#    - The divider row fences the legend strip from the workspace. The generator
#      blocks the entire legend side plus the divider before placing sprites.
#    - Sprite placement uses 8-neighborhood dilation of the occupied mask, so grey
#      sprites never overlap and keep at least one background cell of separation.
# e) Visual observability
#    - Across examples a human sees that the top or bottom strip contains only solid
#      color bars, never holey reference shapes. Meanwhile the grey objects that
#      change color are small irregular blobs whose outlines vary even when they
#      share the same recoloring result. That makes area the only stable common cue.
# f) Visual cleanliness
#    - The legend is confined to a single strip with one-cell gaps between entries.
#      The workspace stays open, and sprites are small and separated, so legend and
#      workspace evidence do not visually tangle.
# g) Color-change inferability
#    - Output colors still come from visible colored legend blocks, so the mapping is
#      grounded. The nontrivial step is inferring the hidden number behind each block
#      from its area and then finding grey sprites with the same pixel count.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, paste


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

# -- fused area logic --
LEGEND_SCALE = 3
MATCH_VALUES = (2, 3, 4, 5, 6)

# -- legend strip --
LEGEND_COUNT_RANGE = (2, 4)
PANEL_HEIGHT = 8
ENTRY_GAP = 1

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
SPRITE_MAX_H = 5
SPRITE_MAX_W = 5

# -- generation --
GEN_RETRIES = 300
_STRUCT8 = generate_binary_structure(2, 2)

_SPRITE_LIBRARY = {
    2: (
        ("11",),
    ),
    3: (
        ("111",),
        ("11", "10"),
    ),
    4: (
        ("111", "100"),
        ("111", "010"),
        ("110", "011"),
    ),
    5: (
        ("111", "010", "010"),
        ("11", "11", "10"),
        ("111", "101"),
        ("111", "001", "001"),
    ),
    6: (
        ("111", "101", "100"),
        ("111", "010", "011"),
        ("110", "011", "001"),
        ("111", "101", "001"),
    ),
}


def _pattern_to_mask(pattern: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in pattern], dtype=bool)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return _crop_mask(out)


def _legend_patch(value: int, color: int) -> np.ndarray:
    """Solid rectangle with area = LEGEND_SCALE * value."""
    if random.random() < 0.5:
        h, w = LEGEND_SCALE, int(value)
    else:
        h, w = int(value), LEGEND_SCALE
    return np.full((h, w), int(color), dtype=int)


def _find_divider_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == DIV)]
    assert len(rows) == 1, "grid must contain exactly one full divider row"
    return int(rows[0])


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Parse the legend strip separated by a full divider row.

    Each legend entry is a single solid colored rectangle whose area divided by
    LEGEND_SCALE yields the match value.
    """
    div_r = _find_divider_row(g)
    nontrivial = (g != bg) & (g != grey) & (g != DIV)
    above_has = bool(np.any(nontrivial[:div_r, :]))
    below_has = bool(np.any(nontrivial[div_r + 1 :, :]))
    assert above_has ^ below_has, "legend must live on exactly one side of the divider"

    if above_has:
        panel = g[:div_r, :]
    else:
        panel = g[div_r + 1 :, :]

    mapping: dict[int, int] = {}
    colors = sorted({int(c) for c in np.unique(panel) if int(c) not in (bg, grey, DIV)})
    for color in colors:
        comps = components_mask(panel == int(color), connectivity=4)
        assert len(comps) == 1, f"color {color} must have exactly one legend component"
        pts = comps[0]
        r0, c0, r1, c1 = bbox_coords(pts)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(len(pts))
        assert area == h * w, f"legend component for color {color} must be a solid rectangle"
        assert area % LEGEND_SCALE == 0, f"legend area for color {color} must be divisible by scale"
        value = area // LEGEND_SCALE
        assert value in MATCH_VALUES, f"legend value {value} outside allowed range"
        assert value not in mapping, f"legend value {value} must map to exactly one color"
        mapping[int(value)] = int(color)
    assert 2 <= len(mapping) <= 4, "legend must contain 2-4 entries"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by decoding solid legend blocks as area-coded values.

    Input structure:
    - Background is the dominant color; grey (5) connected sprites sit in the workspace.
    - A single full divider row isolates a legend strip on the top or bottom side.
      Inside that strip, each legend entry is one solid colored rectangle with no holes.

    Rule:
    - For each legend rectangle, compute `match_value = rectangle_area / LEGEND_SCALE`.
    - For every grey sprite, compute its pixel area by counting grey cells in its
      connected component.
    - Recolor sprites whose area matches a legend value; erase grey distractors whose
      area is absent from the legend.

    Key insight:
    - Both halves of the puzzle now speak in area rather than topology. The legend
      hides a number inside scaled solid-rectangle area, and the grey sprites must be
      grouped by their own cell counts instead of by shape or holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    legend = _legend_mapping(g, bg, GREY)
    out = g.copy()
    for pts in components_mask(g == GREY, connectivity=4):
        area = int(len(pts))
        newc = int(legend[area]) if area in legend else int(bg)
        for r, c in pts:
            out[r, c] = newc
    return out


def _choose_legends() -> list[tuple[int, int]]:
    count = random.randint(*LEGEND_COUNT_RANGE)
    values = list(random.choice(list(itertools.combinations(MATCH_VALUES, count))))
    random.shuffle(values)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(values, colors))


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Place a top or bottom legend strip separated by a full divider row."""
    legends = _choose_legends()
    patches = [(value, color, _legend_patch(value, color)) for value, color in legends]
    total_w = sum(p.shape[1] for _, _, p in patches) + ENTRY_GAP * (len(patches) - 1)
    assert total_w <= GRID_SIZE - 2, "legend strip must fit with side margins"

    top = random.choice([True, False])
    if top:
        panel_r0 = 0
        div_r = PANEL_HEIGHT
        blocked[: div_r + 1, :] = 9
    else:
        div_r = GRID_SIZE - PANEL_HEIGHT - 1
        panel_r0 = div_r + 1
        blocked[div_r:, :] = 9
    grid[div_r, :] = DIV

    c = random.randint(1, GRID_SIZE - total_w - 1)
    for _, _, patch in patches:
        r = random.randint(panel_r0 + 1, panel_r0 + PANEL_HEIGHT - patch.shape[0] - 1)
        paste(grid, patch, r, c)
        c += patch.shape[1] + ENTRY_GAP
    return legends


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (area_value, output_color_or_None)."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors

    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, count in enumerate(extras) if count < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None]] = []
    for i, (value, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((int(value), int(color)))

    legend_values = {value for value, _ in legends}
    distractor_values = [value for value in MATCH_VALUES if value not in legend_values]
    for _ in range(n_distractors):
        plan.append((int(random.choice(distractor_values)), None))
    random.shuffle(plan)
    return plan


def _make_sprite(area_value: int, seen: set[tuple[tuple[int, ...], ...]]) -> np.ndarray:
    """Pick a small connected grey polyomino with the requested area."""
    candidates = []
    for pattern in _SPRITE_LIBRARY[int(area_value)]:
        base = _pattern_to_mask(pattern)
        for _ in range(8):
            mask = _transform_mask(base)
            if mask.shape[0] > SPRITE_MAX_H or mask.shape[1] > SPRITE_MAX_W:
                continue
            sig = _mask_signature(mask)
            candidates.append((sig, mask))
    unique: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for sig, mask in candidates:
        unique[sig] = mask
    fresh = [(sig, mask) for sig, mask in unique.items() if sig not in seen]
    pool = fresh if fresh else list(unique.items())
    sig, mask = random.choice(pool)
    seen.add(sig)
    return np.where(mask, GREY, BG).astype(int)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate and place all sprites, then color matched ones in `expected`."""
    seen_shapes_by_area = {value: set() for value in MATCH_VALUES}
    items = []
    for area_value, color in _sprite_plan(legends):
        sprite = _make_sprite(area_value, seen_shapes_by_area[int(area_value)])
        items.append((sprite.size, sprite, color))
    items.sort(key=lambda x: x[0], reverse=True)

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


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
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
