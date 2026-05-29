# Combo variant of e3721c99 (index 20)
# combo_spec: D1/1+D5/1
# Name: 面积编码图例+面积匹配

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task: legend topology encoded a hole count, and each grey sprite was recolored by its own hole count.
#    Fused task: every legend is now a solid 3-cell-thick bar. Its area is 3*v, so dividing by the fixed factor 3
#    yields the encoded value v. Grey sprites are no longer grouped by holes at all; their filled-cell area must equal v.
#    Example: a 3x4 legend bar has area 12, so it encodes value 4. Any grey component with exactly 4 cells gets that
#    legend color even if it has no hole and even if another 4-cell shape looks different.
# b) Complexity calibration:
#    Values are restricted to 3..7. This is large enough to avoid trivial 1-2 cell sprites, small enough to be counted
#    by eye, and supports 2-4 distinct legends with repeated evidence through multiple matching sprites.
# c) Implicit-assumption audit:
#    The original solver assumed legend components differ by hole count and grey sprites may vary widely in area.
#    That assumption is removed. The new solver instead assumes:
#    - each legend component has area divisible by LEGEND_FACTOR,
#    - each legend value is unique,
#    - grey sprites are 4-connected components whose area is the matching key.
#    The generator enforces all three.
# d) Placement legality:
#    The divider/layout machinery is preserved from the source task, so legends still live in a reserved edge/corner panel.
#    Grey sprites are placed only in the workspace, separated by at least one background cell via 8-neighborhood padding.
# e) Visual observability:
#    The input/output pairs visibly show two kinds of area reasoning: long legend bars of different sizes and scattered
#    grey polyominoes with corresponding cell counts. Because all legend bars are hole-free, the old hole-count story no
#    longer explains the recoloring.
# f) Visual cleanliness:
#    Legends remain aligned in a strip with one-cell gaps; sprites come from a small catalog of compact connected masks,
#    which avoids noisy organic blobs and keeps counting feasible.
# g) Inferability of color change:
#    Each legend color appears on at least one output sprite, often more than once, so the color mapping is supported by
#    repeated examples. Distractors use non-legend areas and are erased, making the area thresholding behavior explicit.


LEGEND_FACTOR = 3


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Map encoded value -> legend color using legend area / LEGEND_FACTOR."""
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
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        area = int(len(candidates[0]))
        assert area % LEGEND_FACTOR == 0, "legend area must be divisible by LEGEND_FACTOR"
        value = area // LEGEND_FACTOR
        assert value not in mapping, f"legend value {value} must map to exactly one color"
        mapping[value] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching their pixel area to values encoded by legend areas.

    Input structure:
    - bg is the most common color, grey (5) sprites are scattered in the workspace, and colored legend bars sit in a
      separated edge or corner panel behind a divider.
    - Each legend is a solid 3-cell-thick component of one color; its area is a multiple of 3, and dividing by 3 gives
      the numeric value associated with that color.

    Rule:
    - Read each legend as value = legend_area // 3.
    - For every grey sprite, count its filled cells.
    - Recolor a grey sprite to the legend color whose encoded value equals that sprite area; erase grey sprites whose
      area is not represented in the legend.

    Key insight:
    - Both sides use area, but the roles differ: legend area is an encoded number after dividing by a fixed factor,
      whereas sprite area is the quantity matched directly to that number.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        area = int(len(pts))
        newc = int(legend[area]) if area in legend else int(bg)
        for r, c in pts:
            out[r, c] = newc
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
LEGEND_VALUES = [3, 4, 5, 6, 7]
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)

_SPRITE_LIBRARY: dict[int, list[np.ndarray]] = {
    3: [
        np.array([[1, 1], [1, 0]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
    5: [
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
    ],
    7: [
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
}


def _legend_span(value: int) -> int:
    """Legend length along the arrangement axis."""
    return int(value)


def _legend_patch(value: int, color: int, arrangement: str) -> np.ndarray:
    """Build a solid 3xV or Vx3 legend bar."""
    if arrangement == "horizontal":
        return np.full((LEGEND_FACTOR, int(value)), int(color), dtype=int)
    return np.full((int(value), LEGEND_FACTOR), int(color), dtype=int)


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(v) for v, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Pick 2-4 legends with distinct encoded values that fit within max_span."""
    del arrangement
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        vals for vals in itertools.combinations(LEGEND_VALUES, count)
        if sum(vals) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends("horizontal", max_span + 1)
    values = list(random.choice(combos))
    random.shuffle(values)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(values, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for value, color in legends:
        patch = _legend_patch(value, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
    n = GRID_SIZE
    cross_dim = LEGEND_FACTOR if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else LEGEND_FACTOR

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


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
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

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_sprite(value: int) -> np.ndarray:
    """Choose a compact connected sprite with exact area=value, then randomize orientation."""
    mask = random.choice(_SPRITE_LIBRARY[value]).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    mask = _trim_mask(mask)
    return np.where(mask, GREY, BG).astype(int)


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Build a shuffled list of (area_value, color_or_None) for all sprites."""
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

    plan: list[tuple[int, int | None]] = []
    for i, (value, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((value, color))

    legend_values = {v for v, _ in legends}
    distractor_values = [v for v in LEGEND_VALUES if v not in legend_values]
    assert distractor_values, "need at least one distractor value outside the legend set"
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_values), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement with a 1-cell visual buffer around sprites."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    candidates = [(r, c) for r in range(1, GRID_SIZE - h) for c in range(1, GRID_SIZE - w)]
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
    items = []
    for value, color in _sprite_plan(legends):
        sprite = _make_sprite(value)
        items.append((int(np.sum(sprite == GREY)), sprite, color))
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
    raise RuntimeError("failed to generate combo variant input")
