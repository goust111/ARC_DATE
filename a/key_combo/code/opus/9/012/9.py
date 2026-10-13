from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LEFT_PATTERN = RED
RIGHT_PATTERN = YELLOW

AREA_RANGE = (5, 12)
HEIGHT_RANGE = (9, 14)
PANEL_WIDTH_RANGE = (7, 12)
MASK_HEIGHT_RANGE = (3, 6)
MASK_WIDTH_RANGE = (3, 6)
GEN_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _reflect_vertical(coords: np.ndarray) -> np.ndarray:
    r0, _c0, r1, _c1 = _bbox_coords(coords)
    reflected = coords.copy()
    reflected[:, 0] = r0 + r1 - reflected[:, 0]
    return reflected


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    return counts.most_common(1)[0][0]


def _analyze_panel(panel: np.ndarray) -> dict:
    colors = [int(c) for c in np.unique(panel) if int(c) != GREY]
    assert len(colors) == 2, "each panel must contain exactly one base color and one pattern color"

    counts = {color: int(np.count_nonzero(panel == color)) for color in colors}
    base_color = max(colors, key=lambda color: counts[color])
    pattern_color = min(colors, key=lambda color: counts[color])
    assert counts[base_color] > counts[pattern_color], "base color must be the majority"

    mask = panel == pattern_color
    components = _components_mask(mask)
    assert len(components) == 1, "pattern mask must be 4-connected"

    coords = components[0]
    return {
        "base_color": int(base_color),
        "pattern_color": int(pattern_color),
        "coords": coords,
    }


def _separator_col(grid: np.ndarray) -> int:
    full = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(full) == 1, "expected exactly one grey separator column"
    return int(full[0])


def _render_panel(height: int, width: int, base_color: int, pattern_color: int, coords: np.ndarray) -> np.ndarray:
    panel = np.full((height, width), int(base_color), dtype=int)
    panel[coords[:, 0], coords[:, 1]] = int(pattern_color)
    return panel


def _swap_only(grid: np.ndarray) -> np.ndarray:
    sep = _separator_col(grid)
    left = _analyze_panel(grid[:, :sep])
    right = _analyze_panel(grid[:, sep + 1 :])
    h, w = grid.shape[0], sep

    out = np.empty_like(grid)
    out[:, sep] = GREY
    out[:, :sep] = _render_panel(h, w, left["base_color"], right["base_color"], right["coords"])
    out[:, sep + 1 :] = _render_panel(h, w, right["base_color"], left["base_color"], left["coords"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two panel masks, then mirror each swapped mask vertically inside its own bounding box.

    Input structure:
    - A single grey full-height separator column splits the grid into two equal rectangular panels.
    - The left panel uses blue as its majority base color and red as its minority pattern color.
    - The right panel uses green as its majority base color and yellow as its minority pattern color.
    - Each panel's minority color forms one 4-connected, non-rectangular mask on top of its base.

    Rule:
    - Find the grey separator column and analyze each side independently.
    - In each panel, identify the majority non-grey color as the base and the minority non-grey color as the pattern mask.
    - Swap the two pattern masks by position: redraw the left panel on its original base, but with the right mask recolored to the right panel's base color; redraw the right panel symmetrically with the left mask recolored to the left panel's base color.
    - After swapping, reflect each swapped mask vertically inside that mask's own bounding box, keeping its columns fixed and keeping the grey separator unchanged.

    Key insight:
    - The mirror is applied to the exchanged panel-local mask, not to the original input mask and not to the whole panel color field.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_col(g)
    assert sep > 0 and sep < g.shape[1] - 1
    assert sep == g.shape[1] - sep - 1, "the two panels must have equal width"

    left_panel = g[:, :sep]
    right_panel = g[:, sep + 1 :]
    left = _analyze_panel(left_panel)
    right = _analyze_panel(right_panel)

    h, w = left_panel.shape
    out = np.empty_like(g)
    out[:, sep] = GREY

    left_swapped = _reflect_vertical(right["coords"])
    right_swapped = _reflect_vertical(left["coords"])
    out[:, :sep] = _render_panel(h, w, left["base_color"], right["base_color"], left_swapped)
    out[:, sep + 1 :] = _render_panel(h, w, right["base_color"], left["base_color"], right_swapped)
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    coords = coords - coords.min(axis=0, keepdims=True)
    return tuple(map(tuple, coords.tolist()))


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray | None:
    area = random.randint(*AREA_RANGE)
    for _ in range(80):
        canvas = np.zeros((max_h, max_w), dtype=bool)
        seed = (random.randrange(max_h), random.randrange(max_w))
        cells = {seed}
        frontier = {seed}
        while len(cells) < area and frontier:
            cr, cc = random.choice(tuple(frontier))
            neighbors = [
                (cr - 1, cc),
                (cr + 1, cc),
                (cr, cc - 1),
                (cr, cc + 1),
            ]
            options = [
                (nr, nc)
                for nr, nc in neighbors
                if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells
            ]
            if not options:
                frontier.remove((cr, cc))
                continue
            cell = random.choice(options)
            cells.add(cell)
            frontier.add(cell)

        if len(cells) != area:
            continue

        for r, c in cells:
            canvas[r, c] = True
        mask = _crop_mask(canvas)
        mh, mw = mask.shape
        if not (MASK_HEIGHT_RANGE[0] <= mh <= min(MASK_HEIGHT_RANGE[1], max_h)):
            continue
        if not (MASK_WIDTH_RANGE[0] <= mw <= min(MASK_WIDTH_RANGE[1], max_w)):
            continue
        if int(mask.sum()) != area:
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if int(mask.sum()) == mh * mw:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _place_mask(panel_h: int, panel_w: int, mask: np.ndarray) -> np.ndarray:
    mh, mw = mask.shape
    max_r = panel_h - mh
    max_c = panel_w - mw
    r0 = random.randint(0, max_r)
    c0 = random.randint(0, max_c)
    coords = np.argwhere(mask)
    coords[:, 0] += r0
    coords[:, 1] += c0
    return coords


def _render_input(height: int, panel_w: int, left_coords: np.ndarray, right_coords: np.ndarray) -> np.ndarray:
    grid = np.empty((height, panel_w * 2 + 1), dtype=int)
    grid[:, panel_w] = GREY
    grid[:, :panel_w] = BLUE
    grid[:, panel_w + 1 :] = GREEN
    grid[left_coords[:, 0], left_coords[:, 1]] = RED
    grid[right_coords[:, 0], panel_w + 1 + right_coords[:, 1]] = YELLOW
    return grid


def _mirror_only(grid: np.ndarray) -> np.ndarray:
    sep = _separator_col(grid)
    left = _analyze_panel(grid[:, :sep])
    right = _analyze_panel(grid[:, sep + 1 :])
    h, w = grid.shape[0], sep

    out = np.empty_like(grid)
    out[:, sep] = GREY
    out[:, :sep] = _render_panel(h, w, left["base_color"], left["pattern_color"], _reflect_vertical(left["coords"]))
    out[:, sep + 1 :] = _render_panel(h, w, right["base_color"], right["pattern_color"], _reflect_vertical(right["coords"]))
    return out


def _validate_case(grid: np.ndarray) -> bool:
    sep = _separator_col(grid)
    left = _analyze_panel(grid[:, :sep])
    right = _analyze_panel(grid[:, sep + 1 :])

    if left["base_color"] != BLUE or left["pattern_color"] != RED:
        return False
    if right["base_color"] != GREEN or right["pattern_color"] != YELLOW:
        return False

    left_mask = np.zeros((grid.shape[0], sep), dtype=bool)
    right_mask = np.zeros((grid.shape[0], sep), dtype=bool)
    left_mask[left["coords"][:, 0], left["coords"][:, 1]] = True
    right_mask[right["coords"][:, 0], right["coords"][:, 1]] = True

    if _shape_signature(left_mask) == _shape_signature(right_mask):
        return False
    if _shape_signature(left_mask) == _shape_signature(np.flipud(right_mask)):
        return False

    out = main(grid)
    if np.array_equal(out, _swap_only(grid)):
        return False
    if np.array_equal(out, _mirror_only(grid)):
        return False

    left_out = _analyze_panel(out[:, :sep])
    right_out = _analyze_panel(out[:, sep + 1 :])
    expected_left = _reflect_vertical(right["coords"])
    expected_right = _reflect_vertical(left["coords"])

    if left_out["base_color"] != BLUE or right_out["base_color"] != GREEN:
        return False
    if left_out["pattern_color"] != GREEN or right_out["pattern_color"] != BLUE:
        return False

    if set(map(tuple, left_out["coords"].tolist())) != set(map(tuple, expected_left.tolist())):
        return False
    if set(map(tuple, right_out["coords"].tolist())) != set(map(tuple, expected_right.tolist())):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        panel_w = random.randint(*PANEL_WIDTH_RANGE)

        left_mask = _random_connected_mask(min(MASK_HEIGHT_RANGE[1], height), min(MASK_WIDTH_RANGE[1], panel_w))
        right_mask = _random_connected_mask(min(MASK_HEIGHT_RANGE[1], height), min(MASK_WIDTH_RANGE[1], panel_w))
        if left_mask is None or right_mask is None:
            continue

        left_coords = _place_mask(height, panel_w, left_mask)
        right_coords = _place_mask(height, panel_w, right_mask)
        grid = _render_input(height, panel_w, left_coords, right_coords)

        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
