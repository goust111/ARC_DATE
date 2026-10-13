from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the repeated boundary color's rectangles one block inward while leaving singleton boundary rectangles in place.

    Input structure:
    - The background is the most common color.
    - Foreground objects are solid monochrome rectangles touching a grid corner.
    - Exactly one foreground color appears as multiple separate 4-connected objects; other colors appear once.

    Rule:
    - Clear the multiple objects of the repeated color from their original positions.
    - For each cleared rectangle, translate it away from the row and column border it touches by exactly its height and width.
    - Keep singleton-color objects unchanged; overlapping translated rectangles of the same color merge.

    Key insight:
    - Multiplicity selects the moving color, and each boundary rectangle's touched edges determine its inward diagonal translation.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        by_color[int(comp["color"])].append(comp)

    moving_colors = [color for color, comps in by_color.items() if len(comps) > 1]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])
    moving_objects = by_color[moving_color]

    out = g.copy()
    for comp in moving_objects:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for comp in moving_objects:
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert int(comp["area"]) == height * width

        touches_top = r0 == 0
        touches_bottom = r1 == H - 1
        touches_left = c0 == 0
        touches_right = c1 == W - 1
        assert touches_top != touches_bottom
        assert touches_left != touches_right

        dr = height if touches_top else -height
        dc = width if touches_left else -width
        shifted = coords + np.array([dr, dc], dtype=int)
        assert bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < H)
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < W)
            )
        )

        target = out[shifted[:, 0], shifted[:, 1]]
        assert bool(np.all((target == bg) | (target == moving_color)))
        out[shifted[:, 0], shifted[:, 1]] = moving_color

    return out

# --------------- generator (o5x) ---------------

# Corner and movement model.
CORNERS = ("TL", "TR", "BL", "BR")
DIAGONAL_PAIRS = ({"TL", "BR"}, {"TR", "BL"})
INWARD_STEP_FACTOR = 2
COMPACT_CANVAS_FACTOR = 3

# Canvas, object count, and retry bounds.
GRID_ROW_BOUNDS = (7, 22)
GRID_COL_BOUNDS = (7, 30)
SQUARE_SIZE_BOUNDS = (1, 4)
MOVING_COUNT_BOUNDS = (2, 4)
GENERATION_RETRIES = 300

# Color palette.
COLOR_VALUES = tuple(range(10))

def _corner_pos(corner: str, H: int, W: int, s: int) -> tuple[int, int]:
    row = 0 if corner[0] == "T" else H - s
    col = 0 if corner[1] == "L" else W - s
    return row, col


def _shifted_pos(corner: str, H: int, W: int, s: int) -> tuple[int, int]:
    row = s if corner[0] == "T" else H - INWARD_STEP_FACTOR * s
    col = s if corner[1] == "L" else W - INWARD_STEP_FACTOR * s
    return row, col


def _valid_sizes(s: int, moving_count: int) -> list[tuple[int, int]]:
    sizes = []
    min_h = max(GRID_ROW_BOUNDS[0], INWARD_STEP_FACTOR * s + 1)
    min_w = max(GRID_COL_BOUNDS[0], INWARD_STEP_FACTOR * s + 1)
    for H in range(min_h, GRID_ROW_BOUNDS[1] + 1):
        for W in range(min_w, GRID_COL_BOUNDS[1] + 1):
            bg_area = H * W - len(CORNERS) * s * s
            if bg_area <= moving_count * s * s:
                continue
            compact = H < COMPACT_CANVAS_FACTOR * s and W < COMPACT_CANVAS_FACTOR * s
            if moving_count == MOVING_COUNT_BOUNDS[1] - 1 and compact:
                continue
            sizes.append((H, W))
    return sizes


def _pick_moving_corners(H: int, W: int, s: int, moving_count: int) -> set[str]:
    if moving_count == len(CORNERS):
        return set(CORNERS)
    compact = H < COMPACT_CANVAS_FACTOR * s and W < COMPACT_CANVAS_FACTOR * s
    if moving_count == MOVING_COUNT_BOUNDS[0] and compact:
        return set(random.choice(DIAGONAL_PAIRS))
    return set(random.sample(CORNERS, moving_count))


def _overlaps(a: tuple[int, int], b: tuple[int, int], s: int) -> bool:
    ar, ac = a
    br, bc = b
    return ar < br + s and br < ar + s and ac < bc + s and bc < ac + s


def _sample_layout():
    """Return a collision-free layout tuple, or None when the sampled size cannot work."""
    s = random.randint(*SQUARE_SIZE_BOUNDS)
    moving_count = random.randint(*MOVING_COUNT_BOUNDS)
    sizes = _valid_sizes(s, moving_count)
    if not sizes:
        return None
    H, W = random.choice(sizes)

    moving = _pick_moving_corners(H, W, s, moving_count)
    singletons = [corner for corner in CORNERS if corner not in moving]

    shifted = {corner: _shifted_pos(corner, H, W, s) for corner in moving}
    singleton_pos = {corner: _corner_pos(corner, H, W, s) for corner in singletons}
    for moved_pos in shifted.values():
        for kept_pos in singleton_pos.values():
            if _overlaps(moved_pos, kept_pos, s):
                return None
    return H, W, s, moving, singletons, shifted, singleton_pos


def _sample_colors(singleton_count: int) -> tuple[int, int, list[int]]:
    colors = list(COLOR_VALUES)
    bg = random.choice(colors)
    colors.remove(bg)
    moving_color = random.choice(colors)
    colors.remove(moving_color)
    singleton_colors = random.sample(colors, singleton_count)
    return bg, moving_color, singleton_colors


def _render_pair(H, W, s, bg, moving_color, moving, singletons, shifted, singleton_pos, singleton_colors):
    """Render the matching input/output grids for an accepted layout and palette."""
    rect = np.full((s, s), moving_color, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    for corner in moving:
        paste(out, rect, *shifted[corner])
    for corner, color in zip(singletons, singleton_colors):
        patch = np.full((s, s), color, dtype=int)
        paste(out, patch, *singleton_pos[corner])

    inp = out.copy()
    for corner in moving:
        r, c = shifted[corner]
        inp[r : r + s, c : c + s] = bg
    for corner in moving:
        paste(inp, rect, *_corner_pos(corner, H, W, s))
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    H, W, s, moving, singletons, shifted, singleton_pos = layout
    bg, moving_color, singleton_colors = _sample_colors(len(singletons))
    inp, out = _render_pair(H, W, s, bg, moving_color, moving, singletons, shifted, singleton_pos, singleton_colors)
    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 22208ba4 input")
