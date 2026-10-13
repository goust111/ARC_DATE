from __future__ import annotations

import math
import random

import numpy as np


BG = 0
NOISE = 5
COUNTED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_POOL = (2, 3, 4, 5)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator controls.
GRID_HEIGHT_RANGE = (17, 24)
GRID_WIDTH_RANGE = (17, 24)
MAX_NOISE_PIXELS = 6
MAX_PLACEMENT_ATTEMPTS = 120
MAX_SHAPE_ATTEMPTS = 120
MAX_FRAGMENT_SPAN = 8
MAX_GENERATION_ATTEMPTS = 300

PROFILE_CYCLE = (
    {"n_colors": 3, "has_noise": True, "prefer_split": True},
    {"n_colors": 4, "has_noise": False, "prefer_split": True},
    {"n_colors": 3, "has_noise": False, "prefer_split": False},
    {"n_colors": 4, "has_noise": True, "prefer_split": False},
)
_GEN_CALL_COUNT = 0


def _perfect_square_side(area: int) -> int:
    side = math.isqrt(area)
    assert side * side == area, f"pixel count {area} is not a perfect square"
    return side


def _ordered_color_sides(grid: np.ndarray) -> list[tuple[int, int]]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, NOISE)]
    assert 3 <= len(colors) <= 4, "expected 3 or 4 counted colors"

    ordered = []
    seen_sides = set()
    for color in colors:
        area = int(np.count_nonzero(grid == color))
        side = _perfect_square_side(area)
        assert 2 <= side <= 5, "counted square side must be in [2, 5]"
        assert side not in seen_sides, "counted colors must have distinct side lengths"
        seen_sides.add(side)
        ordered.append((color, side))

    ordered.sort(key=lambda item: item[1], reverse=True)
    return ordered


def _render_expected(color_to_side: dict[int, int]) -> np.ndarray:
    ordered = sorted(color_to_side.items(), key=lambda item: item[1], reverse=True)
    outer_color, outer_side = ordered[0]
    out = np.full((outer_side, outer_side), outer_color, dtype=int)
    for color, side in ordered[1:]:
        out[:side, :side] = color
    inner_side = ordered[-1][1]
    diag = np.arange(inner_side)
    out[diag, diag] = outer_color
    return out


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = [(r - min_r, c - min_c) for r, c in cells]
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _is_filled_square(mask: np.ndarray) -> bool:
    return mask.shape[0] == mask.shape[1] and int(mask.sum()) == mask.shape[0] * mask.shape[1]


def _random_connected_shape(area: int, avoid_filled_square: bool) -> np.ndarray | None:
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in NEIGHBORS4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        mask = _cells_to_mask(cells)
        if max(mask.shape) > MAX_FRAGMENT_SPAN:
            continue
        if avoid_filled_square and _is_filled_square(mask):
            continue
        return mask
    return None


def _partition_area(area: int, prefer_split: bool) -> list[int]:
    max_parts = min(4, area)
    if prefer_split and max_parts >= 2:
        choices = list(range(2, max_parts + 1))
        weights = [6, 4, 2][: len(choices)]
        part_count = random.choices(choices, weights=weights, k=1)[0]
    else:
        choices = list(range(1, max_parts + 1))
        weights = [5, 6, 3, 1][: len(choices)]
        part_count = random.choices(choices, weights=weights, k=1)[0]

    parts = [1] * part_count
    for _ in range(area - part_count):
        parts[random.randrange(part_count)] += 1
    parts.sort(reverse=True)
    return parts


def _place_cells(grid: np.ndarray, blocked: np.ndarray, cells: np.ndarray, color: int) -> bool:
    if blocked[cells[:, 0], cells[:, 1]].any():
        return False

    h, w = grid.shape
    grid[cells[:, 0], cells[:, 1]] = color
    for r, c in cells:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        blocked[r0:r1, c0:c1] = True
    return True


def _place_fragment(grid: np.ndarray, blocked: np.ndarray, color: int, area: int) -> bool:
    avoid_filled_square = area > 1
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        mask = _random_connected_shape(area, avoid_filled_square=avoid_filled_square)
        if mask is None:
            continue
        rel = np.argwhere(mask)
        sh, sw = mask.shape
        origins = [
            (r0, c0)
            for r0 in range(grid.shape[0] - sh + 1)
            for c0 in range(grid.shape[1] - sw + 1)
        ]
        random.shuffle(origins)
        for r0, c0 in origins:
            cells = rel + np.array([r0, c0], dtype=int)
            if _place_cells(grid, blocked, cells, color):
                return True
    return False


def _place_noise(grid: np.ndarray, blocked: np.ndarray, count: int) -> bool:
    if count == 0:
        return True

    candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if not blocked[r, c]]
    random.shuffle(candidates)
    if len(candidates) < count:
        return False

    for r, c in candidates[:count]:
        cells = np.array([[r, c]], dtype=int)
        if not _place_cells(grid, blocked, cells, NOISE):
            return False
    return True


def _build_scene(
    color_to_side: dict[int, int],
    height: int,
    width: int,
    has_noise: bool,
    prefer_split: bool,
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = list(color_to_side.keys())
    split_favored = set(random.sample(colors, k=max(1, len(colors) - 1)))
    for color, side in sorted(color_to_side.items(), key=lambda item: item[1], reverse=True):
        parts = _partition_area(side * side, prefer_split and color in split_favored)
        for part in parts:
            if not _place_fragment(grid, blocked, color, part):
                return None

    noise_count = random.randint(1, MAX_NOISE_PIXELS) if has_noise else 0
    if not _place_noise(grid, blocked, noise_count):
        return None
    return grid


def _sample_color_sides(n_colors: int) -> dict[int, int]:
    sides = random.sample(SIDE_POOL, n_colors)
    colors = random.sample(COUNTED_COLORS, n_colors)
    random.shuffle(colors)
    return {color: side for color, side in zip(colors, sides)}


def _validate_semantics(grid: np.ndarray, alt_grid: np.ndarray, color_to_side: dict[int, int]) -> bool:
    expected = _render_expected(color_to_side)
    stripped = grid.copy()
    stripped[stripped == NOISE] = BG

    try:
        if not np.array_equal(main(grid), expected):
            return False
        if not np.array_equal(main(stripped), expected):
            return False
        if not np.array_equal(main(alt_grid), expected):
            return False
    except Exception:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Recover count-defined nested upper-left squares and redraw the innermost diagonal with the outer color.

    Input structure:
    - The input is a black-background grid with 3 or 4 counted colors scattered as arbitrary pixels or fragments.
    - Grey pixels are isolated noise and are not part of any counted object.
    - Each counted color appears a distinct perfect-square number of times, giving a side length from 2 to 5.

    Rule:
    - Ignore all grey noise pixels.
    - Count each remaining color and convert its population P into a side length S = sqrt(P).
    - Sort colors by side length from largest to smallest and paint nested solid squares anchored at the output's top-left corner, overwriting smaller ones on top of larger ones.
    - Inside the smallest square only, repaint the main diagonal with the largest-square color.

    Key insight:
    - The input layout is a distractor: only per-color pixel counts matter, and the visible L-shaped layers arise from repeated overwriting at one fixed top-left anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered = _ordered_color_sides(g)
    outer_color, outer_side = ordered[0]
    out = np.full((outer_side, outer_side), outer_color, dtype=int)
    for color, side in ordered[1:]:
        out[:side, :side] = color

    inner_side = ordered[-1][1]
    diag = np.arange(inner_side)
    out[diag, diag] = outer_color
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(MAX_GENERATION_ATTEMPTS):
        profile = PROFILE_CYCLE[_GEN_CALL_COUNT % len(PROFILE_CYCLE)]
        _GEN_CALL_COUNT += 1

        color_to_side = _sample_color_sides(profile["n_colors"])
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        grid = _build_scene(
            color_to_side,
            height=height,
            width=width,
            has_noise=profile["has_noise"],
            prefer_split=profile["prefer_split"],
        )
        if grid is None:
            continue

        alt_grid = None
        for _alt in range(8):
            alt_height = random.randint(*GRID_HEIGHT_RANGE)
            alt_width = random.randint(*GRID_WIDTH_RANGE)
            alt_grid = _build_scene(
                color_to_side,
                height=alt_height,
                width=alt_width,
                has_noise=not profile["has_noise"],
                prefer_split=not profile["prefer_split"],
            )
            if alt_grid is None:
                continue
            if alt_grid.shape != grid.shape or not np.array_equal(alt_grid, grid):
                break
        if alt_grid is None:
            continue

        if _validate_semantics(grid, alt_grid, color_to_side):
            return grid

    raise RuntimeError("failed to generate a valid nested-square counting input")
