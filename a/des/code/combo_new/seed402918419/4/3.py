from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

ARM_PALETTE = (BLUE, RED, GREEN, YELLOW)
SEED_PALETTE = (6, 7, 8, 9)
QUADRANTS = ("tl", "tr", "bl", "br")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _random_shape(max_h: int, max_w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_cells = min(7, max_h * max_w)
    if max_cells < 2:
        return None

    for _ in range(80):
        target = rng.randint(2, max_cells)
        cells = {(0, 0)}

        while len(cells) < target:
            frontier: list[set[tuple[int, int]]] = []
            for r, c in list(cells):
                dirs = list(DIRS4)
                rng.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    grown = _normalize(cells | {(nr, nc)})
                    h, w = _shape_size(grown)
                    if h <= max_h and w <= max_w:
                        frontier.append(grown)
            if not frontier:
                break
            cells = rng.choice(frontier)

        if len(cells) != target:
            continue

        if rng.random() < 0.5:
            cells = {(c, r) for r, c in cells}
            cells = _normalize(cells)
            h, w = _shape_size(cells)
            if h > max_h or w > max_w:
                continue

        return sorted(cells)

    return None


def _quadrant_of(r: int, c: int, center_r: int, center_c: int) -> str:
    if r < center_r and c < center_c:
        return "tl"
    if r < center_r and c > center_c:
        return "tr"
    if r > center_r and c < center_c:
        return "bl"
    if r > center_r and c > center_c:
        return "br"
    raise AssertionError("seed pixels may not lie on the cross")


def _arm_colors_for_quadrant(quadrant: str, arms: dict[str, int]) -> tuple[int, int]:
    horizontal = arms["left"] if quadrant.endswith("l") else arms["right"]
    vertical = arms["up"] if quadrant.startswith("t") else arms["down"]
    return horizontal, vertical


def _find_cross(grid: np.ndarray) -> tuple[int, int, dict[str, int]]:
    gray_pts = np.argwhere(grid == GRAY)
    assert gray_pts.shape[0] == 1, "expected a unique gray center"
    center_r, center_c = map(int, gray_pts[0])
    h, w = grid.shape
    assert 0 < center_r < h - 1 and 0 < center_c < w - 1

    up = grid[:center_r, center_c]
    down = grid[center_r + 1 :, center_c]
    left = grid[center_r, :center_c]
    right = grid[center_r, center_c + 1 :]

    for arm in (up, down, left, right):
        assert arm.size > 0
        assert np.all(arm == arm[0]), "each arm must have a single color"
        assert int(arm[0]) in ARM_PALETTE

    arms = {
        "up": int(up[0]),
        "down": int(down[0]),
        "left": int(left[0]),
        "right": int(right[0]),
    }
    assert set(arms.values()) == set(ARM_PALETTE), "arms must use the four palette colors exactly once"
    return center_r, center_c, arms


def _same_color_components(grid: np.ndarray, center_r: int, center_c: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if r == center_r or c == center_c or visited[r, c]:
                continue
            color = int(grid[r, c])
            if color == BG:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if nr == center_r or nc == center_c or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            components.append((color, coords))

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each seed block into the other quadrants and recolor the copies from the shared cross arm.

    Input structure:
    - The background is black.
    - A one-pixel-wide cross sits at the exact grid center, with red, blue, green, and yellow assigned to its four arms and a gray center pixel.
    - One or two quadrants contain a small monochrome connected seed block that stays at least one cell away from the cross.

    Rule:
    - Find every seed block outside the central cross and determine its source quadrant.
    - Create three mirrored copies of that block: one across the horizontal axis, one across the vertical axis, and one by point reflection through the center.
    - Recolor the horizontal-axis copy with the source quadrant's horizontal arm color, the vertical-axis copy with the source quadrant's vertical arm color, and the diagonal copy with the same horizontal arm color.
    - Keep the original cross and the original seed blocks unchanged.

    Key insight:
    - Geometry comes from quadrant symmetry around the cross center, but copy colors come from which cross arm the source quadrant shares with the mirrored target, not from the seed color itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center_r, center_c, arms = _find_cross(g)
    seeds = _same_color_components(g, center_r, center_c)
    assert 1 <= len(seeds) <= 2, "expected one or two seed components"

    out = g.copy()
    h, w = g.shape

    for color, coords in seeds:
        assert color not in (BG, GRAY)
        quadrants = {_quadrant_of(r, c, center_r, center_c) for r, c in coords}
        assert len(quadrants) == 1, "each seed must stay within one quadrant"
        quadrant = next(iter(quadrants))
        horizontal_arm, vertical_arm = _arm_colors_for_quadrant(quadrant, arms)

        copies = (
            ([(2 * center_r - r, c) for r, c in coords], horizontal_arm),
            ([(r, 2 * center_c - c) for r, c in coords], vertical_arm),
            ([(2 * center_r - r, 2 * center_c - c) for r, c in coords], horizontal_arm),
        )

        for transformed, paint_color in copies:
            for rr, cc in transformed:
                assert 0 <= rr < h and 0 <= cc < w
                assert rr != center_r and cc != center_c, "copies may not touch the cross"
                existing = int(out[rr, cc])
                assert existing in (BG, paint_color), "mirrored copies must not conflict"
                out[rr, cc] = paint_color

    return out


def _usable_bounds(quadrant: str, center_r: int, center_c: int, h: int, w: int) -> tuple[int, int, int, int]:
    if quadrant.startswith("t"):
        row_lo, row_hi = 0, center_r - 2
    else:
        row_lo, row_hi = center_r + 2, h - 1

    if quadrant.endswith("l"):
        col_lo, col_hi = 0, center_c - 2
    else:
        col_lo, col_hi = center_c + 2, w - 1

    return row_lo, row_hi, col_lo, col_hi


def _orbit(coords: list[tuple[int, int]], center_r: int, center_c: int) -> dict[str, set[tuple[int, int]]]:
    source = {(int(r), int(c)) for r, c in coords}
    return {
        "source": source,
        "flip_h": {(2 * center_r - r, c) for r, c in source},
        "flip_v": {(r, 2 * center_c - c) for r, c in source},
        "point": {(2 * center_r - r, 2 * center_c - c) for r, c in source},
    }


def _has_neighbor_conflict(points: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in points:
        if (r, c) in occupied:
            return True
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied:
                return True
    return False


def _expected_output(
    grid: np.ndarray,
    seeds: list[dict[str, object]],
    arms: dict[str, int],
    center_r: int,
    center_c: int,
) -> np.ndarray:
    out = grid.copy()
    for seed in seeds:
        quadrant = str(seed["quadrant"])
        orbit = seed["orbit"]
        assert isinstance(orbit, dict)
        horizontal_arm, vertical_arm = _arm_colors_for_quadrant(quadrant, arms)
        for r, c in orbit["flip_h"]:
            out[r, c] = horizontal_arm
        for r, c in orbit["flip_v"]:
            out[r, c] = vertical_arm
        for r, c in orbit["point"]:
            out[r, c] = horizontal_arm
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(13, 30, 2)
    w = rng.randrange(13, 30, 2)
    center_r, center_c = h // 2, w // 2

    grid = np.full((h, w), BG, dtype=int)
    arm_colors = list(ARM_PALETTE)
    rng.shuffle(arm_colors)
    arms = {
        "up": arm_colors[0],
        "down": arm_colors[1],
        "left": arm_colors[2],
        "right": arm_colors[3],
    }

    grid[:center_r, center_c] = arms["up"]
    grid[center_r + 1 :, center_c] = arms["down"]
    grid[center_r, :center_c] = arms["left"]
    grid[center_r, center_c + 1 :] = arms["right"]
    grid[center_r, center_c] = GRAY

    seed_count = 1 if rng.random() < 0.5 else 2
    quadrants = rng.sample(QUADRANTS, seed_count)
    seed_colors = rng.sample(SEED_PALETTE, seed_count)

    occupied_output: set[tuple[int, int]] = set()
    seeds: list[dict[str, object]] = []

    for quadrant, color in zip(quadrants, seed_colors, strict=False):
        row_lo, row_hi, col_lo, col_hi = _usable_bounds(quadrant, center_r, center_c, h, w)
        max_h = row_hi - row_lo + 1
        max_w = col_hi - col_lo + 1
        if max_h < 1 or max_w < 1:
            return None

        placed = False
        for _ in range(120):
            shape = _random_shape(min(5, max_h), min(5, max_w), rng)
            if shape is None:
                continue
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            if shape_h > max_h or shape_w > max_w:
                continue

            origin_r = rng.randint(row_lo, row_hi - shape_h + 1)
            origin_c = rng.randint(col_lo, col_hi - shape_w + 1)
            coords = [(origin_r + r, origin_c + c) for r, c in shape]
            orbit = _orbit(coords, center_r, center_c)
            family = set().union(*orbit.values())

            if any(r == center_r or c == center_c for r, c in family):
                continue
            if _has_neighbor_conflict(family, occupied_output, h, w):
                continue

            for r, c in orbit["source"]:
                grid[r, c] = color

            occupied_output |= family
            seeds.append(
                {
                    "quadrant": quadrant,
                    "color": color,
                    "orbit": orbit,
                }
            )
            placed = True
            break

        if not placed:
            return None

    expected = _expected_output(grid, seeds, arms, center_r, center_c)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
