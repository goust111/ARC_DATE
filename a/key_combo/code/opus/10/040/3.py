from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GENERATION_RETRIES = 500
MARKER_RETRIES = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_info(grid: np.ndarray) -> tuple[np.ndarray, int]:
    red_components = _connected_components_of_color(grid, RED)
    assert len(red_components) == 1, "expected exactly one red marker object"
    coords = red_components[0]
    assert 4 <= len(coords) <= 8, "marker size must stay within the intended range"

    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert w >= 3 and w % 2 == 1, "marker must have an integer vertical symmetry axis"

    mask = np.zeros((h, w), dtype=bool)
    shifted = coords - np.array([[r0, c0]])
    mask[shifted[:, 0], shifted[:, 1]] = True
    assert np.array_equal(mask, mask[:, ::-1]), "red marker must be vertically mirror-symmetric"

    axis_col = c0 + w // 2
    assert 0 < axis_col < grid.shape[1] - 1, "borrowed axis must lie strictly inside the grid"
    return coords, axis_col


def _invert_binary_region(region: np.ndarray) -> np.ndarray:
    vals = set(map(int, np.unique(region)))
    assert vals.issubset({BLACK, BLUE}), "invertible region must contain only blue foreground on black background"
    return np.where(region == BLUE, BLACK, BLUE)


def _apply_rule_from_axis(grid: np.ndarray, axis_col: int, *, erase_marker: bool = True, mode: str = "right") -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if erase_marker:
        out[out == RED] = BLACK

    if mode == "right":
        if axis_col + 1 < out.shape[1]:
            out[:, axis_col + 1 :] = _invert_binary_region(out[:, axis_col + 1 :])
    elif mode == "right_inclusive":
        out[:, axis_col:] = _invert_binary_region(out[:, axis_col:])
    elif mode == "left":
        out[:, :axis_col] = _invert_binary_region(out[:, :axis_col])
    elif mode == "all":
        out[:, :] = _invert_binary_region(out[:, :])
    else:
        raise ValueError(f"unknown inversion mode: {mode}")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Borrow the red marker's vertical symmetry axis, erase the marker, and invert only the blue/black pattern on the axis's right side.

    Input structure:
    - The grid is a black background with scattered blue pixels and one small red connected marker object.
    - The red marker is vertically mirror-symmetric and therefore determines a unique interior grid column that acts as its symmetry axis.
    - Blue pixels may appear on both sides of that borrowed axis, while red and blue never overlap.

    Rule:
    - Find the single red connected object and recover its vertical mirror axis column C.
    - Erase every red marker cell to black.
    - For columns with index greater than C, invert blue and black independently in each row: blue becomes black and black becomes blue.
    - Leave columns with index less than or equal to C unchanged after the marker has been erased.

    Key insight:
    - The symmetric red object is not copied or reflected; it only donates a geometric divider that limits where the row-wise inversion is allowed to act.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "this task uses only black, blue, and red"

    _coords, axis_col = _marker_info(g)
    return _apply_rule_from_axis(g, axis_col, erase_marker=True, mode="right")


def _orbit_adjacent(mask: np.ndarray, orbit: list[tuple[int, int]]) -> bool:
    for r, c in orbit:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                return True
    return False


def _sample_marker_mask() -> np.ndarray | None:
    for _ in range(MARKER_RETRIES):
        h = random.randint(2, 4)
        half = random.randint(1, 2)
        w = 2 * half + 1
        target = random.randint(4, 8)

        mask = np.zeros((h, w), dtype=bool)
        axis = half
        seed = (random.randrange(h), axis)
        mask[seed] = True
        pixels = 1

        while pixels < target:
            candidates: list[list[tuple[int, int]]] = []

            for r in range(h):
                if not mask[r, axis]:
                    orbit = [(r, axis)]
                    if pixels + 1 <= target and _orbit_adjacent(mask, orbit):
                        candidates.append(orbit)

                for left_c in range(half):
                    right_c = w - 1 - left_c
                    if mask[r, left_c] or mask[r, right_c]:
                        continue
                    orbit = [(r, left_c), (r, right_c)]
                    if pixels + 2 <= target and _orbit_adjacent(mask, orbit):
                        candidates.append(orbit)

            if not candidates:
                break

            orbit = random.choice(candidates)
            for r, c in orbit:
                mask[r, c] = True
            pixels += len(orbit)

        if pixels != target or not np.any(mask[:, :axis]):
            continue

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        tight = mask[r0 : r1 + 1, c0 : c1 + 1]
        if tight.shape[1] < 3 or tight.shape[1] % 2 != 1:
            continue
        if not np.array_equal(tight, tight[:, ::-1]):
            continue
        if len(_connected_components_of_color(tight.astype(int), 1)) != 1:
            continue
        return tight

    return None


def _place_marker(grid: np.ndarray, marker_mask: np.ndarray, axis_col: int) -> np.ndarray | None:
    h, w = marker_mask.shape
    local_axis = w // 2
    c0 = axis_col - local_axis
    if c0 < 0 or c0 + w > grid.shape[1]:
        return None

    r0 = random.randint(0, grid.shape[0] - h)
    out = np.array(grid, copy=True)
    sub = out[r0 : r0 + h, c0 : c0 + w]
    sub[marker_mask] = RED
    return out


def _paint_blue_rows(grid: np.ndarray, axis_col: int, want_axis_blue: bool) -> np.ndarray | None:
    out = np.array(grid, copy=True)
    axis_blue_rows: list[int] = []
    for r in range(out.shape[0]):
        left_avail = [c for c in range(axis_col) if int(out[r, c]) == BLACK]
        right_avail = [c for c in range(axis_col + 1, out.shape[1]) if int(out[r, c]) == BLACK]
        if len(left_avail) < 2 or len(right_avail) < 2:
            return None

        left_n = random.randint(1, min(4, len(left_avail) - 1))
        right_n = random.randint(1, min(5, len(right_avail) - 1))
        for c in random.sample(left_avail, left_n):
            out[r, c] = BLUE
        for c in random.sample(right_avail, right_n):
            out[r, c] = BLUE

        if int(out[r, axis_col]) == BLACK and random.random() < 0.45:
            axis_blue_rows.append(r)

    if want_axis_blue:
        if not axis_blue_rows:
            candidates = [r for r in range(out.shape[0]) if int(out[r, axis_col]) == BLACK]
            if not candidates:
                return None
            axis_blue_rows = [random.choice(candidates)]
        for r in axis_blue_rows:
            out[r, axis_col] = BLUE

    return out


def _rows_have_both_sides(grid: np.ndarray, axis_col: int) -> bool:
    for r in range(grid.shape[0]):
        if not np.any(grid[r, :axis_col] == BLUE):
            return False
        if not np.any(grid[r, axis_col + 1 :] == BLUE):
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        marker_coords, axis_col = _marker_info(grid)
        actual = main(grid)
    except Exception:
        return False

    if not _rows_have_both_sides(grid, axis_col):
        return False

    expected = _apply_rule_from_axis(grid, axis_col, erase_marker=True, mode="right")
    if not np.array_equal(actual, expected):
        return False

    wrong_versions = [
        _apply_rule_from_axis(grid, axis_col, erase_marker=True, mode="all"),
        _apply_rule_from_axis(grid, axis_col, erase_marker=True, mode="left"),
        _apply_rule_from_axis(grid, axis_col, erase_marker=True, mode="right_inclusive"),
    ]

    no_erase = expected.copy()
    no_erase[marker_coords[:, 0], marker_coords[:, 1]] = RED
    wrong_versions.append(no_erase)

    if axis_col - 1 >= 1:
        wrong_versions.append(_apply_rule_from_axis(grid, axis_col - 1, erase_marker=True, mode="right"))
    if axis_col + 1 <= grid.shape[1] - 2:
        wrong_versions.append(_apply_rule_from_axis(grid, axis_col + 1, erase_marker=True, mode="right"))

    if any(np.array_equal(expected, wrong) for wrong in wrong_versions):
        return False

    if np.any(actual == RED):
        return False
    if not np.array_equal(actual[:, : axis_col + 1], expected[:, : axis_col + 1]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(14, 18)
    axis_col = random.randint(4, w - 5)

    marker_mask = _sample_marker_mask()
    if marker_mask is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    placed = _place_marker(grid, marker_mask, axis_col)
    if placed is None:
        return None

    want_axis_blue = random.random() < 0.5
    painted = _paint_blue_rows(placed, axis_col, want_axis_blue)
    if painted is None:
        return None

    return painted if _semantic_checks(painted) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
