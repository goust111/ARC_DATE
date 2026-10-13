from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

QUADRANTS = ("NE", "SE", "SW", "NW")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _rotate_coord(r: int, c: int, center: tuple[float, float], turns: int) -> tuple[int, int]:
    rr = float(r) - center[0]
    cc = float(c) - center[1]
    turns %= 4
    for _ in range(turns):
        rr, cc = cc, -rr
    return int(round(rr + center[0])), int(round(cc + center[1]))


def _bbox_center(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _score_center(grid: np.ndarray, center: tuple[float, float]) -> tuple[int, int, int, int, float]:
    blues = np.argwhere(grid == BLUE)
    if blues.size == 0:
        return (-10**9, 0, 0, 0, 0.0)

    same = 0
    fillable = 0
    off = 0
    conflict = 0
    for r, c in blues:
        for turns in (1, 2, 3):
            rr, cc = _rotate_coord(int(r), int(c), center, turns)
            if not _in_bounds(grid.shape, rr, cc):
                off += 1
                continue
            value = int(grid[rr, cc])
            if value == BLUE:
                same += 1
            elif value in (BLACK, ORANGE):
                fillable += 1
            else:
                conflict += 1

    bbox_r, bbox_c = _bbox_center(grid == BLUE)
    dist = (center[0] - bbox_r) ** 2 + (center[1] - bbox_c) ** 2
    score = 8 * same + 2 * fillable - 30 * off - 60 * conflict
    return score, same, -off, -conflict, -dist


def _all_candidate_centers(grid: np.ndarray):
    bbox_r, bbox_c = _bbox_center(grid == BLUE)
    rows = sorted({bbox_r - 0.5, bbox_r, bbox_r + 0.5})
    cols = sorted({bbox_c - 0.5, bbox_c, bbox_c + 0.5})
    for r in rows:
        for c in cols:
            yield float(r), float(c)


def _best_rotation_center(grid: np.ndarray) -> tuple[tuple[float, float], tuple[int, int, int, int, float], tuple[int, int, int, int, float]]:
    best_center = None
    best_key = None
    second_key = (-10**9, -10**9, -10**9, -10**9, -10**9)
    for center in _all_candidate_centers(grid):
        key = _score_center(grid, center)
        if best_key is None or key > best_key:
            second_key = best_key if best_key is not None else second_key
            best_key = key
            best_center = center
        elif key > second_key:
            second_key = key
    assert best_center is not None and best_key is not None
    return best_center, best_key, second_key


def _detect_rotation_center(grid: np.ndarray) -> tuple[float, float]:
    center, _best, _second = _best_rotation_center(grid)
    return center


def _complete_shape(grid: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in np.argwhere(grid == BLUE):
        for turns in (1, 2, 3):
            rr, cc = _rotate_coord(int(r), int(c), center, turns)
            if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BLACK:
                out[rr, cc] = RED
    return out


def _flood_exterior(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    exterior = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing rotational quadrant, then flood-fill only the newly enclosed marked cavities and add an exterior outline.

    Input structure:
    - The background is black.
    - Blue pixels form an almost fourfold rotationally symmetric figure with one quadrant-like section missing.
    - Orange marker pixels sit in black space that is border-reachable before the completion.

    Rule:
    - Detect the figure's rotational center from the blue pixels.
    - Rotate every blue pixel by 90, 180, and 270 degrees clockwise; any rotated position that is still black becomes red, while existing blue stays blue.
    - Using the completed blue+red figure as walls, find every 4-connected black/orange region unreachable from the grid border.
    - If such a closed region contains at least one orange marker, recolor only its black cells yellow and keep the orange marker cells unchanged.
    - Finally, recolor every exterior black cell that 8-neighbors the completed figure to green; closed regions are never outlined.

    Key insight:
    - The orange markers are meaningful only after rotational completion because the missing quadrant is what prevents those regions from being enclosed in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.any(g == BLUE), "expected blue pixels"

    center = _detect_rotation_center(g)
    out = _complete_shape(g, center)
    shape = (out == BLUE) | (out == RED)
    passable = (out == BLACK) | (out == ORANGE)
    exterior = _flood_exterior(passable)
    closed = passable & ~exterior

    for comp in _components(closed):
        values = out[comp[:, 0], comp[:, 1]]
        if np.any(values == ORANGE):
            black_mask = values == BLACK
            out[comp[black_mask, 0], comp[black_mask, 1]] = YELLOW

    for r, c in np.argwhere((out == BLACK) & exterior):
        if any(_in_bounds(out.shape, nr, nc) and shape[nr, nc] for nr, nc in _neighbors8(int(r), int(c))):
            out[int(r), int(c)] = GREEN

    return out


def _quadrant_member(r: int, c: int, center: tuple[int, int], quadrant: str) -> bool:
    dr = r - center[0]
    dc = c - center[1]
    if quadrant == "NE":
        return dr < 0 and dc > 0
    if quadrant == "SE":
        return dr > 0 and dc > 0
    if quadrant == "SW":
        return dr > 0 and dc < 0
    return dr < 0 and dc < 0


def _square_ring(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for d in range(-radius, radius + 1):
        cells.add((cr - radius, cc + d))
        cells.add((cr + radius, cc + d))
        cells.add((cr + d, cc - radius))
        cells.add((cr + d, cc + radius))
    return cells


def _axis_tabs(center: tuple[int, int], radius: int, length: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for step in range(1, length + 1):
        cells.add((cr - radius - step, cc))
        cells.add((cr + radius + step, cc))
        cells.add((cr, cc - radius - step))
        cells.add((cr, cc + radius + step))
    return cells


def _shape_to_mask(shape_cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in shape_cells:
        mask[r, c] = True
    return mask


def _choose_radii() -> list[int]:
    for _ in range(200):
        rings = random.randint(2, 4)
        inner = random.randint(2, 3)
        radii = [inner]
        for _idx in range(rings - 1):
            radii.append(radii[-1] + random.randint(2, 3))
        if radii[-1] <= 9:
            return radii
    raise RuntimeError("failed to sample ring radii")


def _build_full_shape() -> tuple[np.ndarray, tuple[int, int], list[int]]:
    radii = _choose_radii()
    outer = radii[-1]
    outer_tabs = random.randint(0, 2)
    local_size = 2 * (outer + outer_tabs) + 1
    center = (local_size // 2, local_size // 2)
    shape_cells: set[tuple[int, int]] = set()
    for radius in radii:
        shape_cells |= _square_ring(center, radius)
    if outer_tabs:
        shape_cells |= _axis_tabs(center, outer, outer_tabs)
    return _shape_to_mask(shape_cells, (local_size, local_size)), center, radii


def _place_patch(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h0, w0 = mask.shape
    max_extra_h = 30 - h0
    max_extra_w = 30 - w0
    top = random.randint(0, min(4, max_extra_h))
    bottom = random.randint(0, min(4, max_extra_h - top))
    left = random.randint(0, min(4, max_extra_w))
    right = random.randint(0, min(4, max_extra_w - left))
    grid = np.zeros((h0 + top + bottom, w0 + left + right), dtype=bool)
    grid[top : top + h0, left : left + w0] = mask
    return grid, (top, left)


def _expected_output(
    input_grid: np.ndarray,
    full_shape: np.ndarray,
    missing_shape: np.ndarray,
    marked_cavities: set[int],
    cavity_components: list[np.ndarray],
    marker_positions: set[tuple[int, int]],
) -> np.ndarray:
    out = np.array(input_grid, copy=True)
    out[missing_shape] = RED

    for idx, comp in enumerate(cavity_components):
        if idx not in marked_cavities:
            continue
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            if (rr, cc) not in marker_positions:
                out[rr, cc] = YELLOW

    exterior = _flood_exterior(~full_shape)
    for r, c in np.argwhere((out == BLACK) & exterior):
        if any(_in_bounds(out.shape, nr, nc) and full_shape[nr, nc] for nr, nc in _neighbors8(int(r), int(c))):
            out[int(r), int(c)] = GREEN
    return out


def _semantic_checks(
    input_grid: np.ndarray,
    expected: np.ndarray,
    full_shape: np.ndarray,
    blue_shape: np.ndarray,
    marked_cavities: set[int],
    cavity_components: list[np.ndarray],
    marker_positions: set[tuple[int, int]],
    center: tuple[int, int],
) -> bool:
    if len(cavity_components) < 2 or not marked_cavities or len(marked_cavities) >= len(cavity_components):
        return False
    if not np.any(expected == RED) or not np.any(expected == GREEN) or not np.any(expected == YELLOW):
        return False

    passable_before = (input_grid == BLACK) | (input_grid == ORANGE)
    exterior_before = _flood_exterior(passable_before)
    if not all(exterior_before[r, c] for r, c in marker_positions):
        return False

    exterior_after = _flood_exterior(~full_shape)
    if any(exterior_after[r, c] for r, c in marker_positions):
        return False

    for idx, comp in enumerate(cavity_components):
        values = expected[comp[:, 0], comp[:, 1]]
        if idx in marked_cavities:
            if not np.any(values == ORANGE):
                return False
            if np.any(values == BLACK):
                return False
        else:
            if np.any(values == YELLOW):
                return False

    detected, best_key, second_key = _best_rotation_center(input_grid)
    if detected != (float(center[0]), float(center[1])):
        return False
    if best_key[0] - second_key[0] < 8:
        return False

    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        local_shape, local_center, _radii = _build_full_shape()
        full_shape, offset = _place_patch(local_shape)
        center = (local_center[0] + offset[0], local_center[1] + offset[1])
        quadrant = random.choice(QUADRANTS)

        blue_shape = full_shape.copy()
        missing_shape = np.zeros_like(full_shape, dtype=bool)
        for r, c in np.argwhere(full_shape):
            rr = int(r)
            cc = int(c)
            if _quadrant_member(rr, cc, center, quadrant):
                blue_shape[rr, cc] = False
                missing_shape[rr, cc] = True

        if not np.any(missing_shape):
            continue

        cavity_components = _components(~full_shape & ~_flood_exterior(~full_shape))
        if len(cavity_components) < 2:
            continue

        max_marked = min(3, len(cavity_components) - 1)
        marked_count = random.randint(1, max_marked)
        marked_cavities = set(random.sample(range(len(cavity_components)), marked_count))

        marker_positions: set[tuple[int, int]] = set()
        for idx in marked_cavities:
            comp = cavity_components[idx]
            coords = [tuple(map(int, rc)) for rc in comp]
            marker_positions.add(random.choice(coords))

        input_grid = np.zeros(full_shape.shape, dtype=int)
        input_grid[blue_shape] = BLUE
        for r, c in marker_positions:
            input_grid[r, c] = ORANGE

        expected = _expected_output(
            input_grid=input_grid,
            full_shape=full_shape,
            missing_shape=missing_shape,
            marked_cavities=marked_cavities,
            cavity_components=cavity_components,
            marker_positions=marker_positions,
        )

        if _semantic_checks(
            input_grid=input_grid,
            expected=expected,
            full_shape=full_shape,
            blue_shape=blue_shape,
            marked_cavities=marked_cavities,
            cavity_components=cavity_components,
            marker_positions=marker_positions,
            center=center,
        ):
            return input_grid

    raise RuntimeError("generate_input: failed to construct a valid sample")
