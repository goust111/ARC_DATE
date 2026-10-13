from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
MARKER = 5

MAX_GENERATION_ATTEMPTS = 600
_GEN_CALL_COUNT = 0


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_marker_row(grid: np.ndarray) -> int:
    marker_rows = np.flatnonzero(np.all(grid == MARKER, axis=1))
    assert len(marker_rows) == 1, "expected exactly one full-width marker row"
    return int(marker_rows[0])


def _reflect_red_pixels(grid: np.ndarray, marker_row: int) -> list[tuple[int, int, int]]:
    red_coords = np.argwhere(grid == RED)
    assert 3 <= len(red_coords) <= 8, "expected a 3-8 pixel red motif"

    reflected: list[tuple[int, int, int]] = []
    for r, c in red_coords:
        rr = 2 * marker_row - int(r)
        assert 0 <= rr < grid.shape[0], "reflected pixel must stay in bounds"
        reflected.append((int(rr), int(c), GREEN))

    items = [(int(r), int(c), RED) for r, c in red_coords]
    items.extend(reflected)
    return items


def _gravity_drop(items: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c, color in items:
        assert 0 <= r < h and 0 <= c < w
        by_row[int(r)].append((int(c), int(color)))

    out = np.full((h, w), BG, dtype=int)
    for row in sorted(by_row.keys(), reverse=True):
        placements: list[tuple[int, int, int]] = []
        for c, color in by_row[row]:
            nr = row
            while nr + 1 < h and out[nr + 1, c] == BG:
                nr += 1
            placements.append((nr, c, color))
        for nr, c, color in placements:
            assert out[nr, c] == BG, "same-row pixels must not overlap after falling"
            out[nr, c] = color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker_row = _find_marker_row(g)
    items = _reflect_red_pixels(g, marker_row)
    return _gravity_drop(items, g.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the red motif across the grey marker row, erase the row, then let both copies fall together under gravity.

    Input structure:
    - The grid is mostly black background with one full-width grey marker row.
    - A single 4-connected red motif sits above that row, separated from it by one or two blank rows.
    - The area below the marker row is otherwise empty, leaving room for the reflected copy and the later fall.

    Rule:
    - Reflect every red pixel vertically across the grey row and paint the reflected pixels green.
    - Remove the grey marker row from the scene.
    - Apply downward gravity to all red and green pixels together, processing source rows from bottom to top.
    - Pixels originating in the same row fall simultaneously and do not block one another, but any already settled red or green pixel stops later pixels.

    Key insight:
    - The reflection creates a second real object, and the final arrangement is determined by the collision interaction between the original red pixels and the new green mirror during the gravity simulation.
    """
    return _solve(grid)


def _rigid_drop(coords: np.ndarray, settled: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, _w = shape
    shift = 0
    while True:
        candidate = coords + np.array([shift + 1, 0], dtype=int)
        if int(candidate[:, 0].max()) >= h:
            break
        if np.any(settled[candidate[:, 0], candidate[:, 1]]):
            break
        shift += 1
    return coords + np.array([shift, 0], dtype=int)


def _wrong_rigid_component_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_row = _find_marker_row(g)
    red_coords = np.argwhere(g == RED).astype(int)
    green_coords = np.array([[2 * marker_row - int(r), int(c)] for r, c in red_coords], dtype=int)
    out = np.full(g.shape, BG, dtype=int)

    settled = np.zeros(g.shape, dtype=bool)
    green_final = _rigid_drop(green_coords, settled, g.shape)
    settled[green_final[:, 0], green_final[:, 1]] = True
    out[green_final[:, 0], green_final[:, 1]] = GREEN

    red_final = _rigid_drop(red_coords, settled, g.shape)
    settled[red_final[:, 0], red_final[:, 1]] = True
    out[red_final[:, 0], red_final[:, 1]] = RED
    return out


def _wrong_no_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_row = _find_marker_row(g)
    out = np.full(g.shape, BG, dtype=int)
    red_coords = np.argwhere(g == RED)
    out[red_coords[:, 0], red_coords[:, 1]] = RED
    for r, c in red_coords:
        out[2 * marker_row - int(r), int(c)] = GREEN
    return out


def _make_shape(mask_h: int, mask_w: int, area: int) -> np.ndarray | None:
    choices = [(r, c) for r in range(mask_h) for c in range(mask_w)]
    for _ in range(120):
        start = random.choice(choices)
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < mask_h and 0 <= nc < mask_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((mask_h, mask_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != mask_h - 1:
            continue
        if not any(int(np.count_nonzero(mask[r])) >= 2 for r in rows):
            continue

        column_counts = [int(np.count_nonzero(mask[:, c])) for c in cols]
        if len(set(column_counts)) == 1:
            continue

        return mask
    return None


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BG, RED, MARKER})

    marker_row = _find_marker_row(g)
    red_mask = g == RED
    comps = _connected_components(red_mask)
    assert len(comps) == 1, "expected one red component"
    coords = comps[0]

    assert 3 <= len(coords) <= 8
    assert int(coords[:, 0].min()) >= 1, "motif must not touch top border"
    assert int(coords[:, 0].max()) < marker_row - 0

    gap = marker_row - int(coords[:, 0].max()) - 1
    assert gap in (1, 2), "motif must be 1-2 rows above the marker"
    assert not np.any(g[marker_row + 1 :, :] != BG), "space below the marker must be empty in the input"

    reflected_rows = 2 * marker_row - coords[:, 0]
    assert int(reflected_rows.min()) > marker_row
    assert int(reflected_rows.max()) < g.shape[0]


def _generate_for_gap(gap: int, attempts: int) -> np.ndarray | None:
    for _ in range(attempts):
        height = random.randint(12, 18)
        width = random.randint(8, 14)
        marker_row = random.randint(4, min(6, height - 7))

        max_shape_h = min(4, marker_row - gap - 1, height - marker_row - gap - 1)
        if max_shape_h < 2:
            continue
        shape_h = random.randint(2, max_shape_h)

        max_shape_w = min(5, width - 2)
        if max_shape_w < 2:
            continue
        shape_w = random.randint(2, max_shape_w)

        max_area = min(8, shape_h * shape_w)
        area = random.randint(3, max_area)
        mask = _make_shape(shape_h, shape_w, area)
        if mask is None:
            continue

        top = marker_row - gap - shape_h
        if top < 1:
            continue
        left = random.randint(1, width - shape_w - 1)

        grid = np.full((height, width), BG, dtype=int)
        grid[marker_row, :] = MARKER
        patch = np.where(mask, RED, BG)
        grid[top : top + shape_h, left : left + shape_w] = np.maximum(
            grid[top : top + shape_h, left : left + shape_w], patch
        )

        try:
            _validate_input(grid)
        except AssertionError:
            continue

        out = _solve(grid)
        rigid = _wrong_rigid_component_gravity(grid)
        no_gravity = _wrong_no_gravity(grid)

        if np.array_equal(out, rigid):
            continue
        if np.array_equal(out, no_gravity):
            continue
        if np.count_nonzero(out == RED) != area or np.count_nonzero(out == GREEN) != area:
            continue
        if np.any(out[marker_row, :] == MARKER):
            continue

        occupied_cols = [c for c in range(width) if np.any(grid[:, c] == RED)]
        if len(occupied_cols) < 2:
            continue
        stack_heights = [int(np.count_nonzero(grid[:, c] == RED)) for c in occupied_cols]
        if len(set(stack_heights)) == 1:
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_gap = 1 if (_GEN_CALL_COUNT % 2 == 0) else 2
    _GEN_CALL_COUNT += 1

    primary = _generate_for_gap(preferred_gap, MAX_GENERATION_ATTEMPTS // 2)
    if primary is not None:
        return primary

    fallback = _generate_for_gap(3 - preferred_gap, MAX_GENERATION_ATTEMPTS // 2)
    if fallback is not None:
        return fallback

    raise RuntimeError("failed to generate a valid ARC input for puzzle 5")
