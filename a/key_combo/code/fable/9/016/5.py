from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
MARKER = 9
PANEL_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_PANEL_SIZE_RANGE = (4, 10)  # upper bound exclusive; height becomes 3*n+2 <= 29
GEN_MAX_ATTEMPTS = 500
GEN_MASK_TRIES = 80


def _corner_rotation(marker_pos: tuple[int, int], n: int) -> int:
    corners = {
        (0, 0): 0,
        (0, n - 1): 1,
        (n - 1, n - 1): 2,
        (n - 1, 0): 3,
    }
    assert marker_pos in corners, "marker must occupy a corner"
    return corners[marker_pos]


def _normalize_panel(panel: np.ndarray) -> tuple[np.ndarray, int]:
    n, m = panel.shape
    assert n == m, "panel must be square"

    marker_coords = np.argwhere(panel == MARKER)
    assert marker_coords.shape == (1, 2), "each panel must contain exactly one marker pixel"
    marker_pos = tuple(map(int, marker_coords[0]))

    colors = {int(c) for c in np.unique(panel)} - {BLACK}
    assert MARKER in colors and len(colors) == 2, "panel must contain one foreground color plus the marker"
    fg_color = next(c for c in colors if c != MARKER)

    restored = np.rot90(panel, _corner_rotation(marker_pos, n)).copy()
    assert int(restored[0, 0]) == MARKER, "marker must indicate the true top-left corner"
    restored[0, 0] = BLACK

    allowed = {BLACK, fg_color}
    assert {int(c) for c in np.unique(restored)} <= allowed, "normalized panel must become binary"
    return restored, fg_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate three marked panels upright, then keep only cells owned by exactly one panel.

    Input structure:
    - The grid is split by two full-width divider rows into three equal square panels.
    - Each panel has black background, one panel-specific foreground color, and one
      color-9 corner marker showing that panel's true top-left corner.
    - Each entire panel may have been rotated by 0, 90, 180, or 270 degrees.

    Rule:
    - For each panel, rotate it until the color-9 marker returns to the top-left corner,
      then delete that marker.
    - Compare the three corrected panels cell by cell.
    - Output a single panel-sized grid where a cell keeps its panel color only when
      exactly one corrected panel has foreground there; counts 0, 2, or 3 become black.

    Key insight:
    - The corner marker fixes orientation first; only after that do the three masks share
      one coordinate system, and the overlay rule is parity-like "exactly one foreground".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    n = int(g.shape[1])
    assert n > 1 and g.shape[0] == 3 * n + 2, "grid must be three n-by-n panels separated by two rows"

    divider_rows = (n, 2 * n + 1)
    for r in divider_rows:
        row = g[r]
        assert np.all(row == row[0]), "divider row must be full-width and uniform"
        assert int(row[0]) != BLACK, "divider row must be non-black"

    panels = [g[:n], g[n + 1 : 2 * n + 1], g[2 * n + 2 :]]
    masks: list[np.ndarray] = []
    colors: list[int] = []
    for panel in panels:
        restored, fg_color = _normalize_panel(panel)
        masks.append(restored == fg_color)
        colors.append(int(fg_color))

    counts = sum(mask.astype(int) for mask in masks)
    exactly_one = counts == 1
    out = np.zeros((n, n), dtype=int)
    for mask, color in zip(masks, colors):
        out[exactly_one & mask] = int(color)
    return out


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    n_components = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            n_components += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return n_components


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _has_full_foreground_line(mask: np.ndarray) -> bool:
    return bool(np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)))


def _grow_mask(n: int, target_area: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    cells = [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]

    seed_count = random.randint(1, min(3, target_area))
    for r, c in random.sample(cells, seed_count):
        mask[r, c] = True

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        occupied = np.argwhere(mask)
        for r, c in occupied:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < n and 0 <= nc < n and not mask[nr, nc] and (nr, nc) != (0, 0):
                    frontier.add((nr, nc))

        empty = [(r, c) for r in range(n) for c in range(n) if not mask[r, c] and (r, c) != (0, 0)]
        if not empty:
            break

        if frontier and (random.random() < 0.88 or len(frontier) == len(empty)):
            r, c = random.choice(tuple(frontier))
        else:
            r, c = random.choice(empty)
        mask[r, c] = True

    return mask


def _make_mask(n: int) -> np.ndarray | None:
    area = n * n
    min_area = max(4, area // 5)
    max_area = min(area - 4, (area * 2) // 5 + 1)
    target_area = random.randint(min_area, max_area)

    for _ in range(GEN_MASK_TRIES):
        mask = _grow_mask(n, target_area)
        if mask[0, 0]:
            continue
        if _has_full_foreground_line(mask):
            continue
        if not _all_rotations_distinct(mask):
            continue
        components = _component_count(mask)
        if not (1 <= components <= 3):
            continue
        return mask
    return None


def _render_panel(mask: np.ndarray, color: int, rotation: int) -> np.ndarray:
    panel = np.zeros(mask.shape, dtype=int)
    panel[mask] = int(color)
    panel[0, 0] = MARKER
    return np.rot90(panel, int(rotation)).astype(int)


def _expected_from_masks(masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    counts = sum(mask.astype(int) for mask in masks)
    out = np.zeros(masks[0].shape, dtype=int)
    exactly_one = counts == 1
    for mask, color in zip(masks, colors):
        out[exactly_one & mask] = int(color)
    return out


def _naive_without_reorientation(input_grid: np.ndarray) -> np.ndarray:
    g = np.asarray(input_grid, dtype=int)
    n = int(g.shape[1])
    panels = [g[:n].copy(), g[n + 1 : 2 * n + 1].copy(), g[2 * n + 2 :].copy()]

    masks: list[np.ndarray] = []
    colors: list[int] = []
    for panel in panels:
        colors_here = {int(c) for c in np.unique(panel)} - {BLACK}
        assert MARKER in colors_here and len(colors_here) == 2
        fg_color = next(c for c in colors_here if c != MARKER)
        panel[panel == MARKER] = BLACK
        masks.append(panel == fg_color)
        colors.append(fg_color)
    return _expected_from_masks(masks, colors)


def _semantic_checks(masks: list[np.ndarray], input_grid: np.ndarray, expected: np.ndarray) -> bool:
    counts = sum(mask.astype(int) for mask in masks)
    exactly_one = counts == 1
    overlapped = counts >= 2

    if not np.any(exactly_one):
        return False
    if not np.any(overlapped):
        return False

    for mask in masks:
        if not np.any(mask & exactly_one):
            return False
        if not np.any(mask & overlapped):
            return False

    if np.array_equal(_naive_without_reorientation(input_grid), expected):
        return False

    if np.array_equal(main(input_grid), expected):
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n = random.randrange(*GEN_PANEL_SIZE_RANGE)
    masks: list[np.ndarray] = []
    for _ in range(3):
        mask = _make_mask(n)
        if mask is None:
            return None
        masks.append(mask)

    if any(np.array_equal(masks[i], masks[j]) for i in range(3) for j in range(i + 1, 3)):
        return None

    rotations = [random.randrange(4) for _ in range(3)]
    if all(rot == 0 for rot in rotations):
        return None

    colors = random.sample(PANEL_COLOR_POOL, 3)
    remaining = [c for c in PANEL_COLOR_POOL if c not in colors]
    if random.random() < 0.5:
        divider_color = random.choice(remaining)
        divider_colors = (divider_color, divider_color)
    else:
        divider_colors = tuple(random.sample(remaining, 2))

    rendered = [
        _render_panel(mask, color, rotation)
        for mask, color, rotation in zip(masks, colors, rotations)
    ]

    grid = np.zeros((3 * n + 2, n), dtype=int)
    grid[:n] = rendered[0]
    grid[n] = divider_colors[0]
    grid[n + 1 : 2 * n + 1] = rendered[1]
    grid[2 * n + 1] = divider_colors[1]
    grid[2 * n + 2 :] = rendered[2]

    expected = _expected_from_masks(masks, colors)
    if not _semantic_checks(masks, grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
