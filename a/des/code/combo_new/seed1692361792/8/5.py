from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
MAROON = 9
OUTPUT_COLORS = (1, 2, 3, 4, 6, 7)
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
GENERATION_RETRIES = 500
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Component:
    color: int
    coords: np.ndarray
    min_r: int
    min_c: int
    max_r: int
    max_c: int


def _detect_scan_axis(grid: np.ndarray) -> int:
    maroon_mask = grid == MAROON
    row_hits = np.flatnonzero(np.all(maroon_mask, axis=1))
    col_hits = np.flatnonzero(np.all(maroon_mask, axis=0))

    if len(row_hits) == 1 and len(col_hits) == 0 and int(np.count_nonzero(maroon_mask)) == grid.shape[1]:
        return 0
    if len(row_hits) == 0 and len(col_hits) == 1 and int(np.count_nonzero(maroon_mask)) == grid.shape[0]:
        return 1
    raise AssertionError("expected exactly one full maroon row or column")


def _extract_components(grid: np.ndarray) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BG, MAROON):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            components.append(
                Component(
                    color=color,
                    coords=arr,
                    min_r=int(mins[0]),
                    min_c=int(mins[1]),
                    max_r=int(maxs[0]),
                    max_c=int(maxs[1]),
                )
            )

    return components


def _scan_key(component: Component, primary_axis: int) -> tuple[int, int, int, int, int]:
    if primary_axis == 0:
        primary_min = component.min_r
        secondary_min = component.min_c
        primary_span = component.max_r - component.min_r
        secondary_span = component.max_c - component.min_c
    else:
        primary_min = component.min_c
        secondary_min = component.min_r
        primary_span = component.max_c - component.min_c
        secondary_span = component.max_r - component.min_r

    return (
        int(primary_min),
        int(secondary_min),
        int(primary_span),
        int(secondary_span),
        int(component.coords.shape[0]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor disconnected blocks by the order in which a scan perpendicular to the maroon guide line first encounters them.

    Input structure:
    - The background is black.
    - Several separated 4-connected monochrome blocks occupy the canvas in arbitrary colors.
    - A single full maroon row or full maroon column spans the grid and does not overlap any block.

    Rule:
    - A horizontal maroon line means a top-to-bottom scan with a horizontal virtual line, while a vertical maroon line means a left-to-right scan with a vertical virtual line.
    - Order the colored blocks by the first row or column where that scan would touch each block, then remove the maroon guide line.
    - Repaint the ordered blocks with the fixed sequence blue, red, green, yellow, magenta, orange, preserving every block's shape and position.

    Key insight:
    - The maroon line contributes no output pixels of its own; it only selects the scan axis, so the solution is to sort blocks by their earliest visible coordinate on that axis.
    """
    g = np.asarray(grid, dtype=int)
    primary_axis = _detect_scan_axis(g)
    components = _extract_components(g)

    out = np.full_like(g, BG)
    if not components:
        return out

    assert len(components) <= len(OUTPUT_COLORS), "too many blocks for the fixed recolor palette"
    ordered = sorted(components, key=lambda comp: _scan_key(comp, primary_axis))

    for color, component in zip(OUTPUT_COLORS, ordered):
        out[component.coords[:, 0], component.coords[:, 1]] = color

    return out


def _random_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        box_h = int(rng.integers(2, 5))
        box_w = int(rng.integers(2, 6))
        max_cells = min(box_h * box_w, 8)
        if max_cells < 3:
            continue
        target = int(rng.integers(3, max_cells + 1))

        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

        if rng.random() < 0.5:
            mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.size) and int(mask.size) > 4 and rng.random() < 0.75:
            continue
        return mask

    raise RuntimeError("failed to sample a usable component mask")


def _line_coordinate(primary_len: int, primary_starts: list[int], primary_sizes: list[int], rng: np.random.Generator) -> int | None:
    used = np.zeros(primary_len, dtype=bool)
    for start, size in zip(primary_starts, primary_sizes):
        used[start : start + size] = True

    occupied = np.flatnonzero(used)
    if occupied.size == 0:
        return None

    lo = int(occupied[0])
    hi = int(occupied[-1])
    free_internal = [idx for idx in range(1, primary_len - 1) if not used[idx] and lo < idx < hi]
    if not free_internal:
        return None

    roomy = [idx for idx in free_internal if not used[idx - 1] and not used[idx + 1]]
    candidates = roomy or free_internal
    return int(candidates[int(rng.integers(len(candidates)))])


def _expected_output(shape: tuple[int, int], ordered_coords: list[np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color, coords in zip(OUTPUT_COLORS, ordered_coords):
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_blocks = int(rng.integers(3, 7))
    horizontal_line = bool(rng.integers(2) == 0)
    colors = [int(c) for c in rng.choice(np.asarray(INPUT_COLORS, dtype=int), size=n_blocks, replace=False)]
    masks = [_random_mask(rng) for _ in range(n_blocks)]

    primary_sizes = [int(mask.shape[0] if horizontal_line else mask.shape[1]) for mask in masks]
    secondary_sizes = [int(mask.shape[1] if horizontal_line else mask.shape[0]) for mask in masks]

    lead_gap = int(rng.integers(1, 3))
    tail_gap = int(rng.integers(1, 3))
    between_gaps = [int(rng.integers(1, 3)) for _ in range(n_blocks - 1)]

    primary_starts: list[int] = []
    cursor = lead_gap
    for idx, size in enumerate(primary_sizes):
        primary_starts.append(cursor)
        cursor += size
        if idx < n_blocks - 1:
            cursor += between_gaps[idx]
    primary_len = cursor + tail_gap
    if primary_len > 30:
        return None

    min_secondary = max(max(secondary_sizes) + 2, 8)
    if min_secondary > 30:
        return None
    max_secondary = min(30, min_secondary + 10)
    secondary_len = int(rng.integers(min_secondary, max_secondary + 1))

    line_primary = _line_coordinate(primary_len, primary_starts, primary_sizes, rng)
    if line_primary is None:
        return None

    if horizontal_line:
        h, w = primary_len, secondary_len
    else:
        h, w = secondary_len, primary_len

    grid = np.full((h, w), BG, dtype=int)
    ordered_coords: list[np.ndarray] = []
    scan_values: list[int] = []

    for color, mask, primary_start, secondary_span in zip(colors, masks, primary_starts, secondary_sizes):
        slack = secondary_len - secondary_span
        if slack < 0:
            return None
        secondary_start = int(rng.integers(slack + 1))
        cells = np.argwhere(mask)
        if horizontal_line:
            cells = cells + np.array([primary_start, secondary_start], dtype=int)
        else:
            cells = cells + np.array([secondary_start, primary_start], dtype=int)

        if np.any(grid[cells[:, 0], cells[:, 1]] != BG):
            return None
        grid[cells[:, 0], cells[:, 1]] = color
        ordered_coords.append(cells)
        scan_values.append(int(primary_start))

    if len(set(scan_values)) != n_blocks:
        return None

    if horizontal_line:
        if np.any(grid[line_primary, :] != BG):
            return None
        grid[line_primary, :] = MAROON
    else:
        if np.any(grid[:, line_primary] != BG):
            return None
        grid[:, line_primary] = MAROON

    components = _extract_components(grid)
    if len(components) != n_blocks:
        return None

    expected = _expected_output(grid.shape, ordered_coords)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid scan-line recoloring instance")
