from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAX_ATTEMPTS = 400


def _active_mask(shape: tuple[int, int], cell_size: int, phase: int) -> np.ndarray:
    rows = np.arange(shape[0], dtype=int)[:, None] // int(cell_size)
    cols = np.arange(shape[1], dtype=int)[None, :] // int(cell_size)
    return ((rows + cols + int(phase)) % 2) == 0


def _valid_candidates(blue_mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(blue_mask)
    assert coords.size > 0, "expected at least one blue sample pixel"

    height, width = blue_mask.shape
    max_cell_size = min(height, width) // 2
    assert max_cell_size >= 1, "grid is too small to contain a checkerboard"

    candidates: list[tuple[int, int]] = []
    for cell_size in range(1, max_cell_size + 1):
        parities = ((coords[:, 0] // cell_size) + (coords[:, 1] // cell_size)) & 1
        if np.all(parities == parities[0]):
            candidates.append((int(cell_size), int(parities[0])))
    return candidates


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[counts.argmax()])

    corners = [
        (0, 0),
        (0, grid.shape[1] - 1),
        (grid.shape[0] - 1, 0),
        (grid.shape[0] - 1, grid.shape[1] - 1),
    ]
    corner_markers = [(r, c) for r, c in corners if int(grid[r, c]) != bg]
    assert len(corner_markers) == 1, "expected exactly one visible non-background corner marker"
    marker_r, marker_c = corner_markers[0]
    red = int(grid[marker_r, marker_c])

    non_bg = [int(v) for v in values if int(v) != bg]
    assert len(non_bg) == 2, "expected exactly two non-background colors"
    blue_candidates = [color for color in non_bg if color != red]
    assert len(blue_candidates) == 1, "expected one blue sample color"
    blue = int(blue_candidates[0])
    return bg, blue, red


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a hidden checkerboard from sparse blue samples and paint the full board.

    Input structure:
    - Black background.
    - Blue sample pixels appear only inside one parity class of a hidden checkerboard.
    - A single red marker pixel sits in a corner and supplies the fill color for
      the alternating cells.

    Rule:
    - Recover the checkerboard cell size and phase from the blue pixels.
    - Fill the entire active parity class with blue.
    - Fill the alternating parity class with red.
    - The corner marker is overwritten by the checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, blue, red = _infer_colors(g)
    blue_mask = g == blue
    candidates = _valid_candidates(blue_mask)
    assert len(candidates) == 1, "checkerboard size/phase must be uniquely determined"
    cell_size, phase = candidates[0]

    out = np.full_like(g, red)
    out[_active_mask(g.shape, cell_size, phase)] = blue
    return out


def _grow_connected_pattern(cell_size: int, target_size: int) -> set[tuple[int, int]]:
    start = (random.randrange(cell_size), random.randrange(cell_size))
    cells = {start}
    while len(cells) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < cell_size and 0 <= cc < cell_size and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _sample_cell_pixels(cell_size: int) -> set[tuple[int, int]]:
    all_pixels = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    mode = random.choices(
        ["single", "scatter", "patch", "solid"],
        weights=[1, 2, 3, 2],
        k=1,
    )[0]

    if mode == "single":
        return {random.choice(all_pixels)}
    if mode == "scatter":
        max_size = max(2, min(len(all_pixels), cell_size * 2))
        size = random.randint(2, max_size)
        return set(random.sample(all_pixels, size))
    if mode == "patch":
        size = random.randint(max(2, cell_size), len(all_pixels))
        return _grow_connected_pattern(cell_size, size)
    return set(all_pixels)


def _cell_blue_mask(cell_rows: int, cell_cols: int, phase: int) -> np.ndarray:
    rows = np.arange(cell_rows, dtype=int)[:, None]
    cols = np.arange(cell_cols, dtype=int)[None, :]
    return ((rows + cols + int(phase)) % 2) == 0


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice([2, 3, 4, 5])
    cell_rows = random.randint(4, 7)
    cell_cols = random.randint(4, 7)
    phase = random.randint(0, 1)

    height = cell_rows * cell_size
    width = cell_cols * cell_size
    grid = np.full((height, width), BG, dtype=int)

    active_cells = [
        (r, c)
        for r, c in np.argwhere(_cell_blue_mask(cell_rows, cell_cols, phase))
    ]
    if len(active_cells) < 6:
        return None

    min_samples = max(4, len(active_cells) // 3)
    max_samples = max(min_samples, (2 * len(active_cells)) // 3)
    sample_count = random.randint(min_samples, max_samples)
    sampled_cells = random.sample(active_cells, sample_count)

    if len({r for r, _c in sampled_cells}) < 2 or len({c for _r, c in sampled_cells}) < 2:
        return None

    for cell_r, cell_c in sampled_cells:
        r0 = int(cell_r * cell_size)
        c0 = int(cell_c * cell_size)
        rel_pixels = _sample_cell_pixels(cell_size)
        for dr, dc in rel_pixels:
            grid[r0 + dr, c0 + dc] = BLUE

    corners = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    ]
    for r, c in corners:
        grid[r, c] = BG
    marker_r, marker_c = random.choice(corners)
    grid[marker_r, marker_c] = RED

    blue_coords = np.argwhere(grid == BLUE)
    if blue_coords.shape[0] < max(6, cell_size * 3):
        return None

    local_offsets = {(int(r % cell_size), int(c % cell_size)) for r, c in blue_coords}
    if len(local_offsets) < min(cell_size * cell_size, max(3, cell_size)):
        return None

    candidates = _valid_candidates(grid == BLUE)
    if len(candidates) != 1 or candidates[0] != (cell_size, phase):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-sampling input")
