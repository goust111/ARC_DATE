import hashlib
import random
from collections import deque

import numpy as np


def _foreground_color(grid: np.ndarray) -> int:
    colors = sorted(int(color) for color in np.unique(grid))
    assert len(colors) == 2 and colors[0] == 0, "expected black plus one foreground color"
    return int(colors[1])


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _invert_binary(grid: np.ndarray, foreground: int) -> np.ndarray:
    return np.where(grid == 0, foreground, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """Count the foreground pixels, invert the two-color pattern, then tile the inverted tile N by N.

    Input structure:
    - A small 2D grid uses exactly two colors: black (0) as background and one non-black foreground color.
    - The foreground pattern may be connected or split into multiple pieces.
    - The number of foreground pixels is between 2 and 4.

    Rule:
    - Count all non-black pixels and call that count N.
    - Swap black with the foreground color across the whole grid.
    - Repeat the swapped grid N times horizontally and N times vertically.

    Key insight:
    - The same foreground pixels serve two roles at once: they define which cells get inverted and their count also fixes the tiling repeat count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    fg = _foreground_color(g)
    n = int(np.count_nonzero(g == fg))
    assert 2 <= n <= 4, "expected 2 to 4 foreground pixels"
    swapped = _invert_binary(g, fg)
    return np.tile(swapped, (n, n))


GRID_SIDE_RANGE = (3, 5)
FOREGROUND_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _random_connected_positions(height: int, width: int, n: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < n:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return []
        cells.add(random.choice(frontier))
    return sorted(cells)


def _random_disconnected_positions(height: int, width: int, n: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(250):
        picked = sorted(random.sample(cells, n))
        mask = np.zeros((height, width), dtype=bool)
        for r, c in picked:
            mask[r, c] = True
        if _count_components(mask) >= 2:
            return picked
    return []


def _positions_match_style(
    positions: list[tuple[int, int]],
    height: int,
    width: int,
    want_connected: bool,
    want_two_dimensional: bool,
) -> bool:
    if not positions:
        return False
    mask = np.zeros((height, width), dtype=bool)
    for r, c in positions:
        mask[r, c] = True
    component_count = _count_components(mask)
    if want_connected and component_count != 1:
        return False
    if not want_connected and component_count < 2:
        return False
    rows = {r for r, _c in positions}
    cols = {c for _r, c in positions}
    if want_two_dimensional and (len(rows) == 1 or len(cols) == 1):
        return False
    if len(rows) == height and len(cols) == width and len(positions) == height * width:
        return False
    return True


def _build_candidate(
    n: int,
    want_connected: bool,
    want_two_dimensional: bool,
    size_index: int,
) -> np.ndarray | None:
    shapes = [(h, w) for h in range(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1) for w in range(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1)]
    height, width = shapes[size_index % len(shapes)]
    if n >= height * width:
        return None

    positions = (
        _random_connected_positions(height, width, n)
        if want_connected
        else _random_disconnected_positions(height, width, n)
    )
    if not _positions_match_style(positions, height, width, want_connected, want_two_dimensional):
        return None

    foreground = random.choice(FOREGROUND_COLORS)
    grid = np.zeros((height, width), dtype=int)
    for r, c in positions:
        grid[r, c] = foreground
    return grid


def _has_ambiguous_family_collision(grid: np.ndarray) -> bool:
    fg = _foreground_color(grid)
    n = int(np.count_nonzero(grid == fg))
    out = main(grid)
    out_h, out_w = out.shape

    for alt_n in range(2, 5):
        if alt_n == n or out_h % alt_n != 0 or out_w % alt_n != 0:
            continue
        tile_h = out_h // alt_n
        tile_w = out_w // alt_n
        if not (3 <= tile_h <= 5 and 3 <= tile_w <= 5):
            continue
        tile = out[:tile_h, :tile_w]
        if not np.array_equal(np.tile(tile, (alt_n, alt_n)), out):
            continue
        candidate = np.where(tile == 0, fg, 0)
        if np.array_equal(candidate, grid):
            continue
        colors = set(map(int, np.unique(candidate)))
        if colors == {0, fg} and int(np.count_nonzero(candidate == fg)) == alt_n:
            return True
    return False


def _validate_instance(grid: np.ndarray, want_connected: bool) -> bool:
    try:
        fg = _foreground_color(grid)
        n = int(np.count_nonzero(grid == fg))
        if not (2 <= n <= 4):
            return False
        if not (grid.ndim == 2 and 3 <= grid.shape[0] <= 5 and 3 <= grid.shape[1] <= 5):
            return False

        component_count = _count_components(grid == fg)
        if want_connected and component_count != 1:
            return False
        if not want_connected and component_count < 2:
            return False

        swapped = _invert_binary(grid, fg)
        out = main(grid)
        if out.shape != (grid.shape[0] * n, grid.shape[1] * n):
            return False
        if max(out.shape) > 30 or not np.array_equal(out, np.tile(swapped, (n, n))):
            return False
        if len(np.unique(out)) != 2:
            return False
        if _has_ambiguous_family_collision(grid):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    if not hasattr(generate_input, "_counter"):
        generate_input._counter = 0
        generate_input._seen_inputs = set()
        generate_input._seen_outputs = set()

    state = int(generate_input._counter)
    generate_input._counter += 1

    target_n = 2 + (state % 3)
    want_connected = ((state // 3) % 2 == 0)
    want_two_dimensional = target_n >= 3 and (state % 4) in (0, 1)
    size_index = state

    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate(target_n, want_connected, want_two_dimensional, size_index + random.randrange(9))
        if grid is None or not _validate_instance(grid, want_connected):
            continue

        input_sig = _grid_signature(grid)
        output_sig = _grid_signature(main(grid))
        if input_sig in generate_input._seen_inputs:
            continue
        if output_sig in generate_input._seen_outputs:
            continue

        generate_input._seen_inputs.add(input_sig)
        generate_input._seen_outputs.add(output_sig)
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 5")
