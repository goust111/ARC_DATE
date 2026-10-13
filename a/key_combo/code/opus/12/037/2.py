from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREEN = 3
STRIPE_COLORS = (1, 2, 4, 5, 6, 7, 9)
NOISE_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 500


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bounded_composition(total: int, parts: int, lows: list[int], highs: list[int]) -> list[int] | None:
    out: list[int] = []
    remaining = total
    for i in range(parts):
        tail_lo = sum(lows[i + 1 :])
        tail_hi = sum(highs[i + 1 :])
        lo = max(lows[i], remaining - tail_hi)
        hi = min(highs[i], remaining - tail_lo)
        if lo > hi:
            return None
        out.append(random.randint(lo, hi))
        remaining -= out[-1]
    if remaining != 0:
        return None
    return out


def _row_dominant_color(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row if int(v) != GREEN)
    assert counts, "each row must retain at least one non-green stripe cell"
    best = counts.most_common()
    top_count = best[0][1]
    winners = [color for color, count in best if count == top_count]
    assert len(winners) == 1, "each row must have a unique dominant non-green color"
    return int(winners[0])


def _denoise_rows(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    out = np.asarray(grid, dtype=int).copy()
    row_colors: list[int] = []
    for r in range(out.shape[0]):
        dominant = _row_dominant_color(out[r, :])
        row_colors.append(dominant)
        mask = out[r, :] != GREEN
        out[r, mask] = dominant
    return out, row_colors


def _green_path(grid: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(np.asarray(grid) == GREEN)}
    assert cells, "expected a green path"

    degrees = {}
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        assert degree in (1, 2), "green structure must be a non-branching path"
        degrees[(r, c)] = degree

    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    ordered: list[Coord] = []
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        ordered.append(cur)
        next_cells = [n for n in _neighbors4(*cur) if n in cells and n != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(ordered) == len(cells), "green path must be 4-connected"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise horizontal stripes, sample their colors along the green path, and repaint that path with the reversed sampled sequence.

    Input structure:
    - The grid is a full horizontal-stripe canvas whose true row colors come from non-green pixels.
    - Each row has one dominant stripe color plus a few non-green noise pixels, while green marks one 4-connected one-pixel-wide path.
    - The path has exactly two endpoints, no branches, and runs from one grid edge to another.

    Rule:
    - For each row, ignore green cells, find the unique dominant remaining color, and repaint every non-green cell in that row to it.
    - Find the two green endpoints, choosing the smaller `(row, col)` endpoint as the traversal start, then walk the path to the other endpoint.
    - Record the denoised row color under each visited path cell, reverse that color sequence, and write it back onto the same path cells in traversal order.
    - Leave every non-path cell as produced by the row denoising step.

    Key insight:
    - The path itself carries no useful colors; it must read the cleaned stripe identity of each crossed row, so row-major denoising and endpoint-ordered traversal are both necessary before the final reversal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    denoised, row_colors = _denoise_rows(g)
    path = _green_path(g)
    sampled = [row_colors[r] for r, _c in path]

    out = denoised.copy()
    for (r, c), color in zip(path, reversed(sampled)):
        out[r, c] = int(color)
    return out


def _append_segment(path: list[Coord], dr: int, dc: int, steps: int) -> None:
    r, c = path[-1]
    for _ in range(steps):
        r += dr
        c += dc
        path.append((r, c))


def _build_vertical_path(height: int, width: int) -> list[Coord] | None:
    feasible_parts = [parts for parts in range(4, 7) if parts <= height - 1 <= 4 * parts]
    if not feasible_parts:
        return None

    parts = random.choice(feasible_parts)
    runs = _bounded_composition(height - 1, parts, [1] * parts, [4] * parts)
    if runs is None:
        return None

    col = random.randint(0, width - 1)
    path: list[Coord] = [(0, col)]
    horiz_count = 0
    for i, run in enumerate(runs):
        _append_segment(path, 1, 0, run)
        if i == len(runs) - 1:
            if random.random() < 0.6:
                candidates = []
                for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
                    new_col = col + delta
                    if 0 <= new_col < width:
                        candidates.append(delta)
                if candidates:
                    delta = random.choice(candidates)
                    _append_segment(path, 0, 1 if delta > 0 else -1, abs(delta))
                    col += delta
                    horiz_count += 1
            continue

        candidates = []
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            new_col = col + delta
            if 0 <= new_col < width:
                candidates.append(delta)
        if not candidates:
            return None
        delta = random.choice(candidates)
        _append_segment(path, 0, 1 if delta > 0 else -1, abs(delta))
        col += delta
        horiz_count += 1

    if horiz_count < 2:
        return None
    return path


def _build_horizontal_same_row_path(height: int, width: int) -> list[Coord] | None:
    parts = _bounded_composition(width - 1, 5, [1, 1, 1, 1, 1], [3, 4, 4, 4, 3])
    if parts is None:
        return None

    start_row = random.randint(0, height - 1)
    other_rows = random.sample([r for r in range(height) if r != start_row], 3)
    random.shuffle(other_rows)
    row_targets = [start_row, other_rows[0], other_rows[1], other_rows[2], start_row]

    path: list[Coord] = [(start_row, 0)]
    cur_row = start_row
    for i in range(4):
        _append_segment(path, 0, 1, parts[i])
        next_row = row_targets[i + 1]
        step = 1 if next_row > cur_row else -1
        _append_segment(path, step, 0, abs(next_row - cur_row))
        cur_row = next_row
    _append_segment(path, 0, 1, parts[4])
    return path


def _path_is_valid(path: list[Coord], height: int, width: int) -> bool:
    if path is None or len(path) < 2:
        return False
    if path[0] == path[-1]:
        return False
    if not any(r in (0, height - 1) or c in (0, width - 1) for r, c in (path[0], path[-1])):
        return False
    if not ((path[0][0] in (0, height - 1) or path[0][1] in (0, width - 1)) and (path[-1][0] in (0, height - 1) or path[-1][1] in (0, width - 1))):
        return False
    seen = set()
    for a, b in zip(path, path[1:]):
        if a in seen:
            return False
        seen.add(a)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    if path[-1] in seen:
        return False

    cells = set(path)
    degrees = {}
    for r, c in path:
        if not (0 <= r < height and 0 <= c < width):
            return False
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        degrees[(r, c)] = degree
        if degree not in (1, 2):
            return False
    endpoints = [coord for coord, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        return False
    touched_rows = {r for r, _c in path}
    return len(touched_rows) >= 4


def _sample_row_colors(height: int) -> list[int]:
    rows = [random.choice(STRIPE_COLORS)]
    for _ in range(1, height):
        choices = [color for color in STRIPE_COLORS if color != rows[-1]]
        rows.append(random.choice(choices))
    return rows


def _path_row_counts(path: list[Coord], height: int) -> list[int]:
    counts = [0] * height
    for r, _c in path:
        counts[r] += 1
    return counts


def _paint_path(grid: np.ndarray, path: list[Coord]) -> None:
    for r, c in path:
        grid[r, c] = GREEN


def _add_noise(grid: np.ndarray, path: list[Coord], row_colors: list[int]) -> bool:
    height, width = grid.shape
    path_cells = set(path)
    row_path_counts = _path_row_counts(path, height)
    path_rows = [r for r, count in enumerate(row_path_counts) if count > 0]
    special_row = random.choice(path_rows)
    used_special = False

    for r in range(height):
        max_noise = min(3, width - row_path_counts[r] - 2)
        if max_noise < 1:
            return False
        noise_count = random.randint(1, max_noise)
        available_cols = [c for c in range(width) if (r, c) not in path_cells]
        if len(available_cols) < noise_count:
            return False
        random.shuffle(available_cols)
        choices = [color for color in NOISE_COLORS if color != row_colors[r]]
        if noise_count > len(choices):
            return False
        colors = random.sample(choices, noise_count)
        for c, color in zip(available_cols[:noise_count], colors):
            grid[r, c] = color
            if r == special_row:
                used_special = True
    return used_special


def _wrong_no_reverse(grid: np.ndarray) -> np.ndarray:
    denoised, row_colors = _denoise_rows(grid)
    out = denoised.copy()
    for r, c in _green_path(grid):
        out[r, c] = row_colors[r]
    return out


def _wrong_keep_noise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    denoised, row_colors = _denoise_rows(grid)
    path = _green_path(grid)
    _ = denoised
    for (r, c), color in zip(path, reversed([row_colors[r] for r, _c in path])):
        g[r, c] = color
    return g


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    path = _green_path(inp)
    sampled = [_row_dominant_color(inp[r, :]) for r, _c in path]
    if sampled == sampled[::-1]:
        return False
    if np.array_equal(_wrong_no_reverse(inp), out):
        return False
    if np.array_equal(_wrong_keep_noise(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(12, 18)
    mode = random.choice(("vertical", "horizontal_same_row"))

    if mode == "vertical":
        path = _build_vertical_path(height, width)
    else:
        path = _build_horizontal_same_row_path(height, width)
    if path is None or not _path_is_valid(path, height, width):
        return None

    row_counts = _path_row_counts(path, height)
    if max(row_counts) > width - 5:
        return None

    row_colors = _sample_row_colors(height)
    grid = np.empty((height, width), dtype=int)
    for r, color in enumerate(row_colors):
        grid[r, :] = color

    _paint_path(grid, path)
    if not _add_noise(grid, path, row_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-path reversal puzzle")
