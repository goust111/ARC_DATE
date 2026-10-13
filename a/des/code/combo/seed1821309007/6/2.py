import math
import random

import numpy as np


BG = 8
OUTPUT_COLORS = (2, 3, 1, 4, 6, 7, 9, 5)
INPUT_COLORS = tuple(c for c in range(10) if c != BG)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _partner(coord: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    return (h - 1 - r, w - 1 - c)


def _distance_key(coord: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int]:
    r, c = coord
    h, w = shape
    dr2 = 2 * r - (h - 1)
    dc2 = 2 * c - (w - 1)
    return (dr2 * dr2 + dc2 * dc2, r, c)


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    steps = math.gcd(abs(dr), abs(dc))
    if steps == 0:
        return [a]
    sr = dr // steps
    sc = dc // steps
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(steps + 1)]


def _extract_pairs(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    pairs = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg:
                continue
            coord = (r, c)
            if coord in seen:
                continue
            mate = _partner(coord, grid.shape)
            seen.add(coord)
            if mate == coord:
                continue
            if not (0 <= mate[0] < h and 0 <= mate[1] < w):
                continue
            if int(grid[mate]) == int(grid[coord]):
                seen.add(mate)
                if coord < mate:
                    pairs.append((coord, mate))
                else:
                    pairs.append((mate, coord))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each center-symmetric same-color pair and erase unpaired pixels."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    pairs = _extract_pairs(g, bg)
    pairs.sort(key=lambda pair: _distance_key(pair[0], g.shape))

    out = np.full_like(g, bg)
    for idx, (a, b) in enumerate(pairs):
        color = OUTPUT_COLORS[idx % len(OUTPUT_COLORS)]
        for r, c in _line_points(a, b):
            out[r, c] = color
    return out


def _candidate_endpoints(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            mate = _partner((r, c), shape)
            if (r, c) >= mate:
                continue
            line = _line_points((r, c), mate)
            if len(line) < 3:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_pairs(shape: tuple[int, int], pair_count: int):
    dist_keys = set()
    used_cells = set()
    chosen = []
    for a in _candidate_endpoints(shape):
        b = _partner(a, shape)
        dist = _distance_key(a, shape)[0]
        if dist in dist_keys:
            continue
        line = _line_points(a, b)
        if any(cell in used_cells for cell in line):
            continue
        chosen.append((a, b, line))
        dist_keys.add(dist)
        used_cells.update(line)
        if len(chosen) == pair_count:
            return chosen
    return None


def _place_orphans(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    target_count: int,
) -> int:
    h, w = grid.shape
    count = 0
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)
    blocked_mates = set()
    for coord in coords:
        if count >= target_count:
            break
        if coord in occupied or coord in blocked_mates:
            continue
        mate = _partner(coord, grid.shape)
        if mate == coord or mate in occupied:
            continue
        color = random.choice(INPUT_COLORS)
        grid[coord] = color
        occupied.add(coord)
        blocked_mates.add(mate)
        count += 1
    return count


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(8, 18)
    w = random.randint(8, 18)

    # Multiple centered lines remain visually clean only when the center is not a cell.
    if h % 2 == 1 and w % 2 == 1:
        if random.random() < 0.75:
            h += 1 if h < 18 else -1
        else:
            w += 1 if w < 18 else -1

    pair_cap = min(6, max(1, min(h, w) // 2))
    pair_count = random.randint(1, pair_cap)
    pairs = _place_pairs((h, w), pair_count)
    if pairs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = set()
    color_pool = list(INPUT_COLORS)
    random.shuffle(color_pool)

    for idx, (a, b, line) in enumerate(pairs):
        color = color_pool[idx % len(color_pool)]
        grid[a] = color
        grid[b] = color
        occupied.update(line)

    orphan_target = random.randint(0, min(4, (h * w) // 20 + 1))
    orphan_count = _place_orphans(grid, occupied, orphan_target)

    out = main(grid)
    non_bg_pairs = len(_extract_pairs(grid, BG))
    if non_bg_pairs != pair_count:
        return None
    if orphan_target > 0 and orphan_count == 0:
        return None
    if len(np.unique(out)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-pair input")
