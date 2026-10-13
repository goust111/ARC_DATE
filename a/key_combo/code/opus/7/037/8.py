from __future__ import annotations

import random

import numpy as np


BLACK = 0
TEAL = 8
SEED_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _center_cell(height: int, width: int) -> tuple[int, int] | None:
    if height % 2 == 1 and width % 2 == 1:
        return height // 2, width // 2
    return None


def _edge_distance(r: int, c: int, height: int, width: int) -> int:
    return min(r, c, height - 1 - r, width - 1 - c)


def _chebyshev_clear(pos: tuple[int, int], used: list[tuple[int, int]], min_gap: int) -> bool:
    r, c = pos
    return all(max(abs(r - ur), abs(c - uc)) >= min_gap for ur, uc in used)


def _transform_details(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tiled = np.tile(np.asarray(grid, dtype=int), (2, 2))
    seed_mask = (tiled != BLACK) & (tiled != TEAL)
    candidate_mask = np.zeros_like(tiled, dtype=bool)

    for r, c in np.argwhere(seed_mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIAGONALS:
            nr = r + dr
            nc = c + dc
            if _in_bounds(tiled.shape, nr, nc) and int(tiled[nr, nc]) == BLACK:
                candidate_mask[nr, nc] = True

    supported_mask = tiled != BLACK
    supported_mask[candidate_mask] = True

    keep_mask = np.zeros_like(tiled, dtype=bool)
    for r, c in np.argwhere(candidate_mask):
        r = int(r)
        c = int(c)
        support = 0
        for dr, dc in ORTHOGONALS:
            nr = r + dr
            nc = c + dc
            if _in_bounds(tiled.shape, nr, nc) and supported_mask[nr, nc]:
                support += 1
        if support >= 2:
            keep_mask[r, c] = True

    out = tiled.copy()
    out[candidate_mask] = TEAL
    out[candidate_mask & ~keep_mask] = BLACK
    return tiled, candidate_mask, keep_mask, out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the seed grid into four quadrants, add diagonal teal marks, then delete unsupported teal pixels.

    Input structure:
    - The input is a small black grid containing one to three non-black, non-teal seed pixels.
    - Seeds may use different colors, but only black is background and teal is reserved for new marks.
    - At least one seed is typically close to an outer edge, so the 2x2 tiling creates visible seam interactions.

    Rule:
    - Copy the whole input into a 2x2 tiled canvas.
    - For every non-black, non-teal seed in the tiled canvas, paint each black diagonal neighbor teal.
    - Evaluate all newly painted teal cells together: count how many orthogonal neighbors are non-black when seeds and all candidate teal marks are both treated as support.
    - Delete a teal cell if it has fewer than two orthogonal non-black neighbors; keep it otherwise.
    - Never modify the original seed pixels.

    Key insight:
    - Tiling can turn far-apart edge seeds into near neighbors across tile seams, so some diagonal teal marks survive only because duplicated tiles support each other.
    """
    _tiled, _candidate_mask, _keep_mask, out = _transform_details(grid)
    return out


def _single_seed_layout() -> tuple[int, int, list[tuple[int, int]]]:
    height = 3
    width = 3
    center = _center_cell(height, width)
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) != center and _edge_distance(r, c, height, width) <= 1
    ]
    return height, width, [random.choice(candidates)]


def _wrap_pair_options(height: int, width: int, orientation: str) -> list[list[tuple[int, int]]]:
    center = _center_cell(height, width)
    options: list[list[tuple[int, int]]] = []

    if orientation == "horizontal":
        left_cols = sorted({0, min(1, width - 1)})
        right_cols = sorted({width - 1, max(0, width - 2)})
        for row in range(height):
            for c0 in left_cols:
                for c1 in right_cols:
                    if abs(c0 - c1) < 2:
                        continue
                    pair = [(row, c0), (row, c1)]
                    if center is not None and center in pair:
                        continue
                    options.append(pair)
    else:
        top_rows = sorted({0, min(1, height - 1)})
        bottom_rows = sorted({height - 1, max(0, height - 2)})
        for col in range(width):
            for r0 in top_rows:
                for r1 in bottom_rows:
                    if abs(r0 - r1) < 2:
                        continue
                    pair = [(r0, col), (r1, col)]
                    if center is not None and center in pair:
                        continue
                    options.append(pair)

    random.shuffle(options)
    return options


def _choose_extra_seed(height: int, width: int, used: list[tuple[int, int]]) -> tuple[int, int] | None:
    center = _center_cell(height, width)
    all_positions = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in used and (center is None or (r, c) != center)
    ]
    used_rows = {r for r, _c in used}
    used_cols = {c for _r, c in used}

    filters = [
        lambda pos: _edge_distance(pos[0], pos[1], height, width) >= 2
        and _chebyshev_clear(pos, used, 2)
        and pos[0] not in used_rows
        and pos[1] not in used_cols,
        lambda pos: _edge_distance(pos[0], pos[1], height, width) >= 1
        and _chebyshev_clear(pos, used, 2)
        and pos[0] not in used_rows
        and pos[1] not in used_cols,
        lambda pos: _chebyshev_clear(pos, used, 2),
        lambda pos: _chebyshev_clear(pos, used, 1),
    ]

    for predicate in filters:
        candidates = [pos for pos in all_positions if predicate(pos)]
        if candidates:
            return random.choice(candidates)
    return None


def _build_grid() -> np.ndarray | None:
    seed_count = random.choices((1, 2, 3), weights=(1, 4, 4), k=1)[0]
    if seed_count == 1:
        height, width, positions = _single_seed_layout()
    else:
        height = random.randint(4, 6)
        width = random.randint(4, 6)
        orientation = random.choice(("horizontal", "vertical"))
        pair_options = _wrap_pair_options(height, width, orientation)
        if not pair_options:
            return None
        positions = list(random.choice(pair_options))
        if seed_count == 3:
            extra = _choose_extra_seed(height, width, positions)
            if extra is None:
                return None
            positions.append(extra)

    grid = np.zeros((height, width), dtype=int)
    for pos, color in zip(positions, random.sample(SEED_COLORS, len(positions))):
        grid[pos] = color
    return grid


def _has_seam_support(keep_mask: np.ndarray, height: int, width: int) -> bool:
    for r, c in np.argwhere(keep_mask):
        r = int(r)
        c = int(c)
        if r in (height - 1, height) or c in (width - 1, width):
            return True
    return False


def _valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = grid.shape
    if max(height, width) > 30:
        return False

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid != BLACK)]
    if not (1 <= len(seeds) <= 3):
        return False

    center = _center_cell(height, width)
    if center is not None and center in seeds:
        return False
    if not any(_edge_distance(r, c, height, width) <= 1 for r, c in seeds):
        return False
    if any(int(grid[r, c]) == TEAL for r, c in seeds):
        return False

    _tiled, candidate_mask, keep_mask, out = _transform_details(grid)
    if max(out.shape) > 30:
        return False

    kept = int(np.count_nonzero(keep_mask))
    deleted = int(np.count_nonzero(candidate_mask & ~keep_mask))
    if kept == 0 or deleted == 0:
        return False
    if not _has_seam_support(keep_mask, height, width):
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if grid is not None and _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
