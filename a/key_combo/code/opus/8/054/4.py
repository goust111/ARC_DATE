from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
START_COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    start_mask = (g != BLACK) & (g != BLUE)
    start_coords = np.argwhere(start_mask)
    if start_coords.size == 0:
        raise ValueError("missing start pixels")

    left_edge = int(start_coords[:, 1].min())
    n = int(w - left_edge)
    if n <= 0 or n > h:
        raise ValueError("invalid small-region size")

    region = g[h - n : h, w - n : w]
    local_starts = np.argwhere((region != BLACK) & (region != BLUE))
    if local_starts.size == 0:
        raise ValueError("small region has no starts")
    if not np.all(local_starts[:, 1] == 0):
        raise ValueError("starts must lie on the left edge of the small region")

    starts = sorted((int(r), int(region[r, 0])) for r in local_starts[:, 0])
    if len({r for r, _color in starts}) != len(starts):
        raise ValueError("start rows must be unique")

    blue_box = _bbox(g == BLUE)
    if blue_box is None:
        raise ValueError("missing large blue pattern")
    pattern = _crop(g, blue_box)
    ph, pw = pattern.shape
    if ph != pw or ph % n != 0:
        raise ValueError("large pattern must be a square integer scale-up")
    k = int(ph // n)
    if k <= 0:
        raise ValueError("invalid scale factor")

    obstacle = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            block = pattern[r * k : (r + 1) * k, c * k : (c + 1) * k]
            obstacle[r, c] = int(np.all(block == BLUE))
    return obstacle, starts, n, k


def _grow_paths(
    obstacle: np.ndarray,
    starts: list[tuple[int, int]],
    *,
    block_on_paths: bool = True,
    keep_order: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    n = int(obstacle.shape[0])
    ordered_starts = sorted(starts) if keep_order else list(starts)

    out = np.zeros((n, n), dtype=int)
    stats = {
        "obstacle_detours": 0,
        "occupied_detours": 0,
        "right_moves": 0,
        "down_moves": 0,
        "reached_right": 0,
        "stop_bottom": 0,
        "stop_obstacle": 0,
        "stop_path": 0,
    }

    for start_r, color in ordered_starts:
        r = int(start_r)
        c = 0
        out[r, c] = int(color)
        while c < n - 1:
            right_is_obstacle = bool(obstacle[r, c + 1] == 1)
            right_is_path = bool(block_on_paths and out[r, c + 1] != 0)
            if not right_is_obstacle and not right_is_path:
                c += 1
                out[r, c] = int(color)
                stats["right_moves"] += 1
                if c == n - 1:
                    stats["reached_right"] += 1
                continue

            if right_is_obstacle:
                stats["obstacle_detours"] += 1
            if right_is_path:
                stats["occupied_detours"] += 1

            nr = r + 1
            if nr >= n:
                stats["stop_bottom"] += 1
                break
            if obstacle[nr, c] == 1:
                stats["stop_obstacle"] += 1
                break
            if block_on_paths and out[nr, c] != 0:
                stats["stop_path"] += 1
                break

            r = nr
            out[r, c] = int(color)
            stats["down_moves"] += 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the large blue pattern into an obstacle map, then grow colored paths inside the small start panel.

    Input structure:
    - The background is black.
    - A large blue-and-black square pattern sits in the upper-left area; it is an integer scale-up of a smaller binary grid, with each kxk block either fully blue or fully black.
    - A separate square region sits flush to the lower-right corner; only its left edge contains 2-3 non-blue start pixels of distinct colors, and the rest of that region is black.

    Rule:
    - Infer the small region size from the lower-right start panel, and infer the scale factor k from the large blue pattern's square bounding box.
    - Downsample the large pattern into a small obstacle map: a cell is an obstacle iff its entire kxk block is blue.
    - Create an all-black output of the small region's size and process start pixels from top to bottom.
    - Each path begins at its start pixel, repeatedly tries to move right, and if the right cell is an obstacle or already occupied by an earlier path, it moves down instead; if that downward move is out of bounds, blocked by an obstacle, or already occupied, the path stops.

    Key insight:
    - The visible blue object is not the answer region itself; after integer downsampling it becomes a hidden navigation map that controls how the colored starts in the lower-right panel can grow.
    """
    obstacle, starts, _n, _k = _parse_input(grid)
    out, _stats = _grow_paths(obstacle, starts, block_on_paths=True, keep_order=True)
    return out


def _valid_pairs() -> list[tuple[int, int, int, int]]:
    pairs: list[tuple[int, int, int, int]] = []
    for n in range(4, 7):
        for k in (3, 4):
            big = n * k
            for top in range(0, 3):
                for left in range(0, 3):
                    for gap_r in range(1, 4):
                        for gap_c in range(1, 4):
                            h = top + big + gap_r + n
                            w = left + big + gap_c + n
                            if h <= 30 and w <= 30:
                                pairs.append((n, k, top, left))
    return pairs


_GEOMETRY_CHOICES = _valid_pairs()


def _random_obstacle(rng: np.random.Generator, n: int, start_rows: list[int]) -> np.ndarray:
    density = float(rng.uniform(0.22, 0.42))
    obstacle = (rng.random((n, n)) < density).astype(int)

    obstacle[start_rows, 0] = 0
    for r in range(n):
        if obstacle[r].all():
            choices = [c for c in range(n) if not (c == 0 and r in start_rows)]
            obstacle[r, int(rng.choice(choices))] = 0

    if int(obstacle[0].sum()) == 0:
        obstacle[0, int(rng.integers(1, n))] = 1
    if int(obstacle[n - 1].sum()) == 0:
        obstacle[n - 1, int(rng.integers(1, n))] = 1

    non_start_rows = [r for r in range(n) if r not in start_rows]
    if not non_start_rows:
        raise ValueError("need a non-start row to anchor the left edge of the blue bbox")
    if int(obstacle[:, 0].sum()) == 0:
        obstacle[int(rng.choice(non_start_rows)), 0] = 1
    if int(obstacle[:, n - 1].sum()) == 0:
        obstacle[int(rng.integers(0, n)), n - 1] = 1

    for r in range(n):
        if obstacle[r].all():
            free_cols = [c for c in range(n) if not (c == 0 and r in start_rows)]
            obstacle[r, int(rng.choice(free_cols))] = 0
    return obstacle


def _build_input(
    obstacle: np.ndarray,
    starts: list[tuple[int, int]],
    *,
    k: int,
    top: int,
    left: int,
    gap_r: int,
    gap_c: int,
) -> np.ndarray:
    n = int(obstacle.shape[0])
    big = np.kron(obstacle.astype(int), np.ones((k, k), dtype=int)) * BLUE
    big_size = int(big.shape[0])
    h = top + big_size + gap_r + n
    w = left + big_size + gap_c + n
    grid = np.zeros((h, w), dtype=int)

    grid[top : top + big_size, left : left + big_size] = big
    sr0 = h - n
    sc0 = w - n
    for r, color in starts:
        grid[sr0 + int(r), sc0] = int(color)
    return grid


def _same_starts(starts_a: list[tuple[int, int]], starts_b: list[tuple[int, int]]) -> bool:
    return sorted(starts_a) == sorted(starts_b)


def _semantic_checks(grid: np.ndarray, obstacle: np.ndarray, starts: list[tuple[int, int]], expected: np.ndarray) -> bool:
    try:
        parsed_obstacle, parsed_starts, _n, _k = _parse_input(grid)
    except Exception:
        return False

    if not np.array_equal(parsed_obstacle, obstacle):
        return False
    if not _same_starts(parsed_starts, starts):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    no_obstacle_out, _ = _grow_paths(np.zeros_like(obstacle), starts, block_on_paths=True, keep_order=True)
    if np.array_equal(no_obstacle_out, expected):
        return False

    reversed_out, _ = _grow_paths(obstacle, list(reversed(sorted(starts))), block_on_paths=True, keep_order=False)
    if np.array_equal(reversed_out, expected):
        return False

    ignore_paths_out, _ = _grow_paths(obstacle, starts, block_on_paths=False, keep_order=True)
    if np.array_equal(ignore_paths_out, expected):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n, k, top, left = _GEOMETRY_CHOICES[int(rng.integers(0, len(_GEOMETRY_CHOICES)))]
    gap_r = int(rng.integers(1, 4))
    gap_c = int(rng.integers(1, 4))
    big_size = n * k
    if top + big_size + gap_r + n > 30 or left + big_size + gap_c + n > 30:
        return None

    start_count = int(rng.choice(np.array([2, 3], dtype=int)))
    start_rows = sorted(int(x) for x in rng.choice(n, size=start_count, replace=False))
    start_colors = [int(x) for x in rng.choice(np.array(START_COLOR_POOL, dtype=int), size=start_count, replace=False)]
    starts = list(zip(start_rows, start_colors))

    obstacle = _random_obstacle(rng, n, start_rows)
    expected, stats = _grow_paths(obstacle, starts, block_on_paths=True, keep_order=True)

    if stats["obstacle_detours"] < 1:
        return None
    if stats["occupied_detours"] < 1:
        return None
    if stats["right_moves"] < n:
        return None
    if stats["down_moves"] < 2:
        return None
    if len(np.unique(expected[expected != 0])) != start_count:
        return None
    if np.array_equal(expected, 0 * expected):
        return None

    grid = _build_input(obstacle, starts, k=k, top=top, left=left, gap_r=gap_r, gap_c=gap_c)
    if not _semantic_checks(grid, obstacle, starts, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
