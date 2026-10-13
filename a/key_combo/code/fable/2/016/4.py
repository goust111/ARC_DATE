from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
CORRIDOR = 5
INK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _on_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _path_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for rr, cc in _neighbors4(r, c):
        if _in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
            out.append((rr, cc))
    return out


def _ordered_corridor(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty corridor"

    endpoints: list[tuple[int, int]] = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        deg = len(_path_neighbors(mask, r, c))
        assert deg in (1, 2), "corridor must be a simple path"
        if deg == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "corridor must have exactly two endpoints"
    border_endpoints = [pt for pt in endpoints if _on_border(mask.shape, *pt)]
    assert len(border_endpoints) == 1, "exactly one endpoint must touch the border"
    start = border_endpoints[0]

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [pt for pt in _path_neighbors(mask, *cur) if pt != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "corridor may not branch"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == int(mask.sum()), "corridor must be one connected path"
    return ordered


def _count_corners(path: list[tuple[int, int]]) -> int:
    corners = 0
    for i in range(1, len(path) - 1):
        pr, pc = path[i - 1]
        r, c = path[i]
        nr, nc = path[i + 1]
        d1 = (r - pr, c - pc)
        d2 = (nr - r, nc - c)
        if d1 != d2:
            corners += 1
    return corners


def _run_lengths(n: int) -> list[int]:
    return list(range(1, n + 1)) + list(range(n - 1, 0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode the corridor by counting its bends and writing a 1..n..1 run pattern along it.

    Input structure
    - The background is black.
    - One gray 4-connected foreground component is a simple corridor path, and exactly one of its endpoints touches the grid border.
    - One additional non-gray foreground pixel is isolated from the corridor and supplies the ink color.

    Rule
    - Follow the corridor from the unique border endpoint.
    - Count the corridor cells where the travel direction changes; call that count n.
    - Recolor corridor cells in runs of lengths 1,2,...,n,n-1,...,1 using the isolated pixel's color, leaving one gray separator cell between consecutive runs.
    - After the sequence ends, keep the rest of the corridor gray and leave the isolated pixel unchanged.

    Key insight
    - The corridor is both the data source and the writing tape: its 2D shape only matters through the 1D traversal order and the number of turns along that order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 2, "expected one corridor color and one ink pixel color"

    counts = Counter(map(int, g.ravel()))
    singleton_colors = [color for color in colors if counts[color] == 1]
    assert len(singleton_colors) == 1, "expected exactly one isolated ink pixel"
    ink = singleton_colors[0]
    corridor_color = [color for color in colors if color != ink][0]

    mask = g == corridor_color
    path = _ordered_corridor(mask)
    n = _count_corners(path)
    assert n >= 1, "the corridor must bend at least once"

    runs = _run_lengths(n)
    needed = sum(runs) + len(runs) - 1
    assert needed <= len(path), "corridor is too short for the encoded sequence"

    out = g.copy()
    idx = 0
    for run_i, run_len in enumerate(runs):
        for _ in range(run_len):
            r, c = path[idx]
            out[r, c] = ink
            idx += 1
        if run_i + 1 < len(runs):
            idx += 1
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[int(rng.integers(parts))] += 1
    return out


def _canonical_corridor(
    n: int,
    total_cells: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    segment_count = n + 1
    vertical_indices = list(range(0, segment_count, 2))
    horizontal_indices = list(range(1, segment_count, 2))

    min_steps = [1] * segment_count
    if len(horizontal_indices) >= 2:
        for idx in vertical_indices:
            min_steps[idx] = 2

    total_steps = total_cells - 1
    min_total = sum(min_steps)
    if total_steps < min_total:
        return None

    extras = _random_composition(total_steps - min_total, segment_count, rng)
    steps = [base + extra for base, extra in zip(min_steps, extras)]

    signs: list[int] = []
    if horizontal_indices:
        cur_sign = int(rng.choice(np.array([-1, 1], dtype=int)))
        signs.append(cur_sign)
        for _ in range(1, len(horizontal_indices)):
            if float(rng.random()) < 0.7:
                cur_sign *= -1
            signs.append(cur_sign)

    x = 0
    x_positions = [0]
    for step, sign in zip((steps[i] for i in horizontal_indices), signs):
        x += sign * step
        x_positions.append(x)

    min_x = min(x_positions)
    max_x = max(x_positions)
    left_slack = int(rng.integers(0, 4))
    right_slack = int(rng.integers(0, 4))
    width = int(max(6, (max_x - min_x) + 3 + left_slack + right_slack))
    start_c = int(1 - min_x + left_slack)

    total_down = sum(steps[i] for i in vertical_indices)
    bottom_margin = int(rng.integers(1, 4))
    height = int(max(6, total_down + 1 + bottom_margin))
    if max(height, width) > 30:
        return None

    path = [(0, start_c)]
    r, c = 0, start_c
    sign_i = 0
    for seg_idx, step in enumerate(steps):
        if seg_idx % 2 == 0:
            for _ in range(step):
                r += 1
                path.append((r, c))
        else:
            sign = signs[sign_i]
            sign_i += 1
            for _ in range(step):
                c += sign
                path.append((r, c))

    if r >= height - 1:
        return None
    if any(not _in_bounds((height, width), rr, cc) for rr, cc in path):
        return None
    if sum(_on_border((height, width), rr, cc) for rr, cc in path) != 1:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for rr, cc in path:
        if mask[rr, cc]:
            return None
        mask[rr, cc] = True

    try:
        ordered = _ordered_corridor(mask)
    except AssertionError:
        return None
    if ordered != path:
        return None
    if _count_corners(path) != n:
        return None
    return (height, width), path


def _transform_path(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    edge: str,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h, w = map(int, shape)
    if edge == "top":
        return shape, path
    if edge == "bottom":
        return (h, w), [(h - 1 - r, c) for r, c in path]
    if edge == "left":
        return (w, h), [(w - 1 - c, r) for r, c in path]
    if edge == "right":
        return (w, h), [(c, h - 1 - r) for r, c in path]
    raise ValueError("unknown edge transform")


def _find_isolated_spot(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    candidates = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(_in_bounds(mask.shape, rr, cc) and mask[rr, cc] for rr, cc in _neighbors4(r, c)):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _paint_first_k(path: list[tuple[int, int]], grid: np.ndarray, ink: int, k: int) -> np.ndarray:
    out = grid.copy()
    for r, c in path[:k]:
        out[r, c] = ink
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    bg = _get_bg(inp)
    colors = sorted(set(map(int, inp.ravel())) - {bg})
    if len(colors) != 2:
        return False

    counts = Counter(map(int, inp.ravel()))
    singleton_colors = [color for color in colors if counts[color] == 1]
    if len(singleton_colors) != 1:
        return False
    ink = singleton_colors[0]
    corridor_color = [color for color in colors if color != ink][0]

    path = _ordered_corridor(inp == corridor_color)
    n = _count_corners(path)
    if n < 2:
        return False

    runs = _run_lengths(n)
    needed = sum(runs) + len(runs) - 1
    if needed >= len(path):
        return False

    recolored = sum(int(inp[r, c] == corridor_color and out[r, c] == ink) for r, c in path)
    if recolored != n * n:
        return False

    if np.array_equal(out, _paint_first_k(path, inp, ink, n * n)):
        return False
    if np.array_equal(out, _paint_first_k(path, inp, ink, needed)):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice(np.array([2, 3, 4, 5], dtype=int)))
    used_cells = n * n + 2 * n - 2
    total_cells = int(used_cells + rng.integers(2, 9))

    built = _canonical_corridor(n, total_cells, rng)
    if built is None:
        return None
    shape, path = built

    edge = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    shape, path = _transform_path(shape, path, edge)

    grid = np.full(shape, BG, dtype=int)
    for r, c in path:
        grid[r, c] = CORRIDOR

    mask = grid == CORRIDOR
    spot = _find_isolated_spot(mask, rng)
    if spot is None:
        return None
    ink = int(rng.choice(np.array(INK_COLORS, dtype=int)))
    grid[spot] = ink

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _semantic_checks(grid, out):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor self-encoding input")
