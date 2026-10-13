from __future__ import annotations

import random
from collections import deque

import numpy as np


ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
SCENARIOS = ("straight", "corner", "tee", "cross")
ARC_COLORS = tuple(range(10))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ORTH_DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


def _find_seed_color(g: np.ndarray) -> int:
    h, w = g.shape
    border = int(g[0, 0])
    assert np.all(g[0, :] == border)
    assert np.all(g[-1, :] == border)
    assert np.all(g[:, 0] == border)
    assert np.all(g[:, -1] == border)

    interior = g[1:-1, 1:-1]
    candidates = []
    for color in map(int, np.unique(interior)):
        seed_mask = interior == color
        if not _is_connected(seed_mask):
            continue

        parity_colors = [set(), set()]
        ok = True
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                cell = int(g[r, c])
                if cell == color:
                    continue
                parity_colors[(r + c) & 1].add(cell)
                if len(parity_colors[(r + c) & 1]) > 1:
                    ok = False
                    break
            if not ok:
                break

        if ok and all(len(colors) == 1 for colors in parity_colors):
            even_color = next(iter(parity_colors[0]))
            odd_color = next(iter(parity_colors[1]))
            if even_color != odd_color and even_color != border and odd_color != border:
                candidates.append(color)

    assert len(candidates) == 1, "expected one connected seed color over a two-color checkerboard"
    return int(candidates[0])


def _seed_cells_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _expansion_targets(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for r, c in cells:
        neighbors = []
        for dr, dc in ORTH_DIRS:
            if (r + dr, c + dc) in cells:
                neighbors.append((dr, dc))

        if len(neighbors) == 1:
            dr, dc = neighbors[0]
            targets.add((r - dr, c - dc))
        elif len(neighbors) == 2:
            (dr1, dc1), (dr2, dc2) = neighbors
            if dr1 == -dr2 and dc1 == -dc2:
                continue
            targets.add((r - dr1 - dr2, c - dc1 - dc2))

    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the seed by one topology-dependent step on the checkerboard without overwriting the border.

    Input structure:
    - A one-cell border of a single color surrounds the grid.
    - Inside the border, two colors form a pixel-by-pixel checkerboard background.
    - One connected foreground seed overlays some checkerboard cells and is the only interior color that does not follow the checkerboard.

    Rule:
    - For each seed cell, inspect its four orthogonal seed neighbors in the original input.
    - If it has exactly one seed neighbor, extend one cell in the opposite direction; if it has two opposite neighbors, do nothing; if it has two orthogonal neighbors, paint the outer diagonal opposite those neighbors.
    - Cells with three or four seed neighbors do nothing, and any attempted growth that would hit the surrounding border is skipped.

    Key insight:
    - The growth direction is determined locally by the seed cell's role in the shape: endpoints grow outward, corners grow on the outside diagonal, and straight or crowded cells stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    h, w = g.shape

    seed_color = _find_seed_color(g)
    seed_mask = g == seed_color
    seed_cells = _seed_cells_from_mask(seed_mask)
    out = g.copy()

    for r, c in _expansion_targets(seed_cells):
        if 0 < r < h - 1 and 0 < c < w - 1:
            out[r, c] = seed_color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    transformed = set(cells)
    for _ in range(random.randrange(4)):
        transformed = {(-c, r) for r, c in transformed}
    if random.random() < 0.5:
        transformed = {(r, -c) for r, c in transformed}
    return _normalize_cells(transformed)


def _shape_stats(cells: set[tuple[int, int]]) -> dict[str, int]:
    stats = {"endpoints": 0, "straight": 0, "corners": 0, "deg3": 0, "deg4": 0}
    for r, c in cells:
        neighbors = []
        for dr, dc in ORTH_DIRS:
            if (r + dr, c + dc) in cells:
                neighbors.append((dr, dc))

        degree = len(neighbors)
        if degree == 1:
            stats["endpoints"] += 1
        elif degree == 2:
            (dr1, dc1), (dr2, dc2) = neighbors
            if dr1 == -dr2 and dc1 == -dc2:
                stats["straight"] += 1
            else:
                stats["corners"] += 1
        elif degree == 3:
            stats["deg3"] += 1
        elif degree == 4:
            stats["deg4"] += 1
    return stats


def _make_straight_shape() -> set[tuple[int, int]]:
    length = random.randint(4, 8)
    return {(0, c) for c in range(length)}


def _make_corner_shape() -> set[tuple[int, int]]:
    arm_a = random.randint(3, 6)
    arm_b = random.randint(3, 6)
    cells = {(0, c) for c in range(arm_a)}
    cells |= {(r, arm_a - 1) for r in range(arm_b)}
    if random.random() < 0.6:
        arm_c = random.randint(2, 4)
        cells |= {(arm_b - 1, arm_a - 1 + c) for c in range(arm_c)}
    return _normalize_cells(cells)


def _make_tee_shape() -> set[tuple[int, int]]:
    trunk = random.randint(5, 8)
    branch_pos = random.randint(1, trunk - 3)
    branch_len = random.randint(2, 4)
    tail_len = random.randint(2, 4)
    branch_side = random.choice((-1, 1))
    tail_side = random.choice((-1, 1))

    cells = {(0, c) for c in range(trunk)}
    cells |= {(branch_side * step, branch_pos) for step in range(1, branch_len + 1)}
    cells |= {(tail_side * step, trunk - 1) for step in range(1, tail_len + 1)}
    return _normalize_cells(cells)


def _make_cross_shape() -> set[tuple[int, int]]:
    arms = {
        "up": random.randint(1, 3),
        "down": random.randint(1, 3),
        "left": random.randint(1, 3),
        "right": random.randint(1, 3),
    }
    if max(arms.values()) == 1:
        arms[random.choice(tuple(arms))] = 2

    cells = {(0, 0)}
    cells |= {(-step, 0) for step in range(1, arms["up"] + 1)}
    cells |= {(step, 0) for step in range(1, arms["down"] + 1)}
    cells |= {(0, -step) for step in range(1, arms["left"] + 1)}
    cells |= {(0, step) for step in range(1, arms["right"] + 1)}

    bend_arm = random.choice(("up", "down", "left", "right"))
    bend_len = random.randint(1, 3)
    if bend_arm == "up":
        endpoint = (-arms["up"], 0)
        bend_dir = random.choice(((0, -1), (0, 1)))
    elif bend_arm == "down":
        endpoint = (arms["down"], 0)
        bend_dir = random.choice(((0, -1), (0, 1)))
    elif bend_arm == "left":
        endpoint = (0, -arms["left"])
        bend_dir = random.choice(((-1, 0), (1, 0)))
    else:
        endpoint = (0, arms["right"])
        bend_dir = random.choice(((-1, 0), (1, 0)))

    dr, dc = bend_dir
    for step in range(1, bend_len + 1):
        cells.add((endpoint[0] + dr * step, endpoint[1] + dc * step))
    return _normalize_cells(cells)


def _make_shape(scenario: str) -> set[tuple[int, int]]:
    if scenario == "straight":
        cells = _make_straight_shape()
    elif scenario == "corner":
        cells = _make_corner_shape()
    elif scenario == "tee":
        cells = _make_tee_shape()
    else:
        cells = _make_cross_shape()

    cells = _transform_cells(cells)
    stats = _shape_stats(cells)
    if scenario == "straight":
        assert stats["endpoints"] >= 2 and stats["straight"] >= 1
    elif scenario == "corner":
        assert stats["corners"] >= 1 and stats["straight"] >= 1
    elif scenario == "tee":
        assert stats["deg3"] >= 1 and stats["corners"] >= 1
    else:
        assert stats["deg4"] >= 1 and stats["corners"] >= 1
    return cells


def _bbox_shape(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _place_shape(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    want_blocked: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    shape_h, shape_w = _bbox_shape(cells)
    feasible = []
    blocked_feasible = []

    for top in range(1, h - shape_h):
        for left in range(1, w - shape_w):
            placed = {(r + top, c + left) for r, c in cells}
            targets = _expansion_targets(placed)
            interior_targets = {
                (r, c)
                for r, c in targets
                if 0 < r < h - 1 and 0 < c < w - 1 and (r, c) not in placed
            }
            if not interior_targets:
                continue

            blocked = any(not (0 < r < h - 1 and 0 < c < w - 1) for r, c in targets)
            item = (placed, interior_targets)
            feasible.append(item)
            if blocked:
                blocked_feasible.append(item)

    if want_blocked and blocked_feasible:
        return random.choice(blocked_feasible)
    if feasible:
        return random.choice(feasible)
    return None


def _render_checkerboard(
    h: int,
    w: int,
    border_color: int,
    color_a: int,
    color_b: int,
    phase: int,
) -> np.ndarray:
    grid = np.full((h, w), border_color, dtype=int)
    rows, cols = np.indices((h - 2, w - 2))
    parity = (rows + cols + phase) & 1
    grid[1:-1, 1:-1] = np.where(parity == 0, color_a, color_b)
    return grid


def _expected_output(
    inp: np.ndarray,
    seed_color: int,
    targets: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in targets:
        out[r, c] = seed_color
    return out


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS)
    shape = _make_shape(scenario)
    shape_h, shape_w = _bbox_shape(shape)

    h = random.randint(max(10, shape_h + 3), min(30, shape_h + 18))
    w = random.randint(max(10, shape_w + 3), min(30, shape_w + 18))
    placement = _place_shape(shape, h, w, want_blocked=random.random() < 0.4)
    if placement is None:
        return None
    placed_cells, targets = placement

    border_color, checker_a, checker_b, seed_color = random.sample(ARC_COLORS, 4)
    phase = random.randint(0, 1)
    inp = _render_checkerboard(h, w, border_color, checker_a, checker_b, phase)
    for r, c in placed_cells:
        inp[r, c] = seed_color

    stats = _shape_stats(placed_cells)
    if scenario == "straight" and stats["corners"] != 0:
        return None
    if scenario == "corner" and stats["corners"] == 0:
        return None
    if scenario == "tee" and stats["deg3"] == 0:
        return None
    if scenario == "cross" and stats["deg4"] == 0:
        return None

    expected = _expected_output(inp, seed_color, targets)
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, solved):
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard growth puzzle")
