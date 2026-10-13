import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _interior_mask(path_mask: np.ndarray) -> np.ndarray:
    h, w = path_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if path_mask[r, c] or outside[r, c]:
                continue
            outside[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if path_mask[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            q.append((nr, nc))

    return (~path_mask) & (~outside)


def _trace_cycle(path_set: set[tuple[int, int]], start: tuple[int, int], first_step: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev = start
    cur = first_step

    while cur != start:
        order.append(cur)
        nxts = [p for p in _neighbors4(*cur) if p in path_set and p != prev]
        assert len(nxts) == 1, "cycle tracing expects degree-2 path pixels"
        prev, cur = cur, nxts[0]

    return order


def _clockwise_score(order: list[tuple[int, int]], inside_mask: np.ndarray) -> int:
    score = 0
    n = len(order)
    for i, (r, c) in enumerate(order):
        nr, nc = order[(i + 1) % n]
        dr, dc = nr - r, nc - c
        right = (r + dc, c - dr)
        left = (r - dc, c + dr)
        if 0 <= right[0] < inside_mask.shape[0] and 0 <= right[1] < inside_mask.shape[1] and inside_mask[right]:
            score += 1
        if 0 <= left[0] < inside_mask.shape[0] and 0 <= left[1] < inside_mask.shape[1] and inside_mask[left]:
            score -= 1
    return score


def _clockwise_order(path_mask: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in np.argwhere(path_mask)]
    assert pts, "expected a non-empty path"
    path_set = set(pts)

    for r, c in pts:
        deg = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c))
        assert deg == 2, "every path pixel must have exactly two path neighbors"

    start = min(pts)
    nbrs = [p for p in _neighbors4(*start) if p in path_set]
    assert len(nbrs) == 2, "start pixel must have degree 2"

    inside_mask = _interior_mask(path_mask)
    assert inside_mask.any(), "path must enclose a non-empty interior"

    cand0 = _trace_cycle(path_set, start, nbrs[0])
    cand1 = _trace_cycle(path_set, start, nbrs[1])
    score0 = _clockwise_score(cand0, inside_mask)
    score1 = _clockwise_score(cand1, inside_mask)
    if score1 > score0:
        return cand1
    return cand0


def _seed_color_at(grid: np.ndarray, r: int, c: int) -> int | None:
    colors = {
        int(grid[nr, nc])
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < grid.shape[0]
        and 0 <= nc < grid.shape[1]
        and int(grid[nr, nc]) not in (BG, PATH)
    }
    if not colors:
        return None
    assert len(colors) == 1, "each path pixel should be adjacent to at most one seed color"
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a grey simple cycle by walking it clockwise and switching the active
    color whenever the walk reaches a seed-adjacent path pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path_mask = g == PATH
    order = _clockwise_order(path_mask)

    out = g.copy()
    active = None
    for r, c in order:
        seed_color = _seed_color_at(g, r, c)
        if seed_color is not None:
            active = seed_color
        out[r, c] = PATH if active is None else active
    return out


def _rectangle_cycle(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    order = []
    for c in range(left, right + 1):
        order.append((top, c))
    for r in range(top + 1, bottom + 1):
        order.append((r, right))
    for c in range(right - 1, left - 1, -1):
        order.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        order.append((r, left))
    return order


def _order_is_valid(order: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(order) < 4 or len(set(order)) != len(order):
        return False

    path_set = set(order)
    for r, c in order:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c)) != 2:
            return False

    for a, b in zip(order, order[1:] + order[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False

    path_mask = np.zeros((h, w), dtype=bool)
    for r, c in order:
        path_mask[r, c] = True
    return bool(_interior_mask(path_mask).any())


def _try_bump(
    order: list[tuple[int, int]],
    h: int,
    w: int,
    idx: int,
    side: str,
) -> list[tuple[int, int]] | None:
    n = len(order)
    if not (1 <= idx < n - 1):
        return None

    a = order[idx - 1]
    b = order[idx]
    c = order[idx + 1]
    d1 = (b[0] - a[0], b[1] - a[1])
    d2 = (c[0] - b[0], c[1] - b[1])
    if d1 != d2:
        return None

    dr, dc = d1
    offset = (-dc, dr) if side == "left" else (dc, -dr)
    inserted = [
        (a[0] + offset[0], a[1] + offset[1]),
        (b[0] + offset[0], b[1] + offset[1]),
        (c[0] + offset[0], c[1] + offset[1]),
    ]
    if any(p in set(order) for p in inserted):
        return None

    cand = order[:idx] + inserted + order[idx + 1 :]
    if _order_is_valid(cand, h, w):
        return cand
    return None


def _seed_candidates(
    order: list[tuple[int, int]],
    h: int,
    w: int,
) -> dict[tuple[int, int], dict[str, list[tuple[int, int]]]]:
    path_set = set(order)
    path_mask = np.zeros((h, w), dtype=bool)
    for r, c in order:
        path_mask[r, c] = True
    inside_mask = _interior_mask(path_mask)

    candidates: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            if (r, c) in path_set:
                continue
            adj = [p for p in _neighbors4(r, c) if p in path_set]
            if len(adj) != 1:
                continue
            key = adj[0]
            side = "inside" if inside_mask[r, c] else "outside"
            if key not in candidates:
                candidates[key] = {"inside": [], "outside": []}
            candidates[key][side].append((r, c))
    return candidates


def _cyclic_distance(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _select_seed_pixels(
    order: list[tuple[int, int]],
    candidates: dict[tuple[int, int], dict[str, list[tuple[int, int]]]],
    n_seeds: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    eligible = [i for i, p in enumerate(order) if p in candidates]
    rng.shuffle(eligible)

    selected_idx: list[int] = []
    for gap in [max(2, len(order) // max(6, 2 * n_seeds)), 3, 2, 1, 0]:
        selected_idx = []
        for idx in eligible:
            if all(_cyclic_distance(idx, prev, len(order)) > gap for prev in selected_idx):
                selected_idx.append(idx)
                if len(selected_idx) == n_seeds:
                    break
        if len(selected_idx) == n_seeds:
            break

    if len(selected_idx) < n_seeds:
        return []

    mode = rng.choices(["mixed", "inside", "outside"], weights=[4, 1, 1])[0]
    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used_sides: list[str] = []

    for idx in selected_idx:
        path_px = order[idx]
        opts = candidates[path_px]
        available = [side for side in ("inside", "outside") if opts[side]]
        if not available:
            return []

        if mode in available:
            side = mode
        else:
            side = rng.choice(available)

        placements.append((path_px, rng.choice(opts[side])))
        used_sides.append(side)

    if mode == "mixed" and len(set(used_sides)) == 1:
        target = "outside" if used_sides[0] == "inside" else "inside"
        for i, (path_px, _seed_pos) in enumerate(placements):
            if candidates[path_px][target]:
                placements[i] = (path_px, rng.choice(candidates[path_px][target]))
                break

    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 22)
    w = rng.randint(12, 22)

    top = rng.randint(2, h - 8)
    left = rng.randint(2, w - 8)
    bottom = rng.randint(top + 4, h - 3)
    right = rng.randint(left + 4, w - 3)
    order = _rectangle_cycle(top, left, bottom, right)

    bump_target = rng.randint(0, min(10, (h + w) // 3))
    for _ in range(bump_target):
        for _ in range(40):
            idx = rng.randint(1, len(order) - 2)
            cand = _try_bump(order, h, w, idx, rng.choice(["left", "right"]))
            if cand is not None:
                order = cand
                break

    candidates = _seed_candidates(order, h, w)
    eligible_pixels = [p for p, sides in candidates.items() if sides["inside"] or sides["outside"]]
    if len(eligible_pixels) < 2:
        return None

    max_seeds = min(5, len(eligible_pixels))
    n_seeds = rng.randint(2, max_seeds)
    placements = _select_seed_pixels(order, candidates, n_seeds, rng)
    if len(placements) != n_seeds:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in order:
        grid[r, c] = PATH

    colors = rng.sample(SEED_COLORS, n_seeds)
    for color, (_path_px, (r, c)) in zip(colors, placements):
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    non_path = grid != PATH
    if not np.array_equal(out[non_path], grid[non_path]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
