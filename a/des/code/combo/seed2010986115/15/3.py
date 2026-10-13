import random
from collections import deque

import numpy as np


BG = 0
RED = 2
YELLOW = 4
TRACK_COLORS = [1, 3, 5, 6, 7, 8, 9]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _flood_outside(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if _in_bounds(blocked.shape, nr, nc) and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _loop_interior(loop_mask: np.ndarray) -> np.ndarray:
    outside = _flood_outside(loop_mask)
    return (~loop_mask) & (~outside)


def _extract_loop(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    allowed = {int(color), RED}
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) in allowed:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _is_simple_cycle(loop_cells: set[tuple[int, int]]) -> bool:
    for r, c in loop_cells:
        deg = sum((nr, nc) in loop_cells for nr, nc in neighbors4(r, c))
        if deg != 2:
            return False
    return True


def _trace_cycle(loop_cells: set[tuple[int, int]], start: tuple[int, int], first_step: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev = start
    cur = first_step
    while True:
        order.append(cur)
        if cur == start:
            break
        nxts = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in loop_cells and (nr, nc) != prev]
        assert len(nxts) == 1, "loop traversal should have a unique continuation"
        prev, cur = cur, nxts[0]
    order = order[:-1]
    assert len(order) == len(loop_cells), "traced order must cover the whole loop"
    return order


def _orientation_score(order: list[tuple[int, int]], interior: np.ndarray) -> int:
    h, w = interior.shape
    score = 0
    n = len(order)
    for i in range(n):
        r0, c0 = order[i]
        r1, c1 = order[(i + 1) % n]
        dr, dc = r1 - r0, c1 - c0
        right = (dc, -dr)
        left = (-dc, dr)
        for rr, cc in ((r0 + right[0], c0 + right[1]), (r1 + right[0], c1 + right[1])):
            if 0 <= rr < h and 0 <= cc < w and interior[rr, cc]:
                score += 1
        for rr, cc in ((r0 + left[0], c0 + left[1]), (r1 + left[0], c1 + left[1])):
            if 0 <= rr < h and 0 <= cc < w and interior[rr, cc]:
                score -= 1
    return score


def _clockwise_order(loop_cells: set[tuple[int, int]], start: tuple[int, int], interior: np.ndarray) -> list[tuple[int, int]]:
    neighs = [(nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in loop_cells]
    assert len(neighs) == 2, "marker cell must lie on a degree-2 loop"
    order_a = _trace_cycle(loop_cells, start, neighs[0])
    order_b = [order_a[0]] + list(reversed(order_a[1:]))
    score_a = _orientation_score(order_a, interior)
    score_b = _orientation_score(order_b, interior)
    assert score_a != score_b, "clockwise direction must be distinguishable"
    return order_a if score_a > score_b else order_b


def _detect_track_color(grid: np.ndarray, red_pos: tuple[int, int]) -> int:
    r, c = red_pos
    candidates = {
        int(grid[nr, nc])
        for nr, nc in neighbors4(r, c)
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in {BG, RED, YELLOW}
    }
    matches = []
    for color in candidates:
        loop_cells = _extract_loop(grid, red_pos, color)
        if sum(int(grid[rr, cc]) == RED for rr, cc in loop_cells) != 1:
            continue
        if not _is_simple_cycle(loop_cells):
            continue
        matches.append(color)
    assert len(matches) == 1, "each red marker must correspond to exactly one loop color"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red marker clockwise around its enclosing 4-connected loop by the
    number of yellow pixels inside that loop, then erase the yellow pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    red_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert red_positions, "expected at least one red marker"

    for red_pos in red_positions:
        track_color = _detect_track_color(g, red_pos)
        loop_cells = _extract_loop(g, red_pos, track_color)

        loop_mask = np.zeros_like(g, dtype=bool)
        for r, c in loop_cells:
            loop_mask[r, c] = True
        interior = _loop_interior(loop_mask)

        yellow_count = int(np.sum((g == YELLOW) & interior))
        order = _clockwise_order(loop_cells, red_pos, interior)
        dst = order[yellow_count % len(order)]

        for r, c in np.argwhere(interior & (g == YELLOW)):
            out[int(r), int(c)] = BG
        out[red_pos] = track_color
        out[dst] = RED

    return out


GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
OBJECT_COUNT_WEIGHTS = [0.22, 0.32, 0.28, 0.18]
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
TARGET_BUMPS_RANGE = (0, 7)
SHAPE_BUILD_TRIES = 60
PLACEMENT_TRIES = 80
GEN_ATTEMPTS = 300
MIN_GAP = 2


def _rectangle_cycle(h: int, w: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for c in range(w):
        order.append((0, c))
    for r in range(1, h):
        order.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        order.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        order.append((r, 0))
    return order


def _normalize_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in order)
    min_c = min(c for _, c in order)
    return [(r - min_r, c - min_c) for r, c in order]


def _valid_cycle_order(order: list[tuple[int, int]]) -> bool:
    if len(set(order)) != len(order):
        return False
    cell_set = set(order)
    n = len(order)
    for i in range(n):
        r0, c0 = order[i]
        r1, c1 = order[(i + 1) % n]
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            return False
    return _is_simple_cycle(cell_set)


def _expand_once(order: list[tuple[int, int]], edge_idx: int) -> list[tuple[int, int]] | None:
    n = len(order)
    a = order[edge_idx]
    b = order[(edge_idx + 1) % n]
    dr, dc = b[0] - a[0], b[1] - a[1]
    outward = (-dc, dr)
    u = (a[0] + outward[0], a[1] + outward[1])
    v = (b[0] + outward[0], b[1] + outward[1])
    if u in order or v in order:
        return None

    if edge_idx == n - 1:
        candidate = order + [u, v]
    else:
        candidate = order[: edge_idx + 1] + [u, v] + order[edge_idx + 1 :]
    if not _valid_cycle_order(candidate):
        return None
    return candidate


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    k = rng.randrange(4)
    return np.rot90(mask, k)


def _build_loop_mask(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_BUILD_TRIES):
        h = rng.randint(*RECT_H_RANGE)
        w = rng.randint(*RECT_W_RANGE)
        order = _rectangle_cycle(h, w)
        target_bumps = rng.randint(*TARGET_BUMPS_RANGE)
        for _ in range(target_bumps):
            indices = list(range(len(order)))
            rng.shuffle(indices)
            for edge_idx in indices:
                nxt = _expand_once(order, edge_idx)
                if nxt is not None:
                    order = nxt
                    break
        order = _normalize_order(order)
        mask = np.zeros(
            (
                max(r for r, _ in order) + 1,
                max(c for _, c in order) + 1,
            ),
            dtype=bool,
        )
        for r, c in order:
            mask[r, c] = True
        mask = _random_transform(mask, rng)

        padded = np.pad(mask, 1, constant_values=False)
        interior = _loop_interior(padded)[1:-1, 1:-1]
        interior_count = int(np.sum(interior))
        if interior_count == 0:
            continue
        return mask
    raise RuntimeError("failed to build a valid loop mask")


def _choose_yellow_count(rng: random.Random, interior_count: int, loop_len: int) -> int:
    nonzero = [k for k in range(1, interior_count + 1) if k % loop_len != 0]
    wrapped = [k for k in range(loop_len + 1, interior_count + 1) if k % loop_len != 0]
    roll = rng.random()
    if roll < 0.12:
        return 0
    if wrapped and roll < 0.42:
        return rng.choice(wrapped)
    if nonzero:
        return rng.choice(nonzero)
    return rng.randint(0, interior_count)


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if ph + 2 * MIN_GAP > gh or pw + 2 * MIN_GAP > gw:
        return False

    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(1, gh - ph - 1)
        c0 = rng.randint(1, gw - pw - 1)
        rs = slice(max(0, r0 - MIN_GAP), min(gh, r0 + ph + MIN_GAP))
        cs = slice(max(0, c0 - MIN_GAP), min(gw, c0 + pw + MIN_GAP))
        if np.any(reserved[rs, cs]):
            continue
        mask = patch != BG
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(mask & (sub != BG)):
            continue
        sub[mask] = patch[mask]
        grid[r0 : r0 + ph, c0 : c0 + pw] = sub
        reserved[rs, cs] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_objects = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    colors = rng.sample(TRACK_COLORS, n_objects)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    yellow_total = 0

    for obj_idx, color in enumerate(colors):
        mask = _build_loop_mask(rng)
        loop_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        loop_len = len(loop_cells)

        padded = np.pad(mask, 1, constant_values=False)
        interior = _loop_interior(padded)[1:-1, 1:-1]
        interior_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        if not interior_cells:
            return None

        yellow_count = _choose_yellow_count(rng, len(interior_cells), loop_len)
        if obj_idx == n_objects - 1 and yellow_total == 0 and yellow_count == 0:
            yellow_count = min(1, len(interior_cells))
        yellow_total += yellow_count

        marker = rng.choice(loop_cells)
        yellow_cells = set(rng.sample(interior_cells, yellow_count)) if yellow_count else set()

        patch = np.full(mask.shape, BG, dtype=int)
        for r, c in loop_cells:
            patch[r, c] = color
        for r, c in yellow_cells:
            patch[r, c] = YELLOW
        patch[marker] = RED

        if not _place_patch(grid, reserved, patch, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.sum(grid == YELLOW)) == 0:
        return None
    if int(np.sum(out == YELLOW)) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
