import random
from collections import Counter, deque

import numpy as np


BG = 4
PALETTE = [c for c in range(10) if c != BG]
BRANCHES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _transform_component_in_place(out: np.ndarray, comp: dict, do_vflip: bool, do_hflip: bool, bg: int) -> None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        out[r, c] = bg
        local[r - r0, c - c0] = True
    if do_vflip:
        local = np.flipud(local)
    if do_hflip:
        local = np.fliplr(local)
    color = int(comp["color"])
    for rr, cc in np.argwhere(local):
        out[r0 + int(rr), c0 + int(cc)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each object according to the row/column-distance parity to its same-color partner.
    Even row distance -> vertical flip. Even column distance -> horizontal flip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors

    pairs = []
    for color in colors:
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            assert (r1 - r0 + 1) % 2 == 1
            assert (c1 - c0 + 1) % 2 == 1
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp["coords"]:
                mask[r - r0, c - c0] = True
            assert not np.array_equal(mask, np.flipud(mask))
            assert not np.array_equal(mask, np.fliplr(mask))
        pairs.append((color, comps))

    out = g.copy()
    for _color, comps in pairs:
        a, b = comps
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        dr_num = abs((ar0 + ar1) - (br0 + br1))
        dc_num = abs((ac0 + ac1) - (bc0 + bc1))
        assert dr_num % 2 == 0
        assert dc_num % 2 == 0
        dr = dr_num // 2
        dc = dc_num // 2
        do_vflip = dr % 2 == 0
        do_hflip = dc % 2 == 0
        _transform_component_in_place(out, a, do_vflip, do_hflip, bg)
        _transform_component_in_place(out, b, do_vflip, do_hflip, bg)

    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _make_shape(rng: random.Random) -> np.ndarray:
    dims = [(3, 3), (3, 5), (5, 3), (5, 5), (5, 7), (7, 5)]
    for _ in range(200):
        h, w = rng.choice(dims)
        target = rng.randint(max(4, (h * w) // 3), h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        frontier = [start]
        mask[start] = True
        while int(mask.sum()) < target and frontier:
            base = frontier[rng.randrange(len(frontier))]
            nbrs = []
            for nr, nc in _neighbors4(base[0], base[1]):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove(base)
                continue
            cell = rng.choice(nbrs)
            mask[cell] = True
            frontier.append(cell)
        if not _connected(mask):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric shape")


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 1 or c0 < 1 or r1 >= H - 1 or c1 >= W - 1:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r1 + 2)
    cc1 = min(W, c1 + 2)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paint_shape(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = color


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _choose_pair_positions(
    rng: random.Random,
    occupied: np.ndarray,
    h: int,
    w: int,
    row_even: bool,
    col_even: bool,
) -> tuple[int, int, int, int] | None:
    H, W = occupied.shape
    rows = list(range(1, H - h))
    cols = list(range(1, W - w))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for _ in range(300):
        r1 = rng.choice(rows)
        c1 = rng.choice(cols)
        if not _can_place(occupied, r1, c1, h, w):
            continue
        candidates = []
        for r2 in rows:
            if (abs(r2 - r1) % 2 == 0) != row_even:
                continue
            for c2 in cols:
                if (abs(c2 - c1) % 2 == 0) != col_even:
                    continue
                if abs(r2 - r1) + abs(c2 - c1) < 4:
                    continue
                if not _can_place(occupied, r2, c2, h, w):
                    continue
                if max(abs(r2 - r1), abs(c2 - c1)) < max(h, w) + 1:
                    continue
                candidates.append((r2, c2))
        if candidates:
            r2, c2 = rng.choice(candidates)
            return r1, c1, r2, c2
    return None


def _sample_branch_multiset(rng: random.Random, num_pairs: int) -> list[tuple[bool, bool]]:
    order = list(BRANCHES)
    rng.shuffle(order)
    if num_pairs <= 4:
        chosen = order[:num_pairs]
    else:
        chosen = order[:]
        while len(chosen) < num_pairs:
            chosen.append(rng.choice(BRANCHES))
    if all(branch == (False, False) for branch in chosen):
        chosen[0] = rng.choice(BRANCHES[1:])
    rng.shuffle(chosen)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(15, 23)
    W = rng.randint(15, 23)
    num_pairs = rng.randint(2, 5)
    if min(H, W) < 16 and num_pairs > 3:
        num_pairs = 3

    colors = rng.sample(PALETTE, num_pairs)
    branches = _sample_branch_multiset(rng, num_pairs)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for color, (row_even, col_even) in zip(colors, branches):
        shape = _make_shape(rng)
        h, w = shape.shape
        pos = _choose_pair_positions(rng, occupied, h, w, row_even, col_even)
        if pos is None:
            return None
        r1, c1, r2, c2 = pos
        _paint_shape(grid, shape, r1, c1, color)
        _paint_shape(grid, shape, r2, c2, color)
        _mark_occupied(occupied, r1, c1, h, w)
        _mark_occupied(occupied, r2, c2, h, w)

    fg = grid[grid != BG]
    if len(fg) == 0:
        return None
    color_counts = Counter(int(v) for v in fg)
    if any(count < 8 for count in color_counts.values()):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
