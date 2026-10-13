import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

SYM_H_ONLY = (True, False)
SYM_V_ONLY = (False, True)
SYM_BOTH = (True, True)
SYM_NONE = (False, False)
ALL_SYMMETRIES = (SYM_H_ONLY, SYM_V_ONLY, SYM_BOTH, SYM_NONE)

FILL_BY_SYMMETRY = {
    SYM_H_ONLY: BLUE,
    SYM_V_ONLY: GREEN,
    SYM_BOTH: YELLOW,
    SYM_NONE: ORANGE,
}

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 300
ROOM_PLACE_ATTEMPTS = 250
SEED_ATTEMPTS = 250

LEGEND_PATTERNS = {
    SYM_H_ONLY: (
        np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
    ),
    SYM_V_ONLY: (
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
        np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    ),
    SYM_BOTH: (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
    ),
    SYM_NONE: (
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    ),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("tight mask requires at least one true cell")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _classify_symmetry(mask: np.ndarray) -> tuple[bool, bool]:
    tight = _tight_mask(mask.astype(bool))
    return bool(np.array_equal(tight, np.flipud(tight))), bool(np.array_equal(tight, np.fliplr(tight)))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    if not bg.any():
        return False
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool((bg & ~seen).any())


def _read_legend(grid: np.ndarray) -> dict[tuple[bool, bool], int]:
    legend_mask = ~np.isin(grid, [BLACK, GREY, RED])
    mapping: dict[tuple[bool, bool], int] = {}
    for comp in _components(legend_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 != 3 or c1 - c0 + 1 != 3 or len(comp) != 9:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch)}
        if MAROON not in colors or len(colors) != 2:
            continue
        fill = next(color for color in colors if color != MAROON)
        mapping[_classify_symmetry(patch == MAROON)] = int(fill)
    if not mapping:
        raise ValueError("legend not found")
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend = _read_legend(g)
    out = g.copy()

    region_mask = (g == BLACK) | (g == RED)
    for comp in _components(region_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        red_cells = [(r, c) for r, c in comp if g[r, c] == RED]
        if not red_cells:
            continue
        sr0, sc0, sr1, sc1 = _bbox(red_cells)
        seed_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in red_cells:
            seed_mask[r - sr0, c - sc0] = True
        symmetry = _classify_symmetry(seed_mask)
        fill = legend[symmetry]
        for r, c in comp:
            if g[r, c] == BLACK:
                out[r, c] = fill
    return out


def _orbit_cells(h: int, w: int, symmetry: tuple[bool, bool], r: int, c: int) -> list[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if symmetry[0]:
        rows.add(h - 1 - r)
    if symmetry[1]:
        cols.add(w - 1 - c)
    return [(rr, cc) for rr in rows for cc in cols]


def _random_seed_mask(symmetry: tuple[bool, bool], rng: random.Random) -> np.ndarray:
    for _ in range(SEED_ATTEMPTS):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        target_area = rng.randint(max(4, (h * w) // 3), max(4, (2 * h * w) // 3))

        if symmetry == SYM_NONE:
            start = (rng.randrange(h), rng.randrange(w))
            mask[start] = True
            frontier = set()
            for dr, dc in NEIGHBORS4:
                nr, nc = start[0] + dr, start[1] + dc
                if 0 <= nr < h and 0 <= nc < w:
                    frontier.add((nr, nc))
            while mask.sum() < target_area and frontier:
                cell = rng.choice(tuple(frontier))
                frontier.remove(cell)
                mask[cell] = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = cell[0] + dr, cell[1] + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if mask.sum() < 4:
                continue
        else:
            if symmetry == SYM_H_ONLY:
                independent = [(r, c) for r in range((h + 1) // 2) for c in range(w)]
            elif symmetry == SYM_V_ONLY:
                independent = [(r, c) for r in range(h) for c in range((w + 1) // 2)]
            else:
                independent = [(r, c) for r in range((h + 1) // 2) for c in range((w + 1) // 2)]
            rng.shuffle(independent)
            anchor = independent[0]
            for rr, cc in _orbit_cells(h, w, symmetry, *anchor):
                mask[rr, cc] = True
            for r, c in independent[1:]:
                if mask.sum() >= target_area:
                    break
                orbit = _orbit_cells(h, w, symmetry, r, c)
                if all(mask[rr, cc] for rr, cc in orbit):
                    continue
                if not any(
                    0 <= rr + dr < h and 0 <= cc + dc < w and mask[rr + dr, cc + dc]
                    for rr, cc in orbit
                    for dr, dc in NEIGHBORS4
                ):
                    continue
                if rng.random() < 0.7:
                    for rr, cc in orbit:
                        mask[rr, cc] = True

        if not mask.any():
            continue
        tight = _tight_mask(mask)
        if tight.shape[0] < 3 or tight.shape[1] < 3 or tight.shape[0] > 5 or tight.shape[1] > 5:
            continue
        if int(tight.sum()) < 4 or int(tight.sum()) >= tight.size:
            continue
        if not _is_connected(tight):
            continue
        if _has_hole(tight):
            continue
        if _classify_symmetry(tight) != symmetry:
            continue
        return tight.astype(bool)
    raise RuntimeError("failed to build a seed mask")


def _legend_tile(symmetry: tuple[bool, bool], rng: random.Random) -> np.ndarray:
    pattern = rng.choice(LEGEND_PATTERNS[symmetry])
    tile = np.full((3, 3), FILL_BY_SYMMETRY[symmetry], dtype=int)
    tile[pattern] = MAROON
    return tile


def _legend_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 7
    if corner == "bl":
        return h - 7, 0
    return h - 7, w - 7


def _place_legend(grid: np.ndarray, corner: str, rng: random.Random) -> None:
    r0, c0 = _legend_origin(*grid.shape, corner)
    positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
    classes = list(ALL_SYMMETRIES)
    rng.shuffle(classes)
    for symmetry, (dr, dc) in zip(classes, positions):
        tile = _legend_tile(symmetry, rng)
        grid[r0 + dr : r0 + dr + 3, c0 + dc : c0 + dc + 3] = tile


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    if mask[rr0:rr1, cc0:cc1].any():
        return False
    mask[rr0:rr1, cc0:cc1] = True
    return True


def _draw_room(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = GREY
    grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = BLACK


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(30, 38)
    w = rng.randint(32, 42)
    if rng.random() < 0.5:
        h, w = w, h
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_corner = rng.choice(("tl", "tr", "bl", "br"))
    lr0, lc0 = _legend_origin(h, w, legend_corner)
    if not _reserve(occupied, lr0, lc0, lr0 + 6, lc0 + 6):
        return None

    room_classes = list(ALL_SYMMETRIES)
    extra_rooms = rng.randint(0, 2)
    room_classes.extend(rng.choice(ALL_SYMMETRIES) for _ in range(extra_rooms))
    rng.shuffle(room_classes)
    rooms: list[tuple[int, int, int, int, tuple[bool, bool]]] = []

    for symmetry in room_classes:
        seed = _random_seed_mask(symmetry, rng)
        sh, sw = seed.shape
        inner_h = rng.randint(max(7, sh + 2), 10)
        inner_w = rng.randint(max(7, sw + 2), 10)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        placed = False
        for _ in range(ROOM_PLACE_ATTEMPTS):
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            if not _reserve(occupied, r0, c0, r0 + outer_h - 1, c0 + outer_w - 1):
                continue
            _draw_room(grid, r0, c0, inner_h, inner_w)
            sr = r0 + 1 + rng.randint(1, inner_h - sh - 1)
            sc = c0 + 1 + rng.randint(1, inner_w - sw - 1)
            patch = grid[sr : sr + sh, sc : sc + sw]
            if np.any(patch != BLACK):
                return None
            patch[seed] = RED
            rooms.append((r0, c0, inner_h, inner_w, symmetry))
            placed = True
            break
        if not placed:
            return None

    _place_legend(grid, legend_corner, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    seen = set()
    for r0, c0, inner_h, inner_w, symmetry in rooms:
        inside = grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
        transformed = out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
        fill = FILL_BY_SYMMETRY[symmetry]
        if not np.all(transformed[inside == BLACK] == fill):
            return None
        if not np.all(transformed[inside == RED] == RED):
            return None
        seen.add(symmetry)
    if seen != set(ALL_SYMMETRIES):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-fill puzzle")
