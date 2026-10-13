from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

# For a 2x2 tower at (r, c), the side-midpoint lies between two cells.
# We break that tie in a cyclic, rotation-preserving way.
EMIT_OFFSETS = {
    UP: (-1, 0),
    RIGHT: (0, 2),
    DOWN: (2, 1),
    LEFT: (1, -1),
}

TOWER_POOL = [BLUE, RED, GREEN, MAGENTA, ORANGE, CYAN]
BG_POOL = [BLACK, BLUE, RED, GREEN, MAGENTA, ORANGE, CYAN]
CORE_H = 14
CORE_W = 18
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _gray_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_towers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color in (bg, GRAY, MAROON):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (4, 2), "each tower color must occupy a 2x2 block"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert r1 == r0 + 1 and c1 == c0 + 1, "tower must be exactly 2x2"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "tower must be solid"
        towers.append((color, r0, c0))
    assert towers, "expected at least one signal tower"
    return towers


def _infer_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirror_cells = grid == GRAY
    mirror_map: dict[tuple[int, int], str] = {}
    for comp in _gray_components(mirror_cells):
        assert 2 <= len(comp) <= 4, "each mirror must have length 2-4"
        rows = sorted(r for r, _c in comp)
        cols = sorted(c for _r, c in comp)
        coords = set(comp)
        if len(set(r - c for r, c in comp)) == 1:
            r0 = min(rows)
            c0 = min(cols)
            expected = {(r0 + i, c0 + i) for i in range(len(comp))}
            assert coords == expected, "backslash mirror must be a contiguous diagonal"
            orient = "\\"
        elif len(set(r + c for r, c in comp)) == 1:
            r0 = min(rows)
            c0 = max(cols)
            expected = {(r0 + i, c0 - i) for i in range(len(comp))}
            assert coords == expected, "slash mirror must be a contiguous diagonal"
            orient = "/"
        else:
            raise AssertionError("gray component must be a slash or backslash mirror")
        for cell in comp:
            mirror_map[cell] = orient
    return mirror_map


def _reflect(direction: int, mirror: str) -> int:
    if mirror == "\\":
        return {RIGHT: DOWN, LEFT: UP, DOWN: RIGHT, UP: LEFT}[direction]
    assert mirror == "/"
    return {RIGHT: UP, LEFT: DOWN, DOWN: LEFT, UP: RIGHT}[direction]


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    towers = _infer_towers(g, bg)
    tower_cells = {(r + dr, c + dc) for _color, r, c in towers for dr in (0, 1) for dc in (0, 1)}
    mirrors = _infer_mirrors(g)
    receivers = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    receiver_set = set(receivers)
    assert receivers, "expected at least one receiver"

    out = g.copy()
    hits = {pos: [] for pos in receivers}
    used_orients: set[str] = set()

    for color, tr, tc in towers:
        for direction in (UP, RIGHT, DOWN, LEFT):
            r = tr + EMIT_OFFSETS[direction][0]
            c = tc + EMIT_OFFSETS[direction][1]
            if not (0 <= r < h and 0 <= c < w):
                continue

            seen: set[tuple[int, int, int]] = set()
            d = direction
            while 0 <= r < h and 0 <= c < w:
                state = (r, c, d)
                if state in seen:
                    break
                seen.add(state)

                if (r, c) in tower_cells:
                    break
                if (r, c) in receiver_set:
                    hits[(r, c)].append(int(color))
                    break

                mirror = mirrors.get((r, c))
                if mirror is not None:
                    used_orients.add(mirror)
                    d = _reflect(d, mirror)
                    dr, dc = DIRS[d]
                    r += dr
                    c += dc
                    continue

                if g[r, c] == bg and out[r, c] == bg:
                    out[r, c] = int(color)

                dr, dc = DIRS[d]
                r += dr
                c += dc

    for pos, beam_colors in hits.items():
        r, c = pos
        if len(beam_colors) == 1:
            out[r, c] = int(beam_colors[0])
        elif len(beam_colors) >= 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = MAROON

    if return_meta:
        return out, {"hits": hits, "used_orients": used_orients}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored beams from 2x2 towers, reflect them on gray diagonal mirrors,
    and recolor maroon receivers by the number of beams they receive.
    """
    return _solve(grid)


def _place_tower(grid: np.ndarray, top: int, left: int, color: int, bg: int) -> bool:
    if top < 0 or left < 0 or top + 1 >= grid.shape[0] or left + 1 >= grid.shape[1]:
        return False
    if not np.all(grid[top : top + 2, left : left + 2] == bg):
        return False
    grid[top : top + 2, left : left + 2] = int(color)
    return True


def _mirror_cells(start_r: int, start_c: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "\\":
        return [(start_r + i, start_c + i) for i in range(length)]
    return [(start_r + i, start_c - i) for i in range(length)]


def _place_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int, bg: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != bg:
            return False
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _place_receiver(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if grid[r, c] != bg:
        return False
    grid[r, c] = MAROON
    return True


def _build_core() -> np.ndarray | None:
    bg = random.choice(BG_POOL)
    tower_colors = random.sample([c for c in TOWER_POOL if c != bg], 4)
    grid = np.full((CORE_H, CORE_W), bg, dtype=int)

    # Active backslash-mirror module: rightward beam turns downward into a receiver.
    a_top = random.randint(1, 2)
    a_left = random.randint(1, 2)
    a_mirror_col = a_left + random.choice((3, 4))
    a_len = random.randint(2, 4)
    a_receiver_row = a_top + random.choice((4, 5))

    if not _place_tower(grid, a_top, a_left, tower_colors[0], bg):
        return None
    if not _place_cells(grid, _mirror_cells(a_top, a_mirror_col, "\\", a_len), GRAY, bg):
        return None
    if not _place_receiver(grid, a_receiver_row, a_mirror_col, bg):
        return None

    # Active slash-mirror module: rightward beam turns upward into a receiver.
    b_top = random.randint(8, 9)
    b_left = random.randint(1, 2)
    b_mirror_col = b_left + random.choice((5, 6))
    b_len = random.randint(2, 4)
    b_receiver_row = b_top - random.choice((2, 3))

    if not _place_tower(grid, b_top, b_left, tower_colors[1], bg):
        return None
    if not _place_cells(grid, _mirror_cells(b_top, b_mirror_col, "/", b_len), GRAY, bg):
        return None
    if not _place_receiver(grid, b_receiver_row, b_mirror_col, bg):
        return None

    # Two-beam receiver module: one horizontal beam and one vertical beam meet.
    c_receiver_row = random.randint(7, 9)
    c_receiver_col = random.randint(13, 14)
    c_left_tower_left = c_receiver_col - random.choice((5, 6))
    c_top_tower_top = c_receiver_row - random.choice((4, 5))

    if not _place_tower(grid, c_receiver_row, c_left_tower_left, tower_colors[2], bg):
        return None
    if not _place_tower(grid, c_top_tower_top, c_receiver_col - 1, tower_colors[3], bg):
        return None
    if not _place_receiver(grid, c_receiver_row, c_receiver_col, bg):
        return None

    # At least one untouched receiver.
    zero_candidates = [(11, 16), (12, 16), (10, 15), (2, 16)]
    random.shuffle(zero_candidates)
    zero_count = random.randint(1, 2)
    placed_zero = 0
    for zr, zc in zero_candidates:
        if _place_receiver(grid, zr, zc, bg):
            placed_zero += 1
        if placed_zero >= zero_count:
            break
    if placed_zero == 0:
        return None

    return grid


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    top = random.randint(0, 3)
    bottom = random.randint(0, 3)
    left = random.randint(0, 3)
    right = random.randint(0, 3)
    bg = _most_common_color(out)
    return np.pad(out, ((top, bottom), (left, right)), constant_values=int(bg))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        core = _build_core()
        if core is None:
            continue
        candidate = _transform_grid(core)
        try:
            _out, meta = _solve(candidate, return_meta=True)
        except AssertionError:
            continue

        hit_counts = sorted(len(colors) for colors in meta["hits"].values())
        if not hit_counts:
            continue
        if min(hit_counts) != 0:
            continue
        if 1 not in hit_counts:
            continue
        if max(hit_counts) < 2:
            continue
        if meta["used_orients"] != {"\\", "/"}:
            continue
        return candidate
    raise RuntimeError("failed to generate a valid beam-and-mirror puzzle")
