import random

import numpy as np


BG = 8       # azure
FG = 5       # gray
END = 2      # red
MID = 3      # green
BRANCH = 1   # blue
CROSS = 4    # yellow

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_ATTEMPTS = 400
_GEN_MODES = ("path", "t", "cross", "mixed")


def _degree_map(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros(mask.shape, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    return deg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fg = g == FG
    deg = _degree_map(fg)
    out[fg & (deg == 1)] = END
    out[fg & (deg == 2)] = MID
    out[fg & (deg == 3)] = BRANCH
    out[fg & (deg == 4)] = CROSS
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(pts)


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
    return cells


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _fit_random(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]] | None:
    cells = _normalize(cells)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    if max_r + 2 >= h or max_c + 2 >= w:
        return None
    top = random.randint(1, h - max_r - 2)
    left = random.randint(1, w - max_c - 2)
    return _translate(cells, top, left)


def _gen_path_shape() -> set[tuple[int, int]] | None:
    for _ in range(80):
        length = random.randint(7, 18)
        cells = [(0, 0)]
        used = {(0, 0)}
        prev = None
        cur = (0, 0)
        while len(cells) < length:
            dirs = _N4[:]
            random.shuffle(dirs)
            moved = False
            for dr, dc in dirs:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in used:
                    continue
                bad = False
                for adr, adc in _N4:
                    adj = (nxt[0] + adr, nxt[1] + adc)
                    if adj in used and adj != cur:
                        bad = True
                        break
                if bad:
                    continue
                used.add(nxt)
                cells.append(nxt)
                prev = cur
                cur = nxt
                moved = True
                break
            if not moved:
                break
        shape = set(cells)
        mask = _cells_to_mask(_normalize(shape))
        deg = _degree_map(mask)[mask]
        if len(shape) >= 7 and np.count_nonzero(deg == 1) == 2 and np.all((deg >= 1) & (deg <= 2)):
            return _normalize(shape)
    return None


def _gen_t_shape() -> set[tuple[int, int]]:
    trunk = random.randint(5, 9)
    stem_up = random.randint(1, trunk - 3)
    stem_down = trunk - 1 - stem_up
    arm = random.randint(2, 5)
    orient = random.choice(["up", "down", "left", "right"])
    if orient in ("up", "down"):
        cells = _line_cells(0, 0, trunk - 1, 0)
        pivot = stem_up
        direction = -1 if orient == "left" else 1
        cells |= _line_cells(pivot, 0, pivot, direction * arm)
    else:
        cells = _line_cells(0, 0, 0, trunk - 1)
        pivot = stem_up
        direction = -1 if orient == "up" else 1
        cells |= _line_cells(0, pivot, direction * arm, pivot)
    return _normalize(cells)


def _gen_cross_shape() -> set[tuple[int, int]]:
    up = random.randint(2, 5)
    down = random.randint(2, 5)
    left = random.randint(2, 5)
    right = random.randint(2, 5)
    cells = _line_cells(-up, 0, down, 0) | _line_cells(0, -left, 0, right)
    return _normalize(cells)


def _gen_mixed_shape() -> set[tuple[int, int]]:
    up = random.randint(2, 4)
    down = random.randint(2, 4)
    left = random.randint(2, 4)
    right = random.randint(2, 4)
    cells = _line_cells(-up, 0, down, 0) | _line_cells(0, -left, 0, right)
    arm = random.choice(["up", "down", "left", "right"])
    if arm == "up":
        dist = random.randint(1, up - 1)
        anchor = (-dist, 0)
        branch_len = random.randint(1, 3)
        sign = random.choice([-1, 1])
        cells |= _line_cells(anchor[0], anchor[1], anchor[0], sign * branch_len)
    elif arm == "down":
        dist = random.randint(1, down - 1)
        anchor = (dist, 0)
        branch_len = random.randint(1, 3)
        sign = random.choice([-1, 1])
        cells |= _line_cells(anchor[0], anchor[1], anchor[0], sign * branch_len)
    elif arm == "left":
        dist = random.randint(1, left - 1)
        anchor = (0, -dist)
        branch_len = random.randint(1, 3)
        sign = random.choice([-1, 1])
        cells |= _line_cells(anchor[0], anchor[1], sign * branch_len, anchor[1])
    else:
        dist = random.randint(1, right - 1)
        anchor = (0, dist)
        branch_len = random.randint(1, 3)
        sign = random.choice([-1, 1])
        cells |= _line_cells(anchor[0], anchor[1], sign * branch_len, anchor[1])
    return _normalize(cells)


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mode_ok(mask: np.ndarray, mode: str) -> bool:
    if not _connected(mask):
        return False
    deg = _degree_map(mask)[mask]
    if np.any((deg < 1) | (deg > 4)):
        return False
    if mode == "path":
        return np.count_nonzero(deg == 1) == 2 and np.all((deg == 1) | (deg == 2))
    if mode == "t":
        return np.count_nonzero(deg == 3) >= 1 and np.count_nonzero(deg == 4) == 0
    if mode == "cross":
        return np.count_nonzero(deg == 4) >= 1
    return np.count_nonzero(deg == 3) >= 1 and np.count_nonzero(deg == 4) >= 1


def _shape_for_mode(mode: str) -> set[tuple[int, int]] | None:
    if mode == "path":
        return _gen_path_shape()
    if mode == "t":
        return _gen_t_shape()
    if mode == "cross":
        return _gen_cross_shape()
    return _gen_mixed_shape()


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 22)
    w = random.randint(10, 22)
    mode = random.choice(_GEN_MODES)
    shape = _shape_for_mode(mode)
    if shape is None:
        return None
    placed = _fit_random(shape, h, w)
    if placed is None:
        return None
    grid = np.full((h, w), BG, dtype=int)
    for r, c in placed:
        grid[r, c] = FG
    mask = grid == FG
    if not _mode_ok(mask, mode):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
