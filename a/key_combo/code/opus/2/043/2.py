import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if not mask[nr, nc] or visited[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    comp.append((nr, nc))
            comps.append(comp)
    return comps


def _region_of_cell(regions: list[list[tuple[int, int]]], cell: tuple[int, int]) -> int | None:
    for i, comp in enumerate(regions):
        if cell in comp:
            return i
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape
    red_mask = grid == RED
    non_red_regions = _components_of_mask(~red_mask)
    if len(non_red_regions) < 2:
        return np.zeros((h, w), dtype=int)

    anchor_cell = (h - 1, 0)
    anchor_idx = _region_of_cell(non_red_regions, anchor_cell)
    if anchor_idx is None:
        anchor_idx = max(range(len(non_red_regions)), key=lambda i: len(non_red_regions[i]))
    idx_a = anchor_idx
    idx_b = 1 - idx_a if idx_a == 0 and len(non_red_regions) >= 2 else (0 if idx_a != 0 else 1)
    region_masks = []
    for ridx in [idx_a, idx_b]:
        reg = np.zeros((h, w), dtype=bool)
        for rr, cc in non_red_regions[ridx]:
            reg[rr, cc] = True
        region_masks.append(reg)

    colors = [c for c in np.unique(grid) if c not in (BLACK, RED)]
    diffs = []
    for c in colors:
        c = int(c)
        mask = grid == c
        a = 0
        b = 0
        for comp in _components_of_mask(mask & region_masks[0]):
            a += 1
        for comp in _components_of_mask(mask & region_masks[1]):
            b += 1
        d = a - b
        if d != 0:
            diffs.append((abs(d), c, d))

    if not diffs:
        return np.zeros((1, 1), dtype=int)

    diffs.sort(reverse=True, key=lambda x: x[0])
    width = diffs[0][0]
    out = np.full((len(diffs), width), BLACK, dtype=int)
    for r, (_, c, d) in enumerate(diffs):
        if d > 0:
            out[r, width - d :] = c
        else:
            out[r, : abs(d)] = c
    return out


def _place_blob(grid: np.ndarray, mask: np.ndarray, color: int, size: int) -> bool:
    h, w = grid.shape
    candidates = np.argwhere(mask & (grid == BLACK))
    if len(candidates) == 0:
        return False
    sr, sc = tuple(candidates[random.randrange(len(candidates))])
    painted = [(sr, sc)]
    grid[sr, sc] = color
    frontier = [(sr, sc)]

    while len(painted) < size:
        if not frontier:
            return False
        r, c = random.choice(frontier)
        neigh = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]
        random.shuffle(neigh)
        progressed = False
        for nr, nc in neigh:
            if (
                0 <= nr < h
                and 0 <= nc < w
                and mask[nr, nc]
                and grid[nr, nc] == BLACK
                and (nr, nc) not in painted
            ):
                grid[nr, nc] = color
                painted.append((nr, nc))
                frontier.append((nr, nc))
                progressed = True
                break
        if not progressed:
            frontier.remove((r, c))
    return True


def generate_input() -> np.ndarray:
    h, w = 18, 12
    colors = [BLUE, GREEN, YELLOW, GREY]
    for _ in range(200):
        g = np.full((h, w), BLACK, dtype=int)
        sep_col = random.randint(5, 6)
        g[:, sep_col] = RED

        color_choices = random.sample(colors, random.randint(3, 4))
        target_abs = random.sample(range(1, 7), len(color_choices))
        used_abs = set()
        records = []

        for color in color_choices:
            while True:
                a = random.randint(1, 4)
                b = random.randint(0, 4)
                d = a - b
                if d != 0 and abs(d) not in used_abs:
                    used_abs.add(abs(d))
                    break
            records.append((color, a, b))

        left_mask = np.zeros((h, w), dtype=bool)
        right_mask = np.zeros((h, w), dtype=bool)
        left_mask[:, :sep_col] = True
        right_mask[:, sep_col + 1 :] = True

        ok = True
        for color, a_count, b_count in records:
            for _ in range(a_count):
                size = random.randint(2, 5)
                if not _place_blob(g, left_mask, color, size):
                    ok = False
                    break
            if not ok:
                break
            for _ in range(b_count):
                size = random.randint(2, 5)
                if not _place_blob(g, right_mask, color, size):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if sep_col < 1 or sep_col >= w - 1:
            continue
        return g
    return np.zeros((h, w), dtype=int)


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


if __name__ == "__main__":
    random.seed(0)
    print(main(generate_input()[0]).shape)
