from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_arrow(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], int] | None:
    cells = set(comp)
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})

    for row in rows:
        row_cols = sorted(c for r, c in comp if r == row)
        if row_cols != list(range(min(row_cols), max(row_cols) + 1)):
            continue
        c0, c1 = row_cols[0], row_cols[-1]
        tail = {(row, c) for c in row_cols}
        if len(tail) >= 2 and cells == tail | {(row - 1, c1), (row + 1, c1)}:
            return ((0, 1), len(tail))
        if len(tail) >= 2 and cells == tail | {(row - 1, c0), (row + 1, c0)}:
            return ((0, -1), len(tail))

    for col in cols:
        col_rows = sorted(r for r, c in comp if c == col)
        if col_rows != list(range(min(col_rows), max(col_rows) + 1)):
            continue
        r0, r1 = col_rows[0], col_rows[-1]
        tail = {(r, col) for r in col_rows}
        if len(tail) >= 2 and cells == tail | {(r1, col - 1), (r1, col + 1)}:
            return ((1, 0), len(tail))
        if len(tail) >= 2 and cells == tail | {(r0, col - 1), (r0, col + 1)}:
            return ((-1, 0), len(tail))

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored object by the distance and direction encoded in its same-colored arrow, leaving a gray residue.

    Input structure:
    - The background is black.
    - Each active color appears in exactly two disconnected components: one arbitrary connected object and one small arrow marker.
    - An arrow consists of a straight tail plus two head pixels at one tail endpoint, and the tail length encodes the translation distance.

    Rule:
    - For each color, identify which component is the arrow, read its direction from the head orientation, and read its step count from the tail length.
    - Remove the arrow from the grid.
    - Recolor the original object cells to gray as a residue.
    - Redraw the object translated by the encoded number of cells along the arrow direction, clipping any translated pixels that leave the grid.

    Key insight:
    - The marker is a compact control symbol: its visible orientation gives the movement direction, while the tail's visible length gives the distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GRAY)):
        comps = _components(g == color)
        assert len(comps) == 2, f"color {color} must have one object and one arrow"
        parsed = [_parse_arrow(comp) for comp in comps]
        if sum(item is not None for item in parsed) == 2:
            arrow_idx = 0 if len(comps[0]) <= len(comps[1]) else 1
        else:
            assert sum(item is not None for item in parsed) == 1
            arrow_idx = 0 if parsed[0] is not None else 1
        obj_idx = 1 - arrow_idx
        (dr, dc), steps = parsed[arrow_idx]  # type: ignore[misc]
        arrow = comps[arrow_idx]
        obj = comps[obj_idx]

        for r, c in arrow:
            out[r, c] = BG
        for r, c in obj:
            out[r, c] = GRAY
        for r, c in obj:
            nr = r + dr * steps
            nc = c + dc * steps
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color

    return out


def _random_object_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    target = random.randint(2, min(h * w, 5))
    while int(mask.sum()) < target:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _arrow_cells(r: int, c: int, steps: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        tail = [(r, c + k) for k in range(steps)]
        head = [(r - 1, c + steps - 1), (r + 1, c + steps - 1)]
    elif (dr, dc) == (0, -1):
        tail = [(r, c - k) for k in range(steps)]
        head = [(r - 1, c - steps + 1), (r + 1, c - steps + 1)]
    elif (dr, dc) == (1, 0):
        tail = [(r + k, c) for k in range(steps)]
        head = [(r + steps - 1, c - 1), (r + steps - 1, c + 1)]
    else:
        tail = [(r - k, c) for k in range(steps)]
        head = [(r - steps + 1, c - 1), (r - steps + 1, c + 1)]
    return tail + head


def _separated(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    sb = set(b)
    for r, c in a:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in sb:
                    return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG, dtype=int)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
        ok = True

        for color in colors:
            placed = False
            for _ in range(120):
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                steps = random.randint(2, 4)
                mask = _random_object_mask()
                oh, ow = mask.shape
                cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
                if _parse_arrow(cells) is not None:
                    continue

                r0 = random.randint(3, h - oh - 4)
                c0 = random.randint(3, w - ow - 4)
                obj = [(r0 + r, c0 + c) for r, c in cells]
                moved = [(r + direction[0] * steps, c + direction[1] * steps) for r, c in obj]
                if not all(0 <= r < h and 0 <= c < w for r, c in moved):
                    continue

                if direction == (0, 1):
                    ar = r0 + oh // 2
                    ac = max(c for _, c in obj) - steps - 3
                elif direction == (0, -1):
                    ar = r0 + oh // 2
                    ac = min(c for _, c in obj) + steps + 3
                elif direction == (1, 0):
                    ar = max(r for r, _ in obj) - steps - 3
                    ac = c0 + ow // 2
                else:
                    ar = min(r for r, _ in obj) + steps + 3
                    ac = c0 + ow // 2

                arrow = _arrow_cells(ar, ac, steps, direction)
                if not all(0 <= r < h and 0 <= c < w for r, c in arrow):
                    continue
                if not _separated(obj, arrow):
                    continue
                if any(grid[r, c] != BG for r, c in obj + arrow):
                    continue
                neighborhood_clear = True
                occupied = set(obj + arrow)
                for r, c in occupied:
                    for nr in range(r - 1, r + 2):
                        for nc in range(c - 1, c + 2):
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and (nr, nc) not in occupied:
                                neighborhood_clear = False
                if not neighborhood_clear:
                    continue
                for r, c in obj + arrow:
                    grid[r, c] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
