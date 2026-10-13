import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2


@dataclass(frozen=True)
class Ring:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    cells: tuple[tuple[int, int], ...]

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def center2(self) -> tuple[int, int]:
        return self.r0 + self.r1, self.c0 + self.c1

    @property
    def area_box(self) -> int:
        return self.h * self.w


def _component_cells(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    stack = [start]
    seen[start] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((int(r), int(c)))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))
    return cells


def _is_rect_border(cells: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(cells) == expected


def _find_rings(grid: np.ndarray) -> list[Ring]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rings: list[Ring] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            cells = _component_cells(grid, (r, c), seen)
            if not _is_rect_border(cells):
                continue
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            rings.append(
                Ring(
                    int(grid[r, c]),
                    min(rows),
                    min(cols),
                    max(rows),
                    max(cols),
                    tuple(sorted(cells)),
                )
            )
    return rings


def _transform_ring(ring: Ring, op: str) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rel = [(r - ring.r0, c - ring.c0) for r, c in ring.cells]
    if op == "rotate":
        nh, nw = ring.w, ring.h
        out = tuple(sorted((c, ring.h - 1 - r) for r, c in rel))
    elif op == "flip":
        nh, nw = ring.h, ring.w
        out = tuple(sorted((r, ring.w - 1 - c) for r, c in rel))
    else:
        nh, nw = ring.h, ring.w
        out = tuple(sorted(rel))
    return nh, nw, out


def _marker_op(grid: np.ndarray, ring: Ring) -> str:
    mr, mc = ring.r0 + 1, ring.c0 + 1
    if not (0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1]):
        return "none"
    marker = int(grid[mr, mc])
    if marker == RED:
        return "rotate"
    if marker == BLUE:
        return "flip"
    return "none"


def _draw_cells(out: np.ndarray, center2: tuple[int, int], shape: tuple[int, int], cells: tuple[tuple[int, int], ...], color: int) -> None:
    nh, nw = shape
    r0 = (int(center2[0]) - nh + 1) // 2
    c0 = (int(center2[1]) - nw + 1) // 2
    for rr, cc in cells:
        r, c = r0 + int(rr), c0 + int(cc)
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    out = np.zeros_like(arr)
    rings = _find_rings(arr)

    groups: dict[tuple[int, int], list[Ring]] = {}
    for ring in rings:
        groups.setdefault(ring.center2, []).append(ring)

    for center2, group in groups.items():
        ordered = sorted(group, key=lambda x: x.area_box, reverse=True)
        for i, ring in enumerate(ordered):
            op = "none" if i == len(ordered) - 1 else _marker_op(arr, ring)
            nh, nw, cells = _transform_ring(ring, op)
            _draw_cells(out, center2, (nh, nw), cells, ring.color)
    return out


def _paint_ring(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    rr0, cc0 = max(0, r0 - pad), max(0, c0 - pad)
    rr1, cc1 = min(occupied.shape[0] - 1, r1 + pad), min(occupied.shape[1] - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _generate_group(grid: np.ndarray, occupied: np.ndarray, rng: random.Random) -> bool:
    layers = rng.choice((2, 3))
    side = rng.randint(11 if layers == 2 else 15, 21)
    if side % 2 == 0:
        side += 1
    h = side
    w = side if rng.random() < 0.55 else side + rng.choice((-4, 4))
    w = max(w, 2 + 4 * layers + 3)
    if w % 2 != h % 2:
        w += 1

    max_r = grid.shape[0] - h - 2
    max_c = grid.shape[1] - w - 2
    if max_r <= 1 or max_c <= 1:
        return False
    for _ in range(80):
        r0 = rng.randint(1, max_r)
        c0 = rng.randint(1, max_c)
        if not _can_place(occupied, r0, c0, h, w, 2):
            continue

        colors = rng.sample([3, 4, 5, 6, 7, 8, 9], layers)
        ops: list[int] = []
        for i in range(layers - 1):
            if h == w:
                ops.append(rng.choice([BLACK, BLUE, RED]))
            else:
                ops.append(rng.choice([BLACK, BLUE]))

        for i in range(layers):
            rr, cc = r0 + 2 * i, c0 + 2 * i
            hh, ww = h - 4 * i, w - 4 * i
            _paint_ring(grid, rr, cc, hh, ww, colors[i])
            if i < layers - 1 and ops[i] != BLACK:
                grid[rr + 1, cc + 1] = ops[i]

        occupied[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        return True
    return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        rows = rng.randint(24, 30)
        cols = rng.randint(24, 30)
        grid = np.zeros((rows, cols), dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        count = rng.randint(1, 3)
        ok = True
        for _ in range(count):
            if not _generate_group(grid, occupied, rng):
                ok = False
                break
        if not ok:
            continue
        if len(_find_rings(grid)) < 2:
            continue
        result = main(grid)
        if isinstance(result, np.ndarray) and result.shape == grid.shape and not np.array_equal(result, grid):
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring puzzle input")
