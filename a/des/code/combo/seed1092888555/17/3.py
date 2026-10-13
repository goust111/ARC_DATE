import random
from typing import List, Tuple

import numpy as np


BACKGROUND = 0
EDGE_COLOR = 2
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int) -> Tuple[Tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_objects(grid: np.ndarray) -> List[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: List[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == area, "objects must be solid rectangles"
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "anchor": (c0, r0),
                }
            )

    objects.sort(key=lambda obj: (obj["anchor"][0], obj["anchor"][1]))
    return objects


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the alignment graph of solid rectangles as an adjacency matrix.

    Objects are monochrome 4-connected solid rectangles on a black background.
    Two objects are adjacent if their bounding boxes overlap in row span or
    column span. Objects are ordered from left to right, then top to bottom.
    The output is an NxN matrix with object colors on the diagonal and red on
    off-diagonal connected pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_objects(g)
    n = len(objects)
    assert 2 <= n <= 4

    out = np.full((n, n), BACKGROUND, dtype=int)
    for i, obj in enumerate(objects):
        out[i, i] = int(obj["color"])

    for i in range(n):
        r0a, c0a, r1a, c1a = objects[i]["bbox"]
        for j in range(i + 1, n):
            r0b, c0b, r1b, c1b = objects[j]["bbox"]
            row_overlap = _intervals_overlap(r0a, r1a, r0b, r1b)
            col_overlap = _intervals_overlap(c0a, c1a, c0b, c1b)
            if row_overlap or col_overlap:
                out[i, j] = EDGE_COLOR
                out[j, i] = EDGE_COLOR

    return out


def _pair_relation(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> str:
    row_overlap = _intervals_overlap(a[0], a[2], b[0], b[2])
    col_overlap = _intervals_overlap(a[1], a[3], b[1], b[3])
    if row_overlap and col_overlap:
        return "both"
    if row_overlap:
        return "row"
    if col_overlap:
        return "col"
    return "none"


def _sample_rectangles(rng: random.Random, h: int, w: int, n: int) -> List[Tuple[int, int, int, int]] | None:
    rects: List[Tuple[int, int, int, int]] = []
    for _ in range(PLACEMENT_RETRIES):
        rects.clear()
        occupied = np.zeros((h, w), dtype=bool)
        for _idx in range(n):
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                rh = rng.randint(2, max(2, min(5, h // 2)))
                rw = rng.randint(2, max(2, min(5, w // 2)))
                if rh >= h or rw >= w:
                    continue
                r0 = rng.randint(0, h - rh)
                c0 = rng.randint(0, w - rw)
                pad_r0 = max(0, r0 - 1)
                pad_c0 = max(0, c0 - 1)
                pad_r1 = min(h, r0 + rh + 1)
                pad_c1 = min(w, c0 + rw + 1)
                if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                    continue
                occupied[r0 : r0 + rh, c0 : c0 + rw] = True
                rects.append((r0, c0, r0 + rh - 1, c0 + rw - 1))
                placed = True
                break
            if not placed:
                break
        if len(rects) != n:
            continue

        relations = set()
        connected_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                rel = _pair_relation(rects[i], rects[j])
                relations.add(rel)
                if rel != "none":
                    connected_pairs += 1

        if connected_pairs == 0:
            continue
        if connected_pairs == n * (n - 1) // 2:
            continue
        if "row" not in relations and "col" not in relations and "both" not in relations:
            continue
        return list(rects)
    return None


def _paint_rect(grid: np.ndarray, rect: Tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    palette = [1, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(GEN_RETRIES):
        h = rng.randint(9, 18)
        w = rng.randint(9, 18)
        n = rng.randint(2, 4)
        rects = _sample_rectangles(rng, h, w, n)
        if rects is None:
            continue

        colors = rng.sample(palette, n)
        grid = np.full((h, w), BACKGROUND, dtype=int)
        for rect, color in zip(rects, colors):
            _paint_rect(grid, rect, color)

        objects = _find_objects(grid)
        if len(objects) != n:
            continue

        out = main(grid)
        if out.shape != (n, n):
            continue

        off_diag = out.copy()
        np.fill_diagonal(off_diag, BACKGROUND)
        if not np.any(off_diag == EDGE_COLOR):
            continue
        if np.all(off_diag == EDGE_COLOR):
            continue
        return grid

    raise RuntimeError("generate_input: failed after repeated attempts")
