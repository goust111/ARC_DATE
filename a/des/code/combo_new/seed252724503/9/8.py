from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
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


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored segments and paint their enclosing rectangles from longest pair to shortest pair.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly two disconnected one-pixel-wide straight segments.
    - Segments may be horizontal or vertical, and different color pairs may have overlapping bounding boxes.

    Rule:
    - Match the two segments of each color.
    - For each color pair, compute the smallest axis-aligned bounding box containing both segments and compute the pair's total segment length.
    - Starting with the largest total length and proceeding to the smallest, fill that bounding box with the pair's color.
    - Later, shorter pairs overwrite earlier, longer pairs where their rectangles overlap.

    Key insight:
    - The segments themselves are only anchors: each color really defines one filled rectangle, and the visible overlap order comes from segment-length ranking rather than spatial position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BG)
    items: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components(g == color)
        assert len(comps) == 2, f"color {color} must appear in exactly two segments"
        all_cells = comps[0] + comps[1]
        items.append((len(all_cells), color, _bbox(all_cells)))

    items.sort(key=lambda item: (-item[0], item[1]))
    for _total_len, color, (r0, c0, r1, c1) in items:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != BG:
            return False
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and (nr, nc) not in cells:
                return False
    return True


def _segment_cells(r: int, c: int, length: int, horizontal: bool) -> list[tuple[int, int]]:
    if horizontal:
        return [(r, c + k) for k in range(length)]
    return [(r + k, c) for k in range(length)]


def _separated(seg1: list[tuple[int, int]], seg2: list[tuple[int, int]]) -> bool:
    s2 = set(seg2)
    for r, c in seg1:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in s2:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(16, 28)
        w = random.randint(16, 28)
        grid = np.full((h, w), BG, dtype=int)

        pair_count = random.randint(2, 4)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], pair_count)
        lengths = random.sample(range(4, 12), pair_count)

        ok = True
        for color, total_len in zip(colors, lengths):
            len1 = random.randint(2, total_len - 2)
            len2 = total_len - len1
            placed = False
            for _ in range(120):
                horizontal1 = bool(random.getrandbits(1))
                horizontal2 = bool(random.getrandbits(1))
                r1 = random.randint(1, h - (1 if horizontal1 else len1) - 2)
                c1 = random.randint(1, w - (len1 if horizontal1 else 1) - 2)
                r2 = random.randint(1, h - (1 if horizontal2 else len2) - 2)
                c2 = random.randint(1, w - (len2 if horizontal2 else 1) - 2)
                seg1 = _segment_cells(r1, c1, len1, horizontal1)
                seg2 = _segment_cells(r2, c2, len2, horizontal2)
                if set(seg1) & set(seg2):
                    continue
                if not _separated(seg1, seg2):
                    continue
                if not _can_place(grid, seg1 + seg2):
                    continue
                for rr, cc in seg1 + seg2:
                    grid[rr, cc] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
