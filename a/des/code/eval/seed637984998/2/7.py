from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5


def _bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _components(grid: np.ndarray, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == BG or (colors is not None and color not in colors):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            comps.append({"color": color, "coords": set(pts), "bbox": (min(rows), min(cols), max(rows), max(cols))})
    return comps


def _panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(grid == GREY)
    if pts.size == 0:
        return None
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _inside(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _is_rect_border(coords: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return coords == expected


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    panel = _panel_bbox(g)
    if panel is None:
        return out

    keys = []
    locks = {}
    for comp in _components(g):
        color = int(comp["color"])
        if color == GREY:
            continue
        coords = comp["coords"]
        inside = all(_inside(panel, r, c) for r, c in coords)
        outside = all(not _inside(panel, r, c) for r, c in coords)
        if inside and _is_rect_border(coords, comp["bbox"]):
            locks[color] = comp
        elif outside:
            keys.append(comp)

    keys.sort(key=lambda comp: (len(comp["coords"]), comp["bbox"][0], comp["bbox"][1]))
    for key in keys:
        color = int(key["color"])
        lock = locks.get(color)
        if lock is None:
            continue
        for r, c in key["coords"]:
            out[r, c] = BG
        r0, c0, r1, c1 = lock["bbox"]
        for r, c in lock["coords"]:
            out[r, c] = GREY
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = color
    return out


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BG


def _can_place(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    pad = grid[max(0, r0 - 1) : min(gh, r0 + h + 1), max(0, c0 - 1) : min(gw, c0 + w + 1)]
    return bool(np.all(pad == BG))


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(24, 32)
        w = random.randint(24, 34)
        grid = np.zeros((h, w), dtype=int)
        pr0, pc0 = 5, 6
        ph = random.randint(13, min(20, h - 3))
        pw = random.randint(13, min(22, w - 8))
        grid[pr0 : pr0 + ph, pc0 : pc0 + pw] = GREY

        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
        key_specs = []
        occupied = set()
        for color in colors:
            ih = random.randint(2, 4)
            iw = random.randint(2, 5)
            placed = False
            for _try in range(100):
                rr = random.randint(pr0 + 2, pr0 + ph - ih - 3)
                cc = random.randint(pc0 + 2, pc0 + pw - iw - 3)
                border = {(r, c) for r in range(rr - 1, rr + ih + 1) for c in range(cc - 1, cc + iw + 1)}
                if border & occupied:
                    continue
                _draw_rect_border(grid, rr - 1, cc - 1, ih + 2, iw + 2, color)
                occupied |= border
                key_specs.append((color, ih, iw))
                placed = True
                break
            if not placed:
                break
        if len(key_specs) != len(colors):
            continue

        for color, ih, iw in sorted(key_specs, key=lambda x: x[1] * x[2]):
            placed = False
            for _try in range(100):
                candidates = []
                if random.random() < 0.5:
                    rr = random.randint(1, h - ih - 2)
                    if pc0 - iw - 2 >= 1:
                        candidates.append((rr, random.randint(1, pc0 - iw - 2)))
                    if pc0 + pw + 1 <= w - iw - 2:
                        candidates.append((rr, random.randint(pc0 + pw + 1, w - iw - 2)))
                else:
                    cc = random.randint(1, w - iw - 2)
                    if pr0 - ih - 2 >= 1:
                        candidates.append((random.randint(1, pr0 - ih - 2), cc))
                    if pr0 + ph + 1 <= h - ih - 2:
                        candidates.append((random.randint(pr0 + ph + 1, h - ih - 2), cc))
                if not candidates:
                    continue
                rr, cc = random.choice(candidates)
                if _can_place(grid, rr, cc, ih, iw):
                    grid[rr : rr + ih, cc : cc + iw] = color
                    placed = True
                    break
            if not placed:
                break
        else:
            return grid
    raise RuntimeError("failed to generate lock/key puzzle")


if __name__ == "__main__":
    for _ in range(20):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
