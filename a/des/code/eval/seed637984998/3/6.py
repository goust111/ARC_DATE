from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
BANDS = {4, 8, 9}
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            comps.append({"color": color, "coords": set(pts), "bbox": (min(rows), min(cols), max(rows), max(cols))})
    return comps


def _arrow_cells(tip: tuple[int, int], direction: str, side: int) -> set[tuple[int, int]]:
    r, c = tip
    if direction == "R":
        return {(r, c), (r, c - 1), (r + side, c)}
    if direction == "L":
        return {(r, c), (r, c + 1), (r + side, c)}
    if direction == "D":
        return {(r, c), (r - 1, c), (r, c + side)}
    if direction == "U":
        return {(r, c), (r + 1, c), (r, c + side)}
    raise ValueError(direction)


def _match_arrow(coords: set[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    for tip in coords:
        for direction in DIRS:
            for side in (-1, 1):
                if coords == _arrow_cells(tip, direction, side):
                    return direction, tip
    return None


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in b:
                    return True
    return False


def _full_bands(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    cells = set()
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            cells.update((r, c) for c in range(w))
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            cells.update((r, c) for r in range(h))
    return cells


def _shadow(
    comp: dict,
    direction: str,
    grid: np.ndarray,
    band_cells: set[tuple[int, int]],
    ignored: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    coords = set(comp["coords"])
    blockers = {
        (r, c)
        for r, c in zip(*np.nonzero(grid != BG))
        if (r, c) not in coords and (r, c) not in band_cells and (r, c) not in ignored
    }
    last_band = None
    step = 0
    while True:
        nxt = {(r + (step + 1) * dr, c + (step + 1) * dc) for r, c in coords}
        if any(r < 0 or c < 0 or r >= h or c >= w or (r, c) in blockers for r, c in nxt):
            break
        for r, c in nxt:
            if (r, c) in band_cells and int(grid[r, c]) != BG:
                last_band = int(grid[r, c])
        step += 1
    if step == 0:
        return set(), GREY
    return {(r + step * dr, c + step * dc) for r, c in coords}, (last_band if last_band is not None else GREY)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    band_cells = _full_bands(g, bg)
    comps = [c for c in _components_by_color(g, bg) if int(c["color"]) not in BANDS]
    arrows = []
    shapes = []
    for comp in comps:
        m = _match_arrow(comp["coords"])
        if m is None:
            shapes.append(comp)
        else:
            direction, tip = m
            arrows.append({"color": comp["color"], "coords": comp["coords"], "direction": direction, "tip": tip})
    arrow_cells = set().union(*(a["coords"] for a in arrows)) if arrows else set()
    for arrow in arrows:
        candidates = [s for s in shapes if int(s["color"]) == int(arrow["color"]) and _touches(s["coords"], arrow["coords"])]
        if not candidates:
            candidates = [s for s in shapes if int(s["color"]) == int(arrow["color"])]
        if not candidates:
            continue
        shape = min(candidates, key=lambda s: min(abs(r - arrow["tip"][0]) + abs(c - arrow["tip"][1]) for r, c in s["coords"]))
        cells, color = _shadow(shape, arrow["direction"], g, band_cells, arrow_cells)
        for r, c in cells:
            if out[r, c] == bg:
                out[r, c] = color
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _place(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shape_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in np.argwhere(mask)}


def _clear(grid: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG:
                    return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(22, 30)
        w = random.randint(22, 32)
        grid = np.zeros((h, w), dtype=int)
        for r, color in zip(sorted(random.sample(range(5, h - 5), 2)), random.sample(list(BANDS), 2)):
            grid[r, :] = color
        colors = random.sample([1, 2, 3, 6, 7], random.randint(2, 4))
        for color in colors:
            placed = False
            for _try in range(200):
                mask = random.choice(SHAPES)
                mh, mw = mask.shape
                top = random.randint(2, h - mh - 3)
                left = random.randint(2, w - mw - 3)
                shape = _shape_cells(mask, top, left)
                if any(grid[r, c] != BG for r, c in shape):
                    continue
                direction = random.choice(list(DIRS))
                sr, sc = random.choice(list(shape))
                if direction == "R":
                    tip = (sr, max(c for r, c in shape if r == sr) + 2)
                elif direction == "L":
                    tip = (sr, min(c for r, c in shape if r == sr) - 2)
                elif direction == "D":
                    tip = (max(r for r, c in shape if c == sc) + 2, sc)
                else:
                    tip = (min(r for r, c in shape if c == sc) - 2, sc)
                arr = _arrow_cells(tip, direction, random.choice([-1, 1]))
                if _clear(grid, arr, 0) and _clear(grid, shape, 1):
                    _place(grid, shape, color)
                    _place(grid, arr, color)
                    placed = True
                    break
            if not placed:
                break
        else:
            return grid
    raise RuntimeError("failed to generate projection puzzle")


if __name__ == "__main__":
    for _ in range(20):
        x = generate_input()
        y = main(x)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
