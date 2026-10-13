from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
PIECE_COLORS = (3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _frame_gaps(grid: np.ndarray) -> list[dict]:
    gaps: list[dict] = []
    for comp in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = []
        for c in range(c0, c1 + 1):
            border.append((r0, c, "up"))
            border.append((r1, c, "down"))
        for r in range(r0 + 1, r1):
            border.append((r, c0, "left"))
            border.append((r, c1, "right"))
        missing = [(r, c, side) for r, c, side in border if int(grid[r, c]) == BG]
        if not missing:
            continue
        by_side: dict[str, list[tuple[int, int]]] = {}
        for r, c, side in missing:
            by_side.setdefault(side, []).append((r, c))
        for side, cells in by_side.items():
            cells = sorted(cells)
            if side in {"up", "down"}:
                cells = sorted(cells, key=lambda x: x[1])
                if len(cells) >= 2 and all(cells[i][1] + 1 == cells[i + 1][1] for i in range(len(cells) - 1)):
                    gaps.append({"cells": cells, "shape": _norm(cells), "side": side, "center": _center(cells)})
            else:
                cells = sorted(cells, key=lambda x: x[0])
                if len(cells) >= 2 and all(cells[i][0] + 1 == cells[i + 1][0] for i in range(len(cells) - 1)):
                    gaps.append({"cells": cells, "shape": _norm(cells), "side": side, "center": _center(cells)})
    return gaps


def _center(cells: list[tuple[int, int]]) -> tuple[float, float]:
    return (sum(r for r, _ in cells) / len(cells), sum(c for _, c in cells) / len(cells))


def _piece_direction(red: tuple[int, int], body: list[tuple[int, int]]) -> str:
    cr, cc = _center(body)
    rr, rc = red
    if abs(rr - cr) >= abs(rc - cc):
        return "up" if rr < cr else "down"
    return "left" if rc < cc else "right"


def _pieces(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != BLUE)
    pieces: list[dict] = []
    for comp in _components(mask):
        reds = [(r, c) for r, c in comp if int(grid[r, c]) == RED]
        if len(reds) != 1:
            continue
        colors = [int(grid[r, c]) for r, c in comp if int(grid[r, c]) not in (RED, BG, BLUE)]
        if not colors:
            continue
        color = max(set(colors), key=colors.count)
        body = [(r, c) for r, c in comp if int(grid[r, c]) == color]
        cells = sorted(comp)
        pieces.append(
            {
                "cells": cells,
                "shape": _norm(cells),
                "color": color,
                "red": reds[0],
                "direction": _piece_direction(reds[0], body),
                "center": _center(cells),
            }
        )
    return pieces


def _ahead(piece: dict, gap: dict) -> bool:
    pr, pc = piece["center"]
    gr, gc = gap["center"]
    dr, dc = DIRS[piece["direction"]]
    vr, vc = gr - pr, gc - pc
    if dr != 0:
        return vc == 0 and vr * dr > 0
    return vr == 0 and vc * dc > 0


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    gaps = _frame_gaps(g)
    used: set[int] = set()
    for piece in _pieces(g):
        candidates = [
            (abs(gap["center"][0] - piece["center"][0]) + abs(gap["center"][1] - piece["center"][1]), idx, gap)
            for idx, gap in enumerate(gaps)
            if idx not in used and gap["shape"] == piece["shape"] and _ahead(piece, gap)
        ]
        assert candidates, "no matching frame gap"
        _dist, idx, gap = min(candidates)
        used.add(idx)
        for r, c in piece["cells"]:
            out[r, c] = BG
        for r, c in gap["cells"]:
            out[r, c] = piece["color"]
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, side: str, start: int, length: int) -> list[tuple[int, int]]:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    if side == "up":
        cells = [(r0, c0 + start + i) for i in range(length)]
    elif side == "down":
        cells = [(r1, c0 + start + i) for i in range(length)]
    elif side == "left":
        cells = [(r0 + start + i, c0) for i in range(length)]
    else:
        cells = [(r0 + start + i, c1) for i in range(length)]
    for r, c in cells:
        grid[r, c] = BG
    return cells


def _can_place_rect(used: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = rect
    for ar0, ac0, ar1, ac1 in used:
        if not (r1 + pad < ar0 or ar1 + pad < r0 or c1 + pad < ac0 or ac1 + pad < c0):
            return False
    return True


def _generate_case() -> np.ndarray | None:
    h = random.randint(24, 32)
    w = random.randint(24, 32)
    grid = np.full((h, w), BG, dtype=int)
    used_rects: list[tuple[int, int, int, int]] = []
    colors = list(PIECE_COLORS)
    random.shuffle(colors)
    count = random.randint(2, 4)
    for i in range(count):
        fh, fw = random.randint(5, 8), random.randint(5, 9)
        side = random.choice(("up", "down", "left", "right"))
        max_len = (fw - 2) if side in {"up", "down"} else (fh - 2)
        length = random.randint(2, min(4, max_len))
        start = random.randint(1, max_len - length + 1)
        placed = False
        for _ in range(200):
            r0 = random.randint(2, h - fh - 3)
            c0 = random.randint(2, w - fw - 3)
            rect = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if _can_place_rect(used_rects, rect, pad=3):
                placed = True
                used_rects.append(rect)
                break
        if not placed:
            return None
        gap = _draw_frame(grid, r0, c0, fh, fw, side, start, length)
        color = colors[i % len(colors)]
        if side in {"up", "down"}:
            gr = gap[0][0]
            gc0 = gap[0][1]
            direction = random.choice(("left", "right"))
            if direction == "right":
                pc0 = max(1, gc0 - random.randint(length + 2, length + 5))
                cells = [(gr, pc0 + k) for k in range(length)]
                red = cells[-1]
            else:
                pc0 = min(w - length - 1, gc0 + random.randint(length + 2, length + 5))
                cells = [(gr, pc0 + k) for k in range(length)]
                red = cells[0]
        else:
            gr0 = gap[0][0]
            gc = gap[0][1]
            direction = random.choice(("up", "down"))
            if direction == "down":
                pr0 = max(1, gr0 - random.randint(length + 2, length + 5))
                cells = [(pr0 + k, gc) for k in range(length)]
                red = cells[-1]
            else:
                pr0 = min(h - length - 1, gr0 + random.randint(length + 2, length + 5))
                cells = [(pr0 + k, gc) for k in range(length)]
                red = cells[0]
        if any(grid[r, c] != BG for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = color
        grid[red] = RED
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-fragment puzzle")
