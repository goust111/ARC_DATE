from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _line_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    lines = []
    occupied = set()
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for comp in _components(grid == color):
            if len(comp) < 2:
                continue
            rows = set(map(int, comp[:, 0]))
            cols = set(map(int, comp[:, 1]))
            if len(rows) == 1:
                r = next(iter(rows))
                cs = sorted(cols)
                if cs == list(range(cs[0], cs[-1] + 1)):
                    coords = [(r, c) for c in cs]
                    lines.append({"color": color, "coords": coords, "orient": "h"})
                    occupied.update(coords)
            elif len(cols) == 1:
                c = next(iter(cols))
                rs = sorted(rows)
                if rs == list(range(rs[0], rs[-1] + 1)):
                    coords = [(r, c) for r in rs]
                    lines.append({"color": color, "coords": coords, "orient": "v"})
                    occupied.update(coords)
    return lines, occupied


def _line_direction(grid: np.ndarray, line: dict, line_cells: set[tuple[int, int]], bg: int) -> tuple[int, int] | None:
    h, w = grid.shape
    endpoints = [line["coords"][0], line["coords"][-1]]
    if line["orient"] == "h":
        candidates = [(-1, 0), (1, 0)]
    else:
        candidates = [(0, -1), (0, 1)]
    found = []
    for er, ec in endpoints:
        for dr, dc in candidates:
            rr, cc = er + dr, ec + dc
            if 0 <= rr < h and 0 <= cc < w:
                val = int(grid[rr, cc])
                if val != bg and (rr, cc) not in line_cells:
                    found.append((dr, dc))
    return found[0] if len(found) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    lines, line_cells = _line_objects(g, bg)
    for line in lines:
        line["dir"] = _line_direction(g, line, line_cells, bg)
    cell_to_line = {cell: i for i, line in enumerate(lines) for cell in line["coords"]}
    out = g.copy()
    h, w = g.shape

    for i, line in enumerate(lines):
        direction = line.get("dir")
        if direction is None:
            continue
        dr, dc = direction
        for start in line["coords"]:
            r, c = start
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                if (r, c) in line_cells:
                    hit = cell_to_line[(r, c)]
                    if hit != i and lines[hit].get("dir") is not None:
                        hdr, hdc = lines[hit]["dir"]
                        rr, cc = r, c
                        while True:
                            rr += hdr
                            cc += hdc
                            if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in line_cells:
                                break
                            if int(g[rr, cc]) == bg:
                                out[rr, cc] = int(lines[hit]["color"])
                            else:
                                break
                    break
                if int(g[r, c]) == bg:
                    out[r, c] = int(line["color"])
                else:
                    break
    return out


def _draw_line(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(16, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(range(1, 10), 4)
    hr = random.randint(3, h // 2)
    c0 = random.randint(3, 6)
    length = random.randint(4, 7)
    hline = [(hr, c) for c in range(c0, c0 + length)]
    hit_c = random.choice(hline[1:-1])[1]
    vr0 = hr + random.randint(3, 6)
    vlen = random.randint(4, 7)
    if vr0 + vlen >= h - 2:
        return None
    vline = [(r, hit_c) for r in range(vr0, vr0 + vlen)]
    if any(grid[r, c] != BG for r, c in hline + vline):
        return None
    _draw_line(grid, hline, colors[0])
    _draw_line(grid, vline, colors[1])
    grid[hr + 1, c0] = colors[2]  # horizontal line emits downward
    grid[vr0, hit_c + 1] = colors[3]  # vertical line relays rightward

    if random.random() < 0.7:
        r2 = random.randint(2, h - 3)
        lo = max(2, hit_c + 4)
        hi = w - 5
        if lo <= hi:
            c2 = random.randint(lo, hi)
            extra = [(r2, c) for c in range(c2, c2 + 3)]
            if all(grid[r, c] == BG for r, c in extra) and grid[r2 - 1, c2] == BG:
                _draw_line(grid, extra, random.choice([c for c in range(1, 10) if c not in colors]))
                grid[r2 - 1, c2] = random.choice([c for c in range(1, 10) if c not in colors])
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate relay-line input")
