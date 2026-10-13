from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PORTAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _portal_gap(r0: int, c0: int, grid: np.ndarray) -> tuple[tuple[int, int], str] | None:
    candidates = {
        (r0, c0 + 1): "up",
        (r0 + 2, c0 + 1): "down",
        (r0 + 1, c0): "left",
        (r0 + 1, c0 + 2): "right",
    }
    gaps = [(cell, side) for cell, side in candidates.items() if int(grid[cell]) == BG]
    return gaps[0] if len(gaps) == 1 else None


def _portals(grid: np.ndarray) -> list[dict]:
    portals: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY)):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0, c1 - c0) != (2, 2):
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if int(sub[1, 1]) != BG or int(np.count_nonzero(sub == color)) != 7:
                continue
            gap = _portal_gap(r0, c0, grid)
            if gap is None:
                continue
            portals.append({"color": color, "bbox": (r0, c0, r1, c1), "gap": gap[0], "side": gap[1]})
    return portals


def _grey_segments(grid: np.ndarray) -> list[dict]:
    segs: list[dict] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == r1:
            cells = sorted(comp, key=lambda x: x[1])
            if len(cells) == c1 - c0 + 1:
                segs.append({"cells": cells, "axis": "h", "center": (r0, (c0 + c1) / 2), "length": len(cells)})
        elif c0 == c1:
            cells = sorted(comp, key=lambda x: x[0])
            if len(cells) == r1 - r0 + 1:
                segs.append({"cells": cells, "axis": "v", "center": ((r0 + r1) / 2, c0), "length": len(cells)})
    return segs


def _entry_candidates(seg: dict, portals: list[dict]) -> list[tuple[float, int, dict]]:
    sr, sc = seg["center"]
    out = []
    for idx, portal in enumerate(portals):
        gr, gc = portal["gap"]
        if seg["axis"] == "h" and gr == sr and gc != sc:
            out.append((abs(gc - sc), idx, portal))
        elif seg["axis"] == "v" and gc == sc and gr != sr:
            out.append((abs(gr - sr), idx, portal))
    return sorted(out)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    portals = _portals(g)
    by_color: dict[int, list[dict]] = {}
    for portal in portals:
        by_color.setdefault(portal["color"], []).append(portal)
    for color, items in by_color.items():
        assert len(items) == 2, f"expected portal pair for color {color}"

    out = g.copy()
    for seg in _grey_segments(g):
        candidates = _entry_candidates(seg, portals)
        assert candidates, "grey segment does not face a portal"
        _dist, _idx, entry = candidates[0]
        pair = [p for p in by_color[entry["color"]] if p is not entry][0]
        for r, c in seg["cells"]:
            out[r, c] = BG
        dr, dc = DIRS[pair["side"]]
        gr, gc = pair["gap"]
        for k in range(1, seg["length"] + 1):
            rr, cc = gr + dr * k, gc + dc * k
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = GREY
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h >= H - 1 or c0 + w >= W - 1:
        return False
    rr0, cc0 = max(0, r0 - pad), max(0, c0 - pad)
    rr1, cc1 = min(H, r0 + h + pad), min(W, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_portal(grid: np.ndarray, occupied: np.ndarray, r0: int, c0: int, color: int, side: str) -> dict:
    grid[r0 : r0 + 3, c0 : c0 + 3] = color
    grid[r0 + 1, c0 + 1] = BG
    gap = {
        "up": (r0, c0 + 1),
        "down": (r0 + 2, c0 + 1),
        "left": (r0 + 1, c0),
        "right": (r0 + 1, c0 + 2),
    }[side]
    grid[gap] = BG
    occupied[max(0, r0 - 1) : r0 + 4, max(0, c0 - 1) : c0 + 4] = True
    return {"color": color, "gap": gap, "side": side}


def _place_portal(grid: np.ndarray, occupied: np.ndarray, color: int, side: str) -> dict | None:
    H, W = grid.shape
    for _ in range(300):
        r0 = random.randint(2, H - 5)
        c0 = random.randint(2, W - 5)
        if _can_place(occupied, r0, c0, 3, 3, pad=3):
            return _draw_portal(grid, occupied, r0, c0, color, side)
    return None


def _line_cells_from_gap(gap: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[side]
    gr, gc = gap
    return [(gr + dr * k, gc + dc * k) for k in range(1, length + 1)]


def _generate_case() -> np.ndarray | None:
    H, W = random.randint(26, 34), random.randint(26, 34)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    colors = list(PORTAL_COLORS)
    random.shuffle(colors)
    pair_count = random.randint(2, 4)
    portals_by_color: dict[int, list[dict]] = {}
    for i in range(pair_count):
        color = colors[i]
        sides = random.sample(list(DIRS), 2)
        a = _place_portal(grid, occupied, color, sides[0])
        b = _place_portal(grid, occupied, color, sides[1])
        if a is None or b is None:
            return None
        portals_by_color[color] = [a, b]

    for color, pair in portals_by_color.items():
        entry = random.choice(pair)
        length = random.randint(2, 5)
        cells = _line_cells_from_gap(entry["gap"], entry["side"], length)
        if any(not (0 <= r < H and 0 <= c < W) or grid[r, c] != BG for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = GREY
            occupied[r, c] = True

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
    raise RuntimeError("failed to generate a valid portal puzzle")
