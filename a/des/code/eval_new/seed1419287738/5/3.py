import random
from collections import defaultdict, deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = defaultdict(list)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps[color].append(cells)
    return comps


def _track_info(cells, h, w):
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    touches = [(r, c) for r, c in cells if r in (0, h - 1) or c in (0, w - 1)]
    if not touches:
        return None
    if len(rows) == 1 and len(cells) >= 4:
        row = next(iter(rows))
        cols_sorted = sorted(cols)
        if cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)):
            edge = min(touches, key=lambda x: min(x[1], w - 1 - x[1]))
            sign = 1 if edge[1] == 0 else -1
            return {"axis": "h", "line": row, "edge": edge, "sign": sign}
    if len(cols) == 1 and len(cells) >= 4:
        col = next(iter(cols))
        rows_sorted = sorted(rows)
        if rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)):
            edge = min(touches, key=lambda x: min(x[0], h - 1 - x[0]))
            sign = 1 if edge[0] == 0 else -1
            return {"axis": "v", "line": col, "edge": edge, "sign": sign}
    return None


def _bbox_mask(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Slide same-color fragments to the attracting end of their edge track.

    Input structure:
    Each non-background color has one straight line segment track touching a grid edge
    and several separate compact fragments of the same color away from the track.

    Rule:
    For each color, the edge-touching track end attracts the fragments.  Fragments are
    sorted by original distance to that end, erased from their old locations, aligned
    to the track row or column, and stacked from the edge inward without rotating.
    Track pixels are retained.

    Key insight:
    Color pairs fragments with their track; the edge-touching endpoint gives direction,
    and distance to that endpoint determines stacking order.
    """
    out = np.array(grid, copy=True)
    h, w = grid.shape
    for color, comps in _components_by_color(grid).items():
        tracks = [(comp, _track_info(comp, h, w)) for comp in comps]
        tracks = [(comp, info) for comp, info in tracks if info is not None]
        if not tracks:
            continue
        track, info = max(tracks, key=lambda item: len(item[0]))
        fragments = [comp for comp in comps if comp is not track]
        if not fragments:
            continue
        er, ec = info["edge"]
        fragments.sort(key=lambda comp: min(abs(r - er) + abs(c - ec) for r, c in comp))
        for comp in fragments:
            for r, c in comp:
                out[r, c] = BG
        offset = 1
        for comp in fragments:
            mask = _bbox_mask(comp)
            mh, mw = mask.shape
            if info["axis"] == "h":
                top = info["line"] - mh // 2
                if info["sign"] == 1:
                    left = offset
                    offset += mw
                else:
                    left = w - offset - mw
                    offset += mw
            else:
                left = info["line"] - mw // 2
                if info["sign"] == 1:
                    top = offset
                    offset += mh
                else:
                    top = h - offset - mh
                    offset += mh
            for rr, cc in zip(*np.where(mask)):
                r, c = int(top + rr), int(left + cc)
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = color
    return out


FRAGMENTS = [
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
]


def _place_mask(grid, mask, top, left, color):
    cells = [(top + int(r), left + int(c)) for r, c in zip(*np.where(mask))]
    margin = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if any(r < 1 or r >= grid.shape[0] - 1 or c < 1 or c >= grid.shape[1] - 1 for r, c in cells):
        return False
    if any(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != BG for r, c in margin):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _draw_track(grid, color, orientation, line, from_start, length):
    if orientation == "h":
        if from_start:
            grid[line, 0:length] = color
        else:
            grid[line, grid.shape[1] - length:grid.shape[1]] = color
    else:
        if from_start:
            grid[0:length, line] = color
        else:
            grid[grid.shape[0] - length:grid.shape[0], line] = color


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(18, 29)
        w = random.randint(18, 29)
        grid = np.zeros((h, w), dtype=int)
        n = random.randint(2, 3)
        colors = random.sample(COLORS, n)
        used_lines = set()
        ok = True
        for color in colors:
            orientation = random.choice(["h", "v"])
            from_start = random.choice([True, False])
            if orientation == "h":
                line = random.randint(3, h - 4)
                while ("h", line) in used_lines:
                    line = random.randint(3, h - 4)
                used_lines.add(("h", line))
            else:
                line = random.randint(3, w - 4)
                while ("v", line) in used_lines:
                    line = random.randint(3, w - 4)
                used_lines.add(("v", line))
            _draw_track(grid, color, orientation, line, from_start, random.randint(4, 6))
            for _j in range(random.randint(2, 3)):
                mask = random.choice(FRAGMENTS)
                placed = False
                for _try in range(100):
                    top = random.randint(2, h - mask.shape[0] - 3)
                    left = random.randint(2, w - mask.shape[1] - 3)
                    if orientation == "h" and abs(top - line) <= 2:
                        continue
                    if orientation == "v" and abs(left - line) <= 2:
                        continue
                    if _place_mask(grid, mask, top, left, color):
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                break
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid magnetic-track puzzle")
