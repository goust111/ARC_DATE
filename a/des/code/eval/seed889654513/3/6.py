from collections import deque
import random

import numpy as np


BG = 8
GREY = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _source_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    out = []
    for color in SOURCE_COLORS:
        seen = np.zeros((h, w), dtype=bool)
        for sr, sc in np.argwhere(g == color):
            sr, sc = int(sr), int(sc)
            if seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 == 2 and c1 - c0 == 2 and len(coords) == 9:
                if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                    out.append({"color": color, "bbox": (r0, c0, r1, c1)})
    return out


def _receivers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    out = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if patch[1, 1] != BG:
                continue
            border = np.ones((3, 3), dtype=bool)
            border[1, 1] = False
            if np.all(patch[border] == GREY):
                out.append({"bbox": (r0, c0, r0 + 2, c0 + 2), "center": (r0 + 1, c0 + 1)})
    return out


def _adjacent_to_source(cell: tuple[int, int], source: dict) -> bool:
    r, c = cell
    r0, c0, r1, c1 = source["bbox"]
    if c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1):
        return True
    if r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1):
        return True
    return False


def _connected_sources(g: np.ndarray, receiver: dict, sources: list[dict]) -> list[tuple[int, dict]]:
    h, w = g.shape
    r0, c0, r1, c1 = receiver["bbox"]
    starts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if g[r, c] == GREY]
    q = deque((r, c, 0) for r, c in starts)
    seen = {(r, c) for r, c in starts}
    found: dict[int, tuple[int, dict]] = {}
    while q:
        r, c, d = q.popleft()
        for i, source in enumerate(sources):
            if i not in found and _adjacent_to_source((r, c), source):
                found[i] = (d, source)
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and g[nr, nc] == GREY:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return sorted(found.values(), key=lambda item: item[0])


def _paint_receiver(out: np.ndarray, receiver: dict, n_sources: int, color: int) -> None:
    cr, cc = receiver["center"]
    if n_sources == 1:
        r0, c0 = cr - 1, cc - 1
        out[r0 : r0 + 3, c0 : c0 + 3] = BG
        out[r0, c0 : c0 + 3] = color
        out[r0 + 2, c0 : c0 + 3] = color
        out[r0 : r0 + 3, c0] = color
        out[r0 : r0 + 3, c0 + 2] = color
    elif n_sources == 2:
        out[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
    elif n_sources == 3:
        r0, c0 = cr - 2, cc - 2
        out[r0 : r0 + 5, c0 : c0 + 5] = BG
        out[r0, c0 : c0 + 5] = color
        out[r0 + 4, c0 : c0 + 5] = color
        out[r0 : r0 + 5, c0] = color
        out[r0 : r0 + 5, c0 + 4] = color
    elif n_sources == 4:
        out[cr - 2 : cr + 3, cc - 2 : cc + 3] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sources = _source_components(g)
    for receiver in _receivers(g):
        linked = _connected_sources(g, receiver, sources)
        n = len(linked)
        if n not in (1, 2, 3, 4):
            continue
        color = linked[0][1]["color"]
        _paint_receiver(out, receiver, n, color)
    return out


def _mark_free(occ: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occ.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if occ[rr, cc]:
                    return False
    for r, c in cells:
        occ[r, c] = True
    return True


def _receiver_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3) if (dr, dc) != (1, 1)]


def _source_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]


def _wire_cells(rr0: int, rc0: int, sr0: int, sc0: int, side: str) -> list[tuple[int, int]]:
    cells = []
    if side == "up":
        c = rc0 + 1
        for r in range(sr0 + 3, rr0 + 1):
            cells.append((r, c))
    elif side == "down":
        c = rc0 + 1
        for r in range(rr0 + 2, sr0):
            cells.append((r, c))
    elif side == "left":
        r = rr0 + 1
        for c in range(sc0 + 3, rc0 + 1):
            cells.append((r, c))
    else:
        r = rr0 + 1
        for c in range(rc0 + 2, sc0):
            cells.append((r, c))
    return cells


def _try_group(grid: np.ndarray, occ: np.ndarray, colors: list[int]) -> bool:
    h, w = grid.shape
    n = random.randint(1, min(4, len(colors)))
    sides = random.sample(list(DIRS), n)
    gaps = random.sample(range(3, 10), n)
    max_gap = max(gaps)
    row_low, row_high = max_gap + 4, h - max_gap - 7
    col_low, col_high = max_gap + 4, w - max_gap - 7
    if row_low > row_high or col_low > col_high:
        return False
    rr0 = random.randint(row_low, row_high)
    rc0 = random.randint(col_low, col_high)

    cells = _receiver_cells(rr0, rc0)
    specs = []
    for side, gap, color in zip(sides, gaps, colors[:n]):
        if side == "up":
            sr0, sc0 = rr0 - gap - 3, rc0
        elif side == "down":
            sr0, sc0 = rr0 + 3 + gap, rc0
        elif side == "left":
            sr0, sc0 = rr0, rc0 - gap - 3
        else:
            sr0, sc0 = rr0, rc0 + 3 + gap
        source = _source_cells(sr0, sc0)
        wire = _wire_cells(rr0, rc0, sr0, sc0, side)
        cells.extend(source)
        cells.extend(wire)
        specs.append((side, color, source, wire))

    if not _mark_free(occ, cells):
        return False

    for r, c in _receiver_cells(rr0, rc0):
        grid[r, c] = GREY
    for _side, color, source, wire in specs:
        for r, c in wire:
            grid[r, c] = GREY
        for r, c in source:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(23, 30)
    w = random.randint(23, 30)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    colors = SOURCE_COLORS[:]
    random.shuffle(colors)
    groups = random.randint(1, 2)
    used = 0
    for _ in range(groups):
        if used >= len(colors):
            break
        for _attempt in range(150):
            before = grid.copy()
            occ_before = occ.copy()
            if _try_group(grid, occ, colors[used:]):
                linked = _connected_sources(grid, _receivers(grid)[-1], _source_components(grid))
                used += len(linked)
                break
            grid[:, :] = before
            occ[:, :] = occ_before
    receivers = _receivers(grid)
    if not receivers:
        return None
    if len(_source_components(grid)) < len(receivers):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal network input")
