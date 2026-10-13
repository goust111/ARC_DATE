import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COUNT_PATTERNS = [
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 0),
    (0, 3),
]
GRID_RANGES = {
    2: ((18, 27), (18, 27)),
    3: ((22, 30), (22, 30)),
    4: ((27, 30), (27, 30)),
}
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _count_symbols(interior: np.ndarray, color: int, horizontal: bool) -> int:
    count = 0
    for comp in _components(interior == color):
        if len(comp) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(comp)
        if horizontal and r0 == r1 and abs(c1 - c0) == 1:
            count += 1
        if (not horizontal) and c0 == c1 and abs(r1 - r0) == 1:
            count += 1
    return count


def _left_right(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (-dc, dr), (dc, -dr)


def _draw_line(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int], length: int, color: int) -> tuple[int, int]:
    r, c = start
    dr, dc = direction
    for step in range(1, length + 1):
        rr, cc = r + dr * step, c + dc * step
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = color
    return r + dr * length, c + dc * length


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, RED, BLUE}:
            continue
        mask = grid == color
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        rows = np.flatnonzero(row_counts == row_counts.max())
        cols = np.flatnonzero(col_counts == col_counts.max())
        if len(rows) != 2 or len(cols) != 2:
            continue
        r0, r1 = map(int, rows)
        c0, c1 = map(int, cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = np.zeros_like(mask, dtype=bool)
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], border[r0 : r1 + 1, c0 : c1 + 1]):
            continue

        extra = np.argwhere(mask & ~border)
        if len(extra) != 1:
            continue
        bud = tuple(map(int, extra[0]))
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        if bud == (r0 - 1, mid_c):
            direction = (-1, 0)
        elif bud == (mid_r, c1 + 1):
            direction = (0, 1)
        elif bud == (r1 + 1, mid_c):
            direction = (1, 0)
        elif bud == (mid_r, c0 - 1):
            direction = (0, -1)
        else:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        h_count = _count_symbols(interior, RED, horizontal=True)
        v_count = _count_symbols(interior, BLUE, horizontal=False)
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "bud": bud,
                "direction": direction,
                "h_count": h_count,
                "v_count": v_count,
            }
        )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in _find_frames(g):
        color = int(frame["color"])
        bud = tuple(frame["bud"])
        direction = tuple(frame["direction"])
        h_count = int(frame["h_count"])
        v_count = int(frame["v_count"])
        endpoint = _draw_line(out, bud, direction, h_count + v_count, color)
        left_dir, right_dir = _left_right(direction)
        _draw_line(out, endpoint, left_dir, h_count, color)
        _draw_line(out, endpoint, right_dir, v_count, color)
    return out


def _symbol_footprint(orientation: str, r: int, c: int, ih: int, iw: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if orientation == "H":
        cells = {(r, c), (r, c + 1)}
        r0, r1 = r - 1, r + 1
        c0, c1 = c - 1, c + 2
    else:
        cells = {(r, c), (r + 1, c)}
        r0, r1 = r - 1, r + 2
        c0, c1 = c - 1, c + 1
    blocked = {
        (rr, cc)
        for rr in range(r0, r1 + 1)
        for cc in range(c0, c1 + 1)
        if 0 <= rr < ih and 0 <= cc < iw
    }
    return cells, blocked


def _place_symbols(ih: int, iw: int, h_count: int, v_count: int) -> list[tuple[int, int, int]]:
    orientations = ["H"] * h_count + ["V"] * v_count
    for _ in range(80):
        random.shuffle(orientations)
        blocked: set[tuple[int, int]] = set()
        placed: list[tuple[int, int, int]] = []
        ok = True
        for orientation in orientations:
            candidates: list[tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
            if orientation == "H":
                rows = range(ih)
                cols = range(iw - 1)
            else:
                rows = range(ih - 1)
                cols = range(iw)
            for r in rows:
                for c in cols:
                    cells, footprint = _symbol_footprint(orientation, r, c, ih, iw)
                    if cells & blocked:
                        continue
                    candidates.append((r, c, cells, footprint))
            if not candidates:
                ok = False
                break
            r, c, _cells, footprint = random.choice(candidates)
            blocked |= footprint
            placed.append((RED if orientation == "H" else BLUE, r, c))
        if ok:
            return placed
    raise RuntimeError("failed to place symbols")


def _growth_cells(height: int, width: int, direction: tuple[int, int], h_count: int, v_count: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(height) for c in range(width)}
    mid_r = height // 2
    mid_c = width // 2
    if direction == (-1, 0):
        bud = (-1, mid_c)
    elif direction == (0, 1):
        bud = (mid_r, width)
    elif direction == (1, 0):
        bud = (height, mid_c)
    else:
        bud = (mid_r, -1)
    cells.add(bud)
    dr, dc = direction
    endpoint = bud
    for step in range(1, h_count + v_count + 1):
        endpoint = (bud[0] + dr * step, bud[1] + dc * step)
        cells.add(endpoint)
    left_dir, right_dir = _left_right(direction)
    lr, lc = left_dir
    rr, rc = right_dir
    for step in range(1, h_count + 1):
        cells.add((endpoint[0] + lr * step, endpoint[1] + lc * step))
    for step in range(1, v_count + 1):
        cells.add((endpoint[0] + rr * step, endpoint[1] + rc * step))
    return cells


def _padded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((r + dr, c + dc))
    return padded


def _sample_frame(direction: tuple[int, int]) -> dict:
    for _ in range(80):
        h_count, v_count = random.choice(COUNT_PATTERNS)
        if h_count + v_count == 0:
            continue
        if h_count + v_count >= 4 or max(h_count, v_count) >= 3:
            size_choices = (9,)
        elif h_count + v_count == 3:
            size_choices = (7, 9)
        else:
            size_choices = (7, 9)
        height = random.choice(size_choices)
        width = random.choice(size_choices)
        try:
            symbols = _place_symbols(height - 2, width - 2, h_count, v_count)
        except RuntimeError:
            continue
        return {
            "height": height,
            "width": width,
            "direction": direction,
            "h_count": h_count,
            "v_count": v_count,
            "symbols": symbols,
            "footprint": _growth_cells(height, width, direction, h_count, v_count),
        }
    raise RuntimeError("failed to sample frame")


def _place_frames(grid_h: int, grid_w: int, frames: list[dict]) -> bool:
    occupied: set[tuple[int, int]] = set()
    order = sorted(range(len(frames)), key=lambda i: len(_padded(frames[i]["footprint"])), reverse=True)
    for idx in order:
        frame = frames[idx]
        padded = _padded(frame["footprint"])
        rows = [r for r, _ in padded]
        cols = [c for _, c in padded]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        candidates: list[tuple[int, int]] = []
        for r0 in range(-min_r, grid_h - max_r):
            for c0 in range(-min_c, grid_w - max_c):
                shifted = {(r0 + r, c0 + c) for r, c in padded}
                if shifted.isdisjoint(occupied):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        frame["top"] = r0
        frame["left"] = c0
        occupied |= {(r0 + r, c0 + c) for r, c in padded}
    return True


def _draw_input(grid: np.ndarray, frame: dict, color: int) -> None:
    r0 = int(frame["top"])
    c0 = int(frame["left"])
    height = int(frame["height"])
    width = int(frame["width"])
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    mid_r = r0 + height // 2
    mid_c = c0 + width // 2
    direction = tuple(frame["direction"])
    if direction == (-1, 0):
        bud = (r0 - 1, mid_c)
    elif direction == (0, 1):
        bud = (mid_r, c1 + 1)
    elif direction == (1, 0):
        bud = (r1 + 1, mid_c)
    else:
        bud = (mid_r, c0 - 1)
    grid[bud] = color

    for sym_color, sr, sc in frame["symbols"]:
        rr = r0 + 1 + sr
        cc = c0 + 1 + sc
        if sym_color == RED:
            grid[rr, cc : cc + 2] = RED
        else:
            grid[rr : rr + 2, cc] = BLUE


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    (h_lo, h_hi), (w_lo, w_hi) = GRID_RANGES[n_frames]
    grid_h = random.randint(h_lo, h_hi)
    grid_w = random.randint(w_lo, w_hi)
    directions = random.sample(DIRS, k=n_frames)
    colors = random.sample(FRAME_COLORS, k=n_frames)

    frames: list[dict] = []
    try:
        for direction in directions:
            frames.append(_sample_frame(direction))
    except RuntimeError:
        return None
    if not _place_frames(grid_h, grid_w, frames):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, frame in zip(colors, frames):
        frame["color"] = color
        _draw_input(grid, frame, color)

    parsed = _find_frames(grid)
    if len(parsed) != n_frames:
        return None
    parsed_by_color = {int(frame["color"]): frame for frame in parsed}
    for frame in frames:
        color = int(frame["color"])
        if color not in parsed_by_color:
            return None
        parsed_frame = parsed_by_color[color]
        if tuple(parsed_frame["direction"]) != tuple(frame["direction"]):
            return None
        if int(parsed_frame["h_count"]) != int(frame["h_count"]):
            return None
        if int(parsed_frame["v_count"]) != int(frame["v_count"]):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
