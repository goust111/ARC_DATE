import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
FRAME = 5
WALL = 6

GEN_RETRIES = 400
ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _find_color_components(grid, FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME):
            continue
        if np.any(patch[~border] == FRAME):
            continue
        if len(coords) != int(border.sum()):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _interior(frame: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _reflect(frame: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = _interior(frame)
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    if inner_w > inner_h:
        return r, ic0 + ic1 - c
    return ir0 + ir1 - r, c


def _frame_source_cells(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = _interior(frame)
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    cells: list[tuple[int, int]] = []
    if inner_w > inner_h:
        mid = inner_w // 2
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic0 + mid):
                cells.append((r, c))
    else:
        mid = inner_h // 2
        for r in range(ir0, ir0 + mid):
            for c in range(ic0, ic1 + 1):
                cells.append((r, c))
    return cells


def _connected_shape(cells: list[tuple[int, int]], size: int) -> list[tuple[int, int]] | None:
    cell_set = set(cells)
    for _ in range(80):
        start = random.choice(cells)
        shape = {start}
        frontier = {nbr for nbr in neighbors4(*start) if nbr in cell_set}
        while len(shape) < size and frontier:
            nxt = random.choice(list(frontier))
            shape.add(nxt)
            frontier = {nbr for nbr in frontier if nbr not in shape}
            for nbr in neighbors4(*nxt):
                if nbr in cell_set and nbr not in shape:
                    frontier.add(nbr)
        if len(shape) != size:
            continue
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(shape):
            continue
        return sorted(shape)
    return None


def _touches_blue(grid: np.ndarray, r: int, c: int) -> bool:
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BLUE:
            return True
    return False


def _place_frame_contents(grid: np.ndarray, frame: tuple[int, int, int, int]) -> bool:
    source_cells = _frame_source_cells(frame)
    max_size = min(8, len(source_cells) - 1)
    if max_size < 3:
        return False

    shape_size = random.randint(3, max_size)
    blue_cells = _connected_shape(source_cells, shape_size)
    if blue_cells is None:
        return False

    targets = {cell: _reflect(frame, *cell) for cell in blue_cells}
    blocked_count = random.randint(1, min(3, len(blue_cells) - 1))
    blocked_sources = set(random.sample(blue_cells, blocked_count))
    reserved_open_targets = {targets[cell] for cell in blue_cells if cell not in blocked_sources}

    for r, c in blue_cells:
        grid[r, c] = BLUE

    for src in blocked_sources:
        tr, tc = targets[src]
        if _touches_blue(grid, tr, tc):
            return False
        grid[tr, tc] = WALL

    # Some blocked targets extend into short wall segments, but never onto open reflection targets.
    for src in blocked_sources:
        if random.random() < 0.45:
            continue
        tr, tc = targets[src]
        axis = random.choice([(0, 1), (1, 0)])
        sign = random.choice([-1, 1])
        extra_len = random.randint(1, 2)
        segment: list[tuple[int, int]] = []
        for step in range(1, extra_len + 1):
            nr = tr + axis[0] * sign * step
            nc = tc + axis[1] * sign * step
            ir0, ic0, ir1, ic1 = _interior(frame)
            if not (ir0 <= nr <= ir1 and ic0 <= nc <= ic1):
                segment = []
                break
            if (nr, nc) in reserved_open_targets or int(grid[nr, nc]) != BG or _touches_blue(grid, nr, nc):
                segment = []
                break
            segment.append((nr, nc))
        for nr, nc in segment:
            grid[nr, nc] = WALL

    return True


def _frame_dims_for_sector(sector_h: int, sector_w: int, orientation: str) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for inner_h in range(3, sector_h - 1):
        for inner_w in range(3, sector_w - 1):
            if inner_h == inner_w:
                continue
            if orientation == "wide":
                if inner_w % 2 != 0 or inner_w <= inner_h:
                    continue
                if inner_h * (inner_w // 2) < 6:
                    continue
            else:
                if inner_h % 2 != 0 or inner_h <= inner_w:
                    continue
                if (inner_h // 2) * inner_w < 6:
                    continue
            candidates.append((inner_h, inner_w))
    if not candidates:
        return None
    weights = [4 if abs(h - w) <= 2 else 2 if abs(h - w) <= 4 else 1 for h, w in candidates]
    idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
    return candidates[idx]


def _make_sectors(h: int, w: int) -> list[tuple[int, int, int, int]]:
    avail_h = h - 3
    avail_w = w - 3
    top_h = random.randint(6, avail_h - 6)
    left_w = random.randint(6, avail_w - 6)
    bottom_h = avail_h - top_h
    right_w = avail_w - left_w
    row_starts = [1, 1 + top_h + 1]
    col_starts = [1, 1 + left_w + 1]
    heights = [top_h, bottom_h]
    widths = [left_w, right_w]
    sectors = []
    for i in range(2):
        for j in range(2):
            sectors.append((row_starts[i], col_starts[j], heights[i], widths[j]))
    return sectors


def _place_frame_outline(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _generate_attempt() -> np.ndarray | None:
    frame_count = random.choice([3, 4])
    h = random.randint(19 if frame_count == 4 else 18, 24)
    w = random.randint(19 if frame_count == 4 else 18, 24)
    grid = np.full((h, w), BG, dtype=int)

    sectors = _make_sectors(h, w)
    chosen_sector_ids = random.sample(range(4), frame_count)
    orientations = ["wide", "tall"] + [random.choice(["wide", "tall"]) for _ in range(frame_count - 2)]
    random.shuffle(orientations)
    frames: list[tuple[int, int, int, int]] = []

    for sector_id, orientation in zip(chosen_sector_ids, orientations):
        sr, sc, sh, sw = sectors[sector_id]
        dims = _frame_dims_for_sector(sh, sw, orientation)
        if dims is None:
            return None
        inner_h, inner_w = dims
        ext_h = inner_h + 2
        ext_w = inner_w + 2
        r0 = random.randint(sr, sr + sh - ext_h)
        c0 = random.randint(sc, sc + sw - ext_w)
        frame = (r0, c0, r0 + ext_h - 1, c0 + ext_w - 1)
        _place_frame_outline(grid, frame)
        frames.append(frame)

    for frame in frames:
        if not _place_frame_contents(grid, frame):
            return None

    if not _is_valid_input(grid):
        return None
    return grid


def _is_connected_blue(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    target = set(coords)
    seen = {coords[0]}
    q = deque([coords[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == target


def _is_valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    frames = _find_frames(grid)
    if len(frames) not in {3, 4}:
        return False

    wide = tall = 0
    for frame in frames:
        ir0, ic0, ir1, ic1 = _interior(frame)
        inner = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        inner_h, inner_w = inner.shape
        if inner_w > inner_h:
            if inner_w % 2 != 0:
                return False
            wide += 1
        elif inner_h > inner_w:
            if inner_h % 2 != 0:
                return False
            tall += 1
        else:
            return False

        blues = [(ir0 + r, ic0 + c) for r, c in np.argwhere(inner == BLUE)]
        if not (3 <= len(blues) <= 8):
            return False
        if not _is_connected_blue(grid, blues):
            return False

        blocked = 0
        open_ = 0
        for r, c in blues:
            tr, tc = _reflect(frame, r, c)
            target = int(grid[tr, tc])
            if target == WALL:
                blocked += 1
            elif target == BG:
                open_ += 1
            else:
                return False

        if blocked < 1 or open_ < 1:
            return False

        for r, c in np.argwhere(inner == WALL):
            rr = ir0 + int(r)
            cc = ic0 + int(c)
            if _touches_blue(grid, rr, cc):
                return False

    if wide < 1 or tall < 1:
        return False

    out = main(grid)
    return bool(np.any(out != grid))


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each blue motif inside a grey room unless its reflected destination is occupied by an internal wall, in which case erase the source pixel.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick rectangular frames define separate rooms.
    - Each room contains a connected irregular blue motif on one side and magenta wall cells inside the room.

    Rule:
    - For each grey frame, inspect only its interior and compare the interior width and height.
    - Wide interiors mirror blue cells across the vertical midline; tall interiors mirror blue cells across the horizontal midline.
    - If a blue cell's reflected destination is black, add a blue copy there and keep the source.
    - If the reflected destination is a magenta wall, erase the source blue cell instead; grey frames and magenta walls stay unchanged.

    Key insight:
    - The frame aspect ratio chooses the mirror axis, but walls act as per-pixel blockers on the reflected destinations, so the same room can simultaneously create mirrored copies and erase blocked sources.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        ir0, ic0, ir1, ic1 = _interior(frame)
        blue_local = np.argwhere(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == BLUE)
        for lr, lc in blue_local:
            r = ir0 + int(lr)
            c = ic0 + int(lc)
            tr, tc = _reflect(frame, r, c)
            target = int(g[tr, tc])
            if target == WALL:
                out[r, c] = BG
            elif target == BG:
                out[tr, tc] = BLUE

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid selective-mirror input")
