import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 9)

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (16, 30)
PALETTE_LEN_RANGE = (2, 6)
FRAME_COUNT_RANGE = (2, 4)
FRAME_INNER_H_RANGE = (2, 5)
FRAME_INNER_W_RANGE = (2, 6)
PLACEMENT_TRIES = 150
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_palette(grid: np.ndarray) -> np.ndarray:
    row = grid[-1]
    cols = np.flatnonzero(row != BLACK)
    assert cols.size > 0, "expected a bottom palette strip"
    c0, c1 = int(cols[0]), int(cols[-1])
    assert np.all(row[c0 : c1 + 1] != BLACK), "palette strip must be contiguous"
    palette = row[c0 : c1 + 1].astype(int)
    assert len(set(map(int, palette.tolist()))) == len(palette), "palette colors must be distinct"
    return palette


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None
    return r0, c0, r1, c1


def _detect_frames(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    frames: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for coords in _components(grid == GREY):
        bbox = _frame_bbox(coords)
        if bbox is not None:
            frames.append((coords, bbox))
    return frames


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame using the bottom palette color indexed by its interior azure count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _find_palette(g)
    out = g.copy()
    frames = _detect_frames(g)
    assert frames, "expected at least one grey frame"

    for coords, (r0, c0, r1, c1) in frames:
        assert r1 < g.shape[0] - 1, "frames must sit above the palette row"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert not np.any((inner != BLACK) & (inner != AZURE)), "frame interiors must contain only bg or azure markers"
        count = int(np.count_nonzero(inner == AZURE))
        assert 1 <= count <= len(palette), "marker count must index the palette"
        color = int(palette[count - 1])
        out[coords[:, 0], coords[:, 1]] = color
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[inner == AZURE] = BLACK

    return out


def _build_expected(grid: np.ndarray, specs: list[tuple[int, int, int, int, int]], palette: list[int]) -> np.ndarray:
    expected = grid.copy()
    for r0, c0, r1, c1, count in specs:
        color = int(palette[count - 1])
        _paint_frame(expected, int(r0), int(c0), int(r1), int(c1), color)
        expected[r0 + 1 : r1, c0 + 1 : c1][expected[r0 + 1 : r1, c0 + 1 : c1] == AZURE] = BLACK
    return expected


def _sample_frame_dims(count: int, h: int, w: int) -> tuple[int, int] | None:
    max_inner_h = min(FRAME_INNER_H_RANGE[1], h - 4)
    max_inner_w = min(FRAME_INNER_W_RANGE[1], w - 2)
    if max_inner_h < FRAME_INNER_H_RANGE[0] or max_inner_w < FRAME_INNER_W_RANGE[0]:
        return None

    for _ in range(40):
        inner_h = random.randint(FRAME_INNER_H_RANGE[0], max_inner_h)
        min_inner_w = max(FRAME_INNER_W_RANGE[0], (count + inner_h - 1) // inner_h)
        if min_inner_w > max_inner_w:
            continue
        inner_w = random.randint(min_inner_w, max_inner_w)
        return int(inner_h), int(inner_w)
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = map(int, occupied.shape)
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(h - 1, int(r1) + 1)
    cc1 = min(w - 1, int(c1) + 1)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_one_frame(grid: np.ndarray, occupied: np.ndarray, count: int) -> tuple[int, int, int, int] | None:
    h, w = map(int, grid.shape)
    for _ in range(PLACEMENT_TRIES):
        dims = _sample_frame_dims(int(count), h, w)
        if dims is None:
            return None
        inner_h, inner_w = dims
        frame_h = int(inner_h + 2)
        frame_w = int(inner_w + 2)
        max_r0 = h - frame_h - 2
        max_c0 = w - frame_w
        if max_r0 < 0 or max_c0 < 0:
            continue
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)
        r1 = int(r0 + frame_h - 1)
        c1 = int(c0 + frame_w - 1)
        if not _can_place(occupied, int(r0), int(c0), int(r1), int(c1)):
            continue

        _paint_frame(grid, int(r0), int(c0), int(r1), int(c1), GREY)
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        markers = random.sample(interior, int(count))
        for mr, mc in markers:
            grid[int(mr), int(mc)] = AZURE
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _validate_generated(grid: np.ndarray, expected: np.ndarray, frame_count: int) -> bool:
    try:
        palette = _find_palette(grid)
        frames = _detect_frames(grid)
        if len(frames) != frame_count:
            return False
        counts = set()
        frame_cells = 0
        for coords, (r0, c0, r1, c1) in frames:
            frame_cells += int(len(coords))
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if np.any((inner != BLACK) & (inner != AZURE)):
                return False
            count = int(np.count_nonzero(inner == AZURE))
            if count in counts or count < 1 or count > len(palette):
                return False
            counts.add(count)
        if frame_cells != int(np.count_nonzero(grid == GREY)):
            return False
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    palette_len_max = min(PALETTE_LEN_RANGE[1], len(PALETTE_COLORS), w)
    if palette_len_max < PALETTE_LEN_RANGE[0]:
        return None
    palette_len = random.randint(PALETTE_LEN_RANGE[0], palette_len_max)
    frame_count = random.randint(FRAME_COUNT_RANGE[0], min(FRAME_COUNT_RANGE[1], palette_len))

    palette = random.sample(PALETTE_COLORS, palette_len)
    counts = sorted(random.sample(list(range(1, palette_len + 1)), frame_count), reverse=True)

    grid = np.zeros((h, w), dtype=int)
    palette_c0 = random.randint(0, w - palette_len)
    grid[-1, palette_c0 : palette_c0 + palette_len] = np.array(palette, dtype=int)

    occupied = np.zeros((h - 1, w), dtype=bool)
    specs: list[tuple[int, int, int, int, int]] = []
    for count in counts:
        placed = _place_one_frame(grid, occupied, int(count))
        if placed is None:
            return None
        r0, c0, r1, c1 = placed
        specs.append((int(r0), int(c0), int(r1), int(c1), int(count)))

    expected = _build_expected(grid, specs, palette)
    if not _validate_generated(grid, expected, frame_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
