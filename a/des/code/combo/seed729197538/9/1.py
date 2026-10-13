import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400
MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 26
MIN_RECT_SIZE = 5
MAX_RECT_SIZE = 11
MAX_RECTS = 4
PLACEMENT_TRIES = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, pts))
    return comps


def _rect_frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = 2 * (height + width) - 4
    if len(points) != expected:
        return None

    point_set = set(points)
    for rr in range(r0, r1 + 1):
        if (rr, c0) not in point_set or (rr, c1) not in point_set:
            return None
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in point_set or (r1, cc) not in point_set:
            return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color, pts in _same_color_components(grid, bg):
        bbox = _rect_frame_bbox(pts)
        if bbox is not None:
            frames.append((color, bbox))
    frames.sort(key=lambda item: item[1])
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for border_color, (r0, c0, r1, c1) in _find_frames(g, bg):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_values = interior[interior != bg]
        assert marker_values.size > 0, "each frame must contain marker pixels"
        marker_colors = {int(v) for v in marker_values.tolist()}
        assert len(marker_colors) == 1, "all markers in one frame must share a color"
        marker_color = int(marker_values[0])
        assert marker_color != border_color, "marker color must differ from the original border"

        _paint_frame(out, (r0, c0, r1, c1), marker_color)
        out[r0 + 1 : r1, c0 + 1 : c1] = border_color

    return out


def _rect_count_for_area(rng: random.Random, area: int) -> int:
    max_count = 1
    if area >= 220:
        max_count = 2
    if area >= 320:
        max_count = 3
    if area >= 430:
        max_count = 4
    return rng.randint(1, min(MAX_RECTS, max_count))


def _valid_positions(occ: np.ndarray, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - rect_h):
        r1 = r0 + rect_h - 1
        for c0 in range(1, w - rect_w):
            c1 = c0 + rect_w - 1
            if occ[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            positions.append((r0, c0))
    return positions


def _marker_slots(rng: random.Random, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
    rng.shuffle(candidates)
    slots: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in slots):
            slots.append((r, c))
    return slots


def _choose_marker_positions(
    rng: random.Random, bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    slots = _marker_slots(rng, bbox)
    if not slots:
        return None
    cap = min(4, len(slots))
    if cap == 1 or rng.random() < 0.5:
        count = 1
    else:
        count = rng.randint(2, cap)
    rng.shuffle(slots)
    return slots[:count]


def _sample_marker_colors(
    rng: random.Random, border_colors: list[int], border_color: int
) -> int:
    other_borders = [c for c in border_colors if c != border_color]
    if other_borders and rng.random() < 0.5:
        return rng.choice(other_borders)
    return rng.choice([c for c in COLOR_POOL if c != border_color])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    target_rects = _rect_count_for_area(rng, height * width)

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    border_colors = rng.sample(list(COLOR_POOL), target_rects)
    max_side = min(MAX_RECT_SIZE, 12 - min(target_rects, 4), height - 2, width - 2)
    if max_side < MIN_RECT_SIZE:
        return None

    placed: list[tuple[int, tuple[int, int, int, int], int]] = []
    for border_color in border_colors:
        placed_this = False
        for _ in range(PLACEMENT_TRIES):
            rect_h = rng.randint(MIN_RECT_SIZE, max_side)
            rect_w = rng.randint(MIN_RECT_SIZE, max_side)
            positions = _valid_positions(occ, rect_h, rect_w)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            markers = _choose_marker_positions(rng, bbox)
            if markers is None:
                continue
            marker_color = _sample_marker_colors(rng, border_colors, border_color)

            _paint_frame(grid, bbox, border_color)
            _paint_frame(expected, bbox, marker_color)
            expected[r0 + 1 : r0 + rect_h - 1, c0 + 1 : c0 + rect_w - 1] = border_color
            for mr, mc in markers:
                grid[mr, mc] = marker_color

            occ[r0 - 1 : r0 + rect_h + 1, c0 - 1 : c0 + rect_w + 1] = True
            placed.append((border_color, bbox, marker_color))
            placed_this = True
            break
        if not placed_this:
            return None

    if len(placed) != target_rects:
        return None
    if np.count_nonzero(grid != BG) >= grid.size // 2:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if len(_find_frames(grid, BG)) != target_rects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
