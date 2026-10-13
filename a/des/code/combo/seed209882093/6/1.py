import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


Grid = np.ndarray
Point = Tuple[int, int]


def _components(mask: np.ndarray) -> List[List[Point]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Point]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Point] = deque([(r, c)])
            comp: List[Point] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_legend_pairs(values: np.ndarray) -> Optional[Dict[int, int]]:
    if values.size < 4 or values.size % 2 != 0 or np.any(values == 0):
        return None
    mapping: Dict[int, int] = {}
    for i in range(0, values.size, 2):
        border = int(values[i])
        fill = int(values[i + 1])
        if border == fill or border in mapping:
            return None
        mapping[border] = fill
    if len(mapping) < 2:
        return None
    return mapping


def _legend_candidates(grid: Grid) -> List[Tuple[str, int, Dict[int, int]]]:
    h, w = grid.shape
    candidates: List[Tuple[str, int, Dict[int, int]]] = []

    for r in range(h):
        nz = np.flatnonzero(grid[r] != 0)
        if nz.size < 4 or nz.size % 2 != 0:
            continue
        if int(nz[-1]) - int(nz[0]) + 1 != int(nz.size):
            continue
        start, end = int(nz[0]), int(nz[-1])
        clear = True
        for c in range(start, end + 1):
            if r > 0 and grid[r - 1, c] != 0:
                clear = False
                break
            if r + 1 < h and grid[r + 1, c] != 0:
                clear = False
                break
        if not clear:
            continue
        mapping = _parse_legend_pairs(grid[r, start : end + 1])
        if mapping is not None:
            candidates.append(("row", r, mapping))

    for c in range(w):
        nz = np.flatnonzero(grid[:, c] != 0)
        if nz.size < 4 or nz.size % 2 != 0:
            continue
        if int(nz[-1]) - int(nz[0]) + 1 != int(nz.size):
            continue
        start, end = int(nz[0]), int(nz[-1])
        clear = True
        for r in range(start, end + 1):
            if c > 0 and grid[r, c - 1] != 0:
                clear = False
                break
            if c + 1 < w and grid[r, c + 1] != 0:
                clear = False
                break
        if not clear:
            continue
        mapping = _parse_legend_pairs(grid[start : end + 1, c])
        if mapping is not None:
            candidates.append(("col", c, mapping))

    return candidates


def _find_legend(grid: Grid) -> Tuple[str, int, Dict[int, int]]:
    candidates = _legend_candidates(grid)
    if not candidates:
        raise ValueError("legend line not found")
    best: Optional[Tuple[str, int, Dict[int, int]]] = None
    best_score = (-1, -1)
    for orientation, index, mapping in candidates:
        frame_count = len(_find_frames(_erase_legend_line(grid, orientation, index)))
        score = (frame_count, len(mapping))
        if score > best_score:
            best = (orientation, index, mapping)
            best_score = score
    assert best is not None
    return best


def _erase_legend_line(grid: Grid, orientation: str, index: int) -> Grid:
    out = grid.copy()
    if orientation == "row":
        out[index, :] = 0
    else:
        out[:, index] = 0
    return out


def _perimeter_points(r0: int, r1: int, c0: int, c1: int) -> List[Point]:
    pts: List[Point] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _find_frames(grid: Grid) -> List[Tuple[int, int, int, int, int, int, int]]:
    frames: List[Tuple[int, int, int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if v != 0):
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            perimeter = _perimeter_points(r0, r1, c0, c1)
            comp_set = set(comp)
            if any(pt not in comp_set for pt in perimeter):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            nonzero = np.argwhere(interior != 0)
            if nonzero.shape[0] != 1:
                continue
            sr, sc = (int(v) for v in nonzero[0])
            seed = (r0 + 1 + sr, c0 + 1 + sc)
            extra = comp_set.difference(perimeter)
            if extra and extra != {seed}:
                continue
            frames.append((r0, r1, c0, c1, color, seed[0], seed[1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    orientation, index, mapping = _find_legend(g)
    out = _erase_legend_line(g, orientation, index)

    work = out.copy()
    for r0, r1, c0, c1, border_color, seed_r, seed_c in _find_frames(work):
        fill_color = mapping.get(border_color)
        if fill_color is None:
            continue
        if int(work[seed_r, seed_c]) != border_color:
            continue
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if work[r, c] == 0:
                    out[r, c] = fill_color
    return out


def _reserve_legend_band(blocked: np.ndarray, orientation: str, index: int) -> None:
    h, w = blocked.shape
    if orientation == "row":
        blocked[max(0, index - 1) : min(h, index + 2), :] = True
    else:
        blocked[:, max(0, index - 1) : min(w, index + 2)] = True


def _place_frame(
    grid: Grid,
    blocked: np.ndarray,
    rng: random.Random,
    border_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape
    max_h = min(8, h - 2)
    max_w = min(8, w - 2)
    if max_h < 5 or max_w < 5:
        return False

    for _ in range(200):
        frame_h = rng.randint(5, max_h)
        frame_w = rng.randint(5, max_w)
        r0 = rng.randint(1, h - frame_h - 1)
        c0 = rng.randint(1, w - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r1 + 2)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c1 + 2)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue

        for r, c in _perimeter_points(r0, r1, c0, c1):
            grid[r, c] = border_color

        seed_r = rng.randint(r0 + 2, r1 - 2)
        seed_c = rng.randint(c0 + 2, c1 - 2)
        grid[seed_r, seed_c] = seed_color
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> Optional[Grid]:
    num_frames = rng.randint(2, 4)
    side_min = 12 + num_frames * 2
    h = rng.randint(side_min, 23)
    w = rng.randint(side_min, 24)
    orientation = rng.choice(["row", "col"])
    if orientation == "row" and h < 7:
        return None
    if orientation == "col" and w < 7:
        return None

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_index = rng.randint(2, h - 3) if orientation == "row" else rng.randint(2, w - 3)
    _reserve_legend_band(blocked, orientation, legend_index)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_colors = [palette.pop() for _ in range(num_frames)]
    fill_colors = [palette.pop() for _ in range(num_frames)]
    mismatch_color = palette.pop()

    active = [False] * num_frames
    on_index = rng.randrange(num_frames)
    off_index = rng.randrange(num_frames - 1)
    if off_index >= on_index:
        off_index += 1
    active[on_index] = True
    for i in range(num_frames):
        if i in (on_index, off_index):
            continue
        active[i] = rng.random() < 0.5

    for border_color, is_active in zip(border_colors, active):
        seed_color = border_color if is_active else mismatch_color
        if not _place_frame(grid, blocked, rng, border_color, seed_color):
            return None

    order = list(range(num_frames))
    rng.shuffle(order)
    span_len = 2 * num_frames
    if orientation == "row":
        if w < span_len + 2:
            return None
        start = rng.randint(1, w - span_len - 1)
        for pos, idx in enumerate(order):
            grid[legend_index, start + 2 * pos] = border_colors[idx]
            grid[legend_index, start + 2 * pos + 1] = fill_colors[idx]
    else:
        if h < span_len + 2:
            return None
        start = rng.randint(1, h - span_len - 1)
        for pos, idx in enumerate(order):
            grid[start + 2 * pos, legend_index] = border_colors[idx]
            grid[start + 2 * pos + 1, legend_index] = fill_colors[idx]

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    candidates = _legend_candidates(grid)
    if len(candidates) != 1:
        return None
    work = _erase_legend_line(grid, orientation, legend_index)
    if len(_find_frames(work)) != num_frames:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
