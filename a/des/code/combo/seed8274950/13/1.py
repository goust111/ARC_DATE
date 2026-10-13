import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_H_RANGE = (11, 22)
GRID_W_RANGE = (11, 22)
MIN_ROOM_SIZE = 3
MAX_LABEL_REGIONS = 6
MAX_GENERATION_ATTEMPTS = 300
GRID_CUT_PATTERNS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 1), (1, 2), (2, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_boundary(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)


def _detect_regions(grid: np.ndarray) -> tuple[int, int, list[tuple[list[tuple[int, int]], tuple[int, int], int]]]:
    """
    Infer wall/background colors and labeled regions.

    A valid wall color is a single 4-connected structure such that every non-wall
    component either:
    - is a labeled region with exactly one non-background pixel, or
    - is an unlabeled boundary-touching outside area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = Counter(int(v) for v in g.flat)
    candidates = []
    for wall_color in counts:
        wall_mask = g == wall_color
        if len(_components(wall_mask)) != 1:
            continue

        remaining = [color for color in counts if color != wall_color]
        if len(remaining) < 2:
            continue
        bg_color = max(remaining, key=lambda color: counts[color])

        labeled_regions: list[tuple[list[tuple[int, int]], tuple[int, int], int]] = []
        ok = True
        for comp in _components(~wall_mask):
            label_cells = [(r, c) for r, c in comp if int(g[r, c]) != bg_color]
            bg_cells = [(r, c) for r, c in comp if int(g[r, c]) == bg_color]
            if len(label_cells) == 0:
                if _touches_boundary(comp, g.shape):
                    continue
                ok = False
                break
            if len(label_cells) != 1 or not bg_cells:
                ok = False
                break
            lr, lc = label_cells[0]
            labeled_regions.append((comp, (int(lr), int(lc)), int(g[lr, lc])))

        if ok and len(labeled_regions) >= 2:
            candidates.append((int(wall_color), int(bg_color), labeled_regions))

    assert len(candidates) == 1, "grid must admit a unique wall/background interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find background regions separated by a wall color, fill each region with its
    single-pixel label color, and erase the label back to background.
    """
    g = np.asarray(grid, dtype=int)
    _wall_color, bg_color, regions = _detect_regions(g)

    out = g.copy()
    changed = 0
    for comp, label_cell, label_color in regions:
        lr, lc = label_cell
        for r, c in comp:
            if int(g[r, c]) == bg_color:
                out[r, c] = label_color
                changed += 1
        out[lr, lc] = bg_color

    assert changed > 0, "expected at least one background cell to be filled"
    return out


def _sample_lengths(total: int, parts: int, min_size: int) -> list[int] | None:
    slack = total - parts * min_size
    if slack < 0:
        return None
    lengths = [min_size] * parts
    for _ in range(slack):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _intervals_from_lengths(start: int, lengths: list[int]) -> list[tuple[int, int]]:
    intervals = []
    cursor = start
    for idx, length in enumerate(lengths):
        intervals.append((cursor, cursor + length - 1))
        cursor += length
        if idx != len(lengths) - 1:
            cursor += 1
    return intervals


def _make_canvas(h: int, w: int, bg_color: int, wall_color: int) -> np.ndarray:
    grid = np.full((h, w), bg_color, dtype=int)
    grid[0, :] = wall_color
    grid[-1, :] = wall_color
    grid[:, 0] = wall_color
    grid[:, -1] = wall_color
    return grid


def _grid_partition(
    grid: np.ndarray,
    wall_color: int,
    horizontal_cuts: int,
    vertical_cuts: int,
) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    row_lengths = _sample_lengths(h - 2 - horizontal_cuts, horizontal_cuts + 1, MIN_ROOM_SIZE)
    col_lengths = _sample_lengths(w - 2 - vertical_cuts, vertical_cuts + 1, MIN_ROOM_SIZE)
    if row_lengths is None or col_lengths is None:
        return None

    row_intervals = _intervals_from_lengths(1, row_lengths)
    col_intervals = _intervals_from_lengths(1, col_lengths)

    for upper, lower in zip(row_intervals, row_intervals[1:]):
        cut_r = upper[1] + 1
        grid[cut_r, 1:-1] = wall_color
    for left, right in zip(col_intervals, col_intervals[1:]):
        cut_c = left[1] + 1
        grid[1:-1, cut_c] = wall_color

    rooms = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            rooms.append((r0, r1, c0, c1))
    return rooms


def _bsp_partition(grid: np.ndarray, wall_color: int) -> list[tuple[int, int, int, int]] | None:
    rooms = [(1, grid.shape[0] - 2, 1, grid.shape[1] - 2)]
    target = random.randint(2, MAX_LABEL_REGIONS)

    for _ in range(target * 4):
        if len(rooms) >= target:
            break
        splittable = []
        for idx, (r0, r1, c0, c1) in enumerate(rooms):
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height >= 2 * MIN_ROOM_SIZE + 1:
                splittable.append((idx, "h"))
            if width >= 2 * MIN_ROOM_SIZE + 1:
                splittable.append((idx, "v"))
        if not splittable:
            break

        idx, orient = random.choice(splittable)
        r0, r1, c0, c1 = rooms.pop(idx)
        if orient == "h":
            lengths = _sample_lengths((r1 - r0 + 1) - 1, 2, MIN_ROOM_SIZE)
            if lengths is None:
                return None
            top_h, bottom_h = lengths
            cut_r = r0 + top_h
            grid[cut_r, c0 : c1 + 1] = wall_color
            rooms.append((r0, cut_r - 1, c0, c1))
            rooms.append((cut_r + 1, r1, c0, c1))
        else:
            lengths = _sample_lengths((c1 - c0 + 1) - 1, 2, MIN_ROOM_SIZE)
            if lengths is None:
                return None
            left_w, right_w = lengths
            cut_c = c0 + left_w
            grid[r0 : r1 + 1, cut_c] = wall_color
            rooms.append((r0, r1, c0, cut_c - 1))
            rooms.append((r0, r1, cut_c + 1, c1))

    return rooms if len(rooms) >= 2 else None


def _label_position(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = room
    inner_rows = list(range(r0 + 1, r1)) or list(range(r0, r1 + 1))
    inner_cols = list(range(c0 + 1, c1)) or list(range(c0, c1 + 1))
    if random.random() < 0.75:
        return random.choice(inner_rows), random.choice(inner_cols)
    return random.randint(r0, r1), random.randint(c0, c1)


def _paint_labels(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    bg_color: int,
    wall_color: int,
) -> None:
    room_count = len(rooms)
    palette = [color for color in range(10) if color not in (bg_color, wall_color)]
    random.shuffle(palette)
    assert room_count <= len(palette)

    for label_color, room in zip(palette[:room_count], rooms):
        lr, lc = _label_position(room)
        assert grid[lr, lc] == bg_color
        grid[lr, lc] = label_color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    colors = list(range(10))
    bg_color, wall_color = random.sample(colors, 2)
    grid = _make_canvas(h, w, bg_color, wall_color)

    mode = random.choice(("grid", "bsp"))
    if mode == "grid":
        horizontal_cuts, vertical_cuts = random.choice(GRID_CUT_PATTERNS)
        rooms = _grid_partition(grid, wall_color, horizontal_cuts, vertical_cuts)
    else:
        rooms = _bsp_partition(grid, wall_color)
    if rooms is None:
        return None
    if len(rooms) > MAX_LABEL_REGIONS:
        return None

    _paint_labels(grid, rooms, bg_color, wall_color)

    try:
        _, detected_bg, regions = _detect_regions(grid)
    except AssertionError:
        return None

    if detected_bg != bg_color or len(regions) != len(rooms):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
