from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BAR_COLORS = [1, 2, 3, 4]
OBJECT_COLORS = [5, 6, 7, 8, 9]

SURVIVOR_SHAPES_RIGHT = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)],
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalize_shape(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _find_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    for r in range(h):
        row = grid[r]
        gaps = np.where(row == bg)[0]
        if len(gaps) != 1:
            continue
        gap = int(gaps[0])
        colors = {int(v) for v in row if int(v) != bg}
        if len(colors) != 1:
            continue
        color = colors.pop()
        if int(np.sum(row == color)) != w - 1:
            continue
        bars.append({"row": r, "gap_col": gap, "color": color})
    assert 3 <= len(bars) <= 4, "expected 3-4 horizontal bars"
    assert len({bar["gap_col"] for bar in bars}) == len(bars), "gap columns must be unique"
    return sorted(bars, key=lambda bar: bar["row"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase ray-hit lower objects and refill each bar gap from the nearest surviving color below it.

    Input structure:
    - The background is black, with 3-4 full-width horizontal bars in the upper half.
    - Each bar is monochrome except for one single-cell gap, and different bars use different gap columns.
    - The lower half contains several small monochrome 4-connected target objects that do not touch the bars.

    Rule:
    - From every bar gap, cast a one-cell-wide vertical ray downward starting on the next row.
    - Consider all rays simultaneously: any lower 4-connected object touched by any ray is erased completely.
    - After all erasures, process each gap independently: scan downward row by row while ignoring the other bar rows, and on the first lower row that still contains foreground, copy the color of the closest surviving object pixel in that row into the gap.
    - If no surviving lower-object pixel remains below a gap, leave that gap as background.

    Key insight:
    - The gaps both emit destructive rays and later receive a color sample, so shared hit objects must be removed globally before any gap can know which surviving color is now directly below it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = _get_bg(g)
    bars = _find_bars(g, bg)
    gap_rows = {bar["gap_col"]: bar["row"] for bar in bars}
    bar_rows = {bar["row"] for bar in bars}

    lower = g.copy()
    for r in bar_rows:
        lower[r, :] = bg

    hit_mask = np.zeros(g.shape, dtype=bool)
    lower_mask = lower != bg
    for coords in _components(lower_mask):
        touched = False
        for r, c in coords:
            c = int(c)
            if c in gap_rows and int(r) > int(gap_rows[c]):
                touched = True
                break
        if touched:
            hit_mask[coords[:, 0], coords[:, 1]] = True

    out = g.copy()
    out[hit_mask] = bg

    scan_grid = out.copy()
    for r in bar_rows:
        scan_grid[r, :] = bg

    for bar in bars:
        gap_r = int(bar["row"])
        gap_c = int(bar["gap_col"])
        fill = bg
        for r in range(gap_r + 1, g.shape[0]):
            row_cols = [int(c) for c in np.where(scan_grid[r] != bg)[0]]
            if not row_cols:
                continue
            best_c = min(row_cols, key=lambda c: (abs(c - gap_c), c))
            fill = int(scan_grid[r, best_c])
            break
        out[gap_r, gap_c] = fill

    return out


def _mirror_shape(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_c = max(c for _, c in shape)
    mirrored = [(r, max_c - c) for r, c in shape]
    min_c = min(c for _, c in mirrored)
    return [(r, c - min_c) for r, c in mirrored]


def _shared_hit_shape(c1: int, c2: int, tail_right: bool) -> list[tuple[int, int]]:
    cells = [(0, c) for c in range(c1, c2 + 1)]
    cells.append((1, c2 if tail_right else c1))
    return cells


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                return False
    return True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _choose_bar_rows(bar_count: int) -> list[int]:
    start = random.randint(1, 2)
    return [start + 2 * i for i in range(bar_count)]


def _choose_gap_columns(bar_count: int, width: int) -> tuple[list[int], list[tuple[int, int]]]:
    interior = list(range(2, width - 2))
    if bar_count == 3:
        for _ in range(200):
            c1 = random.choice(interior[:-4])
            d = random.randint(2, 4)
            c2 = c1 + d
            if c2 not in interior:
                continue
            third_choices = [c for c in interior if c not in {c1, c2} and abs(c - c1) >= 3 and abs(c - c2) >= 3]
            if not third_choices:
                continue
            c3 = random.choice(third_choices)
            cols = [c1, c2, c3]
            random.shuffle(cols)
            return cols, [(c1, c2)]
        raise RuntimeError("failed to choose 3 valid gap columns")

    for _ in range(200):
        left_start = random.randint(2, max(2, width - 12))
        d1 = random.randint(2, 4)
        p1 = (left_start, left_start + d1)
        right_start_min = p1[1] + 3
        right_start_max = width - 3 - 2
        if right_start_min > right_start_max:
            continue
        right_start = random.randint(right_start_min, right_start_max)
        d2 = random.randint(2, 4)
        p2 = (right_start, right_start + d2)
        if p2[1] > width - 3:
            continue
        cols = [p1[0], p1[1], p2[0], p2[1]]
        random.shuffle(cols)
        return cols, [p1, p2]
    raise RuntimeError("failed to choose 4 valid gap columns")


def _survivor_side(gap: int, pairs: list[tuple[int, int]], width: int) -> str:
    for a, b in pairs:
        if gap == a:
            return "left"
        if gap == b:
            return "right"
    return random.choice(["left", "right"]) if 3 <= gap <= width - 4 else ("right" if gap < width // 2 else "left")


def _shape_width(shape: list[tuple[int, int]]) -> int:
    return 1 + max(c for _, c in shape)


def _translate(shape: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in shape]


def _build_attempt() -> np.ndarray | None:
    bar_count = random.randint(3, 4)
    height = random.randint(15, 19)
    width = random.randint(14, 18)
    bg = 0

    bar_rows = _choose_bar_rows(bar_count)
    if bar_rows[-1] + 8 >= height:
        return None

    gap_cols, hit_pairs = _choose_gap_columns(bar_count, width)
    bar_colors = random.sample(BAR_COLORS, bar_count)
    bars = list(zip(bar_rows, gap_cols, bar_colors, strict=True))

    grid = np.zeros((height, width), dtype=int)
    for r, gap_c, color in bars:
        grid[r, :] = color
        grid[r, gap_c] = bg

    object_colors = random.sample(OBJECT_COLORS, min(len(OBJECT_COLORS), bar_count + len(hit_pairs)))
    while len(object_colors) < bar_count + len(hit_pairs):
        object_colors.append(random.choice(OBJECT_COLORS))
    random.shuffle(object_colors)

    hit_row = bar_rows[-1] + 2
    survivor_row = hit_row + 4
    if survivor_row + 3 >= height:
        return None

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    color_idx = 0

    for pair_index, (c1, c2) in enumerate(hit_pairs):
        cells = _shared_hit_shape(c1, c2, tail_right=bool(pair_index % 2))
        sig = _normalize_shape(cells)
        if sig in used_shapes:
            return None
        row = hit_row + (pair_index % 2)
        cells = _translate(cells, row, 0)
        if not _can_place(grid, cells, bg):
            return None
        _paint(grid, cells, object_colors[color_idx])
        color_idx += 1
        used_shapes.add(sig)

    survivor_shapes = SURVIVOR_SHAPES_RIGHT[:]
    random.shuffle(survivor_shapes)
    gap_to_bar_row = {gap_c: r for r, gap_c, _ in bars}
    sorted_gaps = sorted(gap_to_bar_row)
    for idx, gap_c in enumerate(sorted_gaps):
        if not survivor_shapes:
            return None
        shape = survivor_shapes.pop()
        side = _survivor_side(gap_c, hit_pairs, width)
        if side == "left":
            shape = _mirror_shape(shape)
            anchor_c = gap_c - _shape_width(shape)
        else:
            anchor_c = gap_c + 1
        row = survivor_row + (idx % 2)
        cells = _translate(shape, row, anchor_c)
        sig = _normalize_shape(cells)
        if sig in used_shapes:
            return None
        if any(c in gap_to_bar_row for _, c in cells):
            return None
        if not _can_place(grid, cells, bg):
            return None
        _paint(grid, cells, object_colors[color_idx])
        color_idx += 1
        used_shapes.add(sig)

    lower = grid.copy()
    for r in bar_rows:
        lower[r, :] = bg
    comps = _components(lower != bg)
    if not (4 <= len(comps) <= 6):
        return None

    gap_set = set(gap_to_bar_row)
    hit_per_gap = {c: 0 for c in gap_set}
    survivor_per_gap = {c: 0 for c in gap_set}
    for coords in comps:
        cols = {int(c) for _, c in coords}
        is_hit = any(c in gap_to_bar_row and any(int(r) > gap_to_bar_row[c] for r, cc in coords if int(cc) == c) for c in cols if c in gap_to_bar_row)
        for gap_c in gap_set:
            if gap_c in cols:
                hit_per_gap[gap_c] += 1
            else:
                nearest = min(abs(int(c) - gap_c) for _, c in coords)
                if nearest == 1:
                    survivor_per_gap[gap_c] += 1
        if len(cols & gap_set) >= 2:
            shared = True
        else:
            shared = False
        _ = shared

    if not any(len({int(c) for _, c in comp} & gap_set) >= 2 for comp in comps):
        return None
    if any(hit_per_gap[c] < 1 or survivor_per_gap[c] < 1 for c in gap_set):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
