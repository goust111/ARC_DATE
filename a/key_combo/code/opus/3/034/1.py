from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, find_components as _arc_find_components, get_bg as _arc_get_bg
except Exception:  # pragma: no cover - fallback for standalone loading
    _arc_find_components = None
    _arc_get_bg = None

    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9


BG = Color.BLACK
SEP = Color.RED
FILL = Color.YELLOW
BAR_COLORS = [c for c in range(10) if c not in (BG, SEP, FILL)]
OBJECT_COLORS = BAR_COLORS[:]

GRID_HEIGHT_RANGE = (14, 21)
GRID_WIDTH_RANGE = (16, 27)
GEN_RETRIES = 400

OBSTACLE_SHAPES = (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
    min_area: int = 1,
    return_format: str = "dict",
):
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    allowed = None if colors is None else {int(c) for c in colors}
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if seen[r0, c0] or color == int(bg):
                continue
            if allowed is not None and color not in allowed:
                continue

            seen[r0, c0] = True
            stack = [(r0, c0)]
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    ncolor = int(grid[nr, nc])
                    if ncolor == int(bg):
                        continue
                    if allowed is not None and ncolor not in allowed:
                        continue
                    if monochromatic and ncolor != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            if len(coords) < int(min_area):
                continue
            coords_arr = np.array(coords, dtype=int)
            rmin, cmin = coords_arr.min(axis=0)
            rmax, cmax = coords_arr.max(axis=0)
            if return_format == "dict":
                out.append(
                    {
                        "color": color,
                        "coords": coords_arr,
                        "bbox": (int(rmin), int(cmin), int(rmax), int(cmax)),
                        "area": int(len(coords)),
                    }
                )
            else:
                obj = np.full_like(grid, int(bg))
                obj[coords_arr[:, 0], coords_arr[:, 1]] = grid[coords_arr[:, 0], coords_arr[:, 1]]
                out.append(obj)

    return out


def _find_components(grid: np.ndarray, **kwargs):
    if _arc_find_components is not None:
        return _arc_find_components(grid, **kwargs)
    return _fallback_find_components(grid, **kwargs)


def _row_segment(row: np.ndarray, bg: int) -> tuple[int, int, int]:
    cols = np.flatnonzero(row != int(bg))
    assert cols.size > 0, "expected a horizontal edge segment"
    color = int(row[cols[0]])
    assert np.all(row[cols] == color), "edge segment must be monochromatic"
    assert int(cols[-1] - cols[0] + 1) == int(cols.size), "edge segment must be contiguous"
    assert int(np.count_nonzero(row != int(bg))) == int(cols.size), "edge row must contain one segment"
    return color, int(cols[0]), int(cols[-1])


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == int(SEP), axis=1))
    assert rows.size == 1, "expected exactly one full red separator row"
    return int(rows[0])


def _component_area_map(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    area_map: dict[tuple[int, int], int] = {}
    for comp in _find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        area = int(comp["area"])
        for r, c in comp["coords"]:
            area_map[(int(r), int(c))] = area
    return area_map


def main(grid: np.ndarray) -> np.ndarray:
    """Fill overlapping bar columns from the longer side until multi-pixel obstacles block each ray.

    Input structure:
    - A black background contains one horizontal segment on the top row and one on the bottom row; both segments share one non-red color but differ in span.
    - Exactly one full red separator row crosses the grid between the two edge segments.
    - Only the region between the separator and the longer edge segment contains extra foreground objects: multi-pixel monochrome 4-connected obstacles and isolated single-pixel distractors.

    Rule:
    - Intersect the top and bottom segment column ranges, and process each overlapping column from the longer segment's side toward the separator.
    - Advance one ray cell at a time. Empty cells are traversed, single-pixel objects are pierced and effectively removed, and any cell belonging to a multi-pixel 4-connected object stops the ray before that row.
    - Color every traversed cell yellow, never overwrite the separator or any multi-pixel obstacle, and leave untouched foreground unchanged apart from pierced singletons being replaced by the yellow fill.

    Key insight:
    - The same visible kind of colored foreground can either be noise or a blocker; the hidden discriminator is connected-component size, so only singleton objects are transparent to the fill ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    top_color, top_c0, top_c1 = _row_segment(g[0], bg)
    bottom_color, bottom_c0, bottom_c1 = _row_segment(g[-1], bg)
    assert top_color == bottom_color, "top and bottom edge bars must share a color"
    assert top_color != int(SEP), "edge bars cannot use the separator color"

    sep_r = _separator_row(g)
    assert 0 < sep_r < g.shape[0] - 1

    top_len = top_c1 - top_c0 + 1
    bottom_len = bottom_c1 - bottom_c0 + 1
    assert top_len != bottom_len, "one side must be strictly longer"

    overlap_c0 = max(top_c0, bottom_c0)
    overlap_c1 = min(top_c1, bottom_c1)
    assert overlap_c0 <= overlap_c1, "edge bars must overlap in columns"

    fill_from_top = top_len > bottom_len
    ray_rows = range(1, sep_r) if fill_from_top else range(g.shape[0] - 2, sep_r, -1)
    area_map = _component_area_map(g, bg)

    out = g.copy()
    for c in range(overlap_c0, overlap_c1 + 1):
        traversed_rows: list[int] = []
        for r in ray_rows:
            value = int(g[r, c])
            if value == int(bg):
                traversed_rows.append(int(r))
                continue

            if int(area_map.get((int(r), int(c)), 1)) == 1:
                traversed_rows.append(int(r))
                continue
            break

        if traversed_rows:
            out[traversed_rows, c] = int(FILL)

    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _cells_by_col(cells: list[tuple[int, int]]) -> dict[int, list[int]]:
    by_col: dict[int, list[int]] = {}
    for r, c in cells:
        by_col.setdefault(int(c), []).append(int(r))
    return by_col


def _can_place_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    *,
    row_lo: int,
    row_hi: int,
) -> bool:
    h, w = grid.shape
    cell_set = {(int(r), int(c)) for r, c in cells}
    for r, c in cell_set:
        if not (row_lo <= r <= row_hi and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(BG):
            return False

    for r, c in cell_set:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cell_set:
                    continue
                if int(grid[nr, nc]) != int(BG):
                    return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _choose_segments(width: int, fill_from_top: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(300):
        long_len = random.randint(7, min(width - 2, 12))
        short_len = random.randint(4, long_len - 1)
        long_start = random.randint(0, width - long_len)
        short_start = random.randint(0, width - short_len)

        overlap = min(long_start + long_len - 1, short_start + short_len - 1) - max(long_start, short_start) + 1
        if overlap < 2:
            continue

        top = (long_start, long_len) if fill_from_top else (short_start, short_len)
        bottom = (short_start, short_len) if fill_from_top else (long_start, long_len)
        return top, bottom
    raise RuntimeError("failed to choose overlapping edge segments")


def _place_required_obstacle(
    grid: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    blocked_col: int,
    protected_col: int,
    used_cols: set[int],
    color: int,
) -> list[tuple[int, int]] | None:
    for _ in range(400):
        shape = random.choice(OBSTACLE_SHAPES)
        height, width = _shape_dims(shape)
        if row_lo + height - 1 > row_hi:
            continue

        local_cols = sorted({c for _, c in shape})
        c0 = blocked_col - random.choice(local_cols)
        cols = {c0 + c for _, c in shape}
        if min(cols) < 0 or max(cols) >= grid.shape[1]:
            continue
        if protected_col in cols or cols & used_cols:
            continue

        r0 = random.randint(row_lo, row_hi - height + 1)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if not _can_place_cells(grid, cells, row_lo=row_lo, row_hi=row_hi):
            continue

        _paint_cells(grid, cells, color)
        used_cols.update(cols)
        return cells
    return None


def _place_extra_obstacle(
    grid: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    protected_col: int,
    used_cols: set[int],
    color: int,
) -> list[tuple[int, int]] | None:
    for _ in range(300):
        shape = random.choice(OBSTACLE_SHAPES)
        height, width = _shape_dims(shape)
        if row_lo + height - 1 > row_hi:
            continue

        c0 = random.randint(0, grid.shape[1] - width)
        cols = {c0 + c for _, c in shape}
        if protected_col in cols or cols & used_cols:
            continue

        r0 = random.randint(row_lo, row_hi - height + 1)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if not _can_place_cells(grid, cells, row_lo=row_lo, row_hi=row_hi):
            continue

        _paint_cells(grid, cells, color)
        used_cols.update(cols)
        return cells
    return None


def _candidate_singleton_rows(
    *,
    fill_from_top: bool,
    grid_height: int,
    sep_r: int,
    obj_row_lo: int,
    obj_row_hi: int,
    obstacle_rows: list[int],
) -> list[int]:
    if fill_from_top:
        stop = min(obstacle_rows) if obstacle_rows else sep_r
        return list(range(obj_row_lo, min(obj_row_hi, stop - 1) + 1))

    stop = max(obstacle_rows) if obstacle_rows else sep_r
    return list(range(max(obj_row_lo, stop + 1), obj_row_hi + 1))


def _place_singleton_in_column(
    grid: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    col: int,
    rows: list[int],
    color: int,
) -> tuple[int, int] | None:
    rows = rows[:]
    random.shuffle(rows)
    for r in rows:
        cells = [(int(r), int(col))]
        if _can_place_cells(grid, cells, row_lo=row_lo, row_hi=row_hi):
            grid[int(r), int(col)] = int(color)
            return int(r), int(col)
    return None


def _place_random_singleton(
    grid: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    allowed_cols: list[int],
    color: int,
) -> tuple[int, int] | None:
    cols = allowed_cols[:]
    random.shuffle(cols)
    for col in cols:
        rows = list(range(row_lo, row_hi + 1))
        random.shuffle(rows)
        placed = _place_singleton_in_column(
            grid,
            row_lo=row_lo,
            row_hi=row_hi,
            col=int(col),
            rows=rows,
            color=color,
        )
        if placed is not None:
            return placed
    return None


def _nonstructural_components(grid: np.ndarray) -> tuple[list[dict], list[dict], int, int, int, int, bool]:
    bg = BG
    sep_r = _separator_row(grid)
    _bar_color, top_c0, top_c1 = _row_segment(grid[0], bg)
    _bar_color2, bottom_c0, bottom_c1 = _row_segment(grid[-1], bg)
    fill_from_top = (top_c1 - top_c0) > (bottom_c1 - bottom_c0)
    overlap_c0 = max(top_c0, bottom_c0)
    overlap_c1 = min(top_c1, bottom_c1)

    singles: list[dict] = []
    multis: list[dict] = []
    for comp in _find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        rows = comp["coords"][:, 0]
        if np.all(rows == 0) or np.all(rows == grid.shape[0] - 1) or np.all(rows == sep_r):
            continue
        if int(comp["area"]) == 1:
            singles.append(comp)
        else:
            multis.append(comp)
    return singles, multis, sep_r, overlap_c0, overlap_c1, bg, fill_from_top


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    singles, multis, sep_r, overlap_c0, overlap_c1, bg, fill_from_top = _nonstructural_components(grid)
    if not singles or not multis:
        return False

    top_color, top_c0, top_c1 = _row_segment(grid[0], bg)
    bottom_color, bottom_c0, bottom_c1 = _row_segment(grid[-1], bg)
    if top_color != bottom_color or top_color in (SEP, FILL):
        return False
    if top_c1 - top_c0 == bottom_c1 - bottom_c0:
        return False
    if overlap_c0 > overlap_c1:
        return False

    blocked_overlap = False
    per_col_multi: dict[int, int] = {}
    for comp in multis:
        area = int(comp["area"])
        if area < 2 or area > 5:
            return False
        rows = comp["coords"][:, 0]
        if fill_from_top:
            if int(rows.min()) <= 1 or int(rows.max()) >= sep_r - 1:
                return False
        else:
            if int(rows.min()) <= sep_r + 1 or int(rows.max()) >= grid.shape[0] - 2:
                return False
        for c in np.unique(comp["coords"][:, 1]):
            per_col_multi[int(c)] = per_col_multi.get(int(c), 0) + 1
            if overlap_c0 <= int(c) <= overlap_c1:
                blocked_overlap = True
    if not blocked_overlap or any(v > 1 for v in per_col_multi.values()):
        return False

    pierced = 0
    for comp in singles:
        r, c = map(int, comp["coords"][0])
        if int(out[r, c]) == int(FILL):
            pierced += 1
    if pierced == 0:
        return False

    sep_adj = sep_r - 1 if fill_from_top else sep_r + 1
    overlap_edge = out[sep_adj, overlap_c0 : overlap_c1 + 1] == int(FILL)
    if not overlap_edge.any() or overlap_edge.all():
        return False

    return bool(np.any(out != grid))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    fill_from_top = bool(random.getrandbits(1))
    sep_r = random.randint(5, height - 6)
    bar_color = random.choice(BAR_COLORS)

    top_seg, bottom_seg = _choose_segments(width, fill_from_top)
    top_start, top_len = top_seg
    bottom_start, bottom_len = bottom_seg
    top_end = top_start + top_len - 1
    bottom_end = bottom_start + bottom_len - 1
    overlap_cols = list(range(max(top_start, bottom_start), min(top_end, bottom_end) + 1))
    if len(overlap_cols) < 2:
        return None

    grid = np.full((height, width), int(BG), dtype=int)
    grid[0, top_start : top_end + 1] = int(bar_color)
    grid[-1, bottom_start : bottom_end + 1] = int(bar_color)
    grid[sep_r, :] = int(SEP)

    if fill_from_top:
        obj_row_lo, obj_row_hi = 2, sep_r - 2
    else:
        obj_row_lo, obj_row_hi = sep_r + 2, height - 3
    if obj_row_lo > obj_row_hi:
        return None

    color_pool = OBJECT_COLORS[:]
    random.shuffle(color_pool)

    protected_col = random.choice(overlap_cols)
    blocked_choices = [c for c in overlap_cols if c != protected_col]
    if not blocked_choices:
        return None
    blocked_col = random.choice(blocked_choices)

    used_obstacle_cols: set[int] = set()
    multi_cells: list[list[tuple[int, int]]] = []

    obstacle = _place_required_obstacle(
        grid,
        row_lo=obj_row_lo,
        row_hi=obj_row_hi,
        blocked_col=blocked_col,
        protected_col=protected_col,
        used_cols=used_obstacle_cols,
        color=color_pool.pop() if color_pool else random.choice(OBJECT_COLORS),
    )
    if obstacle is None:
        return None
    multi_cells.append(obstacle)

    extra_obstacles = random.randint(0, 2)
    for _ in range(extra_obstacles):
        obstacle = _place_extra_obstacle(
            grid,
            row_lo=obj_row_lo,
            row_hi=obj_row_hi,
            protected_col=protected_col,
            used_cols=used_obstacle_cols,
            color=color_pool.pop() if color_pool else random.choice(OBJECT_COLORS),
        )
        if obstacle is not None:
            multi_cells.append(obstacle)

    obstacle_rows_by_col: dict[int, list[int]] = {}
    for cells in multi_cells:
        for c, rows in _cells_by_col(cells).items():
            obstacle_rows_by_col.setdefault(int(c), []).extend(rows)

    singleton_positions: list[tuple[int, int]] = []
    required_cols = [protected_col]
    blocked_rows = _candidate_singleton_rows(
        fill_from_top=fill_from_top,
        grid_height=height,
        sep_r=sep_r,
        obj_row_lo=obj_row_lo,
        obj_row_hi=obj_row_hi,
        obstacle_rows=obstacle_rows_by_col.get(blocked_col, []),
    )
    if blocked_rows:
        required_cols.append(blocked_col)

    for col in required_cols:
        rows = _candidate_singleton_rows(
            fill_from_top=fill_from_top,
            grid_height=height,
            sep_r=sep_r,
            obj_row_lo=obj_row_lo,
            obj_row_hi=obj_row_hi,
            obstacle_rows=obstacle_rows_by_col.get(col, []),
        )
        placed = _place_singleton_in_column(
            grid,
            row_lo=obj_row_lo,
            row_hi=obj_row_hi,
            col=col,
            rows=rows,
            color=color_pool.pop() if color_pool else random.choice(OBJECT_COLORS),
        )
        if placed is None:
            return None
        singleton_positions.append(placed)

    extra_singletons = random.randint(0, 2)
    outer_cols = [c for c in range(width) if c not in overlap_cols]
    for _ in range(extra_singletons):
        placed = _place_random_singleton(
            grid,
            row_lo=obj_row_lo,
            row_hi=obj_row_hi,
            allowed_cols=outer_cols or overlap_cols,
            color=color_pool.pop() if color_pool else random.choice(OBJECT_COLORS),
        )
        if placed is not None:
            singleton_positions.append(placed)

    if len(singleton_positions) < len(required_cols):
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-overlap fill task")
