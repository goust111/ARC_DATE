import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
OBSTACLE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

ROW_SHAPES = {
    "h2": [(0, 0), (0, 1)],
    "h3": [(0, 0), (0, 1), (0, 2)],
}
COL_SHAPES = {
    "v2": [(0, 0), (1, 0)],
}

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _shape_size(shape: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_key(comp: list[tuple[int, int]]) -> tuple[int, int]:
    return min(comp)


def _boundary_cells(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(comp)
    out = []
    for r, c in comp:
        if any((r + dr, c + dc) not in cells for dr, dc in DIR4):
            out.append((r, c))
    return sorted(out)


def _settle_obstacles(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for c in range(w):
        rows = [r for r in range(h) if out[r, c] == OBSTACLE]
        if not rows:
            continue
        out[rows, c] = BACKGROUND
        for offset, r in enumerate(range(h - len(rows), h)):
            out[r, c] = OBSTACLE
    return out


def _scan_color_order(grid: np.ndarray) -> list[int]:
    order: list[int] = []
    seen = {BACKGROUND, OBSTACLE}
    for value in map(int, grid.ravel()):
        if value not in seen:
            seen.add(value)
            order.append(value)
    return order


def _path_hv(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    horiz = [(ar, c) for c in range(min(ac, bc), max(ac, bc) + 1)]
    vert = [(r, bc) for r in range(min(ar, br), max(ar, br) + 1)]
    return horiz + vert[1:]


def _path_vh(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    vert = [(r, ac) for r in range(min(ar, br), max(ar, br) + 1)]
    horiz = [(br, c) for c in range(min(ac, bc), max(ac, bc) + 1)]
    return vert + horiz[1:]


def _path_clear(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> bool:
    for r, c in path:
        if int(grid[r, c]) not in (BACKGROUND, color):
            return False
    return True


def _best_l_path(
    grid: np.ndarray,
    comp_a: list[tuple[int, int]],
    comp_b: list[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    candidates: list[tuple[tuple[int, ...], list[tuple[int, int]]]] = []
    for a in _boundary_cells(comp_a):
        for b in _boundary_cells(comp_b):
            for orient, builder in (("HV", _path_hv), ("VH", _path_vh)):
                path = builder(a, b)
                if not _path_clear(grid, path, color):
                    continue
                orient_rank = 0 if orient == "HV" else 1
                horiz_row = a[0] if orient == "HV" else grid.shape[0] + grid.shape[1]
                key = (
                    len(path),
                    orient_rank,
                    horiz_row,
                    a[0],
                    a[1],
                    b[0],
                    b[1],
                )
                candidates.append((key, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _solve(
    grid: np.ndarray,
    *,
    settle: bool,
    strict: bool = True,
    return_paths: bool = False,
):
    base = _settle_obstacles(grid) if settle else np.asarray(grid, dtype=int).copy()
    out = base.copy()
    paths: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for color_index, color in enumerate(_scan_color_order(grid)):
        comps = _components(base == color)
        comps.sort(key=_component_key)
        for pair_index, (comp_a, comp_b) in enumerate(zip(comps, comps[1:])):
            path = _best_l_path(out, comp_a, comp_b, color)
            if path is None:
                if strict:
                    raise AssertionError(f"no valid L path for color {color}")
                return None
            paths[(color_index, pair_index)] = tuple(path)
            for r, c in path:
                if out[r, c] == BACKGROUND:
                    out[r, c] = color
    if return_paths:
        return out, paths
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the grey obstacle cells to the bottom of their columns, then connect each color's components with shortest L paths.

    Input structure:
    - The grid is a black background with grey obstacle pixels and 2-3 non-grey foreground colors.
    - Each non-grey color appears as 2-3 separate 4-connected components made of small blobs.
    - Grey cells may float above empty space; they are the only cells that move.

    Rule:
    - First settle every grey cell straight downward within its own column, clearing its old position and packing all grey cells into the bottom of that column.
    - Then, for each non-grey color in the row-major order of its first appearance in the input, sort that color's components by the row-major position of their top-left pixel.
    - Connect each consecutive pair using the shortest clear orthogonal path with at most one right-angle turn; paths may pass only through black cells or already-present cells of the same color.
    - Among equally short L paths, prefer the horizontal-then-vertical route, and among those ties use the one whose horizontal segment is on the smaller row.
    - Paint only the black cells on the chosen path; previously drawn colors become blockers for later colors.

    Key insight:
    - The decisive geometry is the obstacle layout after the grey cells have settled, so suspended blockers can change which shortest L route becomes available.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g, settle=True, strict=True, return_paths=False)


def _can_place(grid: np.ndarray, shape: list[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    h, w = grid.shape
    cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != BACKGROUND:
            return False
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) not in cells and grid[nr, nc] != BACKGROUND:
                return False
    return True


def _paint_shape(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    anchor: tuple[int, int],
    color: int,
) -> None:
    for dr, dc in shape:
        grid[anchor[0] + dr, anchor[1] + dc] = color


def _place_shape(
    grid: np.ndarray,
    shape_names: list[str],
    shape_bank: dict[str, list[tuple[int, int]]],
    row_choices: list[int],
    col_min: int,
    col_max: int,
    color: int,
    avoid_cols: set[int] | None = None,
) -> tuple[str, tuple[int, int]] | None:
    choices = shape_names[:]
    random.shuffle(choices)
    for name in choices:
        shape = shape_bank[name]
        sh, sw = _shape_size(shape)
        valid_rows = [r for r in row_choices if 0 <= r and r + sh <= grid.shape[0]]
        valid_cols = [c for c in range(col_min, col_max + 1) if 0 <= c and c + sw <= grid.shape[1]]
        random.shuffle(valid_rows)
        random.shuffle(valid_cols)
        for r in valid_rows:
            for c in valid_cols:
                if avoid_cols and any(c + dc in avoid_cols for _, dc in shape):
                    continue
                if _can_place(grid, shape, (r, c)):
                    _paint_shape(grid, shape, (r, c), color)
                    return name, (r, c)
    return None


def _zone_layout(num_colors: int, comp_counts: list[int]) -> tuple[int, list[tuple[int, int]]]:
    bottom_clear = 2
    heights = [5 if count == 3 else 4 for count in comp_counts]
    total = sum(heights) + (num_colors - 1) + bottom_clear
    max_h = 16
    if total > max_h:
        raise AssertionError("zone layout exceeds height budget")
    h = total if num_colors == 3 else random.randint(total, min(max_h, total + 2))
    starts = [0]
    for prev_h in heights[:-1]:
        starts.append(starts[-1] + prev_h + 1)
    zones = [(r0, r0 + height - 1) for r0, height in zip(starts, heights)]
    return h, zones


def _add_grey_block(
    grid: np.ndarray,
    row: int,
    band_cols: list[int],
    grey_counts: dict[int, int],
    *,
    allow_pair: bool,
) -> bool:
    random_cols = band_cols[:]
    random.shuffle(random_cols)
    if allow_pair and len(band_cols) >= 2 and random.random() < 0.4:
        for c0 in range(min(band_cols), max(band_cols)):
            cells = [(row, c0), (row, c0 + 1)]
            if c0 not in band_cols or c0 + 1 not in band_cols:
                continue
            if any(grey_counts[c] >= 2 for _, c in cells):
                continue
            if all(grid[r, c] == BACKGROUND for r, c in cells):
                for r, c in cells:
                    grid[r, c] = OBSTACLE
                    grey_counts[c] += 1
                return True
    for c in random_cols:
        if grey_counts[c] >= 2 or grid[row, c] != BACKGROUND:
            continue
        grid[row, c] = OBSTACLE
        grey_counts[c] += 1
        return True
    return False


def _color_component_counts(grid: np.ndarray) -> list[int]:
    counts = []
    for color in _scan_color_order(grid):
        counts.append(len(_components(grid == color)))
    return counts


def generate_input() -> np.ndarray:
    for _ in range(500):
        num_colors = random.choice([2, 2, 3])
        comp_counts = [2] * num_colors
        if num_colors == 2 and random.random() < 0.6:
            comp_counts[random.randrange(num_colors)] = 3

        h, zones = _zone_layout(num_colors, comp_counts)
        w = random.randint(13, 16)
        band_w = 3 if w >= 15 and random.random() < 0.7 else 2
        band_c0 = w // 2 - band_w // 2
        band_cols = list(range(band_c0, band_c0 + band_w))
        left_col_max = band_c0 - 3
        right_col_min = band_cols[-1] + 3
        if left_col_max < 2 or right_col_min > w - 3:
            continue

        grid = np.zeros((h, w), dtype=int)
        colors = random.sample(PALETTE, num_colors)
        critical_index = random.randrange(num_colors)
        grey_counts: dict[int, int] = defaultdict(int)

        ok = True
        for idx, ((r0, r1), color, comp_count) in enumerate(zip(zones, colors, comp_counts)):
            zone_h = r1 - r0 + 1
            a_rows = [r0] if zone_h == 4 else [r0, r0 + 1]
            placed_a = _place_shape(
                grid,
                ["h2", "h3"],
                ROW_SHAPES,
                a_rows,
                1,
                left_col_max - 1,
                color,
            )
            if placed_a is None:
                ok = False
                break
            a_name, a_anchor = placed_a
            a_row = a_anchor[0]

            b_row = r0 + 2
            if b_row + 2 > r1 + 1:
                ok = False
                break
            placed_b = _place_shape(
                grid,
                ["v2"],
                COL_SHAPES,
                [b_row],
                right_col_min,
                w - 2,
                color,
            )
            if placed_b is None:
                ok = False
                break
            _, b_anchor = placed_b

            if comp_count == 3:
                c_row = r1
                c_col_min = right_col_min
                c_col_max = w - 4
                placed_c = _place_shape(
                    grid,
                    ["h2", "h3"],
                    ROW_SHAPES,
                    [c_row],
                    c_col_min,
                    c_col_max,
                    color,
                )
                if placed_c is None:
                    ok = False
                    break

            must_shift = idx == critical_index
            if must_shift or random.random() < 0.35:
                if not _add_grey_block(
                    grid,
                    a_row,
                    band_cols,
                    grey_counts,
                    allow_pair=must_shift and band_w >= 2,
                ):
                    ok = False
                    break

        if not ok:
            continue

        extra_rows = [r for r in range(1, h - 2) if all(grid[r, c] == BACKGROUND for c in band_cols)]
        random.shuffle(extra_rows)
        for row in extra_rows[: random.randint(0, 2)]:
            _add_grey_block(grid, row, band_cols, grey_counts, allow_pair=False)

        if any(np.any(grid[h - 2 :, c] == OBSTACLE) for c in range(w)):
            continue
        if not np.any(grid == OBSTACLE):
            continue

        counts = _color_component_counts(grid)
        if counts != comp_counts:
            continue

        settled = _settle_obstacles(grid)
        if any(len(_components(settled == color)) != count for color, count in zip(colors, comp_counts)):
            continue

        solved = _solve(grid, settle=True, strict=False, return_paths=True)
        presolved = _solve(grid, settle=False, strict=False, return_paths=True)
        if solved is None or presolved is None:
            continue
        out, settled_paths = solved
        _, raw_paths = presolved

        if not any(settled_paths.get(key) != raw_paths.get(key) for key in settled_paths):
            continue
        if np.array_equal(grid, out):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for task 9")
