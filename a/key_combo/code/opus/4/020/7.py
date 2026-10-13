from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_MODES = ("a_wins", "b_wins", "tie_a_wins")
MAX_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


@dataclass(frozen=True)
class PathSpec:
    r_top: int
    c_start: int
    r_bottom: int
    c_end: int
    color: int = -1

    @property
    def corner(self) -> tuple[int, int]:
        return int(self.r_top), int(self.c_end)

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int(self.r_top), int(self.c_start)), (int(self.r_bottom), int(self.c_end))

    @property
    def horizontal_span(self) -> tuple[int, int]:
        return min(int(self.c_start), int(self.c_end)), max(int(self.c_start), int(self.c_end))

    def cells(self) -> set[tuple[int, int]]:
        left, right = self.horizontal_span
        cells = {(int(self.r_top), c) for c in range(left, right + 1)}
        cells.update((r, int(self.c_end)) for r in range(int(self.r_top), int(self.r_bottom) + 1))
        return cells

    def recolor(self, color: int) -> "PathSpec":
        return PathSpec(
            r_top=int(self.r_top),
            c_start=int(self.c_start),
            r_bottom=int(self.r_bottom),
            c_end=int(self.c_end),
            color=int(color),
        )


def _non_bg_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != BG)


def _contributors(paths: list[PathSpec]) -> dict[tuple[int, int], list[PathSpec]]:
    cover: dict[tuple[int, int], list[PathSpec]] = {}
    for path in paths:
        for cell in path.cells():
            cover.setdefault(cell, []).append(path)
    return cover


def _corner_distance(path: PathSpec, cell: tuple[int, int]) -> int:
    r, c = cell
    cr, cc = path.corner
    return abs(int(r) - int(cr)) + abs(int(c) - int(cc))


def _path_winner(paths: list[PathSpec], cell: tuple[int, int]) -> int:
    winner = min(paths, key=lambda path: (_corner_distance(path, cell), int(path.color)))
    return int(winner.color)


def _extract_paths(grid: np.ndarray) -> list[PathSpec]:
    g = np.asarray(grid, dtype=int)
    colors = _non_bg_colors(g)
    assert 2 <= len(colors) <= 4, "expected 2 to 4 foreground colors"

    paths: list[PathSpec] = []
    for color in colors:
        pts = np.argwhere(g == int(color))
        assert pts.shape == (2, 2), "each color must mark exactly two endpoints"

        a = tuple(map(int, pts[0]))
        b = tuple(map(int, pts[1]))
        assert a[0] != b[0] and a[1] != b[1], "paired endpoints must differ in both row and column"
        top, bottom = (a, b) if a[0] < b[0] else (b, a)
        paths.append(PathSpec(top[0], top[1], bottom[0], bottom[1], int(color)))

    return paths


def _render_distance_rule(paths: list[PathSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for cell, owners in _contributors(paths).items():
        out[cell] = owners[0].color if len(owners) == 1 else _path_winner(owners, cell)
    for path in paths:
        for endpoint in path.endpoints:
            out[endpoint] = int(path.color)
    return out


def _render_last_draw(paths: list[PathSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for path in sorted(paths, key=lambda item: int(item.color)):
        for cell in path.cells():
            out[cell] = int(path.color)
    for path in paths:
        for endpoint in path.endpoints:
            out[endpoint] = int(path.color)
    return out


def _render_smallest_color(paths: list[PathSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for cell, owners in _contributors(paths).items():
        out[cell] = min(int(path.color) for path in owners)
    for path in paths:
        for endpoint in path.endpoints:
            out[endpoint] = int(path.color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each color pair by a fixed right-angle path and recolor crossings by nearest corner.

    Input structure:
    - The grid is a black background with 2 to 4 foreground colors.
    - Each foreground color appears in exactly two cells, which are that color's endpoints.
    - Within each pair, the two endpoints are in different rows and different columns.

    Rule:
    - For each color pair, pick the endpoint with the smaller row as the start.
    - Draw that path horizontally along the start row until the other endpoint's column, then vertically down that column to the other endpoint; the corner is at `(r_top, c_bottom)`.
    - Cells covered by exactly one path keep that path's color.
    - If multiple paths cover the same cell, compare the Manhattan distance from that cell to each contributing path's corner and keep the color of the nearest corner; ties go to the smaller color id.
    - Restore the original endpoint pixels in their original colors.

    Key insight:
    - The hidden structure is a network of L-shaped paths whose overlap cells are not decided by draw order, but by which path's corner is geometrically closer.
    """
    g = np.asarray(grid, dtype=int)
    return _render_distance_rule(_extract_paths(g), g.shape)


def _sample_grid_shape(n_pairs: int) -> tuple[int, int]:
    base = max(11, 2 * int(n_pairs) + 3)
    upper = min(18, base + 4)
    return random.randint(base, upper), random.randint(base, upper)


def _build_forced_pair(
    height: int,
    width: int,
    mode: str,
    dir_a: int,
    prefer_b_left: bool,
) -> tuple[PathSpec, PathSpec, tuple[int, int]] | None:
    for _ in range(300):
        r_top_a = random.randint(2, height - 4)
        max_width = min(width - 3, 7)
        min_width = 4
        if max_width < min_width:
            return None
        span = random.randint(min_width, max_width)
        left = random.randint(1, width - span - 2)
        right = left + span
        c_start_a, c_end_a = (left, right) if dir_a > 0 else (right, left)

        max_dcol = span - 1
        if mode == "tie_a_wins":
            dcols = [d for d in range(1, max_dcol + 1) if d <= r_top_a - 1]
            if not dcols:
                continue
            dcol = random.choice(dcols)
            drow = dcol
        elif mode == "a_wins":
            candidates = []
            for dcol in range(1, max_dcol + 1):
                for drow in range(dcol + 1, min(r_top_a - 1, dcol + 3) + 1):
                    candidates.append((dcol, drow))
            if not candidates:
                continue
            dcol, drow = random.choice(candidates)
        else:
            candidates = []
            for drow in range(1, min(3, r_top_a - 1) + 1):
                for dcol in range(drow + 1, min(max_dcol, drow + 3) + 1):
                    candidates.append((dcol, drow))
            if not candidates:
                continue
            dcol, drow = random.choice(candidates)

        c_cross = c_end_a - dir_a * dcol
        r_top_b = r_top_a - drow
        lower_rows = [r for r in range(r_top_a + 1, height - 1)]
        if len(lower_rows) < 2:
            continue
        r_bottom_a, r_bottom_b = random.sample(lower_rows, 2)

        blocked_cols = {c_start_a, c_end_a, c_cross}
        left_choices = [c for c in range(1, c_cross) if c not in blocked_cols]
        right_choices = [c for c in range(c_cross + 1, width - 1) if c not in blocked_cols]
        if prefer_b_left and left_choices:
            c_start_b = random.choice(left_choices)
        elif not prefer_b_left and right_choices:
            c_start_b = random.choice(right_choices)
        else:
            other_choices = left_choices + right_choices
            if not other_choices:
                continue
            c_start_b = random.choice(other_choices)

        path_a = PathSpec(r_top_a, c_start_a, r_bottom_a, c_end_a)
        path_b = PathSpec(r_top_b, c_start_b, r_bottom_b, c_cross)
        cross = (int(r_top_a), int(c_cross))
        if cross in path_a.endpoints or cross in path_b.endpoints:
            continue
        return path_a, path_b, cross
    return None


def _add_extra_path(
    height: int,
    width: int,
    used_rows: set[int],
    used_cols: set[int],
    occupied: set[tuple[int, int]],
) -> PathSpec | None:
    available_rows = [r for r in range(1, height - 1) if r not in used_rows]
    available_cols = [c for c in range(1, width - 1) if c not in used_cols]
    if len(available_rows) < 2 or len(available_cols) < 2:
        return None

    for _ in range(300):
        r_top, r_bottom = sorted(random.sample(available_rows, 2))
        c_start, c_end = random.sample(available_cols, 2)
        candidate = PathSpec(r_top, c_start, r_bottom, c_end)
        if candidate.cells().isdisjoint(occupied):
            return candidate
    return None


def _render_input(paths: list[PathSpec], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in paths:
        for endpoint in path.endpoints:
            grid[endpoint] = int(path.color)
    return grid


def _validate_generated_example(
    grid: np.ndarray,
    paths: list[PathSpec],
    forced_pair: tuple[PathSpec, PathSpec],
    cross: tuple[int, int],
    mode: str,
) -> bool:
    colors = _non_bg_colors(grid)
    if len(colors) != len(paths):
        return False
    if any(np.count_nonzero(grid == color) != 2 for color in colors):
        return False

    endpoint_rows = []
    endpoint_cols = []
    for path in paths:
        for r, c in path.endpoints:
            endpoint_rows.append(int(r))
            endpoint_cols.append(int(c))
    if len(endpoint_rows) != len(set(endpoint_rows)) or len(endpoint_cols) != len(set(endpoint_cols)):
        return False

    cover = _contributors(paths)
    overlaps = [cell for cell, owners in cover.items() if len(owners) > 1]
    if cross not in overlaps or len(overlaps) < 1:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    expected = _render_distance_rule(paths, grid.shape)
    if not np.array_equal(actual, expected):
        return False

    path_a, path_b = forced_pair
    dist_a = _corner_distance(path_a, cross)
    dist_b = _corner_distance(path_b, cross)
    if mode == "a_wins":
        if not (dist_a < dist_b and int(actual[cross]) == int(path_a.color)):
            return False
        if int(_render_last_draw(paths, grid.shape)[cross]) == int(actual[cross]):
            return False
    elif mode == "b_wins":
        if not (dist_b < dist_a and int(actual[cross]) == int(path_b.color)):
            return False
        if int(_render_smallest_color(paths, grid.shape)[cross]) == int(actual[cross]):
            return False
    else:
        if not (dist_a == dist_b and int(path_a.color) < int(path_b.color) and int(actual[cross]) == int(path_a.color)):
            return False
        if int(_render_last_draw(paths, grid.shape)[cross]) == int(actual[cross]):
            return False

    return bool((actual != grid).any())


def _generate_candidate(call_index: int) -> np.ndarray | None:
    n_pairs = 2 + (call_index % 3)
    mode = GEN_MODES[call_index % len(GEN_MODES)]
    height, width = _sample_grid_shape(n_pairs)

    dir_a = 1 if call_index % 2 == 0 else -1
    prefer_b_left = bool((call_index // 2) % 2 == 0)
    forced = _build_forced_pair(height, width, mode, dir_a, prefer_b_left)
    if forced is None:
        return None
    geom_a, geom_b, cross = forced

    used_rows = {geom_a.r_top, geom_a.r_bottom, geom_b.r_top, geom_b.r_bottom}
    used_cols = {geom_a.c_start, geom_a.c_end, geom_b.c_start, geom_b.c_end}
    occupied = geom_a.cells() | geom_b.cells()
    geoms = [geom_a, geom_b]

    while len(geoms) < n_pairs:
        extra = _add_extra_path(height, width, used_rows, used_cols, occupied)
        if extra is None:
            return None
        geoms.append(extra)
        used_rows.update([extra.r_top, extra.r_bottom])
        used_cols.update([extra.c_start, extra.c_end])
        occupied |= extra.cells()

    colors = sorted(random.sample(PALETTE, n_pairs))
    extras = geoms[2:]
    random.shuffle(extras)
    colored_paths = [geoms[0].recolor(colors[0]), geoms[1].recolor(colors[1])]
    for geom, color in zip(extras, colors[2:]):
        colored_paths.append(geom.recolor(color))

    grid = _render_input(colored_paths, (height, width))
    forced_pair = (colored_paths[0], colored_paths[1])
    return grid if _validate_generated_example(grid, colored_paths, forced_pair, cross, mode) else None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_candidate(call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid right-angle crossing puzzle")
