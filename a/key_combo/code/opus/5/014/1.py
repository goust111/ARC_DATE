from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
CONTROL = 1
RAY = 5
POINT_COLORS = (2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 13
GRID_MAX = 18
GEN_RETRIES = 500

# A fifth point would force two fills to share a corner, which contradicts the
# required "no overlap before truncation" condition because every full fill
# rectangle includes its corner cell.
POINT_COUNT_CHOICES = (3, 4)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            pr, pc = stack.pop()
            pts.append((pr, pc))
            for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _corners(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _nearest_corner(shape: tuple[int, int], point: tuple[int, int]) -> tuple[int, int]:
    corners = _corners(shape)
    dists = [abs(point[0] - r) + abs(point[1] - c) for r, c in corners]
    best = min(dists)
    assert dists.count(best) == 1, "foreground point must have a unique nearest corner"
    return corners[dists.index(best)]


def _rect_from_diagonal(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _marker_cells(center: tuple[int, int], step: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r, c = center
    if step[0] != 0:
        bar = [(r, c - 1), (r, c), (r, c + 1)]
    else:
        bar = [(r - 1, c), (r, c), (r + 1, c)]
    control = (r - step[0], c - step[1])
    return bar, control


def _ray_cells(
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
    *,
    include_center: bool,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = center
    if not include_center:
        r += step[0]
        c += step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += step[0]
        c += step[1]
    return cells


def _parse_marker(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[int], set[int]]:
    grey = np.argwhere(grid == RAY)
    assert grey.shape[0] == 3, "expected one three-pixel grey marker bar"
    rows = sorted(set(int(v) for v in grey[:, 0]))
    cols = sorted(set(int(v) for v in grey[:, 1]))

    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        center = (rows[0], cols[1])
        axis = "horizontal"
    elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        center = (rows[1], cols[0])
        axis = "vertical"
    else:
        raise AssertionError("grey pixels must form a straight contiguous three-pixel bar")

    blue = np.argwhere(grid == CONTROL)
    assert blue.shape[0] == 1, "expected one blue control pixel"
    control = tuple(int(v) for v in blue[0])
    assert abs(control[0] - center[0]) + abs(control[1] - center[1]) == 1

    if axis == "horizontal":
        assert control[1] == center[1], "horizontal bar must have a vertical control neighbor"
        step = (center[0] - control[0], 0)
    else:
        assert control[0] == center[0], "vertical bar must have a horizontal control neighbor"
        step = (0, center[1] - control[1])

    assert abs(step[0]) + abs(step[1]) == 1
    bar, expected_control = _marker_cells(center, step)
    assert control == expected_control, "ray direction must point away from the control pixel"

    marker_pixels = set(bar)
    marker_pixels.add(control)
    marker_rows = {r for r, _ in marker_pixels}
    marker_cols = {c for _, c in marker_pixels}
    return center, step, marker_pixels, marker_rows, marker_cols


def _rect_hit_by_ray(
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
) -> bool:
    h, w = map(int, shape)
    r0, c0, r1, c1 = rect

    if step[0] != 0:
        c_line = center[1]
        ray_r0 = center[0] if step[0] > 0 else 0
        ray_r1 = h - 1 if step[0] > 0 else center[0]
        return c0 <= c_line <= c1 and not (r1 < ray_r0 or r0 > ray_r1)

    r_line = center[0]
    ray_c0 = center[1] if step[1] > 0 else 0
    ray_c1 = w - 1 if step[1] > 0 else center[1]
    return r0 <= r_line <= r1 and not (c1 < ray_c0 or c0 > ray_c1)


def _truncate_rect(
    rect: tuple[int, int, int, int],
    point: tuple[int, int],
    shape: tuple[int, int],
    center: tuple[int, int],
    step: tuple[int, int],
) -> tuple[int, int, int, int]:
    if not _rect_hit_by_ray(rect, shape, center, step):
        return rect

    r0, c0, r1, c1 = rect
    pr, pc = point
    if step[0] != 0:
        line = center[1]
        assert pc != line, "foreground point cannot lie on the ray column"
        if pc < line:
            c1 = min(c1, line - 1)
        else:
            c0 = max(c0, line + 1)
    else:
        line = center[0]
        assert pr != line, "foreground point cannot lie on the ray row"
        if pr < line:
            r1 = min(r1, line - 1)
        else:
            r0 = max(r0, line + 1)

    assert r0 <= pr <= r1 and c0 <= pc <= c1
    return r0, c0, r1, c1


def _analyze_input(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    center, step, _, marker_rows, marker_cols = _parse_marker(g)

    fg_mask = (g != bg) & (g != CONTROL) & (g != RAY)
    comps = _components4(fg_mask)
    assert 3 <= len(comps) <= 4, "expected three or four isolated foreground pixels"

    infos: list[dict] = []
    for comp in comps:
        assert comp.shape[0] == 1, "every foreground object must be a singleton pixel"
        pr, pc = map(int, comp[0])
        assert pr not in marker_rows and pc not in marker_cols, "foreground pixels cannot share marker rows or columns"
        corner = _nearest_corner(g.shape, (pr, pc))
        full_rect = _rect_from_diagonal((pr, pc), corner)
        kept_rect = _truncate_rect(full_rect, (pr, pc), g.shape, center, step)
        infos.append(
            {
                "point": (pr, pc),
                "color": int(g[pr, pc]),
                "corner": corner,
                "full_rect": full_rect,
                "kept_rect": kept_rect,
                "hit": full_rect != kept_rect,
            }
        )

    infos.sort(key=lambda item: (item["point"][0], item["point"][1], item["color"]))
    return bg, center, step, infos


def _render(
    shape: tuple[int, int],
    bg: int,
    center: tuple[int, int],
    step: tuple[int, int],
    infos: list[dict],
    *,
    truncate: bool,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    key = "kept_rect" if truncate else "full_rect"
    for info in infos:
        r0, c0, r1, c1 = info[key]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(info["color"])
    for r, c in _ray_cells(shape, center, step, include_center=True):
        out[r, c] = RAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each singleton toward its nearest corner, but clip the fill at the marker-directed ray barrier.

    Input structure:
    - A black background grid contains one grey three-pixel bar, one blue control pixel adjacent to the bar center, and three or four isolated colored singleton pixels.
    - The grey bar is horizontal or vertical and lies strictly inside the grid.
    - Foreground singleton pixels avoid every row and column used by the bar or the control pixel.

    Rule:
    - Read the ray direction from the bar center away from the adjacent blue control pixel, then extend a grey ray from that center cell to the grid boundary.
    - Remove the original three-pixel bar and the blue control pixel; only the grey ray remains in the output.
    - For each colored singleton, find its unique nearest grid corner by Manhattan distance and form the axis-aligned rectangle having the singleton and that corner as opposite corners.
    - If the grey ray crosses that rectangle, keep only the side containing the singleton and exclude the ray cells themselves; otherwise keep the full rectangle.
    - Fill every kept rectangle with the singleton's color. The grey ray stays visible on top.

    Key insight:
    - The bar-and-control marker does not erase objects here; it creates a one-sided geometric barrier that truncates some nearest-corner fills while leaving others untouched.
    """
    g = np.asarray(grid, dtype=int)
    bg, center, step, infos = _analyze_input(g)
    return _render(g.shape, bg, center, step, infos, truncate=True)


def _marker_options(h: int, w: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                center = (r, c)
                bar, control = _marker_cells(center, step)
                marker = bar + [control]
                if not all(1 <= rr < h - 1 and 1 <= cc < w - 1 for rr, cc in marker):
                    continue
                if len(_ray_cells((h, w), center, step, include_center=True)) < 5:
                    continue
                options.append((center, step))
    return options


def _candidate_points(
    shape: tuple[int, int],
    marker_rows: set[int],
    marker_cols: set[int],
    center: tuple[int, int],
    step: tuple[int, int],
) -> dict[tuple[int, int], list[dict]]:
    h, w = map(int, shape)
    out: dict[tuple[int, int], list[dict]] = {corner: [] for corner in _corners(shape)}

    for r in range(h):
        if r in marker_rows:
            continue
        for c in range(w):
            if c in marker_cols:
                continue
            try:
                corner = _nearest_corner(shape, (r, c))
            except AssertionError:
                continue
            full_rect = _rect_from_diagonal((r, c), corner)
            kept_rect = _truncate_rect(full_rect, (r, c), shape, center, step)
            full_area = _rect_area(full_rect)
            kept_area = _rect_area(kept_rect)
            removed = full_area - kept_area
            if full_area < 4 or full_area > shape[0] * shape[1] // 3:
                continue
            if kept_area < 2:
                continue
            if removed and removed < 3:
                continue
            out[corner].append(
                {
                    "point": (r, c),
                    "corner": corner,
                    "full_rect": full_rect,
                    "kept_rect": kept_rect,
                    "hit": full_rect != kept_rect,
                }
            )

    return out


def _points_well_spaced(points: list[tuple[int, int]]) -> bool:
    for (r1, c1), (r2, c2) in itertools.combinations(points, 2):
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False
    return True


def _rects_disjoint(rects: list[tuple[int, int, int, int]]) -> bool:
    for a, b in itertools.combinations(rects, 2):
        if _rects_overlap(a, b):
            return False
    return True


def _choose_point_layout(
    shape: tuple[int, int],
    marker_rows: set[int],
    marker_cols: set[int],
    center: tuple[int, int],
    step: tuple[int, int],
) -> list[dict] | None:
    by_corner = _candidate_points(shape, marker_rows, marker_cols, center, step)
    usable_corners = [corner for corner, items in by_corner.items() if items]
    if len(usable_corners) < 3:
        return None

    counts = list(POINT_COUNT_CHOICES)
    random.shuffle(counts)
    for count in counts:
        if count > len(usable_corners):
            continue
        corner_sets = list(itertools.combinations(usable_corners, count))
        random.shuffle(corner_sets)
        for chosen_corners in corner_sets:
            for _ in range(250):
                picked = [dict(random.choice(by_corner[corner])) for corner in chosen_corners]
                points = [item["point"] for item in picked]
                full_rects = [item["full_rect"] for item in picked]
                kept_rects = [item["kept_rect"] for item in picked]
                hit_count = sum(int(item["hit"]) for item in picked)
                if hit_count == 0 or hit_count == len(picked):
                    continue
                if not _points_well_spaced(points):
                    continue
                if not _rects_disjoint(full_rects):
                    continue
                if not _rects_disjoint(kept_rects):
                    continue
                out_area = sum(_rect_area(rect) for rect in kept_rects)
                if not (10 <= out_area <= shape[0] * shape[1] // 2):
                    continue
                if len({item["corner"] for item in picked}) < 2:
                    continue
                return picked
    return None


def _valid_instance(inp: np.ndarray) -> bool:
    try:
        bg, center, step, infos = _analyze_input(inp)
    except AssertionError:
        return False

    if len(infos) not in POINT_COUNT_CHOICES:
        return False
    if len({info["color"] for info in infos}) != len(infos):
        return False
    if len({info["corner"] for info in infos}) < 2:
        return False
    if not _points_well_spaced([info["point"] for info in infos]):
        return False
    if not _rects_disjoint([info["full_rect"] for info in infos]):
        return False
    if not _rects_disjoint([info["kept_rect"] for info in infos]):
        return False

    hit_count = sum(int(info["hit"]) for info in infos)
    if hit_count == 0 or hit_count == len(infos):
        return False

    actual = main(inp)
    expected = _render(inp.shape, bg, center, step, infos, truncate=True)
    baseline = _render(inp.shape, bg, center, step, infos, truncate=False)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, baseline):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    marker_options = _marker_options(h, w)
    if not marker_options:
        return None
    center, step = random.choice(marker_options)
    bar, control = _marker_cells(center, step)
    marker_pixels = set(bar)
    marker_pixels.add(control)
    marker_rows = {r for r, _ in marker_pixels}
    marker_cols = {c for _, c in marker_pixels}

    layout = _choose_point_layout((h, w), marker_rows, marker_cols, center, step)
    if layout is None:
        return None

    for r, c in bar:
        grid[r, c] = RAY
    grid[control] = CONTROL

    colors = random.sample(POINT_COLORS, len(layout))
    random.shuffle(layout)
    for color, item in zip(colors, layout):
        r, c = item["point"]
        grid[r, c] = color

    if not _valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
