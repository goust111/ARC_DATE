import random
from collections import deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_HEIGHT_RANGE = (11, 23)
GRID_WIDTH_CHOICES = tuple(range(13, 30, 2))
CORRIDOR_HALF_WIDTH = 2
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_AREA_RANGE = (2, 7)
OBJECT_HEIGHT_RANGE = (2, 5)
OBJECT_WIDTH_RANGE = (1, 4)
ANOMALY_AREA_RANGE = (2, 6)
ANOMALY_WIDTH_CHOICES = (1, 3, 5)
ANOMALY_HEIGHT_RANGE = (1, 5)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _component_support_size(grid: np.ndarray, points: list[tuple[int, int]]) -> int:
    seen = set()
    total = 0
    for point in points:
        if point in seen:
            continue
        comp = _same_color_component(grid, point)
        seen.update(comp)
        total += len(comp)
    return total


def _is_4_connected(points: list[tuple[int, int]]) -> bool:
    if not points:
        return False
    point_set = set(points)
    q = deque([points[0]])
    seen = {points[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in point_set and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(point_set)


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_connected_shape(
    max_h: int,
    max_w: int,
    min_area: int,
    max_area: int,
    require_odd_width: bool = False,
) -> list[tuple[int, int]] | None:
    max_area = min(max_area, max_h * max_w)
    if max_area < min_area:
        return None

    for _ in range(120):
        target_area = random.randint(min_area, max_area)
        start = (random.randrange(max_h), random.randrange(max_w))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        shape = _normalize_cells(cells)
        width = max(c for _, c in shape) + 1
        if require_odd_width and width % 2 == 0:
            continue
        return shape
    return None


def _translated_bbox(shape: list[tuple[int, int]], r0: int, c0: int) -> tuple[int, int, int, int]:
    rows = [r0 + r for r, _ in shape]
    cols = [c0 + c for _, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _region_is_clear(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    axis: int,
    axis_color: int,
) -> bool:
    h, w = grid.shape
    for r in range(max(0, r0), min(h - 1, r1) + 1):
        for c in range(max(0, c0), min(w - 1, c1) + 1):
            value = int(grid[r, c])
            if c == axis:
                if value != axis_color:
                    return False
            elif value != bg:
                return False
    return True


def _can_place_on_half(
    occupied: np.ndarray,
    shape: list[tuple[int, int]],
    r0: int,
    c0: int,
    pad: int = PLACEMENT_PADDING,
) -> bool:
    h, w = occupied.shape
    rows = [r0 + r for r, _ in shape]
    cols = [c0 + c for _, c in shape]
    if min(rows) < 0 or max(rows) >= h or min(cols) < 0 or max(cols) >= w:
        return False
    rmin, cmin, rmax, cmax = min(rows), min(cols), max(rows), max(cols)
    rr0 = max(0, rmin - pad)
    cc0 = max(0, cmin - pad)
    rr1 = min(h - 1, rmax + pad)
    cc1 = min(w - 1, cmax + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _paint_shape(grid: np.ndarray, shape: list[tuple[int, int]], r0: int, c0: int, color: int) -> None:
    for r, c in shape:
        grid[r0 + r, c0 + c] = color


def _mirror_half(half: np.ndarray, bg: int, axis_color: int) -> np.ndarray:
    h, half_w = half.shape
    w = half_w * 2 + 1
    axis = half_w
    out = np.full((h, w), bg, dtype=int)
    out[:, axis] = axis_color
    out[:, :axis] = half
    out[:, axis + 1 :] = np.fliplr(half)
    return out


def _defect_signature(grid: np.ndarray, points: list[tuple[int, int]], bg: int) -> tuple[int, int, int]:
    bg_count = sum(int(grid[r, c]) == bg for r, c in points)
    support = _component_support_size(grid, points)
    colors = len({int(grid[r, c]) for r, c in points})
    return bg_count, support, colors


def _stamp_on_axis(
    repaired: np.ndarray,
    defect_points: list[tuple[int, int]],
    defect_colors: dict[tuple[int, int], int],
    axis: int,
) -> np.ndarray:
    out = repaired.copy()
    cols = [c for _, c in defect_points]
    center = (min(cols) + max(cols)) // 2
    assert (max(cols) - min(cols)) % 2 == 0, "defect bbox width must be odd to center on the axis"
    for r, c in defect_points:
        sc = axis + (c - center)
        assert 0 <= sc < out.shape[1]
        out[r, sc] = defect_colors[(r, c)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the broken patch in a left-right mirror scene, repair the symmetry, and stamp that patch onto the axis.

    Input structure:
    - The grid has odd width and is almost mirror-symmetric across its vertical middle column.
    - The middle column is a solid special-color axis marker.
    - Away from the axis, mirrored foreground objects use a small palette on a dominant background.
    - Exactly one small 4-connected patch on either the left or right half disagrees with its mirrored counterpart.

    Rule:
    - Compare every left-half cell with its mirrored right-half cell to collect all mismatch pairs.
    - Identify which side is the true defect side: the broken patch is the more localized side among the mismatches.
    - Repair the scene by replacing each defect-side mismatch cell with its mirror counterpart's color.
    - Then translate the defect patch itself, keeping its original wrong colors and row positions, so that its bbox is centered on the axis column.

    Key insight:
    - The near-perfect bilateral symmetry is only a detector: it isolates the anomalous patch, whose shape is then reused as a centered stamp on the marked axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] % 2 == 1
    h, w = g.shape
    axis = w // 2

    axis_color = int(g[0, axis])
    assert np.all(g[:, axis] == axis_color), "expected a solid axis column"
    bg = _mode(np.concatenate([g[:, :axis].ravel(), g[:, axis + 1 :].ravel()]))

    pairs: list[tuple[int, int, int]] = []
    left_points: list[tuple[int, int]] = []
    right_points: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(axis):
            rc = w - 1 - c
            if int(g[r, c]) == int(g[r, rc]):
                continue
            pairs.append((r, c, rc))
            left_points.append((r, c))
            right_points.append((r, rc))

    assert pairs, "expected one asymmetric defect patch"

    left_sig = _defect_signature(g, left_points, bg)
    right_sig = _defect_signature(g, right_points, bg)
    assert left_sig != right_sig, "defect side must be distinguishable"
    defect_on_left = left_sig < right_sig
    defect_points = left_points if defect_on_left else right_points
    assert _is_4_connected(defect_points), "defect patch must be 4-connected"

    repaired = g.copy()
    for r, lc, rc in pairs:
        if defect_on_left:
            repaired[r, lc] = g[r, rc]
        else:
            repaired[r, rc] = g[r, lc]

    defect_colors = {point: int(g[point]) for point in defect_points}
    return _stamp_on_axis(repaired, defect_points, defect_colors, axis)


def _build_base_grid(h: int, w: int, bg: int, axis_color: int, fg_colors: list[int]) -> np.ndarray | None:
    axis = w // 2
    left = np.full((h, axis), bg, dtype=int)
    occupied = np.zeros((h, axis), dtype=bool)

    usable_max_col = axis - CORRIDOR_HALF_WIDTH - 1
    if usable_max_col <= 0:
        return None

    object_count = random.randint(max(OBJECT_COUNT_RANGE[0], len(fg_colors)), OBJECT_COUNT_RANGE[1])
    color_schedule = fg_colors.copy()
    while len(color_schedule) < object_count:
        color_schedule.append(random.choice(fg_colors))
    random.shuffle(color_schedule)

    for color in color_schedule:
        placed = False
        for _ in range(120):
            shape = _random_connected_shape(
                random.randint(*OBJECT_HEIGHT_RANGE),
                random.randint(*OBJECT_WIDTH_RANGE),
                OBJECT_AREA_RANGE[0],
                OBJECT_AREA_RANGE[1],
            )
            if shape is None:
                continue
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            if shape_w > usable_max_col + 1 or shape_h > h:
                continue
            r0 = random.randint(0, h - shape_h)
            c0 = random.randint(0, usable_max_col - shape_w + 1)
            if not _can_place_on_half(occupied, shape, r0, c0):
                continue
            _paint_shape(left, shape, r0, c0, color)
            for r, c in shape:
                occupied[r0 + r, c0 + c] = True
            placed = True
            break
        if not placed:
            return None

    return _mirror_half(left, bg, axis_color)


def _random_anomaly_shape(axis: int) -> list[tuple[int, int]] | None:
    width_choices = [width for width in ANOMALY_WIDTH_CHOICES if width <= axis]
    random.shuffle(width_choices)
    for width in width_choices:
        for _ in range(80):
            shape = _random_connected_shape(
                random.randint(*ANOMALY_HEIGHT_RANGE),
                width,
                ANOMALY_AREA_RANGE[0],
                ANOMALY_AREA_RANGE[1],
                require_odd_width=True,
            )
            if shape is None:
                continue
            actual_width = max(c for _, c in shape) + 1
            if actual_width % 2 == 1:
                return shape
    return None


def _anomaly_candidates(
    base: np.ndarray,
    shape: list[tuple[int, int]],
    side: str,
    bg: int,
    axis: int,
    axis_color: int,
) -> list[tuple[int, int]]:
    h, w = base.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    candidates = []

    if side == "left":
        col_range = range(0, axis - shape_w + 1)
    else:
        col_range = range(axis + 1, w - shape_w + 1)

    for r0 in range(0, h - shape_h + 1):
        for c0 in col_range:
            translated = [(r0 + r, c0 + c) for r, c in shape]
            mirrored = [(r, w - 1 - c) for r, c in translated]
            if any(int(base[r, c]) != bg for r, c in translated):
                continue
            if any(int(base[r, c]) != bg for r, c in mirrored):
                continue

            tr0, tc0, tr1, tc1 = _translated_bbox(shape, r0, c0)
            mr0, mc0, mr1, mc1 = min(r for r, _ in mirrored), min(c for _, c in mirrored), max(r for r, _ in mirrored), max(c for _, c in mirrored)
            if not _region_is_clear(base, tr0 - 1, tc0 - 1, tr1 + 1, tc1 + 1, bg, axis, axis_color):
                continue
            if not _region_is_clear(base, mr0 - 1, mc0 - 1, mr1 + 1, mc1 + 1, bg, axis, axis_color):
                continue
            candidates.append((r0, c0))
    return candidates


def _color_anomaly(
    fg_colors: list[int],
    shape: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    usable_colors = fg_colors[:]
    random.shuffle(usable_colors)
    color_count = random.randint(1, min(2, len(usable_colors)))
    picked = usable_colors[:color_count]
    coords = shape[:]
    random.shuffle(coords)

    out: dict[tuple[int, int], int] = {}
    for i, coord in enumerate(coords):
        if i < color_count:
            out[coord] = picked[i]
        else:
            out[coord] = random.choice(picked)
    return out


def _expected_output(
    base: np.ndarray,
    defect_points: list[tuple[int, int]],
    defect_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    axis = base.shape[1] // 2
    return _stamp_on_axis(base, defect_points, defect_colors, axis)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.choice(GRID_WIDTH_CHOICES)
    axis = w // 2

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    axis_color = random.choice(palette)
    palette.remove(axis_color)
    fg_count = random.randint(2, 3)
    fg_colors = random.sample(palette, fg_count)

    base = _build_base_grid(h, w, bg, axis_color, fg_colors)
    if base is None:
        return None

    shape = _random_anomaly_shape(axis)
    if shape is None:
        return None

    side = random.choice(["left", "right"])
    candidates = _anomaly_candidates(base, shape, side, bg, axis, axis_color)
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)

    local_colors = _color_anomaly(fg_colors, shape)
    defect_points = [(r0 + r, c0 + c) for r, c in shape]
    defect_colors = {(r0 + r, c0 + c): color for (r, c), color in local_colors.items()}
    assert _is_4_connected(defect_points)

    inp = base.copy()
    for point, color in defect_colors.items():
        inp[point] = color

    expected = _expected_output(base, defect_points, defect_colors)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # The opposite-side mismatches must be plain background so the defect side is unique.
    for r, c in defect_points:
        mr, mc = r, w - 1 - c
        if int(base[mr, mc]) != bg:
            return None
    if int(_mode(np.concatenate([inp[:, :axis].ravel(), inp[:, axis + 1 :].ravel()]))) != bg:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-defect ARC input")
