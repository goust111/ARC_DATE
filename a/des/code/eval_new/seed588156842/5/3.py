import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

TOP_H_MIN = 14
TOP_H_MAX = 20
TOP_W_MIN = 15
TOP_W_MAX = 24
BASE_MIN = 2
BASE_MAX = 3
GEN_ATTEMPTS = 400
DISTRACTOR_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    uniq, counts = np.unique(flat, return_counts=True)
    return int(uniq[counts.argmax()])


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _trim_rectangle_corners(
    cells: set[tuple[int, int]],
    corners: list[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    order = corners[:]
    random.shuffle(order)
    out = set(cells)
    for corner in order:
        if corner in protected or random.random() < 0.45:
            continue
        trial = out - {corner}
        if _connected(trial):
            out = trial
            break
    return out


def _horizontal_island(
    top: int,
    left: int,
    height: int,
    width: int,
    bridge_row: int,
    arm_side: str,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    cells = {(r, c) for r in range(top, top + height) for c in range(left, left + width)}
    if arm_side == "right":
        arm = (bridge_row, left + width)
        attach = (bridge_row, left + width - 1)
    else:
        arm = (bridge_row, left - 1)
        attach = (bridge_row, left)
    cells.add(arm)
    corners = [
        (top, left),
        (top, left + width - 1),
        (top + height - 1, left),
        (top + height - 1, left + width - 1),
    ]
    cells = _trim_rectangle_corners(cells, corners, {attach})
    return cells, arm


def _vertical_island(
    top: int,
    left: int,
    height: int,
    width: int,
    bridge_col: int,
    arm_side: str,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    cells = {(r, c) for r in range(top, top + height) for c in range(left, left + width)}
    if arm_side == "down":
        arm = (top + height, bridge_col)
        attach = (top + height - 1, bridge_col)
    else:
        arm = (top - 1, bridge_col)
        attach = (top, bridge_col)
    cells.add(arm)
    corners = [
        (top, left),
        (top, left + width - 1),
        (top + height - 1, left),
        (top + height - 1, left + width - 1),
    ]
    cells = _trim_rectangle_corners(cells, corners, {attach})
    return cells, arm


def _random_blob_local() -> set[tuple[int, int]]:
    height = random.randint(BASE_MIN, BASE_MAX)
    width = random.randint(BASE_MIN, BASE_MAX)
    cells = {(r, c) for r in range(height) for c in range(width)}
    side = random.choice(["up", "down", "left", "right"])
    if side in ("up", "down"):
        arm_col = random.randint(0, width - 1)
        arm = (-1, arm_col) if side == "up" else (height, arm_col)
        attach = (0, arm_col) if side == "up" else (height - 1, arm_col)
    else:
        arm_row = random.randint(0, height - 1)
        arm = (arm_row, -1) if side == "left" else (arm_row, width)
        attach = (arm_row, 0) if side == "left" else (arm_row, width - 1)
    cells.add(arm)
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    cells = _trim_rectangle_corners(cells, corners, {attach})
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _fits_with_padding(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            candidates.append(r)
    assert len(candidates) == 1, f"expected one separator row, got {candidates}"
    return int(candidates[0])


def _parse_routes(lower: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    routes: list[tuple[int, int, int]] = []
    for r in range(lower.shape[0]):
        cols = [int(c) for c in np.flatnonzero(lower[r] != bg)]
        if not cols:
            continue
        assert len(cols) == 3, f"route row {r} must have three colored cells"
        assert cols == list(range(cols[0], cols[0] + 3)), f"route row {r} must be contiguous"
        a, bridge, b = [int(v) for v in lower[r, cols[0] : cols[0] + 3]]
        routes.append((a, bridge, b))
    assert routes, "expected at least one route row"
    return routes


def _bridge_candidates(
    region: np.ndarray,
    color_a: int,
    color_b: int,
    bg: int,
) -> list[tuple[int, str, tuple[int, int], tuple[int, int]]]:
    pts_a = [tuple(map(int, p)) for p in np.argwhere(region == color_a)]
    pts_b = [tuple(map(int, p)) for p in np.argwhere(region == color_b)]
    assert pts_a and pts_b, f"missing island color(s) {color_a}, {color_b}"

    seen: dict[tuple[str, tuple[int, int], tuple[int, int]], int] = {}
    for ar, ac in pts_a:
        for br, bc in pts_b:
            if ar == br:
                left_c, right_c = sorted((ac, bc))
                if right_c - left_c <= 1:
                    continue
                if np.all(region[ar, left_c + 1 : right_c] == bg):
                    start = (ar, left_c)
                    end = (ar, right_c)
                    seen[("h", start, end)] = right_c - left_c - 1
            if ac == bc:
                top_r, bottom_r = sorted((ar, br))
                if bottom_r - top_r <= 1:
                    continue
                if np.all(region[top_r + 1 : bottom_r, ac] == bg):
                    start = (top_r, ac)
                    end = (bottom_r, ac)
                    seen[("v", start, end)] = bottom_r - top_r - 1

    candidates = [(gap, orient, start, end) for (orient, start, end), gap in seen.items()]
    candidates.sort(key=lambda item: (item[0], 0 if item[1] == "h" else 1, item[2], item[3]))
    return candidates


def _best_bridge(
    region: np.ndarray,
    color_a: int,
    color_b: int,
    bg: int,
) -> tuple[int, str, tuple[int, int], tuple[int, int]]:
    candidates = _bridge_candidates(region, color_a, color_b, bg)
    assert candidates, f"no clear bridge path for colors {color_a} and {color_b}"
    return candidates[0]


def _draw_bridge(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    if start[0] == end[0]:
        r = start[0]
        for c in range(start[1] + 1, end[1]):
            out[r, c] = color
        return
    c = start[1]
    for r in range(start[0] + 1, end[0]):
        out[r, c] = color


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep = _find_separator_row(g, bg)
    upper = g[:sep, :].copy()
    routes = _parse_routes(g[sep + 1 :, :], bg)

    out = upper.copy()
    for color_a, bridge_color, color_b in routes:
        _gap, _orient, start, end = _best_bridge(upper, color_a, color_b, bg)
        _draw_bridge(out, start, end, bridge_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw bridge-colored straight connectors between routed color islands, then discard the routing legend.

    Input structure:
    - The background is black.
    - A full grey horizontal separator row splits the grid into an upper island area and a lower routing-table area.
    - Above the separator, several differently colored connected islands appear as solid irregular blobs separated by black cells.
    - Below the separator, each non-empty row contains exactly one contiguous 3-cell color triplet of the form [A, C, B].

    Rule:
    - Read every lower triplet as a routing instruction: connect the island of color A to the island of color B using bridge color C.
    - For each routed pair, search all same-row and same-column pixel pairs between the two islands and keep the clear orthogonal pair with the smallest black gap between endpoints.
    - Paint only the black cells strictly between that endpoint pair with color C, leaving the islands themselves unchanged.
    - Return only the upper area: remove the grey separator and the routing table.

    Key insight:
    - The bottom legend does not name bridge locations directly; it names two endpoint-island colors and places the bridge color in the middle, so the solver must decode a color mapping first and then find the shortest straight black corridor between the referenced islands.
    """
    return _transform(grid)


def _generate_attempt() -> np.ndarray | None:
    top_h = random.randint(TOP_H_MIN, TOP_H_MAX)
    top_w = random.randint(TOP_W_MIN, TOP_W_MAX)

    colors = random.sample(PALETTE, 6)
    island_colors = colors[:4]
    bridge_colors = colors[4:]

    top = np.full((top_h, top_w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    # Horizontal bridge pair.
    h1 = random.randint(BASE_MIN, BASE_MAX)
    w1 = random.randint(BASE_MIN, BASE_MAX)
    h2 = random.randint(BASE_MIN, BASE_MAX)
    w2 = random.randint(BASE_MIN, BASE_MAX)
    bridge_row_h = random.randint(1, min(3, top_h - 1))
    left_tops = [t for t in range(0, 5) if t <= bridge_row_h <= t + h1 - 1]
    right_tops = [t for t in range(0, 5) if t <= bridge_row_h <= t + h2 - 1]
    if not left_tops or not right_tops:
        return None
    top1 = random.choice(left_tops)
    top2 = random.choice(right_tops)
    max_gap_h = top_w - w1 - w2 - 2
    if max_gap_h < 1:
        return None
    gap_h = random.randint(1, min(7, max_gap_h))
    max_left = top_w - w1 - gap_h - 2 - w2
    if max_left < 0:
        return None
    left1 = random.randint(0, max_left)
    left2 = left1 + w1 + gap_h + 2
    cells1, arm1 = _horizontal_island(top1, left1, h1, w1, bridge_row_h, "right")
    cells2, arm2 = _horizontal_island(top2, left2, h2, w2, bridge_row_h, "left")

    # Vertical bridge pair.
    h3 = random.randint(BASE_MIN, BASE_MAX)
    w3 = random.randint(BASE_MIN, BASE_MAX)
    h4 = random.randint(BASE_MIN, BASE_MAX)
    w4 = random.randint(BASE_MIN, BASE_MAX)
    bridge_col_v = random.randint(1, top_w - 2)
    left3_options = [c for c in range(0, top_w - w3 + 1) if c <= bridge_col_v <= c + w3 - 1]
    left4_options = [c for c in range(0, top_w - w4 + 1) if c <= bridge_col_v <= c + w4 - 1]
    if not left3_options or not left4_options:
        return None
    left3 = random.choice(left3_options)
    left4 = random.choice(left4_options)
    gap_v = random.randint(1, 4)
    max_top3 = top_h - (h3 + gap_v + 2 + h4)
    if max_top3 < 5:
        return None
    top3 = random.randint(5, max_top3)
    top4 = top3 + h3 + gap_v + 2
    cells3, arm3 = _vertical_island(top3, left3, h3, w3, bridge_col_v, "down")
    cells4, arm4 = _vertical_island(top4, left4, h4, w4, bridge_col_v, "up")

    core_islands = [cells1, cells2, cells3, cells4]
    for island in core_islands:
        if not _fits_with_padding(island, occupied, top_h, top_w):
            return None
        occupied |= island

    _paint_cells(top, cells1, island_colors[0])
    _paint_cells(top, cells2, island_colors[1])
    _paint_cells(top, cells3, island_colors[2])
    _paint_cells(top, cells4, island_colors[3])

    reserved = set()
    for c in range(min(arm1[1], arm2[1]) + 1, max(arm1[1], arm2[1])):
        reserved.add((arm1[0], c))
    for r in range(min(arm3[0], arm4[0]) + 1, max(arm3[0], arm4[0])):
        reserved.add((r, arm3[1]))

    remaining_colors = [color for color in PALETTE if color not in island_colors and color not in bridge_colors]
    extra_count = random.randint(0, len(remaining_colors))
    random.shuffle(remaining_colors)
    for color in remaining_colors[:extra_count]:
        placed = False
        blob = _random_blob_local()
        blob_h = max(r for r, _ in blob) + 1
        blob_w = max(c for _, c in blob) + 1
        if blob_h > top_h or blob_w > top_w:
            continue
        for _ in range(DISTRACTOR_ATTEMPTS):
            top_r = random.randint(0, top_h - blob_h)
            left_c = random.randint(0, top_w - blob_w)
            absolute = {(top_r + r, left_c + c) for r, c in blob}
            if absolute & reserved:
                continue
            if not _fits_with_padding(absolute, occupied, top_h, top_w):
                continue
            occupied |= absolute
            _paint_cells(top, absolute, color)
            placed = True
            break
        if not placed:
            continue

    expected = top.copy()
    _draw_bridge(expected, min(arm1, arm2, key=lambda p: p[1]), max(arm1, arm2, key=lambda p: p[1]), bridge_colors[0])
    _draw_bridge(expected, min(arm3, arm4, key=lambda p: p[0]), max(arm3, arm4, key=lambda p: p[0]), bridge_colors[1])

    # Validate that the intended arm pairs are the unique shortest connections.
    h_candidates = _bridge_candidates(top, island_colors[0], island_colors[1], BG)
    v_candidates = _bridge_candidates(top, island_colors[2], island_colors[3], BG)
    if not h_candidates or not v_candidates:
        return None
    if h_candidates[0][1] != "h" or v_candidates[0][1] != "v":
        return None
    if len(h_candidates) > 1 and h_candidates[1][0] == h_candidates[0][0]:
        return None
    if len(v_candidates) > 1 and v_candidates[1][0] == v_candidates[0][0]:
        return None
    expected_h = tuple(sorted((arm1, arm2), key=lambda p: p[1]))
    expected_v = tuple(sorted((arm3, arm4), key=lambda p: p[0]))
    if (h_candidates[0][2], h_candidates[0][3]) != expected_h:
        return None
    if (v_candidates[0][2], v_candidates[0][3]) != expected_v:
        return None

    full = np.full((top_h + 3, top_w), BG, dtype=int)
    full[:top_h, :] = top
    full[top_h, :] = GREY

    route_specs = [
        (island_colors[0], bridge_colors[0], island_colors[1]),
        (island_colors[2], bridge_colors[1], island_colors[3]),
    ]
    random.shuffle(route_specs)
    for i, route in enumerate(route_specs):
        a, bridge, b = route
        if random.random() < 0.5:
            triplet = [a, bridge, b]
        else:
            triplet = [b, bridge, a]
        start = random.randint(0, top_w - 3)
        full[top_h + 1 + i, start : start + 3] = np.array(triplet, dtype=int)

    if not np.array_equal(main(full), expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid routing-island puzzle")
