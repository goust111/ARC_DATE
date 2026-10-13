from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MOVING_PALETTE = (1, 2, 3, 4, 6, 8, 9)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert len(markers) == 2, "expected exactly two grey markers"
    (r0, c0), (r1, c1) = markers
    assert (r0 == r1) ^ (c0 == c1), "grey markers must share exactly one axis"
    assert abs(r0 - r1) + abs(c0 - c1) >= 4, "markers must be separated by at least three cells"
    return markers


def _axis_from_markers(markers: list[tuple[int, int]]) -> tuple[str, int]:
    (r0, c0), (r1, c1) = markers
    if r0 == r1:
        return "horizontal", r0
    return "vertical", c0


def _collect_moves(
    grid: np.ndarray, markers: list[tuple[int, int]], bg: int
) -> list[dict[str, int | tuple[int, int]]]:
    moving_cells = np.argwhere((grid != bg) & (grid != GREY) & (grid != ORANGE))
    assert 3 <= len(moving_cells) <= 4, "the non-conflicting setup allows three or four moving pixels"

    planned: dict[tuple[int, int], int] = {}
    used_markers: set[tuple[int, int]] = set()
    used_sides: set[int] = set()
    moves: list[dict[str, int | tuple[int, int]]] = []
    for r_raw, c_raw in moving_cells:
        r = int(r_raw)
        c = int(c_raw)
        color = int(grid[r, c])

        aligned = [(mr, mc) for mr, mc in markers if (r == mr) ^ (c == mc)]
        assert len(aligned) == 1, "each moving pixel must align with exactly one marker"
        mr, mc = aligned[0]

        dr = int(np.sign(r - mr))
        dc = int(np.sign(c - mc))
        assert abs(dr) + abs(dc) == 1, "movement must be cardinal"
        nr = mr + dr
        nc = mc + dc
        assert 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1], "landing cell must stay in bounds"
        assert (nr, nc) == (r, c) or int(grid[nr, nc]) == bg, "landing cell must be empty"
        assert (nr, nc) not in planned, "two moving pixels cannot share a landing cell"

        used_markers.add((mr, mc))
        used_sides.add(dr if dr else dc)
        planned[(nr, nc)] = color
        moves.append(
            {
                "src": (r, c),
                "dst": (nr, nc),
                "marker": (mr, mc),
                "color": color,
            }
        )

    assert len(used_markers) == 2, "both markers must act as compression anchors"
    assert used_sides == {-1, 1}, "examples must expose both approach sides"
    return moves


def _orange_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == ORANGE
    assert mask.any(), "expected one orange source rectangle"
    rect = _bbox(mask)
    r0, c0, r1, c1 = rect
    assert 2 <= r1 - r0 + 1 <= 3 and 2 <= c1 - c0 + 1 <= 3, "orange source must be a 2x2 to 3x3 rectangle"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "orange source must be solid"
    return rect


def _reflect_rect(rect: tuple[int, int, int, int], orientation: str, axis: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    if orientation == "horizontal":
        return 2 * axis - r1, c0, 2 * axis - r0, c1
    return r0, 2 * axis - c1, r1, 2 * axis - c0


def _corridor(rect: tuple[int, int, int, int], reflected: tuple[int, int, int, int], orientation: str):
    if orientation == "horizontal":
        fill_r0 = min(rect[2], reflected[2]) + 1
        fill_r1 = max(rect[0], reflected[0]) - 1
        return fill_r0, rect[1], fill_r1, rect[3]
    fill_c0 = min(rect[3], reflected[3]) + 1
    fill_c1 = max(rect[1], reflected[1]) - 1
    return rect[0], fill_c0, rect[2], fill_c1


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each aligned colored pixel onto its grey anchor, then reflect the orange rectangle across the axis implied by the two anchors.

    Input structure:
    - The background is black.
    - Exactly two grey singleton markers share one row or one column, with several non-grey/non-orange single pixels aligned to exactly one of them.
    - One solid orange rectangle lies strictly on one side of the row or column implied by the grey pair.

    Rule:
    - Keep the two grey markers fixed.
    - Move every scattered colored pixel straight toward its unique aligned grey marker until it occupies the marker-adjacent cell on that same row or column.
    - Treat the full row or column through the two grey markers as a reflection axis, mirror the orange rectangle across it, and fill the rectangular corridor between the original and mirrored rectangles with grey.
    - Preserve the compressed colored pixels and the grey markers.

    Key insight:
    - The same two grey points have a dual role: they are both the compression anchors for the scattered pixels and the endpoints that implicitly define the rectangle's reflection axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BLACK, "background must be black"

    markers = _parse_markers(g)
    orientation, axis = _axis_from_markers(markers)
    moves = _collect_moves(g, markers, bg)
    rect = _orange_rect(g)
    reflected = _reflect_rect(rect, orientation, axis)

    rr0, rc0, rr1, rc1 = reflected
    assert 0 <= rr0 <= rr1 < g.shape[0] and 0 <= rc0 <= rc1 < g.shape[1], "reflected rectangle must stay in bounds"

    out = g.copy()
    for move in moves:
        sr, sc = move["src"]
        out[sr, sc] = bg

    fr0, fc0, fr1, fc1 = _corridor(rect, reflected, orientation)
    assert fr0 <= fr1 and fc0 <= fc1, "corridor must be non-empty"
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = GREY
    _paint_rect(out, reflected, ORANGE)

    for mr, mc in markers:
        out[mr, mc] = GREY
    for move in moves:
        dr, dc = move["dst"]
        out[dr, dc] = int(move["color"])

    return out


def _expected_output(grid: np.ndarray, spec: dict) -> np.ndarray:
    out = grid.copy()
    rect = spec["rect"]
    reflected = _reflect_rect(rect, spec["orientation"], spec["axis"])
    for move in spec["moves"]:
        sr, sc = move["src"]
        out[sr, sc] = BLACK
    fr0, fc0, fr1, fc1 = _corridor(rect, reflected, spec["orientation"])
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = GREY
    _paint_rect(out, reflected, ORANGE)
    for mr, mc in spec["markers"]:
        out[mr, mc] = GREY
    for move in spec["moves"]:
        dr, dc = move["dst"]
        out[dr, dc] = move["color"]
    return out


def _positions_with_gap(length: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(1, length - 1) for b in range(a + 4, length - 1)]


def _spans_avoiding_points(length: int, span: int, points: tuple[int, int], margin: int = 1) -> list[int]:
    starts: list[int] = []
    p0, p1 = points
    for start in range(0, length - span + 1):
        end = start + span - 1
        if start - margin <= p0 <= end + margin:
            continue
        if start - margin <= p1 <= end + margin:
            continue
        starts.append(start)
    return starts


def _pick_direction_layout(rng: np.random.Generator, count: int) -> list[list[int]]:
    assert count in (3, 4)
    if count == 4:
        return [[-1, 1], [-1, 1]]
    full = int(rng.integers(0, 2))
    other = 1 - full
    layout = [[], []]
    layout[full] = [-1, 1]
    layout[other] = [int(rng.choice([-1, 1]))]
    return layout


def _build_horizontal(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, dict] | None:
    rect_h = int(rng.integers(2, 4))
    rect_w = int(rng.integers(2, 4))
    feasible_gaps = [gap for gap in range(2, 5) if h >= 2 * (rect_h + gap) - 1]
    if not feasible_gaps:
        return None
    gap = int(rng.choice(feasible_gaps))
    axis_min = gap + rect_h - 1
    axis_max = h - gap - rect_h
    axis = int(rng.integers(axis_min, axis_max + 1))

    marker_pairs = _positions_with_gap(w)
    if not marker_pairs:
        return None
    markers_cols = marker_pairs[int(rng.integers(0, len(marker_pairs)))]
    rect_starts = _spans_avoiding_points(w, rect_w, markers_cols, margin=1)
    if not rect_starts:
        return None
    c0 = int(rng.choice(rect_starts))
    c1 = c0 + rect_w - 1

    side = str(rng.choice(["top", "bottom"]))
    if side == "top":
        r1 = axis - gap
        r0 = r1 - rect_h + 1
    else:
        r0 = axis + gap
        r1 = r0 + rect_h - 1

    count = int(rng.integers(3, 5))
    direction_layout = _pick_direction_layout(rng, count)
    colors = [int(c) for c in rng.choice(MOVING_PALETTE, size=count, replace=False)]

    grid = np.full((h, w), BLACK, dtype=int)
    markers = [(axis, markers_cols[0]), (axis, markers_cols[1])]
    for mr, mc in markers:
        grid[mr, mc] = GREY
    rect = (r0, c0, r1, c1)
    _paint_rect(grid, rect, ORANGE)

    moves: list[dict] = []
    color_index = 0
    for marker_idx, dirs in enumerate(direction_layout):
        mr, mc = markers[marker_idx]
        for direction in dirs:
            max_dist = mr if direction == -1 else (h - 1 - mr)
            if max_dist < 2:
                return None
            dist = int(rng.integers(2, max_dist + 1))
            src = (mr + direction * dist, mc)
            dst = (mr + direction, mc)
            color = colors[color_index]
            color_index += 1
            if grid[src] != BLACK or grid[dst] != BLACK:
                return None
            grid[src] = color
            moves.append({"src": src, "dst": dst, "marker": (mr, mc), "color": color})

    spec = {
        "orientation": "horizontal",
        "axis": axis,
        "markers": markers,
        "rect": rect,
        "moves": moves,
    }
    return grid, spec


def _build_vertical(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, dict] | None:
    rect_h = int(rng.integers(2, 4))
    rect_w = int(rng.integers(2, 4))
    feasible_gaps = [gap for gap in range(2, 5) if w >= 2 * (rect_w + gap) - 1]
    if not feasible_gaps:
        return None
    gap = int(rng.choice(feasible_gaps))
    axis_min = gap + rect_w - 1
    axis_max = w - gap - rect_w
    axis = int(rng.integers(axis_min, axis_max + 1))

    marker_pairs = _positions_with_gap(h)
    if not marker_pairs:
        return None
    markers_rows = marker_pairs[int(rng.integers(0, len(marker_pairs)))]
    rect_starts = _spans_avoiding_points(h, rect_h, markers_rows, margin=1)
    if not rect_starts:
        return None
    r0 = int(rng.choice(rect_starts))
    r1 = r0 + rect_h - 1

    side = str(rng.choice(["left", "right"]))
    if side == "left":
        c1 = axis - gap
        c0 = c1 - rect_w + 1
    else:
        c0 = axis + gap
        c1 = c0 + rect_w - 1

    count = int(rng.integers(3, 5))
    direction_layout = _pick_direction_layout(rng, count)
    colors = [int(c) for c in rng.choice(MOVING_PALETTE, size=count, replace=False)]

    grid = np.full((h, w), BLACK, dtype=int)
    markers = [(markers_rows[0], axis), (markers_rows[1], axis)]
    for mr, mc in markers:
        grid[mr, mc] = GREY
    rect = (r0, c0, r1, c1)
    _paint_rect(grid, rect, ORANGE)

    moves: list[dict] = []
    color_index = 0
    for marker_idx, dirs in enumerate(direction_layout):
        mr, mc = markers[marker_idx]
        for direction in dirs:
            max_dist = mc if direction == -1 else (w - 1 - mc)
            if max_dist < 2:
                return None
            dist = int(rng.integers(2, max_dist + 1))
            src = (mr, mc + direction * dist)
            dst = (mr, mc + direction)
            color = colors[color_index]
            color_index += 1
            if grid[src] != BLACK or grid[dst] != BLACK:
                return None
            grid[src] = color
            moves.append({"src": src, "dst": dst, "marker": (mr, mc), "color": color})

    spec = {
        "orientation": "vertical",
        "axis": axis,
        "markers": markers,
        "rect": rect,
        "moves": moves,
    }
    return grid, spec


def _validate_generated_case(grid: np.ndarray, spec: dict) -> bool:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(get_bg(grid)) == BLACK

    markers = _parse_markers(grid)
    assert markers == spec["markers"]
    orientation, axis = _axis_from_markers(markers)
    assert orientation == spec["orientation"] and axis == spec["axis"]

    rect = _orange_rect(grid)
    assert rect == spec["rect"]
    reflected = _reflect_rect(rect, orientation, axis)
    rr0, rc0, rr1, rc1 = reflected
    assert 0 <= rr0 <= rr1 < grid.shape[0] and 0 <= rc0 <= rc1 < grid.shape[1]

    marker_usage: dict[tuple[int, int], int] = {marker: 0 for marker in markers}
    sides: set[int] = set()
    seen_sources: set[tuple[int, int]] = set()
    seen_targets: set[tuple[int, int]] = set()
    for move in spec["moves"]:
        src = tuple(move["src"])
        dst = tuple(move["dst"])
        marker = tuple(move["marker"])
        color = int(move["color"])
        assert src not in seen_sources and dst not in seen_targets
        seen_sources.add(src)
        seen_targets.add(dst)
        assert int(grid[src]) == color
        assert int(grid[dst]) == BLACK
        assert marker in markers
        marker_usage[marker] += 1

        if orientation == "horizontal":
            assert src[1] == marker[1] and dst[1] == marker[1]
            direction = int(np.sign(src[0] - marker[0]))
            assert direction in (-1, 1)
            assert src[0] == marker[0] + direction * abs(src[0] - marker[0])
            assert dst == (marker[0] + direction, marker[1])
        else:
            assert src[0] == marker[0] and dst[0] == marker[0]
            direction = int(np.sign(src[1] - marker[1]))
            assert direction in (-1, 1)
            assert dst == (marker[0], marker[1] + direction)
        sides.add(direction)

    assert all(count > 0 for count in marker_usage.values())
    assert sides == {-1, 1}

    expected = _expected_output(grid, spec)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(11, 16))
        w = int(rng.integers(11, 16))
        orientation = str(rng.choice(["horizontal", "vertical"]))
        built = _build_horizontal(rng, h, w) if orientation == "horizontal" else _build_vertical(rng, h, w)
        if built is None:
            continue
        grid, spec = built
        if _validate_generated_case(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
