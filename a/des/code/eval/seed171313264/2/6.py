from __future__ import annotations

import random
from collections import Counter

import numpy as np


LEFT_MARKER = 2
RIGHT_MARKER = 1

BG_COLOR = 0
MARKER_COLORS = (LEFT_MARKER, RIGHT_MARKER)
NON_MARKER_POOL = (3, 4, 5, 6, 7, 8, 9)
TURN_COUNT_CHOICES = (1, 3)
GRID_BOUNDS = (20, 30)
TURN_MARGIN = 5
TURN_GAP = 6
PLACEMENT_GAP = 1
GEN_RETRIES = 300

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _solid_square_component(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    size_r = int(r1 - r0 + 1)
    size_c = int(c1 - c0 + 1)
    if size_r != size_c or size_r not in {2, 3, 4}:
        return None
    if int(coords.shape[0]) != size_r * size_r:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool((patch == int(color)).all()):
        return None
    return {
        "color": int(color),
        "size": int(size_r),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": coords,
    }


def _extract_squares(grid: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        for coords in _component_coords(grid == int(color)):
            sq = _solid_square_component(grid, color, coords)
            if sq is not None:
                squares.append(sq)
    sizes = [int(sq["size"]) for sq in squares]
    assert len(sizes) == len(set(sizes)), "square sizes must be distinct"
    return squares


def _find_path_color(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    candidates: list[int] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        mask = grid == int(color)
        comps = _component_coords(mask)
        if len(comps) != 1:
            continue
        coords = comps[0]
        if int(coords.shape[0]) < 3:
            continue
        deg1 = 0
        ok = True
        boundary_endpoints = 0
        for r, c in coords:
            degree = sum(1 for nr, nc in _neighbors4(int(r), int(c), height, width) if bool(mask[nr, nc]))
            if degree == 1:
                deg1 += 1
                if int(r) in {0, height - 1} or int(c) in {0, width - 1}:
                    boundary_endpoints += 1
            elif degree != 2:
                ok = False
                break
        if ok and deg1 == 2 and boundary_endpoints == 2:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected exactly one simple path color"
    return int(candidates[0])


def _path_endpoints(path_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = path_mask.shape
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(path_mask):
        degree = sum(1 for nr, nc in _neighbors4(int(r), int(c), height, width) if bool(path_mask[nr, nc]))
        if degree == 1:
            endpoints.append((int(r), int(c)))
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    return endpoints


def _traverse_path(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = path_mask.shape
    path: list[tuple[int, int]] = [tuple(map(int, start))]
    prev: tuple[int, int] | None = None
    cur = tuple(map(int, start))
    while True:
        nbrs = [(nr, nc) for nr, nc in _neighbors4(cur[0], cur[1], height, width) if bool(path_mask[nr, nc])]
        nxts = [pt for pt in nbrs if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be non-branching"
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    return int(np.sign(dr)), int(np.sign(dc))


def _turn_kind(prev_pt: tuple[int, int], cur_pt: tuple[int, int], next_pt: tuple[int, int]) -> str:
    d1 = _step_dir(prev_pt, cur_pt)
    d2 = _step_dir(cur_pt, next_pt)
    if d1 == d2:
        return "straight"
    i1 = DIR_INDEX[d1]
    i2 = DIR_INDEX[d2]
    delta = int((i2 - i1) % 4)
    assert delta in {1, 3}, "turn must be left or right"
    return "right" if delta == 1 else "left"


def _marker_for_turn(
    grid: np.ndarray,
    square_mask: np.ndarray,
    path_color: int,
    bg: int,
    turn: tuple[int, int],
) -> tuple[int, int, int]:
    height, width = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for nr, nc in _neighbors4(turn[0], turn[1], height, width):
        color = int(grid[nr, nc])
        if color == int(bg) or color == int(path_color) or bool(square_mask[nr, nc]):
            continue
        candidates.append((int(nr), int(nc), int(color)))
    assert len(candidates) == 1, "each turn must have exactly one adjacent marker"
    return candidates[0]


def _ordered_turns(
    grid: np.ndarray,
    path_color: int,
    squares: list[dict],
    bg: int,
) -> list[dict]:
    path_mask = grid == int(path_color)
    endpoints = _path_endpoints(path_mask)
    square_mask = np.zeros_like(grid, dtype=bool)
    for sq in squares:
        coords = sq["coords"]
        square_mask[coords[:, 0], coords[:, 1]] = True

    candidates: list[list[dict]] = []
    for start in endpoints:
        path = _traverse_path(path_mask, start)
        turns: list[dict] = []
        valid = True
        for i in range(1, len(path) - 1):
            kind = _turn_kind(path[i - 1], path[i], path[i + 1])
            if kind == "straight":
                continue
            mr, mc, marker_color = _marker_for_turn(grid, square_mask, path_color, bg, path[i])
            expected = LEFT_MARKER if kind == "left" else RIGHT_MARKER
            if int(marker_color) != int(expected):
                valid = False
                break
            turns.append(
                {
                    "turn": tuple(map(int, path[i])),
                    "kind": kind,
                    "marker": (int(mr), int(mc)),
                    "marker_color": int(marker_color),
                }
            )
        if valid:
            candidates.append(turns)

    assert candidates, "could not infer path direction from markers"
    if len(candidates) == 1:
        return candidates[0]

    # Ambiguous cases are not generated, but keep a deterministic fallback.
    candidates.sort(key=lambda turns: [info["turn"] for info in turns])
    return candidates[0]


def _anchor_at_turn(turn: tuple[int, int], size: int) -> tuple[int, int]:
    # Even squares cannot be centered exactly on one cell; use upper-left-biased integer centering.
    return int(turn[0] - size // 2), int(turn[1] - size // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the solid squares to successive turning points of the polyline path.

    Squares are matched to turns by ascending square size. The moved square takes
    the adjacent turn-marker color, while the original path and marker pixels stay visible.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    squares = _extract_squares(g, bg)
    path_color = _find_path_color(g, bg)
    turns = _ordered_turns(g, path_color, squares, bg)
    assert len(squares) == len(turns), "square count must match turn count"

    out = g.copy()
    for sq in squares:
        r0, c0, r1, c1 = sq["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    for sq, turn_info in zip(sorted(squares, key=lambda item: int(item["size"])), turns):
        size = int(sq["size"])
        color = int(turn_info["marker_color"])
        r0, c0 = _anchor_at_turn(turn_info["turn"], size)
        r1 = int(r0 + size)
        c1 = int(c0 + size)
        assert 0 <= r0 and 0 <= c0 and r1 <= g.shape[0] and c1 <= g.shape[1]
        out[r0:r1, c0:c1] = int(color)

    path_mask = g == int(path_color)
    out[path_mask] = int(path_color)
    marker_mask = np.isin(g, MARKER_COLORS) & ~path_mask
    out[marker_mask] = g[marker_mask]
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _sample_points(turn_count: int, height: int, width: int) -> list[tuple[int, int]] | None:
    y0 = random.randint(TURN_MARGIN, height - TURN_MARGIN - 1)
    if turn_count == 1:
        x1 = random.randint(TURN_MARGIN, width - TURN_MARGIN - 1)
        end_r = 0 if random.random() < 0.5 else height - 1
        return [(int(y0), 0), (int(y0), int(x1)), (int(end_r), int(x1))]

    if width < 2 * TURN_MARGIN + TURN_GAP + 2:
        return None

    x1_hi = width - TURN_MARGIN - TURN_GAP - 1
    if x1_hi < TURN_MARGIN:
        return None
    x1 = random.randint(TURN_MARGIN, x1_hi)
    y_choices = [y for y in range(TURN_MARGIN, height - TURN_MARGIN) if abs(int(y - y0)) >= TURN_GAP]
    if not y_choices:
        return None
    y1 = random.choice(y_choices)
    x2 = random.randint(x1 + TURN_GAP, width - TURN_MARGIN - 1)
    end_r = 0 if random.random() < 0.5 else height - 1
    return [(int(y0), 0), (int(y0), int(x1)), (int(y1), int(x1)), (int(y1), int(x2)), (int(end_r), int(x2))]


def _marker_specs(
    points: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    used = set()
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1]
        turn = points[i]
        next_pt = points[i + 1]
        kind = _turn_kind(prev_pt, turn, next_pt)
        if kind == "straight":
            continue
        color = LEFT_MARKER if kind == "left" else RIGHT_MARKER
        choices = []
        for nr, nc in _neighbors4(turn[0], turn[1], height, width):
            if (nr, nc) in path_cells or (nr, nc) in used:
                continue
            choices.append((int(nr), int(nc)))
        assert len(choices) >= 1
        mr, mc = random.choice(choices)
        used.add((int(mr), int(mc)))
        markers.append((int(mr), int(mc), int(color)))
    return markers


def _rect_conflicts(r0: int, c0: int, size: int, rects: list[tuple[int, int, int]]) -> bool:
    for rr, cc, ss in rects:
        if not (
            r0 + size - 1 < rr - PLACEMENT_GAP
            or rr + ss - 1 < r0 - PLACEMENT_GAP
            or c0 + size - 1 < cc - PLACEMENT_GAP
            or cc + ss - 1 < c0 - PLACEMENT_GAP
        ):
            return True
    return False


def _place_squares(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
    sizes: list[int],
    colors: list[int],
) -> list[dict] | None:
    height, width = grid.shape
    forbidden = path_cells | marker_cells
    placed_rects: list[tuple[int, int, int]] = []
    placed: list[dict] = []
    order = list(range(len(sizes)))
    random.shuffle(order)
    for idx in order:
        size = int(sizes[idx])
        color = int(colors[idx])
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - size + 1):
            for c0 in range(width - size + 1):
                rr0 = max(0, int(r0 - PLACEMENT_GAP))
                cc0 = max(0, int(c0 - PLACEMENT_GAP))
                rr1 = min(height - 1, int(r0 + size - 1 + PLACEMENT_GAP))
                cc1 = min(width - 1, int(c0 + size - 1 + PLACEMENT_GAP))
                blocked = False
                for rr in range(rr0, rr1 + 1):
                    for cc in range(cc0, cc1 + 1):
                        if (rr, cc) in forbidden:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked or _rect_conflicts(r0, c0, size, placed_rects):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + size, c0 : c0 + size] = int(color)
        placed_rects.append((int(r0), int(c0), int(size)))
        placed.append({"size": int(size), "color": int(color), "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1))})
        for rr in range(r0, r0 + size):
            for cc in range(c0, c0 + size):
                forbidden.add((int(rr), int(cc)))
    return placed


def _paint_expected(
    inp: np.ndarray,
    path_color: int,
    squares: list[dict],
    points: list[tuple[int, int]],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for sq in squares:
        r0, c0, r1, c1 = sq["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG_COLOR

    turn_infos = []
    marker_map = {(int(r), int(c)): int(color) for r, c, color in markers}
    for i in range(1, len(points) - 1):
        kind = _turn_kind(points[i - 1], points[i], points[i + 1])
        if kind == "straight":
            continue
        turn = points[i]
        adj_markers = [(nr, nc) for nr, nc in _neighbors4(turn[0], turn[1], inp.shape[0], inp.shape[1]) if (nr, nc) in marker_map]
        assert len(adj_markers) == 1
        turn_infos.append({"turn": turn, "marker_color": int(marker_map[adj_markers[0]])})

    for sq, turn_info in zip(sorted(squares, key=lambda item: int(item["size"])), turn_infos):
        size = int(sq["size"])
        r0, c0 = _anchor_at_turn(turn_info["turn"], size)
        out[r0 : r0 + size, c0 : c0 + size] = int(turn_info["marker_color"])

    path_mask = inp == int(path_color)
    out[path_mask] = int(path_color)
    marker_mask = np.isin(inp, MARKER_COLORS) & ~path_mask
    out[marker_mask] = inp[marker_mask]
    return out


def _rot90_times(grid: np.ndarray, times: int) -> np.ndarray:
    return np.rot90(grid, k=int(times)).copy()


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_BOUNDS)
    width = random.randint(*GRID_BOUNDS)
    turn_count = random.choice(TURN_COUNT_CHOICES)
    points = _sample_points(turn_count, height, width)
    if points is None:
        return None

    path_cells = _polyline_cells(points)
    path_set = set(path_cells)
    if len(path_set) != len(path_cells):
        return None

    markers = _marker_specs(points, path_set, height, width)
    if len(markers) != turn_count:
        return None

    available = list(NON_MARKER_POOL)
    random.shuffle(available)
    path_color = int(available.pop())
    if turn_count == 1:
        sizes = [random.choice([2, 3, 4])]
    else:
        sizes = [2, 3, 4]
    square_colors = [int(available.pop()) for _ in sizes]

    inp = np.full((height, width), BG_COLOR, dtype=int)
    for r, c in path_cells:
        inp[r, c] = int(path_color)
    for r, c, color in markers:
        assert int(inp[r, c]) == BG_COLOR
        inp[r, c] = int(color)

    placed = _place_squares(inp, path_set, {(r, c) for r, c, _color in markers}, sizes, square_colors)
    if placed is None:
        return None

    expected = _paint_expected(inp, path_color, placed, points, markers)
    rot = random.randint(0, 3)
    inp = _rot90_times(inp, rot)
    expected = _rot90_times(expected, rot)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
