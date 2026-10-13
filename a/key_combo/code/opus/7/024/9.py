from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PUSH_FROM_CONTACT = {
    (1, 0): ("T", (-1, 0)),
    (-1, 0): ("B", (1, 0)),
    (0, 1): ("L", (0, -1)),
    (0, -1): ("R", (0, 1)),
}

GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
MARKER_DIMS = [(2, 3), (2, 4), (3, 2), (4, 2)]
TEMPLATES = [
    {BLUE: ("T", "B"), RED: ("L", "R"), GREEN: ("T", "R")},
    {BLUE: ("T", "B"), RED: ("T", "L"), GREEN: ("B", "R")},
    {BLUE: ("L", "R"), RED: ("T", "B"), GREEN: ("B", "L")},
    {BLUE: ("L", "R"), RED: ("T", "R"), GREEN: ("B", "L")},
]


def _components(grid: np.ndarray, ignore: set[int] | None = None) -> list[dict]:
    ignore = set() if ignore is None else {int(v) for v in ignore}
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": len(coords),
                    "bbox": _bbox(arr),
                }
            )
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contact_side_and_push(grid: np.ndarray, coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    h, w = grid.shape
    contacts: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MAROON:
                contacts.add((dr, dc))
    assert len(contacts) == 1, "each object must touch the maroon marker on exactly one side"
    return PUSH_FROM_CONTACT[next(iter(contacts))]


def _steps_to_edge(shape: tuple[int, int], coords: np.ndarray, direction: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(h - 1 - coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _centroid_int(coords: np.ndarray) -> tuple[int, int]:
    area = int(coords.shape[0])
    row_sum = int(coords[:, 0].sum())
    col_sum = int(coords[:, 1].sum())
    assert row_sum % area == 0 and col_sum % area == 0, "centroid must land on integer grid coordinates"
    return row_sum // area, col_sum // area


def _ordered_endpoints(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if a[0] != b[0]:
        return (a, b) if a[0] < b[0] else (b, a)
    return (a, b) if a[1] <= b[1] else (b, a)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = _ordered_endpoints(a, b)
    if start[0] == end[0] or start[1] == end[1]:
        return _segment(start, end)
    turn = (start[0], end[1])
    return _segment(start, turn) + _segment(turn, end)[1:]


def _draw_dashed_path(out: np.ndarray, path: list[tuple[int, int]], blocked: np.ndarray) -> None:
    dash_index = 0
    for r, c in path:
        if blocked[r, c]:
            continue
        if dash_index % 2 == 0 and int(out[r, c]) == BG:
            out[r, c] = GREY
        dash_index += 1


def _is_straight_triomino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 3:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        cols = sorted(int(c) for c in coords[:, 1])
        return cols == list(range(cols[0], cols[0] + 3))
    if c0 == c1:
        rows = sorted(int(r) for r in coords[:, 0])
        return rows == list(range(rows[0], rows[0] + 3))
    return False


def _pair_kind(a: tuple[int, int], b: tuple[int, int]) -> str:
    if a[0] == b[0]:
        return "horizontal"
    if a[1] == b[1]:
        return "vertical"
    return "elbow"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push maroon-touching same-color objects to the boundary and connect each pair with grey dashes.

    Input structure:
    - The background is black.
    - One compact maroon 4-connected marker component sits in the central region.
    - Exactly six small monochrome 4-connected objects use blue, red, and green, with each color appearing in exactly two separate objects.
    - Every object lies wholly on one cardinal side of the maroon marker and is 4-adjacent to it on that side only.

    Rule:
    - For each non-maroon object, detect which side of the maroon marker it touches and push the whole object straight away from the marker until it reaches the grid boundary.
    - Erase each object from its original position, keep the maroon marker unchanged, and preserve every object's shape and color during the push.
    - After all pushes, pair the two objects of each color and compute their centroids.
    - Connect each same-color centroid pair with a grey dashed path: horizontal if they share a row, vertical if they share a column, otherwise an L path that runs horizontally on the smaller-row centroid's row and then vertically to the other centroid.
    - Along the path, count only the cells that are not maroon and not part of a pushed object; paint the even-numbered counted cells grey and leave the odd-numbered counted cells unchanged. Draw blue first, red second, and green last, without overwriting earlier grey crossings.

    Key insight:
    - The maroon marker first acts as a launcher that determines each object's outward direction, and then the pushed landing positions of the same-color objects become the unique geometric anchors that determine both the pairing layout and the dash rhythm.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    marker_comps = [comp for comp in _components(g, ignore={BG}) if int(comp["color"]) == MAROON]
    assert len(marker_comps) == 1

    movable = _components(g, ignore={BG, MAROON})
    assert len(movable) == 6

    out = g.copy()
    moved: list[dict] = []
    for comp in movable:
        coords = comp["coords"]
        side, direction = _contact_side_and_push(g, coords)
        steps = _steps_to_edge(g.shape, coords, direction)
        assert steps >= 1

        dr, dc = direction
        own = {tuple(map(int, cell)) for cell in coords.tolist()}
        for step in range(1, steps + 1):
            shifted = coords + np.array([dr * step, dc * step], dtype=int)
            for rr, cc in shifted:
                cell = (int(rr), int(cc))
                if cell not in own:
                    assert int(g[cell]) == BG, "push path must stay clear until the boundary"

        target = coords + np.array([dr * steps, dc * steps], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BG
        moved.append({"color": int(comp["color"]), "coords": target, "side": side})

    for comp in moved:
        coords = comp["coords"]
        assert bool(np.all(out[coords[:, 0], coords[:, 1]] == BG))
        out[coords[:, 0], coords[:, 1]] = int(comp["color"])

    blocked = out != BG
    by_color: dict[int, list[np.ndarray]] = {BLUE: [], RED: [], GREEN: []}
    for comp in moved:
        by_color[int(comp["color"])].append(comp["coords"])

    for color in (BLUE, RED, GREEN):
        pair = by_color[color]
        assert len(pair) == 2
        path = _path_cells(_centroid_int(pair[0]), _centroid_int(pair[1]))
        _draw_dashed_path(out, path, blocked)

    return out


def _marker_grid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), MAROON, dtype=int)


def _input_object_cells(side: str, slot: int, marker_top: int, marker_left: int, marker_h: int, marker_w: int) -> list[tuple[int, int]]:
    if side == "T":
        col = marker_left + slot
        return [(marker_top - 3 + k, col) for k in range(3)]
    if side == "B":
        col = marker_left + slot
        return [(marker_top + marker_h + k, col) for k in range(3)]
    if side == "L":
        row = marker_top + slot
        return [(row, marker_left - 3 + k) for k in range(3)]
    row = marker_top + slot
    return [(row, marker_left + marker_w + k) for k in range(3)]


def _central_choices(low: int, high: int, span: int) -> list[int]:
    values = list(range(low, high + 1))
    midpoint = (span - 1) / 2.0
    chosen = [v for v in values if abs((v + 0.5) - midpoint) <= max(1.5, span / 6.0)]
    return chosen if chosen else values


def _assign_slots(rng: random.Random, template: dict[int, tuple[str, str]], capacities: dict[str, int]) -> dict[int, tuple[tuple[str, int], tuple[str, int]]] | None:
    available = {side: list(range(cap)) for side, cap in capacities.items()}
    order = sorted(template, key=lambda color: 0 if set(template[color]) in ({"T", "B"}, {"L", "R"}) else 1)
    assignment: dict[int, tuple[tuple[str, int], tuple[str, int]]] = {}

    def rec(index: int) -> bool:
        if index == len(order):
            return True
        color = order[index]
        side_a, side_b = template[color]
        options: list[tuple[int, int]] = []
        if set((side_a, side_b)) in ({"T", "B"}, {"L", "R"}):
            shared = sorted(set(available[side_a]) & set(available[side_b]))
            options = [(slot, slot) for slot in shared]
        else:
            options = [(slot_a, slot_b) for slot_a in available[side_a] for slot_b in available[side_b]]
        rng.shuffle(options)
        for slot_a, slot_b in options:
            available[side_a].remove(slot_a)
            available[side_b].remove(slot_b)
            assignment[color] = ((side_a, slot_a), (side_b, slot_b))
            if rec(index + 1):
                return True
            del assignment[color]
            available[side_a].append(slot_a)
            available[side_b].append(slot_b)
            available[side_a].sort()
            available[side_b].sort()
        return False

    if rec(0):
        return assignment
    return None


def _build_input(
    h: int,
    w: int,
    marker_top: int,
    marker_left: int,
    marker_h: int,
    marker_w: int,
    assignment: dict[int, tuple[tuple[str, int], tuple[str, int]]],
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    marker = _marker_grid(marker_h, marker_w)
    grid[marker_top : marker_top + marker_h, marker_left : marker_left + marker_w] = marker

    occupied = {
        (r, c)
        for r in range(marker_top, marker_top + marker_h)
        for c in range(marker_left, marker_left + marker_w)
    }
    for color in (BLUE, RED, GREEN):
        for side, slot in assignment[color]:
            cells = _input_object_cells(side, slot, marker_top, marker_left, marker_h, marker_w)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                return None
            if any((r, c) in occupied for r, c in cells):
                return None
            for r, c in cells:
                occupied.add((r, c))
                grid[r, c] = color
    return grid


def _path_free_count(path: list[tuple[int, int]], blocked: np.ndarray) -> int:
    return sum(0 if blocked[r, c] else 1 for r, c in path)


def _validate_instance(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30

    marker_comps = [comp for comp in _components(inp, ignore={BG}) if int(comp["color"]) == MAROON]
    assert len(marker_comps) == 1
    marker = marker_comps[0]
    assert 4 <= int(marker["area"]) <= 8
    mr0, mc0, mr1, mc1 = marker["bbox"]

    movable = _components(inp, ignore={BG, MAROON})
    assert len(movable) == 6
    assert Counter(int(comp["color"]) for comp in movable) == Counter({BLUE: 2, RED: 2, GREEN: 2})

    moved_cells: set[tuple[int, int]] = set()
    moved_by_color: dict[int, list[np.ndarray]] = {BLUE: [], RED: [], GREEN: []}
    side_by_color: dict[int, list[str]] = {BLUE: [], RED: [], GREEN: []}
    sides_seen: set[str] = set()

    for comp in movable:
        coords = comp["coords"]
        color = int(comp["color"])
        assert _is_straight_triomino(coords)

        side, direction = _contact_side_and_push(inp, coords)
        sides_seen.add(side)
        side_by_color[color].append(side)

        if side == "T":
            assert int(coords[:, 0].max()) < mr0
        elif side == "B":
            assert int(coords[:, 0].min()) > mr1
        elif side == "L":
            assert int(coords[:, 1].max()) < mc0
        else:
            assert int(coords[:, 1].min()) > mc1

        steps = _steps_to_edge(inp.shape, coords, direction)
        assert steps >= 1
        dr, dc = direction
        own = {tuple(map(int, cell)) for cell in coords.tolist()}
        for step in range(1, steps + 1):
            shifted = coords + np.array([dr * step, dc * step], dtype=int)
            for rr, cc in shifted:
                cell = (int(rr), int(cc))
                if cell not in own:
                    assert int(inp[cell]) == BG

        moved = coords + np.array([dr * steps, dc * steps], dtype=int)
        for rr, cc in moved:
            cell = (int(rr), int(cc))
            assert cell not in moved_cells
            moved_cells.add(cell)
        moved_by_color[color].append(moved)

        if side == "T":
            assert int(moved[:, 0].min()) == 0
        elif side == "B":
            assert int(moved[:, 0].max()) == inp.shape[0] - 1
        elif side == "L":
            assert int(moved[:, 1].min()) == 0
        else:
            assert int(moved[:, 1].max()) == inp.shape[1] - 1

    assert len(sides_seen) >= 3
    for color in (BLUE, RED, GREEN):
        assert len(side_by_color[color]) == 2
        assert side_by_color[color][0] != side_by_color[color][1]

    placed = np.full_like(inp, BG)
    placed[marker["coords"][:, 0], marker["coords"][:, 1]] = MAROON
    for color in (BLUE, RED, GREEN):
        for moved in moved_by_color[color]:
            placed[moved[:, 0], moved[:, 1]] = color
    blocked = placed != BG

    pair_kinds = set()
    free_counts = []
    for color in (BLUE, RED, GREEN):
        centroids = [_centroid_int(coords) for coords in moved_by_color[color]]
        kind = _pair_kind(centroids[0], centroids[1])
        pair_kinds.add(kind)
        path = _path_cells(centroids[0], centroids[1])
        free_counts.append(_path_free_count(path, blocked))
        assert free_counts[-1] >= 3

    assert "elbow" in pair_kinds
    assert "horizontal" in pair_kinds or "vertical" in pair_kinds

    out = main(inp)
    assert out.shape == inp.shape
    assert not np.array_equal(out, inp)
    assert np.any(out == GREY)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_h, marker_w = rng.choice(MARKER_DIMS)

    top_low = 4
    top_high = h - marker_h - 4
    left_low = 4
    left_high = w - marker_w - 4
    if top_low > top_high or left_low > left_high:
        return None

    marker_top = rng.choice(_central_choices(top_low, top_high, h))
    marker_left = rng.choice(_central_choices(left_low, left_high, w))

    template = rng.choice(TEMPLATES)
    capacities = {"T": marker_w, "B": marker_w, "L": marker_h, "R": marker_h}
    assignment = _assign_slots(rng, template, capacities)
    if assignment is None:
        return None

    inp = _build_input(h, w, marker_top, marker_left, marker_h, marker_w, assignment)
    if inp is None:
        return None

    try:
        _validate_instance(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle-9 input")
