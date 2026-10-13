from __future__ import annotations

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATE_ATTEMPTS = 400
STAIR_PATTERNS = ((1, 2, 2), (1, 2, 3), (2, 2, 3))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _shape_kind(coords: np.ndarray) -> str:
    if len(coords) == 1:
        return "pixel"
    if len(coords) == 2:
        rows = sorted(int(r) for r, _ in coords.tolist())
        cols = {int(c) for _, c in coords.tolist()}
        if len(cols) == 1 and rows[1] == rows[0] + 1:
            return "domino"
    raise AssertionError("expected each green object to be a single pixel or a vertical 2-pixel segment")


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(grid == GREEN):
        kind = _shape_kind(coords)
        rows = coords[:, 0]
        cols = coords[:, 1]
        objects.append(
            {
                "coords": coords.copy(),
                "kind": kind,
                "top": int(rows.min()),
                "bottom": int(rows.max()),
                "col": int(cols.min()),
            }
        )
    return objects


def _floor_segment(grid: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert len(coords) > 0, "expected a grey floor"
    rows = sorted({int(r) for r, _ in coords.tolist()})
    assert len(rows) == 1, "expected a one-row floor segment"
    row = rows[0]
    cols = sorted(int(c) for _, c in coords.tolist())
    assert cols == list(range(cols[0], cols[-1] + 1)), "expected the floor to be contiguous"
    return row, cols[0], cols[-1]


def _make_floor_mask(shape: tuple[int, int], floor_row: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(floor_row), int(c0) : int(c1) + 1] = True
    return mask


def _initial_blocked(floor_mask: np.ndarray) -> np.ndarray:
    blocked = floor_mask.copy()
    above = np.zeros_like(blocked)
    above[:-1, :] = floor_mask[1:, :]
    blocked |= above
    return blocked


def _mark_object(blocked: np.ndarray, coords: np.ndarray, add_buffer: bool) -> None:
    blocked[coords[:, 0], coords[:, 1]] = True
    if not add_buffer:
        return
    above = coords.copy()
    above[:, 0] -= 1
    good = above[:, 0] >= 0
    blocked[above[good, 0], above[good, 1]] = True


def _drop_one(coords: np.ndarray, blocked: np.ndarray, h: int) -> np.ndarray:
    cur = coords.copy()
    while True:
        shifted = cur + np.array([1, 0], dtype=int)
        if int(shifted[:, 0].max()) >= h:
            return cur
        if np.any(blocked[shifted[:, 0], shifted[:, 1]]):
            return cur
        cur = shifted


def _land_objects(
    objects: list[dict],
    floor_mask: np.ndarray,
    *,
    add_object_buffer: bool,
    accumulate: bool,
) -> tuple[list[dict], np.ndarray]:
    h, _w = floor_mask.shape
    blocked = _initial_blocked(floor_mask)
    landed = []
    order = sorted(objects, key=lambda obj: (-obj["bottom"], obj["col"], obj["top"]))
    for obj in order:
        coords = _drop_one(obj["coords"], blocked, h)
        rows = coords[:, 0]
        cols = coords[:, 1]
        landed_obj = {
            "coords": coords,
            "kind": obj["kind"],
            "top": int(rows.min()),
            "bottom": int(rows.max()),
            "col": int(cols.min()),
        }
        landed.append(landed_obj)
        if accumulate:
            _mark_object(blocked, coords, add_object_buffer)
    return landed, blocked


def _render_visible(shape: tuple[int, int], floor_mask: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[floor_mask] = GREY
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = GREEN
    return out


def _center(coords: np.ndarray) -> tuple[int, int]:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    r, c = coords[order[0]]
    return int(r), int(c)


def _connector_gaps(landed_grid: np.ndarray, blocked: np.ndarray, objects: list[dict]) -> list[list[tuple[int, int]]]:
    centers = [_center(obj["coords"]) for obj in objects]
    gaps: list[list[tuple[int, int]]] = []
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            dr = int(r2 - r1)
            dc = int(c2 - c1)
            if abs(dr) != abs(dc):
                continue
            step_r = 0 if dr == 0 else dr // abs(dr)
            step_c = 0 if dc == 0 else dc // abs(dc)
            gap = [(r1 + k * step_r, c1 + k * step_c) for k in range(1, abs(dr))]
            if all(int(landed_grid[r, c]) == BG and not blocked[r, c] for r, c in gap):
                gaps.append(gap)
    return gaps


def _solve_core(
    grid: np.ndarray,
    *,
    add_object_buffer: bool = True,
    accumulate: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    floor_row, floor_c0, floor_c1 = _floor_segment(g)
    floor_mask = _make_floor_mask(g.shape, floor_row, floor_c0, floor_c1)
    objects = _extract_objects(g)
    landed, blocked = _land_objects(
        objects,
        floor_mask,
        add_object_buffer=add_object_buffer,
        accumulate=accumulate,
    )
    settled = _render_visible(g.shape, floor_mask, landed)
    gaps = _connector_gaps(settled, blocked, landed)
    out = settled.copy()
    for gap in gaps:
        for r, c in gap:
            out[r, c] = RED
    return out, settled, landed, gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop green objects with a one-row landing cushion, then draw red diagonal connectors between clear aligned centers.

    Input structure:
    - The grid is a black background with one grey horizontal floor segment near the bottom.
    - Above the floor are several separated green objects, each either a single pixel or a vertical 2-pixel segment.
    - Green objects start above the floor's one-row cushion and do not touch each other.

    Rule:
    - Treat every floor pixel and the cell directly above it as blocked.
    - Process green objects from lower initial row to higher initial row, breaking ties from left to right, and drop each object straight down until one more step would hit blocked space, a landed object, or that landed object's one-row cushion.
    - After all objects land, use each object's center pixel (the unique pixel for size 1, the upper pixel for size 2) and connect every pair whose centers lie on the same 45-degree diagonal and whose interior diagonal cells are unobstructed background by coloring those interior cells red.
    - Keep the floor and landed green objects unchanged.

    Key insight:
    - The invisible cushion above every landed object changes later landing heights, and those emergent heights determine which diagonal connectors become possible.
    """
    out, _settled, _landed, _gaps = _solve_core(grid)
    return out


def _height(kind: str) -> int:
    return 1 if kind == "pixel" else 2


def _coords_from_top(kind: str, top: int, col: int) -> np.ndarray:
    if kind == "pixel":
        return np.array([[top, col]], dtype=int)
    return np.array([[top, col], [top + 1, col]], dtype=int)


def _build_input_grid(
    h: int,
    w: int,
    floor_row: int,
    floor_c0: int,
    floor_c1: int,
    stacks: list[tuple[int, str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    floor_mask = _make_floor_mask((h, w), floor_row, floor_c0, floor_c1)
    grid = np.full((h, w), BG, dtype=int)
    grid[floor_mask] = GREY

    landed_objects = []
    for col, kind, count in stacks:
        step = _height(kind) + 1
        base_top = floor_row - _height(kind) - 1
        highest_top = base_top - (count - 1) * step
        lift = int(np.random.randint(1, highest_top + 1))
        for level in range(count):
            landed_top = base_top - level * step
            input_top = landed_top - lift
            coords = _coords_from_top(kind, input_top, col)
            grid[coords[:, 0], coords[:, 1]] = GREEN
            landed_objects.append(
                {
                    "coords": _coords_from_top(kind, landed_top, col),
                    "kind": kind,
                    "top": landed_top,
                    "bottom": landed_top + _height(kind) - 1,
                    "col": col,
                }
            )
    expected = _render_visible((h, w), floor_mask, landed_objects)
    blocked = _initial_blocked(floor_mask)
    for obj in landed_objects:
        _mark_object(blocked, obj["coords"], True)
    for gap in _connector_gaps(expected, blocked, landed_objects):
        for r, c in gap:
            expected[r, c] = RED
    return grid, expected


def _green_objects_are_separate(grid: np.ndarray) -> bool:
    try:
        parts = _extract_objects(grid)
    except AssertionError:
        return False
    cells = []
    owners = {}
    for idx, obj in enumerate(parts):
        for r, c in obj["coords"].tolist():
            owners[(int(r), int(c))] = idx
            cells.append((int(r), int(c)))
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) == (r, c):
                    continue
                if (nr, nc) in owners and owners[(nr, nc)] != owners[(r, c)]:
                    return False
    return True


def _input_validity(grid: np.ndarray) -> bool:
    colors = set(int(x) for x in np.unique(grid))
    if not colors.issubset({BG, GREEN, GREY}):
        return False
    try:
        floor_row, _c0, _c1 = _floor_segment(grid)
    except AssertionError:
        return False
    try:
        objects = _extract_objects(grid)
    except AssertionError:
        return False
    if not (5 <= len(objects) <= 8):
        return False
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[floor_row, :] = grid[floor_row, :] == GREY
    if floor_row > 0:
        blocked[floor_row - 1, :] = grid[floor_row, :] == GREY
    for obj in objects:
        coords = obj["coords"]
        if np.any(blocked[coords[:, 0], coords[:, 1]]):
            return False
    return _green_objects_are_separate(grid)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if not _input_validity(inp):
        return False
    actual, settled, _landed, gaps = _solve_core(inp)
    if not np.array_equal(actual, expected):
        return False
    if len(gaps) < 2:
        return False
    if sum(len(gap) for gap in gaps) < 2:
        return False
    no_buffer, no_buffer_settled, _a, _b = _solve_core(inp, add_object_buffer=False, accumulate=True)
    no_dynamic, no_dynamic_settled, _c, _d = _solve_core(inp, add_object_buffer=True, accumulate=False)
    if np.array_equal(settled, no_buffer_settled):
        return False
    if np.array_equal(settled, no_dynamic_settled):
        return False
    if np.array_equal(actual, no_buffer) or np.array_equal(actual, no_dynamic):
        return False
    return True


def _candidate_extra_columns(floor_c0: int, floor_c1: int, primary_cols: list[int]) -> list[int]:
    used = set(primary_cols)
    candidates = []
    for c in range(floor_c0 + 1, floor_c1):
        if any(abs(c - pc) <= 1 for pc in used):
            continue
        if primary_cols[0] - 1 <= c <= primary_cols[-1] + 1:
            continue
        candidates.append(c)
    np.random.shuffle(candidates)
    return candidates


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = int(np.random.randint(15, 21))
    w = int(np.random.randint(16, 23))
    floor_row = h - 1
    floor_c0 = int(np.random.randint(0, 2))
    floor_c1 = w - 1 - int(np.random.randint(0, 2))

    primary_kind = "pixel" if np.random.rand() < 0.5 else "domino"
    primary_step = _height(primary_kind) + 1
    pattern = STAIR_PATTERNS[int(np.random.randint(0, len(STAIR_PATTERNS)))]

    c0_min = floor_c0 + 1
    c0_max = floor_c1 - 1 - 2 * primary_step
    if c0_min > c0_max:
        return None
    start_col = int(np.random.randint(c0_min, c0_max + 1))
    primary_cols = [start_col + i * primary_step for i in range(3)]
    stacks: list[tuple[int, str, int]] = [(primary_cols[i], primary_kind, pattern[i]) for i in range(3)]

    total_objects = sum(pattern)
    extra_target = int(np.random.randint(0, min(8 - total_objects, 2) + 1))
    extra_cols = _candidate_extra_columns(floor_c0, floor_c1, primary_cols)
    while extra_target > 0 and extra_cols:
        col = extra_cols.pop(0)
        kind = ("domino" if primary_kind == "pixel" else "pixel") if np.random.rand() < 0.7 else primary_kind
        count = 2 if extra_target >= 2 and np.random.rand() < 0.35 else 1
        stacks.append((col, kind, count))
        total_objects += count
        extra_target = max(0, min(8 - total_objects, extra_target - count))

    if not (5 <= total_objects <= 8):
        return None

    inp, expected = _build_input_grid(h, w, floor_row, floor_c0, floor_c1, stacks)
    if not _semantic_checks(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            inp, _expected = case
            return inp
    raise RuntimeError("failed to generate a valid input")
