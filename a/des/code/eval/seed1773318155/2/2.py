from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
SIDES = ("top", "right", "bottom", "left")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    return -dc, dr


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    return dc, -dr


def _is_path_value(value: int) -> bool:
    return int(value) in (GREY, ORANGE)


def _grey_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY:
            out.append((rr, cc))
    return out


def _collect_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for r in range(int(grid.shape[0])):
        for c in range(int(grid.shape[1])):
            value = int(grid[r, c])
            if value in (BG, GREY, ORANGE):
                continue
            if len(_grey_neighbors(grid, r, c)) == 1:
                sources.append((r, c))
    return sources


def _choose_next_direction(
    grid: np.ndarray, r: int, c: int, direction: tuple[int, int]
) -> tuple[int, int] | None:
    forward = tuple(map(int, direction))
    left = _left_of(forward)
    right = _right_of(forward)

    choices: dict[str, tuple[tuple[int, int], int]] = {}
    for name, step in (("F", forward), ("L", left), ("R", right)):
        rr = int(r + step[0])
        cc = int(c + step[1])
        if not _in_bounds(grid.shape, rr, cc):
            continue
        value = int(grid[rr, cc])
        if _is_path_value(value):
            choices[name] = (step, value)

    has_f = "F" in choices
    has_l = "L" in choices
    has_r = "R" in choices

    # Turning is controlled by the set of available forward/left/right exits:
    # single side branch -> take it; stem-entry T -> turn right; cross -> turn left.
    if has_l and has_f and has_r:
        return choices["L"][0]
    if has_l and has_r and not has_f:
        return choices["R"][0]
    if has_l ^ has_r:
        return choices["L"][0] if has_l else choices["R"][0]
    if has_f:
        return choices["F"][0]
    return None


def _trace_signal(
    grid: np.ndarray, source_r: int, source_c: int
) -> tuple[int, int, int] | None:
    source_color = int(grid[source_r, source_c])
    grey_nbrs = _grey_neighbors(grid, source_r, source_c)
    if len(grey_nbrs) != 1:
        return None

    r, c = grey_nbrs[0]
    direction = (int(r - source_r), int(c - source_c))
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(direction[0]), int(direction[1]))
        if state in seen:
            return None
        seen.add(state)

        step = _choose_next_direction(grid, r, c, direction)
        if step is None:
            return None

        rr = int(r + step[0])
        cc = int(c + step[1])
        value = int(grid[rr, cc])
        if value == ORANGE:
            return rr, cc, source_color
        if value != GREY:
            return None
        r, c = rr, cc
        direction = step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored signal through a grey orthogonal path network and
    recolor the reached orange endpoint.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for source_r, source_c in _collect_sources(g):
        result = _trace_signal(g, source_r, source_c)
        if result is None:
            continue
        target_r, target_c, color = result
        out[target_r, target_c] = int(color)

    return out


def _paint_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    lo = min(int(c0), int(c1))
    hi = max(int(c0), int(c1))
    grid[int(row), lo : hi + 1] = int(color)


def _paint_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    lo = min(int(r0), int(r1))
    hi = max(int(r0), int(r1))
    grid[lo : hi + 1, int(col)] = int(color)


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    for _ in range(int(k) % 4):
        r, c = int(w - 1 - c), int(r)
        h, w = int(w), int(h)
    return int(r), int(c)


def _rotate_side(side: str | None, k: int) -> str | None:
    if side is None:
        return None
    idx = SIDES.index(side)
    return SIDES[(idx - int(k)) % 4]


def _rotate_component(component: dict, k: int) -> dict:
    grid = np.rot90(component["grid"], int(k))
    shape = component["grid"].shape
    return {
        "grid": grid,
        "target": None
        if component["target"] is None
        else _rotate_coord(component["target"], shape, k),
        "source": _rotate_coord(component["source"], shape, k),
        "oranges": [_rotate_coord(pos, shape, k) for pos in component["oranges"]],
        "edge_side": _rotate_side(component["edge_side"], k),
    }


def _build_single_branch(source_color: int) -> dict:
    lead = random.randint(1, 4)
    forward = random.randint(1, 4)
    branch = random.randint(1, 4)

    height = branch + 3
    width = lead + forward + 2
    center_c = lead

    grid = np.full((height, width), BG, dtype=int)
    grid[1, 0] = int(source_color)
    _paint_h(grid, 1, 1, center_c + forward, GREY)
    _paint_v(grid, center_c, 2, branch + 1, GREY)
    grid[1, center_c + forward + 1] = ORANGE
    grid[branch + 2, center_c] = ORANGE

    return {
        "grid": grid,
        "source": (1, 0),
        "target": (branch + 2, center_c),
        "oranges": [(1, center_c + forward + 1), (branch + 2, center_c)],
        "edge_side": None,
    }


def _build_two_branch_right(source_color: int) -> dict:
    stem = random.randint(1, 4)
    left_span = random.randint(1, 4)
    right_span = random.randint(1, 4)

    height = stem + 2
    width = left_span + right_span + 3
    center_c = left_span + 1

    grid = np.full((height, width), BG, dtype=int)
    _paint_h(grid, 0, 1, width - 2, GREY)
    _paint_v(grid, center_c, 1, stem, GREY)
    grid[0, 0] = ORANGE
    grid[0, width - 1] = ORANGE
    grid[stem + 1, center_c] = int(source_color)

    return {
        "grid": grid,
        "source": (stem + 1, center_c),
        "target": (0, width - 1),
        "oranges": [(0, 0), (0, width - 1)],
        "edge_side": None,
    }


def _build_three_branch_left(source_color: int) -> dict:
    up_span = random.randint(1, 4)
    down_span = random.randint(1, 4)
    left_span = random.randint(1, 4)
    right_span = random.randint(1, 4)

    center_r = up_span + 1
    center_c = left_span + 1
    height = up_span + down_span + 3
    width = left_span + right_span + 3

    grid = np.full((height, width), BG, dtype=int)
    _paint_h(grid, center_r, 1, width - 2, GREY)
    _paint_v(grid, center_c, 1, center_r - 1, GREY)
    _paint_v(grid, center_c, center_r + 1, center_r + down_span, GREY)
    grid[center_r, 0] = ORANGE
    grid[0, center_c] = ORANGE
    grid[center_r, width - 1] = ORANGE
    grid[center_r + down_span + 1, center_c] = int(source_color)

    return {
        "grid": grid,
        "source": (center_r + down_span + 1, center_c),
        "target": (center_r, 0),
        "oranges": [(center_r, 0), (0, center_c), (center_r, width - 1)],
        "edge_side": None,
    }


def _build_edge_exit(source_color: int) -> dict:
    line_len = random.randint(2, 6)
    width = line_len + 1
    grid = np.full((1, width), BG, dtype=int)
    grid[0, 0] = int(source_color)
    _paint_h(grid, 0, 1, width - 1, GREY)
    return {
        "grid": grid,
        "source": (0, 0),
        "target": None,
        "oranges": [],
        "edge_side": "right",
    }


def _build_component(kind: str, source_color: int) -> dict:
    if kind == "single":
        base = _build_single_branch(source_color)
    elif kind == "two_right":
        base = _build_two_branch_right(source_color)
    elif kind == "three_left":
        base = _build_three_branch_left(source_color)
    else:
        assert kind == "edge"
        base = _build_edge_exit(source_color)
    return _rotate_component(base, random.randint(0, 3))


def _bbox_is_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, int(r0 - 1))
    rr1 = min(int(occupied.shape[0]), int(r0 + h + 1))
    cc0 = max(0, int(c0 - 1))
    cc1 = min(int(occupied.shape[1]), int(c0 + w + 1))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_component(
    canvas: np.ndarray, occupied: np.ndarray, component: dict
) -> dict | None:
    comp = component["grid"]
    h, w = map(int, comp.shape)
    H, W = map(int, canvas.shape)
    side = component["edge_side"]

    positions: list[tuple[int, int]] = []
    if side is None:
        if H - h < 2 or W - w < 2:
            return None
        for _ in range(80):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            positions.append((r0, c0))
    elif side == "left":
        if H - h < 0:
            return None
        lo = 1 if H - h >= 2 else 0
        hi = H - h - 1 if H - h >= 2 else H - h
        for _ in range(80):
            positions.append((random.randint(lo, hi), 0))
    elif side == "right":
        if H - h < 0 or W - w < 0:
            return None
        lo = 1 if H - h >= 2 else 0
        hi = H - h - 1 if H - h >= 2 else H - h
        for _ in range(80):
            positions.append((random.randint(lo, hi), W - w))
    elif side == "top":
        if H - h < 0:
            return None
        lo = 1 if W - w >= 2 else 0
        hi = W - w - 1 if W - w >= 2 else W - w
        for _ in range(80):
            positions.append((0, random.randint(lo, hi)))
    else:
        assert side == "bottom"
        if H - h < 0 or W - w < 0:
            return None
        lo = 1 if W - w >= 2 else 0
        hi = W - w - 1 if W - w >= 2 else W - w
        for _ in range(80):
            positions.append((H - h, random.randint(lo, hi)))

    random.shuffle(positions)
    for r0, c0 in positions:
        if not _bbox_is_clear(occupied, r0, c0, h, w):
            continue
        mask = comp != BG
        canvas[r0 : r0 + h, c0 : c0 + w][mask] = comp[mask]
        occupied[r0 : r0 + h, c0 : c0 + w][mask] = True

        placed = {
            "source": (r0 + component["source"][0], c0 + component["source"][1]),
            "target": None
            if component["target"] is None
            else (r0 + component["target"][0], c0 + component["target"][1]),
            "oranges": [(r0 + rr, c0 + cc) for rr, cc in component["oranges"]],
        }
        return placed
    return None


def _validate_candidate(grid: np.ndarray, placed: list[dict]) -> bool:
    out = main(grid)
    for item in placed:
        source_r, source_c = item["source"]
        source_color = int(grid[source_r, source_c])
        target = item["target"]
        if target is None:
            for rr, cc in item["oranges"]:
                if int(out[rr, cc]) != ORANGE:
                    return False
            continue
        tr, tc = target
        if int(out[tr, tc]) != source_color:
            return False
        for rr, cc in item["oranges"]:
            if (rr, cc) == (tr, tc):
                continue
            if int(out[rr, cc]) != ORANGE:
                return False

    expected_changes = sum(1 for item in placed if item["target"] is not None)
    actual_changes = int(np.sum((grid == ORANGE) & (out != ORANGE)))
    return actual_changes == expected_changes


def _gen_attempt() -> np.ndarray | None:
    kinds = ["single", "two_right", "three_left"]
    if random.random() < 0.5:
        kinds.append("edge")
    if random.random() < 0.35:
        kinds.append(random.choice(("single", "two_right", "three_left")))
    random.shuffle(kinds)

    if len(kinds) > len(SOURCE_COLORS):
        return None

    colors = random.sample(SOURCE_COLORS, len(kinds))
    components = [_build_component(kind, color) for kind, color in zip(kinds, colors)]

    max_h = max(int(comp["grid"].shape[0]) for comp in components)
    max_w = max(int(comp["grid"].shape[1]) for comp in components)
    min_h = max(15, max_h + 2)
    min_w = max(15, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    canvas = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    ordered = sorted(
        components,
        key=lambda comp: (
            0 if comp["edge_side"] is not None else 1,
            -(int(comp["grid"].shape[0]) * int(comp["grid"].shape[1])),
        ),
    )

    placed_meta: list[dict] = []
    for component in ordered:
        placed = _place_component(canvas, occupied, component)
        if placed is None:
            return None
        placed_meta.append(placed)

    if not _validate_candidate(canvas, placed_meta):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing input")
