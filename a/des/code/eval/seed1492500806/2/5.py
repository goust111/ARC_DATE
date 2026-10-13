from __future__ import annotations

import random

import numpy as np


MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_BY_NAME = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perp_dirs(vec: tuple[int, int]) -> list[tuple[int, int]]:
    return [(-1, 0), (1, 0)] if vec[0] == 0 else [(0, -1), (0, 1)]


def _component_same_color(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    banned: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    stack = [start]
    seen = set()
    shape = grid.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or (r, c) in banned or int(grid[r, c]) != int(color):
            continue
        seen.add((r, c))
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                stack.append((nr, nc))
    return seen


def _has_perpendicular_neighbor(
    grid: np.ndarray,
    pos: tuple[int, int],
    color: int,
    vec_toward_stamp: tuple[int, int],
) -> bool:
    r, c = pos
    for dr, dc in _perp_dirs(vec_toward_stamp):
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(color):
            return True
    return False


def _stamp_span(stamp: set[tuple[int, int]], guide_vec: tuple[int, int]) -> int:
    if guide_vec[0] != 0:
        rows = [r for r, _c in stamp]
        return max(rows) - min(rows) + 1
    cols = [c for _r, c in stamp]
    return max(cols) - min(cols) + 1


def _extract_object(
    grid: np.ndarray,
    marker_pos: tuple[int, int],
    bg: int,
) -> dict:
    mr, mc = marker_pos
    neighbors = []
    for dr, dc in DIR4:
        nr, nc = mr + dr, mc + dc
        if not _in_bounds(grid.shape, nr, nc):
            continue
        value = int(grid[nr, nc])
        if value != int(bg) and value != MAROON:
            neighbors.append((nr, nc))

    assert len(neighbors) == 1, "each maroon endpoint must touch exactly one guide cell"
    line_start = neighbors[0]
    color = int(grid[line_start])
    toward_stamp = (line_start[0] - mr, line_start[1] - mc)
    guide_vec = (-toward_stamp[0], -toward_stamp[1])

    line_cells: list[tuple[int, int]] = []
    current = line_start
    for _ in range(grid.size):
        if _has_perpendicular_neighbor(grid, current, color, toward_stamp):
            attach = current
            break
        line_cells.append(current)
        nr, nc = current[0] + toward_stamp[0], current[1] + toward_stamp[1]
        assert _in_bounds(grid.shape, nr, nc), "guide line must lead into a stamp"
        assert int(grid[nr, nc]) == int(color), "guide line must remain monochrome"
        current = (nr, nc)
    else:
        raise AssertionError("failed to locate stamp attachment")

    stamp = _component_same_color(grid, attach, color, set(line_cells))
    assert attach in stamp and stamp, "stamp component must remain after removing the guide"

    span = _stamp_span(stamp, guide_vec)
    distance = abs(mr - attach[0]) + abs(mc - attach[1])
    assert span >= 2, "stamp span must be at least two cells"
    assert distance % span == 0, "endpoint distance must be a multiple of the stamp span"
    repeats = distance // span
    assert repeats >= 1, "each guide must request at least one extra stamping step"

    return {
        "color": color,
        "stamp": stamp,
        "guide_vec": guide_vec,
        "step": span,
        "repeats": repeats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each small monochrome stamp along its same-color guide line.

    The maroon endpoint identifies how far the copies must continue; the
    translation step is the stamp's own span in the guide direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    markers = [tuple(map(int, pos)) for pos in np.argwhere(g == MAROON)]
    assert markers, "expected at least one maroon endpoint"

    out = g.copy()
    out[out == MAROON] = int(bg)

    for marker in markers:
        obj = _extract_object(g, marker, bg)
        color = int(obj["color"])
        dr, dc = obj["guide_vec"]
        step = int(obj["step"])
        repeats = int(obj["repeats"])
        stamp = obj["stamp"]

        for k in range(1, repeats + 1):
            shift_r = dr * step * k
            shift_c = dc * step * k
            for r, c in stamp:
                rr = r + shift_r
                cc = c + shift_c
                assert _in_bounds(out.shape, rr, cc), "stamped copy must fit inside the grid"
                out[rr, cc] = color

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_DIRS = ["right", "left", "down", "up"]

GEN_OBJECTS_RANGE = (1, 4)
GEN_SIDE_RANGE_BY_COUNT = {
    1: (12, 18),
    2: (15, 22),
    3: (18, 25),
    4: (21, 28),
}
GEN_SHAPE_ATTEMPTS = 200
GEN_SCENE_ATTEMPTS = 400
GEN_PADDING = 1


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _is_connected(mask: np.ndarray) -> bool:
    points = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    if not points:
        return False
    stack = [points[0]]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or not bool(mask[r, c]):
            continue
        seen.add((r, c))
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                stack.append((nr, nc))
    return len(seen) == len(points)


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _seed_stamp(direction: str, h: int, w: int) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    if direction in ("right", "left"):
        attach_r = random.randrange(h)
        side_c = w - 1 if direction == "right" else 0
        inward_c = w - 2 if direction == "right" else 1
        branch_r = attach_r + (1 if attach_r == 0 else -1 if attach_r == h - 1 else random.choice([-1, 1]))
        attach = (attach_r, side_c)
        required = {attach, (branch_r, side_c), (attach_r, inward_c)}
        return attach, required

    attach_c = random.randrange(w)
    side_r = h - 1 if direction == "down" else 0
    inward_r = h - 2 if direction == "down" else 1
    branch_c = attach_c + (1 if attach_c == 0 else -1 if attach_c == w - 1 else random.choice([-1, 1]))
    attach = (side_r, attach_c)
    required = {attach, (side_r, branch_c), (inward_r, attach_c)}
    return attach, required


def _make_stamp(direction: str) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = random.choice((3, 4))
        w = random.choice((3, 4))
        attach, cells = _seed_stamp(direction, h, w)
        area_lo = max(len(cells) + 1, 5)
        area_hi = min(h * w - 1, 10)
        if area_lo > area_hi:
            continue
        target_area = random.randint(area_lo, area_hi)

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _is_connected(mask):
            continue
        if not _touches_all_bbox_edges(mask):
            continue
        if direction in ("right", "left"):
            if not any(bool(mask[r, attach[1]]) for r in range(h) if r != attach[0]):
                continue
        else:
            if not any(bool(mask[attach[0], c]) for c in range(w) if c != attach[1]):
                continue
        return mask, attach

    raise RuntimeError("failed to build a valid stamp")


def _stamp_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pos)) for pos in np.argwhere(mask)]


def _step_from_mask(mask: np.ndarray, direction: str) -> int:
    return int(mask.shape[1] if direction in ("right", "left") else mask.shape[0])


def _shift_cells(
    cells: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _combined_footprint(spec: dict) -> set[tuple[int, int]]:
    direction = spec["direction"]
    step = spec["step"]
    repeats = spec["repeats"]
    dr, dc = DIR_BY_NAME[direction]
    stamp = spec["stamp_abs"]
    attach_r, attach_c = spec["attach_abs"]

    coords = set(stamp)
    for t in range(1, repeats * step):
        coords.add((attach_r + dr * t, attach_c + dc * t))
    coords.add((attach_r + dr * repeats * step, attach_c + dc * repeats * step))
    for k in range(1, repeats + 1):
        coords.update(_shift_cells(stamp, dr * step * k, dc * step * k))
    return coords


def _can_place(
    occupied: np.ndarray,
    coords: set[tuple[int, int]],
    padding: int,
) -> bool:
    H, W = occupied.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        r0 = max(0, r - padding)
        r1 = min(H, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(W, c + padding + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, coords: set[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _top_left_ranges(H: int, W: int, mask: np.ndarray, direction: str, repeats: int) -> tuple[range, range]:
    h, w = mask.shape
    step = _step_from_mask(mask, direction)
    r_min, r_max = 0, H - h
    c_min, c_max = 0, W - w
    dr, dc = DIR_BY_NAME[direction]
    if dr == 1:
        r_max -= repeats * step
    elif dr == -1:
        r_min += repeats * step
    if dc == 1:
        c_max -= repeats * step
    elif dc == -1:
        c_min += repeats * step
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _place_objects(H: int, W: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((H, W), dtype=bool)
    placed: list[dict] = []

    specs = sorted(
        specs,
        key=lambda spec: (
            spec["repeats"] * spec["step"],
            spec["mask"].shape[0] * spec["mask"].shape[1],
        ),
        reverse=True,
    )

    for base in specs:
        rows, cols = _top_left_ranges(H, W, base["mask"], base["direction"], base["repeats"])
        positions = [(r, c) for r in rows for c in cols]
        if not positions:
            return None
        random.shuffle(positions)

        stamp_rel = _stamp_cells(base["mask"])
        attach_rel = base["attach_rel"]
        found = None
        for r0, c0 in positions:
            stamp_abs = [(r0 + r, c0 + c) for r, c in stamp_rel]
            attach_abs = (r0 + attach_rel[0], c0 + attach_rel[1])
            spec = {
                **base,
                "top_left": (r0, c0),
                "stamp_abs": stamp_abs,
                "attach_abs": attach_abs,
            }
            footprint = _combined_footprint(spec)
            if _can_place(occupied, footprint, GEN_PADDING):
                _mark_occupied(occupied, footprint)
                spec["footprint"] = footprint
                found = spec
                break

        if found is None:
            return None
        placed.append(found)

    return placed


def _render_input(H: int, W: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((H, W), GEN_BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        _paint_cells(grid, spec["stamp_abs"], color)
        dr, dc = DIR_BY_NAME[spec["direction"]]
        attach_r, attach_c = spec["attach_abs"]
        for t in range(1, spec["repeats"] * spec["step"]):
            grid[attach_r + dr * t, attach_c + dc * t] = color
        marker_r = attach_r + dr * spec["repeats"] * spec["step"]
        marker_c = attach_c + dc * spec["repeats"] * spec["step"]
        grid[marker_r, marker_c] = MAROON
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    out[out == MAROON] = GEN_BG
    for spec in specs:
        color = int(spec["color"])
        dr, dc = DIR_BY_NAME[spec["direction"]]
        for k in range(1, spec["repeats"] + 1):
            shifted = _shift_cells(spec["stamp_abs"], dr * spec["step"] * k, dc * spec["step"] * k)
            _paint_cells(out, shifted, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*GEN_OBJECTS_RANGE)
    side_lo, side_hi = GEN_SIDE_RANGE_BY_COUNT[n_objects]
    H = random.randint(side_lo, side_hi)
    W = random.randint(side_lo, side_hi)

    directions = random.sample(GEN_DIRS, n_objects)
    colors = random.sample(GEN_COLORS, n_objects)
    specs = []

    for direction, color in zip(directions, colors, strict=False):
        mask, attach_rel = _make_stamp(direction)
        specs.append(
            {
                "direction": direction,
                "color": color,
                "mask": mask,
                "attach_rel": attach_rel,
                "step": _step_from_mask(mask, direction),
                "repeats": random.randint(1, 3),
            }
        )

    placed = _place_objects(H, W, specs)
    if placed is None:
        return None

    inp = _render_input(H, W, placed)
    expected = _render_expected(inp, placed)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
