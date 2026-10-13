from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_complete_square(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c or side_r < 2:
        return False
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == BLUE)
        and np.all(grid[r1, c0 : c1 + 1] == BLUE)
        and np.all(grid[r0 : r1 + 1, c0] == BLUE)
        and np.all(grid[r0 : r1 + 1, c1] == BLUE)
    )


def _corner_marker_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components_of_color(grid, BLUE):
        box = _bbox(coords)
        objects.append(
            {
                "coords": coords,
                "bbox": box,
                "complete": _is_complete_square(grid, coords),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect complete blue squares, mark their outside corners, and transfer only their aligned top edge-strip columns to the bottom edge.

    Input structure:
    - The grid has black background, a 2-row grey strip along the top edge, and several
      separated blue 4-connected objects in the interior.
    - Some blue objects are complete squares: their bounding box is square and every cell
      on that bbox border is blue; others are distractor shapes.
    - Red pixels may appear inside the top strip, including above complete-square columns
      and above non-square columns.

    Rule:
    - Find every blue 4-connected object and keep only the ones that are complete squares.
    - For each complete square, paint green on the two orthogonally outside cells of each
      corner whenever those cells are in bounds.
    - Activate every column in each complete square's bbox column span; copy the 2-row top
      strip in those columns to the bottom 2 rows, then erase those top-strip columns.
    - Finally erase every red pixel everywhere, whether it stayed on top or was copied.

    Key insight:
    - The same complete-square test gates two different effects at once: local green
      corner marking and global edge-strip transfer. Red markers alone are not enough,
      because non-square objects can share marked strip columns as distractors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    active_cols = np.zeros(w, dtype=bool)
    complete_boxes: list[tuple[int, int, int, int]] = []
    for obj in _analyze_objects(g):
        if not obj["complete"]:
            continue
        box = obj["bbox"]
        complete_boxes.append(box)
        active_cols[box[1] : box[3] + 1] = True

    for box in complete_boxes:
        for r, c in _corner_marker_coords(box):
            if _in_bounds(g.shape, r, c):
                out[r, c] = GREEN

    if h >= 2:
        out[-2:, active_cols] = g[:2, active_cols]
        out[:2, active_cols] = BG

    out[out == RED] = BG
    return out


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.argwhere(mask)}


def _shape_size(coords: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _rotate_coords(coords: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    mask_h, mask_w = _shape_size(coords)
    mask = np.zeros((mask_h, mask_w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return _coords_from_mask(np.rot90(mask, turns))


def _filled_square(side: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(side) for c in range(side)}


def _frame_square(side: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(side)
        for c in range(side)
        if r in (0, side - 1) or c in (0, side - 1)
    }


def _broken_frame(side: int) -> set[tuple[int, int]]:
    coords = _frame_square(side)
    removable = (
        [(0, c) for c in range(1, side - 1)]
        + [(side - 1, c) for c in range(1, side - 1)]
        + [(r, 0) for r in range(1, side - 1)]
        + [(r, side - 1) for r in range(1, side - 1)]
    )
    coords.remove(random.choice(removable))
    return coords


def _make_complete_shape() -> dict:
    side = random.randint(2, 4)
    if side == 2 or random.random() < 0.5:
        coords = _filled_square(side)
        kind = "filled_square"
    else:
        coords = _frame_square(side)
        kind = "frame_square"
    return {"coords": coords, "kind": kind, "complete": True}


def _make_fake_square_shape() -> dict:
    if random.random() < 0.5:
        coords = {(0, 0), (1, 0), (1, 1)}
        coords = _rotate_coords(coords, random.randint(0, 3))
        kind = "l_shape"
    else:
        coords = _broken_frame(random.randint(3, 4))
        kind = "broken_frame"
    return {"coords": coords, "kind": kind, "complete": False, "fake_square": True}


def _make_other_nonsquare_shape() -> dict:
    choice = random.choice(("bar", "t_shape", "step", "rectangle", "broken_frame"))
    if choice == "bar":
        if random.random() < 0.5:
            length = random.randint(3, 5)
            coords = {(0, c) for c in range(length)}
        else:
            length = random.randint(3, 5)
            coords = {(r, 0) for r in range(length)}
    elif choice == "t_shape":
        coords = {(0, 0), (0, 1), (0, 2), (1, 1)}
        coords = _rotate_coords(coords, random.randint(0, 3))
    elif choice == "step":
        coords = {(0, 0), (0, 1), (1, 1), (1, 2)}
        coords = _rotate_coords(coords, random.randint(0, 3))
    elif choice == "rectangle":
        h = random.randint(2, 3)
        w = random.randint(3, 4)
        if h == w:
            w += 1
        coords = {(r, c) for r in range(h) for c in range(w)}
    else:
        coords = _broken_frame(random.randint(3, 4))
    return {"coords": coords, "kind": choice, "complete": False, "fake_square": False}


def _place_objects(grid: np.ndarray, specs: list[dict]) -> list[dict] | None:
    h, w = grid.shape
    randomized = specs[:]
    random.shuffle(randomized)

    widths = [_shape_size(spec["coords"])[1] for spec in randomized]
    required_width = sum(widths) + max(0, len(widths) - 1)
    usable_width = w - 2
    if required_width > usable_width:
        return None

    extras = usable_width - required_width
    gaps = [0] * (len(randomized) + 1)
    for idx in range(1, len(randomized)):
        gaps[idx] = 1
    for _ in range(extras):
        gaps[random.randint(0, len(gaps) - 1)] += 1

    cursor = 1 + gaps[0]
    placed: list[dict] = []
    for idx, spec in enumerate(randomized):
        coords = spec["coords"]
        shape_h, shape_w = _shape_size(coords)
        min_r0 = 3
        max_r0 = h - shape_h - 3
        if max_r0 < min_r0:
            return None
        r0 = random.randint(min_r0, max_r0)
        c0 = cursor
        shifted = np.array([(r0 + r, c0 + c) for r, c in coords], dtype=int)
        grid[shifted[:, 0], shifted[:, 1]] = BLUE
        box = _bbox(shifted)
        placed.append(
            {
                "bbox": box,
                "complete": bool(spec["complete"]),
                "kind": spec["kind"],
                "fake_square": bool(spec.get("fake_square", False)),
            }
        )
        cursor += shape_w + gaps[idx + 1]
    return placed


def _complete_marker_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cols = list(range(c0, c1 + 1))
    if len(cols) <= 2:
        marked_cols = [cols[(r0 + c0) % len(cols)]]
    elif (r0 + c1) % 2 == 0:
        marked_cols = [cols[0], cols[-1]]
    else:
        marked_cols = [cols[len(cols) // 2]]
    if len(marked_cols) == 1:
        return [((r0 + c0) % 2, marked_cols[0])]
    return [(0, marked_cols[0]), (1, marked_cols[1])]


def _distractor_marker_cells(placed: list[dict], active_cols: set[int]) -> list[tuple[int, int]]:
    preferred_cols: list[int] = []
    for obj in placed:
        if obj["fake_square"]:
            preferred_cols.extend(c for c in range(obj["bbox"][1], obj["bbox"][3] + 1) if c not in active_cols)
    other_cols: list[int] = []
    for obj in placed:
        if obj["complete"]:
            continue
        other_cols.extend(c for c in range(obj["bbox"][1], obj["bbox"][3] + 1) if c not in active_cols)

    cols = preferred_cols or other_cols
    if not cols:
        return []

    unique_cols = sorted(set(cols))
    cells = [(0, unique_cols[0])]
    if len(unique_cols) >= 3 and len(cells) < 2:
        cells.append((1, unique_cols[-1]))
    return cells


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    h, w = grid.shape
    assert h >= 8 and w >= 8
    assert set(np.unique(grid)).issubset({BG, BLUE, RED, GREY})
    assert np.all(np.isin(grid[:2, :], [GREY, RED]))
    assert np.all(grid[-2:, :] == BG)
    assert not np.any(grid[2:-2, :] == RED)

    objects = _analyze_objects(grid)
    assert 3 <= len(objects) <= 5
    complete = [obj for obj in objects if obj["complete"]]
    nonsquare = [obj for obj in objects if not obj["complete"]]
    assert complete and nonsquare
    assert any((obj["bbox"][2] - obj["bbox"][0]) == (obj["bbox"][3] - obj["bbox"][1]) for obj in nonsquare)

    active_cols = set()
    for obj in complete:
        _, c0, _, c1 = obj["bbox"]
        active_cols.update(range(c0, c1 + 1))
    assert active_cols and len(active_cols) < w

    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    red_cols = {c for _, c in red_cells}
    assert red_cols - active_cols
    assert active_cols - red_cols

    for obj in complete:
        _, c0, _, c1 = obj["bbox"]
        count = sum(c0 <= c <= c1 for _, c in red_cells)
        assert 1 <= count <= 2

    assert any(
        any(obj["bbox"][1] <= c <= obj["bbox"][3] for _, c in red_cells)
        for obj in nonsquare
    )

    out = main(grid)
    assert not np.any(out == RED)

    active_mask = np.zeros(w, dtype=bool)
    for c in active_cols:
        active_mask[c] = True
    inactive_mask = ~active_mask

    assert np.all(out[:2, active_mask] == BG)
    expected_bottom = np.where(grid[:2, active_mask] == RED, BG, grid[:2, active_mask])
    assert np.array_equal(out[-2:, active_mask], expected_bottom)
    expected_top_inactive = np.where(grid[:2, inactive_mask] == RED, BG, grid[:2, inactive_mask])
    assert np.array_equal(out[:2, inactive_mask], expected_top_inactive)

    for obj in complete:
        for r, c in _corner_marker_coords(obj["bbox"]):
            if _in_bounds(out.shape, r, c):
                assert int(out[r, c]) == GREEN


def _generate_attempt() -> np.ndarray | None:
    num_objects = random.randint(3, 5)
    complete_count = random.randint(1, min(3, num_objects - 1))
    nonsquare_count = num_objects - complete_count

    specs = [_make_complete_shape() for _ in range(complete_count)]
    specs.append(_make_fake_square_shape())
    for _ in range(nonsquare_count - 1):
        specs.append(_make_other_nonsquare_shape())

    widths = [_shape_size(spec["coords"])[1] for spec in specs]
    heights = [_shape_size(spec["coords"])[0] for spec in specs]

    min_w = max(14, sum(widths) + max(0, len(specs) - 1) + 2)
    max_w = min(26, min_w + 6)
    if min_w > max_w:
        return None
    min_h = max(12, max(heights) + 7)
    max_h = min(18, min_h + 4)
    if min_h > max_h:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.full((h, w), BG, dtype=int)
    grid[:2, :] = GREY

    placed = _place_objects(grid, specs)
    if placed is None:
        return None

    active_cols: set[int] = set()
    for obj in placed:
        if obj["complete"]:
            active_cols.update(range(obj["bbox"][1], obj["bbox"][3] + 1))
    if not active_cols or len(active_cols) >= w:
        return None

    red_cells: set[tuple[int, int]] = set()
    for obj in placed:
        if obj["complete"]:
            red_cells.update(_complete_marker_cells(obj["bbox"]))
    red_cells.update(_distractor_marker_cells(placed, active_cols))
    if not red_cells:
        return None

    for r, c in red_cells:
        grid[r, c] = RED

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-strip square-gating puzzle")
