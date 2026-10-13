from __future__ import annotations

import random

import numpy as np


BG = 0
LEFT_WALL = 1
RIGHT_WALL = 2
PATH = 5

GRID_HEIGHT_RANGE = (13, 17)
GRID_WIDTH_RANGE = (13, 17)
COUNT_RANGE = (3, 4)
GEN_RETRIES = 400
OFFSET_RETRIES = 200

MASK_LIBRARY = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
)

HORIZONTAL_MASKS = tuple(mask for mask in MASK_LIBRARY if mask.shape[0] == 1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _component_order(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _find_objects(grid: np.ndarray, color: int, wall_col: int) -> list[np.ndarray]:
    movable = (grid == int(color)).copy()
    movable[:, wall_col] = False
    return _components_mask(movable)


def _slide_objects(grid: np.ndarray, color: int, wall_col: int, direction: int) -> list[dict]:
    h, w = grid.shape
    objects = _find_objects(grid, color, wall_col)
    objects.sort(key=lambda coords: _component_order(coords, direction))

    settled = np.zeros((h, w), dtype=bool)
    settled[:, wall_col] = True
    moved: list[dict] = []

    for coords in objects:
        shift = 0
        while True:
            candidate = coords + np.array([0, shift + direction])
            if int(candidate[:, 1].min()) < 0 or int(candidate[:, 1].max()) >= w:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += direction

        landed = coords + np.array([0, shift])
        assert not np.any(settled[landed[:, 0], landed[:, 1]])
        settled[landed[:, 0], landed[:, 1]] = True
        moved.append(
            {
                "color": int(color),
                "original": coords.copy(),
                "landed": landed,
                "distance": abs(int(shift)),
                "shift": int(shift),
            }
        )
    return moved


def _draw_paths(out: np.ndarray, moved: list[dict], bg: int) -> np.ndarray:
    fixed = out != int(bg)
    for item in sorted(
        moved,
        key=lambda obj: (
            -int(obj["distance"]),
            int(obj["color"]),
            int(obj["original"][:, 0].min()),
            int(obj["original"][:, 1].min()),
        ),
    ):
        shift = int(item["shift"])
        for r_raw, c_raw in item["original"]:
            r = int(r_raw)
            c0 = int(c_raw)
            c1 = c0 + shift
            lo, hi = sorted((c0, c1))
            for c in range(lo, hi + 1):
                if not fixed[r, c]:
                    out[r, c] = PATH
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each wall-colored object into its matching side wall, then mark the swept horizontal path in gray.

    Input structure
    - The background is black.
    - The full left column is a blue wall and the full right column is a red wall.
    - Away from the walls, there are several small 4-connected blue objects and red objects of the same colors as the two walls.

    Rule
    - Treat every non-wall blue component as a rigid object that slides left, and every non-wall red component as a rigid object that slides right.
    - For each wall color, process objects from nearest to farthest from that wall; each object stops just before the wall or a previously settled object of the same color.
    - Draw the two wall columns unchanged, place every object at its landed position, then paint gray on every vacated cell swept between the object's original and landed positions.
    - Gray path cells never overwrite the fixed walls or any landed object pixels.

    Key insight
    - The output combines two linked steps: the final landing point comes from same-color wall-compaction, and that landing shift determines the exact horizontal sweep that becomes the gray trajectory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3

    bg = _get_bg(g)
    h, w = g.shape
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])

    assert left_color != right_color
    assert np.all(g[:, 0] == left_color)
    assert np.all(g[:, w - 1] == right_color)

    moved = []
    moved.extend(_slide_objects(g, left_color, 0, -1))
    moved.extend(_slide_objects(g, right_color, w - 1, 1))

    out = np.full((h, w), int(bg), dtype=int)
    out[:, 0] = left_color
    out[:, w - 1] = right_color

    occupied = out != int(bg)
    for item in moved:
        landed = item["landed"]
        assert not np.any(occupied[landed[:, 0], landed[:, 1]])
        out[landed[:, 0], landed[:, 1]] = int(item["color"])
        occupied[landed[:, 0], landed[:, 1]] = True

    return _draw_paths(out, moved, bg)


def _choose_mask(force_horizontal: bool = False) -> np.ndarray:
    source = HORIZONTAL_MASKS if force_horizontal else MASK_LIBRARY
    return random.choice(source).copy()


def _can_place_row(start: int, height: int, intervals: list[tuple[int, int]]) -> bool:
    end = start + height - 1
    return all(end + 1 < r0 or r1 + 1 < start for r0, r1 in intervals)


def _row_layout(height: int, masks: list[np.ndarray], forced_row: int) -> list[int] | None:
    starts: list[int | None] = [None] * len(masks)
    intervals: list[tuple[int, int]] = []

    first_h = masks[0].shape[0]
    if forced_row + first_h > height:
        return None
    starts[0] = forced_row
    intervals.append((forced_row, forced_row + first_h - 1))

    order = list(range(1, len(masks)))
    random.shuffle(order)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        mh = masks[idx].shape[0]
        candidates = list(range(height - mh + 1))
        random.shuffle(candidates)
        for start in candidates:
            if not _can_place_row(start, mh, intervals):
                continue
            starts[idx] = start
            intervals.append((start, start + mh - 1))
            if backtrack(i + 1):
                return True
            intervals.pop()
            starts[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(x) for x in starts]


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)])


def _render_objects(shape: tuple[int, int], specs: list[dict], *, final: bool) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[:, 0] = LEFT_WALL
    grid[:, shape[1] - 1] = RIGHT_WALL
    for spec in specs:
        coords = spec["final_coords"] if final else spec["input_coords"]
        grid[coords[:, 0], coords[:, 1]] = int(spec["color"])
    return grid


def _path_cells(spec: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    shift = int(spec["final_coords"][0, 1] - spec["input_coords"][0, 1])
    for r_raw, c_raw in spec["input_coords"]:
        r = int(r_raw)
        c0 = int(c_raw)
        c1 = c0 + shift
        lo, hi = sorted((c0, c1))
        for c in range(lo, hi + 1):
            cells.add((r, c))
    return cells


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = _render_objects(shape, specs, final=True)
    fixed = out != BG
    for spec in sorted(
        specs,
        key=lambda item: (
            -int(item["distance"]),
            int(item["color"]),
            int(item["input_coords"][:, 0].min()),
            int(item["input_coords"][:, 1].min()),
        ),
    ):
        for r, c in sorted(_path_cells(spec)):
            if not fixed[r, c]:
                out[r, c] = PATH
    return out


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _has_required_overlap(specs: list[dict]) -> bool:
    by_color = {
        LEFT_WALL: [spec for spec in specs if spec["color"] == LEFT_WALL],
        RIGHT_WALL: [spec for spec in specs if spec["color"] == RIGHT_WALL],
    }
    for left in by_color[LEFT_WALL]:
        left_cells = _path_cells(left)
        for right in by_color[RIGHT_WALL]:
            if int(left["distance"]) == int(right["distance"]):
                continue
            if left_cells & _path_cells(right):
                return True
    return False


def _rows_disjoint(specs: list[dict]) -> bool:
    seen: set[int] = set()
    for spec in specs:
        rows = set(map(int, spec["input_coords"][:, 0]))
        if seen & rows:
            return False
        seen |= rows
    return True


def _validate_scene(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    left_specs = [spec for spec in specs if spec["color"] == LEFT_WALL]
    right_specs = [spec for spec in specs if spec["color"] == RIGHT_WALL]
    if not (COUNT_RANGE[0] <= len(left_specs) <= COUNT_RANGE[1]):
        return False
    if not (COUNT_RANGE[0] <= len(right_specs) <= COUNT_RANGE[1]):
        return False
    if not _rows_disjoint(left_specs) or not _rows_disjoint(right_specs):
        return False
    if not _has_required_overlap(specs):
        return False

    distances = {int(spec["distance"]) for spec in specs}
    if len(distances) < 3:
        return False

    blue_mask = (inp == LEFT_WALL).copy()
    blue_mask[:, 0] = False
    red_mask = (inp == RIGHT_WALL).copy()
    red_mask[:, -1] = False
    blue_components = _components_mask(blue_mask)
    red_components = _components_mask(red_mask)
    if len(blue_components) != len(left_specs) or len(red_components) != len(right_specs):
        return False
    if any(not (2 <= len(comp) <= 4) for comp in blue_components + red_components):
        return False
    if not np.all(inp[:, 0] == LEFT_WALL) or not np.all(inp[:, -1] == RIGHT_WALL):
        return False
    if set(map(int, np.unique(inp))) != {BG, LEFT_WALL, RIGHT_WALL}:
        return False
    if not set(map(int, np.unique(expected))).issubset({BG, LEFT_WALL, RIGHT_WALL, PATH}):
        return False
    return True


def _build_specs(height: int, width: int) -> list[dict] | None:
    shared_row = random.randrange(1, height - 1)
    blue_count = random.randint(*COUNT_RANGE)
    red_count = random.randint(*COUNT_RANGE)

    blue_masks = [_choose_mask(force_horizontal=True)] + [_choose_mask() for _ in range(blue_count - 1)]
    red_masks = [_choose_mask(force_horizontal=True)] + [_choose_mask() for _ in range(red_count - 1)]

    blue_rows = _row_layout(height, blue_masks, shared_row)
    red_rows = _row_layout(height, red_masks, shared_row)
    if blue_rows is None or red_rows is None:
        return None

    specs: list[dict] = []
    for color, masks, rows in ((LEFT_WALL, blue_masks, blue_rows), (RIGHT_WALL, red_masks, red_rows)):
        for idx, (mask, row) in enumerate(zip(masks, rows, strict=True)):
            _, mw = mask.shape
            final_c0 = 1 if color == LEFT_WALL else width - 1 - mw
            specs.append(
                {
                    "id": f"{color}-{idx}",
                    "color": color,
                    "mask": mask.copy(),
                    "row": int(row),
                    "final_c0": int(final_c0),
                    "distance": None,
                    "input_coords": None,
                    "final_coords": _coords_from_mask(mask, row, final_c0),
                }
            )
    return specs


def _assign_offsets(height: int, width: int, specs: list[dict]) -> bool:
    left0 = next(spec for spec in specs if spec["id"] == "1-0")
    right0 = next(spec for spec in specs if spec["id"] == "2-0")
    left_w = left0["mask"].shape[1]
    right_w = right0["mask"].shape[1]
    max_left = width - left_w - 2
    max_right = width - right_w - 2
    pair_choices = []
    for left_offset in range(1, max_left + 1):
        for right_offset in range(1, max_right + 1):
            left_l = 1 + left_offset
            left_r = left_l + left_w - 1
            right_l = width - 1 - right_w - right_offset
            right_r = right_l + right_w - 1
            if left_l <= right_r:
                continue
            pair_choices.append((left_offset, right_offset))
    if not pair_choices:
        return False

    for _ in range(OFFSET_RETRIES):
        occupied = np.zeros((height, width), dtype=bool)
        occupied[:, 0] = True
        occupied[:, width - 1] = True

        chosen_pair = random.choice(pair_choices)
        for spec in specs:
            _, mw = spec["mask"].shape
            max_offset = width - mw - 2
            if spec["id"] == "1-0":
                offset = chosen_pair[0]
                input_c0 = 1 + offset
            elif spec["id"] == "2-0":
                offset = chosen_pair[1]
                input_c0 = spec["final_c0"] - offset
            elif spec["color"] == LEFT_WALL:
                offset = random.randint(1, max_offset)
                input_c0 = 1 + offset
            else:
                offset = random.randint(1, max_offset)
                input_c0 = spec["final_c0"] - offset

            coords = _coords_from_mask(spec["mask"], spec["row"], input_c0)
            if np.any(occupied[coords[:, 0], coords[:, 1]]):
                break
            spec["distance"] = int(offset)
            spec["input_coords"] = coords
            occupied[coords[:, 0], coords[:, 1]] = True
        else:
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        specs = _build_specs(height, width)
        if specs is None:
            continue
        if not _assign_offsets(height, width, specs):
            continue

        inp = _render_objects((height, width), specs, final=False)
        expected = _expected_output((height, width), specs)
        if not _validate_scene(inp, expected, specs):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
