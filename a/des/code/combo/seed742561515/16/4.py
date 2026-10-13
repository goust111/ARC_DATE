from __future__ import annotations

import random

import numpy as np


BG = 8
DIVIDER = 5
CENTER = 0
ARROW = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)
DIR_UP_LEFT = (-1, -1)
DIR_UP_RIGHT = (-1, 1)
DIR_DOWN_LEFT = (1, -1)
DIR_DOWN_RIGHT = (1, 1)

ALL_DIRECTIONS = (
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP_LEFT,
    DIR_UP_RIGHT,
    DIR_DOWN_LEFT,
    DIR_DOWN_RIGHT,
)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIRECTION_BAG: list[tuple[int, int]] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    assert rows.size > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _apply_transform(arr: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    if direction == DIR_UP:
        return np.flipud(arr)
    if direction == DIR_DOWN:
        return arr.copy()
    if direction == DIR_LEFT:
        return np.fliplr(arr)
    if direction == DIR_RIGHT:
        return np.rot90(arr, 2)
    if direction == DIR_UP_LEFT:
        return np.rot90(arr, 1)
    if direction == DIR_UP_RIGHT:
        return np.rot90(arr, -1)
    if direction == DIR_DOWN_LEFT:
        return arr.T.copy()
    if direction == DIR_DOWN_RIGHT:
        return np.rot90(arr.T, 2)
    raise AssertionError(f"unknown direction: {direction}")


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = set()
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen.add(start)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return
    rr = top + rows
    cc = left + cols
    assert rr.min() >= 0 and cc.min() >= 0
    assert rr.max() < canvas.shape[0] and cc.max() < canvas.shape[1]
    assert np.all(canvas[rr, cc] == BG)
    canvas[rr, cc] = int(color)


def _decode_instruction_blocks(lower: np.ndarray) -> dict[int, tuple[int, int]]:
    centers = np.argwhere(lower == CENTER)
    assert centers.size > 0
    mapping: dict[int, tuple[int, int]] = {}
    for rr, cc in centers:
        r = int(rr)
        c = int(cc)
        assert 0 < r < lower.shape[0] - 1 and 0 < c < lower.shape[1] - 1
        patch = lower[r - 1 : r + 2, c - 1 : c + 2]
        assert patch.shape == (3, 3)
        assert int(patch[1, 1]) == CENTER

        arrow_offsets = []
        base_colors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                value = int(patch[dr + 1, dc + 1])
                if value == ARROW:
                    arrow_offsets.append((dr, dc))
                else:
                    base_colors.append(value)

        assert len(arrow_offsets) == 1
        assert len(base_colors) == 7
        assert len(set(base_colors)) == 1
        color = base_colors[0]
        assert color not in {BG, DIVIDER, CENTER, ARROW}
        assert color not in mapping
        mapping[color] = arrow_offsets[0]
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the transformation encoded by each lower instruction block to the same-colored object above the divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7

    divider_rows = [r for r in range(h) if bool(np.all(g[r, :] == DIVIDER))]
    assert len(divider_rows) == 1
    divider_row = int(divider_rows[0])
    assert 0 < divider_row < h - 1

    upper = g[:divider_row, :]
    lower = g[divider_row + 1 :, :]
    instruction_by_color = _decode_instruction_blocks(lower)

    object_colors = [int(c) for c in np.unique(upper) if int(c) != BG]
    assert object_colors
    assert set(object_colors) == set(instruction_by_color)

    out = np.full_like(upper, BG)
    for color in object_colors:
        mask = upper == color
        assert _is_single_component(mask)
        r0, c0, r1, c1 = _bbox(mask)
        sprite = mask[r0 : r1 + 1, c0 : c1 + 1]
        transformed = _apply_transform(sprite, instruction_by_color[color])
        _paint_mask(out, transformed, r0, c0, color)

    return out


def _take_directions(count: int) -> list[tuple[int, int]]:
    global _DIRECTION_BAG
    out: list[tuple[int, int]] = []
    while len(out) < count:
        if not _DIRECTION_BAG:
            _DIRECTION_BAG = list(ALL_DIRECTIONS)
            random.shuffle(_DIRECTION_BAG)
        out.append(_DIRECTION_BAG.pop())
    return out


def _grow_connected_shape(height: int, width: int, direction: tuple[int, int]) -> np.ndarray | None:
    area = height * width
    lo = max(4, (area + 2) // 3)
    hi = area - 1 if area > 4 else area
    for _ in range(250):
        target = random.randint(lo, hi)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        frontier = set()
        for dr, dc in NEIGHBORS4:
            rr = seed[0] + dr
            cc = seed[1] + dc
            if 0 <= rr < height and 0 <= cc < width:
                frontier.add((rr, cc))

        while frontier and len(cells) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        r0, c0, r1, c1 = _bbox(mask)
        if (r0, c0, r1, c1) != (0, 0, height - 1, width - 1):
            continue
        if not _is_single_component(mask):
            continue
        transformed = _apply_transform(mask, direction)
        if direction != DIR_DOWN and mask.shape == transformed.shape and np.array_equal(mask, transformed):
            continue
        return mask
    return None


def _split_rows(count: int) -> list[list[int]]:
    order = list(range(count))
    random.shuffle(order)
    if count == 1:
        return [order]
    if count >= 4:
        row_count = 2
    else:
        row_count = random.choice((1, 2))
    if row_count == 1:
        return [order]
    first_len = random.randint(max(1, count // 2), min(count - 1, (count + 1) // 2))
    return [order[:first_len], order[first_len:]]


def _layout_rectangles(rectangles: list[tuple[int, int]], gap_range: tuple[int, int], row_gap_range: tuple[int, int]) -> tuple[int, int, list[tuple[int, int]]]:
    rows = _split_rows(len(rectangles))
    row_specs = []
    inner_width = 0
    inner_height = 0
    row_gap_values = [random.randint(*row_gap_range) for _ in range(max(0, len(rows) - 1))]

    for row in rows:
        dims = [rectangles[i] for i in row]
        row_height = max(h for h, _ in dims)
        gaps = [random.randint(*gap_range) for _ in range(max(0, len(row) - 1))]
        x = 0
        starts = []
        for idx, (height, width) in zip(row, dims):
            starts.append((idx, x, height, width))
            x += width
            if len(starts) <= len(gaps):
                x += gaps[len(starts) - 1]
        row_width = x
        row_specs.append((row_height, row_width, starts))
        inner_width = max(inner_width, row_width)

    for i, (row_height, _row_width, _starts) in enumerate(row_specs):
        inner_height += row_height
        if i < len(row_gap_values):
            inner_height += row_gap_values[i]

    positions = [(-1, -1)] * len(rectangles)
    y = 0
    for i, (row_height, row_width, starts) in enumerate(row_specs):
        x_offset = random.randint(0, inner_width - row_width)
        for idx, x0, height, width in starts:
            y_offset = random.randint(0, row_height - height)
            positions[idx] = (y + y_offset, x_offset + x0)
        y += row_height
        if i < len(row_gap_values):
            y += row_gap_values[i]

    assert all(r >= 0 and c >= 0 for r, c in positions)
    return inner_height, inner_width, positions


def _make_instruction_patch(color: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    patch[1, 1] = CENTER
    patch[1 + direction[0], 1 + direction[1]] = ARROW
    return patch


def _build_expected_top(objects: list[dict], height: int, width: int) -> np.ndarray:
    top = np.full((height, width), BG, dtype=int)
    for obj in objects:
        transformed = _apply_transform(obj["mask"], obj["direction"])
        _paint_mask(top, transformed, obj["r0"], obj["c0"], obj["color"])
    return top


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, len(OBJECT_COLORS))
    colors = random.sample(OBJECT_COLORS, count)
    directions = _take_directions(count)

    objects: list[dict] = []
    for color, direction in zip(colors, directions):
        shape = None
        for _ in range(80):
            height = random.randint(2, 5)
            width = random.randint(2, 5)
            if direction in {DIR_UP_LEFT, DIR_UP_RIGHT, DIR_DOWN_LEFT, DIR_DOWN_RIGHT} and random.random() < 0.65:
                while width == height:
                    width = random.randint(2, 5)
            shape = _grow_connected_shape(height, width, direction)
            if shape is not None:
                break
        if shape is None:
            return None
        h, w = shape.shape
        safe = max(h, w)
        objects.append(
            {
                "color": color,
                "direction": direction,
                "mask": shape,
                "safe": safe,
                "box": (safe, safe),
            }
        )

    object_boxes = [obj["box"] for obj in objects]
    upper_inner_h, upper_inner_w, upper_positions = _layout_rectangles(object_boxes, gap_range=(2, 4), row_gap_range=(2, 4))

    block_order = list(range(count))
    random.shuffle(block_order)
    block_boxes = [(3, 3)] * count
    lower_inner_h, lower_inner_w, lower_positions = _layout_rectangles(block_boxes, gap_range=(1, 3), row_gap_range=(1, 3))

    outer_margin = random.randint(1, 2)
    upper_top_margin = random.randint(1, 2)
    upper_bottom_margin = random.randint(1, 2)
    lower_top_margin = random.randint(1, 2)
    lower_bottom_margin = random.randint(1, 2)

    inner_width = max(upper_inner_w, lower_inner_w)
    width = inner_width + 2 * outer_margin
    upper_height = upper_inner_h + upper_top_margin + upper_bottom_margin
    lower_height = lower_inner_h + lower_top_margin + lower_bottom_margin
    height = upper_height + 1 + lower_height

    grid = np.full((height, width), BG, dtype=int)
    grid[upper_height, :] = DIVIDER

    upper_x_base = outer_margin + random.randint(0, inner_width - upper_inner_w)
    lower_x_base = outer_margin + random.randint(0, inner_width - lower_inner_w)

    for obj, (local_r, local_c) in zip(objects, upper_positions):
        obj["r0"] = upper_top_margin + local_r
        obj["c0"] = upper_x_base + local_c
        _paint_mask(grid, obj["mask"], obj["r0"], obj["c0"], obj["color"])

    for slot_idx, obj_idx in enumerate(block_order):
        local_r, local_c = lower_positions[slot_idx]
        r0 = upper_height + 1 + lower_top_margin + local_r
        c0 = lower_x_base + local_c
        patch = _make_instruction_patch(objects[obj_idx]["color"], objects[obj_idx]["direction"])
        assert np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == BG)
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch

    expected = _build_expected_top(objects, upper_height, width)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
