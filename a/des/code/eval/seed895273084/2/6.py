from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

INSTRUCTION_COLORS = (1, 2, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 400


def _has_adjacent_color(grid: np.ndarray, row: int, col: int, color: int) -> bool:
    height, width = grid.shape
    for dr, dc in NEIGHBORS4:
        rr = row + dr
        cc = col + dc
        if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) == int(color):
            return True
    return False


def _separator_col(grid: np.ndarray) -> int:
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    assert len(cols) == 1, "expected exactly one maroon separator column"
    return cols[0]


def _instruction_vectors(left: np.ndarray) -> list[tuple[int, int]]:
    instructions: list[tuple[int, int, int, int]] = []
    for color in np.unique(left):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        coords = np.argwhere(left == color)
        assert len(coords) == 2, "each instruction color must appear exactly twice"
        marked = [
            _has_adjacent_color(left, int(r), int(c), GREY)
            for r, c in coords
        ]
        assert sum(marked) == 1, "grey marker must identify exactly one start pixel"
        start_idx = marked.index(True)
        sr, sc = map(int, coords[start_idx])
        tr, tc = map(int, coords[1 - start_idx])
        instructions.append((sr, sc, tr - sr, tc - sc))
    instructions.sort(key=lambda item: (item[0], item[1]))
    return [(int(dr), int(dc)) for _sr, _sc, dr, dc in instructions]


def _anchor_positions(vectors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    anchors = {(0, 0)}
    for dr, dc in vectors:
        translated = {(r + dr, c + dc) for r, c in anchors}
        anchors |= translated
    return anchors


def _motif_offsets(right: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    yellow = np.argwhere(right == YELLOW)
    green = np.argwhere(right == GREEN)
    assert len(yellow) == 1, "expected exactly one yellow anchor"
    assert len(green) > 0, "expected a green motif"
    anchor_r, anchor_c = map(int, yellow[0])
    offsets = [(int(r) - anchor_r, int(c) - anchor_c) for r, c in green]
    return offsets, (anchor_r, anchor_c)


def _render(shape: tuple[int, int], anchor: tuple[int, int], offsets: list[tuple[int, int]], vectors: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    height, width = shape
    anchor_r, anchor_c = anchor
    for shift_r, shift_c in _anchor_positions(vectors):
        for off_r, off_c in offsets:
            rr = anchor_r + shift_r + off_r
            cc = anchor_c + shift_c + off_c
            if 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode vector instructions from the left half and repeatedly duplicate the
    green motif on the right half, then return only the transformed right area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    vectors = _instruction_vectors(left)
    offsets, anchor = _motif_offsets(right)
    return _render(right.shape, anchor, offsets, vectors)


def _trim_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_connected_shape(height: int, width: int, area: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for row, col in cells:
            for dr, dc in NEIGHBORS4:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return set()
        cells.add(random.choice(frontier))
    return _trim_cells(cells)


def _shape_candidates(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bbox_h = max(r for r, _c in cells) + 1
    bbox_w = max(c for _r, c in cells) + 1
    neighbors = set()
    for row, col in cells:
        for dr, dc in NEIGHBORS4:
            pos = (row + dr, col + dc)
            if pos not in cells:
                neighbors.add(pos)
    inside = [
        (row, col)
        for row, col in neighbors
        if 0 <= row < bbox_h and 0 <= col < bbox_w
    ]
    outside = [
        (row, col)
        for row, col in neighbors
        if row < 0 or row >= bbox_h or col < 0 or col >= bbox_w
    ]
    return inside, outside


def _sample_motif() -> tuple[list[tuple[int, int]], bool]:
    want_inside = random.random() < 0.5
    for _ in range(200):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        max_area = min(height * width, 7)
        min_area = 3
        if want_inside:
            max_area = min(max_area, height * width - 1)
            if max_area < min_area:
                continue
        area = random.randint(min_area, max_area)
        cells = _random_connected_shape(height, width, area)
        if not cells:
            continue
        inside, outside = _shape_candidates(cells)
        if want_inside:
            if not inside:
                continue
            anchor = random.choice(inside)
        else:
            if not outside:
                continue
            anchor = random.choice(outside)
        offsets = sorted((r - anchor[0], c - anchor[1]) for r, c in cells)
        return offsets, want_inside
    raise RuntimeError("failed to sample motif")


def _expanded8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set()
    for row, col in cells:
        for dr, dc in NEIGHBORS8:
            expanded.add((row + dr, col + dc))
    return expanded


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _c in cells),
        max(r for r, _c in cells),
        min(c for _r, c in cells),
        max(c for _r, c in cells),
    )


def _sample_vector(current_green: set[tuple[int, int]], anchors: set[tuple[int, int]], direction: str) -> tuple[int, int] | None:
    r0, r1, c0, c1 = _bbox(current_green)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    expanded = _expanded8(current_green)

    for _ in range(250):
        if direction == "h":
            dc_mag = random.randint(width + 1, width + 3)
            dc = dc_mag if random.random() < 0.5 else -dc_mag
            dr = random.randint(-max(1, height // 3), max(1, height // 3))
        elif direction == "v":
            dr_mag = random.randint(height + 1, height + 3)
            dr = dr_mag if random.random() < 0.5 else -dr_mag
            dc = random.randint(-max(1, width // 3), max(1, width // 3))
        else:
            dr_mag = random.randint(height + 1, height + 2)
            dc_mag = random.randint(width + 1, width + 2)
            dr = dr_mag if random.random() < 0.5 else -dr_mag
            dc = dc_mag if random.random() < 0.5 else -dc_mag
        if dr == 0 and dc == 0:
            continue

        shifted_anchors = {(r + dr, c + dc) for r, c in anchors}
        if anchors & shifted_anchors:
            continue

        shifted_green = {(r + dr, c + dc) for r, c in current_green}
        if shifted_green & expanded:
            continue

        return int(dr), int(dc)
    return None


def _direction_plan(count: int) -> list[str]:
    plans = {
        1: [["h"], ["v"], ["d"]],
        2: [["h", "v"], ["v", "h"], ["h", "d"], ["v", "d"], ["d", "h"], ["d", "v"]],
        3: [["h", "v", "h"], ["v", "h", "v"], ["h", "v", "d"], ["v", "h", "d"], ["d", "h", "v"], ["d", "v", "h"]],
    }
    return list(random.choice(plans[count]))


def _sample_vectors(offsets: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    green = set(offsets)
    anchors = {(0, 0)}
    vectors: list[tuple[int, int]] = []
    plan = _direction_plan(count)

    for direction in plan:
        vector = _sample_vector(green, anchors, direction)
        if vector is None:
            return None
        dr, dc = vector
        vectors.append((dr, dc))
        shifted_green = {(r + dr, c + dc) for r, c in green}
        shifted_anchors = {(r + dr, c + dc) for r, c in anchors}
        green |= shifted_green
        anchors |= shifted_anchors

    return vectors


def _final_green_offsets(offsets: list[tuple[int, int]], vectors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    anchors = _anchor_positions(vectors)
    return {
        (anchor_r + off_r, anchor_c + off_c)
        for anchor_r, anchor_c in anchors
        for off_r, off_c in offsets
    }


def _sample_grey_dir(dr: int, dc: int) -> tuple[int, int] | None:
    choices = []
    for grey_dr, grey_dc in NEIGHBORS4:
        if (grey_dr, grey_dc) == (dr, dc):
            continue
        if abs(grey_dr - dr) + abs(grey_dc - dc) == 1:
            continue
        choices.append((grey_dr, grey_dc))
    return random.choice(choices) if choices else None


def _left_layout(vectors: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int, int, int, int, int]]] | None:
    grey_dirs: list[tuple[int, int]] = []
    for dr, dc in vectors:
        grey_dir = _sample_grey_dir(dr, dc)
        if grey_dir is None:
            return None
        grey_dirs.append(grey_dir)

    cursor = 0
    placements = []
    width_need = 0
    for (dr, dc), (grey_dr, grey_dc) in zip(vectors, grey_dirs, strict=False):
        rel_rows = (0, dr, grey_dr)
        rel_cols = (0, dc, grey_dc)
        min_row = min(rel_rows)
        max_row = max(rel_rows)
        start_row = cursor + 1 - min_row
        cursor += (max_row - min_row) + random.randint(2, 3)
        width_need = max(width_need, max(rel_cols) - min(rel_cols) + 3)
        placements.append((start_row, dr, dc, grey_dr, grey_dc, min(rel_cols), max(rel_cols)))

    left_height = cursor + 1
    left_width = width_need + random.randint(1, 3)
    if left_height > MAX_GRID_SIDE:
        return None

    result = []
    for start_row, dr, dc, grey_dr, grey_dc, min_col, max_col in placements:
        low = 1 - min_col
        high = left_width - 2 - max_col
        if low > high:
            return None
        start_col = random.randint(low, high)
        result.append((start_row, start_col, dr, dc, grey_dr, grey_dc))
    return left_height, left_width, result


def _make_input_grid() -> np.ndarray | None:
    offsets, _inside_anchor = _sample_motif()
    instruction_count = random.randint(1, 3)
    vectors = _sample_vectors(offsets, instruction_count)
    if vectors is None:
        return None

    final_green = _final_green_offsets(offsets, vectors)
    min_r, max_r, min_c, max_c = _bbox(final_green)
    right_height = (max_r - min_r + 1) + random.randint(2, 5)
    right_width = (max_c - min_c + 1) + random.randint(2, 5)
    if right_height > MAX_GRID_SIDE or right_width > MAX_GRID_SIDE:
        return None

    left_layout = _left_layout(vectors)
    if left_layout is None:
        return None
    left_height, left_width, instructions = left_layout

    height = max(left_height, right_height)
    width = left_width + 1 + right_width
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    top_margin = random.randint(1, height - (max_r - min_r + 1) - 1)
    left_margin = random.randint(1, right_width - (max_c - min_c + 1) - 1)
    anchor_r = top_margin - min_r
    anchor_c = left_margin - min_c

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, left_width] = MAROON

    colors = random.sample(INSTRUCTION_COLORS, len(vectors))
    for color, (start_row, start_col, dr, dc, grey_dr, grey_dc) in zip(colors, instructions, strict=False):
        grid[start_row, start_col] = color
        grid[start_row + dr, start_col + dc] = color
        grid[start_row + grey_dr, start_col + grey_dc] = GREY

    right = grid[:, left_width + 1 :]
    for off_r, off_c in offsets:
        right[anchor_r + off_r, anchor_c + off_c] = GREEN
    right[anchor_r, anchor_c] = YELLOW

    expected = _render(right.shape, (anchor_r, anchor_c), offsets, vectors)
    if np.array_equal(expected, right):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_input_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
