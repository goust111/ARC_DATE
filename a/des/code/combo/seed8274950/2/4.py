from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separators(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    bg = _most_common_color(grid)

    row_candidates: list[tuple[int, int]] = []
    for r in range(h):
        values, counts = np.unique(grid[r, :], return_counts=True)
        idx = int(np.argmax(counts))
        color = int(values[idx])
        count = int(counts[idx])
        if color != bg and count >= w - 1 and len(values) <= 2:
            row_candidates.append((r, color))

    col_candidates: list[tuple[int, int]] = []
    for c in range(w):
        values, counts = np.unique(grid[:, c], return_counts=True)
        idx = int(np.argmax(counts))
        color = int(values[idx])
        count = int(counts[idx])
        if color != bg and count >= h - 1 and len(values) <= 2:
            col_candidates.append((c, color))

    matches: list[tuple[int, int, int, int, int]] = []
    for r, row_color in row_candidates:
        row_mismatch = [int(c) for c in np.flatnonzero(grid[r, :] != row_color)]
        for c, col_color in col_candidates:
            if row_color == col_color:
                continue
            col_mismatch = [int(rr) for rr in np.flatnonzero(grid[:, c] != col_color)]
            if any(cc != c for cc in row_mismatch):
                continue
            if any(rr != r for rr in col_mismatch):
                continue
            if int(grid[r, c]) not in (row_color, col_color):
                continue
            matches.append((r, c, row_color, col_color, bg))

    assert len(matches) == 1, "expected one horizontal and one vertical separator"
    return matches[0]


def _monochrome_components(region: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append((color, cells))
    return components


def _rotate_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    if turns == 0:
        return np.array(mask, dtype=bool, copy=True)
    if turns == 1:
        return np.array(np.rot90(mask, -1), dtype=bool, copy=True)
    if turns == 2:
        return np.array(np.rot90(mask, 2), dtype=bool, copy=True)
    if turns == 3:
        return np.array(np.rot90(mask, 1), dtype=bool, copy=True)
    raise AssertionError("invalid rotation count")


def _rotated_cells(
    cells: list[tuple[int, int]],
    origin_r: int,
    origin_c: int,
    turns: int,
) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    rotated = _rotate_mask(mask, turns)
    rh, rw = map(int, rotated.shape)
    assert (h - rh) % 2 == 0 and (w - rw) % 2 == 0, "quarter-turn rotation requires a lattice-aligned center"

    nr0 = origin_r + r0 + (h - rh) // 2
    nc0 = origin_c + c0 + (w - rw) // 2
    return [(nr0 + int(dr), nc0 + int(dc)) for dr, dc in np.argwhere(rotated)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the grid by a crossing row/column separator pair and rotate each quadrant's objects in place.

    Rule:
    - top-left: unchanged
    - top-right: 90 degrees clockwise
    - bottom-right: 180 degrees
    - bottom-left: 90 degrees counterclockwise
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep_r, sep_c, row_color, col_color, bg = _find_separators(g)
    out = np.full_like(g, bg)
    out[sep_r, :] = g[sep_r, :]
    out[:, sep_c] = g[:, sep_c]

    quadrants = [
        ((0, sep_r, 0, sep_c), 0),
        ((0, sep_r, sep_c + 1, g.shape[1]), 1),
        ((sep_r + 1, g.shape[0], sep_c + 1, g.shape[1]), 2),
        ((sep_r + 1, g.shape[0], 0, sep_c), 3),
    ]

    assert row_color != bg and col_color != bg and row_color != col_color
    for (r_start, r_stop, c_start, c_stop), turns in quadrants:
        region = g[r_start:r_stop, c_start:c_stop]
        for color, cells in _monochrome_components(region, bg):
            for rr, cc in _rotated_cells(cells, r_start, c_start, turns):
                if r_start <= rr < r_stop and c_start <= cc < c_stop:
                    out[rr, cc] = color

    return out


# --------------- generator ---------------

QUADRANT_SIZE_MIN = 5
QUADRANT_SIZE_MAX = 9
MAX_BOX_SIDE = 5
SHAPE_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 180
MAX_GENERATION_ATTEMPTS = 500
EDGE_BIAS_PROBABILITY = 0.35


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_shape(box_h: int, box_w: int, min_area: int) -> np.ndarray | None:
    max_area = box_h * box_w
    if min_area > max_area:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        target_area = int(RNG.integers(min_area, max_area + 1))
        mask = np.zeros((box_h, box_w), dtype=bool)
        seed_r = int(RNG.integers(box_h))
        seed_c = int(RNG.integers(box_w))
        mask[seed_r, seed_c] = True

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(RNG.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target_area:
            continue
        if not _touches_all_sides(mask):
            continue
        return mask
    return None


def _mark_buffer(blocked: np.ndarray, cells: set[tuple[int, int]], radius: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                blocked[rr, cc] = True


def _rotated_local_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    turns: int,
    qh: int,
    qw: int,
) -> set[tuple[int, int]]:
    rotated = _rotate_mask(mask, turns)
    h, w = map(int, mask.shape)
    rh, rw = map(int, rotated.shape)
    if (h - rh) % 2 or (w - rw) % 2:
        return set()

    nr0 = top + (h - rh) // 2
    nc0 = left + (w - rw) // 2
    cells: set[tuple[int, int]] = set()
    for dr, dc in np.argwhere(rotated):
        rr = nr0 + int(dr)
        cc = nc0 + int(dc)
        if 0 <= rr < qh and 0 <= cc < qw:
            cells.add((rr, cc))
    return cells


def _candidate_boxes(qh: int, qw: int, turns: int) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int]] = []
    for box_h in range(1, min(MAX_BOX_SIDE, qh) + 1):
        for box_w in range(1, min(MAX_BOX_SIDE, qw) + 1):
            if box_h * box_w < 2:
                continue
            if turns in (1, 3) and (box_h - box_w) % 2 != 0:
                continue
            if turns != 0 and box_h == 1 and box_w == 1:
                continue
            boxes.append((box_h, box_w))
    return boxes


def _sample_position(qh: int, qw: int, box_h: int, box_w: int, turns: int) -> tuple[int, int]:
    max_r = qh - box_h
    max_c = qw - box_w

    if turns in (1, 3) and box_h != box_w and float(RNG.random()) < EDGE_BIAS_PROBABILITY:
        if box_w > box_h and max_r > 0:
            r0 = int(RNG.choice(np.array([0, max_r], dtype=int)))
            c0 = int(RNG.integers(max_c + 1))
            return r0, c0
        if box_h > box_w and max_c > 0:
            r0 = int(RNG.integers(max_r + 1))
            c0 = int(RNG.choice(np.array([0, max_c], dtype=int)))
            return r0, c0

    return int(RNG.integers(max_r + 1)), int(RNG.integers(max_c + 1))


def _place_quadrant_objects(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    turns: int,
    colors: list[int],
) -> bool:
    r_start, r_stop = row_span
    c_start, c_stop = col_span
    qh = r_stop - r_start
    qw = c_stop - c_start
    boxes = _candidate_boxes(qh, qw, turns)
    if not boxes:
        return False

    area = qh * qw
    max_objects = 1 + int(area >= 36) + int(area >= 64)
    object_count = int(RNG.integers(1, max_objects + 1))

    blocked_in = np.zeros((qh, qw), dtype=bool)
    blocked_out = np.zeros((qh, qw), dtype=bool)

    for _ in range(object_count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            box_h, box_w = boxes[int(RNG.integers(len(boxes)))]
            min_area = max(2, max(box_h, box_w))
            mask = _random_shape(box_h, box_w, min_area=min_area)
            if mask is None:
                continue

            top, left = _sample_position(qh, qw, box_h, box_w, turns)
            input_cells = {
                (top + int(dr), left + int(dc))
                for dr, dc in np.argwhere(mask)
            }
            if any(blocked_in[r, c] for r, c in input_cells):
                continue

            output_cells = _rotated_local_cells(mask, top, left, turns, qh, qw)
            if not output_cells:
                continue
            if turns != 0 and output_cells == input_cells:
                continue
            if any(blocked_out[r, c] for r, c in output_cells):
                continue

            color = int(colors[int(RNG.integers(len(colors)))])
            for r, c in input_cells:
                grid[r_start + r, c_start + c] = color

            _mark_buffer(blocked_in, input_cells, radius=1)
            _mark_buffer(blocked_out, output_cells, radius=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _background_is_unique(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[order[0]]) != bg:
        return False
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = int(RNG.integers(QUADRANT_SIZE_MIN, QUADRANT_SIZE_MAX + 1))
    bottom_h = int(RNG.integers(QUADRANT_SIZE_MIN, QUADRANT_SIZE_MAX + 1))
    left_w = int(RNG.integers(QUADRANT_SIZE_MIN, QUADRANT_SIZE_MAX + 1))
    right_w = int(RNG.integers(QUADRANT_SIZE_MIN, QUADRANT_SIZE_MAX + 1))

    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w
    bg = int(RNG.integers(10))
    palette = [color for color in range(10) if color != bg]
    if len(palette) < 2:
        return None

    row_color, col_color = [int(x) for x in RNG.choice(np.array(palette, dtype=int), size=2, replace=False)]
    object_colors = [color for color in palette if color not in (row_color, col_color)]
    if not object_colors:
        return None

    grid = np.full((height, width), bg, dtype=int)
    sep_r = top_h
    sep_c = left_w
    grid[sep_r, :] = row_color
    grid[:, sep_c] = col_color
    if float(RNG.random()) < 0.5:
        grid[sep_r, sep_c] = row_color

    quadrants = [
        ((0, sep_r), (0, sep_c), 0),
        ((0, sep_r), (sep_c + 1, width), 1),
        ((sep_r + 1, height), (sep_c + 1, width), 2),
        ((sep_r + 1, height), (0, sep_c), 3),
    ]

    for row_span, col_span, turns in quadrants:
        if not _place_quadrant_objects(grid, row_span, col_span, turns, object_colors):
            return None

    if not _background_is_unique(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
