import random
from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    counts = np.sum(grid == color, axis=axis)
    if counts.max() == 0:
        return []
    threshold = max(2, int(np.ceil(counts.max() * 0.6)))
    return [int(i) for i, v in enumerate(counts) if int(v) >= threshold]


def _cell_boxes(row_lines: list[int], col_lines: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            r0, r1 = int(row_lines[i]), int(row_lines[i + 1])
            c0, c1 = int(col_lines[j]), int(col_lines[j + 1])
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                boxes.append((r0, c0, r1, c1))
    return boxes


def _border_sides(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    r0, c0, r1, c1 = box
    return {
        "top": grid[r0, c0 + 1 : c1],
        "bottom": grid[r1, c0 + 1 : c1],
        "left": grid[r0 + 1 : r1, c0],
        "right": grid[r0 + 1 : r1, c1],
    }


def _complete_border(grid: np.ndarray, box: tuple[int, int, int, int], line_color: int) -> bool:
    sides = _border_sides(grid, box)
    return all(side.size > 0 and bool((side == line_color).all()) for side in sides.values())


def _reference_info(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], line_color: int) -> tuple[tuple[int, int, int, int], int]:
    best_box = None
    best_color = None
    best_score = -1
    for box in boxes:
        if not _complete_border(grid, box, line_color):
            continue
        r0, c0, r1, c1 = box
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        vals = [int(v) for v in interior.flatten() if int(v) != line_color and int(v) != 0]
        if not vals:
            continue
        score = len(vals)
        if score > best_score:
            best_score = score
            best_box = box
            best_color = int(Counter(vals).most_common(1)[0][0])
    if best_box is None or best_color is None:
        raise AssertionError("reference cell not found")
    return best_box, best_color


def _gap_descriptor(grid: np.ndarray, box: tuple[int, int, int, int], line_color: int, bg: int) -> tuple[str, int] | None:
    missing = {}
    for side_name, side in _border_sides(grid, box).items():
        n = int(np.count_nonzero(side == bg))
        if n > 0:
            if bool(((side != bg) & (side != line_color)).any()):
                return None
            missing[side_name] = n
        elif not bool((side == line_color).all()):
            return None
    if len(missing) != 1:
        return None
    side_name, gap_size = next(iter(missing.items()))
    if gap_size not in (1, 2, 3):
        return None
    return side_name, gap_size


def _paint_stripes(out: np.ndarray, box: tuple[int, int, int, int], fill_color: int, horizontal: bool, gap_size: int) -> None:
    r0, c0, r1, c1 = box
    period = gap_size + 1
    if horizontal:
        for r in range(r0 + 1, r1, period):
            out[r, c0 + 1 : c1] = fill_color
    else:
        for c in range(c0 + 1, c1, period):
            out[r0 + 1 : r1, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    non_bg = [int(v) for v in g.flatten() if int(v) != bg]
    if not non_bg:
        return g.copy()
    line_color = int(Counter(non_bg).most_common(1)[0][0])

    row_lines = _line_positions(g, line_color, axis=1)
    col_lines = _line_positions(g, line_color, axis=0)
    assert len(row_lines) >= 2 and len(col_lines) >= 2

    boxes = _cell_boxes(row_lines, col_lines)
    ref_box, stripe_color = _reference_info(g, boxes, line_color)

    out = g.copy()
    for box in boxes:
        if box == ref_box:
            continue
        desc = _gap_descriptor(g, box, line_color, bg)
        if desc is None:
            continue
        side_name, gap_size = desc
        horizontal = side_name in ("top", "bottom")
        _paint_stripes(out, box, stripe_color, horizontal, gap_size)
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_gap(grid: np.ndarray, box: tuple[int, int, int, int], side: str, gap_size: int, bg: int) -> None:
    r0, c0, r1, c1 = box
    if side in ("top", "bottom"):
        width = c1 - c0 - 1
        start = c0 + 1 + (width - gap_size) // 2
        row = r0 if side == "top" else r1
        grid[row, start : start + gap_size] = bg
    else:
        height = r1 - r0 - 1
        start = r0 + 1 + (height - gap_size) // 2
        col = c0 if side == "left" else c1
        grid[start : start + gap_size, col] = bg


def _draw_reference_pattern(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    stripe_color: int,
    horizontal: bool,
    gap_size: int,
) -> None:
    _paint_stripes(grid, box, stripe_color, horizontal, gap_size)


def _corner_index(rows: int, cols: int, idx: int) -> tuple[int, int]:
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    return corners[idx]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    colors = list(range(1, 10))
    line_color = random.choice(colors)
    colors.remove(line_color)
    stripe_color = random.choice(colors)

    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 5)
    cell_inner = random.randint(4, 7)
    step = cell_inner + 1

    grid_h = n_rows * cell_inner + n_rows + 1
    grid_w = n_cols * cell_inner + n_cols + 1
    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom_pad = random.randint(1, 4)
    right_pad = random.randint(1, 4)

    grid = np.full((top + grid_h + bottom_pad, left + grid_w + right_pad), bg, dtype=int)

    boxes = []
    row_lines = [top + i * step for i in range(n_rows + 1)]
    col_lines = [left + j * step for j in range(n_cols + 1)]
    for i in range(n_rows):
        for j in range(n_cols):
            box = (row_lines[i], col_lines[j], row_lines[i + 1], col_lines[j + 1])
            boxes.append(box)
            _draw_frame(grid, box, line_color)

    ref_r, ref_c = _corner_index(n_rows, n_cols, random.randrange(4))
    ref_box = boxes[ref_r * n_cols + ref_c]
    ref_horizontal = random.choice([True, False])
    ref_gap_size = random.randint(1, 3)
    _draw_reference_pattern(grid, ref_box, stripe_color, ref_horizontal, ref_gap_size)

    other_indices = [k for k, box in enumerate(boxes) if box != ref_box]
    random.shuffle(other_indices)
    gap_count = random.randint(max(2, len(other_indices) // 2), len(other_indices))

    side_cycle = ["top", "bottom", "left", "right"]
    size_cycle = [1, 2, 3]
    forced = []
    for i, side in enumerate(side_cycle):
        forced.append((side, size_cycle[i % 3]))
    random.shuffle(forced)

    for pos, idx in enumerate(other_indices[:gap_count]):
        box = boxes[idx]
        if pos < len(forced):
            side, gap_size = forced[pos]
        else:
            side = random.choice(side_cycle)
            gap_size = random.choice(size_cycle)
        max_gap = box[2] - box[0] - 1 if side in ("left", "right") else box[3] - box[1] - 1
        gap_size = min(gap_size, max_gap)
        _make_gap(grid, box, side, gap_size, bg)

    try:
        produced = main(grid)
    except Exception:
        return None

    if np.array_equal(produced, grid):
        return None

    unique_gap_sizes = set()
    horizontal_seen = False
    vertical_seen = False
    for box in boxes:
        if box == ref_box:
            continue
        desc = _gap_descriptor(grid, box, line_color, bg)
        if desc is None:
            continue
        side_name, gap_size = desc
        unique_gap_sizes.add(gap_size)
        if side_name in ("top", "bottom"):
            horizontal_seen = True
        else:
            vertical_seen = True

    if not (horizontal_seen and vertical_seen):
        return None
    if len(unique_gap_sizes) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
