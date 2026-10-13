from __future__ import annotations

import random

import numpy as np


GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7
OPS = (GREEN, ORANGE, MAGENTA)
TEMPLATE_COLORS = (1, 2, 4, 8, 9)

GEN_RETRIES = 400
MASK_RETRIES = 200
LAYOUT_RETRIES = 200


def _full_line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [idx for idx in range(grid.shape[0]) if np.all(grid[idx, :] == color)]
    return [idx for idx in range(grid.shape[1]) if np.all(grid[:, idx] == color)]


def _regular_step(lines: list[int]) -> int | None:
    if len(lines) < 3:
        return None
    diffs = np.diff(lines)
    if diffs.size == 0 or np.any(diffs != diffs[0]) or int(diffs[0]) <= 1:
        return None
    return int(diffs[0])


def _detect_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    h, w = grid.shape
    best: tuple[int, int, list[int], list[int], int] | None = None

    for color in np.unique(grid):
        color = int(color)
        rows = _full_line_positions(grid, color, axis=0)
        cols = _full_line_positions(grid, color, axis=1)
        if len(rows) < 3 or len(cols) < 3:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        row_step = _regular_step(rows)
        col_step = _regular_step(cols)
        if row_step is None or col_step is None or row_step != col_step:
            continue
        if (h - 1) % row_step != 0 or (w - 1) % col_step != 0:
            continue
        cell_size = row_step - 1
        score = len(rows) + len(cols)
        candidate = (score, color, rows, cols, cell_size)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        raise ValueError("failed to detect grey lattice")
    return best[1], best[2], best[3], best[4]


def _cell_bounds(lines: list[int], idx: int) -> tuple[int, int]:
    return lines[idx] + 1, lines[idx + 1]


def _cell_view(grid: np.ndarray, row_lines: list[int], col_lines: list[int], i: int, j: int) -> np.ndarray:
    r0, r1 = _cell_bounds(row_lines, i)
    c0, c1 = _cell_bounds(col_lines, j)
    return grid[r0:r1, c0:c1]


def _background_color(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> int:
    corner = _cell_view(grid, row_lines, col_lines, 0, 0)
    values, counts = np.unique(corner, return_counts=True)
    return int(values[np.argmax(counts)])


def _template_mask_and_color(cell: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    fg = [int(v) for v in np.unique(cell) if int(v) != bg]
    if len(fg) != 1:
        raise ValueError("template cell must contain exactly one foreground color")
    color = fg[0]
    return cell == color, color


def _apply_op(mask_a: np.ndarray, mask_b: np.ndarray, op_color: int) -> np.ndarray:
    if op_color == GREEN:
        return np.logical_and(mask_a, mask_b)
    if op_color == ORANGE:
        return np.logical_or(mask_a, mask_b)
    if op_color == MAGENTA:
        return np.logical_xor(mask_a, mask_b)
    raise ValueError(f"unexpected operator color: {op_color}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    line_color, row_lines, col_lines, cell_size = _detect_lattice(g)
    _ = line_color  # line color is detected structurally; the copy already preserves it.
    bg = _background_color(g, row_lines, col_lines)

    row_count = len(row_lines) - 1
    col_count = len(col_lines) - 1
    if row_count < 2 or col_count < 2:
        raise ValueError("expected header row and header column")

    row_templates = []
    for i in range(1, row_count):
        cell = _cell_view(g, row_lines, col_lines, i, 0)
        if cell.shape != (cell_size, cell_size):
            raise ValueError("inconsistent row template size")
        row_templates.append(_template_mask_and_color(cell, bg))

    col_templates = []
    for j in range(1, col_count):
        cell = _cell_view(g, row_lines, col_lines, 0, j)
        if cell.shape != (cell_size, cell_size):
            raise ValueError("inconsistent column template size")
        col_templates.append(_template_mask_and_color(cell, bg))

    out = g.copy()
    for i in range(1, row_count):
        row_mask, row_color = row_templates[i - 1]
        for j in range(1, col_count):
            cell = _cell_view(g, row_lines, col_lines, i, j)
            op_color = int(cell[-1, -1])
            result = _apply_op(row_mask, col_templates[j - 1][0], op_color)
            painted = np.full((cell_size, cell_size), bg, dtype=int)
            painted[result] = row_color
            r0, r1 = _cell_bounds(row_lines, i)
            c0, c1 = _cell_bounds(col_lines, j)
            out[r0:r1, c0:c1] = painted
    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return out


def _grow_blob(size: int, area: int) -> np.ndarray:
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rectangle_mask(size: int) -> np.ndarray:
    h = random.randint(2, size)
    w = random.randint(2, size)
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _frame_mask(size: int) -> np.ndarray:
    if size < 3:
        return _rectangle_mask(size)
    h = random.randint(3, size)
    w = random.randint(3, size)
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    mask = np.zeros((size, size), dtype=bool)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _plus_mask(size: int) -> np.ndarray:
    row_band = random.randint(1, min(2, size - 1))
    col_band = random.randint(1, min(2, size - 1))
    r0 = random.randint(0, size - row_band)
    c0 = random.randint(0, size - col_band)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r0 + row_band, :] = True
    mask[:, c0 : c0 + col_band] = True
    return mask


def _stripe_mask(size: int) -> np.ndarray:
    axis = random.choice((0, 1))
    step = random.choice((2, 3))
    band = random.choice((1, 1, 2))
    offset = random.randrange(step)
    mask = np.zeros((size, size), dtype=bool)
    for start in range(offset, size, step):
        stop = min(size, start + band)
        if axis == 0:
            mask[start:stop, :] = True
        else:
            mask[:, start:stop] = True
    return mask


def _diagonal_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    use_main = random.random() < 0.8
    use_anti = random.random() < 0.6
    if not use_main and not use_anti:
        use_main = True
    for idx in range(size):
        if use_main:
            mask[idx, idx] = True
        if use_anti:
            mask[idx, size - 1 - idx] = True
    return mask


def _l_mask(size: int) -> np.ndarray:
    height = random.randint(2, size)
    width = random.randint(2, size)
    thickness = random.randint(1, min(2, min(height, width) - 1))
    mask = np.zeros((size, size), dtype=bool)
    mask[:height, :thickness] = True
    mask[:thickness, :width] = True
    return mask


def _diamond_mask(size: int) -> np.ndarray:
    center_r = random.randrange(size)
    center_c = random.randrange(size)
    radius = random.randint(1, max(1, size // 2))
    rr, cc = np.indices((size, size))
    return (np.abs(rr - center_r) + np.abs(cc - center_c)) <= radius


def _blob_mask(size: int) -> np.ndarray:
    min_area = max(3, size - 1)
    max_area = size * size - max(2, size // 2)
    area = random.randint(min_area, max_area)
    return _grow_blob(size, area)


def _mask_ok(mask: np.ndarray) -> bool:
    size = mask.shape[0]
    area = int(mask.sum())
    if area < max(3, size - 1):
        return False
    if area > size * size - max(2, size // 2):
        return False
    if area == 0 or area == size * size:
        return False
    if int(mask.any(axis=0).sum()) < 2 and int(mask.any(axis=1).sum()) < 2:
        return False
    return True


def _random_mask(size: int) -> np.ndarray:
    generators = (
        _rectangle_mask,
        _frame_mask,
        _plus_mask,
        _stripe_mask,
        _diagonal_mask,
        _l_mask,
        _diamond_mask,
        _blob_mask,
    )
    for _ in range(MASK_RETRIES):
        mask = random.choice(generators)(size)
        mask = _random_transform(mask)
        if _mask_ok(mask):
            return mask
    raise RuntimeError("failed to generate a non-degenerate mask")


def _pair_positions(row_masks: list[np.ndarray], col_masks: list[np.ndarray]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    and_pairs: list[tuple[int, int]] = []
    xor_pairs: list[tuple[int, int]] = []
    partial_pairs: list[tuple[int, int]] = []
    for i, row_mask in enumerate(row_masks):
        for j, col_mask in enumerate(col_masks):
            inter = np.logical_and(row_mask, col_mask)
            xorm = np.logical_xor(row_mask, col_mask)
            if np.any(inter):
                and_pairs.append((i, j))
            if np.any(xorm):
                xor_pairs.append((i, j))
            if np.any(inter) and np.any(xorm):
                partial_pairs.append((i, j))
    return and_pairs, xor_pairs, partial_pairs


def _choose_templates(row_count: int, col_count: int, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    for _ in range(LAYOUT_RETRIES):
        row_masks = [_random_mask(size) for _ in range(row_count)]
        col_masks = [_random_mask(size) for _ in range(col_count)]
        if len({mask.tobytes() for mask in row_masks}) < min(2, row_count):
            continue
        if len({mask.tobytes() for mask in col_masks}) < min(2, col_count):
            continue
        and_pairs, xor_pairs, partial_pairs = _pair_positions(row_masks, col_masks)
        if and_pairs and xor_pairs and partial_pairs:
            return row_masks, col_masks
    raise RuntimeError("failed to generate compatible templates")


def _choose_featured_positions(
    row_masks: list[np.ndarray], col_masks: list[np.ndarray]
) -> dict[tuple[int, int], int]:
    and_pairs, xor_pairs, partial_pairs = _pair_positions(row_masks, col_masks)
    all_positions = [(i, j) for i in range(len(row_masks)) for j in range(len(col_masks))]
    for _ in range(LAYOUT_RETRIES):
        and_pos = random.choice(and_pairs)
        xor_choices = [pos for pos in xor_pairs if pos != and_pos]
        if not xor_choices:
            continue
        xor_pos = random.choice(xor_choices)
        or_choices = [pos for pos in partial_pairs if pos not in {and_pos, xor_pos}]
        if not or_choices:
            or_choices = [pos for pos in all_positions if pos not in {and_pos, xor_pos}]
        if not or_choices:
            continue
        or_pos = random.choice(or_choices)
        return {
            and_pos: GREEN,
            or_pos: ORANGE,
            xor_pos: MAGENTA,
        }
    raise RuntimeError("failed to assign distinct operator showcase cells")


def _empty_lattice(total_rows: int, total_cols: int, cell_size: int, bg: int, line_color: int) -> np.ndarray:
    height = total_rows * cell_size + total_rows + 1
    width = total_cols * cell_size + total_cols + 1
    grid = np.full((height, width), line_color, dtype=int)
    for i in range(total_rows):
        for j in range(total_cols):
            r0 = 1 + i * (cell_size + 1)
            c0 = 1 + j * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = bg
    return grid


def _paint_template(cell: np.ndarray, mask: np.ndarray, color: int, bg: int) -> None:
    cell[:, :] = bg
    cell[mask] = color


def _build_case(
    row_masks: list[np.ndarray],
    row_colors: list[int],
    col_masks: list[np.ndarray],
    col_colors: list[int],
    ops: np.ndarray,
    cell_size: int,
    bg: int = 0,
    line_color: int = GREY,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = len(row_masks) + 1
    total_cols = len(col_masks) + 1
    inp = _empty_lattice(total_rows, total_cols, cell_size, bg, line_color)
    out = inp.copy()

    for i, (mask, color) in enumerate(zip(row_masks, row_colors), start=1):
        _paint_template(_cell_view(inp, list(range(0, inp.shape[0], cell_size + 1)), list(range(0, inp.shape[1], cell_size + 1)), i, 0), mask, color, bg)
        _paint_template(_cell_view(out, list(range(0, out.shape[0], cell_size + 1)), list(range(0, out.shape[1], cell_size + 1)), i, 0), mask, color, bg)

    for j, (mask, color) in enumerate(zip(col_masks, col_colors), start=1):
        _paint_template(_cell_view(inp, list(range(0, inp.shape[0], cell_size + 1)), list(range(0, inp.shape[1], cell_size + 1)), 0, j), mask, color, bg)
        _paint_template(_cell_view(out, list(range(0, out.shape[0], cell_size + 1)), list(range(0, out.shape[1], cell_size + 1)), 0, j), mask, color, bg)

    row_lines = list(range(0, inp.shape[0], cell_size + 1))
    col_lines = list(range(0, inp.shape[1], cell_size + 1))
    for i, row_mask in enumerate(row_masks, start=1):
        for j, col_mask in enumerate(col_masks, start=1):
            in_cell = _cell_view(inp, row_lines, col_lines, i, j)
            in_cell[:, :] = bg
            in_cell[-1, -1] = int(ops[i - 1, j - 1])

            result = _apply_op(row_mask, col_mask, int(ops[i - 1, j - 1]))
            out_cell = _cell_view(out, row_lines, col_lines, i, j)
            out_cell[:, :] = bg
            out_cell[result] = row_colors[i - 1]

    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    line_color = GREY
    total_rows = random.randint(3, 5)
    total_cols = random.randint(3, 5)
    cell_size = random.randint(4, 7)
    row_count = total_rows - 1
    col_count = total_cols - 1

    row_masks, col_masks = _choose_templates(row_count, col_count, cell_size)
    featured = _choose_featured_positions(row_masks, col_masks)

    row_colors = random.sample(TEMPLATE_COLORS, row_count)
    col_colors = random.sample(TEMPLATE_COLORS, col_count)

    ops = np.empty((row_count, col_count), dtype=int)
    for i in range(row_count):
        for j in range(col_count):
            ops[i, j] = random.choice(OPS)
    for pos, op in featured.items():
        ops[pos] = op

    inp, out = _build_case(row_masks, row_colors, col_masks, col_colors, ops, cell_size, bg=bg, line_color=line_color)

    if not np.array_equal(main(inp), out):
        raise RuntimeError("generator produced a case inconsistent with main()")
    if len(set(int(v) for v in ops.ravel())) != 3:
        raise RuntimeError("generator failed to cover all operator branches")

    result_signatures = {
        _apply_op(row_masks[i], col_masks[j], int(ops[i, j])).tobytes()
        for i in range(row_count)
        for j in range(col_count)
    }
    if len(result_signatures) < 3:
        raise RuntimeError("generated outputs are too repetitive")

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _ = _generate_case()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input")
