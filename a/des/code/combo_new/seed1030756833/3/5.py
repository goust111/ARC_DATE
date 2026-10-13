import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
SEED_COLORS = [1, 2, 3, 4, 6, 7, 9]
INSTRUCTIONS = ["hflip", "vflip", "cw", "ccw"]


def _intervals_from_lines(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = sorted(int(x) for x in lines)
    if not bounds or bounds[0] != 0:
        bounds = [-1] + bounds
    if bounds[-1] != limit - 1:
        bounds = bounds + [limit]

    intervals = []
    for a, b in zip(bounds, bounds[1:]):
        start, stop = a + 1, b
        if start < stop:
            intervals.append((start, stop))
    return intervals


def _cell_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_intervals = _intervals_from_lines(grey_rows, grid.shape[0])
    col_intervals = _intervals_from_lines(grey_cols, grid.shape[1])
    assert row_intervals and col_intervals, "expected grey separator lines"
    return row_intervals, col_intervals


def _seed_pattern(seed_cell: np.ndarray) -> tuple[np.ndarray, int]:
    colors = sorted(
        int(c)
        for c in np.unique(seed_cell)
        if int(c) not in (BLACK, GREY, AZURE)
    )
    assert len(colors) == 1, "seed cell must contain exactly one seed color"
    color = colors[0]
    return seed_cell == color, color


def _read_instruction(cell: np.ndarray) -> str | None:
    coords = np.argwhere(cell == AZURE)
    if coords.size == 0:
        return None
    assert len(coords) == 3, "instruction symbols must contain exactly three azure pixels"

    rows = coords[:, 0]
    cols = coords[:, 1]
    if len(set(map(int, rows))) == 1:
        ordered = sorted(int(c) for c in cols)
        assert ordered[2] - ordered[0] == 2 and len(set(ordered)) == 3
        return "hflip"
    if len(set(map(int, cols))) == 1:
        ordered = sorted(int(r) for r in rows)
        assert ordered[2] - ordered[0] == 2 and len(set(ordered)) == 3
        return "vflip"

    order = np.argsort(rows)
    ordered_rows = [int(rows[i]) for i in order]
    ordered_cols = [int(cols[i]) for i in order]
    assert ordered_rows[2] - ordered_rows[0] == 2 and len(set(ordered_rows)) == 3
    if ordered_cols[0] + 1 == ordered_cols[1] and ordered_cols[1] + 1 == ordered_cols[2]:
        return "cw"
    if ordered_cols[0] - 1 == ordered_cols[1] and ordered_cols[1] - 1 == ordered_cols[2]:
        return "ccw"
    raise AssertionError("unknown azure instruction symbol")


def _apply_instruction(pattern: np.ndarray, instruction: str) -> np.ndarray:
    if instruction == "hflip":
        return np.fliplr(pattern)
    if instruction == "vflip":
        return np.flipud(pattern)
    if instruction == "cw":
        return np.rot90(pattern, -1)
    if instruction == "ccw":
        return np.rot90(pattern, 1)
    raise AssertionError(instruction)


def _paste_pattern(out: np.ndarray, cell_bounds: tuple[int, int, int, int], pattern: np.ndarray, color: int) -> None:
    r0, r1, c0, c1 = cell_bounds
    out[r0:r1, c0:c1] = BLACK
    ph, pw = pattern.shape
    ch, cw = r1 - r0, c1 - c0
    assert ph <= ch and pw <= cw, "final pattern must fit in the output cell"
    top = r0 + (ch - ph) // 2
    left = c0 + (cw - pw) // 2
    region = out[top : top + ph, left : left + pw]
    region[pattern] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply ordered geometric instruction symbols to a seed pattern and draw the result in the final cell.

    Input structure:
    - Black background is divided by full Grey separator lines into a regular grid of equal rectangular cells.
    - The upper-left cell contains one monochrome seed sprite.
    - The other non-final cells contain Azure three-pixel symbols: horizontal line, vertical line, descending diagonal, or ascending diagonal.
    - The lower-right cell is empty and reserved for the answer.

    Rule:
    - Extract the seed sprite from the upper-left cell.
    - Read all non-empty Azure instruction cells in row-major order, skipping the seed cell and final empty cell.
    - A horizontal line flips the current sprite left-right; a vertical line flips it up-down; a descending diagonal rotates it clockwise; an ascending diagonal rotates it counter-clockwise.
    - Copy the input grid and paint the transformed sprite into the lower-right cell using the seed color.

    Key insight:
    - Each small Azure line shape is a visible transform code, and the codes must be composed in reading order before the final sprite is drawn.
    """
    g = np.asarray(grid, dtype=int)
    row_intervals, col_intervals = _cell_slices(g)

    sr0, sr1 = row_intervals[0]
    sc0, sc1 = col_intervals[0]
    pattern, seed_color = _seed_pattern(g[sr0:sr1, sc0:sc1])

    for ri, (r0, r1) in enumerate(row_intervals):
        for ci, (c0, c1) in enumerate(col_intervals):
            if ri == 0 and ci == 0:
                continue
            if ri == len(row_intervals) - 1 and ci == len(col_intervals) - 1:
                continue
            instruction = _read_instruction(g[r0:r1, c0:c1])
            if instruction is not None:
                pattern = _apply_instruction(pattern, instruction)

    out = g.copy()
    tr0, tr1 = row_intervals[-1]
    tc0, tc1 = col_intervals[-1]
    _paste_pattern(out, (tr0, tr1, tc0, tc1), pattern, seed_color)
    return out


# --------------- generator ---------------

LAYOUTS = [(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4)]
GENERATION_RETRIES = 300


def _cell_origin(row: int, col: int, side: int) -> tuple[int, int]:
    return 1 + row * (side + 1), 1 + col * (side + 1)


def _draw_separators(rows: int, cols: int, side: int) -> np.ndarray:
    h = rows * side + rows + 1
    w = cols * side + cols + 1
    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(rows + 1):
        grid[r * (side + 1), :] = GREY
    for c in range(cols + 1):
        grid[:, c * (side + 1)] = GREY
    return grid


def _connected_sprite(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    cells = {(r, c)}
    while len(cells) < area:
        frontier = []
        for cr, cc in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        cells.add((nr, nc))
        mask[nr, nc] = True
    return mask


def _make_seed_mask(side: int) -> np.ndarray | None:
    max_box = min(4, side - 1)
    for _ in range(100):
        h = random.randint(2, max_box)
        w = random.randint(2, max_box)
        area = random.randint(3, min(h * w - 1, max(3, h + w)))
        sprite = _connected_sprite(h, w, area)
        if not sprite.any():
            continue

        top_choices = range(0, side - h + 1)
        left_choices = range(0, side - w + 1)
        top = random.choice(list(top_choices))
        left = random.choice(list(left_choices))

        mask = np.zeros((side, side), dtype=bool)
        mask[top : top + h, left : left + w] = sprite
        variants = {
            mask.tobytes(),
            np.fliplr(mask).tobytes(),
            np.flipud(mask).tobytes(),
            np.rot90(mask, -1).tobytes(),
            np.rot90(mask, 1).tobytes(),
        }
        if len(variants) >= 4:
            return mask
    return None


def _draw_seed(grid: np.ndarray, side: int, mask: np.ndarray, color: int) -> None:
    r0, c0 = _cell_origin(0, 0, side)
    grid[r0 : r0 + side, c0 : c0 + side][mask] = color


def _draw_instruction(grid: np.ndarray, side: int, row: int, col: int, instruction: str) -> None:
    r0, c0 = _cell_origin(row, col, side)
    cr, cc = r0 + side // 2, c0 + side // 2
    if instruction == "hflip":
        pts = [(cr, cc - 1), (cr, cc), (cr, cc + 1)]
    elif instruction == "vflip":
        pts = [(cr - 1, cc), (cr, cc), (cr + 1, cc)]
    elif instruction == "cw":
        pts = [(cr - 1, cc - 1), (cr, cc), (cr + 1, cc + 1)]
    elif instruction == "ccw":
        pts = [(cr - 1, cc + 1), (cr, cc), (cr + 1, cc - 1)]
    else:
        raise AssertionError(instruction)
    for r, c in pts:
        grid[r, c] = AZURE


def _transformed(mask: np.ndarray, sequence: list[str]) -> np.ndarray:
    result = mask.copy()
    for instruction in sequence:
        result = _apply_instruction(result, instruction)
    return result


def _sample_layout_and_side() -> tuple[int, int, int]:
    rows, cols = random.choice(LAYOUTS)
    max_side = min(7, (30 - 1) // max(rows, cols) - 1)
    side = random.randint(5, max_side)
    return rows, cols, side


def _gen_attempt() -> np.ndarray | None:
    rows, cols, side = _sample_layout_and_side()
    n_instructions = rows * cols - 2
    if n_instructions < len(INSTRUCTIONS):
        return None

    seed_mask = _make_seed_mask(side)
    if seed_mask is None:
        return None

    sequence = INSTRUCTIONS[:] + random.choices(INSTRUCTIONS, k=n_instructions - len(INSTRUCTIONS))
    random.shuffle(sequence)
    if np.array_equal(seed_mask, _transformed(seed_mask, sequence)):
        return None

    grid = _draw_separators(rows, cols, side)
    _draw_seed(grid, side, seed_mask, random.choice(SEED_COLORS))

    slot = 0
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue
            if r == rows - 1 and c == cols - 1:
                continue
            _draw_instruction(grid, side, r, c, sequence[slot])
            slot += 1

    try:
        out = main(grid)
    except Exception:
        return None
    tr0, tc0 = _cell_origin(rows - 1, cols - 1, side)
    if not np.any(out[tr0 : tr0 + side, tc0 : tc0 + side] != BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
