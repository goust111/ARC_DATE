import random
from collections import Counter

import numpy as np


BG = 0
GRAY = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, GRAY))

GRID_HEIGHT_RANGE = (6, 16)
GRID_WIDTH_RANGE = (6, 16)
MARKED_ROW_COUNT_RANGE = (1, 6)
MAX_GENERATION_ATTEMPTS = 300


def _find_control_column(grid: np.ndarray) -> int:
    gray_pixels = np.argwhere(grid == GRAY)
    assert len(gray_pixels) > 0, "expected at least one gray control pixel"
    control_cols = np.unique(gray_pixels[:, 1])
    assert len(control_cols) == 1, "gray pixels must lie in a single control column"
    return int(control_cols[0])


def _find_template_row(grid: np.ndarray, control_col: int) -> tuple[int, np.ndarray, np.ndarray]:
    full_rows = np.flatnonzero((grid != BG).all(axis=1))
    if len(full_rows) == 1:
        row = int(full_rows[0])
        cols = np.arange(grid.shape[1], dtype=int)
        return row, cols, grid[row, cols].copy()

    partial_rows: list[int] = []
    for r in range(grid.shape[0]):
        zeros = np.flatnonzero(grid[r] == BG)
        if len(zeros) == 1 and int(zeros[0]) == control_col:
            mask = np.ones(grid.shape[1], dtype=bool)
            mask[control_col] = False
            if np.all(grid[r, mask] != BG):
                partial_rows.append(int(r))

    assert len(partial_rows) == 1, "expected a unique template row"
    row = partial_rows[0]
    cols = np.flatnonzero(np.arange(grid.shape[1]) != control_col)
    return row, cols, grid[row, cols].copy()


def _first_occurrence_map(sequence: np.ndarray) -> dict[int, int]:
    first: dict[int, int] = {}
    for idx, value in enumerate(sequence.tolist()):
        value = int(value)
        if value not in first:
            first[value] = idx
    return first


def _nearest_shift_color(grid: np.ndarray, row: int, control_col: int) -> int:
    candidates = []
    for col, value in enumerate(grid[row].tolist()):
        value = int(value)
        if value not in (BG, GRAY):
            candidates.append((abs(col - control_col), col, value))

    assert candidates, f"row {row} must contain a shift marker"
    best_distance = min(distance for distance, _col, _value in candidates)
    nearest = [(col, value) for distance, col, value in candidates if distance == best_distance]
    assert len(nearest) == 1, f"row {row} has ambiguous nearest shift markers"
    return int(nearest[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the template-row color sequence by marker-indicated offsets and copy it onto selected rows.

    Input structure:
    - The background is black.
    - One unique template row is filled by a non-black color sequence; some inputs may leave only the control-column crossing black while keeping the rest of that row filled.
    - One unique control column contains gray pixels that mark which rows should be rewritten.
    - Each marked row contains at least one non-black, non-gray shift marker, and the nearest such marker to the control column uses a color that also appears in the template row.

    Rule:
    - Find the control column from the gray pixels and find the unique template row.
    - Read the template row left-to-right as the source sequence, ignoring only a possible black control-column hole.
    - For each gray-marked row, read the nearest shift-marker color and locate that color's first occurrence index in the template sequence.
    - Left-rotate the template sequence by that index and write the rotated sequence onto the marked row.
    - Start the output from an all-black grid and keep only the template row plus the rewritten marked rows.

    Key insight:
    - The row marker color is an indirect pointer into the template row: its first position in the template determines the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    control_col = _find_control_column(g)
    template_row, template_cols, template_seq = _find_template_row(g, control_col)
    first_occurrence = _first_occurrence_map(template_seq)

    out = np.full_like(g, BG)
    out[template_row, template_cols] = template_seq

    marked_rows = [int(r) for r in np.flatnonzero(g[:, control_col] == GRAY) if int(r) != template_row]
    for row in marked_rows:
        shift_color = _nearest_shift_color(g, row, control_col)
        assert shift_color in first_occurrence, "shift marker color must appear in the template row"
        shift = first_occurrence[shift_color]
        out[row, template_cols] = np.roll(template_seq, -shift)

    return out


def _build_template_sequence(width: int) -> np.ndarray:
    if random.random() < 0.5:
        motif_len = random.randint(2, min(5, width - 1))
        color_count = random.randint(2, min(motif_len, 5, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        motif = colors.copy()
        while len(motif) < motif_len:
            motif.append(random.choice(colors))
        random.shuffle(motif)
        sequence = (motif * ((width + motif_len - 1) // motif_len))[:width]
        for _ in range(random.randint(0, max(1, width // 4))):
            idx = random.randrange(width)
            sequence[idx] = random.choice(colors)
        return np.array(sequence, dtype=int)

    color_count = random.randint(2, min(width, len(PALETTE)))
    colors = random.sample(PALETTE, color_count)
    random.shuffle(colors)
    sequence = colors[:]
    while len(sequence) < width:
        options = colors if len(colors) == 1 else [color for color in colors if color != sequence[-1]]
        sequence.append(random.choice(options))
    return np.array(sequence[:width], dtype=int)


def _side_choices(control_col: int, width: int, count: int) -> list[str]:
    available = []
    if control_col > 0:
        available.append("left")
    if control_col < width - 1:
        available.append("right")
    assert available

    if len(available) == 2 and count >= 2 and random.random() < 0.7:
        sides = ["left", "right"]
        while len(sides) < count:
            sides.append(random.choice(available))
        random.shuffle(sides)
        return sides

    return [random.choice(available) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if width < 3 or height < 2:
        return None

    grid = np.full((height, width), BG, dtype=int)
    template_row = random.randrange(height)
    control_col = random.randint(1, width - 2)

    template_seq = _build_template_sequence(width)
    grid[template_row] = template_seq

    available_rows = [row for row in range(height) if row != template_row]
    marked_count = random.randint(MARKED_ROW_COUNT_RANGE[0], min(MARKED_ROW_COUNT_RANGE[1], len(available_rows)))
    marked_rows = random.sample(available_rows, marked_count)

    repeated_colors = [color for color, count in Counter(template_seq.tolist()).items() if count > 1]
    forced_duplicate_row = random.choice(marked_rows) if repeated_colors and random.random() < 0.7 else None

    sides = _side_choices(control_col, width, marked_count)
    for row, side in zip(marked_rows, sides):
        grid[row, control_col] = GRAY

        if side == "left":
            marker_col = random.randint(0, control_col - 1)
        else:
            marker_col = random.randint(control_col + 1, width - 1)

        if row == forced_duplicate_row:
            marker_color = int(random.choice(repeated_colors))
        else:
            marker_color = int(random.choice(template_seq.tolist()))
        grid[row, marker_col] = marker_color

    return grid if _is_valid_input(grid) else None


def _is_valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or min(grid.shape) <= 0:
        return False

    gray_pixels = np.argwhere(grid == GRAY)
    if len(gray_pixels) == 0:
        return False
    if len(np.unique(gray_pixels[:, 1])) != 1:
        return False
    control_col = int(gray_pixels[0, 1])

    full_rows = np.flatnonzero((grid != BG).all(axis=1))
    if len(full_rows) != 1:
        return False
    template_row = int(full_rows[0])

    if np.any(grid[template_row] == GRAY):
        return False

    template_colors = set(int(value) for value in grid[template_row].tolist())
    if BG in template_colors or GRAY in template_colors:
        return False

    marked_rows = [int(r) for r in np.flatnonzero(grid[:, control_col] == GRAY)]
    if template_row in marked_rows:
        return False
    if not marked_rows:
        return False

    for row in range(grid.shape[0]):
        non_bg = np.flatnonzero((grid[row] != BG) & (grid[row] != GRAY))
        if row == template_row:
            if len(non_bg) != grid.shape[1]:
                return False
            continue

        if row in marked_rows:
            if len(non_bg) != 1:
                return False
            if int(grid[row, int(non_bg[0])]) not in template_colors:
                return False
        elif len(non_bg) != 0:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    except Exception:
        return False

    return isinstance(out, np.ndarray) and out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
