import numpy as np


RED = 2
BLUE = 1
GREY = 5

KEEP = 0
MIRROR = 1
ERASE = 2

HEIGHT_RANGE = (6, 13)
LEFT_WIDTH_RANGE = (4, 9)
RIGHT_WIDTH_RANGE = (5, 13)
PALETTE_SIZE_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 400
ROW_PATTERN_ATTEMPTS = 80

BACKGROUND_CHOICES = (0, 3, 4, 6, 7, 8, 9)
PATTERN_COLOR_CHOICES = (0, 3, 4, 6, 7, 8, 9)


def _dominant_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_and_bg(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h > 0 and w >= 3, "expected a non-empty grid with room for two halves"

    provisional_bg = _dominant_color(g)
    candidates: list[tuple[int, int]] = []

    for c in range(1, w - 1):
        if not np.all(g[:, c] == g[0, c]):
            continue

        separator_color = int(g[0, c])
        if separator_color == provisional_bg:
            continue

        left = g[:, :c]
        right = g[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue

        bg = _dominant_color(np.concatenate((left.ravel(), right.ravel())))
        instruction_colors: set[int] = set()
        ok = True

        for r in range(h):
            row_colors = set(map(int, np.unique(left[r])))
            row_colors.discard(bg)
            if len(row_colors) > 1:
                ok = False
                break
            instruction_colors |= row_colors

        if ok and instruction_colors and instruction_colors <= {RED, BLUE}:
            candidates.append((c, bg))

    assert len(candidates) == 1, "expected a unique vertical separator column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode row-wise instructions from the left half and apply them to the right half.

    Input structure
    - A full-height separator column splits the grid into left and right halves.
    - The left half is mostly background; each row is either all background,
      contains a red stripe, or contains a blue stripe.
    - The right half contains the row content to keep, mirror, or erase.

    Rule
    - Output only the right half.
    - Background-only left row -> keep the right row unchanged.
    - Red left row -> mirror the right row horizontally.
    - Blue left row -> erase the right row to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_col, bg = _find_separator_and_bg(g)
    right = g[:, separator_col + 1 :]
    assert right.shape[1] > 0, "right half must be non-empty"

    out = right.copy()
    for r in range(g.shape[0]):
        row_colors = set(map(int, np.unique(g[r, :separator_col])))
        row_colors.discard(bg)
        assert len(row_colors) <= 1, "each instruction row must use at most one non-background color"

        if not row_colors:
            continue

        instruction = next(iter(row_colors))
        if instruction == RED:
            out[r] = right[r, ::-1]
        elif instruction == BLUE:
            out[r] = bg
        else:
            raise AssertionError("unexpected instruction color")

    return out


def _positive_parts(total: int, parts: int) -> np.ndarray:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return np.array([total], dtype=int)
    extras = np.random.multinomial(total - parts, np.full(parts, 1.0 / parts))
    return extras.astype(int) + 1


def _nonnegative_parts(total: int, parts: int) -> np.ndarray:
    assert parts >= 1 and total >= 0
    if parts == 1:
        return np.array([total], dtype=int)
    return np.random.multinomial(total, np.full(parts, 1.0 / parts)).astype(int)


def _sample_operations(height: int) -> list[int]:
    ops = [KEEP, MIRROR, ERASE] * (height // 3)
    remainder = height - len(ops)
    if remainder:
        extras = np.random.choice([KEEP, MIRROR, ERASE], size=remainder, replace=False)
        ops.extend(map(int, extras.tolist()))
    np.random.shuffle(ops)
    return [int(op) for op in ops]


def _paint_instruction(row: np.ndarray, color: int, bg: int) -> None:
    row[:] = bg
    width = int(row.size)
    min_len = max(2, width // 2)
    stripe_len = int(np.random.randint(min_len, width + 1))
    start = int(np.random.randint(0, width - stripe_len + 1))
    row[start : start + stripe_len] = color


def _make_right_row(
    width: int,
    bg: int,
    palette: tuple[int, ...],
    must_change_under_mirror: bool,
    used_signatures: set[bytes],
) -> np.ndarray | None:
    for _ in range(ROW_PATTERN_ATTEMPTS):
        row = np.full(width, bg, dtype=int)

        max_segments = min(4, width - 1)
        segment_count = int(np.random.randint(1, max_segments + 1))
        min_occupied = max(2, segment_count)
        max_occupied = min(width - max(1, segment_count - 1), max(min_occupied, (3 * width) // 5))
        if max_occupied < min_occupied:
            continue

        occupied = int(np.random.randint(min_occupied, max_occupied + 1))
        segment_lengths = _positive_parts(occupied, segment_count)

        remaining_bg = width - occupied
        min_internal_gaps = segment_count - 1
        if remaining_bg < min_internal_gaps:
            continue

        gap_extras = _nonnegative_parts(remaining_bg - min_internal_gaps, segment_count + 1)
        pos = int(gap_extras[0])
        prev_color = None

        for idx, seg_len in enumerate(segment_lengths.tolist()):
            choices = [color for color in palette if color != prev_color]
            color = int(np.random.choice(choices))
            row[pos : pos + seg_len] = color
            pos += seg_len
            if idx < segment_count - 1:
                pos += 1 + int(gap_extras[idx + 1])
            prev_color = color

        if not np.any(row != bg):
            continue
        if must_change_under_mirror and np.array_equal(row, row[::-1]):
            continue

        signature = row.tobytes()
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        return row

    return None


def _has_unique_non_bg_separator(grid: np.ndarray, bg: int, separator_col: int) -> bool:
    g = np.asarray(grid, dtype=int)
    cols = [
        c
        for c in range(1, g.shape[1] - 1)
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg
    ]
    return cols == [separator_col]


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*HEIGHT_RANGE))
    left_w = int(np.random.randint(*LEFT_WIDTH_RANGE))
    right_w = int(np.random.randint(*RIGHT_WIDTH_RANGE))
    bg = int(np.random.choice(BACKGROUND_CHOICES))

    palette_pool = [color for color in PATTERN_COLOR_CHOICES if color != bg]
    palette_size = int(np.random.randint(*PALETTE_SIZE_RANGE))
    palette = tuple(int(c) for c in np.random.choice(palette_pool, size=palette_size, replace=False))

    ops = _sample_operations(height)
    if set(ops) != {KEEP, MIRROR, ERASE}:
        return None

    grid = np.full((height, left_w + 1 + right_w), bg, dtype=int)
    grid[:, left_w] = GREY
    expected = np.empty((height, right_w), dtype=int)
    used_rows: set[bytes] = set()

    for r, op in enumerate(ops):
        right_row = _make_right_row(
            right_w,
            bg,
            palette,
            must_change_under_mirror=(op == MIRROR),
            used_signatures=used_rows,
        )
        if right_row is None:
            return None

        grid[r, left_w + 1 :] = right_row

        if op == KEEP:
            expected[r] = right_row
        elif op == MIRROR:
            _paint_instruction(grid[r, :left_w], RED, bg)
            expected[r] = right_row[::-1]
        else:
            _paint_instruction(grid[r, :left_w], BLUE, bg)
            expected[r] = bg

    if _dominant_color(grid) != bg:
        return None
    if not _has_unique_non_bg_separator(grid, bg, left_w):
        return None

    for r, op in enumerate(ops):
        right_row = grid[r, left_w + 1 :]
        if op == MIRROR and np.array_equal(right_row, expected[r]):
            return None
        if op == ERASE and np.array_equal(right_row, expected[r]):
            return None

    try:
        produced = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-instruction puzzle")
