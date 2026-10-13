import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _separator_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == SEP))]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    sep = int(cols[0])
    assert 0 < sep < grid.shape[1] - 1, "separator must split non-empty left/right panels"
    return sep


def _extract_bars(left_panel: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    colors = sorted(int(c) for c in np.unique(left_panel) if int(c) != BG)
    assert colors, "expected at least one instruction bar"

    lengths: dict[int, int] = {}
    rows: dict[int, int] = {}
    for color in colors:
        coords = np.argwhere(left_panel == color)
        assert coords.size > 0
        uniq_rows = np.unique(coords[:, 0])
        assert len(uniq_rows) == 1, "each instruction color must form one horizontal bar"
        row = int(uniq_rows[0])
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        assert int(coords.shape[0]) == c1 - c0 + 1, "bar must be contiguous"
        assert np.all(left_panel[row, c0 : c1 + 1] == color), "bar cells must be solid"
        lengths[color] = c1 - c0 + 1
        rows[color] = row

    assert len(set(lengths.values())) == len(lengths), "bar lengths must be distinct"
    return lengths, rows


def main(grid: np.ndarray) -> np.ndarray:
    """Use the left-panel bars as color-addressed cyclic row-shift instructions for the right panel.

    Input structure:
    - One full-height grey separator column divides the grid into a narrow left panel and a wider right panel.
    - The left panel contains several horizontal solid bars, each in a different non-background color and with a distinct length.
    - The right panel is a colored pattern where each row contains pixels of at most one instruction color from the left panel.

    Rule:
    - Read each left-panel bar as an instruction: bar color c with length n means every right-panel row containing color c is shifted n cells to the right with wraparound.
    - Rows containing no instruction color are left unchanged.
    - Return only the transformed right panel, cropping away the left panel and the grey separator.

    Key insight:
    - Bar placement is a decoy: the acting rows are selected by color occurrence inside the right panel, while the shift amount is set by the matching bar length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0

    shift_by_color, _bar_rows = _extract_bars(left)
    instruction_colors = set(shift_by_color)

    out = right.copy()
    touched = False
    for r in range(right.shape[0]):
        row = right[r]
        present = {int(v) for v in np.unique(row) if int(v) != BG}
        hits = sorted(present & instruction_colors)
        assert len(hits) <= 1, "each right-panel row may contain at most one instruction color"
        if hits:
            touched = True
            out[r] = np.roll(row, shift_by_color[hits[0]] % right.shape[1])

    assert touched, "expected at least one addressed row"
    return out


def _scatter_sum(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    bins = np.zeros(parts, dtype=int)
    for _ in range(total):
        bins[int(rng.integers(parts))] += 1
    return bins


def _segment_colors(
    rng: np.random.Generator,
    fillers: list[int],
    n_segments: int,
    instruction_color: int | None,
) -> list[int]:
    colors: list[int] = []
    pool = fillers.copy()
    rng.shuffle(pool)
    if instruction_color is not None:
        instr_pos = int(rng.integers(n_segments))
    else:
        instr_pos = -1
    cursor = 0
    for i in range(n_segments):
        if i == instr_pos:
            colors.append(instruction_color)
            continue
        if cursor >= len(pool):
            pool = fillers.copy()
            rng.shuffle(pool)
            cursor = 0
        color = pool[cursor]
        cursor += 1
        if colors and color == colors[-1]:
            alts = [c for c in fillers if c != colors[-1]]
            if alts:
                color = int(rng.choice(alts))
        colors.append(int(color))
    return colors


def _base_row(
    width: int,
    segment_colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    n_segments = len(segment_colors)
    max_fill = min(width, 4 * n_segments)
    min_fill = max(n_segments + 1, width // 2)
    filled = int(rng.integers(min_fill, max_fill + 1))

    lengths = np.ones(n_segments, dtype=int)
    extra = filled - n_segments
    while extra > 0:
        idx = int(rng.integers(n_segments))
        if lengths[idx] < 4:
            lengths[idx] += 1
            extra -= 1

    gaps = _scatter_sum(width - filled, n_segments + 1, rng)
    row = np.zeros(width, dtype=int)
    cursor = int(gaps[0])
    for color, length, gap_after in zip(segment_colors, lengths, gaps[1:]):
        row[cursor : cursor + int(length)] = int(color)
        cursor += int(length) + int(gap_after)
    return row


def _make_row(
    width: int,
    rng: np.random.Generator,
    instruction_color: int | None,
    shift: int,
    filler_colors: list[int],
    instruction_colors: set[int],
    used_shifts: set[int],
) -> np.ndarray:
    for _ in range(300):
        n_segments = int(rng.integers(3, min(6, width) + 1))
        seg_colors = _segment_colors(rng, filler_colors, n_segments, instruction_color)
        row = _base_row(width, seg_colors, rng)

        for offset in rng.permutation(width):
            candidate = np.roll(row, int(offset))
            present = {int(v) for v in np.unique(candidate) if int(v) != BG}
            hits = present & instruction_colors
            if instruction_color is None:
                if hits:
                    continue
                if len(present) < 2:
                    continue
                if any(np.array_equal(candidate, np.roll(candidate, n % width)) for n in used_shifts):
                    continue
                return candidate

            if hits != {instruction_color}:
                continue
            if len(present) < 2:
                continue
            if not np.any(candidate[-shift:] != BG):
                continue
            shifted = np.roll(candidate, shift % width)
            if np.array_equal(candidate, shifted):
                continue
            return candidate

    raise RuntimeError("failed to build a non-degenerate row")


def _shift_only_color(row: np.ndarray, color: int, shift: int) -> np.ndarray:
    out = row.copy()
    mask = row == color
    out[mask] = BG
    out[np.roll(mask, shift % row.size)] = color
    return out


def _build_expected(right_panel: np.ndarray, row_color: list[int | None], shift_by_color: dict[int, int]) -> np.ndarray:
    out = right_panel.copy()
    for r, color in enumerate(row_color):
        if color is not None:
            out[r] = np.roll(out[r], shift_by_color[color] % out.shape[1])
    return out


def _semantic_checks(
    grid: np.ndarray,
    right_panel: np.ndarray,
    expected: np.ndarray,
    shift_by_color: dict[int, int],
    bar_rows: dict[int, int],
    row_color: list[int | None],
) -> bool:
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return False

    instruction_colors = set(shift_by_color)
    if np.array_equal(right_panel, expected):
        return False

    used_rows = {color: [] for color in instruction_colors}
    unaffected_rows = []
    for r, color in enumerate(row_color):
        present = {int(v) for v in np.unique(right_panel[r]) if int(v) != BG}
        hits = present & instruction_colors
        if color is None:
            if hits:
                return False
            unaffected_rows.append(r)
        else:
            if hits != {color}:
                return False
            if r == bar_rows[color]:
                return False
            used_rows[color].append(r)

    if not unaffected_rows:
        return False
    if not all(rows for rows in used_rows.values()):
        return False
    if not any(len(rows) >= 2 for rows in used_rows.values()):
        return False

    width = right_panel.shape[1]
    for color, rows in used_rows.items():
        shift = shift_by_color[color] % width
        if shift == 0:
            return False
        wrong_shift = (shift + 1) % width
        if wrong_shift == 0:
            wrong_shift = (shift + 2) % width
        if wrong_shift == shift:
            return False
        if not any(not np.array_equal(np.roll(right_panel[r], wrong_shift), expected[r]) for r in rows):
            return False
        for r in rows:
            if np.array_equal(_shift_only_color(right_panel[r], color, shift), expected[r]):
                return False
            if np.array_equal(right_panel[r], expected[r]):
                return False
            if not np.any(right_panel[r, -shift:] != BG):
                return False

    used_shifts = {shift_by_color[color] % width for color in instruction_colors}
    for shift in used_shifts:
        if shift == 0:
            return False
        if np.array_equal(np.roll(right_panel, shift, axis=1), expected):
            return False

    same_row = right_panel.copy()
    for color, row in bar_rows.items():
        same_row[row] = np.roll(same_row[row], shift_by_color[color] % width)
    if np.array_equal(same_row, expected):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(8, 13))
    num_bars = int(rng.integers(2, 5))
    right_width = int(rng.integers(9, 16))

    max_bar_length = min(5, right_width - 2)
    if max_bar_length < num_bars + 1:
        return None

    lengths = list(rng.choice(np.arange(2, max_bar_length + 1), size=num_bars, replace=False))
    rng.shuffle(lengths)
    instruction_colors = list(rng.choice(PALETTE, size=num_bars, replace=False))

    remaining = [c for c in PALETTE if c not in instruction_colors]
    rng.shuffle(remaining)
    filler_count = int(rng.integers(2, min(4, len(remaining)) + 1))
    filler_colors = remaining[:filler_count]

    left_width = max(lengths) + int(rng.integers(2, 5))
    if left_width + 1 + right_width > 30:
        return None

    bar_row_list = list(rng.choice(np.arange(height), size=num_bars, replace=False))
    bar_rows = {color: int(row) for color, row in zip(instruction_colors, bar_row_list)}
    shift_by_color = {color: int(length) for color, length in zip(instruction_colors, lengths)}

    target_count = {color: 1 for color in instruction_colors}
    multi_color = int(rng.choice(instruction_colors))
    target_count[multi_color] = int(rng.integers(2, 4))
    extras = int(rng.integers(0, 2))
    for _ in range(extras):
        color = int(rng.choice(instruction_colors))
        if sum(target_count.values()) < height - 1:
            target_count[color] += 1

    if sum(target_count.values()) > height - 1:
        return None

    row_color: list[int | None] = [None] * height
    available_rows = list(range(height))
    ordered_colors = sorted(instruction_colors, key=lambda c: (-target_count[c], bar_rows[c]))
    for color in ordered_colors:
        candidates = [r for r in available_rows if r != bar_rows[color]]
        if len(candidates) < target_count[color]:
            return None
        chosen = list(rng.choice(np.array(candidates), size=target_count[color], replace=False))
        for r in chosen:
            row_color[int(r)] = int(color)
            available_rows.remove(int(r))

    if all(color is not None for color in row_color):
        return None

    right_panel = np.zeros((height, right_width), dtype=int)
    used_shifts = {shift_by_color[color] % right_width for color in instruction_colors}
    if 0 in used_shifts:
        return None

    instruction_set = set(instruction_colors)
    for r, color in enumerate(row_color):
        shift = shift_by_color[color] if color is not None else 1
        right_panel[r] = _make_row(
            right_width,
            rng,
            color,
            shift,
            filler_colors,
            instruction_set,
            used_shifts,
        )

    left_panel = np.zeros((height, left_width), dtype=int)
    for color in instruction_colors:
        row = bar_rows[color]
        length = shift_by_color[color]
        start_max = left_width - length
        if start_max < 0:
            return None
        start = int(rng.integers(0, start_max + 1))
        left_panel[row, start : start + length] = color

    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left_panel
    grid[:, left_width] = SEP
    grid[:, left_width + 1 :] = right_panel

    expected = _build_expected(right_panel, row_color, shift_by_color)
    if not _semantic_checks(grid, right_panel, expected, shift_by_color, bar_rows, row_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-addressed row-shift puzzle")
