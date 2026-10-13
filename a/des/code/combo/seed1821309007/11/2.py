import numpy as np


BG = 8
NON_BG_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MIN_LINES = 4
MAX_LINES = 8
MAX_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dominant_non_bg_color(line: np.ndarray, bg: int) -> int:
    fg = line[line != bg]
    assert fg.size > 0, "each data line must contain foreground pixels"
    values, counts = np.unique(fg, return_counts=True)
    best = int(np.max(counts))
    winners = values[counts == best]
    assert winners.size == 1, "dominant color must be unique"
    return int(winners[0])


def _solve_rowwise(grid: np.ndarray) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 2:
        return None

    h, w = map(int, g.shape)
    if w != h - 1:
        return None

    bg = BG if np.any(g == BG) else _get_bg(g)
    control = g[0]
    if np.any(control == bg):
        return None
    if np.unique(control).size != w:
        return None

    rows = g[1:]
    dominant_to_row: dict[int, np.ndarray] = {}
    for row in rows:
        dom = _dominant_non_bg_color(row, bg)
        if dom in dominant_to_row:
            return None
        dominant_to_row[dom] = row.copy()

    if set(map(int, control)) != set(dominant_to_row):
        return None

    ordered = [dominant_to_row[int(color)] for color in control]
    return np.stack(ordered, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a border control sequence into a permutation and reorder interior lines.

    The primary form uses the first row as the control sequence and reorders the
    remaining rows by their dominant non-background colors. The transposed form
    uses the first column analogously to reorder columns.
    """
    g = np.asarray(grid, dtype=int)

    row_result = _solve_rowwise(g)
    if row_result is not None:
        return row_result

    col_result = _solve_rowwise(g.T)
    if col_result is not None:
        return col_result.T

    raise AssertionError("grid does not match the expected control-line structure")


def _sample_pattern_mask(length: int, target_count: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    kind = int(rng.integers(4))

    if kind == 0:
        start = int(rng.integers(length - target_count + 1))
        mask[start : start + target_count] = True
    elif kind == 1:
        start = int(rng.integers(length - target_count + 1))
        left = int(rng.integers(0, target_count))
        right = target_count - left
        mask[start : start + left] = True
        gap_start = start + left
        gap_end = min(length, gap_start + 2)
        tail_start = max(gap_end, start + left + int(rng.integers(1, 3)))
        if tail_start + right > length:
            tail_start = length - right
        mask[tail_start : tail_start + right] = True
    elif kind == 2:
        center = (length - 1) / 2.0
        order = sorted(range(length), key=lambda idx: (abs(idx - center), idx))
        for idx in order[:target_count]:
            mask[idx] = True
    else:
        step = 2 if length >= 5 else 1
        candidates = list(range(int(rng.integers(step)), length, step))
        rng.shuffle(candidates)
        for idx in candidates[:target_count]:
            mask[idx] = True
        if mask.sum() < target_count:
            extras = [idx for idx in range(length) if not mask[idx]]
            rng.shuffle(extras)
            for idx in extras[: target_count - int(mask.sum())]:
                mask[idx] = True

    assert int(mask.sum()) == target_count
    return mask


def _make_line(length: int, dominant: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    line = np.full(length, BG, dtype=int)
    dom_count = int(rng.integers(length // 2 + 1, length))
    dom_mask = _sample_pattern_mask(length, dom_count, rng)
    line[dom_mask] = dominant

    remaining = np.flatnonzero(~dom_mask)
    usable = remaining[:-1].copy()
    rng.shuffle(usable)
    accent_budget = min(int(rng.integers(0, 3)), len(usable))
    if accent_budget > 0:
        accent_colors = [c for c in palette if c != dominant]
        rng.shuffle(accent_colors)
        cursor = 0
        for color in accent_colors[:accent_budget]:
            if cursor >= len(usable):
                break
            take = 1
            if cursor + 1 < len(usable) and rng.random() < 0.3 and dom_count >= 3:
                take = 2
            cells = usable[cursor : cursor + take]
            line[cells] = color
            cursor += take

    assert _dominant_non_bg_color(line, BG) == dominant
    assert np.any(line == BG), "every data line should retain visible background"
    return line


def _build_row_control_instance(rng: np.random.Generator) -> np.ndarray:
    line_count = int(rng.integers(MIN_LINES, MAX_LINES + 1))
    colors = rng.choice(np.array(NON_BG_COLORS, dtype=int), size=line_count, replace=False).tolist()

    rows = [_make_line(line_count, int(color), colors, rng) for color in colors]
    perm = rng.permutation(line_count)
    if np.all(perm == np.arange(line_count)):
        perm[[0, 1]] = perm[[1, 0]]
    control = np.array([colors[i] for i in perm], dtype=int)

    grid = np.full((line_count + 1, line_count), BG, dtype=int)
    grid[0] = control
    for idx, row in enumerate(rows, start=1):
        grid[idx] = row

    expected = np.stack([rows[i] for i in perm], axis=0)
    assert np.array_equal(main(grid), expected)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_row_control_instance(rng)
        if rng.random() < 0.5:
            grid = grid.T
        if np.unique(grid).size >= 3:
            return grid
    raise RuntimeError("failed to generate a valid input")
