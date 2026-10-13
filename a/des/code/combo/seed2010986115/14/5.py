import numpy as np


def _score_assignment(
    inner: np.ndarray,
    row_index: np.ndarray,
    col_index: np.ndarray,
    bg: int,
    front: int,
) -> tuple[int, np.ndarray]:
    row_on = row_index == front
    col_on = col_index == front
    mask = np.outer(col_on, row_on)

    violations = int(np.count_nonzero(inner[mask] == bg))
    violations += int(np.count_nonzero(inner[~mask] == front))

    for color in np.unique(inner):
        color = int(color)
        if color in {bg, front}:
            continue
        open_count = int(np.count_nonzero(inner[mask] == color))
        closed_count = int(np.count_nonzero(inner[~mask] == color))
        violations += min(open_count, closed_count)

    return violations, mask


def _infer_palette(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    row_colors = {int(v) for v in np.unique(grid[0, :])}
    col_colors = {int(v) for v in np.unique(grid[:, 0])}
    shared = sorted(row_colors & col_colors)
    border_colors = shared if len(shared) == 2 else sorted(row_colors | col_colors)
    assert len(border_colors) == 2, "index row and index column must use exactly two colors"

    inner = grid[1:, 1:]
    row_index = grid[0, 1:]
    col_index = grid[1:, 0]

    a, b = map(int, border_colors)
    options = []
    for bg, front in ((a, b), (b, a)):
        score, mask = _score_assignment(inner, row_index, col_index, bg, front)
        options.append((score, bg, front, mask))
    options.sort(key=lambda item: item[0])

    _, bg, front, mask = options[0]
    remaining = [int(c) for c in np.unique(inner) if int(c) not in {bg, front}]
    assert len(remaining) == 2, "internal area must contain background/front plus two decoded colors"

    stats = []
    for color in remaining:
        open_count = int(np.count_nonzero(inner[mask] == color))
        closed_count = int(np.count_nonzero(inner[~mask] == color))
        stats.append((color, open_count, closed_count))

    mix = max(stats, key=lambda item: (item[1] - item[2], item[1], -item[2]))[0]
    back = max(stats, key=lambda item: (item[2] - item[1], item[2], -item[1]))[0]
    assert mix != back, "mix color and back color must be distinguishable"
    return bg, front, back, mix, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the back layer from a top-row/left-column outer-product front mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    bg, _front, back, mix, _mask = _infer_palette(g)
    inner = g[1:, 1:]

    out = np.full(inner.shape, bg, dtype=int)
    out[(inner == back) | (inner == mix)] = back
    return out


_MAX_ATTEMPTS = 300
_COLOR_POOL = np.arange(10, dtype=int)


def _sample_binary_vector(length: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(64):
        mode = int(rng.integers(0, 3))
        bits = np.zeros(length, dtype=bool)

        if mode == 0:
            count = int(rng.integers(1, length))
            bits[rng.choice(length, size=count, replace=False)] = True
        elif mode == 1:
            runs = int(rng.integers(1, min(3, length) + 1))
            for _ in range(runs):
                start = int(rng.integers(0, length))
                stop = int(rng.integers(start + 1, length + 1))
                bits[start:stop] = True
        else:
            density = float(rng.uniform(0.25, 0.75))
            bits = rng.random(length) < density

        if 0 < int(bits.sum()) < length:
            return bits

    bits = np.zeros(length, dtype=bool)
    count = max(1, length // 2)
    bits[rng.choice(length, size=count, replace=False)] = True
    return bits


def _make_back_pattern(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = map(int, shape)
    back = np.zeros((height, width), dtype=bool)
    mode = int(rng.integers(0, 3))

    if mode == 0:
        rects = int(rng.integers(1, 4))
        for _ in range(rects):
            rh = int(rng.integers(1, height + 1))
            rw = int(rng.integers(1, width + 1))
            r0 = int(rng.integers(0, height - rh + 1))
            c0 = int(rng.integers(0, width - rw + 1))
            back[r0 : r0 + rh, c0 : c0 + rw] = True
        if rng.random() < 0.4:
            back &= ~(rng.random((height, width)) < 0.12)
    elif mode == 1:
        row_bits = _sample_binary_vector(height, rng)
        col_bits = _sample_binary_vector(width, rng)
        if rng.random() < 0.7:
            back[row_bits, :] = True
        if rng.random() < 0.7:
            back[:, col_bits] = True
        if not back.any():
            back |= np.outer(row_bits, col_bits)
    else:
        density = float(rng.uniform(0.2, 0.6))
        back = rng.random((height, width)) < density

    return back


def _enforce_branch_coverage(mask: np.ndarray, back: np.ndarray, rng: np.random.Generator) -> None:
    open_coords = np.argwhere(mask)
    closed_coords = np.argwhere(~mask)
    assert len(open_coords) >= 2 and len(closed_coords) >= 2

    open_pick = rng.choice(len(open_coords), size=2, replace=False)
    closed_pick = rng.choice(len(closed_coords), size=2, replace=False)

    r, c = map(int, open_coords[int(open_pick[0])])
    back[r, c] = True
    r, c = map(int, open_coords[int(open_pick[1])])
    back[r, c] = False

    r, c = map(int, closed_coords[int(closed_pick[0])])
    back[r, c] = True
    r, c = map(int, closed_coords[int(closed_pick[1])])
    back[r, c] = False


def _render_grid(
    row_bits: np.ndarray,
    col_bits: np.ndarray,
    back_mask: np.ndarray,
    bg: int,
    front: int,
    back: int,
    mix: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.outer(col_bits, row_bits)
    inner = np.full(mask.shape, bg, dtype=int)
    inner[mask & ~back_mask] = front
    inner[mask & back_mask] = mix
    inner[~mask & back_mask] = back

    grid = np.full((mask.shape[0] + 1, mask.shape[1] + 1), bg, dtype=int)
    grid[0, 1:] = np.where(row_bits, front, bg)
    grid[1:, 0] = np.where(col_bits, front, bg)
    grid[0, 0] = front if rng.random() < 0.5 else bg
    grid[1:, 1:] = inner

    expected = np.full(mask.shape, bg, dtype=int)
    expected[back_mask] = back
    return grid, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_h = int(rng.integers(4, 12))
    inner_w = int(rng.integers(4, 12))

    row_bits = _sample_binary_vector(inner_w, rng)
    col_bits = _sample_binary_vector(inner_h, rng)
    mask = np.outer(col_bits, row_bits)

    open_count = int(mask.sum())
    closed_count = int(mask.size - open_count)
    if open_count < 2 or closed_count < 2:
        return None

    bg, front, back, mix = map(int, rng.choice(_COLOR_POOL, size=4, replace=False))
    back_mask = _make_back_pattern((inner_h, inner_w), rng)
    _enforce_branch_coverage(mask, back_mask, rng)

    grid, expected = _render_grid(row_bits, col_bits, back_mask, bg, front, back, mix, rng)
    if len(np.unique(grid)) != 4:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
