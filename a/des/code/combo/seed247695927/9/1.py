import numpy as np


MAX_RETRIES = 300


def _infer_cell_grid(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    for color in map(int, np.unique(g)):
        rows = np.where(np.all(g == color, axis=1))[0]
        cols = np.where(np.all(g == color, axis=0))[0]
        if len(rows) < 3 or len(cols) < 3:
            continue
        if rows[0] != 0 or cols[0] != 0 or rows[-1] != h - 1 or cols[-1] != w - 1:
            continue
        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        step = int(row_diffs[0])
        if step <= 1 or step != int(col_diffs[0]):
            continue
        n = len(rows) - 1
        if len(cols) != n + 1:
            continue
        if h != n * step + 1 or w != n * step + 1:
            continue
        inner = step - 1
        return color, rows, cols, step, inner
    raise AssertionError("failed to infer regular separator grid")


def _extract_patterns(grid: np.ndarray) -> tuple[list[list[tuple[int, ...]]], dict[tuple[int, ...], np.ndarray]]:
    _sep, rows, cols, _step, inner = _infer_cell_grid(grid)
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_patterns: list[list[tuple[int, ...]]] = []
    n = len(rows) - 1
    for i in range(n):
        row_patterns: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = int(rows[i] + 1)
            c0 = int(cols[j] + 1)
            patch = np.asarray(grid[r0 : r0 + inner, c0 : c0 + inner], dtype=int)
            key = tuple(map(int, patch.ravel()))
            if key not in patterns:
                patterns[key] = patch.copy()
            row_patterns.append(key)
        cell_patterns.append(row_patterns)
    return cell_patterns, patterns


def _candidate_placements(mask: np.ndarray, existing: np.ndarray) -> list[tuple[int, int]]:
    n = existing.shape[0]
    k = mask.shape[0]
    candidates: list[tuple[int, int]] = []
    for top in range(n - k + 1):
        for left in range(n - k + 1):
            ok = True
            for r, c in np.argwhere(existing):
                rr = int(r) - top
                cc = int(c) - left
                if rr < 0 or cc < 0 or rr >= k or cc >= k or not bool(mask[rr, cc]):
                    ok = False
                    break
            if ok:
                candidates.append((top, left))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale a minority cell's inner pattern up to the cell-grid level, completing a fractal overlay.

    Input structure:
    - Grid divided by single-color separator lines into an N x N grid of equal square cells.
    - Most cells share one interior pattern; a minority of cells share another.

    Rule:
    - Compare the minority cell pattern against the majority cell pattern.
    - The pixels where they differ form a binary blueprint inside the minority cell.
    - Treat each blueprint pixel as one whole cell at the macro level.
    - Place that k x k blueprint on the cell grid so all existing minority cells land on active blueprint pixels.
    - Fill every remaining active blueprint position with the minority pattern.
    """
    g = np.asarray(grid, dtype=int)
    _sep, rows, cols, _step, inner = _infer_cell_grid(g)
    n = len(rows) - 1

    cell_patterns, patterns = _extract_patterns(g)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_patterns:
        for key in row:
            counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 2, "expected exactly majority and minority cell patterns"
    majority_key = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_key = min(counts.items(), key=lambda kv: kv[1])[0]
    assert counts[majority_key] > counts[minority_key], "minority pattern must be strictly rarer"

    majority = patterns[majority_key]
    minority = patterns[minority_key]
    assert majority.shape == minority.shape == (inner, inner)

    mask = minority != majority
    k = mask.shape[0]
    assert mask.shape[0] == mask.shape[1], "minority blueprint must be square"
    assert bool(mask.any()) and not bool(mask.all()), "blueprint must have active and inactive pixels"

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_patterns[i][j] == minority_key

    candidates = _candidate_placements(mask, existing)
    assert len(candidates) == 1, "blueprint placement must be unique"
    top, left = candidates[0]

    out = g.copy()
    for i in range(k):
        for j in range(k):
            r0 = int(rows[top + i] + 1)
            c0 = int(cols[left + j] + 1)
            out[r0 : r0 + inner, c0 : c0 + inner] = minority if bool(mask[i, j]) else majority
    return out


def _random_majority_pattern(inner: int, colors: list[int]) -> np.ndarray:
    base = int(colors[0])
    patch = np.full((inner, inner), base, dtype=int)
    if inner >= 2 and np.random.random() < 0.7:
        motif_color = int(colors[1])
        motif_count = int(np.random.randint(1, min(inner * inner - 1, inner + 1) + 1))
        for idx in np.random.choice(inner * inner, size=motif_count, replace=False):
            patch.flat[int(idx)] = motif_color
    return patch


def _random_mask(inner: int) -> np.ndarray:
    total = inner * inner
    min_on = 2 if total >= 4 else 1
    max_on = max(min_on, total - 1)
    count = int(np.random.randint(min_on, max_on + 1))
    mask = np.zeros((inner, inner), dtype=bool)
    chosen = np.random.choice(total, size=count, replace=False)
    mask.flat[chosen] = True
    if mask.all() or not mask.any():
        return _random_mask(inner)
    if len({int(r) for r, _ in np.argwhere(mask)}) < 2 and inner >= 2:
        return _random_mask(inner)
    if len({int(c) for _, c in np.argwhere(mask)}) < 2 and inner >= 2:
        return _random_mask(inner)
    return mask


def _build_input(
    n: int,
    inner: int,
    sep: int,
    majority: np.ndarray,
    minority: np.ndarray,
    existing: np.ndarray,
) -> np.ndarray:
    step = inner + 1
    size = n * step + 1
    grid = np.full((size, size), sep, dtype=int)
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            grid[r0 : r0 + inner, c0 : c0 + inner] = minority if bool(existing[i, j]) else majority
    return grid


def _gen_attempt() -> np.ndarray | None:
    inner = int(np.random.randint(2, 5))
    n = int(np.random.randint(inner + 1, inner + 5))
    colors = list(map(int, np.random.choice(10, size=4, replace=False)))
    sep = colors[0]
    majority = _random_majority_pattern(inner, colors[1:3])
    active_color = int(colors[3])

    mask = _random_mask(inner)
    minority = majority.copy()
    minority[mask] = active_color
    if np.array_equal(minority, majority):
        return None

    full_active = np.argwhere(mask)
    top = int(np.random.randint(0, n - inner + 1))
    left = int(np.random.randint(0, n - inner + 1))

    translated = [(top + int(r), left + int(c)) for r, c in full_active]
    if len(translated) < 2:
        return None

    order = np.random.permutation(len(translated))
    subset_sizes = list(range(1, len(translated)))
    np.random.shuffle(subset_sizes)
    chosen_existing: np.ndarray | None = None

    for subset_size in subset_sizes:
        for _ in range(12):
            subset_ids = order[:subset_size] if subset_size == len(order) else np.random.choice(len(translated), size=subset_size, replace=False)
            existing = np.zeros((n, n), dtype=bool)
            for idx in np.asarray(subset_ids).tolist():
                r, c = translated[int(idx)]
                existing[r, c] = True
            if len(_candidate_placements(mask, existing)) == 1:
                chosen_existing = existing
                break
        if chosen_existing is not None:
            break

    if chosen_existing is None:
        return None

    grid = _build_input(n, inner, sep, majority, minority, chosen_existing)
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
