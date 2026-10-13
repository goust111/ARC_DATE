import random

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_BG_COLOR = 0
_LINE_COLOR = 9
_PATTERN_COLOR = 4

_CELL_SIZE_CHOICES = (3, 4)
_MODE_CHOICES = ("simple", "chain", "future", "both")
_GEN_RETRIES = 400


def _line_indices_1d(arr: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [idx for idx in range(arr.shape[0]) if np.all(arr[idx, :] == color)]
    return [idx for idx in range(arr.shape[1]) if np.all(arr[:, idx] == color)]


def _regular_step(indices: list[int]) -> int | None:
    if len(indices) < 2:
        return None
    step = indices[1] - indices[0]
    if step <= 1:
        return None
    for i in range(2, len(indices)):
        if indices[i] - indices[i - 1] != step:
            return None
    return step


def _detect_grid_structure(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int], int]] = []
    for color in map(int, np.unique(grid)):
        row_lines = _line_indices_1d(grid, color, axis=0)
        col_lines = _line_indices_1d(grid, color, axis=1)
        row_step = _regular_step(row_lines)
        col_step = _regular_step(col_lines)
        if row_step is None or col_step is None or row_step != col_step:
            continue
        if row_lines[0] != 0 or col_lines[0] != 0:
            continue
        if row_lines[-1] != h - 1 or col_lines[-1] != w - 1:
            continue
        if row_step * (len(row_lines) - 1) != h - 1:
            continue
        if col_step * (len(col_lines) - 1) != w - 1:
            continue
        candidates.append((color, row_lines, col_lines, row_step - 1))

    assert len(candidates) == 1, "expected exactly one regular square grid"
    line_color, row_lines, col_lines, cell_size = candidates[0]
    assert cell_size >= 1
    return line_color, row_lines, col_lines, cell_size


def _infer_colors(
    grid: np.ndarray, row_lines: list[int], col_lines: list[int], line_color: int
) -> tuple[int, int]:
    line_mask = np.zeros_like(grid, dtype=bool)
    line_mask[row_lines, :] = True
    line_mask[:, col_lines] = True
    inner = grid[~line_mask]
    values, counts = np.unique(inner, return_counts=True)
    counts_by_color = {int(v): int(c) for v, c in zip(values, counts)}
    bg = max(counts_by_color, key=counts_by_color.get)
    pattern_colors = [color for color in counts_by_color if color not in (bg, line_color)]
    assert len(pattern_colors) == 1, "expected exactly one pattern color inside grid cells"
    return bg, pattern_colors[0]


def _extract_patterns(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    bg_color: int,
    pattern_color: int,
    cell_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    patterns = np.zeros((n_rows, n_cols, cell_size, cell_size), dtype=bool)
    filled = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        rs = slice(row_lines[r] + 1, row_lines[r + 1])
        for c in range(n_cols):
            cs = slice(col_lines[c] + 1, col_lines[c + 1])
            block = grid[rs, cs]
            assert block.shape == (cell_size, cell_size)
            assert not np.any(block == grid[row_lines[r], col_lines[c]])
            assert np.all((block == bg_color) | (block == pattern_color))
            pattern = block == pattern_color
            patterns[r, c] = pattern
            filled[r, c] = bool(pattern.any())

    return patterns, filled


def _propagate_patterns(patterns: np.ndarray, initially_filled: np.ndarray) -> np.ndarray:
    out = patterns.copy()
    filled = initially_filled.copy()
    n_rows, n_cols = filled.shape

    for r in range(n_rows):
        for c in range(n_cols):
            if filled[r, c]:
                continue
            acc = np.zeros_like(out[r, c], dtype=bool)
            has_neighbor = False
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and filled[nr, nc]:
                    acc ^= out[nr, nc]
                    has_neighbor = True
            assert has_neighbor, "each empty cell must have at least one filled neighbor at fill time"
            out[r, c] = acc
            filled[r, c] = True

    return out


def _render_patterns(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
    bg_color: int,
    line_color: int,
    pattern_color: int,
    patterns: np.ndarray,
) -> np.ndarray:
    grid = np.full(shape, bg_color, dtype=int)
    grid[row_lines, :] = line_color
    grid[:, col_lines] = line_color
    for r in range(patterns.shape[0]):
        rs = slice(row_lines[r] + 1, row_lines[r + 1])
        for c in range(patterns.shape[1]):
            cs = slice(col_lines[c] + 1, col_lines[c + 1])
            block = grid[rs, cs]
            block[patterns[r, c]] = pattern_color
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill empty board cells by xor-combining the patterns of their filled orthogonal neighbors.

    Input structure:
    - The background is black, and a single-pixel maroon lattice forms a complete square cell grid with outer border lines.
    - Each cell interior has the same square size and contains either a yellow binary pattern or only black pixels.
    - Yellow pixels appear only inside cell interiors, never on the maroon grid lines.

    Rule:
    - Detect the evenly spaced maroon rows and columns to recover the cell boundaries and each cell's interior bitmap.
    - Treat every initially non-empty cell as already filled; then scan cells from top to bottom and left to right.
    - When an initially empty cell is visited, xor together the interior bitmaps of all currently filled 4-neighbors and write that bitmap into the cell; the cell then counts as filled for later cells.

    Key insight:
    - The task is governed by cell-level state propagation: the visible pattern in each empty cell is the pixelwise parity sum of adjacent filled patterns under scanline order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_color, row_lines, col_lines, cell_size = _detect_grid_structure(g)
    bg_color, pattern_color = _infer_colors(g, row_lines, col_lines, line_color)
    patterns, filled = _extract_patterns(g, row_lines, col_lines, bg_color, pattern_color, cell_size)
    completed = _propagate_patterns(patterns, filled)
    return _render_patterns(g.shape, row_lines, col_lines, bg_color, line_color, pattern_color, completed)


def _choose_dimensions(rng: random.Random) -> tuple[int, int, int]:
    cell_size = rng.choice(_CELL_SIZE_CHOICES)
    max_dim = (30 - 1) // (cell_size + 1)
    while True:
        n_rows = rng.randint(2, max_dim)
        n_cols = rng.randint(2, max_dim)
        if n_rows * n_cols >= 6 and max(n_rows, n_cols) >= 3:
            return n_rows, n_cols, cell_size


def _mask_features(seed_mask: np.ndarray) -> dict[str, bool | int] | None:
    n_rows, n_cols = seed_mask.shape
    empty_count = int((~seed_mask).sum())
    if empty_count < 2:
        return None

    filled = seed_mask.copy()
    used_multi = False
    used_chain = False
    used_future = False

    for r in range(n_rows):
        for c in range(n_cols):
            if seed_mask[r, c]:
                continue
            neighbors: list[tuple[int, int]] = []
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and filled[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                return None
            if len(neighbors) >= 2:
                used_multi = True
            if any(not seed_mask[nr, nc] for nr, nc in neighbors):
                used_chain = True
            if any(nr > r or nc > c for nr, nc in neighbors):
                used_future = True
            filled[r, c] = True

    return {
        "empty_count": empty_count,
        "used_multi": used_multi,
        "used_chain": used_chain,
        "used_future": used_future,
    }


def _valid_for_mode(features: dict[str, bool | int], mode: str) -> bool:
    if not bool(features["used_multi"]):
        return False
    if mode == "chain":
        return bool(features["used_chain"])
    if mode == "future":
        return bool(features["used_future"])
    if mode == "both":
        return bool(features["used_chain"]) and bool(features["used_future"])
    return True


def _sample_seed_mask(n_rows: int, n_cols: int, rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(_MODE_CHOICES)
    total = n_rows * n_cols
    for _ in range(200):
        p = rng.uniform(0.28, 0.55)
        flat = [rng.random() < p for _ in range(total)]
        seed_mask = np.array(flat, dtype=bool).reshape(n_rows, n_cols)
        if int(seed_mask.sum()) < 2 or int(seed_mask.sum()) > total - 2:
            continue
        features = _mask_features(seed_mask)
        if features is None or not _valid_for_mode(features, mode):
            continue
        return seed_mask
    return None


def _random_pattern(cell_size: int, rng: random.Random, used: set[bytes]) -> np.ndarray | None:
    area = cell_size * cell_size
    for _ in range(500):
        count = rng.randint(1, area - 1)
        indices = rng.sample(range(area), count)
        pattern = np.zeros((cell_size, cell_size), dtype=bool)
        for idx in indices:
            pattern[idx // cell_size, idx % cell_size] = True
        key = pattern.tobytes()
        if key in used:
            continue
        used.add(key)
        return pattern
    return None


def _seed_patterns(seed_mask: np.ndarray, cell_size: int, rng: random.Random) -> np.ndarray | None:
    patterns = np.zeros(seed_mask.shape + (cell_size, cell_size), dtype=bool)
    used: set[bytes] = set()
    for r, c in zip(*np.nonzero(seed_mask)):
        pattern = _random_pattern(cell_size, rng, used)
        if pattern is None:
            return None
        patterns[r, c] = pattern
    return patterns


def _simulate_completion(seed_mask: np.ndarray, seed_patterns: np.ndarray) -> np.ndarray | None:
    completed = _propagate_patterns(seed_patterns, seed_mask)
    for r, c in zip(*np.nonzero(~seed_mask)):
        if not completed[r, c].any():
            return None
    return completed


def _grid_lines(cell_count: int, cell_size: int) -> list[int]:
    return [idx * (cell_size + 1) for idx in range(cell_count + 1)]


def _render_from_seed_mask(seed_mask: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    n_rows, n_cols = seed_mask.shape
    cell_size = patterns.shape[-1]
    row_lines = _grid_lines(n_rows, cell_size)
    col_lines = _grid_lines(n_cols, cell_size)
    visible = np.zeros_like(patterns)
    visible[seed_mask] = patterns[seed_mask]
    shape = (row_lines[-1] + 1, col_lines[-1] + 1)
    return _render_patterns(
        shape,
        row_lines,
        col_lines,
        _BG_COLOR,
        _LINE_COLOR,
        _PATTERN_COLOR,
        visible,
    )


def _render_completed(patterns: np.ndarray) -> np.ndarray:
    n_rows, n_cols = patterns.shape[:2]
    cell_size = patterns.shape[-1]
    row_lines = _grid_lines(n_rows, cell_size)
    col_lines = _grid_lines(n_cols, cell_size)
    shape = (row_lines[-1] + 1, col_lines[-1] + 1)
    return _render_patterns(
        shape,
        row_lines,
        col_lines,
        _BG_COLOR,
        _LINE_COLOR,
        _PATTERN_COLOR,
        patterns,
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows, n_cols, cell_size = _choose_dimensions(rng)
    seed_mask = _sample_seed_mask(n_rows, n_cols, rng)
    if seed_mask is None:
        return None

    seed_patterns = _seed_patterns(seed_mask, cell_size, rng)
    if seed_patterns is None:
        return None

    completed = _simulate_completion(seed_mask, seed_patterns)
    if completed is None:
        return None

    inp = _render_from_seed_mask(seed_mask, seed_patterns)
    out = _render_completed(completed)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {_GEN_RETRIES} attempts")
