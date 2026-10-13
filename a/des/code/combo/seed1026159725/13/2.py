import math
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
CELL_SIZES = (3, 5, 7)
GRID_FACTORS = (
    (2, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
    (5, 6),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _divisors(n: int) -> list[int]:
    out: set[int] = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            out.add(d)
            out.add(n // d)
    return sorted(out)


def _vote_pattern(grid: np.ndarray, cell_size: int, bg: int) -> tuple[np.ndarray, int]:
    h, w = grid.shape
    pattern = np.full((cell_size, cell_size), bg, dtype=int)
    mismatches = 0

    for r in range(cell_size):
        for c in range(cell_size):
            samples = grid[r:h:cell_size, c:w:cell_size].ravel()
            non_bg = samples[samples != bg]
            if non_bg.size:
                values, counts = np.unique(non_bg, return_counts=True)
                winner = int(values[np.argmax(counts)])
            else:
                winner = int(bg)
            pattern[r, c] = winner
            mismatches += int(np.sum(samples != winner))

    return pattern, mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all equal square cells and merge their local observations by non-background majority vote.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    bg = _background_color(g)

    candidates: list[tuple[int, int, np.ndarray]] = []
    for cell_size in _divisors(math.gcd(h, w)):
        if cell_size < 2:
            continue
        if h % cell_size or w % cell_size:
            continue
        if h // cell_size < 2 or w // cell_size < 2:
            continue
        pattern, mismatches = _vote_pattern(g, cell_size, bg)
        candidates.append((mismatches, cell_size, pattern))

    assert candidates, "expected at least one valid square tiling"
    _, _, best = min(candidates, key=lambda item: (item[0], item[1]))
    return best


def _paint_rect(canvas: np.ndarray, color: int) -> None:
    side = canvas.shape[0]
    h = int(np.random.randint(1, side + 1))
    w = int(np.random.randint(1, side + 1))
    r0 = int(np.random.randint(0, side - h + 1))
    c0 = int(np.random.randint(0, side - w + 1))
    canvas[r0 : r0 + h, c0 : c0 + w] = color


def _paint_hline(canvas: np.ndarray, color: int) -> None:
    side = canvas.shape[0]
    r = int(np.random.randint(0, side))
    c0 = int(np.random.randint(0, side))
    c1 = int(np.random.randint(c0 + 1, side + 1))
    canvas[r, c0:c1] = color


def _paint_vline(canvas: np.ndarray, color: int) -> None:
    side = canvas.shape[0]
    c = int(np.random.randint(0, side))
    r0 = int(np.random.randint(0, side))
    r1 = int(np.random.randint(r0 + 1, side + 1))
    canvas[r0:r1, c] = color


def _paint_diag(canvas: np.ndarray, color: int, anti: bool) -> None:
    side = canvas.shape[0]
    length = int(np.random.randint(2, side + 1))
    r0 = int(np.random.randint(0, side - length + 1))
    c0 = int(np.random.randint(0, side - length + 1))
    for k in range(length):
        rr = r0 + k
        cc = c0 + (length - 1 - k if anti else k)
        canvas[rr, cc] = color


def _paint_plus(canvas: np.ndarray, color: int) -> None:
    side = canvas.shape[0]
    r = int(np.random.randint(0, side))
    c = int(np.random.randint(0, side))
    arm_r = int(np.random.randint(1, side))
    arm_c = int(np.random.randint(1, side))
    r0 = max(0, r - arm_r)
    r1 = min(side, r + arm_r + 1)
    c0 = max(0, c - arm_c)
    c1 = min(side, c + arm_c + 1)
    canvas[r0:r1, c] = color
    canvas[r, c0:c1] = color


def _paint_l(canvas: np.ndarray, color: int) -> None:
    side = canvas.shape[0]
    h = int(np.random.randint(2, side + 1))
    w = int(np.random.randint(2, side + 1))
    r0 = int(np.random.randint(0, side - h + 1))
    c0 = int(np.random.randint(0, side - w + 1))
    if bool(np.random.randint(0, 2)):
        canvas[r0 : r0 + h, c0] = color
        canvas[r0 + h - 1, c0 : c0 + w] = color
    else:
        canvas[r0, c0 : c0 + w] = color
        canvas[r0 : r0 + h, c0 + w - 1] = color


def _make_target_pattern(cell_size: int, bg: int, fg_colors: list[int]) -> np.ndarray | None:
    painters = (
        _paint_rect,
        _paint_hline,
        _paint_vline,
        lambda canvas, color: _paint_diag(canvas, color, anti=False),
        lambda canvas, color: _paint_diag(canvas, color, anti=True),
        _paint_plus,
        _paint_l,
    )

    for _ in range(100):
        canvas = np.full((cell_size, cell_size), bg, dtype=int)
        steps = int(np.random.randint(3, 6))
        used_colors: list[int] = []
        for _step in range(steps):
            color = int(fg_colors[np.random.randint(0, len(fg_colors))])
            used_colors.append(color)
            painter = painters[int(np.random.randint(0, len(painters)))]
            painter(canvas, color)

        fg_mask = canvas != bg
        fg_count = int(fg_mask.sum())
        visible_colors = {int(v) for v in np.unique(canvas) if int(v) != bg}
        if len(visible_colors) < 2:
            continue
        if fg_count < max(4, cell_size):
            continue
        if fg_count > int(cell_size * cell_size * 0.75):
            continue
        if int((~fg_mask).sum()) < 2:
            continue
        if np.unique(fg_mask.any(axis=1)).size == 1 and np.unique(fg_mask.any(axis=0)).size == 1:
            return canvas
        return canvas
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(np.random.choice(range(1, total), size=parts - 1, replace=False).tolist())
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _fill_observations(target: np.ndarray, bg: int, fg_colors: list[int], n_cells: int) -> tuple[np.ndarray, bool]:
    cell_size = target.shape[0]
    cells = np.full((n_cells, cell_size, cell_size), bg, dtype=int)

    fg_positions = [(int(r), int(c)) for r, c in np.argwhere(target != bg)]
    if len(fg_positions) < 2:
        return cells, False

    order = np.random.permutation(len(fg_positions))
    n_conflict = int(np.random.randint(1, len(fg_positions)))
    conflict_positions = {fg_positions[int(i)] for i in order[:n_conflict]}

    had_stable = False
    had_conflict = False
    had_alt_noise = False
    had_bg_omission = False

    cell_indices = np.arange(n_cells)
    for r in range(cell_size):
        for c in range(cell_size):
            winner = int(target[r, c])
            if winner == bg:
                continue

            shuffled = np.random.permutation(cell_indices)
            if (r, c) in conflict_positions:
                had_conflict = True
                winner_count = int(np.random.randint(n_cells // 2 + 1, n_cells))
                remaining = n_cells - winner_count
                alt_total = int(np.random.randint(1, remaining + 1))
                alt_colors = [color for color in fg_colors if int(color) != winner]
                alt_parts = int(np.random.randint(1, min(len(alt_colors), alt_total) + 1))
                alt_counts = _random_composition(alt_total, alt_parts)
                alt_pick = np.random.choice(alt_colors, size=alt_parts, replace=False)

                winner_cells = shuffled[:winner_count]
                cells[winner_cells, r, c] = winner

                cursor = winner_count
                for alt_color, alt_count in zip(alt_pick, alt_counts):
                    dest = shuffled[cursor : cursor + alt_count]
                    cells[dest, r, c] = int(alt_color)
                    cursor += alt_count
                if alt_total > 0:
                    had_alt_noise = True
                if cursor < n_cells:
                    had_bg_omission = True
            else:
                had_stable = True
                min_count = max(1, (n_cells + 2) // 3)
                winner_count = int(np.random.randint(min_count, n_cells + 1))
                winner_cells = shuffled[:winner_count]
                cells[winner_cells, r, c] = winner
                if winner_count < n_cells:
                    had_bg_omission = True

    if not (had_stable and had_conflict and had_alt_noise and had_bg_omission):
        return cells, False

    blank_cells = np.where(np.all(cells == bg, axis=(1, 2)))[0]
    if blank_cells.size:
        stable_positions = [pos for pos in fg_positions if pos not in conflict_positions]
        if not stable_positions:
            return cells, False
        rr, cc = stable_positions[0]
        for idx in blank_cells:
            cells[int(idx), rr, cc] = int(target[rr, cc])

    return cells, True


def _cells_to_grid(cells: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    n_cells, cell_size, _ = cells.shape
    assert n_cells == n_rows * n_cols
    rows = []
    cursor = 0
    for _ in range(n_rows):
        rows.append(np.hstack([cells[cursor + j] for j in range(n_cols)]))
        cursor += n_cols
    return np.vstack(rows)


def _gen_attempt() -> np.ndarray | None:
    cell_size = int(CELL_SIZES[np.random.randint(0, len(CELL_SIZES))])
    n_rows, n_cols = GRID_FACTORS[int(np.random.randint(0, len(GRID_FACTORS)))]
    if bool(np.random.randint(0, 2)):
        n_rows, n_cols = n_cols, n_rows

    n_cells = n_rows * n_cols
    bg = int(np.random.randint(0, 10))
    fg_pool = [color for color in range(10) if color != bg]
    n_fg_colors = int(np.random.randint(2, min(5, len(fg_pool)) + 1))
    fg_colors = [int(v) for v in np.random.choice(fg_pool, size=n_fg_colors, replace=False)]

    target = _make_target_pattern(cell_size, bg, fg_colors)
    if target is None:
        return None

    cells, ok = _fill_observations(target, bg, fg_colors, n_cells)
    if not ok:
        return None

    if len({cells[i].tobytes() for i in range(n_cells)}) < 2:
        return None

    grid = _cells_to_grid(cells, n_rows, n_cols)
    recovered = main(grid)
    if recovered.shape != target.shape:
        return None
    if not np.array_equal(recovered, target):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
