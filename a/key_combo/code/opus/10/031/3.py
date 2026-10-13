from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MIN_SIDE = 4
MAX_SIDE = 5
MIN_FOREGROUND = 8
MAX_FOREGROUND = 12
MIN_COLOR_COUNT = 3
MAX_GENERATION_ATTEMPTS = 2000


def _color_cells(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(grid == int(color))}


def _mirror_column(k: int, c: int) -> int:
    return int(k) - int(c)


def _exact_vertical_axes(cells: set[tuple[int, int]], width: int) -> list[int]:
    if not cells:
        return []

    axes: list[int] = []
    for k in range(0, 2 * (int(width) - 1) + 1):
        ok = True
        for r, c in cells:
            cc = _mirror_column(k, c)
            if not (0 <= cc < width and (r, cc) in cells):
                ok = False
                break
        if ok:
            axes.append(int(k))
    return axes


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND)


def _classify_colors(grid: np.ndarray) -> tuple[int, int, int]:
    colors = _foreground_colors(grid)
    assert len(colors) == 2, "expected exactly two non-background colors"

    symmetric: list[tuple[int, int]] = []
    asymmetric: list[int] = []
    for color in colors:
        axes = _exact_vertical_axes(_color_cells(grid, color), grid.shape[1])
        if len(axes) == 1:
            symmetric.append((color, axes[0]))
        elif len(axes) == 0:
            asymmetric.append(color)
        else:
            raise AssertionError("symmetric color axis must be unique")

    assert len(symmetric) == 1 and len(asymmetric) == 1, "expected one symmetric and one asymmetric color"
    return symmetric[0][0], asymmetric[0], symmetric[0][1]


def _stamp_output(grid: np.ndarray, mirrored_colors: set[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mirrored = np.fliplr(g)
    out = np.zeros((h * h, w * w), dtype=int)

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BACKGROUND:
                continue
            patch = mirrored if color in mirrored_colors else g
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = patch
    return out


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _is_4_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _frontier(cells: set[tuple[int, int]], shape: tuple[int, int], blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) not in blocked:
                out.add((nr, nc))
    return out


def _cross_color_contacts(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                count += 1
    return count


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _row_col_span_ok(cells: set[tuple[int, int]]) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return len(rows) >= 2 and len(cols) >= 2


def _position_rule_output(grid: np.ndarray, axis_k: int, mode: str) -> np.ndarray:
    mirror_colors: set[int] = set()
    h, w = grid.shape
    mirrored = np.fliplr(grid)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND:
                continue
            left = 2 * c < axis_k
            right = 2 * c > axis_k
            on_or_left = 2 * c <= axis_k
            on_or_right = 2 * c >= axis_k
            use_mirror = (
                (mode == "left" and left)
                or (mode == "right" and right)
                or (mode == "on_or_left" and on_or_left)
                or (mode == "on_or_right" and on_or_right)
            )
            patch = mirrored if use_mirror else grid
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = patch
    return out


def _semantic_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        return False
    if max(h * h, w * w) > 30:
        return False

    colors = _foreground_colors(grid)
    if len(colors) != 2:
        return False

    counts = {color: int(np.sum(grid == color)) for color in colors}
    if any(count < MIN_COLOR_COUNT for count in counts.values()):
        return False
    if not (MIN_FOREGROUND <= sum(counts.values()) <= MAX_FOREGROUND):
        return False

    foreground = set().union(*(_color_cells(grid, color) for color in colors))
    if not _is_4_connected(foreground):
        return False
    if np.array_equal(grid, np.fliplr(grid)):
        return False

    try:
        symmetric_color, asymmetric_color, axis_k = _classify_colors(grid)
    except AssertionError:
        return False

    sym_cells = _color_cells(grid, symmetric_color)
    asym_cells = _color_cells(grid, asymmetric_color)
    if not (_row_col_span_ok(sym_cells) and _row_col_span_ok(asym_cells)):
        return False
    if _cross_color_contacts(sym_cells, asym_cells) < 2:
        return False

    if not any(2 * c < axis_k for _, c in asym_cells):
        return False
    if not any(2 * c > axis_k for _, c in asym_cells):
        return False

    sr0, sc0, sr1, sc1 = _bbox(sym_cells)
    ar0, ac0, ar1, ac1 = _bbox(asym_cells)
    row_overlap = min(sr1, ar1) - max(sr0, ar0) + 1
    col_overlap = min(sc1, ac1) - max(sc0, ac0) + 1
    if row_overlap <= 0 or col_overlap <= 0:
        return False

    truth = _stamp_output(grid, {symmetric_color})
    if np.array_equal(truth, _stamp_output(grid, set())):
        return False
    if np.array_equal(truth, _stamp_output(grid, {symmetric_color, asymmetric_color})):
        return False
    if np.array_equal(truth, _stamp_output(grid, {asymmetric_color})):
        return False
    for mode in ("left", "right", "on_or_left", "on_or_right"):
        if np.array_equal(truth, _position_rule_output(grid, axis_k, mode)):
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, truth)


def _orbits_for_axis(shape: tuple[int, int], axis_k: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = shape
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            cc = _mirror_column(axis_k, c)
            if not (0 <= cc < w):
                continue
            orbit = tuple(sorted({(r, c), (r, cc)}))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _build_symmetric_cells(shape: tuple[int, int], target_count: int) -> tuple[set[tuple[int, int]], int] | None:
    h, w = shape
    candidate_axes = list(range(1, 2 * w - 2))
    random.shuffle(candidate_axes)

    for axis_k in candidate_axes:
        orbits = _orbits_for_axis(shape, axis_k)
        pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
        center_orbits = [orbit for orbit in orbits if len(orbit) == 1]
        if not pair_orbits:
            continue

        for _ in range(200):
            cells: set[tuple[int, int]] = set()
            pair = random.choice(pair_orbits)
            cells.update(pair)
            remaining = target_count - len(cells)
            available_pairs = [orbit for orbit in pair_orbits if orbit != pair]
            available_centers = center_orbits[:]
            random.shuffle(available_pairs)
            random.shuffle(available_centers)

            if remaining < 0:
                continue
            if remaining % 2 == 1:
                if not available_centers:
                    continue
                center = available_centers.pop()
                cells.update(center)
                remaining -= 1

            if remaining < 0:
                continue
            if remaining > 2 * len(available_pairs):
                continue

            while remaining > 0 and available_pairs:
                orbit = available_pairs.pop()
                cells.update(orbit)
                remaining -= 2

            if remaining != 0:
                continue
            if not _row_col_span_ok(cells):
                continue
            axes = _exact_vertical_axes(cells, w)
            if axes != [axis_k]:
                continue
            return cells, axis_k
    return None


def _build_asymmetric_cells(
    shape: tuple[int, int],
    symmetric_cells: set[tuple[int, int]],
    axis_k: int,
    target_count: int,
) -> set[tuple[int, int]] | None:
    h, w = shape
    blocked = set(symmetric_cells)
    left_seed_candidates = [
        cell for cell in _frontier(symmetric_cells, shape, blocked) if 2 * cell[1] < axis_k
    ]
    right_seed_candidates = [
        cell for cell in _frontier(symmetric_cells, shape, blocked) if 2 * cell[1] > axis_k
    ]
    if not left_seed_candidates or not right_seed_candidates:
        return None

    for _ in range(300):
        asym = {
            random.choice(left_seed_candidates),
            random.choice(right_seed_candidates),
        }
        blocked_now = blocked | asym
        while len(asym) < target_count:
            frontier = _frontier(symmetric_cells | asym, shape, blocked_now)
            if not frontier:
                break
            preferred = [
                cell
                for cell in frontier
                if any(nei in symmetric_cells for nei in _neighbors4(*cell))
            ]
            choice_pool = preferred if preferred and random.random() < 0.7 else list(frontier)
            cell = random.choice(choice_pool)
            asym.add(cell)
            blocked_now.add(cell)

        if len(asym) != target_count:
            continue
        if _exact_vertical_axes(asym, w):
            continue
        if not _row_col_span_ok(asym):
            continue
        if _cross_color_contacts(symmetric_cells, asym) < 2:
            continue
        if not _is_4_connected(symmetric_cells | asym):
            continue
        return asym
    return None


def _make_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)

    colors = random.sample(NON_BLACK_COLORS, 2)
    symmetric_color = random.choice(colors)
    asymmetric_color = colors[1] if colors[0] == symmetric_color else colors[0]

    total = random.randint(MIN_FOREGROUND, MAX_FOREGROUND)
    symmetric_count = random.randint(MIN_COLOR_COUNT, min(6, total - MIN_COLOR_COUNT))
    asymmetric_count = total - symmetric_count

    built = _build_symmetric_cells((h, w), symmetric_count)
    if built is None:
        return None
    symmetric_cells, axis_k = built

    asymmetric_cells = _build_asymmetric_cells((h, w), symmetric_cells, axis_k, asymmetric_count)
    if asymmetric_cells is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in symmetric_cells:
        grid[r, c] = symmetric_color
    for r, c in asymmetric_cells:
        grid[r, c] = asymmetric_color

    if random.random() < 0.5:
        grid = np.flipud(grid).copy()
    if random.random() < 0.5:
        grid = np.fliplr(grid).copy()

    if not _semantic_ok(grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the sprite into a block grid, mirroring the stamp only for pixels whose color layer is vertically symmetric.

    Input structure:
    - The input is a small black-background sprite using exactly two non-black colors.
    - One color's own pixel set is left-right symmetric about a unique vertical axis; the other color's set is not.
    - The whole input sprite is the stamp template for every non-black source pixel.

    Rule:
    - Determine, for each non-black color independently, whether that color layer is exactly vertically symmetric.
    - Build an output of shape (H*H, W*W) and partition it into HxW blocks matching the input size.
    - A black input pixel maps to an all-black block.
    - A pixel of the asymmetric color stamps an unmodified copy of the full input sprite into its block.
    - A pixel of the symmetric color stamps a left-right mirrored copy of the full input sprite into its block.

    Key insight:
    - The deciding property is not the source pixel's position but the symmetry status of its color layer, so the solver must first classify colors by per-layer vertical symmetry before choosing each block's stamp orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    symmetric_color, _asymmetric_color, _axis_k = _classify_colors(g)
    return _stamp_output(g, {symmetric_color})


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
