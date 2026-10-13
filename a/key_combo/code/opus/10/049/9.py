from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


BG = 0
A_COLORS = (1, 2)
B_COLORS = (3, 4)
_RNG = np.random.default_rng()

GRID_ROWS_RANGE = (11, 16)  # stop-exclusive for integers()
GRID_COLS_RANGE = (11, 16)
MAX_ATTEMPTS = 400


def _orbit_members(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    rows = {int(r), int(h - 1 - r)}
    cols = {int(c), int(w - 1 - c)}
    return tuple(sorted((rr, cc) for rr in rows for cc in cols))


def _restore_layer(grid: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    layer = np.zeros_like(grid)
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in range(h):
        for c in range(w):
            orbit = _orbit_members(h, w, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            values = {int(grid[rr, cc]) for rr, cc in orbit if int(grid[rr, cc]) in colors}
            if not values:
                continue
            assert len(values) == 1, "each reflection orbit must reveal a unique layer color"
            value = next(iter(values))
            for rr, cc in orbit:
                layer[rr, cc] = value

    return layer


def _overlay(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    out = bottom.copy()
    mask = top != BG
    out[mask] = top[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore two mutually occluded symmetric layers and redraw them in area order.

    Input structure:
    - The grid has black background and exactly two foreground color groups: A uses blue/red {1,2} and B uses green/yellow {3,4}.
    - Each group is an incomplete view of a larger pattern that is symmetric under both horizontal and vertical reflection about the whole-grid center.
    - Where the two complete layers overlap, the input shows only one of the two colors, so each group acts as an occluder for the other.

    Rule:
    - Reconstruct A by treating any visible A-colored cell as evidence that its full horizontal/vertical reflection orbit belongs to A, with one consistent A color across that orbit.
    - Reconstruct B in the same way from the visible B-colored cells.
    - Count the restored foreground area of both complete layers.
    - Draw the smaller layer first on black background, then draw the larger layer on top; if the areas tie, A is the top layer.

    Key insight:
    Each layer's symmetry can be inferred independently from its own visible reflection orbits, so both hidden shapes can be completed in parallel before the final area-based overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    a_layer = _restore_layer(g, A_COLORS)
    b_layer = _restore_layer(g, B_COLORS)
    a_area = int(np.count_nonzero(a_layer))
    b_area = int(np.count_nonzero(b_layer))

    if a_area >= b_area:
        return _overlay(b_layer, a_layer)
    return _overlay(a_layer, b_layer)


def _rep_neighbors(cell: tuple[int, int], rep_h: int, rep_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rep_h and 0 <= cc < rep_w:
            out.append((rr, cc))
    return out


def _grow_connected(
    rep_h: int,
    rep_w: int,
    count: int,
    *,
    seed_shape: set[tuple[int, int]] | None = None,
    blocked: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    count = int(count)
    blocked = set() if blocked is None else set(blocked)
    if seed_shape is None:
        available = [(r, c) for r in range(rep_h) for c in range(rep_w) if (r, c) not in blocked]
        if not available:
            return None
        start = available[int(_RNG.integers(0, len(available)))]
        shape = {start}
        target = count
    else:
        shape = set(seed_shape)
        target = len(shape) + count
        if shape & blocked:
            return None

    while len(shape) < target:
        frontier = sorted(
            {
                nbr
                for cell in shape
                for nbr in _rep_neighbors(cell, rep_h, rep_w)
                if nbr not in shape and nbr not in blocked
            }
        )
        if not frontier:
            return None
        pick = frontier[int(_RNG.integers(0, len(frontier)))]
        shape.add(pick)
    return shape


def _grow_within_allowed(
    allowed: set[tuple[int, int]], target_size: int, rep_h: int, rep_w: int
) -> set[tuple[int, int]] | None:
    allowed = set(allowed)
    target_size = int(target_size)
    if not 1 <= target_size <= len(allowed):
        return None

    ordered = sorted(allowed)
    shape = {ordered[int(_RNG.integers(0, len(ordered)))]}
    while len(shape) < target_size:
        frontier = sorted(
            {
                nbr
                for cell in shape
                for nbr in _rep_neighbors(cell, rep_h, rep_w)
                if nbr in allowed and nbr not in shape
            }
        )
        if not frontier:
            return None
        shape.add(frontier[int(_RNG.integers(0, len(frontier)))])
    return shape


def _choose_branch() -> tuple[str, int, int, int, str]:
    branch = str(
        _RNG.choice(
            [
                "a_top_same_visible",
                "a_top_opposite_visible",
                "b_top_same_visible",
                "b_top_opposite_visible",
                "tie",
            ]
        )
    )

    if branch == "a_top_same_visible":
        shared = int(_RNG.integers(2, 4))
        b_only = int(_RNG.integers(3, 6))
        a_only = b_only + int(_RNG.integers(1, 3))
        return branch, shared, a_only, b_only, "A"
    if branch == "a_top_opposite_visible":
        shared = 3
        b_only = int(_RNG.integers(4, 6))
        a_only = b_only + 1
        return branch, shared, a_only, b_only, "B"
    if branch == "b_top_same_visible":
        shared = int(_RNG.integers(2, 4))
        a_only = int(_RNG.integers(3, 6))
        b_only = a_only + int(_RNG.integers(1, 3))
        return branch, shared, a_only, b_only, "B"
    if branch == "b_top_opposite_visible":
        shared = 3
        a_only = int(_RNG.integers(4, 6))
        b_only = a_only + 1
        return branch, shared, a_only, b_only, "A"

    shared = int(_RNG.integers(2, 4))
    exclusive = int(_RNG.integers(4, 6))
    visible_winner = str(_RNG.choice(["A", "B"]))
    return branch, shared, exclusive, exclusive, visible_winner


def _assign_layer_colors(
    shape: set[tuple[int, int]], palette: tuple[int, int], rep_h: int, rep_w: int
) -> dict[tuple[int, int], int] | None:
    if len(shape) < 2:
        return None
    color_a = palette[0]
    color_b = palette[1]

    subset_size = int(_RNG.integers(1, len(shape)))
    subset = _grow_within_allowed(shape, subset_size, rep_h, rep_w)
    if not subset or subset == shape:
        return None

    color_map = {cell: color_b for cell in shape}
    for cell in subset:
        color_map[cell] = color_a
    if set(color_map.values()) != {color_a, color_b}:
        return None
    return color_map


def _render_layer(
    h: int, w: int, shape: set[tuple[int, int]], color_map: dict[tuple[int, int], int]
) -> np.ndarray:
    layer = np.zeros((h, w), dtype=int)
    for rep in shape:
        color = int(color_map[rep])
        for rr, cc in _orbit_members(h, w, rep[0], rep[1]):
            layer[rr, cc] = color
    return layer


def _choose_shared_visibility(
    shared_count: int, size_a: int, size_b: int, visible_winner: str
) -> tuple[int, ...] | None:
    overlap = 4 * shared_count
    candidates: list[tuple[int, ...]] = []
    for ks in product((1, 2, 3), repeat=shared_count):
        shown_a = int(sum(ks))
        visible_a = int(size_a - overlap + shown_a)
        visible_b = int(size_b - shown_a)
        if 10 * visible_a < 7 * size_a or 10 * visible_b < 7 * size_b:
            continue
        if visible_winner == "A" and visible_a <= visible_b:
            continue
        if visible_winner == "B" and visible_b <= visible_a:
            continue
        candidates.append(tuple(int(k) for k in ks))
    if not candidates:
        return None
    return candidates[int(_RNG.integers(0, len(candidates)))]


def _paint_input(
    h: int,
    w: int,
    a_shape: set[tuple[int, int]],
    b_shape: set[tuple[int, int]],
    a_colors: dict[tuple[int, int], int],
    b_colors: dict[tuple[int, int], int],
    shared_ks: tuple[int, ...],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    shared = sorted(a_shape & b_shape)
    for rep in sorted(a_shape | b_shape):
        orbit = list(_orbit_members(h, w, rep[0], rep[1]))
        if rep in a_shape and rep not in b_shape:
            for rr, cc in orbit:
                grid[rr, cc] = int(a_colors[rep])
            continue
        if rep in b_shape and rep not in a_shape:
            for rr, cc in orbit:
                grid[rr, cc] = int(b_colors[rep])
            continue

        idx = shared.index(rep)
        show_a = int(shared_ks[idx])
        order = orbit.copy()
        _RNG.shuffle(order)
        visible_a_cells = set(order[:show_a])
        for rr, cc in orbit:
            grid[rr, cc] = int(a_colors[rep] if (rr, cc) in visible_a_cells else b_colors[rep])
    return grid


def _expected_output(a_layer: np.ndarray, b_layer: np.ndarray) -> np.ndarray:
    a_area = int(np.count_nonzero(a_layer))
    b_area = int(np.count_nonzero(b_layer))
    return _overlay(b_layer, a_layer) if a_area >= b_area else _overlay(a_layer, b_layer)


def _visible_count(grid: np.ndarray, colors: Iterable[int]) -> int:
    return int(np.isin(grid, list(colors)).sum())


def _validate_instance(
    grid: np.ndarray,
    a_layer: np.ndarray,
    b_layer: np.ndarray,
    branch: str,
    visible_winner: str,
    shared_count: int,
) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    if not np.array_equal(a_layer, a_layer[::-1, :]) or not np.array_equal(a_layer, a_layer[:, ::-1]):
        return False
    if not np.array_equal(b_layer, b_layer[::-1, :]) or not np.array_equal(b_layer, b_layer[:, ::-1]):
        return False

    a_area = int(np.count_nonzero(a_layer))
    b_area = int(np.count_nonzero(b_layer))
    if shared_count == 0:
        return False
    if np.count_nonzero((a_layer != 0) & (b_layer == 0)) == 0:
        return False
    if np.count_nonzero((b_layer != 0) & (a_layer == 0)) == 0:
        return False
    if np.array_equal(grid, _expected_output(a_layer, b_layer)):
        return False

    visible_a = _visible_count(grid, A_COLORS)
    visible_b = _visible_count(grid, B_COLORS)
    if 10 * visible_a < 7 * a_area or 10 * visible_b < 7 * b_area:
        return False
    if visible_winner == "A" and visible_a <= visible_b:
        return False
    if visible_winner == "B" and visible_b <= visible_a:
        return False

    restored_a = _restore_layer(grid, A_COLORS)
    restored_b = _restore_layer(grid, B_COLORS)
    if not np.array_equal(restored_a, a_layer) or not np.array_equal(restored_b, b_layer):
        return False
    if not np.array_equal(main(grid), _expected_output(a_layer, b_layer)):
        return False

    if branch.startswith("a_top") and not (a_area > b_area):
        return False
    if branch.startswith("b_top") and not (b_area > a_area):
        return False
    if branch == "tie" and a_area != b_area:
        return False

    shared_mask = (a_layer != 0) & (b_layer != 0)
    both_visible = 0
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r, c in np.argwhere(shared_mask):
        orbit = _orbit_members(h, w, int(r), int(c))
        if orbit in seen:
            continue
        seen.add(orbit)
        vals = {int(grid[rr, cc]) for rr, cc in orbit}
        if any(v in A_COLORS for v in vals) and any(v in B_COLORS for v in vals):
            both_visible += 1
    return both_visible == shared_count


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_ROWS_RANGE))
    w = int(_RNG.integers(*GRID_COLS_RANGE))
    rep_h, rep_w = h // 2, w // 2
    if rep_h * rep_w < 14:
        return None

    branch, shared_count, a_only_count, b_only_count, visible_winner = _choose_branch()
    if shared_count + a_only_count + b_only_count > rep_h * rep_w:
        return None

    shared_shape = _grow_connected(rep_h, rep_w, shared_count)
    if shared_shape is None:
        return None

    first_is_a = a_only_count >= b_only_count
    if first_is_a:
        a_shape = _grow_connected(rep_h, rep_w, a_only_count, seed_shape=shared_shape)
        if a_shape is None:
            return None
        b_shape = _grow_connected(rep_h, rep_w, b_only_count, seed_shape=shared_shape, blocked=a_shape - shared_shape)
        if b_shape is None:
            return None
    else:
        b_shape = _grow_connected(rep_h, rep_w, b_only_count, seed_shape=shared_shape)
        if b_shape is None:
            return None
        a_shape = _grow_connected(rep_h, rep_w, a_only_count, seed_shape=shared_shape, blocked=b_shape - shared_shape)
        if a_shape is None:
            return None

    a_color_map = _assign_layer_colors(a_shape, A_COLORS, rep_h, rep_w)
    b_color_map = _assign_layer_colors(b_shape, B_COLORS, rep_h, rep_w)
    if a_color_map is None or b_color_map is None:
        return None

    a_layer = _render_layer(h, w, a_shape, a_color_map)
    b_layer = _render_layer(h, w, b_shape, b_color_map)
    shared_ks = _choose_shared_visibility(shared_count, int(np.count_nonzero(a_layer)), int(np.count_nonzero(b_layer)), visible_winner)
    if shared_ks is None:
        return None

    grid = _paint_input(h, w, a_shape, b_shape, a_color_map, b_color_map, shared_ks)
    if not _validate_instance(grid, a_layer, b_layer, branch, visible_winner, shared_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mutually occluded symmetric puzzle")
