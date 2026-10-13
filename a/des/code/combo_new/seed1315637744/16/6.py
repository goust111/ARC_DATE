from __future__ import annotations

from collections import Counter
import numpy as np


WALL_COLOR = 0
RECEIVER_COLOR = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()

GRID_H_RANGE = (14, 27)
GRID_W_RANGE = (21, 30)
CORE_GAP = 2
EXTRA_TOWERS_MAX = 2
EXTRA_RECEIVERS_MAX = 4
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sample_spaced_positions(count: int, lo: int, hi: int, min_gap: int) -> list[int]:
    span = hi - lo + 1
    need = count + (count - 1) * (min_gap - 1)
    if need > span:
        raise ValueError("range too small for requested spacing")

    compressed = span - (min_gap - 1) * (count - 1)
    picks = np.sort(_RNG.choice(np.arange(compressed), size=count, replace=False))
    return [int(lo + picks[i] + i * (min_gap - 1)) for i in range(count)]


def _apply_symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if code >= 4:
        out = out.T
    if code & 1:
        out = np.flipud(out)
    if code & 2:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _trace_signals(
    grid: np.ndarray,
    bg: int | None = None,
) -> tuple[dict[tuple[int, int], set[int]], Counter[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g) if bg is None else int(bg)
    height, width = g.shape

    tower_mask = (g != bg) & (g != WALL_COLOR) & (g != RECEIVER_COLOR)
    tower_positions = np.argwhere(tower_mask)
    tower_counts: Counter[int] = Counter(int(g[r, c]) for r, c in tower_positions)

    receivers = [tuple(map(int, coord)) for coord in np.argwhere(g == RECEIVER_COLOR)]
    received: dict[tuple[int, int], set[int]] = {coord: set() for coord in receivers}

    for r, c in tower_positions:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            while 0 <= rr < height and 0 <= cc < width:
                if int(g[rr, cc]) == WALL_COLOR or bool(tower_mask[rr, cc]):
                    break
                if int(g[rr, cc]) == RECEIVER_COLOR:
                    received[(int(rr), int(cc))].add(color)
                rr += dr
                cc += dc

    return received, tower_counts


def _resolved_receiver_color(received_colors: set[int], tower_counts: Counter[int]) -> int:
    if not received_colors:
        return RECEIVER_COLOR
    if len(received_colors) == 1:
        return next(iter(received_colors))

    best = max(int(tower_counts[color]) for color in received_colors)
    winners = [int(color) for color in received_colors if int(tower_counts[color]) == best]
    return RECEIVER_COLOR if len(winners) != 1 else int(winners[0])


def _receiver_category(received_colors: set[int], tower_counts: Counter[int]) -> str:
    if not received_colors:
        return "none"
    if len(received_colors) == 1:
        return "single"

    best = max(int(tower_counts[color]) for color in received_colors)
    winners = [int(color) for color in received_colors if int(tower_counts[color]) == best]
    return "multi_tie" if len(winners) != 1 else "multi_unique"


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    received, tower_counts = _trace_signals(g, bg=bg)

    out = g.copy()
    for (r, c), colors in received.items():
        out[r, c] = _resolved_receiver_color(colors, tower_counts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey receivers by tracing tower broadcasts through wall-bounded corridors.

    Input structure:
    - A dominant background contains black wall pixels, several colored one-cell signal towers, and grey one-cell receivers.
    - Towers and receivers sit on open cells; walls act as opaque blockers.
    - Each tower color may appear once or twice globally, and multiple receivers can share the same tower rays.

    Rule:
    - From every tower, cast a straight ray up, down, left, and right.
    - A ray is invisible and leaves background cells unchanged; it stops at the grid edge, at a black wall, or immediately before another tower.
    - Each grey receiver collects every tower color whose ray passes through it.
    - A receiver with no colors stays grey; with one color it becomes that color; with multiple colors it becomes the received color whose tower count is largest in the whole grid, and any remaining tie stays grey.

    Key insight:
    - Receiver colors depend on two interacting cues: local line-of-sight through walls and the global multiplicity of each tower color, which breaks multi-color ties.
    """
    return _solve(np.asarray(grid, dtype=int))


def _add_extra_towers(
    grid: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    protected_rows: set[int],
    protected_cols: set[int],
    colors: list[int],
) -> None:
    height, width = grid.shape
    candidate_rows = [r for r in range(1, height - 1) if r not in used_rows and r not in protected_rows]
    candidate_cols = [c for c in range(1, width - 1) if c not in used_cols and c not in protected_cols]
    if not candidate_rows or not candidate_cols or not colors:
        return

    tower_count = int(_RNG.integers(0, min(EXTRA_TOWERS_MAX, len(colors), len(candidate_rows), len(candidate_cols)) + 1))
    if tower_count == 0:
        return

    chosen_colors = list(_RNG.choice(np.array(colors, dtype=int), size=tower_count, replace=False))
    row_order = list(_RNG.permutation(candidate_rows))[:tower_count]
    col_order = list(_RNG.permutation(candidate_cols))[:tower_count]

    for color, r, c in zip(chosen_colors, row_order, col_order):
        grid[int(r), int(c)] = int(color)
        used_rows.add(int(r))
        used_cols.add(int(c))


def _add_extra_receivers(grid: np.ndarray, tower_rows: set[int], tower_cols: set[int]) -> None:
    height, width = grid.shape
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if int(grid[r, c]) == _get_bg(grid) and r not in tower_rows and c not in tower_cols
    ]
    if not candidates:
        return

    receiver_count = int(_RNG.integers(0, min(EXTRA_RECEIVERS_MAX, len(candidates)) + 1))
    if receiver_count == 0:
        return

    picks = _RNG.choice(len(candidates), size=receiver_count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = candidates[int(idx)]
        grid[int(r), int(c)] = RECEIVER_COLOR


def _core_branch_set(grid: np.ndarray) -> set[str]:
    bg = _get_bg(grid)
    received, tower_counts = _trace_signals(grid, bg=bg)
    return {_receiver_category(colors, tower_counts) for colors in received.values()}


def _build_scene() -> np.ndarray | None:
    height = int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    bg, color_a, color_b, color_c = map(
        int,
        _RNG.choice(
            np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int),
            size=4,
            replace=False,
        ),
    )
    extra_palette = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in {bg, color_a, color_b, color_c}]

    try:
        row_tie, row_unique, row_blocked, row_single = _sample_spaced_positions(4, 1, height - 2, CORE_GAP)
        col_a1, col_tie, col_b1, col_a2, col_unique, col_c1, col_b2, col_blocked = _sample_spaced_positions(
            8,
            1,
            width - 2,
            CORE_GAP,
        )
    except ValueError:
        return None

    grid = np.full((height, width), bg, dtype=int)

    core_towers = {
        (row_tie, col_a1): color_a,
        (row_tie, col_b1): color_b,
        (row_unique, col_a2): color_a,
        (row_unique, col_c1): color_c,
        (row_blocked, col_b2): color_b,
    }
    core_receivers = {
        (row_tie, col_tie),
        (row_unique, col_unique),
        (row_blocked, col_blocked),
        (row_single, col_a1),
    }

    for (r, c), color in core_towers.items():
        grid[int(r), int(c)] = int(color)
    for r, c in core_receivers:
        grid[int(r), int(c)] = RECEIVER_COLOR

    grid[row_blocked, col_b2 + 1 : col_blocked] = WALL_COLOR

    tower_rows = {int(r) for r, _c in core_towers}
    tower_cols = {int(c) for _r, c in core_towers}
    receiver_rows = {int(r) for r, _c in core_receivers}
    receiver_cols = {int(c) for _r, c in core_receivers}

    _add_extra_towers(
        grid,
        used_rows=tower_rows,
        used_cols=tower_cols,
        protected_rows=receiver_rows,
        protected_cols=receiver_cols,
        colors=extra_palette,
    )
    _add_extra_receivers(grid, tower_rows=tower_rows, tower_cols=tower_cols)

    symmetry = int(_RNG.integers(0, 8))
    grid = _apply_symmetry(grid, symmetry)

    bg_after = _get_bg(grid)
    tower_mask = (grid != bg_after) & (grid != WALL_COLOR) & (grid != RECEIVER_COLOR)
    tower_counts = Counter(int(grid[r, c]) for r, c in np.argwhere(tower_mask))
    if not tower_counts or max(tower_counts.values()) > 2:
        return None

    if _core_branch_set(grid) != {"none", "single", "multi_unique", "multi_tie"}:
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if int(np.count_nonzero(grid == RECEIVER_COLOR)) < 4:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower-broadcast scene")
