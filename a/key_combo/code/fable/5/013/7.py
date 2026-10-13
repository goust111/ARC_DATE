from __future__ import annotations

from collections import Counter

import numpy as np


# Blue and red are the visible alternating growth palette.  Grey is reserved
# for obstacles; the background is inferred from the grid.
GROWTH_COLORS = (1, 2)
OBSTACLE_COLOR = 5
BACKGROUND_COLORS = (0, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _parse_input(grid: np.ndarray) -> tuple[int, int, int]:
    """Return (background, seed row, seed color) for a valid puzzle grid."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    bg = _background_color(g)
    assert bg not in (*GROWTH_COLORS, OBSTACLE_COLOR), "background conflicts with role colors"
    allowed = {bg, *GROWTH_COLORS, OBSTACLE_COLOR}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input color"

    seed_mask = np.isin(g, GROWTH_COLORS)
    seed_rows = np.flatnonzero(seed_mask.any(axis=1))
    assert len(seed_rows) == 1, "all seed pixels must occupy one row"
    seed_row = int(seed_rows[0])
    seed_values = set(map(int, g[seed_row, seed_mask[seed_row]]))
    assert len(seed_values) == 1, "the seed row must be monochromatic"
    seed_color = int(next(iter(seed_values)))
    assert int(seed_mask[seed_row].sum()) >= 2, "expected multiple seed pixels"

    obstacle_rows = np.argwhere(g == OBSTACLE_COLOR)
    if obstacle_rows.size:
        assert bool(np.all(obstacle_rows[:, 0] > seed_row)), (
            "grey obstacles must all lie below the seed row"
        )
    return bg, seed_row, seed_color


def _growth_trace(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, frozenset[int]]]]:
    """Simulate successful active rows and return their coordinates by generation."""
    g = np.asarray(grid, dtype=int)
    bg, seed_row, seed_color = _parse_input(g)
    height, width = map(int, g.shape)
    active = g[seed_row] == seed_color
    parent_row = seed_row
    levels: list[tuple[int, frozenset[int]]] = []

    while parent_row + 2 < height:
        child_row = parent_row + 2
        children: list[int] = []
        for c in range(1, width - 1):
            # The middle of the live-empty-live triple is interpreted in the
            # current active row, using the puzzle's own row/column frame.
            is_live_gap_live = (
                bool(active[c - 1])
                and not bool(active[c])
                and bool(active[c + 1])
                and int(g[parent_row, c]) == bg
            )
            if is_live_gap_live and int(g[child_row, c]) == bg:
                children.append(c)

        if not children:
            break
        levels.append((child_row, frozenset(children)))
        active = np.zeros(width, dtype=bool)
        active[children] = True
        parent_row = child_row

    return bg, seed_row, seed_color, levels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating-color descendants downward while grey birth obstacles prune whole branches.

    Input structure:
    - A dominant uniform background contains one monochromatic blue-or-red seed row near the top.
    - The seed row includes isolated pixels and a consecutive run; sparse grey pixels occur only below it.
    - Blue and red form the visible two-color growth palette, while grey is the obstacle color.

    Rule:
    - In each active row, every live-background-live triple births a pixel at the middle column two rows lower.
    - Generations alternate between blue and red, beginning with the color opposite the seed color.
    - A child whose exact birth cell is occupied does not appear; only successful children form the next active
      row, so every later descendant requiring a blocked child also disappears.  Grey obstacles stay unchanged.
    - Continue until a generation has no successful births or its birth row would leave the grid.

    Key insight:
    - Blocking is recursive pruning in the generation tree, not a vertical shadow; each active generation must
      be recomputed from the surviving children of the immediately preceding generation.
    """
    g = np.asarray(grid, dtype=int)
    bg, _seed_row, seed_color, levels = _growth_trace(g)
    other_color = GROWTH_COLORS[1] if seed_color == GROWTH_COLORS[0] else GROWTH_COLORS[0]

    out = g.copy()
    for generation, (row, columns) in enumerate(levels, start=1):
        color = other_color if generation % 2 == 1 else seed_color
        for c in columns:
            assert int(out[row, c]) == bg
            out[row, c] = color
    return out


def _trace_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {
        (row, c)
        for row, columns in _growth_trace(grid)[3]
        for c in columns
    }


def _has_consecutive_run_and_isolated_seed(grid: np.ndarray, seed_row: int, seed_color: int) -> bool:
    mask = np.asarray(grid[seed_row] == seed_color, dtype=bool)
    has_run = bool(np.any(mask[:-1] & mask[1:]))
    isolated = [
        c
        for c in range(mask.size)
        if mask[c]
        and (c == 0 or not mask[c - 1])
        and (c == mask.size - 1 or not mask[c + 1])
    ]
    return has_run and len(isolated) >= 4


def _semantic_validation(
    grid: np.ndarray,
    effective_obstacles: list[tuple[int, int]],
    distractor_obstacles: list[tuple[int, int]],
) -> bool:
    """Check uniqueness, alternation, recursive pruning, and anti-shortcut examples."""
    try:
        bg, seed_row, seed_color, levels = _growth_trace(grid)
        out = main(grid)
    except (AssertionError, ValueError):
        return False

    if not _has_consecutive_run_and_isolated_seed(grid, seed_row, seed_color):
        return False
    if len(levels) < 2 or not effective_obstacles or not distractor_obstacles:
        return False
    if not np.array_equal(out[grid == OBSTACLE_COLOR], grid[grid == OBSTACLE_COLOR]):
        return False

    # Alternation is checked directly, rather than merely comparing main to a
    # helper that shares its implementation.
    other = GROWTH_COLORS[1] if seed_color == GROWTH_COLORS[0] else GROWTH_COLORS[0]
    for generation, (row, columns) in enumerate(levels, start=1):
        expected_color = other if generation % 2 else seed_color
        if any(int(out[row, c]) != expected_color for c in columns):
            return False

    actual_cells = _trace_cells(grid)

    # Every designated blocker must matter even in the presence of all other
    # blockers: removing it restores a direct birth and at least one deeper
    # descendant.  This rules out simple obstacle overlay and vertical-shadow
    # interpretations.
    for r, c in effective_obstacles:
        restored = grid.copy()
        restored[r, c] = bg
        restored_cells = _trace_cells(restored)
        gained = restored_cells - actual_cells
        if (r, c) not in gained or not any(rr >= r + 2 for rr, _ in gained):
            return False

    # Distractors live on skipped spacer rows.  They remain visible but must not
    # alter the cascade, and at least one surviving child lies below one in the
    # same column to explicitly refute a geometric shadow shortcut.
    shadow_counterexample = False
    for r, c in distractor_obstacles:
        removed = grid.copy()
        removed[r, c] = bg
        if _trace_cells(removed) != actual_cells:
            return False
        if any(rr > r and cc == c for rr, cc in actual_cells):
            shadow_counterexample = True
    if not shadow_counterexample:
        return False

    # At least one branch must continue below the first blocked generation.
    first_block_row = min(r for r, _ in effective_obstacles)
    if not any(r >= first_block_row + 2 for r, _ in actual_cells):
        return False
    return out.shape == grid.shape and max(out.shape) <= 30


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    seed_count = int(rng.integers(8, 11))
    run_length = int(rng.integers(2, 5))
    separation = int(rng.integers(3, 6))
    chain_span = 2 * seed_count - 1
    minimum_width = chain_span + run_length + separation + 2
    if minimum_width > 30:
        return None
    width = int(rng.integers(minimum_width, 31))

    seed_row = int(rng.integers(1, 4))
    height = int(rng.integers(max(13, seed_row + 8), 27))
    bg = int(rng.choice(BACKGROUND_COLORS))
    seed_color = int(rng.choice(GROWTH_COLORS))
    grid = np.full((height, width), bg, dtype=int)

    total_used = chain_span + run_length + separation
    left_margin = int(rng.integers(1, width - total_used))
    run_on_left = bool(rng.integers(0, 2))
    if run_on_left:
        run_start = left_margin
        chain_start = run_start + run_length + separation
    else:
        chain_start = left_margin
        run_start = chain_start + chain_span + separation

    chain_columns = [chain_start + 2 * i for i in range(seed_count)]
    run_columns = list(range(run_start, run_start + run_length))
    grid[seed_row, chain_columns + run_columns] = seed_color

    # Compute the unobstructed tree, then choose an interior first-generation
    # node.  Its removal leaves two surviving branches but deletes descendants.
    try:
        nominal_levels = _growth_trace(grid)[3]
    except AssertionError:
        return None
    if len(nominal_levels) < 4:
        return None
    first_row, first_columns_set = nominal_levels[0]
    first_columns = sorted(first_columns_set)
    blocker_indices = list(range(2, len(first_columns) - 2))
    if not blocker_indices:
        return None
    blocker = (first_row, first_columns[int(rng.choice(blocker_indices))])
    grid[blocker] = OBSTACLE_COLOR
    effective = [blocker]

    # Roughly half the examples contain a second genuinely cascading blocker.
    if bool(rng.integers(0, 2)):
        before_cells = _trace_cells(grid)
        candidates = sorted((r, c) for r, c in before_cells if r + 2 < height)
        rng.shuffle(candidates)
        for candidate in candidates:
            test = grid.copy()
            test[candidate] = OBSTACLE_COLOR
            after_cells = _trace_cells(test)
            gained_if_removed = before_cells - after_cells
            if (
                candidate in gained_if_removed
                and any(rr >= candidate[0] + 2 for rr, _ in gained_if_removed)
                and any(rr >= candidate[0] + 2 for rr, _ in after_cells)
            ):
                grid = test
                effective.append(candidate)
                break

    # Put a harmless grey obstacle on a skipped spacer row directly above a
    # surviving later child.  Growth below it makes the non-shadow behavior
    # visually explicit.
    current_cells = _trace_cells(grid)
    later_cells = sorted((r, c) for r, c in current_cells if r >= seed_row + 4)
    if not later_cells:
        return None
    below_r, below_c = later_cells[int(rng.integers(len(later_cells)))]
    distractor = (below_r - 1, below_c)
    if int(grid[distractor]) != bg:
        return None
    grid[distractor] = OBSTACLE_COLOR
    distractors = [distractor]

    # Optional additional spacer-row distractors vary obstacle count without
    # changing the logical tree.
    for _ in range(int(rng.integers(0, 3))):
        spacer_rows = list(range(seed_row + 1, height, 2))
        r = int(rng.choice(spacer_rows))
        c = int(rng.integers(1, width - 1))
        if int(grid[r, c]) == bg:
            grid[r, c] = OBSTACLE_COLOR
            distractors.append((r, c))

    if not _semantic_validation(grid, effective, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate cascade-pruning puzzle input."""
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-pruning input")
