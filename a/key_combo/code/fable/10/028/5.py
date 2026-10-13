from __future__ import annotations

import random
import numpy as np


BG = 0
GRID_LIMIT = 30
PALETTE = tuple(range(1, 10))
PANEL_H_RANGE = (4, 10)
WIDTH_RANGE = (8, 16)
MANDATORY_CONFLICT_COLUMNS = 3
EXTRA_CONFLICT_COLUMNS = (1, 3)
EXCLUSIVE_STAMPS_PER_PANEL = (3, 6)
MAX_GENERATION_ATTEMPTS = 400

STAMP_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    height = int(grid.shape[0])
    candidates = [
        row
        for row in range(height)
        if np.all(grid[row, :] != bg) and row > 0 and row < height - 1 and (2 * row + 1 == height)
    ]
    assert len(candidates) == 1, "expected exactly one full-width separator row splitting equal panels"
    return candidates[0]


def _overlay_panels(
    top: np.ndarray, bottom: np.ndarray, divider: np.ndarray, bg: int
) -> tuple[np.ndarray, list[dict[str, int | bool]]]:
    panel_h, width = top.shape
    out = np.full((panel_h, width), bg, dtype=int)
    conflicts: list[dict[str, int | bool]] = []
    for r in range(panel_h):
        for c in range(width):
            upper = int(top[r, c])
            lower = int(bottom[r, c])
            if upper == bg and lower == bg:
                continue
            if upper == bg:
                out[r, c] = lower
                continue
            if lower == bg:
                out[r, c] = upper
                continue
            if upper == lower:
                out[r, c] = upper
                continue
            arb = int(divider[c])
            assert arb in (upper, lower), "divider color must match one of the two conflicting colors"
            out[r, c] = arb
            conflicts.append(
                {
                    "row": r,
                    "col": c,
                    "top": upper,
                    "bottom": lower,
                    "winner": arb,
                    "loser": lower if arb == upper else upper,
                    "winner_on_top": arb == upper,
                }
            )
    return out, conflicts


def _conflict_columns(conflicts: list[dict[str, int | bool]]) -> dict[int, list[dict[str, int | bool]]]:
    by_col: dict[int, list[dict[str, int | bool]]] = {}
    for info in conflicts:
        by_col.setdefault(int(info["col"]), []).append(info)
    return by_col


def _admits_global_priority(conflicts: list[dict[str, int | bool]]) -> bool:
    edges = {(int(info["winner"]), int(info["loser"])) for info in conflicts}
    colors = {color for edge in edges for color in edge}
    indeg = {color: 0 for color in colors}
    graph = {color: set() for color in colors}
    for winner, loser in edges:
        if loser not in graph[winner]:
            graph[winner].add(loser)
            indeg[loser] += 1
    stack = [color for color, deg in indeg.items() if deg == 0]
    seen = 0
    while stack:
        color = stack.pop()
        seen += 1
        for nxt in graph[color]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                stack.append(nxt)
    return seen == len(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the equal-height panels above and below a multicolor separator row, using that row as a per-column tiebreak table.

    Input structure:
    - A dominant background surrounds two equal-height panels split by one full-width non-background row.
    - The separator row is fully filled and can contain several colors, while the panels above and below each contain foreground pixels.
    - Some local coordinates contain foreground in both panels, creating color conflicts in specific columns.

    Rule:
    - Delete the separator row and align the bottom panel over the top panel by local coordinates.
    - If only one panel contributes foreground at a cell, keep that color directly.
    - If both panels contribute different foreground colors at the same cell, inspect the separator color in that column and keep the matching side.

    Key insight:
    - The separator row is not decoration: each of its columns independently decides which conflicting color survives in that column after overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= GRID_LIMIT, "ARC grids must stay within 30x30"

    bg = _get_bg(g)
    divider_row = _find_separator_row(g, bg)
    divider = g[divider_row, :]
    assert np.unique(divider).size >= 2, "separator row must be visibly multicolor"

    top = g[:divider_row, :]
    bottom = g[divider_row + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "separator must split two non-empty equal-height panels"

    out, conflicts = _overlay_panels(top, bottom, divider, bg)
    assert conflicts, "expected at least one conflicting overlap"
    return out


def _random_divider(width: int, colors: list[int]) -> np.ndarray:
    divider = np.empty(width, dtype=int)
    c = 0
    last = None
    while c < width:
        run = random.randint(1, 3)
        options = [color for color in colors if color != last] or colors
        color = random.choice(options)
        divider[c : min(width, c + run)] = color
        last = color
        c += run
    return divider


def _pick_distinct_columns(width: int, count: int) -> list[int]:
    cols = random.sample(range(width), count)
    cols.sort()
    return cols


def _paint_conflict_run(
    top: np.ndarray,
    bottom: np.ndarray,
    divider: np.ndarray,
    column: int,
    winner: int,
    loser: int,
    winner_on_top: bool,
) -> None:
    panel_h = int(top.shape[0])
    run_len = random.randint(1, 2 if panel_h < 7 else 3)
    start = random.randint(0, panel_h - run_len)
    for r in range(start, start + run_len):
        assert top[r, column] == BG and bottom[r, column] == BG
        top[r, column] = winner if winner_on_top else loser
        bottom[r, column] = loser if winner_on_top else winner
    divider[column] = winner


def _place_exclusive_stamp(
    panel: np.ndarray,
    other_panel: np.ndarray,
    colors: list[int],
    blocked_cols: set[int],
) -> bool:
    panel_h, width = panel.shape
    for _ in range(60):
        shape = random.choice(STAMP_SHAPES)
        max_r = max(dr for dr, _ in shape)
        max_c = max(dc for _, dc in shape)
        anchor_r = random.randint(0, panel_h - 1 - max_r)
        anchor_c = random.randint(0, width - 1 - max_c)
        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if any(c in blocked_cols for _, c in cells):
            continue
        if any(panel[r, c] != BG or other_panel[r, c] != BG for r, c in cells):
            continue
        color = random.choice(colors)
        for r, c in cells:
            panel[r, c] = color
        return True
    return False


def _count_unique_side_cells(top: np.ndarray, bottom: np.ndarray, bg: int) -> tuple[int, int]:
    top_only = int(np.sum((top != bg) & (bottom == bg)))
    bottom_only = int(np.sum((bottom != bg) & (top == bg)))
    return top_only, bottom_only


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= GRID_LIMIT
    bg = _get_bg(g)
    assert bg == BG, "generator should keep black as the dominant background"

    divider_row = _find_separator_row(g, bg)
    top = g[:divider_row, :]
    divider = g[divider_row, :]
    bottom = g[divider_row + 1 :, :]
    assert top.shape == bottom.shape
    assert np.unique(divider).size >= 3, "divider should look decorative rather than binary"

    out, conflicts = _overlay_panels(top, bottom, divider, bg)
    assert conflicts and len(conflicts) >= 3, "need several conflict cells"

    by_col = _conflict_columns(conflicts)
    assert len(by_col) >= MANDATORY_CONFLICT_COLUMNS, "need several conflict columns"
    for col, infos in by_col.items():
        winner_colors = {int(info["winner"]) for info in infos}
        assert winner_colors == {int(divider[col])}, "all conflicts in one column must agree with the divider"

    sides = {bool(info["winner_on_top"]) for info in conflicts}
    assert sides == {False, True}, "both panels must win in some conflict columns"
    assert not _admits_global_priority(conflicts), "conflicts must defeat any global color-priority shortcut"

    top_only, bottom_only = _count_unique_side_cells(top, bottom, bg)
    assert top_only >= 4 and bottom_only >= 4, "both panels need clear single-sided evidence"

    camouflage_cols = [c for c in range(g.shape[1]) if c not in by_col]
    assert len(camouflage_cols) >= 2, "need separator columns that never arbitrate a conflict"
    assert any(np.all(top[:, c] == bg) and np.all(bottom[:, c] == bg) for c in camouflage_cols), (
        "at least one separator column should be pure camouflage with no panel foreground"
    )

    other_full_rows = [
        r
        for r in range(g.shape[0])
        if r != divider_row and np.all(g[r, :] != bg)
    ]
    assert not other_full_rows, "separator row must be the only full-width filled row"

    direct = main(g)
    assert np.array_equal(direct, out)


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    width = random.randint(*WIDTH_RANGE)
    if 2 * panel_h + 1 > GRID_LIMIT or width > GRID_LIMIT:
        return None

    color_count = random.randint(4, 6)
    colors = random.sample(PALETTE, color_count)
    cycle_colors = colors[:3]

    top = np.full((panel_h, width), BG, dtype=int)
    bottom = np.full((panel_h, width), BG, dtype=int)
    divider = _random_divider(width, colors)

    conflict_column_count = MANDATORY_CONFLICT_COLUMNS + random.randint(*EXTRA_CONFLICT_COLUMNS)
    if conflict_column_count >= width:
        return None
    conflict_cols = _pick_distinct_columns(width, conflict_column_count)
    silent_cols = set(random.sample([c for c in range(width) if c not in conflict_cols], 2))

    mandatory_pairs = [
        (cycle_colors[0], cycle_colors[1]),
        (cycle_colors[1], cycle_colors[2]),
        (cycle_colors[2], cycle_colors[0]),
    ]
    winner_sides = [random.choice((False, True)) for _ in mandatory_pairs]
    if len(set(winner_sides)) == 1:
        winner_sides[random.randrange(len(winner_sides))] = not winner_sides[0]

    for column, (winner, loser), winner_on_top in zip(conflict_cols[:3], mandatory_pairs, winner_sides):
        _paint_conflict_run(top, bottom, divider, column, winner, loser, winner_on_top)

    for column in conflict_cols[3:]:
        a, b = random.sample(colors, 2)
        winner = random.choice((a, b))
        loser = b if winner == a else a
        winner_on_top = random.choice((False, True))
        _paint_conflict_run(top, bottom, divider, column, winner, loser, winner_on_top)

    stamp_target = random.randint(*EXCLUSIVE_STAMPS_PER_PANEL)
    for _ in range(stamp_target):
        if not _place_exclusive_stamp(top, bottom, colors, silent_cols):
            return None
    for _ in range(stamp_target):
        if not _place_exclusive_stamp(bottom, top, colors, silent_cols):
            return None

    grid = np.vstack([top, divider[None, :], bottom])

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-arbitration overlay input")
