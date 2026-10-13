from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MAROON = 9

PANEL_HEIGHT_RANGE = (4, 7)
PANEL_WIDTH_RANGE = (14, 22)
MAX_GENERATION_ATTEMPTS = 400
OBSTACLE_PALETTE = (4, 5, 6, 7, 8)


def _is_foreground(value: int) -> bool:
    return int(value) not in (BG, MAROON)


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == MAROON)]
    assert len(rows) == 1, "expected exactly one maroon separator row"
    sep = rows[0]
    assert sep > 0 and sep < grid.shape[0] - 1, "separator cannot lie on the border"
    assert sep == grid.shape[0] - sep - 1, "separator must split equal-height panels"
    return sep


def _corridor_mask(panel: np.ndarray, corridor_color: int) -> np.ndarray:
    height, width = panel.shape
    mask = np.zeros_like(panel)
    for r in range(height):
        if (
            _is_foreground(int(panel[r, 0]))
            and _is_foreground(int(panel[r, width - 1]))
            and np.all(panel[r, 1 : width - 1] == BG)
        ):
            mask[r, 1 : width - 1] = corridor_color
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse two separator-stacked panels into a single overlay of their independently extended row corridors.

    Input structure:
    - A full maroon separator row splits the grid into equal-height upper and lower panels.
    - Each panel has a black background with optional foreground markers on the left and right border columns.
    - The upper panel uses red border/corridor markers, the lower panel uses blue border/corridor markers, and interior non-black pixels act as blockers.

    Rule:
    - In each panel independently, inspect every row: if both border cells are foreground and every interior cell is black, fill that row's interior with the panel's corridor color.
    - Ignore blocked rows and remove the maroon separator row.
    - Overlay the two filled panels cell by cell: red+blue corridor overlap becomes green, upper-only corridor stays red, lower-only corridor stays blue, and cells with no corridor stay black.
    - On the leftmost and rightmost columns, copy the original border marker from the upper panel when present, otherwise from the lower panel.

    Key insight:
    - The output depends on comparing two independently computed corridor masks; the green cells appear only where both panels are clear on the same aligned row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape

    top_mask = _corridor_mask(top, RED)
    bottom_mask = _corridor_mask(bottom, BLUE)

    out = np.zeros_like(top)
    both = (top_mask == RED) & (bottom_mask == BLUE)
    top_only = (top_mask == RED) & (bottom_mask == BG)
    bottom_only = (top_mask == BG) & (bottom_mask == BLUE)
    out[both] = GREEN
    out[top_only] = RED
    out[bottom_only] = BLUE

    for c in (0, out.shape[1] - 1):
        for r in range(out.shape[0]):
            if _is_foreground(int(top[r, c])):
                out[r, c] = int(top[r, c])
            elif _is_foreground(int(bottom[r, c])):
                out[r, c] = int(bottom[r, c])

    return out


def _candidate_clear_counts(height: int) -> list[tuple[int, int, int]]:
    choices: list[tuple[int, int, int]] = []
    for common in range(1, height - 1):
        for top_only in range(1, height - common):
            for bottom_only in range(1, height - common - top_only + 1):
                top_count = common + top_only
                bottom_count = common + bottom_only
                if top_count <= height - 2 and bottom_count <= height - 2:
                    choices.append((common, top_only, bottom_only))
    return choices


def _sample_corridor_sets(height: int) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    counts = random.choice(_candidate_clear_counts(height))
    common_n, top_only_n, bottom_only_n = counts

    rows = list(range(height))
    random.shuffle(rows)
    common_rows = set(rows[:common_n])
    cursor = common_n
    top_only_rows = set(rows[cursor : cursor + top_only_n])
    cursor += top_only_n
    bottom_only_rows = set(rows[cursor : cursor + bottom_only_n])

    top_clear = common_rows | top_only_rows
    bottom_clear = common_rows | bottom_only_rows
    return top_clear, bottom_clear, common_rows, top_only_rows, bottom_only_rows


def _sample_blocked_rows(clear_rows: set[int], counterpart_only: set[int], height: int) -> set[int]:
    candidates = set(range(height)) - clear_rows
    assert len(candidates) >= 2
    blocked = {random.choice(sorted(counterpart_only))}
    extra_pool = sorted(candidates - blocked)
    extra_cap = min(2, len(candidates) - 1)
    extra_count = random.randint(0, extra_cap)
    if extra_count > 0:
        blocked |= set(random.sample(extra_pool, extra_count))
    if len(candidates - blocked) < 1:
        removable = sorted(blocked - set(counterpart_only))
        assert removable
        blocked.remove(removable[0])
    return blocked


def _place_row_obstacles(panel: np.ndarray, row: int, obstacle_colors: tuple[int, ...]) -> None:
    width = panel.shape[1]
    obstacle_count = random.randint(1, min(3, width - 2))
    cols = sorted(random.sample(range(1, width - 1), obstacle_count))
    for c in cols:
        panel[row, c] = random.choice(obstacle_colors)


def _build_panel(
    height: int,
    width: int,
    border_color: int,
    clear_rows: set[int],
    blocked_rows: set[int],
    edge_only_row: int,
    edge_only_side: int,
    obstacle_colors: tuple[int, ...],
) -> np.ndarray:
    panel = np.zeros((height, width), dtype=int)

    for r in clear_rows:
        panel[r, 0] = border_color
        panel[r, width - 1] = border_color

    for r in blocked_rows:
        panel[r, 0] = border_color
        panel[r, width - 1] = border_color
        _place_row_obstacles(panel, r, obstacle_colors)

    panel[edge_only_row, edge_only_side] = border_color
    if random.random() < 0.45:
        other_side = width - 1 - edge_only_side
        panel[edge_only_row, other_side] = border_color
        _place_row_obstacles(panel, edge_only_row, obstacle_colors)

    return panel


def _row_signature(panel: np.ndarray, row: int) -> tuple[bool, bool, bool]:
    has_left = _is_foreground(int(panel[row, 0]))
    has_right = _is_foreground(int(panel[row, panel.shape[1] - 1]))
    has_inner = bool(np.any(panel[row, 1 : panel.shape[1] - 1] != BG))
    return has_left, has_right, has_inner


def _validate_generated_input(
    grid: np.ndarray,
    top_clear_target: set[int],
    bottom_clear_target: set[int],
    top_block_target: set[int],
    bottom_block_target: set[int],
    top_edge_only_row: int,
    bottom_edge_only_row: int,
) -> None:
    sep = _separator_row(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    height, width = top.shape
    assert width >= 4

    top_mask = _corridor_mask(top, RED)
    bottom_mask = _corridor_mask(bottom, BLUE)
    top_clear = {r for r in range(height) if np.any(top_mask[r, 1 : width - 1] == RED)}
    bottom_clear = {r for r in range(height) if np.any(bottom_mask[r, 1 : width - 1] == BLUE)}
    assert top_clear == top_clear_target
    assert bottom_clear == bottom_clear_target

    top_blocked = {
        r
        for r in range(height)
        if _is_foreground(int(top[r, 0]))
        and _is_foreground(int(top[r, width - 1]))
        and np.any(top[r, 1 : width - 1] != BG)
    }
    bottom_blocked = {
        r
        for r in range(height)
        if _is_foreground(int(bottom[r, 0]))
        and _is_foreground(int(bottom[r, width - 1]))
        and np.any(bottom[r, 1 : width - 1] != BG)
    }
    assert top_block_target <= top_blocked
    assert bottom_block_target <= bottom_blocked

    top_edge_sig = _row_signature(top, top_edge_only_row)
    bottom_edge_sig = _row_signature(bottom, bottom_edge_only_row)
    assert top_edge_sig in {(True, False, False), (False, True, False), (True, True, True)}
    assert bottom_edge_sig in {(True, False, False), (False, True, False), (True, True, True)}

    overlap = top_clear & bottom_clear
    assert overlap
    assert top_clear - bottom_clear
    assert bottom_clear - top_clear

    out = main(grid)
    assert np.any(out[:, 1 : width - 1] == GREEN), "need overlap cells"
    assert np.any(out[:, 1 : width - 1] == RED), "need upper-only corridor cells"
    assert np.any(out[:, 1 : width - 1] == BLUE), "need lower-only corridor cells"

    top_boundary_only = False
    bottom_boundary_only = False
    for c in (0, width - 1):
        for r in range(height):
            if _is_foreground(int(top[r, c])) and not _is_foreground(int(bottom[r, c])):
                top_boundary_only = True
            if _is_foreground(int(bottom[r, c])) and not _is_foreground(int(top[r, c])):
                bottom_boundary_only = True
    assert top_boundary_only, "boundary priority needs an upper-only example"
    assert bottom_boundary_only, "boundary priority needs a lower-only example"

    for r in range(height):
        if r not in top_clear:
            assert not np.all(top[r, 1 : width - 1] == BG) or not (
                _is_foreground(int(top[r, 0])) and _is_foreground(int(top[r, width - 1]))
            ), "top non-corridor rows must be blocked or missing an endpoint"
        if r not in bottom_clear:
            assert not np.all(bottom[r, 1 : width - 1] == BG) or not (
                _is_foreground(int(bottom[r, 0])) and _is_foreground(int(bottom[r, width - 1]))
            ), "bottom non-corridor rows must be blocked or missing an endpoint"


def _generate_one() -> np.ndarray:
    height = random.randint(*PANEL_HEIGHT_RANGE)
    width = random.randint(*PANEL_WIDTH_RANGE)

    top_clear, bottom_clear, _common, top_only, bottom_only = _sample_corridor_sets(height)
    top_blocked = _sample_blocked_rows(top_clear, bottom_only, height)
    bottom_blocked = _sample_blocked_rows(bottom_clear, top_only, height)

    top_idle = sorted(set(range(height)) - top_clear - top_blocked)
    bottom_idle = sorted(set(range(height)) - bottom_clear - bottom_blocked)
    assert top_idle and bottom_idle

    top_edge_only_row = random.choice(top_idle)
    bottom_edge_only_choices = [r for r in bottom_idle if r != top_edge_only_row] or bottom_idle
    bottom_edge_only_row = random.choice(bottom_edge_only_choices)

    top_side = random.choice((0, width - 1))
    bottom_side = random.choice((0, width - 1))
    if bottom_edge_only_row == top_edge_only_row and bottom_side == top_side:
        bottom_side = width - 1 - bottom_side

    top_obstacle_colors = tuple(random.sample(OBSTACLE_PALETTE, k=2))
    bottom_obstacle_colors = tuple(random.sample(OBSTACLE_PALETTE, k=2))

    top = _build_panel(
        height,
        width,
        RED,
        top_clear,
        top_blocked,
        top_edge_only_row,
        top_side,
        top_obstacle_colors,
    )
    bottom = _build_panel(
        height,
        width,
        BLUE,
        bottom_clear,
        bottom_blocked,
        bottom_edge_only_row,
        bottom_side,
        bottom_obstacle_colors,
    )

    grid = np.vstack([top, np.full((1, width), MAROON, dtype=int), bottom])
    _validate_generated_input(
        grid,
        top_clear,
        bottom_clear,
        top_blocked,
        bottom_blocked,
        top_edge_only_row,
        bottom_edge_only_row,
    )
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_one()
        except AssertionError:
            continue
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
