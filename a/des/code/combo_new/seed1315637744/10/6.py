from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _region_winner(grid: np.ndarray, region: list[tuple[int, int]]) -> int:
    height, width = grid.shape
    scores: dict[int, int] = {}
    first_pos: dict[int, tuple[int, int]] = {}

    for r, c in region:
        color = int(grid[r, c])
        if color in (BLACK, GREY):
            continue
        touch_score = sum(int(grid[nr, nc] == GREY) for nr, nc in _neighbors4(r, c, height, width))
        scores[color] = scores.get(color, 0) + touch_score
        pos = (r, c)
        if color not in first_pos or pos < first_pos[color]:
            first_pos[color] = pos

    assert scores, "each enclosed region must contain at least one colored marker"
    best_score = max(scores.values())
    winners = [color for color, score in scores.items() if score == best_score]
    winners.sort(key=lambda color: first_pos[color])
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed compartment with the color whose markers have the greatest total grey-wall contact.

    Input structure:
    - Black background surrounds a set of one-cell-thick grey horizontal and vertical walls.
    - The grey walls form closed rectangular regions, and each region contains a few non-grey single-pixel markers.
    - A marker may sit in the interior of its region or directly against one or more grey wall cells.

    Rule:
    - Treat grey cells as barriers and find every non-grey connected region that does not touch the grid edge.
    - For each colored marker, count how many of its 4-neighbors are grey; this count is that marker's contact score.
    - Sum contact scores by color inside each region, choose the highest-scoring color, break ties by the topmost occurrence and then the leftmost, and fill the whole non-grey region with that color while leaving grey walls unchanged.

    Key insight:
    - The winning color is determined by weighted boundary contact, not by raw frequency: pixels pressed against more grey edges vote more strongly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    open_cells = g != GREY
    regions = _components(open_cells)
    assert regions, "expected at least one non-grey component"

    out = g.copy()
    height, width = g.shape
    for region in regions:
        if any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in region):
            continue
        winner = _region_winner(g, region)
        for r, c in region:
            out[r, c] = winner
    return out


REGION_ROWS_RANGE = (2, 4)
REGION_COLS_RANGE = (2, 4)
CELL_SIZE_RANGE = (4, 6)
MARKER_COUNT_RANGE = (2, 5)
GEN_RETRIES = 300
MODE_NAMES = (
    "corner_win",
    "repeat_win",
    "score_tie_top",
    "score_tie_left",
    "all_zero",
)


def _wall_positions(sizes: list[int]) -> list[int]:
    walls = [0]
    for size in sizes:
        walls.append(walls[-1] + size + 1)
    return walls


def _region_positions(r0: int, c0: int, r1: int, c1: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    corners: list[tuple[int, int]] = []
    edges: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            score = int(r in (r0, r1)) + int(c in (c0, c1))
            if score == 2:
                corners.append((r, c))
            elif score == 1:
                edges.append((r, c))
            else:
                interior.append((r, c))
    return corners, edges, interior


def _sample_distinct(seq: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    assert len(seq) >= k
    return random.sample(seq, k)


def _place_markers_for_mode(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    mode: str,
) -> None:
    corners, edges, interior = _region_positions(r0, c0, r1, c1)
    colors = random.sample(PALETTE, 6)
    placements: list[tuple[tuple[int, int], int]] = []

    if mode == "corner_win":
        total_markers = random.randint(*MARKER_COUNT_RANGE)
        winner = colors[0]
        winner_pos = random.choice(corners)
        placements.append((winner_pos, winner))

        remaining = total_markers - 1
        loser_positions = _sample_distinct(edges + interior, remaining)
        for idx, pos in enumerate(loser_positions, start=1):
            placements.append((pos, colors[idx]))

    elif mode == "repeat_win":
        total_markers = random.randint(*MARKER_COUNT_RANGE)
        winner = colors[0]
        winner_positions = _sample_distinct(edges, 2)
        placements.extend((pos, winner) for pos in winner_positions)

        remaining = total_markers - 2
        pool = [pos for pos in edges + interior if pos not in winner_positions]
        loser_positions = _sample_distinct(pool, remaining)
        for idx, pos in enumerate(loser_positions, start=1):
            placements.append((pos, colors[idx]))

    elif mode == "score_tie_top":
        total_markers = random.randint(*MARKER_COUNT_RANGE)
        winner = colors[0]
        loser = colors[1]
        if random.choice((False, True)):
            winner_pos = random.choice([pos for pos in edges if pos[0] == r0])
            loser_pos = random.choice([pos for pos in edges if pos[0] == r1])
        else:
            winner_pos = random.choice([(r0, c0), (r0, c1)])
            loser_pos = random.choice([(r1, c0), (r1, c1)])
        placements.append((winner_pos, winner))
        placements.append((loser_pos, loser))

        remaining = total_markers - 2
        pool = [pos for pos in interior if pos not in (winner_pos, loser_pos)]
        extras = _sample_distinct(pool, remaining)
        for idx, pos in enumerate(extras, start=2):
            placements.append((pos, colors[idx]))

    elif mode == "score_tie_left":
        total_markers = random.randint(*MARKER_COUNT_RANGE)
        winner = colors[0]
        loser = colors[1]
        top_edges = [pos for pos in edges if pos[0] == r0]
        winner_pos, loser_pos = sorted(_sample_distinct(top_edges, 2))
        placements.append((winner_pos, winner))
        placements.append((loser_pos, loser))

        remaining = total_markers - 2
        pool = [pos for pos in interior if pos not in (winner_pos, loser_pos)]
        extras = _sample_distinct(pool, remaining)
        for idx, pos in enumerate(extras, start=2):
            placements.append((pos, colors[idx]))

    else:
        assert mode == "all_zero"
        total_markers = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(interior)))
        sorted_interior = sorted(_sample_distinct(interior, total_markers))
        winner = colors[0]
        placements.append((sorted_interior[0], winner))
        for idx, pos in enumerate(sorted_interior[1:], start=1):
            placements.append((pos, colors[idx]))

    used_positions: set[tuple[int, int]] = set()
    for pos, color in placements:
        assert pos not in used_positions
        used_positions.add(pos)
        grid[pos] = color


def _build_grid() -> np.ndarray:
    region_rows = random.randint(*REGION_ROWS_RANGE)
    region_cols = random.randint(*REGION_COLS_RANGE)
    cell_heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(region_rows)]
    cell_widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(region_cols)]

    row_walls = _wall_positions(cell_heights)
    col_walls = _wall_positions(cell_widths)
    height = row_walls[-1] + 1
    width = col_walls[-1] + 1
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_walls, :] = GREY
    grid[:, col_walls] = GREY

    region_boxes = []
    for rr in range(region_rows):
        for cc in range(region_cols):
            region_boxes.append(
                (
                    row_walls[rr] + 1,
                    col_walls[cc] + 1,
                    row_walls[rr + 1] - 1,
                    col_walls[cc + 1] - 1,
                )
            )

    if len(region_boxes) <= len(MODE_NAMES):
        mode_cycle = random.sample(MODE_NAMES, len(region_boxes))
    else:
        mode_cycle = list(MODE_NAMES)
        mode_cycle.extend(random.choice(MODE_NAMES) for _ in range(len(region_boxes) - len(MODE_NAMES)))
        random.shuffle(mode_cycle)

    for box, mode in zip(region_boxes, mode_cycle, strict=False):
        _place_markers_for_mode(grid, *box, mode)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_grid()
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid weighted-vote region puzzle")
