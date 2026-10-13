from __future__ import annotations

import random

import numpy as np


BLACK = 0
SEPARATOR = 5
COLORS = tuple(color for color in range(10) if color not in (BLACK, SEPARATOR))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MODE_SAME_MAJORITY = "same_majority"
MODE_DIFFERENT_MAJORITY = "different_majority"
MODE_SAME_TIE = "same_tie"
MODE_DIFFERENT_TIE = "different_tie"
PANEL_MODES = (
    MODE_SAME_MAJORITY,
    MODE_DIFFERENT_MAJORITY,
    MODE_SAME_TIE,
    MODE_DIFFERENT_TIE,
)

LAYOUT_OPTIONS = tuple(
    (panel_rows, panel_cols, panel_h, panel_w)
    for panel_rows in range(2, 5)
    for panel_cols in range(2, 5)
    for panel_h in range(5, 8)
    for panel_w in range(5, 8)
    if 15 <= panel_rows * panel_h + (panel_rows - 1) <= 25
    if 15 <= panel_cols * panel_w + (panel_cols - 1) <= 25
)

GEN_RETRY_LIMIT = 300
PANEL_RETRY_LIMIT = 120


def _separator_lines(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0])]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == g[0, c])]
    colors = {int(g[r, 0]) for r in row_lines} | {int(g[0, c]) for c in col_lines}
    assert colors, "expected full separator lines"
    assert len(colors) == 1, "separator rows and columns must share one color"
    return int(next(iter(colors))), row_lines, col_lines


def _panel_intervals(size: int, lines: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + [int(line) for line in lines] + [int(size)]
    intervals: list[tuple[int, int]] = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start > 1:
            intervals.append((start + 1, end))
    assert intervals, "expected at least one panel interval"
    lengths = {end - start for start, end in intervals}
    assert len(lengths) == 1, "panels must be equal-sized along each axis"
    return intervals


def _background_color(g: np.ndarray, separator: int) -> int:
    values, counts = np.unique(g[g != separator], return_counts=True)
    assert values.size > 0
    return int(values[np.argmax(counts)])


def _components(panel: np.ndarray, bg: int, separator: int) -> list[tuple[int, np.ndarray]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(panel[r, c])
            if color in (bg, separator):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append((color, np.array(coords, dtype=int)))

    return comps


def _vote_winner(votes: list[tuple[int, int, int]]) -> int:
    ordered = sorted(votes, key=lambda item: (item[0], item[1]))
    counts: dict[int, int] = {}
    first_pos: dict[int, tuple[int, int]] = {}
    for r, c, color in ordered:
        counts[color] = counts.get(color, 0) + 1
        first_pos.setdefault(color, (int(r), int(c)))

    winner = None
    best_count = -1
    best_pos: tuple[int, int] | None = None
    for color, count in counts.items():
        pos = first_pos[color]
        if count > best_count or (count == best_count and (best_pos is None or pos < best_pos)):
            winner = int(color)
            best_count = int(count)
            best_pos = pos

    assert winner is not None
    return winner


def _panel_result(panel: np.ndarray, bg: int, separator: int) -> tuple[np.ndarray, int]:
    comps = _components(panel, bg, separator)
    assert comps, "expected a pattern and votes inside each panel"

    patterns = [(color, coords) for color, coords in comps if coords.shape[0] > 1]
    votes = [(int(coords[0, 0]), int(coords[0, 1]), color) for color, coords in comps if coords.shape[0] == 1]

    assert len(patterns) == 1, "expected one multi-cell pattern per panel"
    pattern_color, pattern_coords = patterns[0]
    assert pattern_coords.shape[0] >= 3, "pattern must contain at least three cells"
    assert len(votes) >= 3, "expected at least three vote pixels per panel"

    winner = _vote_winner(votes)
    out = np.full(panel.shape, bg, dtype=int)
    out[pattern_coords[:, 0], pattern_coords[:, 1]] = int(winner)
    return out, int(pattern_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel's connected pattern to the winning vote color and erase the vote pixels.

    Input structure:
    - A black background is partitioned into equal rectangular panels by one-cell separator lines.
    - Inside each panel, one monochrome 4-connected pattern occupies at least three cells.
    - The other non-background, non-separator cells in the panel are isolated one-pixel vote points.

    Rule:
    - Recover the panel lattice from the full separator rows and columns.
    - In each panel, treat the unique non-singleton monochrome component as the pattern and every singleton component as a vote.
    - Count vote colors; if several colors tie for the maximum count, choose the color whose vote appears first in top-to-bottom, left-to-right scan order.
    - Recolor the entire pattern to that winning color, erase all vote pixels to background, and keep the separator lines unchanged.

    Key insight:
    - Connectivity and component size separate the two roles: the only multi-cell component is the drawable object, while isolated singleton pixels are just votes for its output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, row_lines, col_lines = _separator_lines(g)
    row_panels = _panel_intervals(g.shape[0], row_lines)
    col_panels = _panel_intervals(g.shape[1], col_lines)
    bg = _background_color(g, separator)

    out = np.full_like(g, bg)
    if row_lines:
        out[row_lines, :] = separator
    if col_lines:
        out[:, col_lines] = separator

    for r0, r1 in row_panels:
        for c0, c1 in col_panels:
            panel_out, _pattern_color = _panel_result(g[r0:r1, c0:c1], bg, separator)
            out[r0:r1, c0:c1] = panel_out

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_shape(max_h: int, max_w: int) -> set[tuple[int, int]]:
    box_h = random.randint(2, min(4, max_h))
    box_w = random.randint(2, min(4, max_w))
    target = random.randint(3, min(5, box_h * box_w))

    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    return _normalize(cells)


def _shifted(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(r + top, c + left) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _vote_count_for_mode(mode: str) -> int:
    if mode in (MODE_SAME_TIE, MODE_DIFFERENT_TIE):
        return random.choice((4, 5))
    return random.randint(3, 5)


def _candidate_vote_cells(panel_h: int, panel_w: int, pattern: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(panel_h):
        for c in range(panel_w):
            if (r, c) in pattern:
                continue
            if any((r + dr, c + dc) in pattern for dr, dc in NEIGHBORS4):
                continue
            candidates.append((r, c))
    return candidates


def _sample_nonadjacent_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    for _ in range(60):
        pool = cells[:]
        random.shuffle(pool)
        picked: list[tuple[int, int]] = []
        for cell in pool:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in picked):
                picked.append(cell)
                if len(picked) == count:
                    return picked
    return None


def _vote_colors(pattern_color: int, mode: str, vote_count: int) -> list[int]:
    other_colors = [color for color in COLORS if color != pattern_color]
    rival = random.choice(other_colors)

    if mode == MODE_SAME_MAJORITY:
        winner_count = random.randint(vote_count // 2 + 1, vote_count - 1)
        colors = [pattern_color] * winner_count + [rival] * (vote_count - winner_count)
        random.shuffle(colors)
        return colors

    if mode == MODE_DIFFERENT_MAJORITY:
        winner_count = random.randint(vote_count // 2 + 1, vote_count - 1)
        colors = [rival] * winner_count + [pattern_color] * (vote_count - winner_count)
        random.shuffle(colors)
        return colors

    spare = None
    if vote_count == 5:
        spare_choices = [color for color in COLORS if color not in (pattern_color, rival)]
        spare = random.choice(spare_choices)

    if mode == MODE_SAME_TIE:
        colors = [pattern_color, pattern_color, rival, rival]
        if spare is not None:
            colors.append(spare)
        head = colors[0]
        tail = colors[1:]
        random.shuffle(tail)
        return [head] + tail

    colors = [rival, rival, pattern_color, pattern_color]
    if spare is not None:
        colors.append(spare)
    head = colors[0]
    tail = colors[1:]
    random.shuffle(tail)
    return [head] + tail


def _mode_sequence(count: int) -> list[str]:
    modes = list(PANEL_MODES)
    random.shuffle(modes)
    if count == len(modes):
        return modes
    while len(modes) < count:
        modes.append(random.choice(PANEL_MODES))
    random.shuffle(modes)
    return modes


def _gen_panel(panel_h: int, panel_w: int, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PANEL_RETRY_LIMIT):
        pattern_color = random.choice(COLORS)
        shape = _random_shape(panel_h, panel_w)
        shape_h, shape_w = _bbox(shape)
        top = random.randint(0, panel_h - shape_h)
        left = random.randint(0, panel_w - shape_w)
        pattern = _shifted(shape, top, left)

        vote_count = _vote_count_for_mode(mode)
        candidates = _candidate_vote_cells(panel_h, panel_w, pattern)
        votes = _sample_nonadjacent_cells(candidates, vote_count)
        if votes is None:
            continue

        vote_cells = sorted(votes)
        colors = _vote_colors(pattern_color, mode, vote_count)

        panel = np.full((panel_h, panel_w), BLACK, dtype=int)
        expected = np.full((panel_h, panel_w), BLACK, dtype=int)
        for r, c in pattern:
            panel[r, c] = pattern_color
        winner = colors[0] if mode in (MODE_SAME_TIE, MODE_DIFFERENT_TIE) else None
        for (r, c), color in zip(vote_cells, colors):
            panel[r, c] = color
        if winner is None:
            winner = pattern_color if mode == MODE_SAME_MAJORITY else next(color for color in colors if color != pattern_color)
        for r, c in pattern:
            expected[r, c] = winner

        return panel, expected

    return None


def _assemble_grid(
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    height = panel_rows * panel_h + (panel_rows - 1)
    width = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    out = np.full((height, width), BLACK, dtype=int)

    for r in range(panel_h, height, panel_h + 1):
        grid[r, :] = SEPARATOR
        out[r, :] = SEPARATOR
    for c in range(panel_w, width, panel_w + 1):
        grid[:, c] = SEPARATOR
        out[:, c] = SEPARATOR

    modes = _mode_sequence(panel_rows * panel_cols)
    idx = 0
    for pr in range(panel_rows):
        r0 = pr * (panel_h + 1)
        for pc in range(panel_cols):
            c0 = pc * (panel_w + 1)
            built = _gen_panel(panel_h, panel_w, modes[idx])
            if built is None:
                return None
            panel, expected = built
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
            out[r0 : r0 + panel_h, c0 : c0 + panel_w] = expected
            idx += 1

    return grid, out


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols, panel_h, panel_w = random.choice(LAYOUT_OPTIONS)
    built = _assemble_grid(panel_rows, panel_cols, panel_h, panel_w)
    if built is None:
        return None
    grid, expected = built
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-voting ARC input")
