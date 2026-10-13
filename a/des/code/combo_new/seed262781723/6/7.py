import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 7
LINE_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_ATTEMPTS = 400


def _majority_color(values: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(int(v) for v in values if int(v) not in excluded)
    assert counts, "expected at least one visible line color"
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "line color must be uniquely recoverable by majority"
    return int(best[0])


def _line_maps(g: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    full_rows = [int(r) for r in np.flatnonzero(np.all(g != bg, axis=1))]
    full_cols = [int(c) for c in np.flatnonzero(np.all(g != bg, axis=0))]
    assert full_rows and full_cols, "expected at least one full horizontal and vertical line"

    row_colors = {
        r: _majority_color(g[r, :], {bg, MARKER})
        for r in full_rows
    }
    col_colors = {
        c: _majority_color(g[:, c], {bg, MARKER})
        for c in full_cols
    }
    return row_colors, col_colors


def _markers(g: np.ndarray, bg: int) -> list[dict[str, tuple[int, int]]]:
    H, W = g.shape
    markers: list[dict[str, tuple[int, int]]] = []
    for r, c in np.argwhere(g == MARKER):
        r = int(r)
        c = int(c)
        assert r in {0, H - 1} or c in {0, W - 1}, "markers must lie on the border"

        neighbors: list[tuple[int, int]] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) not in {bg, MARKER}:
                neighbors.append((int(rr), int(cc)))

        assert len(neighbors) == 1, "each marker must point to exactly one inward line cell"
        rr, cc = neighbors[0]
        markers.append(
            {
                "marker": (r, c),
                "start": (rr, cc),
                "step": (int(rr - r), int(cc - c)),
            }
        )

    assert markers, "expected at least one start marker"
    markers.sort(key=lambda item: item["marker"])
    return markers


def _trace_positions(base: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = base.shape
    dr, dc = step
    assert (dr == 0) ^ (dc == 0), "path step must be axis-aligned"

    positions: list[tuple[int, int]] = []
    r, c = start
    while 0 <= r < H and 0 <= c < W and int(base[r, c]) != MARKER and int(base[r, c]) != BG:
        positions.append((int(r), int(c)))
        r += dr
        c += dc

    assert positions, "a marker must lead into a non-empty line"
    return positions


def _paint_path(
    base: np.ndarray,
    out: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> None:
    positions = _trace_positions(base, start, step)
    dr, dc = step

    if dc != 0:
        line_color = int(row_colors[start[0]])
        crossings = [
            (i, int(col_colors[c]))
            for i, (_r, c) in enumerate(positions)
            if int(c) in col_colors
        ]
    else:
        line_color = int(col_colors[start[1]])
        crossings = [
            (i, int(row_colors[r]))
            for i, (r, _c) in enumerate(positions)
            if int(r) in row_colors
        ]

    hidden: set[int] = set()
    crossing_modes: dict[int, tuple[bool, int]] = {}
    for cross_index, (pos_index, cross_color) in enumerate(crossings):
        is_over = (cross_index % 2) == 0
        crossing_modes[int(pos_index)] = (bool(is_over), int(cross_color))
        if not is_over:
            for offset in (-1, 0, 1):
                idx = pos_index + offset
                if 0 <= idx < len(positions):
                    hidden.add(int(idx))

    for idx in hidden:
        r, c = positions[idx]
        out[r, c] = base[r, c]

    current_color = int(line_color)
    for idx, (r, c) in enumerate(positions):
        if idx not in hidden:
            out[r, c] = int(current_color)

        event = crossing_modes.get(int(idx))
        if event is None:
            continue

        is_over, cross_color = event
        if not is_over:
            out[r, c] = int(cross_color)
        current_color = int(cross_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave each edge-marked path along the sparse line grid, alternating over and under crossings while inheriting the crossed line's color.

    Input structure:
    - The background is black.
    - Several thin horizontal and vertical colored lines form a sparse crossing grid.
    - Orange border markers sit at selected line endpoints and indicate the inward travel direction by their placement on the line extension.

    Rule:
    - Start from each orange marker and move inward along its attached line.
    - At the first orthogonal crossing, pass over it: keep the path visible through the intersection.
    - At the second crossing, pass under it: leave the crossed line visible and hide the path on the intersection cell and the immediately adjacent cell on each side.
    - Continue alternating over and under at later crossings.
    - After every crossing, change the path color to the color of the crossed line, and stop when the path reaches the opposite border.

    Key insight:
    - The visible result is produced by two independent states that advance together: an over/under toggle per crossing and a current path color inherited from the most recent crossed line.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    row_colors, col_colors = _line_maps(g, bg)
    starts = _markers(g, bg)

    out = g.copy()
    base = g.copy()
    for item in starts:
        _paint_path(base, out, item["start"], item["step"], row_colors, col_colors)
    return out


def _sample_spaced(low: int, high: int, count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(low, high + 1))
    if len(values) < count:
        return None

    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _draw_scene(
    H: int,
    W: int,
    active_horizontal: bool,
    same_positions: list[int],
    passive_positions: list[int],
    active_positions: set[int],
    same_colors: list[int],
    passive_colors: list[int],
    directions: dict[int, int],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)

    if active_horizontal:
        row_colors = dict(zip(same_positions, same_colors))
        col_colors = dict(zip(passive_positions, passive_colors))

        for r, color in row_colors.items():
            grid[int(r), :] = int(color)
        for r, step in directions.items():
            grid[int(r), 0 if step > 0 else W - 1] = MARKER
        for c, color in col_colors.items():
            grid[:, int(c)] = int(color)
    else:
        col_colors = dict(zip(same_positions, same_colors))
        row_colors = dict(zip(passive_positions, passive_colors))

        for c, color in col_colors.items():
            grid[:, int(c)] = int(color)
        for c, step in directions.items():
            grid[0 if step > 0 else H - 1, int(c)] = MARKER
        for r, color in row_colors.items():
            grid[int(r), :] = int(color)

    assert len(active_positions) == len(directions)
    return grid


def _generate_attempt() -> np.ndarray | None:
    active_horizontal = random.random() < 0.5
    n_active = random.randint(1, 3)
    n_extra = random.choice([0, 0, 1])
    n_passive = random.randint(2, 4)
    n_same = n_active + n_extra
    total_lines = n_same + n_passive
    if total_lines > len(LINE_COLORS):
        return None

    if active_horizontal:
        min_h = max(14, 5 + (n_same - 1) * 3)
        min_w = max(14, 5 + (n_passive - 1) * 4)
        H = random.randint(min_h, 28)
        W = random.randint(min_w, 28)
        same_positions = _sample_spaced(2, H - 3, n_same, 3)
        passive_positions = _sample_spaced(2, W - 3, n_passive, 4)
    else:
        min_h = max(14, 5 + (n_passive - 1) * 4)
        min_w = max(14, 5 + (n_same - 1) * 3)
        H = random.randint(min_h, 28)
        W = random.randint(min_w, 28)
        same_positions = _sample_spaced(2, W - 3, n_same, 3)
        passive_positions = _sample_spaced(2, H - 3, n_passive, 4)

    if same_positions is None or passive_positions is None:
        return None

    colors = random.sample(LINE_COLORS, total_lines)
    same_colors = colors[:n_same]
    passive_colors = colors[n_same:]

    active_positions = set(random.sample(same_positions, n_active))
    directions = {int(pos): random.choice([-1, 1]) for pos in active_positions}

    grid = _draw_scene(
        H=H,
        W=W,
        active_horizontal=active_horizontal,
        same_positions=same_positions,
        passive_positions=passive_positions,
        active_positions=active_positions,
        same_colors=same_colors,
        passive_colors=passive_colors,
        directions=directions,
    )

    if np.any(grid == BG) is False:
        return None
    if int(np.sum(grid == MARKER)) != n_active:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-grid instance")
