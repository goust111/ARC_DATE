import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GREY = 5
LEGEND_COLORS = [1, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    reserved = {bg, RED, GREY}
    for c in (0, w - 1):
        rows = [r for r in range(h) if int(grid[r, c]) not in reserved]
        if not 2 <= len(rows) <= 5:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(colors) != len(set(colors)):
            continue
        if any(int(grid[r, c]) != bg for r in range(h) if r not in rows):
            continue
        candidates.append((len(rows), c, rows, colors))
    assert candidates, "legend not found"
    _, c, rows, colors = max(candidates, key=lambda item: (item[0], -item[1]))
    legend_positions = {(r, c) for r in rows}
    length_map = {color: idx + 1 for idx, color in enumerate(colors)}
    return length_map, legend_positions


def _find_trunk(
    grid: np.ndarray, legend_positions: set[tuple[int, int]], legend_colors: set[int]
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int, int]]]:
    mask = np.isin(grid, [GREY, *sorted(legend_colors)])
    for r, c in legend_positions:
        mask[r, c] = False

    candidates = []
    for comp in _components(mask):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            ordered = sorted(comp, key=lambda pt: pt[1])
            orientation = "h"
        elif len(cols) == 1:
            ordered = sorted(comp, key=lambda pt: pt[0])
            orientation = "v"
        else:
            continue
        values = [int(grid[r, c]) for r, c in ordered]
        if GREY not in values:
            continue
        if not any(v in legend_colors for v in values):
            continue
        if orientation == "h":
            coords = [c for _, c in ordered]
        else:
            coords = [r for r, _ in ordered]
        if coords != list(range(coords[0], coords[-1] + 1)):
            continue
        markers = [(r, c, int(grid[r, c])) for r, c in ordered if int(grid[r, c]) != GREY]
        candidates.append((len(ordered), len(markers), orientation, ordered, markers))

    assert candidates, "trunk not found"
    _, _, orientation, ordered, markers = max(candidates, key=lambda item: (item[0], item[1]))
    return orientation, ordered, markers


def _marker_direction(grid: np.ndarray, orientation: str, r: int, c: int) -> tuple[int, int]:
    h, w = grid.shape
    if orientation == "h":
        choices = [(-1, 0), (1, 0)]
    else:
        choices = [(0, -1), (0, 1)]
    arrows = []
    for dr, dc in choices:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == RED:
            arrows.append((dr, dc))
    assert len(arrows) == 1, "each marker must have exactly one perpendicular arrow"
    return arrows[0]


def _draw_branches(
    base_grid: np.ndarray,
    markers: list[tuple[int, int, int, tuple[int, int]]],
    length_map: dict[int, int],
    bg: int,
) -> np.ndarray:
    out = np.array(base_grid, copy=True)
    out[out == RED] = bg
    h, w = out.shape
    for r, c, color, direction in markers:
        dr, dc = direction
        for step in range(1, length_map[color] + 1):
            nr = r + dr * step
            nc = c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(out[nr, nc]) != bg:
                break
            out[nr, nc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored side-branches from a marked trunk using an edge legend as the branch-length key.

    Input structure:
    - The background is black.
    - One grid edge contains a vertical legend column of distinct single-pixel colors; reading top to bottom maps those colors to lengths 1..N.
    - The main area contains one straight grey trunk, horizontal or vertical, with some trunk cells replaced by legend colors.
    - Each colored trunk marker has one adjacent red pixel on the side perpendicular to the trunk, indicating the branch direction.

    Rule:
    - Keep the legend, the trunk, and the colored trunk markers.
    - For each colored trunk marker, look up its branch length from the legend and grow a same-color straight branch from the next cell in the red-indicated direction.
    - Clear every red direction marker to background before drawing branches.
    - Stop a branch early if the next cell would leave the grid or hit any non-background pixel.

    Key insight:
    - The edge legend does not recolor anything directly; it encodes branch lengths, while the red adjacent pixels encode which perpendicular direction each branch should grow.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    length_map, legend_positions = _find_legend(g, bg)
    orientation, _trunk, raw_markers = _find_trunk(g, legend_positions, set(length_map))
    markers = []
    for r, c, color in raw_markers:
        direction = _marker_direction(g, orientation, r, c)
        markers.append((r, c, color, direction))
    return _draw_branches(g, markers, length_map, bg)


def _choose_extreme_or_inner(rng: random.Random, values: list[int]) -> int:
    assert values
    values = sorted(values)
    extreme = sorted(set(values[:2] + values[-2:]))
    inner = [v for v in values if v not in extreme]
    if inner and rng.random() < 0.5:
        return rng.choice(inner)
    return rng.choice(extreme)


def _choose_segment_positions(rng: random.Random, length: int, count: int) -> list[int]:
    assert 1 <= count <= (length + 1) // 2
    base = sorted(rng.sample(range(length - count + 1), count))
    return [base[idx] + idx for idx in range(count)]


def _balanced_directions(
    rng: random.Random, count: int, option_a: tuple[int, int], option_b: tuple[int, int]
) -> list[tuple[int, int]]:
    counts = [count // 2, count // 2]
    counts[rng.randrange(2)] += count % 2
    dirs = [option_a] * counts[0] + [option_b] * counts[1]
    rng.shuffle(dirs)
    return dirs


def _sample_marker_colors(
    rng: random.Random, legend_colors: list[int], marker_count: int
) -> list[int]:
    used_count = rng.randint(2, min(len(legend_colors), marker_count))
    chosen = rng.sample(legend_colors, used_count)
    colors = chosen[:]
    while len(colors) < marker_count:
        colors.append(rng.choice(chosen))
    rng.shuffle(colors)
    return colors


def _expected_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int, tuple[int, int]]],
    legend_colors: list[int],
) -> np.ndarray:
    length_map = {color: idx + 1 for idx, color in enumerate(legend_colors)}
    return _draw_branches(grid, sorted(markers), length_map, BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 26)
    w = rng.randint(12, 26)
    orientation = rng.choice(["h", "v"])
    legend_len = rng.randint(2, 5)
    legend_colors = rng.sample(LEGEND_COLORS, legend_len)
    legend_side = rng.choice(["left", "right"])

    grid = np.full((h, w), BG, dtype=int)

    legend_col = 0 if legend_side == "left" else w - 1
    legend_start = rng.randint(0, h - legend_len)
    for idx, color in enumerate(legend_colors):
        grid[legend_start + idx, legend_col] = color

    markers: list[tuple[int, int, int, tuple[int, int]]] = []

    if orientation == "h":
        row = _choose_extreme_or_inner(rng, list(range(1, h - 1)))
        min_c = 2 if legend_side == "left" else 1
        max_c = w - 2 if legend_side == "left" else w - 3
        if max_c - min_c + 1 < 6:
            return None
        trunk_len = rng.randint(6, max_c - min_c + 1)
        start_c = rng.randint(min_c, max_c - trunk_len + 1)
        end_c = start_c + trunk_len - 1
        grid[row, start_c : end_c + 1] = GREY

        max_markers = min(5, (trunk_len + 1) // 2)
        marker_count = rng.randint(2, max_markers)
        positions = _choose_segment_positions(rng, trunk_len, marker_count)
        colors = _sample_marker_colors(rng, legend_colors, marker_count)
        dirs = _balanced_directions(rng, marker_count, (-1, 0), (1, 0))

        for offset, color, direction in zip(positions, colors, dirs):
            c = start_c + offset
            grid[row, c] = color
            ar, ac = row + direction[0], c
            if grid[ar, ac] != BG:
                return None
            grid[ar, ac] = RED
            markers.append((row, c, color, direction))
    else:
        min_c = 2 if legend_side == "left" else 1
        max_c = w - 2 if legend_side == "left" else w - 3
        if min_c > max_c:
            return None
        col = _choose_extreme_or_inner(rng, list(range(min_c, max_c + 1)))
        min_r, max_r = 1, h - 2
        if max_r - min_r + 1 < 6:
            return None
        trunk_len = rng.randint(6, max_r - min_r + 1)
        start_r = rng.randint(min_r, max_r - trunk_len + 1)
        end_r = start_r + trunk_len - 1
        grid[start_r : end_r + 1, col] = GREY

        max_markers = min(5, (trunk_len + 1) // 2)
        marker_count = rng.randint(2, max_markers)
        positions = _choose_segment_positions(rng, trunk_len, marker_count)
        colors = _sample_marker_colors(rng, legend_colors, marker_count)
        dirs = _balanced_directions(rng, marker_count, (0, -1), (0, 1))

        for offset, color, direction in zip(positions, colors, dirs):
            r = start_r + offset
            grid[r, col] = color
            ar, ac = r, col + direction[1]
            if grid[ar, ac] != BG:
                return None
            grid[ar, ac] = RED
            markers.append((r, col, color, direction))

    expected = _expected_output(grid, markers, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GEN_RETRIES} attempts")
