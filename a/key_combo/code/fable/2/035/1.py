from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BG = 0
PARTICLE = 8
GREY = 5
WALL_COLORS = (1, 2, 3, 4, 6, 7, 9)
UP = -1
DOWN = 1
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    legend_map: dict[int, int] = {}
    legend_mask = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        choices: list[tuple[int, int, int]] = []
        if r > 0:
            color = int(grid[r - 1, c])
            if color not in {bg, GREY, PARTICLE}:
                choices.append((color, DOWN, r - 1))
        if r + 1 < h:
            color = int(grid[r + 1, c])
            if color not in {bg, GREY, PARTICLE}:
                choices.append((color, UP, r + 1))
        assert len(choices) == 1, "each grey legend cell must pair with exactly one colored cell"
        color, direction, wall_r = choices[0]
        prev = legend_map.setdefault(int(color), int(direction))
        assert prev == int(direction), "legend color cannot map to two directions"
        legend_mask[r, c] = True
        legend_mask[wall_r, c] = True

    assert legend_map, "expected at least one legend item"
    scene_colors = set(map(int, np.unique(grid))) - {bg, GREY, PARTICLE}
    assert scene_colors == set(legend_map), "every non-legend foreground color must appear in the legend"
    return legend_map, legend_mask


def _wall_grid(grid: np.ndarray, legend_map: dict[int, int], legend_mask: np.ndarray) -> np.ndarray:
    wall = np.zeros_like(grid)
    for color in legend_map:
        mask = (grid == int(color)) & (~legend_mask)
        wall[mask] = int(color)
    return wall


def _trace_particle(start_r: int, wall: np.ndarray, legend_map: dict[int, int]) -> tuple[np.ndarray, list[int]]:
    h, w = wall.shape
    r = int(start_r)
    c = 0
    vertical = 0
    visited = np.zeros((h, w), dtype=bool)
    visited[r, c] = True
    hits: list[int] = []

    for _ in range(h * w * 4):
        if vertical == 0:
            if c == w - 1:
                return visited, hits
            color = int(wall[r, c + 1])
            if color == 0:
                c += 1
                visited[r, c] = True
                continue
            vertical = int(legend_map[color])
            hits.append(color)

        r += vertical
        assert 0 <= r < h, "particle detour must remain inside the grid"
        visited[r, c] = True
        if c == w - 1 or int(wall[r, c + 1]) == 0:
            vertical = 0

    raise AssertionError("particle trace did not terminate")


def _render_output(
    grid: np.ndarray,
    wall: np.ndarray,
    legend_mask: np.ndarray,
    start_rows: list[int],
    legend_map: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    preserve = (wall != 0) | legend_mask
    path = np.zeros_like(grid, dtype=bool)
    for row in start_rows:
        trace, _hits = _trace_particle(int(row), wall, legend_map)
        path |= trace
    out[path & (~preserve)] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace every cyan particle across the grid, detouring around colored vertical walls according to the bottom-left legend.

    Input structure:
    - The background is black, with several cyan particle start cells on the left border.
    - Away from the left border there are several partial vertical wall segments, each in its own color.
    - In the lower-left legend, each wall color appears in a vertical two-cell item together with one grey cell.

    Rule:
    - Each particle starts on the left border and paints its visited cells cyan while moving one step at a time.
    - A particle normally moves right; if the cell immediately to its right is a wall, the legend decides whether it detours upward or downward along its current column.
    - While detouring, after each vertical step the particle checks its right neighbor again; once the right side is no longer a wall, the next step resumes rightward motion.
    - All wall cells and legend cells stay unchanged, and all particle traces remain cyan in the output.

    Key insight:
    - The legend colors do not recolor anything directly; each legend item assigns a wall color to an avoidance direction, and that behavior controls the full traced path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    particle_coords = np.argwhere(g == PARTICLE)
    assert particle_coords.size, "expected at least one cyan particle"
    assert np.all(particle_coords[:, 1] == 0), "particles must start on the left border"

    legend_map, legend_mask = _parse_legend(g, bg)
    wall = _wall_grid(g, legend_map, legend_mask)
    start_rows = [int(r) for r in particle_coords[:, 0]]
    return _render_output(g, wall, legend_mask, start_rows, legend_map)


def _sample_spaced_columns(rng: np.random.Generator, lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count:
        return None
    for _ in range(300):
        cols = sorted(int(x) for x in rng.choice(values, size=count, replace=False))
        if all(b - a >= int(min_gap) for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_walls(
    rng: np.random.Generator,
    h: int,
    w: int,
    colors: list[int],
    directions: list[int],
) -> list[dict[str, int]] | None:
    legend_last_col = 1 + 2 * (len(colors) - 1)
    col_lo = legend_last_col + 4
    col_hi = w - 2
    cols = _sample_spaced_columns(rng, col_lo, col_hi, len(colors), min_gap=3)
    if cols is None:
        return None

    max_bottom = h - 5
    max_len = min(9, max_bottom)
    if max_len < 4:
        return None

    walls = []
    order = list(range(len(colors)))
    rng.shuffle(order)
    for idx in order:
        length = int(rng.integers(4, max_len + 1))
        top_hi = max_bottom - length + 1
        if top_hi < 1:
            return None
        top = int(rng.integers(1, top_hi + 1))
        walls.append(
            {
                "color": int(colors[idx]),
                "dir": int(directions[idx]),
                "col": int(cols[idx]),
                "r0": top,
                "r1": top + length - 1,
            }
        )
    walls.sort(key=lambda wall: wall["col"])
    return walls


def _legend_and_walls_grid(h: int, w: int, walls: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    grid = np.full((h, w), BG, dtype=int)
    legend_mask = np.zeros((h, w), dtype=bool)
    wall = np.zeros((h, w), dtype=int)
    legend_map = {int(item["color"]): int(item["dir"]) for item in walls}

    for item in walls:
        color = int(item["color"])
        c = int(item["col"])
        r0 = int(item["r0"])
        r1 = int(item["r1"])
        grid[r0 : r1 + 1, c] = color
        wall[r0 : r1 + 1, c] = color

    for i, item in enumerate(walls):
        color = int(item["color"])
        c = 1 + 2 * i
        if int(item["dir"]) == UP:
            grid[h - 2, c] = GREY
            grid[h - 1, c] = color
            legend_mask[h - 2, c] = True
            legend_mask[h - 1, c] = True
        else:
            grid[h - 2, c] = color
            grid[h - 1, c] = GREY
            legend_mask[h - 2, c] = True
            legend_mask[h - 1, c] = True

    return grid, wall, legend_mask, legend_map


def _candidate_rows(
    h: int,
    wall: np.ndarray,
    legend_map: dict[int, int],
) -> list[dict[str, object]]:
    info: list[dict[str, object]] = []
    for row in range(h - 3):
        try:
            mask, hits = _trace_particle(row, wall, legend_map)
        except AssertionError:
            continue
        if not hits:
            continue
        info.append({"row": int(row), "mask": mask, "hits": tuple(int(x) for x in hits)})
    return info


def _combo_output(
    base: np.ndarray,
    wall: np.ndarray,
    legend_mask: np.ndarray,
    rows: list[int],
    legend_map: dict[int, int],
) -> np.ndarray | None:
    try:
        return _render_output(base, wall, legend_mask, rows, legend_map)
    except AssertionError:
        return None


def _choose_start_rows(
    rng: np.random.Generator,
    base: np.ndarray,
    wall: np.ndarray,
    legend_mask: np.ndarray,
    legend_map: dict[int, int],
    candidates: list[dict[str, object]],
) -> tuple[list[int], np.ndarray] | None:
    sizes = list(range(2, min(4, len(candidates)) + 1))
    rng.shuffle(sizes)
    target_colors = set(legend_map)
    valid: list[tuple[list[int], np.ndarray, int]] = []

    for size in sizes:
        combos = list(itertools.combinations(range(len(candidates)), size))
        rng.shuffle(combos)
        for combo in combos:
            chosen = [candidates[i] for i in combo]
            rows = sorted(int(item["row"]) for item in chosen)
            if any(b - a < 2 for a, b in zip(rows, rows[1:])):
                continue

            hit_colors = set()
            sequences = set()
            union_mask = np.zeros_like(base, dtype=bool)
            for item in chosen:
                hits = tuple(int(x) for x in item["hits"])
                hit_colors.update(hits)
                sequences.add(hits)
                union_mask |= np.asarray(item["mask"], dtype=bool)

            if hit_colors != target_colors:
                continue
            if {legend_map[color] for color in hit_colors} != {UP, DOWN}:
                continue
            if len(sequences) < 2:
                continue

            expected = _combo_output(base, wall, legend_mask, rows, legend_map)
            if expected is None:
                continue

            up_only = _combo_output(base, wall, legend_mask, rows, {color: UP for color in legend_map})
            down_only = _combo_output(base, wall, legend_mask, rows, {color: DOWN for color in legend_map})
            if up_only is not None and np.array_equal(expected, up_only):
                continue
            if down_only is not None and np.array_equal(expected, down_only):
                continue

            changed = 0
            for color in sorted(hit_colors):
                flipped = dict(legend_map)
                flipped[color] = -flipped[color]
                alt = _combo_output(base, wall, legend_mask, rows, flipped)
                if alt is None or not np.array_equal(expected, alt):
                    changed += 1
            if changed < min(2, len(hit_colors)):
                continue

            area = int(union_mask.sum())
            if area <= len(rows) * 2 or area >= int(base.size * 0.45):
                continue
            valid.append((rows, expected, area))

    if not valid:
        return None

    areas = np.array([item[2] for item in valid], dtype=float)
    weights = areas / areas.sum()
    pick = int(rng.choice(len(valid), p=weights))
    rows, expected, _area = valid[pick]
    return rows, expected


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    wall_count = int(rng.integers(2, 5))
    legend_last_col = 1 + 2 * (wall_count - 1)
    min_width = legend_last_col + 4 + 3 * (wall_count - 1) + 3
    h = int(rng.integers(16, 29))
    w = int(rng.integers(max(18, min_width), 31))

    colors = [int(c) for c in rng.choice(WALL_COLORS, size=wall_count, replace=False)]
    directions = [UP, DOWN] + [int(rng.choice([UP, DOWN])) for _ in range(wall_count - 2)]
    rng.shuffle(directions)
    walls = _build_walls(rng, h, w, colors, directions)
    if walls is None:
        return None

    base, wall, legend_mask, legend_map = _legend_and_walls_grid(h, w, walls)
    candidates = _candidate_rows(h, wall, legend_map)
    if len(candidates) < 2:
        return None

    chosen = _choose_start_rows(rng, base, wall, legend_mask, legend_map, candidates)
    if chosen is None:
        return None
    start_rows, expected = chosen

    inp = np.array(base, copy=True)
    for row in start_rows:
        inp[int(row), 0] = PARTICLE

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-detour ARC input")
