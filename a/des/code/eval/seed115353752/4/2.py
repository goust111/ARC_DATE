from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4


def _color_rank(priority: dict[int, int], color: int) -> int:
    return priority.get(int(color), len(priority) + 10)


def _priority_from_legend(row: np.ndarray) -> list[int]:
    colors = [int(v) for v in row if int(v) != BG]
    assert colors, "legend must contain at least one color"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _trace_side(core: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int | None, int]:
    h, w = core.shape
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        steps += 1
        value = int(core[nr, nc])
        if value == BG:
            return None, steps
        if value != YELLOW:
            return value, steps
        nr += dr
        nc += dc
    return None, steps


def _resolve_axis_color(side_infos: list[tuple[int | None, int]], priority: dict[int, int]) -> int | None:
    stats: dict[int, tuple[int, int]] = {}
    for color, dist in side_infos:
        if color is None:
            continue
        count, best_dist = stats.get(color, (0, 10**9))
        stats[color] = (count + 1, min(best_dist, dist))
    if not stats:
        return None
    return min(stats, key=lambda color: (-stats[color][0], stats[color][1], _color_rank(priority, color)))


def _infer_cross_colors(core: np.ndarray, r: int, c: int, priority: dict[int, int]) -> tuple[int | None, int | None]:
    left = _trace_side(core, r, c, 0, -1)
    right = _trace_side(core, r, c, 0, 1)
    up = _trace_side(core, r, c, -1, 0)
    down = _trace_side(core, r, c, 1, 0)
    h_color = _resolve_axis_color([left, right], priority)
    v_color = _resolve_axis_color([up, down], priority)
    return h_color, v_color


def _paint_if_stronger(out: np.ndarray, r: int, c: int, color: int, priority: dict[int, int]) -> None:
    if _color_rank(priority, color) < _color_rank(priority, int(out[r, c])):
        out[r, c] = color


def _propagate(
    out: np.ndarray,
    core: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    winner: int,
    priority: dict[int, int],
) -> None:
    h, w = core.shape
    r, c = start_r + dr, start_c + dc
    while 0 <= r < h and 0 <= c < w:
        value = int(core[r, c])
        if value == BG or value == YELLOW:
            break
        _paint_if_stronger(out, r, c, winner, priority)
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve yellow-marked path crossings by legend priority, then erase the legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid plus legend row"

    priority_colors = _priority_from_legend(g[-1])
    priority = {color: idx for idx, color in enumerate(priority_colors)}

    core = g[:-1].copy()
    out = core.copy()

    for rr, cc in np.argwhere(core == YELLOW):
        r, c = int(rr), int(cc)
        h_color, v_color = _infer_cross_colors(core, r, c, priority)

        if h_color is None and v_color is None:
            out[r, c] = BG
            continue
        if h_color is None:
            out[r, c] = int(v_color)
            continue
        if v_color is None:
            out[r, c] = int(h_color)
            continue

        if h_color == v_color:
            out[r, c] = int(h_color)
            continue

        if _color_rank(priority, h_color) < _color_rank(priority, v_color):
            winner = int(h_color)
            loser_dirs = [(-1, 0), (1, 0)]
        else:
            winner = int(v_color)
            loser_dirs = [(0, -1), (0, 1)]

        out[r, c] = winner
        for dr, dc in loser_dirs:
            _propagate(out, core, r, c, dr, dc, winner, priority)

    result = np.zeros_like(g)
    result[:-1] = out
    return result


# --------------- generator ---------------

COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
CORE_H_RANGE = (10, 18)
GRID_W_RANGE = (14, 24)
LINE_GAP = 2
REGION_GAP = 3
GEN_RETRIES = 400


def _choose_spaced_values(count: int, lo: int, hi: int, min_gap: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    if hi < lo or hi - lo + 1 < count:
        return None
    values = list(range(lo, hi + 1))
    for _ in range(80):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    if hi - lo + 1 >= 1 + (count - 1) * min_gap:
        picks = [lo + i * min_gap for i in range(count)]
        if picks[-1] <= hi:
            return picks
    return None


def _cluster_spec(hi: int, lo: int, winner_orientation: str, rng: random.Random) -> dict[str, int | str]:
    winner_count = rng.randint(2, 3) if rng.random() < 0.7 else 1
    loser_count = rng.randint(1, 3)
    if winner_count == 1 and loser_count == 1 and rng.random() < 0.5:
        loser_count = 2

    if winner_orientation == "horizontal":
        h_color, v_color = hi, lo
        h_count, v_count = winner_count, loser_count
    else:
        h_color, v_color = lo, hi
        h_count, v_count = loser_count, winner_count

    min_h = max(5, 1 + (h_count - 1) * LINE_GAP)
    min_w = max(5, 1 + (v_count - 1) * LINE_GAP)
    return {
        "winner_orientation": winner_orientation,
        "h_color": h_color,
        "v_color": v_color,
        "h_count": h_count,
        "v_count": v_count,
        "min_h": min_h,
        "min_w": min_w,
    }


def _build_regions(
    specs: list[dict[str, int | str]],
    legend_color_count: int,
    rng: random.Random,
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    legend_min_w = 2 * legend_color_count - 1

    if len(specs) == 1:
        spec = specs[0]
        core_h_lo = max(CORE_H_RANGE[0], int(spec["min_h"]))
        grid_w_lo = max(GRID_W_RANGE[0], legend_min_w, int(spec["min_w"]))
        if core_h_lo > CORE_H_RANGE[1] or grid_w_lo > GRID_W_RANGE[1]:
            return None
        core_h = rng.randint(core_h_lo, CORE_H_RANGE[1])
        grid_w = rng.randint(grid_w_lo, GRID_W_RANGE[1])
        return core_h, grid_w, [(0, 0, core_h - 1, grid_w - 1)]

    a, b = specs
    layouts: list[str] = []
    if GRID_W_RANGE[1] >= int(a["min_w"]) + int(b["min_w"]) + REGION_GAP and CORE_H_RANGE[1] >= max(int(a["min_h"]), int(b["min_h"])):
        layouts.append("vertical")
    if CORE_H_RANGE[1] >= int(a["min_h"]) + int(b["min_h"]) + REGION_GAP and GRID_W_RANGE[1] >= max(int(a["min_w"]), int(b["min_w"]), legend_min_w):
        layouts.append("horizontal")
    if not layouts:
        return None

    layout = rng.choice(layouts)
    if layout == "vertical":
        core_h_lo = max(CORE_H_RANGE[0], int(a["min_h"]), int(b["min_h"]))
        grid_w_lo = max(GRID_W_RANGE[0], legend_min_w, int(a["min_w"]) + int(b["min_w"]) + REGION_GAP)
        if core_h_lo > CORE_H_RANGE[1] or grid_w_lo > GRID_W_RANGE[1]:
            return None
        core_h = rng.randint(core_h_lo, CORE_H_RANGE[1])
        grid_w = rng.randint(grid_w_lo, GRID_W_RANGE[1])
        left_w = rng.randint(int(a["min_w"]), grid_w - REGION_GAP - int(b["min_w"]))
        regions = [
            (0, 0, core_h - 1, left_w - 1),
            (0, left_w + REGION_GAP, core_h - 1, grid_w - 1),
        ]
        return core_h, grid_w, regions

    core_h_lo = max(CORE_H_RANGE[0], int(a["min_h"]) + int(b["min_h"]) + REGION_GAP)
    grid_w_lo = max(GRID_W_RANGE[0], legend_min_w, int(a["min_w"]), int(b["min_w"]))
    if core_h_lo > CORE_H_RANGE[1] or grid_w_lo > GRID_W_RANGE[1]:
        return None
    core_h = rng.randint(core_h_lo, CORE_H_RANGE[1])
    grid_w = rng.randint(grid_w_lo, GRID_W_RANGE[1])
    top_h = rng.randint(int(a["min_h"]), core_h - REGION_GAP - int(b["min_h"]))
    regions = [
        (0, 0, top_h - 1, grid_w - 1),
        (top_h + REGION_GAP, 0, core_h - 1, grid_w - 1),
    ]
    return core_h, grid_w, regions


def _paint_cluster(
    core: np.ndarray,
    region: tuple[int, int, int, int],
    spec: dict[str, int | str],
    rng: random.Random,
) -> bool:
    r0, c0, r1, c1 = region
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    min_h = int(spec["min_h"])
    min_w = int(spec["min_w"])
    if region_h < min_h or region_w < min_w:
        return False

    rect_h = rng.randint(min_h, region_h)
    rect_w = rng.randint(min_w, region_w)
    top = rng.randint(r0, r1 - rect_h + 1)
    left = rng.randint(c0, c1 - rect_w + 1)

    if rng.random() < 0.45:
        top = rng.choice([r0, r1 - rect_h + 1])
    if rng.random() < 0.45:
        left = rng.choice([c0, c1 - rect_w + 1])

    rows = _choose_spaced_values(int(spec["h_count"]), top, top + rect_h - 1, LINE_GAP, rng)
    cols = _choose_spaced_values(int(spec["v_count"]), left, left + rect_w - 1, LINE_GAP, rng)
    if rows is None or cols is None:
        return False

    h_color = int(spec["h_color"])
    v_color = int(spec["v_color"])

    core_slice = core[top : top + rect_h, left : left + rect_w]
    if np.any(core_slice != BG):
        return False

    for row in rows:
        core[row, left : left + rect_w] = h_color
    for col in cols:
        core[top : top + rect_h, col] = v_color
    for row in rows:
        for col in cols:
            core[row, col] = YELLOW
    return True


def _legend_row(width: int, priority_colors: list[int], rng: random.Random) -> np.ndarray:
    needed = 2 * len(priority_colors) - 1
    assert width >= needed
    start = rng.randint(0, width - needed)
    row = np.zeros(width, dtype=int)
    col = start
    for color in priority_colors:
        row[col] = color
        col += 2
    return row


def _cluster_pairs(priority_colors: list[int]) -> list[tuple[int, int]]:
    if len(priority_colors) == 2:
        return [(priority_colors[0], priority_colors[1])]
    if len(priority_colors) == 3:
        return [(priority_colors[0], priority_colors[1]), (priority_colors[1], priority_colors[2])]
    return [(priority_colors[0], priority_colors[1]), (priority_colors[2], priority_colors[3])]


def _validate_grid(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and grid.shape[0] >= 2
    core = grid[:-1]
    priority_colors = _priority_from_legend(grid[-1])
    priority = {color: idx for idx, color in enumerate(priority_colors)}

    used = {int(v) for v in core.ravel() if int(v) not in (BG, YELLOW)}
    assert used
    assert set(priority_colors).issubset(used), "every legend color should appear in the paths"
    assert np.any(core == YELLOW), "must contain at least one crossing"

    for rr, cc in np.argwhere(core == YELLOW):
        h_color, v_color = _infer_cross_colors(core, int(rr), int(cc), priority)
        assert h_color is not None and v_color is not None
        assert h_color != v_color

    out = main(grid)
    assert out.shape == grid.shape
    assert np.all(out[-1] == BG), "legend row must be erased"
    assert not np.any(out[:-1] == YELLOW), "cross markers must be resolved"
    assert np.any(out != grid), "transformation should change the grid"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 4)
    priority_colors = rng.sample(COLOR_POOL, color_count)
    pairs = _cluster_pairs(priority_colors)

    if len(pairs) == 1:
        winner_orientations = [rng.choice(["horizontal", "vertical"])]
    else:
        first = rng.choice(["horizontal", "vertical"])
        second = "vertical" if first == "horizontal" else "horizontal"
        winner_orientations = [first, second]

    specs = [_cluster_spec(hi, lo, orient, rng) for (hi, lo), orient in zip(pairs, winner_orientations)]
    layout = _build_regions(specs, len(priority_colors), rng)
    if layout is None:
        return None

    core_h, grid_w, regions = layout
    core = np.zeros((core_h, grid_w), dtype=int)
    for spec, region in zip(specs, regions):
        if not _paint_cluster(core, region, spec, rng):
            return None

    grid = np.vstack([core, _legend_row(grid_w, priority_colors, rng)])
    _validate_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
