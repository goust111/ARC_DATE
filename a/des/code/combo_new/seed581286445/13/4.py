import random
from collections import Counter

import numpy as np


GEN_RETRIES = 400
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (12, 24)
ACTIVE_COUNT_RANGE = (3, 5)
UNUSED_LEGEND_PROB = 0.35
MAX_PLACEMENT_TRIES = 80
OVERLAP_PROB = 0.7
MAX_OCCUPANCY_RATIO = 0.3


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _plus_cells(center, arm):
    r, c = center
    cells = {(r, c)}
    for d in range(1, arm + 1):
        cells.add((r - d, c))
        cells.add((r + d, c))
        cells.add((r, c - d))
        cells.add((r, c + d))
    return cells


def _center_fits(center, arm, h, w):
    r, c = center
    return arm <= r <= h - 1 - arm and arm + 2 <= c <= w - 1 - arm


def _parse_legend(grid: np.ndarray, bg: int):
    legend = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != bg:
            legend.append((r, color))
    assert legend, "expected a non-empty legend in the leftmost column"
    colors = [color for _, color in legend]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return legend


def _visible_points(grid: np.ndarray, color: int):
    pts = []
    for r, c in np.argwhere(grid == color):
        if int(c) > 0:
            pts.append((int(r), int(c)))
    return pts


def _recover_cross_fast(grid: np.ndarray, bg: int, color: int):
    pts = _visible_points(grid, color)
    if not pts:
        return None

    row_counts = Counter(r for r, _ in pts)
    col_counts = Counter(c for _, c in pts)
    rows = [r for r, count in row_counts.items() if count >= 2]
    cols = [c for c, count in col_counts.items() if count >= 2]
    if len(rows) != 1 or len(cols) != 1:
        return None

    center = (rows[0], cols[0])
    if int(grid[center]) != color:
        return None

    arm = max(abs(r - center[0]) + abs(c - center[1]) for r, c in pts)
    if not 1 <= arm <= 3:
        return None

    cells = _plus_cells(center, arm)
    if not set(pts).issubset(cells):
        return None

    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 1 <= c < w):
            return None
        value = int(grid[r, c])
        if value != bg and value != color:
            return None

    return {"color": color, "center": center, "arm": arm, "cells": cells}


def _recover_crosses_search(grid: np.ndarray, bg: int, legend_colors):
    h, w = grid.shape
    bg_cells = {(r, c) for r in range(h) for c in range(1, w) if int(grid[r, c]) == bg}
    candidates_by_color = []

    for color in legend_colors:
        pts = _visible_points(grid, color)
        if not pts:
            candidates_by_color.append([None])
            continue

        point_set = set(pts)
        seen = set()
        color_candidates = []
        for center in pts:
            for arm in (1, 2, 3):
                cells = _plus_cells(center, arm)
                if any(not (0 <= r < h and 1 <= c < w) for r, c in cells):
                    continue
                if not point_set.issubset(cells):
                    continue
                ok = True
                for r, c in cells:
                    value = int(grid[r, c])
                    if value != bg and value != color:
                        ok = False
                        break
                if not ok:
                    continue
                key = tuple(sorted(cells))
                if key in seen:
                    continue
                seen.add(key)
                color_candidates.append(
                    {
                        "color": color,
                        "center": center,
                        "arm": arm,
                        "cells": cells,
                        "bg_cells": cells & bg_cells,
                    }
                )

        assert color_candidates, f"no valid cross candidate for color {color}"
        candidates_by_color.append(color_candidates)

    future_cover = [set() for _ in range(len(candidates_by_color) + 1)]
    for i in range(len(candidates_by_color) - 1, -1, -1):
        cover = set(future_cover[i + 1])
        for cand in candidates_by_color[i]:
            if cand is not None:
                cover.update(cand["bg_cells"])
        future_cover[i] = cover

    selected = []
    bg_cover_count = {}
    one_cells = set()
    solution = None

    def apply_candidate(cand):
        changed = []
        if cand is None:
            return changed
        for cell in cand["bg_cells"]:
            prev = bg_cover_count.get(cell, 0)
            bg_cover_count[cell] = prev + 1
            if prev == 0:
                one_cells.add(cell)
            elif prev == 1:
                one_cells.discard(cell)
            changed.append(cell)
        return changed

    def undo_candidate(changed):
        for cell in reversed(changed):
            prev = bg_cover_count[cell]
            if prev == 1:
                del bg_cover_count[cell]
                one_cells.discard(cell)
            elif prev == 2:
                bg_cover_count[cell] = 1
                one_cells.add(cell)
            else:
                bg_cover_count[cell] = prev - 1

    def dfs(idx):
        nonlocal solution
        if solution is not None:
            return
        if idx == len(candidates_by_color):
            if not one_cells:
                solution = [cand for cand in selected if cand is not None]
            return

        for cand in candidates_by_color[idx]:
            changed = apply_candidate(cand)
            if all(cell in future_cover[idx + 1] for cell in one_cells):
                selected.append(cand)
                dfs(idx + 1)
                selected.pop()
            undo_candidate(changed)
            if solution is not None:
                return

    dfs(0)
    assert solution is not None, "failed to recover crosses"
    return solution


def _recover_crosses(grid: np.ndarray, bg: int, legend_colors):
    fast = []
    for color in legend_colors:
        if not _visible_points(grid, color):
            continue
        cross = _recover_cross_fast(grid, bg, color)
        if cross is None:
            return _recover_crosses_search(grid, bg, legend_colors)
        fast.append(cross)
    return fast


def _solve_scene(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _parse_legend(g, bg)
    legend_colors = [color for _, color in legend]
    priority = {color: idx for idx, color in enumerate(legend_colors)}
    crosses = _recover_crosses(g, bg, legend_colors)

    covered_by = {}
    for cross in crosses:
        for cell in cross["cells"]:
            covered_by.setdefault(cell, []).append(cross["color"])

    out = g.copy()
    for (r, c), colors in covered_by.items():
        if int(g[r, c]) == bg and len(colors) >= 2:
            out[r, c] = min(colors, key=lambda color: priority[color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore blanked overlap pixels of monochrome crosses using the left-edge legend order as priority.

    Input structure:
    - The most common color is the background.
    - The leftmost column is a priority legend: distinct non-background colors appear as single pixels from top to bottom, with background elsewhere in that column.
    - The remaining columns contain monochrome plus-shaped crosses whose colors come from the legend; each cross has a center pixel and equal arm length 1-3 in the four cardinal directions.
    - When multiple crosses overlap on arm cells, those overlap pixels are shown as background in the input.

    Rule:
    - Recover each cross's full footprint from its visible center-aligned row and column fragments.
    - Find every background pixel that lies inside the footprints of two or more recovered crosses.
    - Fill each such conflict pixel with the color whose legend entry is highest in the left-edge priority order.
    - Keep the legend unchanged, keep already colored cross pixels unchanged, and leave non-conflict background pixels unchanged.

    Key insight:
    - The apparent blanks inside the scene are hidden arm-overlap markers: infer the missing cross coverage first, then use the legend as an explicit priority table to decide which color wins each shared pixel.
    """
    return _solve_scene(grid)


def _make_candidate(arm, h, w, rng, base=None):
    if base is not None:
        base_r, base_c = base["center"]
        base_arm = base["arm"]
        if rng.random() < 0.5:
            lo = max(base_arm, arm) + 1
            hi = base_arm + arm
            if lo <= hi:
                distance = rng.randint(lo, hi)
                options = []
                for sign in (-1, 1):
                    cand_c = base_c + sign * distance
                    if _center_fits((base_r, cand_c), arm, h, w):
                        options.append((base_r, cand_c))
                if options:
                    r, c = rng.choice(options)
                    return {"center": (r, c), "arm": arm}
        else:
            lo = max(base_arm, arm) + 1
            hi = base_arm + arm
            if lo <= hi:
                distance = rng.randint(lo, hi)
                options = []
                for sign in (-1, 1):
                    cand_r = base_r + sign * distance
                    if _center_fits((cand_r, base_c), arm, h, w):
                        options.append((cand_r, base_c))
                if options:
                    r, c = rng.choice(options)
                    return {"center": (r, c), "arm": arm}

    r = rng.randint(arm, h - 1 - arm)
    c = rng.randint(arm + 2, w - 1 - arm)
    return {"center": (r, c), "arm": arm}


def _can_add_cross(crosses, candidate, h, w):
    new_center = candidate["center"]
    new_cells = _plus_cells(candidate["center"], candidate["arm"])
    if not _center_fits(new_center, candidate["arm"], h, w):
        return False
    if any(r < 0 or r >= h or c < 2 or c >= w for r, c in new_cells):
        return False

    overlap_budget = Counter()
    for existing in crosses:
        overlap_budget.update(existing["cells"])
    for cell in new_cells:
        if overlap_budget[cell] >= 2:
            return False

    for existing in crosses:
        if new_center in existing["cells"]:
            return False
        if existing["center"] in new_cells:
            return False
    return True


def _render_scene(h, w, bg, legend_rows, legend_colors, active_colors, crosses):
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for row, color in zip(legend_rows, legend_colors):
        inp[row, 0] = color
        out[row, 0] = color

    priority = {color: idx for idx, color in enumerate(legend_colors)}
    cover_map = {}
    for cross in crosses:
        for cell in cross["cells"]:
            cover_map.setdefault(cell, []).append(cross["color"])

    for cell, owners in cover_map.items():
        r, c = cell
        if len(owners) == 1:
            inp[r, c] = owners[0]
            out[r, c] = owners[0]
        else:
            winner = min(owners, key=lambda color: priority[color])
            out[r, c] = winner

    return inp, out, cover_map


def _scene_is_valid(inp, out, bg, legend_colors, active_colors, crosses, cover_map):
    if _get_bg(inp) != bg:
        return False
    if np.array_equal(inp, out):
        return False

    total_play_cells = inp.shape[0] * (inp.shape[1] - 1)
    occupied = sum(len(cross["cells"]) for cross in crosses)
    if occupied / total_play_cells > MAX_OCCUPANCY_RATIO:
        return False

    conflict_cells = [cell for cell, owners in cover_map.items() if len(owners) >= 2]
    if not conflict_cells:
        return False

    for cross in crosses:
        center = cross["center"]
        arm = cross["arm"]
        visible = [cell for cell in cross["cells"] if len(cover_map[cell]) == 1]
        if center not in visible:
            return False
        if not any(r == center[0] and c != center[1] for r, c in visible):
            return False
        if not any(c == center[1] and r != center[0] for r, c in visible):
            return False
        if not any(abs(r - center[0]) + abs(c - center[1]) == arm for r, c in visible):
            return False

        row_counts = Counter(r for r, _ in visible)
        col_counts = Counter(c for _, c in visible)
        if {r for r, count in row_counts.items() if count >= 2} != {center[0]}:
            return False
        if {c for c, count in col_counts.items() if count >= 2} != {center[1]}:
            return False

        recovered = _recover_cross_fast(inp, bg, cross["color"])
        if recovered is None:
            return False
        if recovered["center"] != center or recovered["arm"] != arm:
            return False

    if not np.array_equal(_solve_scene(inp), out):
        return False

    present_colors = {int(value) for value in np.unique(inp[:, 1:]) if int(value) != bg}
    if not set(active_colors).issubset(present_colors):
        return False
    return True


def _gen_attempt(rng):
    active_count = rng.randint(*ACTIVE_COUNT_RANGE)
    extra_legend = 1 if rng.random() < UNUSED_LEGEND_PROB else 0
    legend_count = active_count + extra_legend

    h_min = max(GRID_H_RANGE[0], legend_count + 3)
    h = rng.randint(h_min, GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])

    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    legend_colors = palette[:legend_count]
    active_colors = legend_colors[:]
    rng.shuffle(active_colors)
    active_colors = active_colors[:active_count]

    legend_rows = sorted(rng.sample(range(h), legend_count))
    crosses = []

    for color in active_colors:
        placed = None
        for _ in range(MAX_PLACEMENT_TRIES):
            arm = rng.randint(1, 3)
            base = rng.choice(crosses) if crosses and rng.random() < OVERLAP_PROB else None
            candidate = _make_candidate(arm, h, w, rng, base=base)
            candidate["color"] = color
            candidate["cells"] = _plus_cells(candidate["center"], candidate["arm"])
            if _can_add_cross(crosses, candidate, h, w):
                placed = candidate
                break
        if placed is None:
            return None
        crosses.append(placed)

    inp, out, cover_map = _render_scene(h, w, bg, legend_rows, legend_colors, active_colors, crosses)
    if not _scene_is_valid(inp, out, bg, legend_colors, active_colors, crosses, cover_map):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-overlap input")
