from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
REGION_COLORS = [1, 2, 5, 8]
SEED_COLORS = {3, 4}
MARKER_COLOR = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[0], -d[1])


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _stem_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    for d in branches:
        if _opp(d) not in branches:
            return d
    raise AssertionError("invalid T marker")


def _parse_t_marker(cells: list[tuple[int, int]]) -> dict | None:
    if len(cells) != 4:
        return None
    cell_set = set(cells)
    for r, c in cells:
        branches = [(dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cell_set]
        if len(branches) != 3:
            continue
        stem = _stem_from_branches(branches)
        expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        expected.remove((r - stem[0], c - stem[1]))
        if cell_set == expected:
            bar = [d for d in branches if d != stem]
            return {
                "center": (r, c),
                "stem": stem,
                "bar": bar,
                "cells": sorted(cell_set),
            }
    return None


def _find_t_markers(grid: np.ndarray) -> list[dict]:
    markers = []
    for comp in _label_components(grid == MARKER_COLOR):
        parsed = _parse_t_marker(comp)
        if parsed is None:
            raise AssertionError("all maroon cells must belong to 4-cell T markers")
        markers.append(parsed)
    if len(markers) != 2:
        raise AssertionError("expected exactly two T markers")
    return markers


def _majority_color(vals: list[int]) -> int:
    counts = Counter(vals)
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _compute_runs(grid: np.ndarray, layout: str) -> list[dict]:
    forbidden = SEED_COLORS | {MARKER_COLOR}
    runs: list[dict] = []
    last_color = None
    last_start = 0
    length = grid.shape[1] if layout == "vertical" else grid.shape[0]
    for idx in range(length):
        line = grid[:, idx] if layout == "vertical" else grid[idx, :]
        vals = [int(v) for v in line if int(v) not in forbidden]
        if not vals:
            raise AssertionError("slice lost all base colors")
        color = _majority_color(vals)
        if last_color is None:
            last_color = color
            last_start = idx
        elif color != last_color:
            runs.append({"start": last_start, "end": idx - 1, "color": last_color})
            last_color = color
            last_start = idx
    runs.append({"start": last_start, "end": length - 1, "color": last_color})
    if len(runs) < 5:
        raise AssertionError("expected at least five large strips")
    return runs


def _run_index_for(runs: list[dict], pos: int) -> int:
    for i, run in enumerate(runs):
        if run["start"] <= pos <= run["end"]:
            return i
    raise AssertionError("position outside runs")


def _apply_recolor(grid: np.ndarray, layout: str, runs: list[dict], markers: list[dict]) -> tuple[np.ndarray, list[int]]:
    out = np.array(grid, copy=True)
    target_runs: list[int] = []
    for marker in markers:
        center = marker["center"]
        stem = marker["stem"]
        source_colors = []
        for br, bc in marker["bar"]:
            rr = center[0] + stem[0] + br
            cc = center[1] + stem[1] + bc
            if not _in_bounds(grid.shape, rr, cc):
                raise AssertionError("source diagonal out of bounds")
            source_colors.append(int(grid[rr, cc]))
        if len(set(source_colors)) != 1 or source_colors[0] in SEED_COLORS | {MARKER_COLOR}:
            raise AssertionError("source diagonals must agree on a base color")
        source_color = source_colors[0]
        rr = center[0] + 2 * stem[0]
        cc = center[1] + 2 * stem[1]
        axis_pos = cc if layout == "vertical" else rr
        run_idx = _run_index_for(runs, axis_pos)
        target_runs.append(run_idx)
        run = runs[run_idx]
        if layout == "vertical":
            out[:, run["start"] : run["end"] + 1] = source_color
        else:
            out[run["start"] : run["end"] + 1, :] = source_color
        for mr, mc in marker["cells"]:
            out[mr, mc] = source_color
    if len(set(target_runs)) != 2:
        raise AssertionError("markers must point to two distinct target strips")
    return out, sorted(target_runs)


def _extract_seed_pattern(grid: np.ndarray, layout: str, corridor: tuple[int, int]) -> dict:
    groups: dict[int, dict[int, int]] = defaultdict(dict)
    for r, c in np.argwhere(np.isin(grid, [3, 4])):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        inside = corridor[0] <= (c if layout == "vertical" else r) <= corridor[1]
        if not inside:
            raise AssertionError("all seed pixels must lie inside the corridor")
        key = c if layout == "vertical" else r
        coord = r if layout == "vertical" else c
        if color in groups[key]:
            raise AssertionError("each seed line can contain only one green and one yellow")
        groups[key][color] = coord
    if len(groups) != 2:
        raise AssertionError("expected two seed pairs")

    period = None
    green_phase = None
    yellow_phase = None
    for pair in groups.values():
        if set(pair) != {3, 4}:
            raise AssertionError("each pair must contain one green and one yellow seed")
        d = abs(pair[4] - pair[3])
        if d < 2:
            raise AssertionError("seed distance must be at least two")
        p = 2 * d
        gmod = pair[3] % p
        ymod = pair[4] % p
        if (ymod - gmod) % p != d:
            raise AssertionError("seed rows or columns must occupy adjacent stripe starts")
        if period is None:
            period = p
            green_phase = gmod
            yellow_phase = ymod
        elif period != p or green_phase != gmod or yellow_phase != ymod:
            raise AssertionError("seed pairs must imply the same stripe pattern")
    assert period is not None
    return {"period": period, "step": period // 2, "green_phase": green_phase, "yellow_phase": yellow_phase}


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    markers = _find_t_markers(g)
    if all(marker["stem"][0] == 0 for marker in markers):
        layout = "vertical"
    elif all(marker["stem"][1] == 0 for marker in markers):
        layout = "horizontal"
    else:
        raise AssertionError("marker stems must agree on one global axis")

    runs = _compute_runs(g, layout)
    recolored, target_runs = _apply_recolor(g, layout, runs, markers)
    left_run, right_run = target_runs
    corridor = (runs[left_run]["end"] + 1, runs[right_run]["start"] - 1)
    if corridor[1] - corridor[0] + 1 < 3:
        raise AssertionError("corridor too narrow")
    pattern = _extract_seed_pattern(g, layout, corridor)
    return {
        "grid": g,
        "layout": layout,
        "runs": runs,
        "recolored": recolored,
        "corridor": corridor,
        "pattern": pattern,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint T-pointed strips into wall bands, then fill the corridor between them with seed-driven stripes.

    Input structure:
    - The grid is tiled by five large monochrome rectangular strips using region colors, plus two maroon T control markers placed on source-strip boundaries.
    - Each T has a 3-cell bar and a 1-cell stem; the stem points toward a neighboring target strip, while a green/yellow seed pair sits inside the corridor strip between the two targets.
    - There are exactly two seed pairs, each made of one green pixel and one yellow pixel on a common row or column inside that corridor.

    Rule:
    - For each T marker, read the source color from the two diagonal cells adjacent to the stem tip, and repaint the entire strip reached two steps forward along the stem into that source color.
    - Repaint the maroon T cells themselves into the same source color so they merge into the wall strip.
    - After both recolorings, treat the space between the two repainted target strips as the corridor.
    - If the walls are left/right of the corridor, use the seed row distance to create alternating horizontal stripes across the full corridor; if the walls are above/below it, use the seed column distance to create alternating vertical stripes. The green seed line starts a green stripe block and the adjacent yellow seed line starts the next yellow block.

    Key insight:
    - The seeds alone do not reveal the stripe location: the T-driven recolor must first weld the source strips into two parallel walls, and only that new wall geometry identifies which strip becomes the stripe-filled corridor.
    """
    info = _analyze(grid)
    out = np.array(info["recolored"], copy=True)
    start, end = info["corridor"]
    step = int(info["pattern"]["step"])
    period = int(info["pattern"]["period"])
    green_phase = int(info["pattern"]["green_phase"])

    if info["layout"] == "vertical":
        for r in range(out.shape[0]):
            color = 3 if ((r - green_phase) % period) < step else 4
            out[r, start : end + 1] = color
    else:
        for c in range(out.shape[1]):
            color = 3 if ((c - green_phase) % period) < step else 4
            out[start : end + 1, c] = color
    return out


def _choose_region_colors() -> list[int]:
    a, b, c, d = random.sample(REGION_COLORS, 4)
    return [a, b, c, b, d]


def _make_vertical_case() -> np.ndarray:
    d = random.randint(2, 4)
    left_source_w = random.randint(2, 4)
    left_target_w = random.randint(2, 4)
    corridor_w = random.randint(5, 8)
    right_target_w = random.randint(2, 4)
    right_source_w = random.randint(2, 4)

    height = random.randint(max(12, 4 * d + 3), 22)
    width = left_source_w + left_target_w + corridor_w + right_target_w + right_source_w
    colors = _choose_region_colors()
    cuts = np.cumsum([left_source_w, left_target_w, corridor_w, right_target_w, right_source_w])

    grid = np.zeros((height, width), dtype=int)
    c0 = 0
    for color, c1 in zip(colors, cuts):
        grid[:, c0:c1] = color
        c0 = c1

    corridor_start = left_source_w + left_target_w
    corridor_end = corridor_start + corridor_w - 1

    seed_cols = [corridor_start + 1, corridor_end - 1]

    phase = random.randint(2, height - 3 - 3 * d)
    green_first = random.choice([True, False])
    sequence = [3, 4, 3, 4] if green_first else [4, 3, 4, 3]
    rows = [phase + k * d for k in range(4)]

    left_center = (rows[1], left_source_w - 2)
    right_center = (rows[2], width - right_source_w + 1)
    for rr, cc in [left_center, (left_center[0], left_center[1] + 1), (left_center[0] - 1, left_center[1]), (left_center[0] + 1, left_center[1])]:
        grid[rr, cc] = MARKER_COLOR
    for rr, cc in [right_center, (right_center[0], right_center[1] - 1), (right_center[0] - 1, right_center[1]), (right_center[0] + 1, right_center[1])]:
        grid[rr, cc] = MARKER_COLOR

    grid[rows[0], seed_cols[0]] = sequence[0]
    grid[rows[1], seed_cols[0]] = sequence[1]
    grid[rows[2], seed_cols[1]] = sequence[2]
    grid[rows[3], seed_cols[1]] = sequence[3]
    return grid


def _make_horizontal_case() -> np.ndarray:
    grid = _make_vertical_case().T
    return np.array(grid, copy=True)


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze(grid)
    out = main(grid)
    if grid.shape != out.shape:
        raise AssertionError("shape must stay unchanged")
    if np.array_equal(grid, out):
        raise AssertionError("output must differ from input")
    if len(np.unique(out)) < 3:
        raise AssertionError("output is too degenerate")
    start, end = info["corridor"]
    if info["layout"] == "vertical":
        corridor = out[:, start : end + 1]
    else:
        corridor = out[start : end + 1, :]
    if not np.isin(corridor, [3, 4]).all():
        raise AssertionError("corridor must be fully striped")
    if info["layout"] == "vertical":
        left_ok = np.array_equal(out[:, :start], info["recolored"][:, :start])
        right_ok = np.array_equal(out[:, end + 1 :], info["recolored"][:, end + 1 :])
        if not (left_ok and right_ok):
            raise AssertionError("non-corridor columns changed")
    else:
        top_ok = np.array_equal(out[:start, :], info["recolored"][:start, :])
        bottom_ok = np.array_equal(out[end + 1 :, :], info["recolored"][end + 1 :, :])
        if not (top_ok and bottom_ok):
            raise AssertionError("non-corridor rows changed")


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _make_vertical_case() if random.choice([True, False]) else _make_horizontal_case()
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
