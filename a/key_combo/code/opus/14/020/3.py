import math
import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(0, 1), (0, 2), (1, 0), (1, 1), (1, -1), (1, 2), (1, -2)]
GEN_ATTEMPTS = 500


def _direction_signature(step: tuple[int, int]) -> tuple[int, int]:
    dr, dc = step
    g = math.gcd(abs(dr), abs(dc))
    return dr // g, dc // g


def _find_separators(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 2, "expected exactly two full-width gray separator rows"
    return rows[0], rows[1]


def _virtual_to_actual_row(vr: int, panel_h: int) -> int:
    if vr < panel_h:
        return vr
    if vr < 2 * panel_h:
        return vr + 1
    return vr + 2


def _actual_to_virtual_row(ar: int, panel_h: int) -> int:
    if ar < panel_h:
        return ar
    if ar < 2 * panel_h + 1:
        return ar - 1
    return ar - 2


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    s1, s2 = _find_separators(g)
    top_h = s1
    mid_h = s2 - s1 - 1
    bot_h = h - s2 - 1
    assert top_h == mid_h == bot_h >= 4, "separator rows must split three equal-height panels"
    panel_h = top_h
    assert h == 3 * panel_h + 2

    lines = []
    panel_colors = []
    all_marker_coords = set()
    panel_starts = [0, s1 + 1, s2 + 1]
    for panel_idx, row0 in enumerate(panel_starts):
        panel = g[row0 : row0 + panel_h, :]
        assert np.all(panel != SEP)
        colors = sorted(set(map(int, np.unique(panel))) - {BG})
        assert len(colors) == 1, "each panel must contain exactly one marker color"
        color = colors[0]
        coords = np.argwhere(panel == color)
        assert coords.shape == (3, 2), "each panel must contain exactly three marker pixels"
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        deltas = np.diff(coords, axis=0)
        assert np.array_equal(deltas[0], deltas[1]), "markers must be equally spaced"
        dr, dc = map(int, deltas[0])
        assert dr != 0 or dc != 0
        actual_coords = [(row0 + int(r), int(c)) for r, c in coords]
        virtual_coords = [(_actual_to_virtual_row(r, panel_h), c) for r, c in actual_coords]
        line = {
            "panel": panel_idx,
            "color": color,
            "step": (dr, dc),
            "markers_actual": actual_coords,
            "markers_virtual": virtual_coords,
        }
        lines.append(line)
        panel_colors.append(color)
        all_marker_coords.update(actual_coords)

    assert len(set(panel_colors)) == 3, "panel colors must be distinct"
    assert len({_direction_signature(line["step"]) for line in lines}) == 3, "line directions must differ"
    return {
        "grid": g,
        "panel_h": panel_h,
        "width": w,
        "lines": lines,
        "panel_colors": panel_colors,
        "marker_coords": all_marker_coords,
    }


def _simulate(parsed: dict) -> tuple[np.ndarray, dict]:
    g = parsed["grid"]
    panel_h = parsed["panel_h"]
    width = parsed["width"]
    lines = parsed["lines"]
    panel_colors = parsed["panel_colors"]
    original_markers = parsed["marker_coords"]

    out = g.copy()
    active = []
    for line_id, line in enumerate(lines):
        dr, dc = line["step"]
        active.append({"line_id": line_id, "current": line["markers_virtual"][0], "step": (-dr, -dc)})
        active.append({"line_id": line_id, "current": line["markers_virtual"][-1], "step": (dr, dc)})

    meta = {
        "line_stats": [
            {
                "origin_panel": line["panel"],
                "painted": 0,
                "cross_painted": 0,
                "panels_touched": set(),
                "stop_reasons": Counter(),
            }
            for line in lines
        ],
        "crossed_lines": 0,
        "collision_stop": 0,
        "marker_stop": 0,
        "new_cells": set(),
    }

    virtual_h = 3 * panel_h
    while active:
        proposals = []
        target_to_props: dict[tuple[int, int], list[int]] = {}
        for state in active:
            line_stat = meta["line_stats"][state["line_id"]]
            vr, vc = state["current"]
            dr, dc = state["step"]
            nr, nc = vr + dr, vc + dc
            if not (0 <= nr < virtual_h and 0 <= nc < width):
                line_stat["stop_reasons"]["boundary"] += 1
                continue
            ar = _virtual_to_actual_row(nr, panel_h)
            target = (ar, nc)
            if target in original_markers:
                line_stat["stop_reasons"]["marker"] += 1
                meta["marker_stop"] += 1
                continue
            if out[target] != BG:
                line_stat["stop_reasons"]["painted"] += 1
                meta["collision_stop"] += 1
                continue
            proposals.append((state, nr, nc, target))
            target_to_props.setdefault(target, []).append(len(proposals) - 1)

        blocked = set()
        for prop_ids in target_to_props.values():
            if len(prop_ids) > 1:
                blocked.update(prop_ids)

        next_active = []
        for prop_id, (state, nr, nc, target) in enumerate(proposals):
            line_stat = meta["line_stats"][state["line_id"]]
            if prop_id in blocked:
                line_stat["stop_reasons"]["painted"] += 1
                meta["collision_stop"] += 1
                continue
            target_panel = nr // panel_h
            out[target] = panel_colors[target_panel]
            line_stat["painted"] += 1
            line_stat["panels_touched"].add(target_panel)
            if target_panel != line_stat["origin_panel"]:
                line_stat["cross_painted"] += 1
            meta["new_cells"].add(target)
            next_active.append({"line_id": state["line_id"], "current": (nr, nc), "step": state["step"]})
        active = next_active

    for stat in meta["line_stats"]:
        if stat["cross_painted"] > 0:
            meta["crossed_lines"] += 1
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each panel's arithmetic marker line across separators, recoloring each new segment with the color of the panel it enters.

    Input structure:
    - The grid has black background and exactly two full-width gray separator rows that split it into three equal-height panels.
    - Each panel contains exactly three same-color marker pixels forming one equally spaced straight arithmetic line.
    - The three panels use different marker colors and different line directions.

    Rule:
    - In each panel, sort the three markers in row-major order and use the gap from the first marker to the second as the step vector.
    - Extend from the first marker backward and from the last marker forward by repeated copies of that step.
    - Keep separator rows and original markers unchanged; when an extension passes through a separator into a neighboring panel, skip the separator row and recolor the new pixels with the neighboring panel's marker color.
    - Stop a ray when the next step would leave the grid, hit any original marker pixel, or hit a pixel already painted by another extension.

    Key insight:
    - The three markers define a line in compressed panel coordinates: separators are skipped during continuation, so one logical line can cross panel boundaries and visibly change color based on its current panel rather than its source panel.
    """
    parsed = _parse_input(grid)
    out, _meta = _simulate(parsed)
    return out


def _valid_starts(panel_h: int, width: int, step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    row_max = panel_h - 1 - 2 * dr
    if row_max < 0:
        return []
    if dc >= 0:
        col_min = 0
        col_max = width - 1 - 2 * dc
    else:
        col_min = -2 * dc
        col_max = width - 1
    if col_max < col_min:
        return []
    return [(r0, c0) for r0 in range(row_max + 1) for c0 in range(col_min, col_max + 1)]


def _markers_from_start(panel_idx: int, panel_h: int, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = step
    panel_row0 = panel_idx * panel_h
    return [(panel_row0 + r0 + k * dr, c0 + k * dc) for k in range(3)]


def _build_input(panel_h: int, width: int, colors: list[int], placements: list[dict]) -> np.ndarray:
    grid = np.zeros((3 * panel_h + 2, width), dtype=int)
    grid[panel_h, :] = SEP
    grid[2 * panel_h + 1, :] = SEP
    panel_actual_starts = [0, panel_h + 1, 2 * panel_h + 2]
    for panel_idx, placement in enumerate(placements):
        row0 = panel_actual_starts[panel_idx]
        r_start, c_start = placement["start"]
        dr, dc = placement["step"]
        color = colors[panel_idx]
        for k in range(3):
            grid[row0 + r_start + k * dr, c_start + k * dc] = color
    return grid


def _features_ok(inp: np.ndarray, scenario: str) -> bool:
    try:
        parsed = _parse_input(inp)
        out, meta = _simulate(parsed)
    except AssertionError:
        return False

    if np.array_equal(inp, out):
        return False
    if len(meta["new_cells"]) < 6:
        return False
    if meta["crossed_lines"] < 1:
        return False
    if not any(stat["painted"] > 0 for stat in meta["line_stats"]):
        return False
    if any(stat["painted"] == 0 for stat in meta["line_stats"]):
        return False
    if not any(
        stat["origin_panel"] in stat["panels_touched"] and len(stat["panels_touched"]) >= 2
        for stat in meta["line_stats"]
    ):
        return False
    if len({int(out[r, c]) for r, c in meta["new_cells"]}) < 2:
        return False

    if scenario == "marker_block":
        if meta["marker_stop"] < 1:
            return False
    return True


def _sample_generic() -> np.ndarray | None:
    panel_h = random.randint(4, 8)
    width = random.randint(8, 13)
    colors = random.sample(PALETTE, 3)

    steps = [step for step in DIRS if _valid_starts(panel_h, width, step)]
    if not steps:
        return None
    families = []
    horizontal = [step for step in steps if step[0] == 0]
    slanted = [step for step in steps if step[0] == 1]
    if horizontal and len(slanted) >= 2:
        families.append([random.choice(horizontal)] + random.sample(slanted, 2))
    if len(slanted) >= 3:
        families.append(random.sample(slanted, 3))
    if len(steps) >= 3:
        families.append(random.sample(steps, 3))
    if not families:
        return None
    chosen = random.choice(families)
    if len({_direction_signature(step) for step in chosen}) < 3:
        return None

    placements = []
    for panel_idx, step in enumerate(chosen):
        starts = _valid_starts(panel_h, width, step)
        if not starts:
            return None
        start = random.choice(starts)
        placements.append({"panel": panel_idx, "step": step, "start": start})

    inp = _build_input(panel_h, width, colors, placements)
    return inp if _features_ok(inp, "generic") else None


def _sample_marker_block() -> np.ndarray | None:
    panel_h = random.randint(5, 8)
    width = random.randint(9, 13)
    colors = random.sample(PALETTE, 3)
    h_step = random.choice([(0, 1), (0, 2)])
    d_step_choices = [step for step in [(1, 1), (1, -1), (1, 2), (1, -2)] if _valid_starts(panel_h, width, step)]
    if not d_step_choices:
        return None
    d_step = random.choice(d_step_choices)

    center_min = h_step[1]
    center_max = width - 1 - h_step[1]
    if center_min > center_max:
        return None
    col = random.randint(center_min, center_max)
    hit_row = random.randint(1, panel_h - 2)
    top_start = (hit_row, col - h_step[1])
    mid_start = (1, col)
    if mid_start not in _valid_starts(panel_h, width, (1, 0)):
        return None

    bottom_starts = _valid_starts(panel_h, width, d_step)
    random.shuffle(bottom_starts)
    for bottom_start in bottom_starts[:20]:
        placements = [
            {"panel": 0, "step": h_step, "start": top_start},
            {"panel": 1, "step": (1, 0), "start": mid_start},
            {"panel": 2, "step": d_step, "start": bottom_start},
        ]
        inp = _build_input(panel_h, width, colors, placements)
        if _features_ok(inp, "marker_block"):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scenario = random.choices(["generic", "marker_block"], weights=[3, 2])[0]
        grid = _sample_marker_block() if scenario == "marker_block" else _sample_generic()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-panel arithmetic-line input")
