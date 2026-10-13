import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4
GREY = 5
MARKER_PALETTE = (2, 3, 6, 7, 8, 9)
STATIC_COLORS = {BG, BLUE, YELLOW, GREY}

PANEL_MIN = 7
PANEL_MAX = 8
OUTER_BORDER = 1
SEP_WIDTH = 1
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_full_line(panel, color, axis):
    if axis == "row":
        rows = [r for r in range(panel.shape[0]) if np.all(panel[r, :] == color)]
        return rows
    cols = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == color)]
    return cols


def _reflect_panel(panel):
    panel = np.asarray(panel, dtype=int)
    p = panel.shape[0]
    row_axes = _find_full_line(panel, YELLOW, "row")
    col_axes = _find_full_line(panel, YELLOW, "col")
    assert panel.shape[0] == panel.shape[1] and p >= 6
    assert (len(row_axes) == 1) ^ (len(col_axes) == 1), "expected exactly one guide line"

    out = np.full_like(panel, BG)
    if row_axes:
        axis = row_axes[0]
        out[axis, :] = YELLOW
        for r in range(p):
            if r == axis:
                continue
            src_r = 2 * axis - r
            if 0 <= src_r < p:
                out[r, :] = panel[src_r, :]
        out[axis, :] = YELLOW
        return out

    axis = col_axes[0]
    out[:, axis] = YELLOW
    for c in range(p):
        if c == axis:
            continue
        src_c = 2 * axis - c
        if 0 <= src_c < p:
            out[:, c] = panel[:, src_c]
    out[:, axis] = YELLOW
    return out


def _separator_index(grid, axis):
    if axis == "row":
        idxs = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    else:
        idxs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(idxs) == 1, "expected exactly one full grey separator line"
    return idxs[0]


def _panel_bounds(grid):
    h, w = grid.shape
    sep_r = _separator_index(grid, "row")
    sep_c = _separator_index(grid, "col")
    top = (OUTER_BORDER, sep_r)
    bottom = (sep_r + SEP_WIDTH, h - OUTER_BORDER)
    left = (OUTER_BORDER, sep_c)
    right = (sep_c + SEP_WIDTH, w - OUTER_BORDER)
    row_spans = [top, bottom]
    col_spans = [left, right]
    panels = []
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            panels.append(
                {
                    "index": pr * 2 + pc,
                    "row_group": pr,
                    "col_group": pc,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                }
            )
    return panels


def _adjacent_markers(grid, panel_info, comp):
    h, w = grid.shape
    r0, r1 = panel_info["r0"], panel_info["r1"]
    c0, c1 = panel_info["c0"], panel_info["c1"]
    hits = {}
    for lr, lc in comp:
        gr, gc = r0 + lr, c0 + lc
        for nr, nc in _neighbors4(gr, gc):
            if r0 <= nr < r1 and c0 <= nc < c1:
                continue
            if 0 <= nr < h and 0 <= nc < w:
                color = int(grid[nr, nc])
                if color not in STATIC_COLORS:
                    hits[(nr, nc)] = color
    return sorted((r, c, color) for (r, c), color in hits.items())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each guided panel, then recolor any mirrored blue component that newly touches an exterior marker.

    Input structure:
    - The grid has four square panels in a 2x2 layout, separated by one full grey row and one full grey column.
    - Each panel contains one 4-connected blue shape and one full yellow guide line, while colored marker pixels sit on the outer canvas border outside the panels.
    - The blue shape starts away from the panel boundary, so any later border contact must come from the mirror step.

    Rule:
    - In each panel, a full yellow row means reflect the panel across that horizontal guide, while a full yellow column means reflect across that vertical guide; the guide line itself stays yellow.
    - After reflection, inspect each blue 4-connected component and look for marker pixels that are 4-neighbor-adjacent just outside the panel boundary.
    - Recolor the whole blue component to the chosen marker color; if several markers touch the same component, use the touching marker with the smallest row index. Background, markers, yellow guides, and grey separator lines stay unchanged.

    Key insight:
    - The mirror changes which blue pixels land on the panel edge, and only those new edge contacts can activate the exterior markers and determine the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    panels = _panel_bounds(g)
    for info in panels:
        panel = g[info["r0"] : info["r1"], info["c0"] : info["c1"]]
        out[info["r0"] : info["r1"], info["c0"] : info["c1"]] = _reflect_panel(panel)

    for info in panels:
        panel = out[info["r0"] : info["r1"], info["c0"] : info["c1"]]
        for comp in _components(panel == BLUE):
            hits = _adjacent_markers(out, info, comp)
            if not hits:
                continue
            chosen_r, chosen_c, chosen_color = min(hits, key=lambda item: (item[0], item[1]))
            _ = (chosen_r, chosen_c)
            for r, c in comp:
                out[info["r0"] + r, info["c0"] + c] = chosen_color
    return out


def _global_panel_bounds(p):
    sep = OUTER_BORDER + p
    top = (OUTER_BORDER, OUTER_BORDER + p)
    bottom = (sep + SEP_WIDTH, sep + SEP_WIDTH + p)
    left = (OUTER_BORDER, OUTER_BORDER + p)
    right = (sep + SEP_WIDTH, sep + SEP_WIDTH + p)
    return [
        {"index": 0, "row_group": 0, "col_group": 0, "r0": top[0], "r1": top[1], "c0": left[0], "c1": left[1]},
        {"index": 1, "row_group": 0, "col_group": 1, "r0": top[0], "r1": top[1], "c0": right[0], "c1": right[1]},
        {"index": 2, "row_group": 1, "col_group": 0, "r0": bottom[0], "r1": bottom[1], "c0": left[0], "c1": left[1]},
        {"index": 3, "row_group": 1, "col_group": 1, "r0": bottom[0], "r1": bottom[1], "c0": right[0], "c1": right[1]},
    ]


def _accessible_sides(info):
    sides = []
    sides.append("top" if info["row_group"] == 0 else "bottom")
    sides.append("left" if info["col_group"] == 0 else "right")
    return tuple(sides)


def _orientation_for_side(side):
    return "h" if side in ("top", "bottom") else "v"


def _touch_axis(side, p):
    return 2 if side in ("top", "left") else p - 3


def _quiet_axis(p):
    return p // 2


def _target_box(side, p, touched):
    if touched:
        if side == "top":
            return (0, 2), (1, p - 1)
        if side == "bottom":
            return (p - 2, p), (1, p - 1)
        if side == "left":
            return (1, p - 1), (0, 2)
        return (1, p - 1), (p - 2, p)
    axis = _quiet_axis(p)
    if side == "top":
        return (axis - 2, axis), (1, p - 1)
    if side == "bottom":
        return (axis + 1, axis + 3), (1, p - 1)
    if side == "left":
        return (1, p - 1), (axis - 2, axis)
    return (1, p - 1), (axis + 1, axis + 3)


def _shape_bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _irregular_enough(cells):
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return False
    r0, c0, r1, c1 = _shape_bbox(cells)
    return len(cells) < (r1 - r0 + 1) * (c1 - c0 + 1)


def _random_connected_shape(row_box, col_box, size, required_cell=None):
    cells = [(r, c) for r in range(*row_box) for c in range(*col_box)]
    allowed = set(cells)
    start = required_cell if required_cell is not None else random.choice(cells)
    shape = {start}
    while len(shape) < size:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))
    return shape


def _boundary_cells(shape, side, p):
    if side == "top":
        return sorted((r, c) for r, c in shape if r == 0)
    if side == "bottom":
        return sorted((r, c) for r, c in shape if r == p - 1)
    if side == "left":
        return sorted((r, c) for r, c in shape if c == 0)
    return sorted((r, c) for r, c in shape if c == p - 1)


def _make_output_shape(side, p, touched, multi_marker):
    row_box, col_box = _target_box(side, p, touched)
    boundary_candidates = [(r, c) for r in range(*row_box) for c in range(*col_box) if (side == "top" and r == 0)
                           or (side == "bottom" and r == p - 1)
                           or (side == "left" and c == 0)
                           or (side == "right" and c == p - 1)]
    for _ in range(120):
        size = random.randint(4, min(8, (row_box[1] - row_box[0]) * (col_box[1] - col_box[0])))
        required = random.choice(boundary_candidates) if touched else None
        shape = _random_connected_shape(row_box, col_box, size, required)
        if shape is None or not _irregular_enough(shape):
            continue
        if touched:
            b_cells = _boundary_cells(shape, side, p)
            if not b_cells:
                continue
            if multi_marker:
                rows = {r for r, _ in b_cells}
                if len(rows) < 2:
                    continue
        return shape
    raise RuntimeError("failed to generate a valid panel shape")


def _invert_shape(shape, orient, axis):
    inverted = set()
    for r, c in shape:
        if orient == "h":
            inverted.add((2 * axis - r, c))
        else:
            inverted.add((r, 2 * axis - c))
    return inverted


def _marker_position(info, local_cell, side):
    r, c = local_cell
    if side == "top":
        return info["r0"] - 1, info["c0"] + c
    if side == "bottom":
        return info["r1"], info["c0"] + c
    if side == "left":
        return info["r0"] + r, info["c0"] - 1
    return info["r0"] + r, info["c1"]


def _pick_distinct_colors(n):
    colors = list(MARKER_PALETTE)
    random.shuffle(colors)
    out = []
    for i in range(n):
        out.append(colors[i % len(colors)])
    return out


def _choose_touched_side(info, orient=None):
    sides = _accessible_sides(info)
    if orient is None:
        return random.choice(sides)
    want = "h" if orient == "h" else "v"
    candidates = [side for side in sides if _orientation_for_side(side) == want]
    return random.choice(candidates)


def _build_panel_plan(info, p, touched, force_orient=None, force_multi=False):
    if touched:
        side = _choose_touched_side(info, force_orient)
        orient = _orientation_for_side(side)
        axis = _touch_axis(side, p)
        multi_marker = force_multi and orient == "v"
        out_shape = _make_output_shape(side, p, True, multi_marker)
        markers_local = _boundary_cells(out_shape, side, p)
        if multi_marker:
            rows = sorted({r for r, _ in markers_local})
            chosen_rows = random.sample(rows, 2)
            chosen_rows.sort()
            marker_cells = []
            for row in chosen_rows:
                row_cells = [(r, c) for r, c in markers_local if r == row]
                marker_cells.append(random.choice(row_cells))
        else:
            marker_cells = [random.choice(markers_local)]
        marker_colors = _pick_distinct_colors(len(marker_cells))
        chosen_color = marker_colors[min(range(len(marker_cells)), key=lambda i: (_marker_position(info, marker_cells[i], side)[0], _marker_position(info, marker_cells[i], side)[1]))]
        extra_marker = None
        if len(marker_cells) == 1 and random.random() < 0.35:
            other_side = [s for s in _accessible_sides(info) if s != side][0]
            extra_marker = (other_side, None, random.choice(MARKER_PALETTE))
    else:
        orient = force_orient if force_orient is not None else random.choice(("h", "v"))
        side = _choose_touched_side(info, orient)
        axis = _quiet_axis(p)
        out_shape = _make_output_shape(side, p, False, False)
        marker_cells = []
        marker_colors = []
        chosen_color = BLUE
        extra_marker = None
        if random.random() < 0.45:
            extra_marker = (side, None, random.choice(MARKER_PALETTE))

    in_shape = _invert_shape(out_shape, orient, axis)
    rows = {r for r, _ in in_shape}
    cols = {c for _, c in in_shape}
    assert all(0 <= r < p and 0 <= c < p for r, c in in_shape)
    assert all(0 < r < p - 1 and 0 < c < p - 1 for r, c in in_shape), "input shape must stay off the boundary"
    assert axis not in rows if orient == "h" else axis not in cols

    panel = np.full((p, p), BG, dtype=int)
    output = np.full((p, p), BG, dtype=int)
    if orient == "h":
        panel[axis, :] = YELLOW
        output[axis, :] = YELLOW
    else:
        panel[:, axis] = YELLOW
        output[:, axis] = YELLOW
    for r, c in in_shape:
        panel[r, c] = BLUE
    for r, c in out_shape:
        output[r, c] = chosen_color

    marker_specs = [(side, cell, color) for cell, color in zip(marker_cells, marker_colors)]
    if extra_marker is not None:
        side_name, _cell, color = extra_marker
        marker_specs.append((side_name, None, color))

    return {
        "panel": panel,
        "expected_panel": output,
        "orient": orient,
        "axis": axis,
        "side": side,
        "touched": touched,
        "out_shape": out_shape,
        "marker_specs": marker_specs,
        "chosen_color": chosen_color,
    }


def _place_markers(grid, expected, info, plan):
    p = plan["panel"].shape[0]
    occupied = set()
    by_side_boundary = {
        "top": [(0, c) for c in range(1, p - 1)],
        "bottom": [(p - 1, c) for c in range(1, p - 1)],
        "left": [(r, 0) for r in range(1, p - 1)],
        "right": [(r, p - 1) for r in range(1, p - 1)],
    }
    touch_boundary = {side: _boundary_cells(plan["out_shape"], side, p) for side in ("top", "bottom", "left", "right")}

    for side, cell, color in plan["marker_specs"]:
        if cell is None:
            candidates = list(by_side_boundary[side])
            random.shuffle(candidates)
            chosen = None
            for candidate in candidates:
                if candidate in occupied:
                    continue
                if candidate in touch_boundary[side]:
                    continue
                chosen = candidate
                break
            assert chosen is not None
            cell = chosen
        pos = _marker_position(info, cell, side)
        grid[pos] = color
        expected[pos] = color
        occupied.add(cell)


def _render_grid(p, plans):
    h = 2 * p + 3
    grid = np.full((h, h), BG, dtype=int)
    expected = np.full((h, h), BG, dtype=int)
    sep = OUTER_BORDER + p
    grid[sep, :] = GREY
    grid[:, sep] = GREY
    expected[sep, :] = GREY
    expected[:, sep] = GREY

    infos = _global_panel_bounds(p)
    for info, plan in zip(infos, plans):
        grid[info["r0"] : info["r1"], info["c0"] : info["c1"]] = plan["panel"]
        expected[info["r0"] : info["r1"], info["c0"] : info["c1"]] = plan["expected_panel"]
    for info, plan in zip(infos, plans):
        _place_markers(grid, expected, info, plan)
    return grid, expected


def _semantic_check(grid, expected):
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    panels = _panel_bounds(grid)
    touched_panels = 0
    unchanged_panels = 0
    h_guides = 0
    v_guides = 0
    recolored_colors = set()

    for info in panels:
        input_panel = grid[info["r0"] : info["r1"], info["c0"] : info["c1"]]
        output_panel = actual[info["r0"] : info["r1"], info["c0"] : info["c1"]]
        row_axes = _find_full_line(input_panel, YELLOW, "row")
        col_axes = _find_full_line(input_panel, YELLOW, "col")
        if row_axes:
            h_guides += 1
        if col_axes:
            v_guides += 1
        blue_comps = _components(input_panel == BLUE)
        if len(blue_comps) != 1 or not 4 <= len(blue_comps[0]) <= 8:
            return False
        if any(r in (0, input_panel.shape[0] - 1) or c in (0, input_panel.shape[1] - 1) for r, c in blue_comps[0]):
            return False
        out_colors = set(int(v) for v in np.unique(output_panel) if int(v) not in {BG, YELLOW})
        if BLUE in out_colors:
            unchanged_panels += 1
        if any(color not in {BLUE} for color in out_colors):
            touched_panels += 1
            recolored_colors |= {color for color in out_colors if color != BLUE}

    if touched_panels < 1 or unchanged_panels < 1:
        return False
    if h_guides < 1 or v_guides < 1:
        return False
    return bool(recolored_colors)


def _generation_attempt():
    p = random.randint(PANEL_MIN, PANEL_MAX)
    infos = _global_panel_bounds(p)

    horizontal_panel = random.randrange(4)
    vertical_panel = random.choice([idx for idx in range(4) if idx != horizontal_panel])
    remaining = [idx for idx in range(4) if idx not in {horizontal_panel, vertical_panel}]
    untouched_panel = random.choice(remaining)

    touched_flags = [False] * 4
    touched_flags[horizontal_panel] = True
    touched_flags[vertical_panel] = True
    touched_flags[untouched_panel] = False
    extra_panel = [idx for idx in remaining if idx != untouched_panel][0]
    touched_flags[extra_panel] = random.random() < 0.5

    plans = [None] * 4
    plans[horizontal_panel] = _build_panel_plan(
        infos[horizontal_panel],
        p,
        touched=True,
        force_orient="h",
        force_multi=False,
    )
    plans[vertical_panel] = _build_panel_plan(
        infos[vertical_panel],
        p,
        touched=True,
        force_orient="v",
        force_multi=(random.random() < 0.6),
    )
    plans[untouched_panel] = _build_panel_plan(
        infos[untouched_panel],
        p,
        touched=False,
        force_orient=random.choice(("h", "v")),
    )
    plans[extra_panel] = _build_panel_plan(
        infos[extra_panel],
        p,
        touched=touched_flags[extra_panel],
        force_orient=None,
        force_multi=(touched_flags[extra_panel] and random.random() < 0.25),
    )

    grid, expected = _render_grid(p, plans)
    if not _semantic_check(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
