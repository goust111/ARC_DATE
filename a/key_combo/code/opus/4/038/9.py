from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
TEAL = 8
CONTROL_COLORS = (BLUE, RED, GREEN, ORANGE)
SIDES = ("top", "bottom", "left", "right")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    out = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return out


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return int(comp["area"]) == patch.size and np.all(patch == color)


def _template_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _run_groups(side_len: int, blocked: set[int]) -> list[tuple[int, int]]:
    groups = []
    idx = 0
    while idx < side_len:
        if idx in blocked:
            idx += 1
            continue
        start = idx
        while idx + 1 < side_len and idx + 1 not in blocked:
            idx += 1
        groups.append((start, idx))
        idx += 1
    return groups


def _group_before(start: int, blocked: set[int]) -> list[int]:
    if start == 0 or start - 1 not in blocked:
        return []
    out = []
    idx = start - 1
    while idx >= 0 and idx in blocked:
        out.append(idx)
        idx -= 1
    out.reverse()
    return out


def _group_after(end: int, side_len: int, blocked: set[int]) -> list[int]:
    if end + 1 >= side_len or end + 1 not in blocked:
        return []
    out = []
    idx = end + 1
    while idx < side_len and idx in blocked:
        out.append(idx)
        idx += 1
    return out


def _control_distance(side: str, control: dict, rect_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect_bbox
    if side == "top":
        return r0 - int(control["r"])
    if side == "bottom":
        return int(control["r"]) - r1
    if side == "left":
        return c0 - int(control["c"])
    return int(control["c"]) - c1


def _pick_group_control(
    side: str,
    rect_bbox: tuple[int, int, int, int],
    group: list[int],
    controls_by_index: dict[int, list[dict]],
) -> dict | None:
    if not group:
        return None
    candidates = [control for idx in group for control in controls_by_index[idx]]
    return min(
        candidates,
        key=lambda control: (_control_distance(side, control, rect_bbox), abs(int(control["index"]) - group[0])),
    )


def _choose_color_and_gap(
    side: str,
    run: tuple[int, int],
    side_len: int,
    controls_by_index: dict[int, list[dict]],
    rect_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    blocked = set(controls_by_index)
    left_group = _group_before(run[0], blocked)
    right_group = _group_after(run[1], side_len, blocked)
    left_control = _pick_group_control(side, rect_bbox, left_group, controls_by_index)
    right_control = _pick_group_control(side, rect_bbox, right_group, controls_by_index)

    if left_control is None and right_control is None:
        return TEAL, 0
    if left_control is None:
        return int(right_control["color"]), len(right_group)
    if right_control is None:
        return int(left_control["color"]), len(left_group)

    left_dist = _control_distance(side, left_control, rect_bbox)
    right_dist = _control_distance(side, right_control, rect_bbox)
    if left_dist < right_dist:
        return int(left_control["color"]), len(left_group)
    if right_dist < left_dist:
        return int(right_control["color"]), len(right_group)
    if int(left_control["index"]) <= int(right_control["index"]):
        return int(left_control["color"]), len(left_group)
    return int(right_control["color"]), len(right_group)


def _stamp_if_visible(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> int:
    h, w = mask.shape
    added = 0
    for dr in range(h):
        for dc in range(w):
            if not mask[dr, dc]:
                continue
            rr, cc = top + dr, left + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(out[rr, cc]) == BG:
                out[rr, cc] = int(color)
                added += 1
    return added


def _apply_rule(
    base: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    template: np.ndarray,
    side_controls: dict[str, list[dict]],
) -> tuple[np.ndarray, list[dict]]:
    out = base.copy()
    r0, c0, r1, c1 = rect_bbox
    th, tw = template.shape
    placements: list[dict] = []

    for side in SIDES:
        side_len = (c1 - c0 + 1) if side in ("top", "bottom") else (r1 - r0 + 1)
        controls_by_index: dict[int, list[dict]] = {}
        for control in side_controls[side]:
            controls_by_index.setdefault(int(control["index"]), []).append(control)
        runs = _run_groups(side_len, set(controls_by_index))

        for run in runs:
            run_len = run[1] - run[0] + 1
            if side in ("top", "bottom") and run_len < tw:
                continue
            if side in ("left", "right") and run_len < th:
                continue

            color, gap = _choose_color_and_gap(side, run, side_len, controls_by_index, rect_bbox)
            step = (th if side in ("top", "bottom") else tw) + gap
            if side == "top":
                row, col = r0 - th, c0 + run[0]
                while row >= 0:
                    added = _stamp_if_visible(out, template, color, row, col)
                    placements.append(
                        {"side": side, "run": run, "gap": gap, "color": color, "anchor": (row, col), "added": added}
                    )
                    row -= step
            elif side == "bottom":
                row, col = r1 + 1, c0 + run[0]
                while row + th <= out.shape[0]:
                    added = _stamp_if_visible(out, template, color, row, col)
                    placements.append(
                        {"side": side, "run": run, "gap": gap, "color": color, "anchor": (row, col), "added": added}
                    )
                    row += step
            elif side == "left":
                row, col = r0 + run[0], c0 - tw
                while col >= 0:
                    added = _stamp_if_visible(out, template, color, row, col)
                    placements.append(
                        {"side": side, "run": run, "gap": gap, "color": color, "anchor": (row, col), "added": added}
                    )
                    col -= step
            else:
                row, col = r0 + run[0], c1 + 1
                while col + tw <= out.shape[1]:
                    added = _stamp_if_visible(out, template, color, row, col)
                    placements.append(
                        {"side": side, "run": run, "gap": gap, "color": color, "anchor": (row, col), "added": added}
                    )
                    col += step

    return out, placements


def _parse_grid(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, dict[str, list[dict]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    components = _find_components(g, bg=bg, connectivity=8)
    assert components, "expected foreground objects"

    rectangles = [comp for comp in components if int(comp["color"]) == TEAL and _is_solid_rectangle(g, comp)]
    assert len(rectangles) == 1, "expected one solid teal rectangle"
    rectangle = rectangles[0]

    templates = [comp for comp in components if int(comp["color"]) == GREY and int(comp["area"]) > 1]
    assert len(templates) == 1, "expected one grey template sprite"
    template_comp = templates[0]
    template = _template_mask(template_comp)

    controls = [comp for comp in components if comp is not rectangle and comp is not template_comp]
    assert controls, "expected control pixels"
    assert all(int(comp["area"]) == 1 for comp in controls), "controls must be singleton pixels"

    rect_bbox = tuple(map(int, rectangle["bbox"]))
    r0, c0, r1, c1 = rect_bbox
    side_controls = {side: [] for side in SIDES}
    for comp in controls:
        color = int(comp["color"])
        assert color in CONTROL_COLORS, "unexpected control color"
        r, c = map(int, comp["coords"][0])
        hits = 0
        if r < r0 and c0 <= c <= c1:
            side_controls["top"].append({"r": r, "c": c, "color": color, "index": c - c0})
            hits += 1
        if r > r1 and c0 <= c <= c1:
            side_controls["bottom"].append({"r": r, "c": c, "color": color, "index": c - c0})
            hits += 1
        if c < c0 and r0 <= r <= r1:
            side_controls["left"].append({"r": r, "c": c, "color": color, "index": r - r0})
            hits += 1
        if c > c1 and r0 <= r <= r1:
            side_controls["right"].append({"r": r, "c": c, "color": color, "index": r - r0})
            hits += 1
        assert hits == 1, "each control must align with exactly one rectangle side"

    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    th, tw = template.shape
    assert th <= rect_h and tw <= rect_w, "template bbox must fit the rectangle projection"
    for side in SIDES:
        side_len = rect_w if side in ("top", "bottom") else rect_h
        assert len({int(control["index"]) for control in side_controls[side]}) == len(side_controls[side])
        assert len(side_controls[side]) < side_len, "every side needs at least one unblocked row/column"

    return rect_bbox, template, side_controls


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the corner template away from a teal rectangle along the unblocked side stripes and recolor each stripe from its nearest blocking control.

    Input structure:
    - The grid has a black background, one solid teal rectangle, several singleton control pixels in blue/red/green/orange, and one grey corner template sprite.
    - Every control lies in a side strip of the rectangle and shares a row or a column with that rectangle.
    - The template sprite's bounding box fits within the rectangle's height/width projections.

    Rule:
    - For each rectangle side, the aligned rows or columns carrying control pixels are excluded exactly as in the side-stripe extension task.
    - The remaining aligned indices form maximal continuous stripe runs; each run acts as a corridor pointing away from the rectangle.
    - Starting from the rectangle-adjacent end of each corridor, stamp the grey template repeatedly outward without rotating it. The gap between consecutive copies on that corridor equals the size of the adjacent excluded block that determines the corridor's color.
    - A corridor inherits the color of the nearest blocking control next to that run; if no blocking control touches that run, use teal. Existing non-background pixels are never overwritten.

    Key insight:
    - The same control pixels do two jobs at once: they cut the rectangle's side extensions into discontinuous runs, and those same cut locations donate the color and spacing for the copied template strip beside them.
    """
    rect_bbox, template, side_controls = _parse_grid(np.asarray(grid, dtype=int))
    out, _ = _apply_rule(np.asarray(grid, dtype=int), rect_bbox, template, side_controls)
    return out


TEMPLATE_MASKS = [
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def _side_count_pattern(total_controls: int) -> dict[str, int]:
    sides = list(SIDES)
    random.shuffle(sides)
    if total_controls == 2:
        active = sides[:2]
        return {side: (1 if side in active else 0) for side in SIDES}
    if total_controls == 3:
        active = sides[:2]
        counts = {side: 0 for side in SIDES}
        counts[active[0]] = 2
        counts[active[1]] = 1
        return counts

    mode = random.choice(["2+1+1", "2+2", "1+1+1+1"])
    counts = {side: 0 for side in SIDES}
    if mode == "2+1+1":
        active = sides[:3]
        counts[active[0]] = 2
        counts[active[1]] = 1
        counts[active[2]] = 1
    elif mode == "2+2":
        active = sides[:2]
        counts[active[0]] = 2
        counts[active[1]] = 2
    else:
        for side in SIDES:
            counts[side] = 1
    return counts


def _pick_indices(side_len: int, count: int) -> list[int]:
    if count == 0:
        return []
    assert count < side_len
    choices = list(range(side_len))
    if count == 1:
        return [random.choice(choices)]
    pairs = [(a, b) for a in choices for b in choices if a < b]
    if random.random() < 0.45:
        adjacent = [pair for pair in pairs if pair[1] == pair[0] + 1]
        if adjacent:
            return list(random.choice(adjacent))
    separated = [pair for pair in pairs if pair[1] > pair[0] + 1]
    return list(random.choice(separated or pairs))


def _choose_spaces(th: int, tw: int, rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    for _ in range(200):
        top = random.randint(max(3, th + 1), 8)
        bottom = random.randint(max(3, th + 1), 8)
        left = random.randint(max(3, tw + 1), 8)
        right = random.randint(max(3, tw + 1), 8)
        if rect_h + top + bottom <= 30 and rect_w + left + right <= 30:
            return top, bottom, left, right
    raise RuntimeError("failed to choose grid margins")


def _place_template(grid: np.ndarray, mask: np.ndarray, corner: str) -> tuple[int, int]:
    th, tw = mask.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, grid.shape[1] - tw
    elif corner == "bl":
        r0, c0 = grid.shape[0] - th, 0
    else:
        r0, c0 = grid.shape[0] - th, grid.shape[1] - tw
    grid[r0 : r0 + th, c0 : c0 + tw][mask] = GREY
    return r0, c0


def _render_spec(spec: dict) -> np.ndarray:
    grid = np.zeros((spec["H"], spec["W"]), dtype=int)
    r0, c0, r1, c1 = spec["rect_bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    _place_template(grid, spec["template"], spec["corner"])

    for control in spec["controls"]:
        if grid[control["r"], control["c"]] != BG:
            raise AssertionError("control overlaps an existing object")
        grid[control["r"], control["c"]] = int(control["color"])
    return grid


def _spec_side_controls(spec: dict) -> dict[str, list[dict]]:
    out = {side: [] for side in SIDES}
    for control in spec["controls"]:
        out[control["side"]].append(control.copy())
    return out


def _validate_semantics(inp: np.ndarray, out: np.ndarray, spec: dict, placements: list[dict]) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False

    visible = [placement for placement in placements if placement["added"] > 0]
    if len(visible) < 3:
        return False

    colors = {int(placement["color"]) for placement in visible}
    if not any(color in CONTROL_COLORS for color in colors):
        return False
    if TEAL not in colors:
        return False

    if not any(placement["gap"] > 0 for placement in visible):
        return False
    if len({placement["side"] for placement in visible}) < 2:
        return False

    rect_bbox = tuple(spec["rect_bbox"])
    template = spec["template"]
    side_controls = _spec_side_controls(spec)
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    reparsed_bbox, reparsed_template, reparsed_controls = _parse_grid(inp)
    if reparsed_bbox != rect_bbox or reparsed_template.shape != template.shape or not np.array_equal(reparsed_template, template):
        return False
    for side in SIDES:
        a = sorted((int(c["index"]), int(c["color"])) for c in side_controls[side])
        b = sorted((int(c["index"]), int(c["color"])) for c in reparsed_controls[side])
        if a != b:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(TEMPLATE_MASKS).copy()
    th, tw = template.shape
    rect_h = random.randint(max(4, th), 6)
    rect_w = random.randint(max(5, tw), 7)
    top, bottom, left, right = _choose_spaces(th, tw, rect_h, rect_w)
    H, W = top + rect_h + bottom, left + rect_w + right
    rect_bbox = (top, left, top + rect_h - 1, left + rect_w - 1)

    total_controls = random.choice((2, 3, 3, 4, 4))
    side_counts = _side_count_pattern(total_controls)
    side_lengths = {"top": rect_w, "bottom": rect_w, "left": rect_h, "right": rect_h}
    distances_max = {"top": top, "bottom": bottom, "left": left, "right": right}

    controls = []
    colors = random.sample(CONTROL_COLORS, total_controls)
    color_iter = iter(colors)
    for side in SIDES:
        count = side_counts[side]
        indices = _pick_indices(side_lengths[side], count)
        distances = random.sample(range(1, distances_max[side] + 1), count) if count else []
        for index, distance in zip(indices, distances):
            color = next(color_iter)
            r0, c0, r1, c1 = rect_bbox
            if side == "top":
                rr, cc = r0 - distance, c0 + index
            elif side == "bottom":
                rr, cc = r1 + distance, c0 + index
            elif side == "left":
                rr, cc = r0 + index, c0 - distance
            else:
                rr, cc = r0 + index, c1 + distance
            controls.append(
                {
                    "side": side,
                    "index": index,
                    "r": rr,
                    "c": cc,
                    "color": color,
                }
            )

    spec = {
        "H": H,
        "W": W,
        "rect_bbox": rect_bbox,
        "template": template,
        "corner": random.choice(("tl", "tr", "bl", "br")),
        "controls": controls,
    }

    try:
        inp = _render_spec(spec)
    except AssertionError:
        return None

    rect_bbox2 = tuple(spec["rect_bbox"])
    side_controls = _spec_side_controls(spec)
    out, placements = _apply_rule(inp, rect_bbox2, template, side_controls)
    if _validate_semantics(inp, out, spec, placements):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
