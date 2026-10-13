from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _part_mask(kind: str, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    if kind == "container":
        mask[-1, :] = True
    elif kind == "roof":
        mask[0, :] = True
    else:
        raise ValueError(f"unknown part kind: {kind}")
    return mask


def _parse_parts(grid: np.ndarray, wall_color: int) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    parts: list[dict] = []
    coord_to_part: dict[tuple[int, int], dict] = {}
    for coords in _components_of_color(grid, wall_color):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True

        kind = None
        if height >= 2 and width >= 3 and np.array_equal(local, _part_mask("container", height, width)):
            kind = "container"
        elif height >= 2 and width >= 4 and np.array_equal(local, _part_mask("roof", height, width)):
            kind = "roof"
        else:
            raise AssertionError("wall component must be a rectangular U container or roof")

        part = {
            "kind": kind,
            "top": r0,
            "left": c0,
            "bottom": r1,
            "right": c1,
            "height": height,
            "width": width,
        }
        parts.append(part)
        for rr, cc in coords:
            coord_to_part[(int(rr), int(cc))] = part
    return parts, coord_to_part


def _markers_on_top(grid: np.ndarray, bg: int, wall_color: int) -> list[tuple[int, int]]:
    markers = [(0, c) for c in range(grid.shape[1]) if int(grid[0, c]) not in (bg, wall_color)]
    assert markers, "expected colored top-row markers"
    return markers


def _earliest_container(parts: list[dict], col: int, row: int) -> dict | None:
    candidates = [
        part
        for part in parts
        if part["kind"] == "container" and part["left"] < col < part["right"] and row < part["top"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda part: part["top"])


def _first_wall_below(grid: np.ndarray, wall_color: int, col: int, row: int) -> int | None:
    for rr in range(row + 1, grid.shape[0]):
        if int(grid[rr, col]) == wall_color:
            return rr
    return None


def _trace_marker(
    grid: np.ndarray,
    wall_color: int,
    parts: list[dict],
    coord_to_part: dict[tuple[int, int], dict],
    start_col: int,
) -> dict:
    h, w = grid.shape
    row = 0
    col = start_col
    roofs_hit: list[tuple[int, int, int, int]] = []

    while True:
        container = _earliest_container(parts, col, row)
        wall_row = _first_wall_below(grid, wall_color, col, row)

        if container is not None and (wall_row is None or container["top"] < wall_row):
            return {"kind": "fill", "part": container, "roofs": roofs_hit, "final_col": col}

        if wall_row is None:
            return {"kind": "stop", "cell": (h - 1, col), "reason": "bottom", "roofs": roofs_hit, "final_col": col}

        part = coord_to_part[(wall_row, col)]
        if part["kind"] == "roof" and wall_row == part["top"] and part["left"] < col < part["right"]:
            left_dist = col - part["left"]
            right_dist = part["right"] - col
            assert left_dist != right_dist, "roof landing must have a unique nearer end"
            if left_dist < right_dist:
                col = part["left"] - 1
            else:
                col = part["right"] + 1
            assert 0 <= col < w, "roof reroute must stay inside the grid"
            roofs_hit.append((part["top"], part["left"], part["bottom"], part["right"]))
            row = part["top"]
            continue

        reason = "wall_top"
        return {
            "kind": "stop",
            "cell": (wall_row - 1, col),
            "reason": reason,
            "roofs": roofs_hit,
            "final_col": col,
        }


def _trace_marker_without_roofs(grid: np.ndarray, wall_color: int, parts: list[dict], start_col: int) -> tuple:
    container = _earliest_container(parts, start_col, 0)
    wall_row = _first_wall_below(grid, wall_color, start_col, 0)
    if container is not None and (wall_row is None or container["top"] < wall_row):
        return ("fill", container["top"], container["left"], container["bottom"], container["right"])
    if wall_row is None:
        return ("stop", grid.shape[0] - 1, start_col)
    return ("stop", wall_row - 1, start_col)


def _trace_signature(trace: dict) -> tuple:
    if trace["kind"] == "fill":
        part = trace["part"]
        return ("fill", part["top"], part["left"], part["bottom"], part["right"])
    return ("stop", trace["cell"][0], trace["cell"][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top-row marker through grey roofs and containers, then draw its final resting place.

    Input structure:
    - Black background with several distinct single-pixel colored markers on the top row.
    - Mid/lower rows contain grey rectangular shell parts: open-up U containers and open-down
      roof pieces made of two side walls plus one horizontal cap.
    - Marker paths are non-interfering, and every roof hit occurs at a top-surface column whose
      distances to the two roof ends are unequal.

    Rule:
    - Remove the original top-row markers, then process each marker independently.
    - The marker falls vertically in its current column. If it reaches a roof's top surface
      (strictly between the two roof ends), it slides to the nearer end, moves one column
      outside that end, and continues falling from there.
    - If it enters a U container through the opening, fill that whole interior cavity with the
      marker color and stop. If it instead hits a container wall top, a roof side-wall top, or
      nothing before the bottom edge, stop in the last empty cell of that vertical fall.
    - Keep every grey part unchanged.

    Key insight:
    - The destination is determined by the full cascade of reroutes: roof orientation turns a
      vertical drop into a different column, and later roofs or containers must be evaluated in
      that updated column rather than under the marker's original start column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg, "expected walls and markers"
    counts = Counter(map(int, g.ravel()))
    wall_color = max(non_bg, key=lambda color: (counts[color], color))
    parts, coord_to_part = _parse_parts(g, wall_color)
    markers = _markers_on_top(g, bg, wall_color)

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg

    for _r, c in markers:
        color = int(g[0, c])
        trace = _trace_marker(g, wall_color, parts, coord_to_part, c)
        if trace["kind"] == "fill":
            part = trace["part"]
            out[part["top"] : part["bottom"], part["left"] + 1 : part["right"]] = color
        else:
            rr, cc = trace["cell"]
            assert int(out[rr, cc]) == bg, "marker destinations must be disjoint and empty"
            out[rr, cc] = color

    return out


# --------------- generator ---------------

_BG = 0
_WALL = 5
_MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_LANE_WIDTH = 8
_LANE_GAP = 1
_LANE_COUNT = 3
_HEIGHT_RANGE = (26, 29)
_TOP_CHOICES = (2, 3, 4)
_PART_HEIGHTS = (3, 4)
_MAX_ATTEMPTS = 400


def _roof_patch(height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), _BG, dtype=int)
    patch[_part_mask("roof", height, width)] = _WALL
    return patch


def _container_patch(height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), _BG, dtype=int)
    patch[_part_mask("container", height, width)] = _WALL
    return patch


def _roof_options(local_col: int, *, need_next_interior: bool) -> list[dict]:
    options = []
    for left in range(_LANE_WIDTH):
        for right in range(left + 3, _LANE_WIDTH):
            if not (left < local_col < right):
                continue
            left_dist = local_col - left
            right_dist = right - local_col
            if left_dist == right_dist:
                continue
            next_col = left - 1 if left_dist < right_dist else right + 1
            if not (0 <= next_col < _LANE_WIDTH):
                continue
            if need_next_interior and not (1 <= next_col <= _LANE_WIDTH - 2):
                continue
            options.append({"left": left, "right": right, "next_col": next_col})
    return options


def _container_options(local_col: int) -> list[dict]:
    options = []
    for left in range(_LANE_WIDTH):
        for right in range(left + 3, _LANE_WIDTH):
            if left < local_col < right:
                options.append({"left": left, "right": right})
    return options


def _wall_options(local_col: int) -> list[dict]:
    options = []
    for kind in ("roof", "container"):
        for left in range(_LANE_WIDTH):
            for right in range(left + 3, _LANE_WIDTH):
                if local_col in (left, right):
                    options.append({"kind": kind, "left": left, "right": right})
    return options


def _pick_lane_specs() -> list[str]:
    patterns = [
        "cascade_container",
        random.choice(["bottom", "wall"]),
        random.choice(["direct_container", "single_container", "single_container", "wall"]),
    ]
    random.shuffle(patterns)
    return patterns


def _plan_lane(pattern: str) -> dict | None:
    if pattern == "cascade_container":
        roof_count = 3
    elif pattern == "single_container":
        roof_count = 1
    elif pattern == "direct_container":
        roof_count = 0
    elif pattern == "bottom":
        roof_count = random.choice([1, 2])
    elif pattern == "wall":
        roof_count = random.choice([0, 1, 2])
    else:
        raise ValueError(pattern)

    start_choices = list(range(2, _LANE_WIDTH - 2)) if roof_count else list(range(1, _LANE_WIDTH - 1))
    for _ in range(120):
        current = random.choice(start_choices)
        start = current
        visited = [current]
        route_structures: list[dict] = []

        ok = True
        for step_i in range(roof_count):
            need_next_interior = step_i < roof_count - 1 or pattern in {"cascade_container", "single_container"}
            options = _roof_options(current, need_next_interior=need_next_interior)
            if not options:
                ok = False
                break
            preferred = [opt for opt in options if opt["next_col"] not in visited]
            if preferred:
                options = preferred
            if pattern == "cascade_container" and step_i == roof_count - 1:
                different = [opt for opt in options if opt["next_col"] != start]
                if different:
                    options = different
            chosen = random.choice(options)
            route_structures.append({"kind": "roof", "left": chosen["left"], "right": chosen["right"]})
            current = chosen["next_col"]
            visited.append(current)
        if not ok:
            continue

        if pattern in {"cascade_container", "single_container", "direct_container"}:
            if pattern != "direct_container" and current == start:
                continue
            options = _container_options(current)
            if not options:
                continue
            final = random.choice(options)
            return {
                "pattern": pattern,
                "start_local": start,
                "final_local": current,
                "route_roofs": roof_count,
                "structures": route_structures + [{"kind": "container", "left": final["left"], "right": final["right"]}],
                "outcome": "fill",
            }

        if pattern == "wall":
            options = _wall_options(current)
            if not options:
                continue
            final = random.choice(options)
            return {
                "pattern": pattern,
                "start_local": start,
                "final_local": current,
                "route_roofs": roof_count,
                "structures": route_structures + [final],
                "outcome": "stop_wall",
            }

        if pattern == "bottom":
            return {
                "pattern": pattern,
                "start_local": start,
                "final_local": current,
                "route_roofs": roof_count,
                "structures": route_structures,
                "outcome": "stop_bottom",
            }

    return None


def _assign_verticals(lane: dict, height: int) -> bool:
    top = random.choice(_TOP_CHOICES)
    for structure in lane["structures"]:
        part_height = random.choice(_PART_HEIGHTS)
        structure["height"] = part_height
        structure["top"] = top
        structure["bottom"] = top + part_height - 1
        structure["width"] = structure["right"] - structure["left"] + 1
        top = structure["bottom"] + 2
    return not lane["structures"] or lane["structures"][-1]["bottom"] <= height - 3


def _grid_width() -> int:
    left_margin = random.choice([0, 1])
    right_margin = random.choice([0, 1])
    return left_margin + _LANE_COUNT * _LANE_WIDTH + (_LANE_COUNT - 1) * _LANE_GAP + right_margin


def _lane_bases(width: int) -> list[int]:
    left_margin = width - (_LANE_COUNT * _LANE_WIDTH + (_LANE_COUNT - 1) * _LANE_GAP)
    offset = 0 if left_margin == 0 else random.randint(0, left_margin)
    return [offset + i * (_LANE_WIDTH + _LANE_GAP) for i in range(_LANE_COUNT)]


def _render_input(height: int, width: int, lane_plans: list[dict], colors: list[int]) -> tuple[np.ndarray, dict[int, dict]]:
    grid = np.full((height, width), _BG, dtype=int)
    plan_by_start_col: dict[int, dict] = {}
    bases = _lane_bases(width)

    for lane, color, base in zip(lane_plans, colors, bases):
        start_col = base + lane["start_local"]
        grid[0, start_col] = color

        for structure in lane["structures"]:
            patch_fn = _container_patch if structure["kind"] == "container" else _roof_patch
            patch = patch_fn(structure["height"], structure["width"])
            top = structure["top"]
            left = base + structure["left"]
            grid[top : top + structure["height"], left : left + structure["width"]] = np.maximum(
                grid[top : top + structure["height"], left : left + structure["width"]],
                patch,
            )

        record = {
            "color": color,
            "route_roofs": lane["route_roofs"],
            "outcome": lane["outcome"],
            "start_col": start_col,
        }
        if lane["outcome"] == "fill":
            final = lane["structures"][-1]
            record["bbox"] = (
                final["top"],
                base + final["left"],
                final["bottom"],
                base + final["right"],
            )
        elif lane["outcome"] == "stop_wall":
            final = lane["structures"][-1]
            record["cell"] = (final["top"] - 1, base + lane["final_local"])
        else:
            record["cell"] = (height - 1, base + lane["final_local"])
        plan_by_start_col[start_col] = record

    return grid, plan_by_start_col


def _expected_output(inp: np.ndarray, plan_by_start_col: dict[int, dict]) -> np.ndarray:
    out = inp.copy()
    for start_col, record in plan_by_start_col.items():
        out[0, start_col] = _BG
        color = record["color"]
        if record["outcome"] == "fill":
            top, left, bottom, right = record["bbox"]
            out[top:bottom, left + 1 : right] = color
        else:
            rr, cc = record["cell"]
            out[rr, cc] = color
    return out


def _semantic_accepts(inp: np.ndarray, plan_by_start_col: dict[int, dict]) -> bool:
    bg = _get_bg(inp)
    counts = Counter(map(int, inp.ravel()))
    non_bg = [int(color) for color in np.unique(inp) if int(color) != bg]
    if not non_bg:
        return False
    wall_color = max(non_bg, key=lambda color: (counts[color], color))
    try:
        parts, coord_to_part = _parse_parts(inp, wall_color)
    except AssertionError:
        return False

    markers = _markers_on_top(inp, bg, wall_color)
    if len(markers) != _LANE_COUNT:
        return False
    if len({int(inp[r, c]) for r, c in markers}) != len(markers):
        return False

    roofs = sum(part["kind"] == "roof" for part in parts)
    containers = sum(part["kind"] == "container" for part in parts)
    if roofs < 1 or containers < 1:
        return False

    seen_cells: set[tuple[int, int]] = set()
    saw_fill = False
    saw_stop = False
    saw_cascade = False
    saw_reroute_dependency = False

    for _r, c in markers:
        if c not in plan_by_start_col:
            return False
        trace = _trace_marker(inp, wall_color, parts, coord_to_part, c)
        plan = plan_by_start_col[c]

        if len(trace["roofs"]) != plan["route_roofs"]:
            return False
        if trace["kind"] == "fill":
            saw_fill = True
            if plan["outcome"] != "fill":
                return False
            top, left, bottom, right = plan["bbox"]
            part = trace["part"]
            if (part["top"], part["left"], part["bottom"], part["right"]) != (top, left, bottom, right):
                return False
            cells = {(rr, cc) for rr in range(top, bottom) for cc in range(left + 1, right)}
        else:
            saw_stop = True
            if plan["outcome"] == "fill" or trace["cell"] != plan["cell"]:
                return False
            cells = {trace["cell"]}

        if seen_cells & cells:
            return False
        seen_cells |= cells

        if len(trace["roofs"]) >= 2:
            saw_cascade = True

        if _trace_signature(trace) != _trace_marker_without_roofs(inp, wall_color, parts, c):
            saw_reroute_dependency = True

    return saw_fill and saw_stop and saw_cascade and saw_reroute_dependency


def _gen_attempt() -> np.ndarray | None:
    lane_specs = _pick_lane_specs()
    lanes = []
    for spec in lane_specs:
        lane = _plan_lane(spec)
        if lane is None:
            return None
        lanes.append(lane)

    height = random.randint(*_HEIGHT_RANGE)
    if not all(_assign_verticals(lane, height) for lane in lanes):
        return None

    width = _grid_width()
    colors = random.sample(_MARKER_COLORS, _LANE_COUNT)
    inp, plan_by_start_col = _render_input(height, width, lanes, colors)
    expected = _expected_output(inp, plan_by_start_col)

    if not _semantic_accepts(inp, plan_by_start_col):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid roof-routing sample")
