import random
from collections import deque

import numpy as np


BG = 0
BRIDGE = 5
RECT_COLORS = tuple(c for c in range(10) if c not in {BG, BRIDGE})

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RECT_SIDE_RANGE = (3, 7)
GROUP_COUNT_OPTIONS = (1, 2)
EXTRA_RECT_MAX = 2
GROUP_BUILD_RETRIES = 80
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400
GROUP_MARGIN = 1
BRIDGE_GAP_RANGE = (2, 5)


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _rect_area(rect: dict) -> int:
    return (rect["r1"] - rect["r0"] + 1) * (rect["c1"] - rect["c0"] + 1)


def _extract_rectangles(grid: np.ndarray) -> dict[int, dict]:
    rectangles: dict[int, dict] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in {BG, BRIDGE}:
            continue
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == color) | (patch == BRIDGE)), "foreground objects must be bridge-cut rectangles"
        rect = {
            "color": color,
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
        }
        assert _rect_area(rect) >= 9, "rectangles must be large enough"
        rectangles[color] = rect
    assert rectangles, "expected at least one rectangle"
    return rectangles


def _extract_bridges(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bridges: list[dict] = []
    for coords in _label_components(grid == BRIDGE):
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            row = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert np.all(grid[row, c0 : c1 + 1] == BRIDGE), "horizontal bridge must be contiguous"
            assert 0 < c0 and c1 < w - 1, "bridge must terminate at rectangles"
            left_color = int(grid[row, c0 - 1])
            right_color = int(grid[row, c1 + 1])
            assert left_color not in {BG, BRIDGE} and right_color not in {BG, BRIDGE}
            assert left_color != right_color
            bridges.append(
                {
                    "orientation": "h",
                    "row": row,
                    "c0": c0,
                    "c1": c1,
                    "end_colors": (left_color, right_color),
                }
            )
        else:
            assert len(cols) == 1, "bridge components must be straight"
            col = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert np.all(grid[r0 : r1 + 1, col] == BRIDGE), "vertical bridge must be contiguous"
            assert 0 < r0 and r1 < h - 1, "bridge must terminate at rectangles"
            top_color = int(grid[r0 - 1, col])
            bottom_color = int(grid[r1 + 1, col])
            assert top_color not in {BG, BRIDGE} and bottom_color not in {BG, BRIDGE}
            assert top_color != bottom_color
            bridges.append(
                {
                    "orientation": "v",
                    "col": col,
                    "r0": r0,
                    "r1": r1,
                    "end_colors": (top_color, bottom_color),
                }
            )
    assert bridges, "expected at least one bridge"
    return bridges


def _paint_bridge(grid: np.ndarray, bridge: dict, color: int) -> None:
    if bridge["orientation"] == "h":
        grid[bridge["row"], bridge["c0"] : bridge["c1"] + 1] = color
    else:
        grid[bridge["r0"] : bridge["r1"] + 1, bridge["col"]] = color


def _restore_crossed_rectangles(grid: np.ndarray, bridge: dict, rectangles: dict[int, dict]) -> None:
    endpoints = set(bridge["end_colors"])
    if bridge["orientation"] == "h":
        row = bridge["row"]
        for color, rect in rectangles.items():
            if color in endpoints or not (rect["r0"] <= row <= rect["r1"]):
                continue
            lo = max(bridge["c0"], rect["c0"])
            hi = min(bridge["c1"], rect["c1"])
            if lo <= hi:
                grid[row, lo : hi + 1] = color
    else:
        col = bridge["col"]
        for color, rect in rectangles.items():
            if color in endpoints or not (rect["c0"] <= col <= rect["c1"]):
                continue
            lo = max(bridge["r0"], rect["r0"])
            hi = min(bridge["r1"], rect["r1"])
            if lo <= hi:
                grid[lo : hi + 1, col] = color


def _apply_bridge_rule(grid: np.ndarray, bridges: list[dict], rectangles: dict[int, dict]) -> np.ndarray:
    out = grid.copy()
    for bridge in bridges:
        a, b = bridge["end_colors"]
        rect_a = rectangles[a]
        rect_b = rectangles[b]
        area_a = _rect_area(rect_a)
        area_b = _rect_area(rect_b)
        assert area_a != area_b, "bridge endpoints must have different rectangle areas"
        base_color = a if area_a < area_b else b
        _paint_bridge(out, bridge, base_color)
        _restore_crossed_rectangles(out, bridge, rectangles)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey bridge with the color of the smaller endpoint rectangle,
    except for bridge segments that pass through another rectangle, which take
    that crossed rectangle's color instead.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"
    rectangles = _extract_rectangles(g)
    bridges = _extract_bridges(g)
    return _apply_bridge_rule(g, bridges, rectangles)


def _sample_top(group_h: int, rect_h: int, line_row: int, require_interior: bool) -> int | None:
    if require_interior:
        lo = max(0, line_row - rect_h + 2)
        hi = min(line_row - 1, group_h - rect_h)
    else:
        lo = max(0, line_row - rect_h + 1)
        hi = min(line_row, group_h - rect_h)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_horizontal_group(colors: list[int], cross: bool) -> dict | None:
    count = 3 if cross else 2
    for _ in range(GROUP_BUILD_RETRIES):
        dims = [
            (random.randint(*RECT_SIDE_RANGE), random.randint(*RECT_SIDE_RANGE))
            for _ in range(count)
        ]
        if dims[0][0] * dims[0][1] == dims[-1][0] * dims[-1][1]:
            continue
        gaps = [random.randint(*BRIDGE_GAP_RANGE) for _ in range(count - 1)]
        group_h = max(h for h, _w in dims) + random.randint(2, 4)
        if group_h < 5:
            continue
        line_row = random.randint(1, group_h - 2)

        tops = []
        ok = True
        for idx, (rect_h, _rect_w) in enumerate(dims):
            require_interior = cross and idx == 1
            top = _sample_top(group_h, rect_h, line_row, require_interior=require_interior)
            if top is None:
                ok = False
                break
            tops.append(top)
        if not ok:
            continue

        rects = []
        cur_c = 0
        for color, (rect_h, rect_w), top in zip(colors, dims, tops):
            rects.append(
                {
                    "color": color,
                    "r0": top,
                    "c0": cur_c,
                    "r1": top + rect_h - 1,
                    "c1": cur_c + rect_w - 1,
                }
            )
            cur_c += rect_w
            if len(rects) < count:
                cur_c += gaps[len(rects) - 1]

        bridge = {
            "orientation": "h",
            "row": line_row,
            "c0": rects[0]["c1"] + 1,
            "c1": rects[-1]["c0"] - 1,
            "end_colors": (rects[0]["color"], rects[-1]["color"]),
        }
        return {"shape": (group_h, cur_c), "rects": rects, "bridges": [bridge]}
    return None


def _transpose_group(group: dict) -> dict:
    rects = []
    for rect in group["rects"]:
        rects.append(
            {
                "color": rect["color"],
                "r0": rect["c0"],
                "c0": rect["r0"],
                "r1": rect["c1"],
                "c1": rect["r1"],
            }
        )
    bridges = []
    for bridge in group["bridges"]:
        assert bridge["orientation"] == "h"
        bridges.append(
            {
                "orientation": "v",
                "col": bridge["row"],
                "r0": bridge["c0"],
                "r1": bridge["c1"],
                "end_colors": bridge["end_colors"],
            }
        )
    h, w = group["shape"]
    return {"shape": (w, h), "rects": rects, "bridges": bridges}


def _build_group(colors: list[int], orientation: str, cross: bool) -> dict | None:
    group = _build_horizontal_group(colors, cross)
    if group is None:
        return None
    return group if orientation == "h" else _transpose_group(group)


def _candidate_positions(blocked: np.ndarray, box_h: int, box_w: int, margin: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions = []
    for r0 in range(h - box_h + 1):
        for c0 in range(w - box_w + 1):
            rr0 = max(0, r0 - margin)
            cc0 = max(0, c0 - margin)
            rr1 = min(h - 1, r0 + box_h - 1 + margin)
            cc1 = min(w - 1, c0 + box_w - 1 + margin)
            if not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                positions.append((r0, c0))
    return positions


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, margin: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r0 + box_h - 1 + margin)
    cc1 = min(w - 1, c0 + box_w - 1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _shift_group(group: dict, top: int, left: int) -> tuple[list[dict], list[dict]]:
    rects = []
    for rect in group["rects"]:
        rects.append(
            {
                "color": rect["color"],
                "r0": rect["r0"] + top,
                "c0": rect["c0"] + left,
                "r1": rect["r1"] + top,
                "c1": rect["c1"] + left,
            }
        )
    bridges = []
    for bridge in group["bridges"]:
        if bridge["orientation"] == "h":
            bridges.append(
                {
                    "orientation": "h",
                    "row": bridge["row"] + top,
                    "c0": bridge["c0"] + left,
                    "c1": bridge["c1"] + left,
                    "end_colors": bridge["end_colors"],
                }
            )
        else:
            bridges.append(
                {
                    "orientation": "v",
                    "col": bridge["col"] + left,
                    "r0": bridge["r0"] + top,
                    "r1": bridge["r1"] + top,
                    "end_colors": bridge["end_colors"],
                }
            )
    return rects, bridges


def _place_extra_rectangles(
    blocked: np.ndarray,
    available_colors: list[int],
) -> list[dict] | None:
    extras: list[dict] = []
    max_extra = min(EXTRA_RECT_MAX, len(available_colors))
    count = random.randint(0, max_extra)
    for color in available_colors[:count]:
        rect_h = random.randint(*RECT_SIDE_RANGE)
        rect_w = random.randint(*RECT_SIDE_RANGE)
        positions = _candidate_positions(blocked, rect_h, rect_w, GROUP_MARGIN)
        if not positions:
            return None
        top, left = random.choice(positions)
        _mark_blocked(blocked, top, left, rect_h, rect_w, GROUP_MARGIN)
        extras.append(
            {
                "color": color,
                "r0": top,
                "c0": left,
                "r1": top + rect_h - 1,
                "c1": left + rect_w - 1,
            }
        )
    return extras


def _render_rectangles(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _make_scene() -> tuple[np.ndarray, np.ndarray] | None:
    group_count = random.choice(GROUP_COUNT_OPTIONS)
    if 3 * group_count > len(RECT_COLORS):
        return None

    colors = list(RECT_COLORS)
    random.shuffle(colors)
    color_idx = 0
    groups = []
    for _ in range(group_count):
        orientation = random.choice(("h", "v"))
        cross = bool(random.getrandbits(1))
        needed = 3 if cross else 2
        group_colors = colors[color_idx : color_idx + needed]
        color_idx += needed
        group = _build_group(group_colors, orientation, cross)
        if group is None:
            return None
        groups.append(group)

    max_group_h = max(group["shape"][0] for group in groups)
    max_group_w = max(group["shape"][1] for group in groups)
    min_h = max(GRID_H_RANGE[0], max_group_h + 2)
    min_w = max(GRID_W_RANGE[0], max_group_w + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    scene_rects: list[dict] = []
    scene_bridges: list[dict] = []
    for group in sorted(groups, key=lambda g: g["shape"][0] * g["shape"][1], reverse=True):
        positions = _candidate_positions(blocked, group["shape"][0], group["shape"][1], GROUP_MARGIN)
        if not positions:
            return None
        top, left = random.choice(positions)
        _mark_blocked(blocked, top, left, group["shape"][0], group["shape"][1], GROUP_MARGIN)
        rects, bridges = _shift_group(group, top, left)
        scene_rects.extend(rects)
        scene_bridges.extend(bridges)

    extras = _place_extra_rectangles(blocked, colors[color_idx:])
    if extras is None:
        return None
    scene_rects.extend(extras)

    rectangles_by_color = {rect["color"]: rect for rect in scene_rects}
    full = _render_rectangles((grid_h, grid_w), scene_rects)
    inp = full.copy()
    for bridge in scene_bridges:
        _paint_bridge(inp, bridge, BRIDGE)

    expected = full.copy()
    for bridge in scene_bridges:
        a, b = bridge["end_colors"]
        base = a if _rect_area(rectangles_by_color[a]) < _rect_area(rectangles_by_color[b]) else b
        _paint_bridge(expected, bridge, base)
        _restore_crossed_rectangles(expected, bridge, rectangles_by_color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        scene = _make_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
