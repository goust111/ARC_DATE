import random
from collections import defaultdict

import numpy as np


BG = 0
PAIR_COLOR_POOL = list(range(1, 10))
GENERATION_RETRIES = 400


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(coords, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "coords": coords,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _extract_scene(grid):
    g = np.asarray(grid, dtype=int)
    comps = _connected_components(g)
    markers_by_color = defaultdict(list)
    object_comps = []
    for comp in comps:
        coords = comp["coords"]
        if comp["area"] == 1:
            r, c = map(int, coords[0])
            markers_by_color[int(g[r, c])].append((r, c))
        else:
            object_comps.append(comp)

    pair_colors = set()
    lines = {}
    for color, pts in markers_by_color.items():
        if len(pts) != 2:
            raise AssertionError("each marker color must appear in exactly two singleton markers")
        (r0, c0), (r1, c1) = pts
        if r0 != r1:
            raise AssertionError("marker pairs must share a row")
        c_lo, c_hi = sorted((c0, c1))
        lines[int(color)] = (int(r0), int(c_lo), int(c_hi))
        pair_colors.add(int(color))

    objects = []
    for comp in object_comps:
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        cell_map = []
        key_cells = []
        body_colors = set()
        for r, c in coords.tolist():
            color = int(g[r, c])
            cell_map.append((int(r - r0), int(c - c0), color))
            if color in pair_colors:
                key_cells.append((int(r - r0), int(c - c0), color))
            else:
                body_colors.add(color)
        if len(key_cells) != 1:
            raise AssertionError("each object must contain exactly one key pixel")
        if len(body_colors) != 1:
            raise AssertionError("each object must have exactly one body color")
        key_dr, key_dc, key_color = key_cells[0]
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": int(r0),
                "left": int(c0),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "cells": cell_map,
                "key_color": int(key_color),
                "key_rel": (int(key_dr), int(key_dc)),
                "body_color": int(next(iter(body_colors))),
            }
        )
    return lines, objects


def _build_line_grid(grid, lines, draw_lines=True):
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for comp in _connected_components(out):
        if comp["area"] > 1:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
    if draw_lines:
        occupied = {}
        for color, (row, c0, c1) in lines.items():
            for c in range(c0, c1 + 1):
                prev = occupied.get((row, c))
                if prev is not None and prev != color:
                    raise AssertionError("different horizontal lines cannot overlap")
                occupied[(row, c)] = color
                out[row, c] = color
    return out


def _drop_objects(grid, lines, objects, *, draw_lines=True, solid_mode="keyed", consider_objects=True, reverse_order=False, return_trace=False):
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    line_grid = _build_line_grid(g, lines, draw_lines=draw_lines)
    out = line_grid.copy()
    settled_mask = np.zeros_like(g, dtype=bool)
    settled_objects = []
    traces = []

    ordered = sorted(
        objects,
        key=lambda obj: (-obj["bbox"][2], -obj["bbox"][0], obj["bbox"][1]),
        reverse=reverse_order,
    )

    for obj in ordered:
        top = int(obj["top"])
        left = int(obj["left"])
        passed_colors = set()
        last_stop = "bottom"
        while True:
            blocked = False
            for dr, dc, _color in obj["cells"]:
                r = top + dr
                c = left + dc
                nr = r + 1
                if nr >= h:
                    blocked = True
                    last_stop = "bottom"
                    break
                if consider_objects and settled_mask[nr, c]:
                    blocked = True
                    last_stop = "object"
                    break
                line_color = int(line_grid[nr, c])
                if line_color != BG:
                    if solid_mode == "all":
                        blocked = True
                        last_stop = f"line:{line_color}"
                        break
                    if solid_mode == "keyed" and line_color == obj["key_color"]:
                        blocked = True
                        last_stop = f"line:{line_color}"
                        break
                    if solid_mode == "keyed" and line_color != obj["key_color"]:
                        passed_colors.add(line_color)
            if blocked:
                break
            top += 1

        settled = dict(obj)
        settled["top"] = int(top)
        settled["bbox"] = (int(top), int(left), int(top + obj["height"] - 1), int(left + obj["width"] - 1))
        settled_objects.append(settled)
        for dr, dc, color in obj["cells"]:
            rr = top + dr
            cc = left + dc
            settled_mask[rr, cc] = True
            out[rr, cc] = color
        traces.append(
            {
                "key_color": int(obj["key_color"]),
                "start_top": int(obj["top"]),
                "final_top": int(top),
                "passed_colors": passed_colors,
                "stop": last_stop,
                "left": int(left),
            }
        )

    if return_trace:
        return out, settled_objects, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each same-color marker pair into a horizontal floor, then let keyed floating objects fall through non-matching floors.

    Input structure:
    - The background is black.
    - Some colors appear as exactly two isolated singleton markers, and each such pair shares one row.
    - Separate multicolor components are floating objects: each object is a single-color body with exactly one embedded key pixel whose color matches one marker pair.

    Rule:
    - First connect every marker pair with a horizontal segment of that color, inclusive of both markers.
    - Then drop the objects from lower initial positions to higher ones, translating each object straight downward as a rigid shape.
    - An object is blocked only by the bottom edge, already settled objects, or a line segment whose color matches that object's key pixel; other colored lines are ignored and may be passed through unchanged.
    - Output the drawn lines together with all settled objects in their original colors.

    Key insight:
    - The marker pairs create hidden color-specific floors, and because objects settle in low-to-high order, later objects may pass through several wrong floors before stopping on either their own colored floor or an earlier fallen object.
    """
    lines, objects = _extract_scene(np.asarray(grid, dtype=int))
    return _drop_objects(np.asarray(grid, dtype=int), lines, objects)


SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
]


def _rotate_mask(mask, k):
    return np.rot90(mask, k)


def _normalized_cells(mask):
    coords = np.argwhere(mask > 0)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    coords = coords - np.array([r0, c0], dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    out = np.zeros((h, w), dtype=int)
    out[coords[:, 0], coords[:, 1]] = 1
    return out


def _sample_shape(*, max_width=None, max_height=None):
    options = []
    for base in SHAPE_LIBRARY:
        for k in range(4):
            mask = _normalized_cells(_rotate_mask(base, k))
            h, w = mask.shape
            if max_width is not None and w > max_width:
                continue
            if max_height is not None and h > max_height:
                continue
            coords = tuple(map(tuple, np.argwhere(mask > 0).tolist()))
            options.append((coords, mask))
    dedup = {}
    for coords, mask in options:
        dedup[coords] = mask
    return random.choice(list(dedup.values())).copy()


def _make_object(mask, body_color, key_color):
    coords = np.argwhere(mask > 0)
    h, w = mask.shape
    sprite = np.zeros((h, w), dtype=int)
    sprite[coords[:, 0], coords[:, 1]] = int(body_color)
    key_r, key_c = map(int, random.choice(coords.tolist()))
    sprite[key_r, key_c] = int(key_color)
    return sprite


def _sprite_cells(sprite):
    coords = np.argwhere(sprite != BG)
    return [(int(r), int(c), int(sprite[r, c])) for r, c in coords.tolist()]


def _paint_sprite(grid, sprite, top, left):
    h, w = sprite.shape
    patch = sprite != BG
    grid[top : top + h, left : left + w][patch] = sprite[patch]


def _line_endpoints(span_start, span_end):
    return int(span_start), int(span_end)


def _row_is_clear(grid, row, c0, c1):
    return np.all(grid[row, c0 : c1 + 1] == BG)


def _place_markers(grid, color, row, c0, c1):
    if grid[row, c0] != BG or grid[row, c1] != BG:
        return False
    grid[row, c0] = int(color)
    grid[row, c1] = int(color)
    return True


def _target_positions(h, w, sprites, include_bottom_object):
    alien_row = random.randint(6, 8)
    c_floor_row = alien_row + random.randint(4, 5)
    a_floor_row = c_floor_row + random.randint(4, 5)
    extra_row = None
    if include_bottom_object:
        extra_row = a_floor_row + random.randint(2, 3)
    if extra_row is not None and extra_row >= h - 3:
        return None
    lane1_left = random.randint(2, 4)
    lane2_left = random.randint(max(8, w // 2 - 1), min(w - 7, w // 2 + 1))
    lane3_left = random.randint(w - 5, w - 3) if include_bottom_object else None
    if lane2_left <= lane1_left + 4:
        return None
    if include_bottom_object and lane3_left <= lane2_left + 2:
        return None

    sprite_a, sprite_b, sprite_c = sprites[:3]
    h_a, w_a = sprite_a.shape
    h_b, w_b = sprite_b.shape
    h_c, w_c = sprite_c.shape
    if lane1_left + w_a >= lane2_left - 2:
        return None
    if lane2_left + w_c >= w - 1:
        return None

    left_a = lane1_left
    left_b_min = left_a
    left_b_max = left_a + w_a - w_b
    if left_b_max < left_b_min:
        return None
    left_b = random.randint(left_b_min, left_b_max)
    left_c = lane2_left

    pos = {
        "alien_row": alien_row,
        "c_floor_row": c_floor_row,
        "a_floor_row": a_floor_row,
        "extra_row": extra_row,
        "A": (a_floor_row - h_a, left_a),
        "B": (a_floor_row - h_a - h_b, left_b),
        "C": (c_floor_row - h_c, left_c),
    }

    if min(pos["A"][0], pos["B"][0], pos["C"][0]) <= alien_row:
        return None

    if include_bottom_object:
        sprite_d = sprites[3]
        h_d, w_d = sprite_d.shape
        left_d = lane3_left
        if left_d + w_d >= w - 1:
            return None
        pos["D"] = (h - h_d, left_d)
        if pos["D"][0] <= extra_row:
            return None
    return pos


def _build_input_from_plan(h, w, line_specs, object_specs):
    grid = np.zeros((h, w), dtype=int)
    occupied_lines = {}
    for color, (row, c0, c1) in line_specs.items():
        if not _row_is_clear(grid, row, c0, c1):
            return None
        for c in range(c0, c1 + 1):
            prev = occupied_lines.get((row, c))
            if prev is not None and prev != color:
                return None
            occupied_lines[(row, c)] = color
        if not _place_markers(grid, color, row, c0, c1):
            return None

    for obj in object_specs:
        top = int(obj["input_top"])
        left = int(obj["left"])
        sprite = obj["sprite"]
        h_s, w_s = sprite.shape
        if top < 0 or left < 0 or top + h_s > h or left + w_s > w:
            return None
        patch = sprite != BG
        if np.any(grid[top : top + h_s, left : left + w_s][patch] != BG):
            return None
        _paint_sprite(grid, sprite, top, left)
    return grid


def _derive_line_specs(h, w, colors, positions, sprites, include_bottom_object):
    color_a, color_c, color_x = colors[:3]
    sprite_a, sprite_b, sprite_c = sprites[:3]
    a_top, a_left = positions["A"]
    c_top, c_left = positions["C"]
    a_span = _line_endpoints(max(1, a_left - 1), min(w - 2, a_left + sprite_a.shape[1]))
    c_span = _line_endpoints(max(1, c_left - 1), min(w - 2, c_left + sprite_c.shape[1]))
    alien_start = max(1, min(a_left, c_left) - 1)
    alien_end = min(w - 2, max(a_left + sprite_b.shape[1], c_left + sprite_c.shape[1]) + 1)
    line_specs = {
        color_a: (positions["a_floor_row"], a_span[0], a_span[1]),
        color_c: (positions["c_floor_row"], c_span[0], c_span[1]),
        color_x: (positions["alien_row"], alien_start, alien_end),
    }
    if include_bottom_object:
        extra_color = colors[3]
        d_top, d_left = positions["D"]
        sprite_d = sprites[3]
        extra_span = _line_endpoints(max(1, d_left - 1), min(w - 2, d_left + sprite_d.shape[1]))
        line_specs[extra_color] = (positions["extra_row"], extra_span[0], extra_span[1])
    return line_specs


def _build_object_specs(pair_colors, body_colors, positions, sprites, include_bottom_object):
    color_a, color_c, color_x = pair_colors[:3]
    sprite_a, sprite_b, sprite_c = sprites[:3]
    share_key = random.random() < 0.6
    key_b = color_a if share_key else random.choice([color_c, color_x])
    highest_marker_row = positions["alien_row"]

    def pick_input_top(final_top, sprite_h, low, high):
        upper = min(high, highest_marker_row - sprite_h - 1, final_top - 3)
        lower = max(0, low)
        if upper < lower:
            return None
        return random.randint(lower, upper)

    a_input_top = pick_input_top(positions["A"][0], sprite_a.shape[0], 2, positions["A"][0] - 6)
    c_input_top = pick_input_top(positions["C"][0], sprite_c.shape[0], 1, positions["C"][0] - 4)
    if a_input_top is None or c_input_top is None:
        return None
    object_specs = [
        {
            "name": "A",
            "sprite": _make_object(sprite_a, body_colors[0], color_a),
            "final_top": positions["A"][0],
            "left": positions["A"][1],
            "input_top": a_input_top,
        },
        {
            "name": "B",
            "sprite": _make_object(sprite_b, body_colors[1], key_b),
            "final_top": positions["B"][0],
            "left": positions["B"][1],
            "input_top": None,
        },
        {
            "name": "C",
            "sprite": _make_object(sprite_c, body_colors[2], color_c),
            "final_top": positions["C"][0],
            "left": positions["C"][1],
            "input_top": c_input_top,
        },
    ]

    a_input_top = object_specs[0]["input_top"]
    max_b_top = min(a_input_top - sprite_b.shape[0] - 2, highest_marker_row - sprite_b.shape[0] - 1, positions["B"][0] - 3)
    if max_b_top < 0:
        return None
    object_specs[1]["input_top"] = random.randint(0, max_b_top)

    if include_bottom_object:
        key_d = random.choice([color_a, color_c])
        sprite_d = sprites[3]
        d_top = positions["D"][0]
        max_d_input_top = min(highest_marker_row - sprite_d.shape[0] - 1, d_top - 8)
        if max_d_input_top < 1:
            return None
        object_specs.append(
            {
                "name": "D",
                "sprite": _make_object(sprite_d, body_colors[3], key_d),
                "final_top": d_top,
                "left": positions["D"][1],
                "input_top": random.randint(1, max_d_input_top),
            }
        )

    return object_specs


def _objects_match_targets(settled_objects, object_specs):
    by_left = {(obj["left"], obj["height"], obj["width"], obj["key_color"]): obj for obj in settled_objects}
    for spec in object_specs:
        sprite = spec["sprite"]
        key_colors = [int(v) for v in sprite[sprite != BG].ravel().tolist()]
        key_color_counts = defaultdict(int)
        for val in key_colors:
            key_color_counts[val] += 1
        key_color = min(key_color_counts, key=lambda c: (key_color_counts[c], c))
        candidates = [obj for obj in settled_objects if obj["left"] == spec["left"] and obj["height"] == sprite.shape[0] and obj["width"] == sprite.shape[1]]
        if not any(obj["top"] == spec["final_top"] for obj in candidates):
            return False
    return True


def _nondegenerate(grid):
    lines, objects = _extract_scene(grid)
    correct, settled, traces = _drop_objects(grid, lines, objects, return_trace=True)
    no_lines = _drop_objects(grid, lines, objects, draw_lines=False)
    all_solid = _drop_objects(grid, lines, objects, solid_mode="all")
    no_objects = _drop_objects(grid, lines, objects, consider_objects=False)
    reversed_order = _drop_objects(grid, lines, objects, reverse_order=True)

    if np.array_equal(correct, no_lines):
        return False
    if np.array_equal(correct, all_solid):
        return False
    if np.array_equal(correct, no_objects):
        return False
    if np.array_equal(correct, reversed_order):
        return False

    if not any(trace["passed_colors"] for trace in traces):
        return False
    if not any(trace["stop"] == "object" for trace in traces):
        return False
    if not any(trace["stop"].startswith("line:") for trace in traces):
        return False
    return True


def _gen_attempt():
    try:
        h = random.randint(20, 26)
        w = random.randint(16, 24)
        include_bottom_object = random.random() < 0.7

        pair_color_count = 4 if include_bottom_object else 3
        pair_colors = random.sample(PAIR_COLOR_POOL, pair_color_count)
        body_colors = random.sample([c for c in PAIR_COLOR_POOL if c not in pair_colors], 4 if include_bottom_object else 3)

        sprite_a = _sample_shape(max_width=3, max_height=3)
        sprite_b = _sample_shape(max_width=2, max_height=3)
        sprite_c = _sample_shape(max_width=3, max_height=3)
        sprites = [sprite_a, sprite_b, sprite_c]
        if include_bottom_object:
            sprites.append(_sample_shape(max_width=2, max_height=3))

        positions = _target_positions(h, w, sprites, include_bottom_object)
        if positions is None:
            return None

        line_specs = _derive_line_specs(h, w, pair_colors, positions, sprites, include_bottom_object)
        object_specs = _build_object_specs(pair_colors, body_colors, positions, sprites, include_bottom_object)
        if object_specs is None:
            return None

        grid = _build_input_from_plan(h, w, line_specs, object_specs)
        if grid is None:
            return None

        lines, objects = _extract_scene(grid)
        out, settled_objects, _traces = _drop_objects(grid, lines, objects, return_trace=True)
        if not _objects_match_targets(settled_objects, object_specs):
            return None

        if not _nondegenerate(grid):
            return None

        if not np.array_equal(main(grid), out):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
