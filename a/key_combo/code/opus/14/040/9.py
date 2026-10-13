from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
YELLOW = 4
BODY_COLORS = (1, 2, 3, 6, 7)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_FROM_VEC = {v: k for k, v in DIRS.items()}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox_from_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int, steps: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _find_markers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers: list[dict] = []
    for r, c in np.argwhere(g == YELLOW):
        r = int(r)
        c = int(c)
        assert 0 < r < h - 1 and 0 < c < w - 1
        arms = [int(g[r - 1, c]), int(g[r + 1, c]), int(g[r, c - 1]), int(g[r, c + 1])]
        assert len(set(arms)) == 1
        arm_color = arms[0]
        assert arm_color in BODY_COLORS
        markers.append({"center": (r, c), "arm_color": arm_color})
    assert 2 <= len(markers) <= 4
    return markers


def _line_info(
    rem: set[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, set[tuple[int, int]]] | None:
    r0, c0, r1, c1 = bbox
    rows = sorted({r for r, _ in rem})
    cols = sorted({c for _, c in rem})
    if len(cols) == 1:
        col = cols[0]
        if not (c0 <= col <= c1):
            return None
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        if rows[-1] == r0 - 1:
            return -1, 0, len(rows), rem
        if rows[0] == r1 + 1:
            return 1, 0, len(rows), rem
        return None
    if len(rows) == 1:
        row = rows[0]
        if not (r0 <= row <= r1):
            return None
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        if cols[-1] == c0 - 1:
            return 0, -1, len(cols), rem
        if cols[0] == c1 + 1:
            return 0, 1, len(cols), rem
        return None
    return None


def _parse_object_component(color: int, comp: list[tuple[int, int]]) -> dict:
    comp_set = {tuple(map(int, cell)) for cell in comp}
    rb0, cb0, rb1, cb1 = _bbox_from_cells(comp_set)
    candidates = []
    for r0 in range(rb0, rb1 + 1):
        for r1 in range(r0 + 1, rb1 + 1):
            for c0 in range(cb0, cb1 + 1):
                for c1 in range(c0 + 1, cb1 + 1):
                    bbox = (r0, c0, r1, c1)
                    rect = _bbox_cells(bbox)
                    if not rect <= comp_set:
                        continue
                    rem = comp_set - rect
                    if not (2 <= len(rem) <= 5):
                        continue
                    info = _line_info(rem, bbox)
                    if info is None:
                        continue
                    dr, dc, rod_len, rod_cells = info
                    candidates.append(
                        {
                            "color": color,
                            "body_bbox": bbox,
                            "body_cells": rect,
                            "rod_cells": set(rod_cells),
                            "dir": (dr, dc),
                            "rod_len": rod_len,
                            "component_cells": rect | set(rod_cells),
                        }
                    )
    assert candidates, f"no valid rectangle+rod parse for color {color}"
    max_area = max(len(c["body_cells"]) for c in candidates)
    candidates = [c for c in candidates if len(c["body_cells"]) == max_area]
    keys = {
        (
            c["body_bbox"],
            tuple(sorted(c["body_cells"])),
            tuple(sorted(c["rod_cells"])),
            c["dir"],
            c["rod_len"],
        )
        for c in candidates
    }
    assert len(keys) == 1, f"ambiguous rectangle+rod parse for color {color}"
    parsed = candidates[0]
    parsed["top_left"] = parsed["body_bbox"][:2]
    return parsed


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    markers = _find_markers(g)
    arm_counts = Counter(m["arm_color"] for m in markers)
    repeated = [color for color, count in arm_counts.items() if count >= 2]
    assert len(repeated) == 1
    priority_color = repeated[0]
    active_colors = set(arm_counts)

    objects: dict[int, dict] = {}
    for color in sorted(set(map(int, np.unique(g))) - {BG, YELLOW}):
        comps = _components_of_color(g, color)
        large = [comp for comp in comps if len(comp) > 1]
        if not large:
            continue
        assert len(large) == 1, f"expected one body component for color {color}"
        objects[color] = _parse_object_component(color, large[0])

    assert 3 <= len(objects) <= 5
    assert active_colors <= set(objects)
    inactive_colors = set(objects) - active_colors
    assert inactive_colors
    return {
        "markers": markers,
        "arm_counts": arm_counts,
        "priority_color": priority_color,
        "active_colors": active_colors,
        "inactive_colors": inactive_colors,
        "objects": objects,
    }


def _simulate_boxes(
    shape: tuple[int, int],
    objects: dict[int, dict],
    active_colors: set[int],
    priority_color: int,
    order_override: list[int] | None = None,
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, int], list[int]]:
    current = {color: obj["body_bbox"] for color, obj in objects.items()}
    final_boxes = {color: obj["body_bbox"] for color, obj in objects.items()}
    move_steps = {color: 0 for color in objects}

    if order_override is None:
        others = sorted(
            (color for color in active_colors if color != priority_color),
            key=lambda color: objects[color]["top_left"],
        )
        order = [priority_color] + others
    else:
        order = list(order_override)

    for color in order:
        obj = objects[color]
        dr, dc = obj["dir"]
        steps = 0
        while steps < obj["rod_len"]:
            trial = _shift_bbox(obj["body_bbox"], dr, dc, steps + 1)
            if not _bbox_in_bounds(trial, shape):
                break
            trial_cells = _bbox_cells(trial)
            blocked = False
            for other_color, other_bbox in current.items():
                if other_color == color:
                    continue
                if trial_cells & _bbox_cells(other_bbox):
                    blocked = True
                    break
            if blocked:
                break
            steps += 1
        final = _shift_bbox(obj["body_bbox"], dr, dc, steps)
        current[color] = final
        final_boxes[color] = final
        move_steps[color] = steps
    return final_boxes, move_steps, order


def _render_from_analysis(
    grid: np.ndarray,
    analysis: dict,
    active_colors: set[int] | None = None,
    order_override: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    if active_colors is None:
        active_colors = set(analysis["active_colors"])
    final_boxes, move_steps, order = _simulate_boxes(
        g.shape,
        analysis["objects"],
        active_colors,
        analysis["priority_color"],
        order_override=order_override,
    )

    out = g.copy()
    for color in active_colors:
        for r, c in analysis["objects"][color]["component_cells"]:
            out[r, c] = BG
    for color in active_colors:
        for r, c in _bbox_cells(final_boxes[color]):
            out[r, c] = color
    return out, move_steps, order


def main(grid: np.ndarray) -> np.ndarray:
    """Use repeated yellow-centered arm colors to choose which rod-linked rectangles slide first and which later stop on collision.

    Input structure:
    - Black background with 3-5 solid monochrome rectangles, each attached to a same-color 1-cell-wide straight rod.
    - Separate yellow-centered plus markers appear elsewhere; each marker has a yellow center and four same-color orthogonal arms.
    - Exactly one arm color is repeated across multiple markers, and at least one rectangle color appears on no marker.

    Rule:
    - Read every yellow-centered marker and collect its arm color; marker colors select the active rectangles.
    - The unique repeated arm color is the priority color, so that rectangle slides first along its own rod direction by up to its rod length, stopping early at the border or another rectangle body.
    - All other active rectangles then slide the same way in row-major order of their input body bounding boxes; rods of moved rectangles are erased, while markers and inactive rectangle+rod objects stay unchanged.

    Key insight:
    - The plus markers encode two hidden instructions at once: arm color selects which rectangles are allowed to move, and repeated arm color overrides the normal row-major motion order to create the decisive collision blocker.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    out, _, _ = _render_from_analysis(np.asarray(grid, dtype=int), analysis)
    return out


def _make_object(
    color: int,
    body_bbox: tuple[int, int, int, int],
    direction: str,
    rod_len: int,
    attach_index: int,
) -> dict:
    r0, c0, r1, c1 = body_bbox
    assert 2 <= rod_len <= 5
    if direction == "right":
        row = r0 + attach_index
        rod = {(row, c1 + d) for d in range(1, rod_len + 1)}
    elif direction == "left":
        row = r0 + attach_index
        rod = {(row, c0 - d) for d in range(1, rod_len + 1)}
    elif direction == "down":
        col = c0 + attach_index
        rod = {(r1 + d, col) for d in range(1, rod_len + 1)}
    else:
        col = c0 + attach_index
        rod = {(r0 - d, col) for d in range(1, rod_len + 1)}

    obj = {
        "color": color,
        "body_bbox": body_bbox,
        "body_cells": _bbox_cells(body_bbox),
        "rod_cells": rod,
        "component_cells": _bbox_cells(body_bbox) | rod,
        "dir": DIRS[direction],
        "rod_len": rod_len,
        "top_left": body_bbox[:2],
    }
    return obj


def _object_in_bounds(obj: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in obj["component_cells"])


def _cells_disjoint(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return not (a & b)


def _draw_object(grid: np.ndarray, obj: dict):
    for r, c in obj["component_cells"]:
        grid[r, c] = obj["color"]


def _place_core_pair(h: int, w: int, priority_color: int, follower_color: int) -> tuple[dict, dict] | None:
    for _ in range(150):
        body_h = random.randint(2, 3)
        follower_w = random.randint(2, 4)
        priority_w = random.randint(2, 4)
        follower_move = random.randint(2, 4)
        follower_rod = follower_move + 1
        priority_move = random.randint(2, 4)

        min_cf = follower_move + 1
        max_cf = w - (follower_w + priority_w + 2)
        max_rf = h - body_h - priority_move - 2
        if min_cf > max_cf or max_rf < 1:
            continue

        final_cf = random.randint(min_cf, max_cf)
        row = random.randint(1, max_rf)
        follower_box = (row, final_cf - follower_move, row + body_h - 1, final_cf - follower_move + follower_w - 1)
        priority_final = (row, final_cf + follower_w, row + body_h - 1, final_cf + follower_w + priority_w - 1)
        priority_box = _shift_bbox(priority_final, 1, 0, priority_move)

        follower = _make_object(
            follower_color,
            follower_box,
            "right",
            follower_rod,
            attach_index=random.randrange(body_h),
        )
        priority = _make_object(
            priority_color,
            priority_box,
            "up",
            priority_move,
            attach_index=random.randint(1, priority_w - 1),
        )
        if not (_object_in_bounds(follower, (h, w)) and _object_in_bounds(priority, (h, w))):
            continue
        if not _cells_disjoint(follower["component_cells"], priority["component_cells"]):
            continue
        return priority, follower
    return None


def _try_place_movable_object(
    h: int,
    w: int,
    color: int,
    input_filled: set[tuple[int, int]],
    output_filled: set[tuple[int, int]],
    motion_reserved_body: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int, int, int]] | None:
    for _ in range(250):
        body_h = random.randint(2, 3)
        body_w = random.randint(2, 4)
        direction = random.choice(list(DIRS))
        move_len = random.randint(2, 4)

        max_r0 = h - body_h
        max_c0 = w - body_w
        if max_r0 < 0 or max_c0 < 0:
            return None
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)
        body_bbox = (r0, c0, r0 + body_h - 1, c0 + body_w - 1)
        attach_index = random.randrange(body_h if direction in ("left", "right") else body_w)
        obj = _make_object(color, body_bbox, direction, move_len, attach_index)
        final_bbox = _shift_bbox(body_bbox, *DIRS[direction], move_len)
        if not (_object_in_bounds(obj, (h, w)) and _bbox_in_bounds(final_bbox, (h, w))):
            continue
        if not _cells_disjoint(obj["component_cells"], input_filled):
            continue
        final_cells = _bbox_cells(final_bbox)
        if not _cells_disjoint(final_cells, output_filled):
            continue

        clear = True
        for step in range(move_len + 1):
            sweep = _bbox_cells(_shift_bbox(body_bbox, *DIRS[direction], step))
            if step > 0 and not _cells_disjoint(sweep, motion_reserved_body):
                clear = False
                break
        if not clear:
            continue
        return obj, final_bbox
    return None


def _try_place_inactive_object(
    h: int,
    w: int,
    color: int,
    input_filled: set[tuple[int, int]],
    output_filled: set[tuple[int, int]],
    motion_reserved_body: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int, int, int]] | None:
    for _ in range(250):
        body_h = random.randint(2, 3)
        body_w = random.randint(2, 4)
        direction = random.choice(list(DIRS))
        move_len = random.randint(2, 4)
        max_r0 = h - body_h
        max_c0 = w - body_w
        if max_r0 < 0 or max_c0 < 0:
            return None
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)
        body_bbox = (r0, c0, r0 + body_h - 1, c0 + body_w - 1)
        attach_index = random.randrange(body_h if direction in ("left", "right") else body_w)
        obj = _make_object(color, body_bbox, direction, move_len, attach_index)
        hypo_final = _shift_bbox(body_bbox, *DIRS[direction], move_len)
        if not (_object_in_bounds(obj, (h, w)) and _bbox_in_bounds(hypo_final, (h, w))):
            continue
        if not _cells_disjoint(obj["component_cells"], input_filled):
            continue
        if not _cells_disjoint(obj["component_cells"], output_filled):
            continue

        clear = True
        for step in range(move_len + 1):
            sweep = _bbox_cells(_shift_bbox(body_bbox, *DIRS[direction], step))
            if step > 0 and not _cells_disjoint(sweep, motion_reserved_body):
                clear = False
                break
        if not clear:
            continue
        return obj, hypo_final
    return None


def _place_markers(
    h: int,
    w: int,
    marker_colors: list[int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    centers: list[tuple[int, int, int]] = []
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for color in marker_colors:
        placed = False
        for r, c in candidates:
            cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            if cells & reserved:
                continue
            if any(max(abs(r - rr), abs(c - cc)) < 4 for rr, cc, _ in centers):
                continue
            centers.append((r, c, color))
            reserved |= cells
            placed = True
            break
        if not placed:
            return None
    return centers


def _apply_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    k = transform_id % 4
    g = np.rot90(grid, k)
    if transform_id >= 4:
        g = np.fliplr(g)
    return np.asarray(g, dtype=int)


def _validate_generated_input(grid: np.ndarray):
    analysis = _analyze_grid(grid)
    assert len(analysis["markers"]) in (3, 4)
    assert len(analysis["active_colors"]) in (2, 3)
    assert analysis["arm_counts"][analysis["priority_color"]] == 2

    correct_out, move_steps, _ = _render_from_analysis(grid, analysis)
    blocked = [
        color
        for color in analysis["active_colors"]
        if color != analysis["priority_color"] and move_steps[color] < analysis["objects"][color]["rod_len"]
    ]
    assert len(blocked) == 1
    follower = blocked[0]
    assert analysis["objects"][analysis["priority_color"]]["top_left"] > analysis["objects"][follower]["top_left"]

    rowmajor_order = sorted(analysis["active_colors"], key=lambda color: analysis["objects"][color]["top_left"])
    wrong_priority_out, wrong_priority_steps, _ = _render_from_analysis(
        grid,
        analysis,
        active_colors=set(analysis["active_colors"]),
        order_override=rowmajor_order,
    )
    assert wrong_priority_steps[follower] > move_steps[follower]
    assert not np.array_equal(correct_out, wrong_priority_out)

    all_active_out, all_active_steps, _ = _render_from_analysis(
        grid,
        analysis,
        active_colors=set(analysis["objects"]),
    )
    inactive_moves = [
        all_active_steps[color]
        for color in sorted(analysis["inactive_colors"])
    ]
    assert any(step > 0 for step in inactive_moves)
    assert not np.array_equal(correct_out, all_active_out)
    assert not np.array_equal(correct_out, grid)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    colors = random.sample(BODY_COLORS, random.randint(3, 5))
    priority_color = colors[0]
    follower_color = colors[1]
    extra_active = colors[2] if len(colors) >= 4 and random.random() < 0.6 else None
    inactive_colors = [c for c in colors if c not in {priority_color, follower_color, extra_active}]
    if not inactive_colors:
        return None

    core = _place_core_pair(h, w, priority_color, follower_color)
    if core is None:
        return None
    priority_obj, follower_obj = core

    objects: dict[int, dict] = {
        priority_color: priority_obj,
        follower_color: follower_obj,
    }
    input_filled = set(priority_obj["component_cells"]) | set(follower_obj["component_cells"])
    output_filled = _bbox_cells(_shift_bbox(priority_obj["body_bbox"], *priority_obj["dir"], priority_obj["rod_len"]))
    output_filled |= _bbox_cells(_shift_bbox(follower_obj["body_bbox"], *follower_obj["dir"], follower_obj["rod_len"] - 1))
    motion_reserved_body = set(priority_obj["body_cells"]) | set(follower_obj["body_cells"])
    motion_reserved_body |= _bbox_cells(_shift_bbox(priority_obj["body_bbox"], *priority_obj["dir"], priority_obj["rod_len"]))
    motion_reserved_body |= _bbox_cells(_shift_bbox(follower_obj["body_bbox"], *follower_obj["dir"], follower_obj["rod_len"] - 1))

    if extra_active is not None:
        placed = _try_place_movable_object(h, w, extra_active, input_filled, output_filled, motion_reserved_body)
        if placed is None:
            return None
        obj, final_bbox = placed
        objects[extra_active] = obj
        input_filled |= obj["component_cells"]
        output_filled |= _bbox_cells(final_bbox)
        motion_reserved_body |= obj["body_cells"] | _bbox_cells(final_bbox)

    for color in inactive_colors:
        placed = _try_place_inactive_object(h, w, color, input_filled, output_filled, motion_reserved_body)
        if placed is None:
            return None
        obj, hypo_final = placed
        objects[color] = obj
        input_filled |= obj["component_cells"]
        output_filled |= obj["component_cells"]
        motion_reserved_body |= obj["body_cells"] | _bbox_cells(hypo_final)

    marker_colors = [priority_color, priority_color, follower_color]
    if extra_active is not None:
        marker_colors.append(extra_active)
    random.shuffle(marker_colors)
    markers = _place_markers(h, w, marker_colors, set(input_filled | output_filled))
    if markers is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for obj in objects.values():
        _draw_object(grid, obj)
    for r, c, color in markers:
        grid[r, c] = YELLOW
        grid[r - 1, c] = color
        grid[r + 1, c] = color
        grid[r, c - 1] = color
        grid[r, c + 1] = color

    transforms = list(range(8))
    random.shuffle(transforms)
    for transform_id in transforms:
        transformed = _apply_transform(grid, transform_id)
        try:
            _validate_generated_input(transformed)
        except AssertionError:
            continue
        return transformed
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-9 input")
