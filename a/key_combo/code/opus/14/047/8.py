from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

STATIC_COLORS = (RED, GREEN, YELLOW, PINK, TEAL)
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _shift_box(box: tuple[int, int, int, int], direction: tuple[int, int], steps: int) -> tuple[int, int, int, int]:
    dr, dc = direction
    r0, c0, r1, c1 = box
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _interior_order(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return [(r, c) for c in range(inner_w - 1, -1, -1) for r in range(inner_h)]
    if (dr, dc) == (0, -1):
        return [(r, c) for c in range(inner_w) for r in range(inner_h)]
    if (dr, dc) == (1, 0):
        return [(r, c) for r in range(inner_h - 1, -1, -1) for c in range(inner_w)]
    return [(r, c) for r in range(inner_h) for c in range(inner_w)]


def _make_container_patch(outer_h: int, outer_w: int, direction: tuple[int, int], orange_count: int) -> np.ndarray:
    patch = np.zeros((outer_h, outer_w), dtype=int)
    inner = np.full((outer_h - 2, outer_w - 2), GREY, dtype=int)
    order = _interior_order(outer_h - 2, outer_w - 2, direction)
    for rr, cc in order[:orange_count]:
        inner[rr, cc] = ORANGE
    patch[1:-1, 1:-1] = inner
    return patch


def _charge_patch(patch: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    inner = out[1:-1, 1:-1]
    charged = 0
    for rr, cc in _interior_order(inner.shape[0], inner.shape[1], direction):
        if int(inner[rr, cc]) == GREY:
            inner[rr, cc] = ORANGE
            charged += 1
            if charged == steps:
                break
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _max_shift_before_obstacle(
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    obstacle_mask: np.ndarray,
) -> int:
    h, w = obstacle_mask.shape
    r0, c0, r1, c1 = box
    dr, dc = direction
    boundary = {
        (0, 1): w - 1 - c1,
        (0, -1): c0,
        (1, 0): h - 1 - r1,
        (-1, 0): r0,
    }[(dr, dc)]
    for steps in range(1, boundary + 1):
        nr0, nc0, nr1, nc1 = _shift_box(box, direction, steps)
        if obstacle_mask[nr0 : nr1 + 1, nc0 : nc1 + 1].any():
            return steps - 1
    return boundary


def _restore_static_layer(
    shape: tuple[int, int],
    statics: list[dict],
) -> np.ndarray:
    layer = np.zeros(shape, dtype=int)
    for rect in sorted(statics, key=lambda item: -_area(item["box"])):
        r0, c0, r1, c1 = rect["box"]
        region = layer[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BLACK] = int(rect["color"])
    return layer


def _detect_static_rectangles(grid: np.ndarray) -> list[dict]:
    statics: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) in STATIC_COLORS):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        statics.append({"color": int(color), "box": _bbox_from_coords(coords)})
    return statics


def _find_head_and_tail(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    checks = [
        ((0, 1), [(r, c1 + 1) for r in range(r0, r1 + 1)] if c1 + 1 < w else []),
        ((0, -1), [(r, c0 - 1) for r in range(r0, r1 + 1)] if c0 - 1 >= 0 else []),
        ((1, 0), [(r1 + 1, c) for c in range(c0, c1 + 1)] if r1 + 1 < h else []),
        ((-1, 0), [(r0 - 1, c) for c in range(c0, c1 + 1)] if r0 - 1 >= 0 else []),
    ]
    for direction, cells in checks:
        dr, dc = direction
        for rr, cc in cells:
            if int(grid[rr, cc]) != MAROON:
                continue
            br, bc = rr - dr, cc - dc
            if 0 <= br < h and 0 <= bc < w and int(grid[br, bc]) == MAROON:
                continue
            tail = 0
            nr, nc = rr + dr, cc + dc
            while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MAROON:
                tail += 1
                nr += dr
                nc += dc
            if tail > 0:
                candidates.append(((rr, cc), direction, tail))
    assert candidates, f"expected one head per container, got {candidates}"
    candidates.sort(key=lambda item: item[2], reverse=True)
    if len(candidates) > 1 and int(candidates[0][2]) == int(candidates[1][2]):
        raise AssertionError(f"expected one head per container, got {candidates}")
    head, direction, tail = candidates[0]
    return head, direction, int(tail)


def _detect_containers(grid: np.ndarray) -> list[dict]:
    interiors = _components(np.isin(grid, [GREY, ORANGE]))
    containers: list[dict] = []
    h, w = grid.shape
    for comp in interiors:
        ir0, ic0, ir1, ic1 = _bbox_from_coords(comp)
        r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape != (r1 - r0 - 1, c1 - c0 - 1):
            continue
        if not np.isin(inner, [GREY, ORANGE]).all():
            continue
        ring = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        border_vals = grid[r0 : r1 + 1, c0 : c1 + 1][ring]
        if not np.isin(border_vals, [BLACK, MAROON]).all():
            continue
        head, direction, tail = _find_head_and_tail(grid, (r0, c0, r1, c1))
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[1:-1, 1:-1] = inner
        containers.append(
            {
                "box": (r0, c0, r1, c1),
                "patch": patch,
                "head": head,
                "direction": direction,
                "tail_length": tail,
            }
        )
    containers.sort(key=lambda item: item["box"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore hidden static rectangles so they can limit how far each tailed container charges forward.

    Input structure:
    - Black background with 2-3 non-maroon static solid rectangles in distinct colors.
    - One rectangle may look broken because non-static foreground pixels cover part of it.
    - 1-2 black-bordered containers have grey/orange interiors; each has one adjacent maroon head, a straight maroon tail extending outward, and a farther maroon target on the same axis.

    Rule:
    - Treat every non-container, non-maroon foreground color as one static rectangle and restore its full bounding box; larger rectangles are restored first and smaller ones cannot overwrite them.
    - For each container, read the movement direction from the side on which its head and tail sit.
    - Let the move limit be the minimum of tail length, current grey-cell count, and the largest shift that keeps the whole container box from overlapping any restored static rectangle or the grid boundary.
    - Shift only the container box by that many cells, then convert that many grey cells to orange from the head-facing side inward; ties are broken top-to-bottom for horizontal moves and left-to-right for vertical moves.
    - Erase every maroon guide pixel.

    Key insight:
    - The partially hidden rectangle is not decoration: restoring its full box reveals an obstacle that can cap movement earlier than either the tail length or the remaining grey fuel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    statics = _detect_static_rectangles(g)
    restored = _restore_static_layer(g.shape, statics)
    obstacle_mask = restored != BLACK
    containers = _detect_containers(g)

    out = np.array(restored, copy=True)
    for container in containers:
        box = container["box"]
        direction = container["direction"]
        tail_length = int(container["tail_length"])
        grey_count = int(np.sum(container["patch"][1:-1, 1:-1] == GREY))
        obstacle_cap = _max_shift_before_obstacle(box, direction, obstacle_mask)
        steps = min(tail_length, grey_count, obstacle_cap)
        charged_patch = _charge_patch(container["patch"], direction, steps)
        nr0, nc0, _nr1, _nc1 = _shift_box(box, direction, steps)
        _paste(out, charged_patch, nr0, nc0)
    return out


def _ray_cells(
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    tail_length: int,
    target_gap: int,
    anchor: int,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = box
    dr, dc = direction
    if (dr, dc) == (0, 1):
        head = (anchor, c1 + 1)
    elif (dr, dc) == (0, -1):
        head = (anchor, c0 - 1)
    elif (dr, dc) == (1, 0):
        head = (r1 + 1, anchor)
    else:
        head = (r0 - 1, anchor)
    tail = [(head[0] + dr * k, head[1] + dc * k) for k in range(1, tail_length + 1)]
    target = (head[0] + dr * (tail_length + target_gap + 1), head[1] + dc * (tail_length + target_gap + 1))
    return head, tail, target


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _place_guides(grid: np.ndarray, head: tuple[int, int], tail: list[tuple[int, int]], target: tuple[int, int]) -> None:
    for cell in [head, *tail, target]:
        if _in_bounds(grid.shape, cell):
            grid[cell] = MAROON


def _render_input(instance: dict) -> np.ndarray:
    grid = np.zeros(instance["shape"], dtype=int)
    for rect in instance["static_draw_order"]:
        r0, c0, r1, c1 = rect["box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    for container in instance["containers"]:
        r0, c0, r1, c1 = container["box"]
        _paste(grid, container["patch"], r0, c0)
    for container in instance["containers"]:
        _place_guides(grid, container["head"], container["tail"], container["target"])
    return grid


def _render_expected(instance: dict) -> np.ndarray:
    restored = _restore_static_layer(instance["shape"], instance["statics"])
    out = np.array(restored, copy=True)
    for container in instance["containers"]:
        shifted = _shift_box(container["box"], container["direction"], container["steps"])
        charged = _charge_patch(container["patch"], container["direction"], container["steps"])
        _paste(out, charged, shifted[0], shifted[1])
    return out


def _static_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, STATIC_COLORS)


def _rand_box(h: int, w: int, box_h: int, box_w: int) -> tuple[int, int, int, int]:
    r0 = random.randint(0, h - box_h)
    c0 = random.randint(0, w - box_w)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _free_box(
    shape: tuple[int, int],
    box_h: int,
    box_w: int,
    forbidden: list[tuple[int, int, int, int]],
    tries: int = 200,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    for _ in range(tries):
        box = _rand_box(h, w, box_h, box_w)
        if any(_boxes_overlap(box, other) for other in forbidden):
            continue
        return box
    return None


def _guide_extent(
    head: tuple[int, int],
    tail: list[tuple[int, int]],
    target: tuple[int, int],
) -> tuple[int, int, int, int]:
    pts = np.array([head, *tail, target], dtype=int)
    return _bbox_from_coords(pts)


def _build_primary_pair(shape: tuple[int, int]) -> dict | None:
    h, w = shape
    direction = random.choice(list(DIRS.values()))
    oh = random.randint(4, 6)
    ow = random.randint(4, 7)
    sec_len = random.randint(4, 6)
    obstacle_depth = random.randint(4, 6)
    step_primary = random.randint(1, 3)
    tail_primary = random.randint(step_primary + 1, step_primary + 4)
    tail_secondary = random.randint(2, 5)
    gap_primary = random.randint(1, 3)
    gap_secondary = random.randint(1, 3)
    dir_name = next(name for name, vec in DIRS.items() if vec == direction)

    if dir_name in ("right", "left"):
        if h < oh + 2:
            return None
        r0 = random.randint(1, h - oh - 1)
        r1 = r0 + oh - 1
        primary_top = random.random() < 0.5
        head_row_primary = r0 if primary_top else r1
        head_row_secondary = r1 if primary_top else r0
        obstacle_rows = (r0 - 1, r1 + 1)
        if dir_name == "right":
            max_end = max(
                ow - 1 + tail_primary + gap_primary + 2,
                ow - 1 + step_primary + obstacle_depth,
                ow - 1 + step_primary + sec_len + tail_secondary + gap_secondary + 2,
            )
            max_c0 = w - 1 - max_end
            if max_c0 < 0:
                return None
            c0 = random.randint(0, max_c0)
            c1 = c0 + ow - 1
            obstacle_box = (obstacle_rows[0], c1 + step_primary + 1, obstacle_rows[1], c1 + step_primary + obstacle_depth)
            sec_box = (r0, obstacle_box[1], r1, obstacle_box[1] + sec_len - 1)
            anchor_primary = head_row_primary
            anchor_secondary = head_row_secondary
        else:
            min_c0 = max(
                tail_primary + gap_primary + 2,
                step_primary + obstacle_depth,
                step_primary + sec_len + tail_secondary + gap_secondary + 2,
            )
            max_c0 = w - ow
            if min_c0 > max_c0:
                return None
            c0 = random.randint(min_c0, max_c0)
            c1 = c0 + ow - 1
            obstacle_box = (obstacle_rows[0], c0 - step_primary - obstacle_depth, obstacle_rows[1], c0 - step_primary - 1)
            sec_box = (r0, obstacle_box[3] - sec_len + 1, r1, obstacle_box[3])
            anchor_primary = head_row_primary
            anchor_secondary = head_row_secondary
    else:
        if w < ow + 2:
            return None
        c0 = random.randint(1, w - ow - 1)
        c1 = c0 + ow - 1
        primary_left = random.random() < 0.5
        head_col_primary = c0 if primary_left else c1
        head_col_secondary = c1 if primary_left else c0
        obstacle_cols = (c0 - 1, c1 + 1)
        if dir_name == "down":
            max_end = max(
                oh - 1 + tail_primary + gap_primary + 2,
                oh - 1 + step_primary + obstacle_depth,
                oh - 1 + step_primary + sec_len + tail_secondary + gap_secondary + 2,
            )
            max_r0 = h - 1 - max_end
            if max_r0 < 0:
                return None
            r0 = random.randint(0, max_r0)
            r1 = r0 + oh - 1
            obstacle_box = (r1 + step_primary + 1, obstacle_cols[0], r1 + step_primary + obstacle_depth, obstacle_cols[1])
            sec_box = (obstacle_box[0], c0, obstacle_box[0] + sec_len - 1, c1)
            anchor_primary = head_col_primary
            anchor_secondary = head_col_secondary
        else:
            min_r0 = max(
                tail_primary + gap_primary + 2,
                step_primary + obstacle_depth,
                step_primary + sec_len + tail_secondary + gap_secondary + 2,
            )
            max_r0 = h - oh
            if min_r0 > max_r0:
                return None
            r0 = random.randint(min_r0, max_r0)
            r1 = r0 + oh - 1
            obstacle_box = (r0 - step_primary - obstacle_depth, obstacle_cols[0], r0 - step_primary - 1, obstacle_cols[1])
            sec_box = (obstacle_box[2] - sec_len + 1, c0, obstacle_box[2], c1)
            anchor_primary = head_col_primary
            anchor_secondary = head_col_secondary

    if not (0 <= obstacle_box[0] <= obstacle_box[2] < h and 0 <= obstacle_box[1] <= obstacle_box[3] < w):
        return None
    if not (0 <= sec_box[0] <= sec_box[2] < h and 0 <= sec_box[1] <= sec_box[3] < w):
        return None
    if _boxes_overlap(_shift_box((r0, c0, r0 + oh - 1, c0 + ow - 1), direction, step_primary), sec_box):
        return None

    primary_box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
    secondary_box = sec_box
    if _boxes_overlap(primary_box, secondary_box):
        return None

    inner_area_primary = (oh - 2) * (ow - 2)
    inner_area_secondary = (secondary_box[2] - secondary_box[0] - 1) * (secondary_box[3] - secondary_box[1] - 1)
    if inner_area_primary <= step_primary or inner_area_secondary <= 0:
        return None

    orange_primary = random.randint(0, inner_area_primary - step_primary - 1)
    orange_secondary = random.randint(0, inner_area_secondary - 1)
    patch_primary = _make_container_patch(oh, ow, direction, orange_primary)
    patch_secondary = _make_container_patch(
        secondary_box[2] - secondary_box[0] + 1,
        secondary_box[3] - secondary_box[1] + 1,
        direction,
        orange_secondary,
    )
    head1, tail1, target1 = _ray_cells(primary_box, direction, tail_primary, gap_primary, anchor_primary)
    head2, tail2, target2 = _ray_cells(secondary_box, direction, tail_secondary, gap_secondary, anchor_secondary)
    for cell in [head1, *tail1, target1, head2, *tail2, target2]:
        if not _in_bounds(shape, cell):
            return None

    return {
        "primary": {
            "box": primary_box,
            "patch": patch_primary,
            "direction": direction,
            "tail_length": tail_primary,
            "head": head1,
            "tail": tail1,
            "target": target1,
            "steps": step_primary,
        },
        "secondary": {
            "box": secondary_box,
            "patch": patch_secondary,
            "direction": direction,
            "tail_length": tail_secondary,
            "head": head2,
            "tail": tail2,
            "target": target2,
            "steps": 0,
        },
        "obstacle_box": obstacle_box,
    }


def _decorate_with_decoys(instance: dict) -> bool:
    input_grid = _render_input(instance)
    expected = _render_expected(instance)
    current_boxes = [item["box"] for item in instance["statics"]] + [item["box"] for item in instance["containers"]]
    current_boxes.extend(_guide_extent(c["head"], c["tail"], c["target"]) for c in instance["containers"])

    extra_count = random.choice([1, 2])
    colors_left = [c for c in STATIC_COLORS if c not in {int(item["color"]) for item in instance["statics"]}]
    random.shuffle(colors_left)
    added = 0
    for _ in range(200):
        if added == extra_count or not colors_left:
            break
        box_h = random.randint(3, 6)
        box_w = random.randint(3, 7)
        box = _free_box(instance["shape"], box_h, box_w, current_boxes, tries=20)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if input_grid[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if expected[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        instance["statics"].append({"color": colors_left.pop(), "box": box})
        instance["static_draw_order"].append(instance["statics"][-1])
        current_boxes.append(box)
        added += 1
    return added >= 1


def _build_instance() -> dict | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    pair = _build_primary_pair(shape)
    if pair is None:
        return None

    primary_color = random.choice(STATIC_COLORS)
    instance = {
        "shape": shape,
        "statics": [{"color": primary_color, "box": pair["obstacle_box"]}],
        "static_draw_order": [{"color": primary_color, "box": pair["obstacle_box"]}],
        "containers": [pair["primary"], pair["secondary"]],
    }
    if not _decorate_with_decoys(instance):
        return None

    obstacle_mask = _restore_static_layer(shape, instance["statics"]) != BLACK
    primary = instance["containers"][0]
    secondary = instance["containers"][1]
    primary["steps"] = min(
        primary["tail_length"],
        int(np.sum(primary["patch"][1:-1, 1:-1] == GREY)),
        _max_shift_before_obstacle(primary["box"], primary["direction"], obstacle_mask),
    )
    secondary["steps"] = min(
        secondary["tail_length"],
        int(np.sum(secondary["patch"][1:-1, 1:-1] == GREY)),
        _max_shift_before_obstacle(secondary["box"], secondary["direction"], obstacle_mask),
    )
    return instance


def _validate_instance(instance: dict) -> None:
    inp = _render_input(instance)
    exp = _render_expected(instance)
    out = main(inp)
    assert np.array_equal(out, exp), "solver disagrees with generator metadata"

    primary = instance["containers"][0]
    restored = _restore_static_layer(instance["shape"], instance["statics"])
    obstacle_mask = restored != BLACK
    visible_mask = _static_mask_from_grid(inp)
    true_cap = _max_shift_before_obstacle(primary["box"], primary["direction"], obstacle_mask)
    visible_cap = _max_shift_before_obstacle(primary["box"], primary["direction"], visible_mask)
    grey_count = int(np.sum(primary["patch"][1:-1, 1:-1] == GREY))
    assert true_cap == int(primary["steps"])
    assert primary["tail_length"] > primary["steps"], "hidden obstacle must be tighter than tail"
    assert grey_count > primary["steps"], "hidden obstacle must be tighter than fuel"
    assert visible_cap > true_cap, "restoration must change the movement cap"

    secondary = instance["containers"][1]
    assert secondary["steps"] == 0, "occluding container should stay put"
    assert np.sum(inp == MAROON) > 0
    assert len(instance["statics"]) in (2, 3)


def generate_input() -> np.ndarray:
    for _ in range(400):
        instance = _build_instance()
        if instance is None:
            continue
        try:
            _validate_instance(instance)
        except AssertionError:
            continue
        return _render_input(instance)
    raise RuntimeError("failed to generate a valid puzzle instance")
