from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SHAPE_CHOICES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _sort_cells(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _decode_markers(grid: np.ndarray) -> tuple[dict[int, str], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= 3 and W >= 3
    assert int(g[0, 0]) == BG and int(g[0, W - 1]) == BG
    assert int(g[H - 1, 0]) == BG and int(g[H - 1, W - 1]) == BG

    directions: dict[int, str] = {}
    marker_cells: set[tuple[int, int]] = set()
    edge_specs = [
        ("up", [(0, c) for c in range(1, W - 1)]),
        ("down", [(H - 1, c) for c in range(1, W - 1)]),
        ("left", [(r, 0) for r in range(1, H - 1)]),
        ("right", [(r, W - 1) for r in range(1, H - 1)]),
    ]
    for direction, cells in edge_specs:
        for r, c in cells:
            color = int(g[r, c])
            if color == BG:
                continue
            assert color not in directions
            directions[color] = direction
            marker_cells.add((int(r), int(c)))

    assert directions
    return directions, marker_cells


def _extract_objects(
    grid: np.ndarray,
    color_dirs: dict[int, str],
    marker_cells: set[tuple[int, int]],
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []
    next_id = 0

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if seen[r, c] or (r, c) in marker_cells:
                continue
            color = int(g[r, c])
            if color == BG:
                continue
            assert color in color_dirs

            q = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or (nr, nc) in marker_cells:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))

            rows = [rr for rr, _cc in cells]
            cols = [cc for _rr, cc in cells]
            objects.append(
                {
                    "id": int(next_id),
                    "color": int(color),
                    "direction": color_dirs[int(color)],
                    "cells": _sort_cells(cells),
                    "input_cells": _sort_cells(cells),
                    "area": int(len(cells)),
                    "top_row": int(min(rows)),
                    "left_col": int(min(cols)),
                }
            )
            next_id += 1

    assert objects
    return objects


def _simulate_motion(objects: list[dict], H: int, W: int) -> None:
    max_steps = H + W
    for _ in range(max_steps):
        occupancy: dict[tuple[int, int], int] = {}
        for obj in objects:
            for cell in obj["cells"]:
                occupancy[cell] = int(obj["id"])

        proposals: dict[int, tuple[tuple[int, int], ...]] = {}
        for obj in objects:
            dr, dc = DIR_TO_VEC[obj["direction"]]
            shifted: list[tuple[int, int]] = []
            ok = True
            for r, c in obj["cells"]:
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    ok = False
                    break
                blocker = occupancy.get((rr, cc))
                if blocker is not None and blocker != int(obj["id"]):
                    ok = False
                    break
                shifted.append((int(rr), int(cc)))
            if ok:
                proposals[int(obj["id"])] = _sort_cells(shifted)

        blocked: set[int] = set()
        proposed_hits: dict[tuple[int, int], list[int]] = {}
        for oid, cells in proposals.items():
            for cell in cells:
                proposed_hits.setdefault(cell, []).append(int(oid))
        for ids in proposed_hits.values():
            if len(ids) > 1:
                blocked.update(map(int, ids))

        moved = False
        for obj in objects:
            oid = int(obj["id"])
            if oid in proposals and oid not in blocked:
                obj["cells"] = proposals[oid]
                moved = True
        if not moved:
            return


def _touching(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    bset = set(b)
    for r, c in a:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) in bset:
                return True
    return False


def _fusion_groups(objects: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    seen: set[int] = set()
    by_id = {int(obj["id"]): obj for obj in objects}

    for obj in objects:
        oid = int(obj["id"])
        if oid in seen:
            continue
        comp = []
        stack = [oid]
        seen.add(oid)
        while stack:
            cur = stack.pop()
            cur_obj = by_id[cur]
            comp.append(cur_obj)
            for other in objects:
                other_id = int(other["id"])
                if other_id in seen or other_id == cur:
                    continue
                if _touching(cur_obj["cells"], other["cells"]):
                    seen.add(other_id)
                    stack.append(other_id)
        groups.append(comp)
    return groups


def _winner_for_group(group: list[dict]) -> tuple[dict, str]:
    max_area = max(int(obj["area"]) for obj in group)
    largest = [obj for obj in group if int(obj["area"]) == max_area]
    if len(largest) == 1:
        return largest[0], "area"
    winner = min(
        largest,
        key=lambda obj: (int(obj["top_row"]), int(obj["left_col"]), int(obj["id"])),
    )
    return winner, "tie_top"


def _solve(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    color_dirs, marker_cells = _decode_markers(g)
    objects = _extract_objects(g, color_dirs, marker_cells)
    _simulate_motion(objects, H, W)

    groups = _fusion_groups(objects)
    out = np.full_like(g, BG)
    meta_groups = []
    for group in groups:
        winner, winner_type = _winner_for_group(group)
        fused = len(group) > 1
        paint_color = int(winner["color"])
        for obj in group:
            for r, c in obj["cells"]:
                out[int(r), int(c)] = paint_color
        meta_groups.append(
            {
                "ids": [int(obj["id"]) for obj in group],
                "colors": [int(obj["color"]) for obj in group],
                "areas": [int(obj["area"]) for obj in group],
                "top_rows": [int(obj["top_row"]) for obj in group],
                "fused": bool(fused),
                "winner_color": int(winner["color"]),
                "winner_type": winner_type if fused else "none",
            }
        )

    if not with_meta:
        return out

    meta = {
        "objects": [
            {
                "id": int(obj["id"]),
                "color": int(obj["color"]),
                "direction": obj["direction"],
                "area": int(obj["area"]),
                "top_row": int(obj["top_row"]),
                "left_col": int(obj["left_col"]),
                "input_cells": obj["input_cells"],
                "final_cells": obj["cells"],
            }
            for obj in objects
        ],
        "groups": meta_groups,
        "fusion_count": int(sum(1 for group in meta_groups if group["fused"])),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode border markers into per-color motion, slide all interior objects synchronously,
    then fuse any final touching groups using area-first color selection.
    """
    return _solve(grid, with_meta=False)


def _box_far_enough(
    box: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
    *,
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = box
    for br0, bc0, br1, bc1 in boxes:
        if not (r1 < br0 - gap or br1 < r0 - gap or c1 < bc0 - gap or bc1 < c0 - gap):
            return False
    return True


def _add_rect_spec(
    specs: list[dict],
    boxes: list[tuple[int, int, int, int]],
    *,
    color: int,
    direction: str,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    if not _box_far_enough(box, boxes, gap=1):
        return False
    boxes.append(box)
    specs.append(
        {
            "color": int(color),
            "direction": direction,
            "r0": int(r0),
            "c0": int(c0),
            "h": int(h),
            "w": int(w),
        }
    )
    return True


def _place_rect_in_region(
    specs: list[dict],
    boxes: list[tuple[int, int, int, int]],
    *,
    color: int,
    direction: str,
    H: int,
    W: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    shape: tuple[int, int] | None = None,
) -> bool:
    shapes = [shape] if shape is not None else list(SHAPE_CHOICES)
    random.shuffle(shapes)
    for h, w in shapes:
        r_lo = int(max(1, row_range[0]))
        r_hi = int(min(H - h - 1, row_range[1]))
        c_lo = int(max(1, col_range[0]))
        c_hi = int(min(W - w - 1, col_range[1]))
        if r_lo > r_hi or c_lo > c_hi:
            continue
        positions = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:40]:
            if _add_rect_spec(
                specs,
                boxes,
                color=int(color),
                direction=direction,
                r0=int(r0),
                c0=int(c0),
                h=int(h),
                w=int(w),
            ):
                return True
    return False


def _render_grid(H: int, W: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(spec["color"])

    by_color = {int(spec["color"]): spec["direction"] for spec in specs}
    edge_slots = {
        "up": [(0, c) for c in range(1, W - 1)],
        "down": [(H - 1, c) for c in range(1, W - 1)],
        "left": [(r, 0) for r in range(1, H - 1)],
        "right": [(r, W - 1) for r in range(1, H - 1)],
    }
    used: set[tuple[int, int]] = set()
    for color, direction in random.sample(list(by_color.items()), len(by_color)):
        slots = edge_slots[direction][:]
        random.shuffle(slots)
        placed = False
        for cell in slots:
            if cell in used:
                continue
            used.add(cell)
            grid[cell] = int(color)
            placed = True
            break
        assert placed
    return grid


def _build_no_fusion_scene(H: int, W: int, colors: list[int]) -> list[dict] | None:
    specs: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    directions = random.sample(list(DIR_TO_VEC), len(colors))
    zone_map = {
        "up": ((H // 2, H - 2), (W // 2, W - 2)),
        "down": ((1, max(1, H // 3)), (1, max(1, W // 3))),
        "left": ((1, max(1, H // 3)), (W // 2, W - 2)),
        "right": ((H // 2, H - 2), (1, max(1, W // 3))),
    }

    for color, direction in zip(colors, directions):
        row_range, col_range = zone_map[direction]
        if not _place_rect_in_region(
            specs,
            boxes,
            color=int(color),
            direction=direction,
            H=H,
            W=W,
            row_range=row_range,
            col_range=col_range,
        ):
            return None

    if random.random() < 0.35:
        dup_idx = random.randrange(len(colors))
        dup_color = int(colors[dup_idx])
        dup_direction = directions[dup_idx]
        row_range, col_range = zone_map[dup_direction]
        _place_rect_in_region(
            specs,
            boxes,
            color=dup_color,
            direction=dup_direction,
            H=H,
            W=W,
            row_range=row_range,
            col_range=col_range,
        )

    return specs


def _build_fusion_scene(H: int, W: int, colors: list[int], mode: str) -> list[dict] | None:
    specs: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    pair_dir = random.choice(["up", "down"] if mode == "tie" else list(DIR_TO_VEC))
    pair_colors = colors[:2]

    if pair_dir in {"up", "down"}:
        lane_w = random.choice([1, 2])
        height_choices = [1, 2, 3] if lane_w == 1 else [1, 2]
        if mode == "tie":
            h_front = h_back = random.choice([1, 2])
        else:
            pair_heights = random.sample(height_choices, 2)
            if random.random() < 0.5:
                h_front, h_back = pair_heights
            else:
                h_back, h_front = pair_heights
        lane_c_lo = 2
        lane_c_hi = W - lane_w - 3
        if lane_c_lo > lane_c_hi:
            return None
        c0 = random.randint(lane_c_lo, lane_c_hi)
        gap = random.randint(2, 4)
        if pair_dir == "up":
            upper_max = H - h_front - h_back - gap - 2
            if upper_max < 2:
                return None
            upper_r = random.randint(2, upper_max)
            lower_r = random.randint(upper_r + h_front + gap, H - h_back - 2)
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[0]),
                direction=pair_dir,
                r0=upper_r,
                c0=c0,
                h=h_front,
                w=lane_w,
            ):
                return None
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[1]),
                direction=pair_dir,
                r0=lower_r,
                c0=c0,
                h=h_back,
                w=lane_w,
            ):
                return None
        else:
            upper_max = H - h_front - h_back - gap - 2
            if upper_max < 2:
                return None
            upper_r = random.randint(2, upper_max)
            lower_r = random.randint(upper_r + h_back + gap, H - h_front - 2)
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[1]),
                direction=pair_dir,
                r0=upper_r,
                c0=c0,
                h=h_back,
                w=lane_w,
            ):
                return None
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[0]),
                direction=pair_dir,
                r0=lower_r,
                c0=c0,
                h=h_front,
                w=lane_w,
            ):
                return None
    else:
        lane_h = random.choice([1, 2])
        width_choices = [1, 2, 3] if lane_h == 1 else [1, 2]
        pair_widths = random.sample(width_choices, 2)
        if random.random() < 0.5:
            w_front, w_back = pair_widths
        else:
            w_back, w_front = pair_widths
        lane_r_lo = 2
        lane_r_hi = H - lane_h - 3
        if lane_r_lo > lane_r_hi:
            return None
        r0 = random.randint(lane_r_lo, lane_r_hi)
        gap = random.randint(2, 4)
        if pair_dir == "left":
            left_max = W - w_front - w_back - gap - 2
            if left_max < 2:
                return None
            left_c = random.randint(2, left_max)
            right_c = random.randint(left_c + w_front + gap, W - w_back - 2)
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[0]),
                direction=pair_dir,
                r0=r0,
                c0=left_c,
                h=lane_h,
                w=w_front,
            ):
                return None
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[1]),
                direction=pair_dir,
                r0=r0,
                c0=right_c,
                h=lane_h,
                w=w_back,
            ):
                return None
        else:
            left_max = W - w_front - w_back - gap - 2
            if left_max < 2:
                return None
            left_c = random.randint(2, left_max)
            right_c = random.randint(left_c + w_back + gap, W - w_front - 2)
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[1]),
                direction=pair_dir,
                r0=r0,
                c0=left_c,
                h=lane_h,
                w=w_back,
            ):
                return None
            if not _add_rect_spec(
                specs,
                boxes,
                color=int(pair_colors[0]),
                direction=pair_dir,
                r0=r0,
                c0=right_c,
                h=lane_h,
                w=w_front,
            ):
                return None

    if len(colors) >= 3:
        extra_dir_choices = [d for d in DIR_TO_VEC if d != pair_dir]
        extra_dir = random.choice(extra_dir_choices)
        zone_map = {
            "up": ((H // 2, H - 2), (1, max(1, W // 3))),
            "down": ((1, max(1, H // 3)), (W // 2, W - 2)),
            "left": ((H // 2, H - 2), (W // 2, W - 2)),
            "right": ((1, max(1, H // 3)), (1, max(1, W // 3))),
        }
        row_range, col_range = zone_map[extra_dir]
        _place_rect_in_region(
            specs,
            boxes,
            color=int(colors[2]),
            direction=extra_dir,
            H=H,
            W=W,
            row_range=row_range,
            col_range=col_range,
        )

    return specs


def _scene_for_mode(mode: str) -> np.ndarray | None:
    H = random.randint(10, 15)
    W = random.randint(10, 15)
    if mode == "none":
        color_count = random.randint(2, 4)
    else:
        color_count = random.randint(3, 4)
    colors = random.sample(range(1, 10), color_count)

    if mode == "none":
        specs = _build_no_fusion_scene(H, W, colors)
    else:
        specs = _build_fusion_scene(H, W, colors, mode)
    if not specs:
        return None

    grid = _render_grid(H, W, specs)
    out, meta = _solve(grid, with_meta=True)
    if not np.array_equal(out, main(grid)):
        return None
    if any(int(out[0, c]) != BG for c in range(W)):
        return None
    if any(int(out[H - 1, c]) != BG for c in range(W)):
        return None
    if any(int(out[r, 0]) != BG for r in range(H)):
        return None
    if any(int(out[r, W - 1]) != BG for r in range(H)):
        return None

    group_types = {group["winner_type"] for group in meta["groups"] if group["fused"]}
    if mode == "none":
        if int(meta["fusion_count"]) != 0:
            return None
    elif mode == "area":
        if "area" not in group_types:
            return None
    else:
        if "tie_top" not in group_types:
            return None

    distinct_dirs = {obj["direction"] for obj in meta["objects"]}
    if len(distinct_dirs) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Sample diverse scenes across three semantic branches:
    - no fusion,
    - fusion with unequal areas,
    - fusion with equal areas resolved by higher input position.
    """
    modes = ["none", "area", "tie"]
    mode = random.choice(modes)
    for _ in range(GEN_ATTEMPTS):
        grid = _scene_for_mode(mode)
        if grid is not None:
            return grid
    for fallback in modes:
        if fallback == mode:
            continue
        for _ in range(max(40, GEN_ATTEMPTS // 4)):
            grid = _scene_for_mode(fallback)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid input")
