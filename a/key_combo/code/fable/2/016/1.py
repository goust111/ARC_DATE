from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
OTHER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_SIDE = 30
GEN_ATTEMPTS = 500

TEMPLATES = (
    {"name": "line2", "children": {0: [1], 1: [2]}, "min_depth": 2, "min_branch": 1, "min_moved": 3, "branching": False},
    {"name": "line3", "children": {0: [1], 1: [2], 2: [3]}, "min_depth": 3, "min_branch": 1, "min_moved": 4, "branching": False},
    {"name": "branch2", "children": {0: [1, 2]}, "min_depth": 1, "min_branch": 2, "min_moved": 3, "branching": True},
    {"name": "branch_deep", "children": {0: [1, 2], 1: [3]}, "min_depth": 2, "min_branch": 2, "min_moved": 4, "branching": True},
    {"name": "fan3", "children": {0: [1, 2, 3]}, "min_depth": 1, "min_branch": 3, "min_moved": 4, "branching": True},
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _make_rect(color: int | None, h: int, w: int, r0: int, c0: int, *, tag: int | None = None) -> dict:
    return {
        "color": None if color is None else int(color),
        "h": int(h),
        "w": int(w),
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "tag": tag,
    }


def _shift_rect(rect: dict, dr: int, dc: int) -> dict:
    r0, c0, r1, c1 = rect["bbox"]
    out = dict(rect)
    out["bbox"] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    return out


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _overlap_1d(a[0], a[2], b[0], b[2]) and _overlap_1d(a[1], a[3], b[1], b[3])


def _bbox_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return _overlap_1d(a[0] - margin, a[2] + margin, b[0], b[2]) and _overlap_1d(
        a[1] - margin, a[3] + margin, b[1], b[3]
    )


def _shifted_bbox(rect: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    h = int(rect["h"])
    return r0 - h, c0, r1 - h, c1


def _fill_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            coords = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            bbox = (int(r0), int(c0), int(r1), int(c1))
            rect_h = _bbox_h(bbox)
            rect_w = _bbox_w(bbox)
            area = int(coords_arr.shape[0])
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert area == rect_h * rect_w, "foreground objects must be solid rectangles"
            assert np.all(patch == color), "each rectangle must be monochromatic"

            rects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox,
                    "h": rect_h,
                    "w": rect_w,
                    "area": area,
                }
            )

    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1], rect["color"]))
    for idx, rect in enumerate(rects):
        rect["id"] = idx
    assert rects, "expected at least one foreground rectangle"
    return rects


def _build_dependency(rects: list[dict]) -> dict[int, list[int]]:
    adjacency = {idx: [] for idx in range(len(rects))}
    for src, rect in enumerate(rects):
        dst_bbox = _shifted_bbox(rect)
        for dst, other in enumerate(rects):
            if src == dst:
                continue
            if _bbox_overlap(dst_bbox, other["bbox"]):
                adjacency[src].append(dst)
    return adjacency


def _reverse_edges(adjacency: dict[int, list[int]]) -> dict[int, list[int]]:
    rev = {idx: [] for idx in adjacency}
    for src, children in adjacency.items():
        for child in children:
            rev[child].append(src)
    return rev


def _closure_from_red(rects: list[dict], adjacency: dict[int, list[int]]) -> tuple[int, set[int], dict[int, int]]:
    red_ids = [idx for idx, rect in enumerate(rects) if int(rect["color"]) == RED]
    assert len(red_ids) == 1, "expected exactly one red trigger rectangle"
    red_id = red_ids[0]

    moved = {red_id}
    q = deque([red_id])
    while q:
        cur = q.popleft()
        for nxt in adjacency[cur]:
            if nxt not in moved:
                moved.add(nxt)
                q.append(nxt)

    depth = {idx: -10**9 for idx in moved}
    depth[red_id] = 0
    for idx in sorted(moved, key=lambda item: rects[item]["bbox"][0], reverse=True):
        for nxt in adjacency[idx]:
            if nxt in moved:
                depth[nxt] = max(depth[nxt], depth[idx] + 1)
    return red_id, moved, depth


def _validate_chain(rects: list[dict], adjacency: dict[int, list[int]], red_id: int, moved: set[int]) -> None:
    dests = []
    for idx in sorted(moved):
        dst_bbox = _shifted_bbox(rects[idx])
        assert dst_bbox[0] >= 0, "moved rectangle must stay inside the top boundary"
        dests.append((idx, dst_bbox))

    for i in range(len(dests)):
        for j in range(i + 1, len(dests)):
            assert not _bbox_overlap(dests[i][1], dests[j][1]), "final moved rectangles must not overlap"

    rev = _reverse_edges(adjacency)
    for idx in moved:
        if idx == red_id:
            continue
        assert any(parent in moved for parent in rev[idx]), "every moved non-trigger must be pressed by a moved rectangle"

    for idx in moved:
        for nxt in adjacency[idx]:
            assert nxt in moved, "reachable collision chain must be closed"


def _render_output(shape: tuple[int, int], rects: list[dict], moved: set[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx, rect in enumerate(rects):
        bbox = _shifted_bbox(rect) if idx in moved else rect["bbox"]
        _fill_bbox(out, bbox, int(rect["color"]))
    return out


def _analyze_scene(grid: np.ndarray) -> dict:
    rects = _extract_rectangles(grid)
    adjacency = _build_dependency(rects)
    red_id, moved, depth = _closure_from_red(rects, adjacency)
    _validate_chain(rects, adjacency, red_id, moved)

    outdegree = {idx: sum(child in moved for child in adjacency[idx]) for idx in moved}
    return {
        "rects": rects,
        "adjacency": adjacency,
        "red_id": red_id,
        "moved": moved,
        "depth": depth,
        "unmoved": set(range(len(rects))) - moved,
        "max_depth": max(depth.values()),
        "max_branch": max(outdegree.values(), default=0),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move the red rectangle upward, then propagate upward jumps through every rectangle its landing area presses.

    Input structure:
    - The background is black.
    - Every foreground object is a solid monochromatic axis-aligned rectangle.
    - Exactly one rectangle is red, and enough space exists above any triggered chain.

    Rule:
    - Move the red rectangle straight upward by its own height.
    - Whenever a moved rectangle's landing box overlaps another rectangle's current box, that pressed rectangle also moves upward by its own height.
    - Continue this collision propagation until no new overlaps remain, then clear all moved originals and keep untouched rectangles fixed.

    Key insight:
    - The chain is determined by causal bbox collisions after each jump, and each rectangle uses its own height as the jump distance rather than inheriting the red rectangle's distance.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_scene(g)
    return _render_output(g.shape, info["rects"], info["moved"])


def _rand_comp(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _sample_spec(template: dict, node_id: int) -> dict:
    children = [_sample_spec(template, child_id) for child_id in template["children"].get(node_id, [])]
    if children:
        h = random.randint(2, 4)
    else:
        h = random.randint(1, 4)

    if not children:
        w = random.randint(1, 4)
    elif len(children) == 1:
        child_w = children[0]["w"]
        w = random.randint(child_w, min(8, child_w + 3))
    else:
        min_w = sum(child["w"] for child in children) + (len(children) - 1)
        if min_w > 10:
            raise ValueError("template too wide for ARC canvas")
        w = random.randint(min_w, min(10, min_w + 3))

    return {"id": node_id, "h": h, "w": w, "children": children}


def _pick_gap(parent_h: int, child: dict) -> int:
    if parent_h <= 1:
        return 0
    low = max(0, parent_h - 2) if child["children"] else 0
    return random.randint(low, parent_h - 1)


def _place_spec(node: dict, r0: int, c0: int, out: list[dict]) -> None:
    out.append(_make_rect(None, node["h"], node["w"], r0, c0, tag=node["id"]))
    children = list(node["children"])
    random.shuffle(children)
    if not children:
        return

    if len(children) == 1:
        child = children[0]
        slack = node["w"] - child["w"]
        child_c0 = c0 + random.randint(0, slack)
        child_r0 = r0 - _pick_gap(node["h"], child) - child["h"]
        _place_spec(child, child_r0, child_c0, out)
        return

    total_child_w = sum(child["w"] for child in children)
    slack = node["w"] - total_child_w - (len(children) - 1)
    extras = _rand_comp(slack, len(children) + 1)
    cursor = c0 + extras[0]
    for idx, child in enumerate(children):
        child_r0 = r0 - _pick_gap(node["h"], child) - child["h"]
        _place_spec(child, child_r0, cursor, out)
        cursor += child["w"]
        if idx + 1 < len(children):
            cursor += 1 + extras[idx + 1]


def _normalize_moving_rects(rects: list[dict]) -> list[dict]:
    min_r = min(min(rect["bbox"][0], _shifted_bbox(rect)[0]) for rect in rects)
    min_c = min(rect["bbox"][1] for rect in rects)
    row_shift = random.randint(0, 3) - min_r
    col_shift = random.randint(0, 3) - min_c
    return [_shift_rect(rect, row_shift, col_shift) for rect in rects]


def _choose_shape(rects: list[dict]) -> tuple[int, int] | None:
    need_h = max(rect["bbox"][2] for rect in rects) + 1
    need_w = max(rect["bbox"][3] for rect in rects) + 1
    if need_h > 24 or need_w > 24:
        return None
    extra_h = random.randint(3, min(8, MAX_SIDE - need_h))
    extra_w = random.randint(3, min(8, MAX_SIDE - need_w))
    return need_h + extra_h, need_w + extra_w


def _place_static_rects(moving_rects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    count = 1 if random.random() < 0.35 else 2
    rects = list(moving_rects)
    moving_dests = [_shifted_bbox(rect) for rect in moving_rects]
    statics: list[dict] = []

    for idx in range(count):
        placed = False
        for _ in range(160):
            rect_h = random.randint(1, 4)
            rect_w = random.randint(1, 5)
            if rect_h > h or rect_w > w:
                continue
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            candidate = _make_rect(None, rect_h, rect_w, r0, c0, tag=100 + idx)
            bbox = candidate["bbox"]
            if any(_bbox_conflict(bbox, rect["bbox"], margin=1) for rect in rects):
                continue
            if any(_bbox_conflict(bbox, dst, margin=1) for dst in moving_dests):
                continue
            statics.append(candidate)
            rects.append(candidate)
            placed = True
            break
        if not placed:
            return None

    return statics


def _assign_colors(moving_rects: list[dict], static_rects: list[dict]) -> list[dict]:
    all_rects = [dict(rect) for rect in moving_rects + static_rects]
    palette = random.sample(list(OTHER_COLORS), len(all_rects) - 1)
    all_rects[0]["color"] = RED
    for rect, color in zip(all_rects[1:], palette):
        rect["color"] = int(color)
    return all_rects


def _paint_input(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _fill_bbox(grid, rect["bbox"], int(rect["color"]))
    return grid


def _mode_matches(info: dict, template: dict) -> bool:
    if len(info["moved"]) < template["min_moved"]:
        return False
    if len(info["unmoved"]) < 1:
        return False
    if info["max_depth"] < template["min_depth"]:
        return False
    if template["branching"]:
        return info["max_branch"] >= template["min_branch"]
    return info["max_branch"] == template["min_branch"]


def _gen_attempt(template: dict) -> np.ndarray | None:
    try:
        spec = _sample_spec(template, 0)
    except ValueError:
        return None

    moving_rects: list[dict] = []
    _place_spec(spec, 0, 0, moving_rects)
    moving_rects = _normalize_moving_rects(moving_rects)

    shape = _choose_shape(moving_rects)
    if shape is None:
        return None

    static_rects = _place_static_rects(moving_rects, shape)
    if static_rects is None:
        return None

    rects = _assign_colors(moving_rects, static_rects)
    grid = _paint_input(shape, rects)

    try:
        info = _analyze_scene(grid)
    except AssertionError:
        return None

    if not _mode_matches(info, template):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if out.shape != grid.shape or max(out.shape) > MAX_SIDE:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template = random.choice(TEMPLATES)
        grid = _gen_attempt(template)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-reaction rectangle scene")
