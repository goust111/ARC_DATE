from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_CORNER = {
    "left": ("bottom", "right"),
    "right": ("bottom", "left"),
    "top": ("bottom", "right"),
    "bottom": ("top", "right"),
}
GEN_ATTEMPTS = 400
GEN_CANVAS_RANGE = (20, 30)
GEN_COLORS = [1, 2, 3, 4, 5]
GEN_COUNT_VALUES = [3, 4, 5]
GEN_COUNT_WEIGHTS = [1, 2, 2]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _outline_set(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    pts: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        pts.add((r0, cc))
        pts.add((r1, cc))
    for rr in range(r0, r1 + 1):
        pts.add((rr, c0))
        pts.add((rr, c1))
    return pts


def _decode_component(comp: dict) -> dict:
    coords = comp["coords"]
    cells = {tuple(map(int, rc)) for rc in coords}
    r0, c0, r1, c1 = comp["bbox"]
    candidates = []

    specs = [
        ("left", r0, c0 + 1, r1, c1),
        ("right", r0, c0, r1, c1 - 1),
        ("top", r0 + 1, c0, r1, c1),
        ("bottom", r0, c0, r1 - 1, c1),
    ]
    for side, fr0, fc0, fr1, fc1 in specs:
        fh = fr1 - fr0 + 1
        fw = fc1 - fc0 + 1
        if fh < 4 or fw < 4:
            continue
        if side in {"left", "right"}:
            tab = (fr0 + fh // 2, fc0 - 1 if side == "left" else fc1 + 1)
        else:
            tab = (fr0 - 1 if side == "top" else fr1 + 1, fc0 + fw // 2)
        expected = _outline_set(fr0, fc0, fh, fw)
        expected.add(tab)
        if cells == expected:
            candidates.append(
                {
                    "color": int(comp["color"]),
                    "side": side,
                    "height": int(fh),
                    "width": int(fw),
                    "area": int(fh * fw),
                }
            )

    assert len(candidates) == 1, f"expected exactly one valid tabbed frame interpretation, got {len(candidates)}"
    return candidates[0]


def _draw_outline(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    bottom = top + h - 1
    right = left + w - 1
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _nest_items(items: list[dict], *, corner_override: tuple[str, str] | None = None) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    outer = items[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)
    placements: list[tuple[int, int, int, int]] = []

    top = left = 0
    _draw_outline(out, top, left, int(outer["height"]), int(outer["width"]), int(outer["color"]))
    placements.append((top, left, int(outer["height"]), int(outer["width"])))

    parent_top, parent_left, parent_h, parent_w = placements[0]
    for item in items[1:]:
        region_top = parent_top + 1
        region_left = parent_left + 1
        region_h = parent_h - 2
        region_w = parent_w - 2
        child_h = int(item["height"])
        child_w = int(item["width"])
        assert child_h <= region_h and child_w <= region_w, "child frame must fit inside parent interior"

        vertical, horizontal = corner_override or SIDE_TO_CORNER[str(item["side"])]
        child_top = region_top if vertical == "top" else region_top + region_h - child_h
        child_left = region_left if horizontal == "left" else region_left + region_w - child_w

        _draw_outline(out, child_top, child_left, child_h, child_w, int(item["color"]))
        placements.append((child_top, child_left, child_h, child_w))
        parent_top, parent_left, parent_h, parent_w = placements[-1]

    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the tabbed rectangular outlines by size and nest them so each tab chooses the opposite corner.

    Input structure:
    - The grid has black background and 3-5 non-overlapping monochrome rectangular frame objects of different colors.
    - Each object is a 1-pixel-thick hollow rectangle plus exactly one same-colored tab pixel protruding outward from the midpoint of one bbox side.
    - The true rectangle bbox excludes the tab; rectangle sizes are all different.

    Rule:
    - Detect each tabbed frame, remove its tab logically, and record the tab side relative to the rectangle bbox.
    - Sort the frames by rectangle bbox area from largest to smallest.
    - Create a new black canvas with the largest rectangle size, place that largest frame at the top-left, then nest every next smaller frame inside the previous frame's interior.
    - The child frame's tab decides the opposite placement corner: left tab -> lower-right, right tab -> lower-left, top tab -> lower-right, bottom tab -> upper-right.

    Key insight:
    - The lone pixel outside each rectangle is not decoration: it simultaneously confirms which rectangle is being measured and encodes the corner used for the next nesting step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g, bg=BLACK)
    assert 3 <= len(comps) <= 5, "expected 3-5 tabbed frames"
    items = [_decode_component(comp) for comp in comps]

    areas = [int(item["area"]) for item in items]
    assert len(set(areas)) == len(areas), "frame bbox areas must be unique"
    items.sort(key=lambda item: int(item["area"]), reverse=True)

    out, _placements = _nest_items(items)
    return out


def _make_patch(h: int, w: int, side: str, color: int) -> np.ndarray:
    if side == "left":
        patch = np.zeros((h, w + 1), dtype=int)
        _draw_outline(patch, 0, 1, h, w, color)
        patch[h // 2, 0] = color
        return patch
    if side == "right":
        patch = np.zeros((h, w + 1), dtype=int)
        _draw_outline(patch, 0, 0, h, w, color)
        patch[h // 2, w] = color
        return patch
    if side == "top":
        patch = np.zeros((h + 1, w), dtype=int)
        _draw_outline(patch, 1, 0, h, w, color)
        patch[0, w // 2] = color
        return patch
    patch = np.zeros((h + 1, w), dtype=int)
    _draw_outline(patch, 0, 0, h, w, color)
    patch[h, w // 2] = color
    return patch


def _random_side_list(count: int) -> list[str]:
    largest = random.choice(list(SIDE_TO_CORNER))
    if count >= 4:
        required = ["right", "bottom", random.choice(["left", "top"])]
        extras = [random.choice(list(SIDE_TO_CORNER)) for _ in range(count - 4)]
        inner = required + extras
    else:
        inner = ["right", random.choice(["left", "top", "bottom"])]
    random.shuffle(inner)
    return [largest, *inner]


def _adjust_for_side(value: int, side: str, axis: str) -> int:
    if side in {"left", "right"} and axis == "h" and value % 2 == 0:
        return value + 1
    if side in {"top", "bottom"} and axis == "w" and value % 2 == 0:
        return value + 1
    return value


def _build_sizes(sides: list[str]) -> list[tuple[int, int]]:
    count = len(sides)
    outer_cap = {3: 20, 4: 19, 5: 18}[count]
    outer_min = max(10, 4 + 3 * (count - 1))

    for _ in range(300):
        outer_side = sides[0]
        h_candidates = [v for v in range(outer_min, outer_cap + 1) if _adjust_for_side(v, outer_side, "h") == v]
        w_candidates = [v for v in range(outer_min, outer_cap + 1) if _adjust_for_side(v, outer_side, "w") == v]
        if not h_candidates or not w_candidates:
            continue
        sizes = [(random.choice(h_candidates), random.choice(w_candidates))]
        ok = True
        for idx in range(1, count):
            side = sides[idx]
            parent_h, parent_w = sizes[-1]
            remaining_inner = count - idx - 1
            min_h = 4 + 3 * remaining_inner
            min_w = 4 + 3 * remaining_inner
            h_candidates = [
                v
                for v in range(min_h, parent_h - 2)
                if _adjust_for_side(v, side, "h") == v
            ]
            w_candidates = [
                v
                for v in range(min_w, parent_w - 2)
                if _adjust_for_side(v, side, "w") == v
            ]
            if not h_candidates or not w_candidates:
                ok = False
                break
            child_h = random.choice(h_candidates)
            child_w = random.choice(w_candidates)
            sizes.append((child_h, child_w))

        if not ok:
            continue

        if sizes[0][0] < 10 or sizes[0][1] < 10:
            continue
        if sizes[0][0] > outer_cap or sizes[0][1] > outer_cap:
            continue
        if any(sizes[i][0] >= sizes[i - 1][0] - 2 or sizes[i][1] >= sizes[i - 1][1] - 2 for i in range(1, count)):
            continue
        return sizes

    raise RuntimeError("failed to build a valid size chain")


def _place_patches(specs: list[dict]) -> np.ndarray | None:
    max_h = max(int(spec["patch"].shape[0]) for spec in specs)
    max_w = max(int(spec["patch"].shape[1]) for spec in specs)
    min_dim = max(GEN_CANVAS_RANGE[0], max(max_h, max_w) + 1)

    for _ in range(250):
        h = random.randint(min_dim, GEN_CANVAS_RANGE[1])
        w = random.randint(min_dim, GEN_CANVAS_RANGE[1])
        if h > 30 or w > 30:
            continue

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int]] = [(-1, -1)] * len(specs)
        order = sorted(range(len(specs)), key=lambda i: int(specs[i]["patch"].size), reverse=True)
        ok = True

        for idx in order:
            patch = specs[idx]["patch"]
            ph, pw = patch.shape
            candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(candidates)
            placed = False
            for top, left in candidates:
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(h, top + ph + 1)
                c1 = min(w, left + pw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                mask = patch != 0
                grid[top : top + ph, left : left + pw][mask] = patch[mask]
                blocked[r0:r1, c0:c1] = True
                placements[idx] = (top, left)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        scan_order = sorted(range(len(specs)), key=lambda i: placements[i])
        area_order = sorted(range(len(specs)), key=lambda i: int(specs[i]["height"] * specs[i]["width"]), reverse=True)
        if scan_order == area_order:
            continue
        return grid

    return None


def _color_order_is_suspicious(specs: list[dict]) -> bool:
    color_order = [int(spec["color"]) for spec in sorted(specs, key=lambda s: int(s["height"] * s["width"]), reverse=True)]
    return color_order == sorted(color_order) or color_order == sorted(color_order, reverse=True)


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        comps = _find_components(grid, bg=BLACK)
        items = [_decode_component(comp) for comp in comps]
        items.sort(key=lambda item: int(item["area"]), reverse=True)
        out, placements = _nest_items(items)
    except AssertionError:
        return False

    if len(items) < 3 or len({int(item["area"]) for item in items}) != len(items):
        return False

    for i in range(1, len(items)):
        parent = placements[i - 1]
        child = placements[i]
        slack_h = parent[2] - 2 - child[2]
        slack_w = parent[3] - 2 - child[3]
        if slack_h < 1 or slack_w < 1:
            return False

    scan_items = [_decode_component(comp) for comp in sorted(comps, key=lambda comp: tuple(map(int, comp["bbox"][:2])))]
    scan_areas = [int(item["area"]) for item in scan_items]
    if scan_areas == sorted(scan_areas, reverse=True):
        return False
    try:
        scan_out, _ = _nest_items(scan_items)
        if np.array_equal(out, scan_out):
            return False
    except AssertionError:
        pass

    always_br, _ = _nest_items(items, corner_override=("bottom", "right"))
    always_bl, _ = _nest_items(items, corner_override=("bottom", "left"))
    always_tr, _ = _nest_items(items, corner_override=("top", "right"))
    if np.array_equal(out, always_br) or np.array_equal(out, always_bl) or np.array_equal(out, always_tr):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        count = random.choices(GEN_COUNT_VALUES, weights=GEN_COUNT_WEIGHTS, k=1)[0]
        sides = _random_side_list(count)
        try:
            sizes = _build_sizes(sides)
        except RuntimeError:
            continue

        colors = random.sample(GEN_COLORS, count)
        specs = []
        for (h, w), side, color in zip(sizes, sides, colors):
            specs.append(
                {
                    "height": int(h),
                    "width": int(w),
                    "side": side,
                    "color": int(color),
                    "patch": _make_patch(int(h), int(w), side, int(color)),
                }
            )

        if _color_order_is_suspicious(specs):
            continue

        grid = _place_patches(specs)
        if grid is None:
            continue
        if not _validate_semantics(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
