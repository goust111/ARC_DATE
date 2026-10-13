from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_AREA_MIN = 5
_AREA_MAX = 30
_OUTER_MAX_DIM = 16
_GEN_ATTEMPTS = 600
_SHAPE_ATTEMPTS = 80
_BBOX_ATTEMPTS = 80
_GRID_H_RANGE = (13, 22)
_GRID_W_RANGE = (13, 22)
_LAYER_COUNTS = (3, 4)
_INNER_DIMS = (3, 4, 5, 6)
_PAD_CHOICES = (2, 3)


def _neighbors4(r: int, c: int):
    for dr, dc in _DIRS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (int(r0 + r1) // 2, int(c0 + c1) // 2)


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_components(grid: np.ndarray, bg: int = _BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _ordered_nested_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _find_components(grid, bg=_BG)
    assert comps, "expected at least one foreground component"

    ordered = sorted(
        comps,
        key=lambda comp: (_bbox_area(comp["bbox"]), int(comp["area"])),
        reverse=True,
    )
    bbox_areas = [_bbox_area(comp["bbox"]) for comp in ordered]
    assert len(set(bbox_areas)) == len(bbox_areas), "bbox areas must define a unique nesting order"
    for i in range(len(ordered) - 1):
        assert _strict_contains(ordered[i]["bbox"], ordered[i + 1]["bbox"]), "bboxes must be strictly nested"
    return ordered


def _expected_output(shape: tuple[int, int], inner_box: tuple[int, int, int, int], colors: list[int]) -> np.ndarray:
    center_r, center_c = _bbox_center(inner_box)
    ring_colors = np.array(list(reversed(colors)), dtype=int)

    rows = np.abs(np.arange(shape[0], dtype=int)[:, None] - center_r)
    cols = np.abs(np.arange(shape[1], dtype=int)[None, :] - center_c)
    dist = np.maximum(rows, cols)
    return ring_colors[dist % len(ring_colors)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the nested color hierarchy into full-grid Chebyshev rings centered on the innermost bbox.

    Input structure:
    - The grid has a black background and several non-black monochrome 4-connected components.
    - The components are strictly nested by bounding box: each larger bbox fully contains the next smaller one.
    - Only the component colors and the innermost component's bbox center matter; the detailed contours are distractors.

    Rule:
    - Detect every non-black 4-connected component and sort them by bbox area from largest to smallest.
    - Take the center pixel of the smallest component's bbox, using the upper/left middle cell when the bbox size is even.
    - Reverse the nesting colors so the innermost color comes first, then fill the whole grid by Chebyshev distance rings around that center.
    - A pixel at distance d gets color ring_colors[d mod n], where n is the number of nested layers.

    Key insight:
    - The input shapes are a code: nesting order determines the cyclic color sequence, and the innermost bbox position determines the ring origin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered = _ordered_nested_components(g)
    colors = [int(comp["color"]) for comp in ordered]
    return _expected_output(g.shape, ordered[-1]["bbox"], colors)


def _relative_box(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return (
        int(inner[0] - outer[0]),
        int(inner[1] - outer[1]),
        int(inner[2] - outer[0]),
        int(inner[3] - outer[1]),
    )


def _expanded_forbidden_mask(
    shape: tuple[int, int], inner_box: tuple[int, int, int, int] | None
) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    if inner_box is None:
        return forbidden

    r0, c0, r1, c1 = inner_box
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return forbidden


def _border_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    if side == "right":
        return [(r, w - 1) for r in range(h) if mask[r, w - 1]]
    raise ValueError(f"unknown side: {side}")


def _pick_anchor(candidates: list[tuple[int, int]], hub: tuple[int, int], axis: int) -> tuple[int, int]:
    pivot = hub[axis]
    best = min(abs(cell[axis] - pivot) for cell in candidates)
    nearest = [cell for cell in candidates if abs(cell[axis] - pivot) == best]
    return random.choice(nearest)


def _shortest_path(
    allowed: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]

    dirs = list(_DIRS4)
    random.shuffle(dirs)
    q = deque([start])
    prev = {start: None}

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]):
                continue
            if not allowed[nr, nc] or nxt in prev:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if goal not in prev:
        return None

    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _frontier(shape_mask: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    frontier: list[tuple[int, int]] = []
    h, w = shape_mask.shape
    for r in range(h):
        for c in range(w):
            if shape_mask[r, c] or not allowed[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and shape_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                frontier.append((r, c))
    return frontier


def _mask_irregular(mask: np.ndarray) -> bool:
    if mask.all():
        return False
    perim = np.zeros_like(mask, dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    return not np.array_equal(mask, perim)


def _generate_shape_mask(
    shape: tuple[int, int], inner_box: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    h, w = shape
    forbidden = _expanded_forbidden_mask(shape, inner_box)
    allowed = ~forbidden
    side_cells = {side: _border_cells(allowed, side) for side in ("top", "bottom", "left", "right")}
    if any(not cells for cells in side_cells.values()):
        return None

    candidates = [(r, c) for r in range(h) for c in range(w) if allowed[r, c]]
    inner_candidates = [(r, c) for r, c in candidates if 0 < r < h - 1 and 0 < c < w - 1]
    hubs = inner_candidates or candidates

    for _ in range(_SHAPE_ATTEMPTS):
        hub = random.choice(hubs)
        anchors = {
            "top": _pick_anchor(side_cells["top"], hub, 1),
            "bottom": _pick_anchor(side_cells["bottom"], hub, 1),
            "left": _pick_anchor(side_cells["left"], hub, 0),
            "right": _pick_anchor(side_cells["right"], hub, 0),
        }

        mask = np.zeros((h, w), dtype=bool)
        mask[hub] = True
        ok = True
        for anchor in anchors.values():
            path = _shortest_path(allowed, hub, anchor)
            if path is None:
                ok = False
                break
            for r, c in path:
                mask[r, c] = True
        if not ok:
            continue

        base_area = int(mask.sum())
        if base_area > _AREA_MAX:
            continue

        target_hi = min(_AREA_MAX, int(allowed.sum()), base_area + 6)
        target_lo = max(_AREA_MIN, base_area)
        if target_lo > target_hi:
            continue
        target_area = random.randint(target_lo, target_hi)

        while int(mask.sum()) < target_area:
            frontier = _frontier(mask, allowed)
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        if int(mask.sum()) < _AREA_MIN or int(mask.sum()) > _AREA_MAX:
            continue
        if not _mask_irregular(mask):
            continue
        return mask

    return None


def _sample_local_bboxes(n_layers: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(_BBOX_ATTEMPTS):
        if n_layers == 4:
            inner_choices = (3, 4)
            pad_choices = (2,)
        else:
            inner_choices = _INNER_DIMS
            pad_choices = _PAD_CHOICES

        inner_h = random.choice(inner_choices)
        inner_w = random.choice(inner_choices)

        pads: list[tuple[int, int, int, int]] = []
        cur_h, cur_w = inner_h, inner_w
        for _ in range(n_layers - 1):
            top = random.choice(pad_choices)
            bottom = random.choice(pad_choices)
            left = random.choice(pad_choices)
            right = random.choice(pad_choices)
            pads.append((top, bottom, left, right))
            cur_h += top + bottom
            cur_w += left + right

        if cur_h > _OUTER_MAX_DIM or cur_w > _OUTER_MAX_DIM:
            continue

        box = (0, 0, cur_h - 1, cur_w - 1)
        boxes = [box]
        for top, bottom, left, right in reversed(pads):
            r0, c0, r1, c1 = box
            box = (r0 + top, c0 + left, r1 - bottom, c1 - right)
            boxes.append(box)
        return boxes
    return None


def _globalize_bboxes(
    local_boxes: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    outer_h = local_boxes[0][2] + 1
    outer_w = local_boxes[0][3] + 1

    h = random.randint(max(_GRID_H_RANGE[0], outer_h + 2), max(_GRID_H_RANGE[0], outer_h + 6))
    w = random.randint(max(_GRID_W_RANGE[0], outer_w + 2), max(_GRID_W_RANGE[0], outer_w + 6))
    if h > 30 or w > 30:
        return None

    top = random.randint(1, h - outer_h - 1)
    left = random.randint(1, w - outer_w - 1)
    global_boxes = [(r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in local_boxes]

    center_r, center_c = _bbox_center(global_boxes[-1])
    grid_center = ((h - 1) // 2, (w - 1) // 2)
    if abs(center_r - grid_center[0]) + abs(center_c - grid_center[1]) < 2:
        return None
    return (h, w), global_boxes


def _paint_mask(
    grid: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, color: int
) -> bool:
    r0, c0, r1, c1 = box
    view = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(view[mask] != _BG):
        return False
    view[mask] = int(color)
    return True


def _shapes_are_separated(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == _BG:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(grid[nr, nc])
                    if other != _BG and other != color:
                        return False
    return True


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    comps = _ordered_nested_components(grid)
    if len(comps) != len(boxes):
        return False
    if [comp["bbox"] for comp in comps] != boxes:
        return False
    if [int(comp["color"]) for comp in comps] != colors:
        return False
    if not _shapes_are_separated(grid):
        return False

    for i, comp in enumerate(comps):
        if not (_AREA_MIN <= int(comp["area"]) <= _AREA_MAX):
            return False
        box = comp["bbox"]
        h = box[2] - box[0] + 1
        w = box[3] - box[1] + 1
        local = np.zeros((h, w), dtype=bool)
        coords = comp["coords"]
        local[coords[:, 0] - box[0], coords[:, 1] - box[1]] = True
        if not _mask_irregular(local):
            return False
        if i < len(comps) - 1 and not _strict_contains(box, comps[i + 1]["bbox"]):
            return False

    center_r, center_c = _bbox_center(boxes[-1])
    grid_center = ((grid.shape[0] - 1) // 2, (grid.shape[1] - 1) // 2)
    if abs(center_r - grid_center[0]) + abs(center_c - grid_center[1]) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.choice(_LAYER_COUNTS)
    local_boxes = _sample_local_bboxes(n_layers)
    if local_boxes is None:
        return None

    globalized = _globalize_bboxes(local_boxes)
    if globalized is None:
        return None
    (h, w), global_boxes = globalized

    colors = random.sample(_PALETTE, n_layers)
    grid = np.full((h, w), _BG, dtype=int)

    for i, box in enumerate(global_boxes):
        local_h = box[2] - box[0] + 1
        local_w = box[3] - box[1] + 1
        next_box = None
        if i + 1 < len(global_boxes):
            next_box = _relative_box(box, global_boxes[i + 1])
        mask = _generate_shape_mask((local_h, local_w), next_box)
        if mask is None or not _paint_mask(grid, box, mask, colors[i]):
            return None

    expected = _expected_output(grid.shape, global_boxes[-1], colors)
    return grid if _validate_case(grid, expected, global_boxes, colors) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring ARC input")
