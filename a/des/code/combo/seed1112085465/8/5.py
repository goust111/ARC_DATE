from __future__ import annotations

from collections import deque
import random

import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SIDES = ("top", "right", "bottom", "left")
_OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _single_color(arr: np.ndarray) -> int:
    vals = np.unique(arr)
    assert len(vals) == 1, "expected a solid-colored frame side"
    return int(vals[0])


def _frame_info(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0, "expected a non-empty frame"

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "expected a sizeable rectangular frame"

    assert int(grid[r0, c0]) == bg
    assert int(grid[r0, c1]) == bg
    assert int(grid[r1, c0]) == bg
    assert int(grid[r1, c1]) == bg

    side_colors = {
        "top": _single_color(grid[r0, c0 + 1 : c1]),
        "right": _single_color(grid[r0 + 1 : r1, c1]),
        "bottom": _single_color(grid[r1, c0 + 1 : c1]),
        "left": _single_color(grid[r0 + 1 : r1, c0]),
    }
    assert len(set(side_colors.values())) == 4, "expected four distinct frame-side colors"
    return (r0, c0, r1, c1), side_colors


def _touching_sides(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[str]:
    h, w = shape
    sides = set()
    for r, c in cells:
        if r == 0:
            sides.add("top")
        if r == h - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == w - 1:
            sides.add("right")
    return sides


def _adjacent_to_object(cells: list[tuple[int, int]], obj_mask: np.ndarray) -> bool:
    h, w = obj_mask.shape
    for r, c in cells:
        for dr, dc in _DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and obj_mask[nr, nc]:
                return True
    return False


def _notch_info(obj_mask: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    bg_comps = _components(~obj_mask)
    candidates: list[tuple[list[tuple[int, int]], str]] = []
    for comp in bg_comps:
        if not _adjacent_to_object(comp, obj_mask):
            continue
        touched = _touching_sides(comp, obj_mask.shape)
        if len(touched) != 1:
            continue
        candidates.append((comp, next(iter(touched))))

    assert len(candidates) == 1, "expected one single-sided concavity per object"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's concavity with the color from the opposite frame side.

    Input structure:
    - Black background.
    - One large rectangular frame with four differently colored sides and black corners.
    - Several grey 4-connected concave objects inside the frame.

    Rule:
    - For each object, find the connected background region inside its bbox that opens
      toward exactly one cardinal direction.
    - Use the color from the opposite frame side to fill that concavity.
    - Leave the frame and the object body unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    (fr0, fc0, fr1, fc1), side_colors = _frame_info(g, bg)

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = g[fr0 + 1 : fr1, fc0 + 1 : fc1] != bg
    objects = _components(interior_mask)
    assert objects, "expected at least one interior object"

    out = g.copy()
    for comp in objects:
        r0, c0, r1, c1 = _bbox(comp)
        obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            obj_mask[r - r0, c - c0] = True

        notch_cells, direction = _notch_info(obj_mask)
        fill_color = side_colors[_OPPOSITE[direction]]
        for rr, cc in notch_cells:
            out[r0 + rr, c0 + cc] = fill_color

    return out


# --------------- generator ---------------

_BG = 0
_FRAME_COLORS = {
    "top": 2,
    "right": 1,
    "bottom": 3,
    "left": 4,
}
_OBJECT_COLOR = 5

_INTERIOR_H_RANGE = (14, 20)
_INTERIOR_W_RANGE = (14, 20)
_OUTER_MARGIN_RANGE = (1, 3)
_OBJECT_COUNT_RANGE = (3, 5)
_OBJECT_H_RANGE = (5, 8)
_OBJECT_W_RANGE = (5, 8)
_FRAME_GAP = 1
_OBJECT_GAP = 1
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 400


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 + 1 : c1] = _FRAME_COLORS["top"]
    grid[r1, c0 + 1 : c1] = _FRAME_COLORS["bottom"]
    grid[r0 + 1 : r1, c0] = _FRAME_COLORS["left"]
    grid[r0 + 1 : r1, c1] = _FRAME_COLORS["right"]


def _sample_directions(count: int) -> list[str]:
    dirs = random.sample(_SIDES, k=min(len(_SIDES), count))
    while len(dirs) < count:
        dirs.append(random.choice(_SIDES))
    random.shuffle(dirs)
    return dirs


def _sample_object(direction: str) -> np.ndarray:
    h = random.randint(*_OBJECT_H_RANGE)
    w = random.randint(*_OBJECT_W_RANGE)
    mask = np.ones((h, w), dtype=bool)

    if direction in ("top", "bottom"):
        notch_w = random.randint(1, w - 2)
        notch_d = random.randint(1, h - 1)
        start = random.randint(1, w - notch_w - 1)
        if direction == "top":
            mask[:notch_d, start : start + notch_w] = False
        else:
            mask[h - notch_d :, start : start + notch_w] = False
    else:
        notch_h = random.randint(1, h - 2)
        notch_d = random.randint(1, w - 1)
        start = random.randint(1, h - notch_h - 1)
        if direction == "left":
            mask[start : start + notch_h, :notch_d] = False
        else:
            mask[start : start + notch_h, w - notch_d :] = False

    _notch_info(mask)
    return mask


def _paint_object(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = _OBJECT_COLOR


def _place_objects(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], directions: list[str]) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    blocked = np.zeros_like(grid, dtype=bool)

    min_r = fr0 + 1 + _FRAME_GAP
    max_inner_r = fr1 - 1 - _FRAME_GAP
    min_c = fc0 + 1 + _FRAME_GAP
    max_inner_c = fc1 - 1 - _FRAME_GAP

    for direction in directions:
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            mask = _sample_object(direction)
            h, w = mask.shape
            max_r = max_inner_r - h + 1
            max_c = max_inner_c - w + 1
            if max_r < min_r or max_c < min_c:
                return False

            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)

            br0 = top - _OBJECT_GAP
            bc0 = left - _OBJECT_GAP
            br1 = top + h - 1 + _OBJECT_GAP
            bc1 = left + w - 1 + _OBJECT_GAP
            if blocked[br0 : br1 + 1, bc0 : bc1 + 1].any():
                continue

            _paint_object(grid, top, left, mask)
            blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_generated(grid: np.ndarray, expected_directions: list[str]) -> bool:
    bg = _most_common_color(grid)
    frame_bbox, side_colors = _frame_info(grid, bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    interior_mask = np.zeros_like(grid, dtype=bool)
    interior_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = grid[fr0 + 1 : fr1, fc0 + 1 : fc1] == _OBJECT_COLOR
    comps = _components(interior_mask)
    if len(comps) != len(expected_directions):
        return False

    detected = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            obj_mask[r - r0, c - c0] = True
        notch_cells, direction = _notch_info(obj_mask)
        detected.append(direction)

        fill_color = side_colors[_OPPOSITE[direction]]
        solved = main(grid)
        for rr, cc in notch_cells:
            if int(solved[r0 + rr, c0 + cc]) != fill_color:
                return False

    if sorted(detected) != sorted(expected_directions):
        return False
    solved = main(grid)
    return np.any(solved != grid)


def _generate_attempt() -> np.ndarray | None:
    inner_h = random.randint(*_INTERIOR_H_RANGE)
    inner_w = random.randint(*_INTERIOR_W_RANGE)
    margin_top = random.randint(*_OUTER_MARGIN_RANGE)
    margin_bottom = random.randint(*_OUTER_MARGIN_RANGE)
    margin_left = random.randint(*_OUTER_MARGIN_RANGE)
    margin_right = random.randint(*_OUTER_MARGIN_RANGE)

    grid = np.full(
        (inner_h + 2 + margin_top + margin_bottom, inner_w + 2 + margin_left + margin_right),
        _BG,
        dtype=int,
    )
    frame_bbox = (
        margin_top,
        margin_left,
        margin_top + inner_h + 1,
        margin_left + inner_w + 1,
    )
    _draw_frame(grid, frame_bbox)

    interior_area = inner_h * inner_w
    max_count = _OBJECT_COUNT_RANGE[1] if interior_area >= 260 else _OBJECT_COUNT_RANGE[1] - 1
    count = random.randint(_OBJECT_COUNT_RANGE[0], max_count)
    directions = _sample_directions(count)

    if not _place_objects(grid, frame_bbox, directions):
        return None
    if not _validate_generated(grid, directions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
