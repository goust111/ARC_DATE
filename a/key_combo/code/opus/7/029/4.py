from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1
SOLID = 2

GEN_ATTEMPTS = 400
GRID_MIN = 15
GRID_MAX = 18
RECT_H_RANGE = (3, 4)
RECT_W_RANGE = (2, 3)
MAX_FRAME_SPAN = 7

ENCODE_POSITIONS = {
    0: [],
    1: [(1, 1)],
    2: [(1, 1), (0, 0)],
    3: [(1, 1), (0, 0), (2, 2)],
    4: [(1, 1), (0, 0), (2, 2), (0, 2)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _perimeter_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _ray_specs(red_box: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = red_box
    return [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]


def _ray_cells(
    start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while _in_bounds(shape, rr, cc):
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _encoding_area(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    return {(r, c) for r in range(h - 3, h) for c in range(w - 3, w)}


def _extract_red_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == SOLID
    box = _bbox(mask)
    r0, c0, r1, c1 = box
    if int(mask.sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
        raise ValueError("red object is not a solid rectangle")
    return box


def _extract_frame_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME
    box = _bbox(mask)
    expected = _perimeter_cells(box)
    actual = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if actual != expected:
        raise ValueError("blue object is not a 1-pixel hollow rectangle")
    return box


def _active_info(
    shape: tuple[int, int],
    red_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None]:
    frame_cells = _perimeter_cells(frame_box)
    frame_corners = {
        (frame_box[0], frame_box[1]),
        (frame_box[0], frame_box[3]),
        (frame_box[2], frame_box[1]),
        (frame_box[2], frame_box[3]),
    }
    encoding = _encoding_area(shape)
    info: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None] = []
    for start, direction in _ray_specs(red_box):
        cells = _ray_cells(start, direction, shape)
        hits = [cell for cell in cells if cell in frame_cells]
        if len(hits) > 1:
            return []
        if not hits:
            info.append(None)
            continue
        corner = hits[0]
        if corner not in frame_corners:
            return []
        idx = cells.index(corner)
        if idx < 1:
            return []
        trail = cells[idx + 1 :]
        if not trail:
            return []
        if any(cell in encoding for cell in trail):
            return []
        info.append((corner, direction, trail))
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """Count corner-gated diagonal rays from the red rectangle and encode that count in the bottom-right 3x3 area.
    Input structure:
    - A black grid contains one solid red rectangle, one blue 1-pixel hollow rectangular frame, and an all-black 3x3 encoding area in the bottom-right corner.
    - The red rectangle is the only red object and the blue frame is the only blue object.
    - The rest of the grid is background.
    Rule:
    - From each red rectangle corner, trace the outward diagonal ray.
    - A ray activates only when the first non-black pixel on that diagonal is a blue frame corner; when activated, paint the background cells beyond that corner red along the same diagonal.
    - Count the activated rays and place blue pixels in the fixed bottom-right 3x3 code pattern for that count.
    Key insight:
    - The output depends on a geometric corner-alignment test, but only through the scalar number of successful rays, which is then re-expressed as a fixed symbolic pattern.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    red_box = _extract_red_box(g)
    frame_box = _extract_frame_box(g)

    active = 0
    for item in _active_info(g.shape, red_box, frame_box):
        if item is None:
            continue
        _corner, _direction, trail = item
        active += 1
        for r, c in trail:
            if g[r, c] != BG:
                break
            out[r, c] = SOLID

    h, w = g.shape
    for dr, dc in ENCODE_POSITIONS[active]:
        out[h - 3 + dr, w - 3 + dc] = FRAME
    return out


def _sample_red_box(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    rh = random.randint(*RECT_H_RANGE)
    rw = random.randint(*RECT_W_RANGE)
    r0_max = min(7, h - rh - 5)
    c0_max = min(7, w - rw - 5)
    r0 = random.randint(4, r0_max)
    c0 = random.randint(4, c0_max)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _sample_frame_box(
    shape: tuple[int, int],
    red_box: tuple[int, int, int, int],
    target_active: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = red_box
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1

    if target_active == 1:
        variants = [
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
            ("left", "top"),
            ("left", "bottom"),
            ("right", "top"),
            ("right", "bottom"),
        ]
    else:
        variants = [("top", None), ("bottom", None), ("left", None), ("right", None)]
    random.shuffle(variants)

    for side, which in variants:
        if side == "top":
            max_d = min(r0 - 1, c0 - 1, w - 2 - c1)
            if max_d < 2:
                continue
            d = random.randint(2, max_d)
            fr1 = r0 - d
            min_h = rh + 1
            max_h = min(MAX_FRAME_SPAN, fr1 + 1)
            if min_h > max_h:
                continue
            fh = random.randint(min_h, max_h)
            fr0 = fr1 - fh + 1
            if target_active == 2:
                fc0 = c0 - d
                fc1 = c1 + d
            elif which == "left":
                fc0 = c0 - d
                lo = fc0 + rw
                hi = c1 + d - 1
                if lo > hi:
                    continue
                fc1 = random.randint(lo, hi)
            else:
                fc1 = c1 + d
                lo = c0 - d + 1
                hi = fc1 - rw
                if lo > hi:
                    continue
                fc0 = random.randint(lo, hi)
        elif side == "bottom":
            max_d = min(h - 2 - r1, c0 - 1, w - 2 - c1)
            if max_d < 2:
                continue
            d = random.randint(2, max_d)
            fr0 = r1 + d
            min_h = rh + 1
            max_h = min(MAX_FRAME_SPAN, h - fr0)
            if min_h > max_h:
                continue
            fh = random.randint(min_h, max_h)
            fr1 = fr0 + fh - 1
            if target_active == 2:
                fc0 = c0 - d
                fc1 = c1 + d
            elif which == "left":
                fc0 = c0 - d
                lo = fc0 + rw
                hi = c1 + d - 1
                if lo > hi:
                    continue
                fc1 = random.randint(lo, hi)
            else:
                fc1 = c1 + d
                lo = c0 - d + 1
                hi = fc1 - rw
                if lo > hi:
                    continue
                fc0 = random.randint(lo, hi)
        elif side == "left":
            max_d = min(c0 - 1, r0 - 1, h - 2 - r1)
            if max_d < 2:
                continue
            d = random.randint(2, max_d)
            fc1 = c0 - d
            min_w = rw + 1
            max_w = min(MAX_FRAME_SPAN, fc1 + 1)
            if min_w > max_w:
                continue
            fw = random.randint(min_w, max_w)
            fc0 = fc1 - fw + 1
            if target_active == 2:
                fr0 = r0 - d
                fr1 = r1 + d
            elif which == "top":
                fr0 = r0 - d
                lo = fr0 + rh
                hi = r1 + d - 1
                if lo > hi:
                    continue
                fr1 = random.randint(lo, hi)
            else:
                fr1 = r1 + d
                lo = r0 - d + 1
                hi = fr1 - rh
                if lo > hi:
                    continue
                fr0 = random.randint(lo, hi)
        else:
            max_d = min(w - 2 - c1, r0 - 1, h - 2 - r1)
            if max_d < 2:
                continue
            d = random.randint(2, max_d)
            fc0 = c1 + d
            min_w = rw + 1
            max_w = min(MAX_FRAME_SPAN, w - fc0)
            if min_w > max_w:
                continue
            fw = random.randint(min_w, max_w)
            fc1 = fc0 + fw - 1
            if target_active == 2:
                fr0 = r0 - d
                fr1 = r1 + d
            elif which == "top":
                fr0 = r0 - d
                lo = fr0 + rh
                hi = r1 + d - 1
                if lo > hi:
                    continue
                fr1 = random.randint(lo, hi)
            else:
                fr1 = r1 + d
                lo = r0 - d + 1
                hi = fr1 - rh
                if lo > hi:
                    continue
                fr0 = random.randint(lo, hi)

        box = (fr0, fc0, fr1, fc1)
        if fr0 < 0 or fc0 < 0 or fr1 >= h or fc1 >= w:
            continue
        if _perimeter_cells(box) & _encoding_area(shape):
            continue
        info = _active_info(shape, red_box, box)
        if info and sum(item is not None for item in info) == target_active:
            return box
    return None


def _render_grid(
    shape: tuple[int, int],
    red_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = red_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = SOLID
    for r, c in _perimeter_cells(frame_box):
        grid[r, c] = FRAME
    return grid


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    red_box = _sample_red_box(shape)
    target_active = random.choice([1, 2])
    frame_box = _sample_frame_box(shape, red_box, target_active)
    if frame_box is None:
        return None

    grid = _render_grid(shape, red_box, frame_box)
    active_info = _active_info(shape, red_box, frame_box)
    if not active_info or sum(item is not None for item in active_info) != target_active:
        return None
    if np.any(grid[-3:, -3:] != BG):
        return None
    out = main(grid)
    if int(np.count_nonzero(out[-3:, -3:] == FRAME)) != target_active:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
