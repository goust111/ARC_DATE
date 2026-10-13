from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 1
TEMPLATE = 3
FRAME = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bbox_from_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame_component(coords: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    if not coords:
        return False, None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, None
    coord_set = set(coords)
    expected = set()
    for rr in range(r0, r1 + 1):
        expected.add((rr, c0))
        expected.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))
    if coord_set != expected:
        return False, None
    return True, (r0, c0, r1, c1)


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _find_components_of_color(grid, FRAME):
        ok, box = _is_rect_frame_component(comp)
        if ok and box is not None:
            frames.append(box)
    return frames


def _adjacent_template_side(
    frame_box: tuple[int, int, int, int], template_comp: list[tuple[int, int]]
) -> str | None:
    r0, c0, r1, c1 = frame_box
    sides = set()
    template_set = set(template_comp)
    for rr, cc in template_set:
        if (rr + 1, cc) in {(r0, x) for x in range(c0, c1 + 1)}:
            sides.add("top")
        if (rr - 1, cc) in {(r1, x) for x in range(c0, c1 + 1)}:
            sides.add("bottom")
        if (rr, cc + 1) in {(x, c0) for x in range(r0, r1 + 1)}:
            sides.add("left")
        if (rr, cc - 1) in {(x, c1) for x in range(r0, r1 + 1)}:
            sides.add("right")
    if len(sides) != 1:
        return None
    return next(iter(sides))


def _place_shape(out: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each frame's interior markers, then tile the touching exterior template outward that many times.

    Input structure:
    - The grid has a black background with 2-4 orange one-pixel-thick rectangular frames.
    - Blue pixels appear both inside frames and elsewhere as distractors.
    - Each frame has exactly one green 4-connected template touching one outside side of that frame.

    Rule:
    - For every orange frame, count the blue pixels strictly inside the frame.
    - Detect which side its green template touches, then copy that template outward away from the frame.
    - Use the template's own bounding-box height or width as the translation step, draw exactly N green copies, and keep all original pixels unchanged.

    Key insight:
    - The blue count is not rendered directly; it becomes the repetition count, while the template's touching side determines the tiling direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_boxes = _detect_frames(g)
    template_comps = _find_components_of_color(g, TEMPLATE)

    assert 2 <= len(frame_boxes) <= 4
    assert len(template_comps) >= len(frame_boxes)

    for frame_box in frame_boxes:
        r0, c0, r1, c1 = frame_box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(interior == MARKER))

        matches = []
        for comp in template_comps:
            side = _adjacent_template_side(frame_box, comp)
            if side is not None:
                matches.append((comp, side))
        assert len(matches) == 1

        comp, side = matches[0]
        tr0, tc0, tr1, tc1 = _bbox_from_coords(comp)
        local = [(r - tr0, c - tc0) for r, c in comp]
        th = tr1 - tr0 + 1
        tw = tc1 - tc0 + 1

        if side == "top":
            step_r, step_c = -th, 0
        elif side == "bottom":
            step_r, step_c = th, 0
        elif side == "left":
            step_r, step_c = 0, -tw
        else:
            step_r, step_c = 0, tw

        for k in range(1, count + 1):
            shifted = [(tr0 + lr + k * step_r, tc0 + lc + k * step_c) for lr, lc in local]
            _place_shape(out, shifted, TEMPLATE)

    return out


def _trim_coords(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(list(cells))
    trimmed = [(r - r0, c - c0) for r, c in cells]
    return sorted(trimmed), int(r1 - r0 + 1), int(c1 - c0 + 1)


def _random_template(side: str) -> tuple[list[tuple[int, int]], int, int]:
    while True:
        h, w = random.choice([(2, 2), (2, 3), (3, 2)])
        if side == "top":
            boundary = [(h - 1, c) for c in range(w)]
        elif side == "bottom":
            boundary = [(0, c) for c in range(w)]
        elif side == "left":
            boundary = [(r, w - 1) for r in range(h)]
        else:
            boundary = [(r, 0) for r in range(h)]

        target = random.randint(2, 4)
        cells = {random.choice(boundary)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        trimmed, th, tw = _trim_coords(cells)
        area = th * tw
        if len(trimmed) < 2:
            continue
        if area == len(trimmed) and area > 2:
            continue
        if side == "top" and not any(r == th - 1 for r, _ in trimmed):
            continue
        if side == "bottom" and not any(r == 0 for r, _ in trimmed):
            continue
        if side == "left" and not any(c == tw - 1 for _, c in trimmed):
            continue
        if side == "right" and not any(c == 0 for _, c in trimmed):
            continue
        return trimmed, th, tw


def _rect_free(blocked: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_rect(blocked: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = rect
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _shape_global(local: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in local]


def _corridor_rect(
    side: str, frame_box: tuple[int, int, int, int], template_top: int, template_left: int, th: int, tw: int, count: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_box
    if side == "top":
        return (r0 - (count + 1) * th, template_left, r0 - 1, template_left + tw - 1)
    if side == "bottom":
        return (r1 + 1, template_left, r1 + (count + 1) * th, template_left + tw - 1)
    if side == "left":
        return (template_top, c0 - (count + 1) * tw, template_top + th - 1, c0 - 1)
    return (template_top, c1 + 1, template_top + th - 1, c1 + (count + 1) * tw)


def _interior_coords(frame_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _draw_frame(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_box
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _copy_positions(
    local: list[tuple[int, int]], template_top: int, template_left: int, th: int, tw: int, side: str, count: int
) -> list[list[tuple[int, int]]]:
    if side == "top":
        delta = (-th, 0)
    elif side == "bottom":
        delta = (th, 0)
    elif side == "left":
        delta = (0, -tw)
    else:
        delta = (0, tw)
    out = []
    for k in range(1, count + 1):
        dr, dc = delta[0] * k, delta[1] * k
        out.append([(template_top + r + dr, template_left + c + dc) for r, c in local])
    return out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()
    blocked = np.zeros((h, w), dtype=bool)

    n_objs = random.randint(2, 4)
    sides = random.sample(["top", "bottom", "left", "right"], k=n_objs)
    random.shuffle(sides)

    specs = []
    for side in sides:
        local, th, tw = _random_template(side)
        inner_h = random.randint(3, 5)
        inner_w = random.randint(3, 5)
        count = random.randint(1, 5)
        specs.append(
            {
                "side": side,
                "local": local,
                "th": th,
                "tw": tw,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "count": count,
                "size_score": (count + 1) * max(th, tw) + inner_h + inner_w,
            }
        )
    specs.sort(key=lambda d: int(d["size_score"]), reverse=True)

    placed = []
    for spec in specs:
        side = spec["side"]
        th = int(spec["th"])
        tw = int(spec["tw"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        count = int(spec["count"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if side == "top":
            r_lo, r_hi = (count + 1) * th, h - outer_h
            c_lo, c_hi = 0, w - outer_w
        elif side == "bottom":
            r_lo, r_hi = 0, h - outer_h - (count + 1) * th
            c_lo, c_hi = 0, w - outer_w
        elif side == "left":
            r_lo, r_hi = 0, h - outer_h
            c_lo, c_hi = (count + 1) * tw, w - outer_w
        else:
            r_lo, r_hi = 0, h - outer_h
            c_lo, c_hi = 0, w - outer_w - (count + 1) * tw
        if r_lo > r_hi or c_lo > c_hi:
            return None

        contact_cells = []
        if side == "top":
            contact_cells = [(r, c) for r, c in spec["local"] if r == th - 1]
        elif side == "bottom":
            contact_cells = [(r, c) for r, c in spec["local"] if r == 0]
        elif side == "left":
            contact_cells = [(r, c) for r, c in spec["local"] if c == tw - 1]
        else:
            contact_cells = [(r, c) for r, c in spec["local"] if c == 0]
        if not contact_cells:
            return None

        success = False
        for _ in range(250):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            frame_box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _rect_free(blocked, frame_box):
                continue

            cr, cc = random.choice(contact_cells)
            if side in ("top", "bottom"):
                frame_c = random.randint(c0, c0 + outer_w - 1)
                template_left = frame_c - cc
                template_top = r0 - th if side == "top" else r0 + outer_h
            else:
                frame_r = random.randint(r0, r0 + outer_h - 1)
                template_top = frame_r - cr
                template_left = c0 - tw if side == "left" else c0 + outer_w

            if template_top < 0 or template_left < 0 or template_top + th > h or template_left + tw > w:
                continue

            corridor = _corridor_rect(side, frame_box, template_top, template_left, th, tw, count)
            if not _rect_free(blocked, corridor):
                continue

            interior = _interior_coords(frame_box)
            if len(interior) < count:
                continue

            marker_cells = random.sample(interior, count)
            template_cells = _shape_global(spec["local"], template_top, template_left)
            copies = _copy_positions(spec["local"], template_top, template_left, th, tw, side, count)

            success = True
            placed.append(
                {
                    "frame_box": frame_box,
                    "side": side,
                    "count": count,
                    "marker_cells": marker_cells,
                    "template_cells": template_cells,
                    "copies": copies,
                }
            )
            _mark_rect(blocked, frame_box, pad=1)
            _mark_rect(blocked, corridor, pad=1)
            break
        if not success:
            return None

    for obj in placed:
        _draw_frame(grid, obj["frame_box"])
        _draw_frame(expected, obj["frame_box"])
        for r, c in obj["marker_cells"]:
            grid[r, c] = MARKER
            expected[r, c] = MARKER
        for r, c in obj["template_cells"]:
            grid[r, c] = TEMPLATE
            expected[r, c] = TEMPLATE
        for copy_cells in obj["copies"]:
            for r, c in copy_cells:
                expected[r, c] = TEMPLATE

    free = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == BG and not bool(blocked[r, c])]
    if len(free) < n_objs + 1:
        return None
    distractor_count = random.randint(n_objs, n_objs + 3)
    if len(free) < distractor_count:
        return None
    for r, c in random.sample(free, distractor_count):
        grid[r, c] = MARKER
        expected[r, c] = MARKER

    if not (2 <= len(_detect_frames(grid)) <= 4):
        return None
    if len(_find_components_of_color(grid, TEMPLATE)) != len(placed):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _attempt_generate()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
