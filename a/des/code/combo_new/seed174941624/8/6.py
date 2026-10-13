from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

STAMP_COLORS = (6, 7, 8, 9)
GRID_SIZES = (18, 22, 26, 30)
QUADRANTS = ("TL", "TR", "BL", "BR")
MODES = ("double", "horizontal_single", "vertical_single")
GEN_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            components.append(pts)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    return len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _scaled_len(length: int) -> int:
    return (int(length) + 1) // 2


def _round_ratio(numer: int, denom: int) -> int:
    return (2 * int(numer) + int(denom)) // (2 * int(denom))


def _map_axis(
    start: int,
    length: int,
    src_start: int,
    src_end: int,
    dst_start: int,
    dst_end: int,
) -> tuple[int, int]:
    src_len = int(src_end - src_start + 1)
    dst_len = int(dst_end - dst_start + 1)
    scaled = _scaled_len(length)
    assert 1 <= length <= src_len
    assert 1 <= scaled <= dst_len

    src_room = int(src_len - length)
    dst_room = int(dst_len - scaled)
    if src_room <= 0 or dst_room <= 0:
        offset = 0
    else:
        offset = _round_ratio((int(start) - int(src_start)) * dst_room, src_room)
    offset = max(0, min(offset, dst_room))
    return int(dst_start + offset), int(scaled)


def _source_box(h: int, w: int, quadrant: str) -> tuple[int, int, int, int]:
    top_rows = (1, h // 2 - 1)
    bottom_rows = (h // 2, h - 2)
    left_cols = (1, w // 2 - 1)
    right_cols = (w // 2, w - 2)

    if quadrant == "TL":
        return top_rows[0], left_cols[0], top_rows[1], left_cols[1]
    if quadrant == "TR":
        return top_rows[0], right_cols[0], top_rows[1], right_cols[1]
    if quadrant == "BL":
        return bottom_rows[0], left_cols[0], bottom_rows[1], left_cols[1]
    if quadrant == "BR":
        return bottom_rows[0], right_cols[0], bottom_rows[1], right_cols[1]
    raise ValueError(quadrant)


def _dest_box(out_h: int, out_w: int, quadrant: str) -> tuple[int, int, int, int]:
    mid_r = out_h // 2
    mid_c = out_w // 2

    if quadrant == "TL":
        return 0, 0, mid_r - 1, mid_c - 1
    if quadrant == "TR":
        return 0, mid_c + 1, mid_r - 1, out_w - 1
    if quadrant == "BL":
        return mid_r + 1, 0, out_h - 1, mid_c - 1
    if quadrant == "BR":
        return mid_r + 1, mid_c + 1, out_h - 1, out_w - 1
    raise ValueError(quadrant)


def _quadrant_for_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> str:
    r0, c0, r1, c1 = bbox

    touches_top = r0 == 1
    touches_bottom = r1 == h - 2
    touches_left = c0 == 1
    touches_right = c1 == w - 2

    assert touches_top or touches_bottom or touches_left or touches_right
    assert not (touches_top and touches_bottom)
    assert not (touches_left and touches_right)

    if touches_top:
        vertical = "T"
    elif touches_bottom:
        vertical = "B"
    else:
        vertical = "T" if (r0 + r1) < (h - 1) else "B"

    if touches_left:
        horizontal = "L"
    elif touches_right:
        horizontal = "R"
    else:
        horizontal = "L" if (c0 + c1) < (w - 1) else "R"

    return vertical + horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink border-touching stamp rectangles and collage them into the matching output quadrant.

    Input structure:
    - A black background fills the grid interior.
    - The outer ring is a border legend: top edge is red, bottom is blue, left is green, and right is yellow.
    - Inside that legend ring sit several solid colored rectangles; each rectangle touches one or two sides of the ring and uses a color different from the legend colors.

    Rule:
    - Output a grid of shape `(input_height // 2, input_width // 2)`.
    - Paint a grey horizontal line and a grey vertical line through the center of the output to form four quadrants.
    - For each input rectangle, read which border sides it touches: top means "up", bottom means "down", left means "left", and right means "right".
    - A rectangle touching two adjacent sides goes to the corresponding corner quadrant. A rectangle touching only one side goes to the nearer of the two candidate quadrants based on whether its original center lies in the left/right or top/bottom half.
    - Shrink each rectangle to half height and half width, rounding half-up, then place it inside its assigned quadrant while preserving its relative position within the matching half of the input.

    Key insight:
    - The colored border is not content to copy; it is a legend that converts edge contact into quadrant membership, after which each stamp is resized and repositioned proportionally.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out_h, out_w = h // 2, w // 2
    assert out_h >= 3 and out_w >= 3

    out = np.full((out_h, out_w), BLACK, dtype=int)
    mid_r = out_h // 2
    mid_c = out_w // 2
    out[mid_r, :] = GREY
    out[:, mid_c] = GREY

    components = _find_components(g)
    ordered = sorted(components, key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1], int(g[pts[0][0], pts[0][1]])))

    for pts in ordered:
        assert _is_solid_rectangle(pts)
        r0, c0, r1, c1 = _bbox(pts)
        color = int(g[r0, c0])
        quadrant = _quadrant_for_bbox((r0, c0, r1, c1), h, w)

        src_r0, src_c0, src_r1, src_c1 = _source_box(h, w, quadrant)
        dst_r0, dst_c0, dst_r1, dst_c1 = _dest_box(out_h, out_w, quadrant)

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert src_r0 <= r0 <= r1 <= src_r1
        assert src_c0 <= c0 <= c1 <= src_c1

        new_r0, new_h = _map_axis(r0, height, src_r0, src_r1, dst_r0, dst_r1)
        new_c0, new_w = _map_axis(c0, width, src_c0, src_c1, dst_c0, dst_c1)
        assert new_r0 + new_h - 1 <= dst_r1
        assert new_c0 + new_w - 1 <= dst_c1

        out[new_r0 : new_r0 + new_h, new_c0 : new_c0 + new_w] = color

    return out


def _paint_border_markers(grid: np.ndarray) -> None:
    grid[0, 1:-1] = RED
    grid[-1, 1:-1] = BLUE
    grid[1:-1, 0] = GREEN
    grid[1:-1, -1] = YELLOW


def _mode_list(count: int) -> list[str]:
    if count == 2:
        modes = ["double", random.choice(["horizontal_single", "vertical_single"])]
    elif count == 3:
        modes = ["double", "horizontal_single", "vertical_single"]
    else:
        modes = ["double", "horizontal_single", "vertical_single", random.choice(MODES)]
    random.shuffle(modes)
    return modes


def _sample_bbox_for_box(
    src_r0: int,
    src_c0: int,
    src_r1: int,
    src_c1: int,
    quadrant: str,
    mode: str,
) -> tuple[int, int, int, int]:
    src_h = src_r1 - src_r0 + 1
    src_w = src_c1 - src_c0 + 1

    touches_top = quadrant in ("TL", "TR")
    touches_left = quadrant in ("TL", "BL")

    min_h = 2
    min_w = 2
    max_h = min(src_h - (2 if mode == "vertical_single" else 1), 7)
    max_w = min(src_w - (2 if mode == "horizontal_single" else 1), 7)
    assert min_h <= max_h and min_w <= max_w

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)

    if mode == "double":
        r0 = src_r0 if touches_top else src_r1 - height + 1
        c0 = src_c0 if touches_left else src_c1 - width + 1
    elif mode == "horizontal_single":
        r0 = src_r0 if touches_top else src_r1 - height + 1
        c0 = random.randint(src_c0 + 1, src_c1 - width)
    elif mode == "vertical_single":
        r0 = random.randint(src_r0 + 1, src_r1 - height)
        c0 = src_c0 if touches_left else src_c1 - width + 1
    else:
        raise ValueError(mode)

    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        h = random.choice(GRID_SIZES)
        w = random.choice(GRID_SIZES)
        count = random.randint(2, 4)
        quadrants = random.sample(QUADRANTS, count)
        colors = random.sample(STAMP_COLORS, count)
        modes = _mode_list(count)

        grid = np.full((h, w), BLACK, dtype=int)
        _paint_border_markers(grid)

        specs: list[tuple[str, str, int, tuple[int, int, int, int]]] = []
        ok = True
        for quadrant, mode, color in zip(quadrants, modes, colors, strict=False):
            src_r0, src_c0, src_r1, src_c1 = _source_box(h, w, quadrant)
            try:
                bbox = _sample_bbox_for_box(src_r0, src_c0, src_r1, src_c1, quadrant, mode)
            except (AssertionError, ValueError):
                ok = False
                break
            r0, c0, r1, c1 = bbox
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            specs.append((quadrant, mode, color, bbox))

        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.shape != (h // 2, w // 2):
            continue

        used_colors = {color for _quadrant, _mode, color, _bbox_info in specs}
        out_colors = {int(v) for v in np.unique(out) if int(v) not in (BLACK, GREY)}
        if out_colors != used_colors:
            continue

        if len({mode for _quadrant, mode, _color, _bbox_info in specs}) == 1 and count >= 3:
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
