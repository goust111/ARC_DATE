from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_frame(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    pts = {tuple(map(int, p)) for p in coords}
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _is_solid_rectangle(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_frame_and_objects(grid: np.ndarray, bg: int) -> tuple[BBox, list[dict[str, int | BBox]]]:
    body = grid[:-1, :-1]
    frame_box: BBox | None = None
    objects: list[dict[str, int | BBox]] = []

    for color in sorted(int(v) for v in np.unique(body) if int(v) != bg):
        for coords in _components(body == color):
            box = _bbox(coords)
            if _is_frame(coords, box):
                assert frame_box is None, "expected a unique grey frame"
                frame_box = box
            else:
                assert _is_solid_rectangle(coords, box), "expected monochrome solid rectangles"
                objects.append({"color": int(color), "bbox": box})

    assert frame_box is not None, "expected a frame component"
    objects.sort(key=lambda obj: tuple(int(x) for x in obj["bbox"]))  # top-left reading order
    return frame_box, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project every solid rectangle to the right/bottom strips, then redraw the frame interior
    from the strip intersection colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(g[-1, -1])
    frame_box, objects = _find_frame_and_objects(g, bg)

    out = g.copy()
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, -1] = color
        out[-1, c0 : c1 + 1] = color

    fr0, fc0, fr1, fc1 = frame_box
    row_proj = out[fr0 + 1 : fr1, -1][:, None]
    col_proj = out[-1, fc0 + 1 : fc1][None, :]
    row_colors = np.repeat(row_proj, fc1 - fc0 - 1, axis=1)
    col_colors = np.repeat(col_proj, fr1 - fr0 - 1, axis=0)

    interior = np.full((fr1 - fr0 - 1, fc1 - fc0 - 1), bg, dtype=int)
    active = (row_colors != bg) & (col_colors != bg)
    same = active & (row_colors == col_colors)
    diff = active & ~same
    interior[same] = row_colors[same]
    interior[diff] = col_colors[diff]
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = interior
    return out


# --------------- generator ---------------

BG = 0
FRAME = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_MIN = 17
GRID_MAX = 24
FRAME_SIDE_MIN = 6
FRAME_SIDE_MAX = 9
MAX_GEN_ATTEMPTS = 500


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_overlap_pair(total_len: int) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(200):
        max_len = min(4, total_len - 1)
        if max_len < 2:
            break
        len1 = random.randint(2, max_len)
        len2 = random.randint(2, max_len)
        max_overlap = min(len1 - 1, len2)
        if max_overlap < 1:
            continue
        overlap = random.randint(1, max_overlap)
        span = len1 + len2 - overlap
        if span > total_len:
            continue
        start1 = random.randint(0, total_len - span)
        start2 = start1 + len1 - overlap
        return (start1, start1 + len1 - 1), (start2, start2 + len2 - 1)
    raise RuntimeError("failed to sample overlapping intervals")


def _branch_counts(out: np.ndarray, frame_box: BBox, bg: int) -> tuple[int, int, int]:
    fr0, fc0, fr1, fc1 = frame_box
    same = diff = blank = 0
    for r in range(fr0 + 1, fr1):
        row_color = int(out[r, -1])
        for c in range(fc0 + 1, fc1):
            col_color = int(out[-1, c])
            if row_color == bg or col_color == bg:
                blank += 1
            elif row_color == col_color:
                same += 1
            else:
                diff += 1
    return same, diff, blank


def _sample_extra_box(zone: BBox) -> BBox | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    if zh < 2 or zw < 2:
        return None
    h = random.randint(2, min(4, zh))
    w = random.randint(2, min(4, zw))
    r0 = random.randint(zr0, zr1 - h + 1)
    c0 = random.randint(zc0, zc1 - w + 1)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    frame_h = random.randint(FRAME_SIDE_MIN, min(FRAME_SIDE_MAX, h - 8))
    frame_w = random.randint(FRAME_SIDE_MIN, min(FRAME_SIDE_MAX, w - 8))
    fr0 = random.randint(3, h - frame_h - 4)
    fc0 = random.randint(3, w - frame_w - 4)
    frame_box = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr0 + frame_h - 2, fc0 + frame_w - 2
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1

    (a0, a1), (d0, d1) = _sample_overlap_pair(inner_h)
    (b0, b1), (c0, c1) = _sample_overlap_pair(inner_w)

    extra_count = random.randint(0, 2)
    colors = random.sample(OBJECT_COLORS, 3 + extra_count)
    same_color, diff_color, row_over_color = colors[:3]
    extra_colors = colors[3:]

    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame_box, FRAME)

    left_width = random.randint(2, min(4, fc0 - 1))
    left_c0 = random.randint(0, fc0 - left_width - 1)
    left_box = (ir0 + a0, left_c0, ir0 + a1, left_c0 + left_width - 1)

    right_width = random.randint(2, min(4, w - (fc0 + frame_w - 1) - 3))
    right_c0 = random.randint(fc0 + frame_w + 1, w - right_width - 1)
    right_box = (ir0 + d0, right_c0, ir0 + d1, right_c0 + right_width - 1)

    top_height = random.randint(2, min(4, fr0 - 1))
    top_r0 = random.randint(0, fr0 - top_height - 1)
    top_box = (top_r0, ic0 + b0, top_r0 + top_height - 1, ic0 + b1)

    bottom_height = random.randint(2, min(4, h - (fr0 + frame_h - 1) - 3))
    bottom_r0 = random.randint(fr0 + frame_h + 1, h - bottom_height - 1)
    bottom_box = (bottom_r0, ic0 + c0, bottom_r0 + bottom_height - 1, ic0 + c1)

    _paint_rect(grid, left_box, same_color)
    _paint_rect(grid, right_box, row_over_color)
    _paint_rect(grid, top_box, same_color)
    _paint_rect(grid, bottom_box, diff_color)

    corner_zones = [
        (0, 0, fr0 - 2, fc0 - 2),
        (0, fc0 + frame_w + 1, fr0 - 2, w - 2),
        (fr0 + frame_h + 1, 0, h - 2, fc0 - 2),
        (fr0 + frame_h + 1, fc0 + frame_w + 1, h - 2, w - 2),
    ]
    valid_zones = [zone for zone in corner_zones if zone[0] <= zone[2] and zone[1] <= zone[3]]
    random.shuffle(valid_zones)
    for color, zone in zip(extra_colors, valid_zones):
        box = _sample_extra_box(zone)
        if box is None:
            return None
        _paint_rect(grid, box, color)

    if np.any(grid[-1, :] != BG) or np.any(grid[:, -1] != BG):
        return None
    if np.mean(grid == BG) < 0.55:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    same, diff, blank = _branch_counts(out, frame_box, BG)
    if min(same, diff, blank) == 0:
        return None

    detected_frame, objects = _find_frame_and_objects(grid, BG)
    if detected_frame != frame_box or len(objects) != 4 + extra_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
