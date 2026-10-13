import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _components_by_color(grid: np.ndarray, bg: int = BG) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        out[int(color)] = _components_for_color(grid, int(color))
    return out


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _translate_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    bh, bw = _bbox_hw(bbox)
    dr = 0
    dc = 0
    if r0 == 0:
        dr += bh
    if r1 == h - 1:
        dr -= bh
    if c0 == 0:
        dc += bw
    if c1 == w - 1:
        dc -= bw
    assert dr != 0 or dc != 0
    nr0, nc0, nr1, nc1 = r0 + dr, c0 + dc, r1 + dr, c1 + dc
    assert 0 <= nr0 <= nr1 < h
    assert 0 <= nc0 <= nc1 < w
    return nr0, nc0, nr1, nc1


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int, int]) -> None:
    r0, c0, h, w, color = rect
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _rect_bbox(rect: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, h, w, _color = rect
    return r0, c0, r0 + h - 1, c0 + w - 1


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.asarray(out, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the repeated boundary rectangles inward and resolve their overlaps by xor parity.

    Input structure:
    - The grid uses black (0) as background.
    - Foreground objects are solid monochrome rectangles, each touching at least one grid border.
    - Exactly one non-background color appears in multiple 4-connected components; every other foreground color appears once.

    Rule:
    - Identify the repeated color by component count and erase all of its original rectangles.
    - Translate each repeated-color rectangle inward by its own size: top/bottom-aligned rectangles move vertically by their height, left/right-aligned rectangles move horizontally by their width, and corner rectangles move along both axes.
    - Keep singleton-color rectangles fixed, then overlay all translated repeated rectangles with xor parity: odd coverage paints the repeated color, even coverage returns to black.
    - Singleton-color cells have priority over translated repeated-color cells and are never overwritten.

    Key insight:
    - The moving objects are selected by repeated color multiplicity, but their final combination is not a merge: translated overlaps cancel in even-covered regions, so parity rather than union determines the repeated-color output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    by_color = _components_by_color(g, bg=BG)
    moving_colors = [color for color, comps in by_color.items() if len(comps) >= 2]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])
    moving = by_color[moving_color]

    out = g.copy()
    for comp in moving:
        r0, c0, r1, c1 = comp["bbox"]
        bh, bw = _bbox_hw(comp["bbox"])
        assert comp["area"] == bh * bw
        assert r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    coverage = np.zeros_like(g, dtype=int)
    for comp in moving:
        r0, c0, r1, c1 = _translate_bbox(comp["bbox"], (h, w))
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1

    odd = (coverage % 2) == 1
    out[odd & (out == BG)] = moving_color
    return out


def _validate_generated(grid: np.ndarray, expect_priority: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    by_color = _components_by_color(grid, bg=BG)
    moving_colors = [color for color, comps in by_color.items() if len(comps) >= 2]
    if len(moving_colors) != 1:
        return False
    moving_color = int(moving_colors[0])
    moving = by_color[moving_color]
    if len(moving) < 3:
        return False

    translated = []
    coverage = np.zeros_like(grid, dtype=int)
    for comp in moving:
        bbox = comp["bbox"]
        bh, bw = _bbox_hw(bbox)
        if comp["area"] != bh * bw:
            return False
        r0, c0, r1, c1 = bbox
        if not (r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1):
            return False
        tb = _translate_bbox(bbox, grid.shape)
        translated.append(tb)
        tr0, tc0, tr1, tc1 = tb
        coverage[tr0 : tr1 + 1, tc0 : tc1 + 1] += 1

    even_overlap = (coverage > 0) & ((coverage % 2) == 0)
    if not np.any(even_overlap):
        return False

    isolated = False
    for i, a in enumerate(translated):
        if all(i == j or _intersection_area(a, b) == 0 for j, b in enumerate(translated)):
            isolated = True
            break
    if not isolated:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out == moving_color):
        return False
    if not np.any(even_overlap & (out == BG)):
        return False

    singleton_mask = (grid != BG) & (grid != moving_color)
    if expect_priority:
        overlap_with_singleton = singleton_mask & (coverage > 0)
        if not np.any(overlap_with_singleton):
            return False
        if not np.array_equal(out[overlap_with_singleton], grid[overlap_with_singleton]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    priority_branch = random.random() < 0.5
    singleton_color_count = 1 if random.random() < 0.55 else 2
    colors = random.sample(PALETTE, 1 + singleton_color_count)
    moving_color = colors[0]
    singleton_colors = colors[1:]

    corner_h = corner_w = 0
    reserve_right = 2
    if priority_branch:
        corner_h = random.randint(2, 3)
        corner_w = random.randint(2, 3)
        reserve_right = max(5, corner_w + 3)

    top_h = random.randint(3, 4)
    top_w = random.randint(3, 4)
    left_h = random.randint(3, 4)
    left_w = random.randint(3, 4)

    top_c_low = left_w + 1
    top_c_high = min(2 * left_w - 2, w - top_w - reserve_right)
    left_r_low = top_h + 1
    left_r_high = min(2 * top_h - 2, h - left_h - 5)
    if top_c_low > top_c_high or left_r_low > left_r_high:
        return None

    top_c = random.randint(top_c_low, top_c_high)
    left_r = random.randint(left_r_low, left_r_high)

    repeated_rects = [
        (0, top_c, top_h, top_w, moving_color),
        (left_r, 0, left_h, left_w, moving_color),
    ]

    singleton_rects: list[tuple[int, int, int, int, int]] = []
    if priority_branch:
        repeated_rects.append((0, w - corner_w, corner_h, corner_w, moving_color))

        single_h = random.randint(2, 4)
        single_w_low = corner_w + 1
        single_w_high = min(5, w - 1)
        single_r_low = corner_h + 1
        single_r_high = min(2 * corner_h - 1, h - single_h - 5)
        if single_w_low > single_w_high or single_r_low > single_r_high:
            return None
        single_w = random.randint(single_w_low, single_w_high)
        single_r = random.randint(single_r_low, single_r_high)
        singleton_rects.append((single_r, w - single_w, single_h, single_w, singleton_colors[0]))
    else:
        bl_h = random.randint(2, 3)
        bl_w = random.randint(2, 3)
        singleton_rects.append((h - bl_h, 0, bl_h, bl_w, singleton_colors[0]))

    iso_h = random.randint(2, 4)
    iso_w = random.randint(2, 4)
    iso_r_low = max(h // 2, left_r + left_h + 2)
    if priority_branch and singleton_rects:
        sr, _sc, sh, _sw, _col = singleton_rects[0]
        iso_r_low = max(iso_r_low, sr + sh + 2)
    iso_r_high = h - iso_h - 1
    if iso_r_low > iso_r_high:
        return None
    iso_r = random.randint(iso_r_low, iso_r_high)
    repeated_rects.append((iso_r, w - iso_w, iso_h, iso_w, moving_color))

    if len(singleton_colors) > 1:
        extra_color = singleton_colors[1]
        if priority_branch:
            extra_h = random.randint(2, 3)
            extra_w = random.randint(2, 3)
            singleton_rects.append((h - extra_h, 0, extra_h, extra_w, extra_color))
        else:
            extra_h = random.randint(2, 3)
            extra_w = random.randint(2, 3)
            singleton_rects.append((0, w - extra_w, extra_h, extra_w, extra_color))

    rects = repeated_rects + singleton_rects
    bboxes = [_rect_bbox(rect) for rect in rects]
    for i, a in enumerate(bboxes):
        for j, b in enumerate(bboxes):
            if i >= j:
                continue
            if _intersection_area(a, b) > 0:
                return None

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        _paint_rect(grid, rect)

    grid = _transform_grid(grid)
    if not _validate_generated(grid, expect_priority=priority_branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
