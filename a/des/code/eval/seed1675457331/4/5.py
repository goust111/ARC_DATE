from __future__ import annotations

import random

import numpy as np

BG = 0

LEGEND_W = 2
LEGEND_H = 2
LEGEND_GAP = 1

MIN_RECTS = 3
MAX_RECTS = 5
MIN_SIDE = 3
MAX_SIDE = 7

BOARD_H_MIN = 12
BOARD_H_MAX = 19
BOARD_W_MIN = 12
BOARD_W_MAX = 17
LEGEND_SEP_MIN = 1
LEGEND_SEP_MAX = 3

GEN_RETRIES = 400
TOP_RETRIES = 80
RECT_RETRIES = 120

Rect = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> Rect:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_from_mask(mask: np.ndarray) -> Rect:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_height(rect: Rect) -> int:
    return rect[2] - rect[0] + 1


def _rect_width(rect: Rect) -> int:
    return rect[3] - rect[1] + 1


def _rect_area(rect: Rect) -> int:
    return _rect_height(rect) * _rect_width(rect)


def _fill_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _rectangles_overlap(a: Rect, b: Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _intersection(a: Rect, b: Rect) -> Rect | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(np.int16)
    return np.pad(arr, ((1, 0), (1, 0)), constant_values=0).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    h, w = grid.shape
    sep_col = None
    for c in range(w - 2, -1, -1):
        if np.all(grid[:, c] == bg) and np.any(grid[:, c + 1 :] != bg):
            sep_col = c
            break
    assert sep_col is not None, "expected a blank separator column before the legend"

    legend = grid[:, sep_col + 1 :]
    comps = _components(legend != bg)
    assert comps, "expected legend blocks on the right side"

    items: list[tuple[int, int]] = []
    for coords in comps:
        colors = np.unique(legend[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "legend blocks must be monochrome"
        items.append((int(coords[:, 0].min()), int(colors[0])))

    items.sort()
    colors = [color for _row, color in items]
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors, int(sep_col + 1)


def _candidate_rectangles(scene: np.ndarray, color: int, later_mask: np.ndarray) -> list[Rect]:
    visible = scene == color
    assert visible.any(), "each legend color must remain visible in the scene"

    r0_vis, c0_vis, r1_vis, c1_vis = _bbox_from_mask(visible)
    total_visible = int(visible.sum())
    unblocked = ~later_mask
    prefix = _prefix_sum(unblocked)

    h, w = scene.shape
    candidates: list[Rect] = []
    for r0 in range(r0_vis + 1):
        for r1 in range(r1_vis, h):
            for c0 in range(c0_vis + 1):
                for c1 in range(c1_vis, w):
                    rect = (r0, c0, r1, c1)
                    if _rect_sum(prefix, rect) == total_visible:
                        candidates.append(rect)
    return candidates


def _visible_boundary_sides(visible: np.ndarray, rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    score = 0
    score += int(visible[r0, c0 : c1 + 1].any())
    score += int(visible[r1, c0 : c1 + 1].any())
    score += int(visible[r0 : r1 + 1, c0].any())
    score += int(visible[r0 : r1 + 1, c1].any())
    return score


def _choose_rectangle(scene: np.ndarray, color: int, later_mask: np.ndarray) -> Rect:
    visible = scene == color
    candidates = _candidate_rectangles(scene, color, later_mask)
    if not candidates:
        return _bbox_from_mask(visible)
    return min(
        candidates,
        key=lambda rect: (
            -_visible_boundary_sides(visible, rect),
            _rect_area(rect),
            rect[0],
            rect[1],
            rect[2],
            rect[3],
        ),
    )


def _reconstruct_rectangles(scene: np.ndarray, order: list[int]) -> dict[int, Rect]:
    later_mask = np.zeros(scene.shape, dtype=bool)
    rects: dict[int, Rect] = {}
    for color in reversed(order):
        rect = _choose_rectangle(scene, color, later_mask)
        rects[color] = rect
        r0, c0, r1, c1 = rect
        later_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every earlier rectangle that overlaps the last legend-listed rectangle.

    The right-side legend gives the drawing order from top to bottom.  Each color
    corresponds to one solid axis-aligned rectangle.  The solver reconstructs the
    rectangle for each color from its visible cells plus the already-reconstructed
    later rectangles, then redraws only the topmost rectangle and any earlier
    rectangles that do not intersect it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    order, legend_start = _extract_legend(g, BG)
    scene = g[:, :legend_start]
    rects = _reconstruct_rectangles(scene, order)

    top_color = order[-1]
    top_rect = rects[top_color]

    out = np.full_like(g, BG)
    for color in order:
        rect = rects[color]
        if color == top_color or not _rectangles_overlap(rect, top_rect):
            _fill_rect(out, rect, color)

    out[:, legend_start:] = g[:, legend_start:]
    return out


def _render_scene(shape: tuple[int, int], colors: list[int], rects: list[Rect]) -> np.ndarray:
    scene = np.full(shape, BG, dtype=int)
    for color, rect in zip(colors, rects, strict=False):
        _fill_rect(scene, rect, color)
    return scene


def _embed_legend(scene: np.ndarray, colors: list[int], sep: int, legend_top: int) -> np.ndarray:
    h, board_w = scene.shape
    total_w = board_w + sep + LEGEND_W
    grid = np.full((h, total_w), BG, dtype=int)
    grid[:, :board_w] = scene
    legend_col = board_w + sep
    for i, color in enumerate(colors):
        r0 = legend_top + i * (LEGEND_H + LEGEND_GAP)
        grid[r0 : r0 + LEGEND_H, legend_col : legend_col + LEGEND_W] = int(color)
    return grid


def _expected_output(
    shape: tuple[int, int],
    colors: list[int],
    rects: list[Rect],
    keep_mask: list[bool],
    sep: int,
    legend_top: int,
) -> np.ndarray:
    kept_scene = np.full(shape, BG, dtype=int)
    for color, rect, keep in zip(colors, rects, keep_mask, strict=False):
        if keep:
            _fill_rect(kept_scene, rect, color)
    return _embed_legend(kept_scene, colors, sep, legend_top)


def _sample_top_rect(h: int, w: int) -> Rect | None:
    max_top_h = min(6, h - 5)
    max_top_w = min(6, w - 5)
    if max_top_h < 3 or max_top_w < 3:
        return None

    for _ in range(TOP_RETRIES):
        rect_h = random.randint(3, max_top_h)
        rect_w = random.randint(3, max_top_w)
        max_r0 = h - rect_h - 2
        max_c0 = w - rect_w - 2
        if max_r0 < 2 or max_c0 < 2:
            continue
        r0 = random.randint(2, max_r0)
        c0 = random.randint(2, max_c0)
        return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1
    return None


def _visible_sides_after_occlusion(rect: Rect, blocker: Rect) -> int:
    r0, c0, r1, c1 = rect
    visible = np.ones((_rect_height(rect), _rect_width(rect)), dtype=bool)
    inter = _intersection(rect, blocker)
    if inter is not None:
        ir0, ic0, ir1, ic1 = inter
        visible[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1] = False
    score = 0
    score += int(visible[0, :].any())
    score += int(visible[-1, :].any())
    score += int(visible[:, 0].any())
    score += int(visible[:, -1].any())
    return score


def _sample_overlap_rect(top: Rect, pattern: str, h: int, w: int) -> Rect | None:
    tr0, tc0, tr1, tc1 = top
    top_h = _rect_height(top)
    top_w = _rect_width(top)

    for _ in range(RECT_RETRIES):
        if pattern in ("nw", "ne"):
            extra_v_max = min(4, tr0)
        else:
            extra_v_max = min(4, h - 1 - tr1)
        if pattern in ("nw", "sw"):
            extra_h_max = min(4, tc0)
        else:
            extra_h_max = min(4, w - 1 - tc1)
        if extra_v_max < 1 or extra_h_max < 1:
            return None

        extra_v = random.randint(1, extra_v_max)
        extra_h = random.randint(1, extra_h_max)
        max_overlap_h = min(top_h - 1, MAX_SIDE - extra_v)
        max_overlap_w = min(top_w - 1, MAX_SIDE - extra_h)
        if max_overlap_h < 1 or max_overlap_w < 1:
            continue
        overlap_h = random.randint(1, max_overlap_h)
        overlap_w = random.randint(1, max_overlap_w)

        if pattern == "nw":
            r0 = tr0 - extra_v
            c0 = tc0 - extra_h
            r1 = tr0 + overlap_h - 1
            c1 = tc0 + overlap_w - 1
        elif pattern == "ne":
            r0 = tr0 - extra_v
            c0 = tc1 - overlap_w + 1
            r1 = tr0 + overlap_h - 1
            c1 = tc1 + extra_h
        elif pattern == "sw":
            r0 = tr1 - overlap_h + 1
            c0 = tc0 - extra_h
            r1 = tr1 + extra_v
            c1 = tc0 + overlap_w - 1
        else:
            assert pattern == "se"
            r0 = tr1 - overlap_h + 1
            c0 = tc1 - overlap_w + 1
            r1 = tr1 + extra_v
            c1 = tc1 + extra_h

        rect = (r0, c0, r1, c1)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if _rect_height(rect) < MIN_SIDE or _rect_width(rect) < MIN_SIDE:
            continue
        if _visible_sides_after_occlusion(rect, top) != 4:
            continue
        if _rect_area(rect) - _rect_area(_intersection(rect, top) or rect) < 4:
            continue
        return rect
    return None


def _sample_disjoint_rect(h: int, w: int, forbidden: list[Rect]) -> Rect | None:
    for _ in range(RECT_RETRIES):
        rect_h = random.randint(MIN_SIDE, min(MAX_SIDE, h))
        rect_w = random.randint(MIN_SIDE, min(MAX_SIDE, w))
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_rectangles_overlap(rect, other) for other in forbidden):
            continue
        return rect
    return None


def _bg_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()]) == BG


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)
    legend_span = n_rects * LEGEND_H + (n_rects - 1) * LEGEND_GAP
    h_min = max(BOARD_H_MIN, legend_span + 1)
    if h_min > BOARD_H_MAX:
        return None

    h = random.randint(h_min, BOARD_H_MAX)
    board_w = random.randint(BOARD_W_MIN, BOARD_W_MAX)
    sep = random.randint(LEGEND_SEP_MIN, LEGEND_SEP_MAX)

    colors = random.sample(range(1, 10), n_rects)
    top_rect = _sample_top_rect(h, board_w)
    if top_rect is None:
        return None

    remove_count = random.randint(1, n_rects - 2)
    keep_count = n_rects - 1 - remove_count

    patterns = random.sample(["nw", "ne", "sw", "se"], remove_count)
    removed_rects: list[Rect] = []
    for pattern in patterns:
        rect = _sample_overlap_rect(top_rect, pattern, h, board_w)
        if rect is None:
            return None
        if any(_rectangles_overlap(rect, other) for other in removed_rects):
            return None
        removed_rects.append(rect)

    keep_rects: list[Rect] = []
    forbidden = [top_rect] + removed_rects
    for _ in range(keep_count):
        rect = _sample_disjoint_rect(h, board_w, forbidden + keep_rects)
        if rect is None:
            return None
        keep_rects.append(rect)

    earlier: list[tuple[Rect, bool]] = [(rect, False) for rect in removed_rects]
    earlier.extend((rect, True) for rect in keep_rects)
    random.shuffle(earlier)

    ordered_rects = [rect for rect, _keep in earlier] + [top_rect]
    keep_mask = [keep for _rect, keep in earlier] + [True]
    ordered_colors = colors

    legend_top = random.randint(0, h - legend_span)
    input_scene = _render_scene((h, board_w), ordered_colors, ordered_rects)
    expected = _expected_output((h, board_w), ordered_colors, ordered_rects, keep_mask, sep, legend_top)
    inp = _embed_legend(input_scene, ordered_colors, sep, legend_top)

    if np.array_equal(inp, expected):
        return None
    if not _bg_is_dominant(inp):
        return None
    if any(np.count_nonzero(input_scene == color) == 0 for color in ordered_colors):
        return None
    if keep_mask[:-1].count(True) == 0 or keep_mask[:-1].count(False) == 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _build_attempt()
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
