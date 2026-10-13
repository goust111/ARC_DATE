from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
ALL_COLORS = tuple(range(10))

GEN_MIN_RECTS = 3
GEN_MAX_RECTS = 5
GEN_MIN_H = 18
GEN_MAX_H = 28
GEN_MIN_W = 18
GEN_MAX_W = 28
GEN_RECT_H_CHOICES = (4, 5, 6)
GEN_RECT_W_CHOICES = (4, 5, 6, 7)
GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_multicolor_components(grid: np.ndarray, bg: int) -> list[dict]:
    grid = np.asarray(grid, dtype=int)
    comps = []
    for coords in components_mask(grid != int(bg)):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _is_corner_legend(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    return (
        c0 == c1
        and comp["area"] == (r1 - r0 + 1)
        and (c0 == 0 or c1 == w - 1)
        and (r0 == 0 or r1 == h - 1)
    )


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], dict, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    comps = find_multicolor_components(g, bg)
    assert len(comps) >= 4, "expected several foreground components"

    legend_candidates = [comp for comp in comps if _is_corner_legend(comp, g.shape)]
    assert len(legend_candidates) == 1, "expected exactly one corner color legend"
    legend = legend_candidates[0]

    rectangles = [comp for comp in comps if comp is not legend]
    assert len(rectangles) >= 3, "expected at least three rectangles"

    rect_palette: set[int] = set()
    for rect in rectangles:
        bbox = rect["bbox"]
        assert rect["area"] == bbox_area(bbox), "expected each foreground object to be a solid rectangle"
        patch = crop_bbox(g, bbox)
        assert np.all(patch != bg), "rectangle bbox must be fully occupied"
        cols = set(map(int, patch.ravel()))
        assert 1 <= len(cols) <= 2, "rectangle must use one fill color and optional marker color"
        rect_palette |= cols

    assert len(rect_palette) == 2, "rectangles must jointly use exactly fill color and marker color"
    totals = {color: int(np.count_nonzero(np.isin(g, [color]) & _rectangles_mask(g.shape, rectangles, g, bg))) for color in rect_palette}
    fill_color = max(totals, key=totals.get)
    marker_color = min(totals, key=totals.get)
    assert fill_color != marker_color and totals[fill_color] > totals[marker_color], "failed to infer fill/marker colors"

    legend_col = legend["bbox"][1]
    legend_colors = [int(x) for x in g[legend["bbox"][0] : legend["bbox"][2] + 1, legend_col]]
    assert legend_colors.count(marker_color) == 1, "legend must contain exactly one marker-colored anchor"
    assert fill_color not in legend_colors, "legend colors must not include rectangle fill color"
    non_anchor = [color for color in legend_colors if color != marker_color]
    assert len(non_anchor) == len(rectangles), "legend must contain one non-anchor color per rectangle"
    assert len(set(non_anchor)) == len(non_anchor), "legend colors must be distinct"

    for rect in rectangles:
        patch = crop_bbox(g, rect["bbox"])
        cols = set(map(int, patch.ravel()))
        assert marker_color in cols, "every rectangle must contain marker pixels"
        assert cols <= {fill_color, marker_color}, "unexpected rectangle colors"
        rect["marker_count"] = int(np.count_nonzero(patch == marker_color))
        assert rect["marker_count"] > 0, "marker counts must be positive"

    counts = [rect["marker_count"] for rect in rectangles]
    assert len(set(counts)) == len(counts), "marker counts must be unique"
    return bg, rectangles, legend, marker_color, legend_colors


def _rectangles_mask(
    shape: tuple[int, int], rectangles: list[dict], grid: np.ndarray, bg: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _legend_read_sequence(legend_colors: list[int], anchor_color: int) -> list[int]:
    anchor_idx = legend_colors.index(int(anchor_color))
    return legend_colors[anchor_idx + 1 :] + legend_colors[:anchor_idx]


def _paint_ranked_rectangles(
    grid: np.ndarray, rectangles: list[dict], ordered_colors: list[int], order_key
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ranked = sorted(rectangles, key=order_key)
    assert len(ranked) == len(ordered_colors)
    for rect, color in zip(ranked, ordered_colors):
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked rectangle by ranking marker counts and cycling through the legend from the marker-colored anchor.

    Input structure:
    - A black background contains several separated solid rectangles sharing one fill color.
    - Inside every rectangle, sparse pixels of one marker color appear, and all rectangles have different marker counts.
    - A one-column color legend sits in a corner; exactly one legend cell has the marker color as an anchor, while the other legend colors are distinct.

    Rule:
    - Count the marker-color pixels inside each rectangle and rank the rectangles from fewest to most.
    - Read legend colors starting from the cell immediately below the marker-colored anchor, wrapping from bottom to top and skipping the anchor itself.
    - Assign the first read color to the rectangle with the fewest markers, the second read color to the next rectangle, and so on.
    - Recolor each rectangle, including its internal marker pixels, to its assigned color; keep the legend and background unchanged.

    Key insight:
    - The marker color has a dual role: inside rectangles it is the counting unit, while inside the legend it is the anchor that rotates the color order used for rank lookup.
    """
    bg, rectangles, _legend, marker_color, legend_colors = _parse_scene(grid)
    assert bg == BG, "expected black background"
    ordered_colors = _legend_read_sequence(legend_colors, marker_color)
    return _paint_ranked_rectangles(grid, rectangles, ordered_colors, lambda rect: rect["marker_count"])


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + 1 <= br or br + bh + 1 <= ar or ac + aw + 1 <= bc or bc + bw + 1 <= ac


def _place_rectangles(
    h: int,
    w: int,
    n_rects: int,
    legend_side: str,
) -> list[tuple[int, int, int, int]] | None:
    rect_specs = [(random.choice(GEN_RECT_H_CHOICES), random.choice(GEN_RECT_W_CHOICES)) for _ in range(n_rects)]
    rect_specs.sort(key=lambda x: x[0] * x[1], reverse=True)
    placements: list[tuple[int, int, int, int]] = []

    c_min = 2 if legend_side == "left" else 0
    c_max_exclusive = w if legend_side == "left" else w - 2
    if c_max_exclusive - c_min < min(GEN_RECT_W_CHOICES):
        return None

    for rh, rw in rect_specs:
        candidates = [
            (r, c, rh, rw)
            for r in range(0, h - rh + 1)
            for c in range(c_min, c_max_exclusive - rw + 1)
            if all(_separated((r, c, rh, rw), placed) for placed in placements)
        ]
        if not candidates:
            return None
        placements.append(random.choice(candidates))
    return placements


def _choose_counts(placements: list[tuple[int, int, int, int]]) -> list[int] | None:
    n_rects = len(placements)
    max_distinct = min((rh * rw) // 3 for _r, _c, rh, rw in placements)
    max_count = min(8, max_distinct)
    if max_count <= n_rects:
        return None

    counts = sorted(random.sample(range(1, max_count + 1), n_rects))
    if counts == list(range(1, n_rects + 1)):
        return None
    return counts


def _legend_cells(rank_colors: list[int], marker_color: int) -> list[int]:
    n = len(rank_colors)
    anchor_idx = random.randint(1, n - 1)
    top_down = rank_colors[anchor_idx:] + [marker_color] + rank_colors[:anchor_idx]
    assert _legend_read_sequence(top_down, marker_color) == rank_colors
    return top_down


def _paint_markers(
    grid: np.ndarray, rect: tuple[int, int, int, int], marker_color: int, marker_count: int
) -> None:
    r, c, rh, rw = rect
    cells = [(rr, cc) for rr in range(r, r + rh) for cc in range(c, c + rw)]
    for rr, cc in random.sample(cells, marker_count):
        grid[rr, cc] = int(marker_color)


def _render_input(
    h: int,
    w: int,
    placements: list[tuple[int, int, int, int]],
    fill_color: int,
    marker_color: int,
    counts_by_rect: list[int],
    legend_colors: list[int],
    legend_side: str,
    legend_vertical: str,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rect, marker_count in zip(placements, counts_by_rect):
        r, c, rh, rw = rect
        grid[r : r + rh, c : c + rw] = int(fill_color)
        _paint_markers(grid, rect, marker_color, marker_count)

    legend_len = len(legend_colors)
    legend_rows = range(0, legend_len) if legend_vertical == "top" else range(h - legend_len, h)
    col = 0 if legend_side == "left" else w - 1
    for rr, color in zip(legend_rows, legend_colors):
        grid[rr, col] = int(color)
    return grid


def _rect_key_rowmajor(rect: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = rect["bbox"]
    return int(r0), int(c0)


def _rect_key_area(rect: dict) -> tuple[int, int, int]:
    area = bbox_area(rect["bbox"])
    r0, c0, _r1, _c1 = rect["bbox"]
    return area, int(r0), int(c0)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        _bg, rectangles, _legend, marker_color, legend_colors = _parse_scene(inp)
    except AssertionError:
        return False

    correct_colors = _legend_read_sequence(legend_colors, marker_color)
    if legend_colors.index(marker_color) in {0, len(legend_colors) - 1}:
        return False

    counts_sorted = sorted(rect["marker_count"] for rect in rectangles)
    if counts_sorted == list(range(1, len(rectangles) + 1)):
        return False

    top_down_colors = [color for color in legend_colors if color != marker_color]
    wrong_topstart = _paint_ranked_rectangles(inp, rectangles, top_down_colors, lambda rect: rect["marker_count"])
    if np.array_equal(wrong_topstart, expected):
        return False

    rowmajor = _paint_ranked_rectangles(inp, rectangles, correct_colors, _rect_key_rowmajor)
    if np.array_equal(rowmajor, expected):
        return False

    areas = [bbox_area(rect["bbox"]) for rect in rectangles]
    if len(set(areas)) > 1:
        by_area = _paint_ranked_rectangles(inp, rectangles, correct_colors, _rect_key_area)
        if np.array_equal(by_area, expected):
            return False

    raw_counts = [rect["marker_count"] for rect in rectangles]
    if max(raw_counts) <= len(rectangles):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    n_rects = random.randint(GEN_MIN_RECTS, GEN_MAX_RECTS)
    h = random.randint(max(GEN_MIN_H, n_rects + 8), GEN_MAX_H)
    w = random.randint(GEN_MIN_W, GEN_MAX_W)
    legend_side = random.choice(("left", "right"))
    legend_vertical = random.choice(("top", "bottom"))

    placements = _place_rectangles(h, w, n_rects, legend_side)
    if placements is None:
        return None

    sorted_counts = _choose_counts(placements)
    if sorted_counts is None:
        return None

    order = list(range(n_rects))
    random.shuffle(order)
    if order == list(range(n_rects)):
        return None
    counts_by_rect = [sorted_counts[idx] for idx in order]

    fill_color, marker_color = random.sample(ALL_COLORS[1:], 2)
    rank_colors = random.sample([c for c in ALL_COLORS[1:] if c not in {fill_color, marker_color}], n_rects)
    legend_colors = _legend_cells(rank_colors, marker_color)

    grid = _render_input(
        h,
        w,
        placements,
        fill_color,
        marker_color,
        counts_by_rect,
        legend_colors,
        legend_side,
        legend_vertical,
    )

    try:
        expected = main(grid)
    except AssertionError:
        return None
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
