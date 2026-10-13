from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    non_bg = grid != bg
    candidates: list[tuple[tuple[int, int], list[int], np.ndarray]] = []

    for comp in _components(non_bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if len(comp) != (r1 - r0 + 1):
            continue
        touches_corner = any(
            (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)} for r, c in comp
        )
        if not touches_corner:
            continue
        if c0 not in {0, w - 1}:
            continue
        if r0 != 0 and r1 != h - 1:
            continue
        palette = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
        mask = np.zeros_like(non_bg, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        distinct = len(set(palette))
        candidates.append(((distinct, len(comp)), palette, mask))

    assert candidates, "expected a vertical palette strip in one corner"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "palette legend must be uniquely identifiable"
    _score, palette, mask = best[0]
    return palette, mask


def _detect_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[BBox]:
    border_colors = sorted(
        set(int(v) for v in np.unique(grid[(grid != bg) & (~legend_mask)]))
    )
    assert len(border_colors) == 1, "expected exactly one frame color outside the legend"
    border_color = border_colors[0]
    frame_mask = (grid == border_color) & (~legend_mask)

    frames: list[BBox] = []
    for comp in _components(frame_mask):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert hh >= 3 and ww >= 3, "frame rectangles must enclose an interior"
        expected_area = 2 * (hh + ww) - 4
        assert len(comp) == expected_area, "frame component must be a 1-cell-thick rectangle"

        local = np.zeros((hh, ww), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        assert np.all(local[0, :]) and np.all(local[-1, :])
        assert np.all(local[:, 0]) and np.all(local[:, -1])
        assert not np.any(local[1:-1, 1:-1]), "frame interior must stay empty in its component"
        frames.append((r0, c0, r1, c1))

    assert frames, "expected at least one frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background region with the legend color indexed by its nesting depth.

    Input structure:
    - Background fills most of the grid.
    - One corner contains a vertical 1-pixel-wide palette strip.
    - Elsewhere, one border color draws one or more hollow rectangular frames,
      possibly nested to different depths.

    Rule:
    - Read the corner strip from top to bottom as colors c1, c2, ...
    - Every background region surrounded by exactly d frames is recolored to cd.
    - The outside background (depth 0) and the frame pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    palette, legend_mask = _detect_legend(g, bg)
    frames = _detect_frames(g, bg, legend_mask)

    out = g.copy()
    bg_mask = (g == bg) & (~legend_mask)
    for comp in _components(bg_mask):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in comp):
            continue
        sample_r, sample_c = comp[0]
        depth = sum(r0 < sample_r < r1 and c0 < sample_c < c1 for r0, c0, r1, c1 in frames)
        if depth == 0:
            continue
        assert depth <= len(palette), "legend must provide a color for every nesting depth"
        fill = int(palette[depth - 1])
        for r, c in comp:
            out[r, c] = fill

    return out


# --------------- generator ---------------

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
MAX_ATTEMPTS = 400

DEPTH_CHOICES = (1, 2, 3, 4)
FAMILY_COUNT_CHOICES = (1, 2, 3)
INNER_SIZE_RANGE = (3, 5)
EXPAND_RANGE = (2, 3)
EDGE_MARGIN = 2
FAMILY_PLACEMENT_RETRIES = 200
SEPARATION_PAD = 1


def _boxes_for_family(depth: int) -> list[BBox]:
    inner_h = random.randint(*INNER_SIZE_RANGE)
    inner_w = random.randint(*INNER_SIZE_RANGE)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = boxes[-1]
        expand_top = random.randint(*EXPAND_RANGE)
        expand_bottom = random.randint(*EXPAND_RANGE)
        expand_left = random.randint(*EXPAND_RANGE)
        expand_right = random.randint(*EXPAND_RANGE)
        boxes.append(
            (
                r0 - expand_top,
                c0 - expand_left,
                r1 + expand_bottom,
                c1 + expand_right,
            )
        )

    outer = boxes[-1]
    shift_r = -outer[0]
    shift_c = -outer[1]
    shifted = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in boxes
    ]
    return shifted


def _shift_boxes(boxes: list[BBox], dr: int, dc: int) -> list[BBox]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _frame_perimeter_cells(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    for r, c in _frame_perimeter_cells(box):
        grid[r, c] = color


def _expanded(box: BBox, pad: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _intersects(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_legend(grid: np.ndarray, palette: list[int], corner: str) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros(grid.shape, dtype=bool)
    if corner == "tl":
        rows = range(0, len(palette))
        col = 0
    elif corner == "tr":
        rows = range(0, len(palette))
        col = w - 1
    elif corner == "bl":
        rows = range(h - len(palette), h)
        col = 0
    else:
        rows = range(h - len(palette), h)
        col = w - 1

    for r, color in zip(rows, palette, strict=False):
        grid[r, col] = int(color)
        mask[r, col] = True
    return mask


def _expected_output(grid: np.ndarray, bg: int, palette: list[int], legend_mask: np.ndarray, frames: list[BBox]) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if legend_mask[r, c] or int(grid[r, c]) != bg:
                continue
            depth = sum(r0 < r < r1 and c0 < c < c1 for r0, c0, r1, c1 in frames)
            if depth > 0:
                out[r, c] = int(palette[depth - 1])
    return out


def _attempt_generate() -> np.ndarray | None:
    bg = random.randint(0, 9)
    border = random.choice([c for c in range(10) if c != bg])
    depth = random.choice(DEPTH_CHOICES)
    palette = random.sample([c for c in range(10) if c not in {bg, border}], depth)
    family_count = random.choice(FAMILY_COUNT_CHOICES)
    corner = random.choice(("tl", "tr", "bl", "br"))

    family_depths = [depth] + [random.randint(1, depth) for _ in range(family_count - 1)]
    random.shuffle(family_depths)
    family_templates = [_boxes_for_family(d) for d in family_depths]

    max_h = max(boxes[-1][2] + 1 for boxes in family_templates)
    max_w = max(boxes[-1][3] + 1 for boxes in family_templates)
    min_side = max(GRID_SIDE_MIN, max_h + 2 * EDGE_MARGIN, max_w + 2 * EDGE_MARGIN)
    if min_side > GRID_SIDE_MAX:
        return None

    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)
    grid = np.full((h, w), bg, dtype=int)
    legend_mask = _place_legend(grid, palette, corner)

    placed_outer: list[BBox] = []
    all_frames: list[BBox] = []
    for boxes in family_templates:
        outer = boxes[-1]
        family_h = outer[2] + 1
        family_w = outer[3] + 1
        placed = False
        for _ in range(FAMILY_PLACEMENT_RETRIES):
            r0 = random.randint(EDGE_MARGIN, h - family_h - EDGE_MARGIN)
            c0 = random.randint(EDGE_MARGIN, w - family_w - EDGE_MARGIN)
            shifted_outer = (r0, c0, r0 + family_h - 1, c0 + family_w - 1)
            if any(
                _intersects(_expanded(shifted_outer, SEPARATION_PAD), _expanded(other, SEPARATION_PAD))
                for other in placed_outer
            ):
                continue
            shifted_boxes = _shift_boxes(boxes, r0, c0)
            for box in shifted_boxes:
                _draw_frame(grid, box, border)
            placed_outer.append(shifted_outer)
            all_frames.extend(shifted_boxes)
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_output(grid, bg, palette, legend_mask, all_frames)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
