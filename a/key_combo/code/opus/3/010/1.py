from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, bbox, bbox_h, bbox_w, colors_present, crop_bbox


RECT_COUNT_CHOICES = (4, 5)
SPAN_FAMILIES = (
    (2, 4, 6, 8, 10),
    (3, 5, 7, 9, 11),
)
MARGIN_RANGE = (1, 3)
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _ring_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _contains_interior(box: tuple[int, int, int, int], rc: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    r, c = map(int, rc)
    return r0 < r < r1 and c0 < c < c1


def _visible_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in colors_present(grid, exclude={Color.BLACK}):
        mask = grid == int(color)
        box = bbox(mask)
        assert box is not None, "every foreground color must define one rectangle"
        h = bbox_h(box)
        w = bbox_w(box)
        assert h >= 5 and w >= 5, "rectangles must have non-empty interiors"

        patch = crop_bbox(mask.astype(np.uint8), box).astype(bool)
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(~patch | border), "visible pixels of one color must lie on a single rectangle border"

        rects.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, box)),
                "ring": _ring_mask(grid.shape, box),
                "interior": _interior_mask(grid.shape, box),
            }
        )
    assert len(rects) >= 2, "expected multiple hollow rectangles"
    return rects


def _infer_order(grid: np.ndarray, rects: list[dict]) -> list[dict]:
    beats = {rect["color"]: 0 for rect in rects}
    for idx, rect_a in enumerate(rects):
        for rect_b in rects[idx + 1 :]:
            overlap = rect_a["ring"] & rect_b["ring"]
            coords = np.argwhere(overlap)
            assert len(coords) > 0, "every pair of rectangles must cross"

            visible = {int(grid[int(r), int(c)]) for r, c in coords}
            pair_colors = {int(rect_a["color"]), int(rect_b["color"])}
            assert visible <= pair_colors, "crossings must only show one of the two participating rectangles"
            assert len(visible) == 1, "all crossings of one pair must agree on the same top rectangle"
            beats[next(iter(visible))] += 1

    n = len(rects)
    expected_scores = list(range(n))
    scores = sorted(beats.values())
    assert scores == expected_scores, "crossings must determine a unique total stacking order"
    return sorted(rects, key=lambda rect: beats[rect["color"]])


def _render_output(grid: np.ndarray, order: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    base_bg = int(Color.BLACK)
    input_bg_mask = grid == base_bg
    for level, rect in enumerate(order):
        fill_color = base_bg if level == 0 else int(order[level - 1]["color"])
        if fill_color == base_bg:
            continue
        fill_mask = rect["interior"] & input_bg_mask
        out[fill_mask] = fill_color
    return out


def _unique_corner_pixels(grid: np.ndarray, rect: dict, rects: list[dict]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect["bbox"])
    candidates = [
        (r0 + 1, c0 + 1),
        (r0 + 1, c1 - 1),
        (r1 - 1, c0 + 1),
        (r1 - 1, c1 - 1),
    ]
    unique: list[tuple[int, int]] = []
    for rc in candidates:
        r, c = rc
        if int(grid[r, c]) != int(Color.BLACK):
            continue
        if sum(_contains_interior(other["bbox"], rc) for other in rects) == 1:
            unique.append(rc)
    return unique


def _validate_instance(grid: np.ndarray) -> None:
    rects = _visible_rectangles(grid)
    assert len(rects) in RECT_COUNT_CHOICES, "generator must emit 4 or 5 rectangles"

    cover = np.zeros(grid.shape, dtype=int)
    for rect in rects:
        cover += rect["ring"].astype(int)
    assert int(cover.max()) <= 2, "avoid triple border overlaps so pairwise order stays readable"

    order = _infer_order(grid, rects)
    out = _render_output(grid, order)
    assert not np.array_equal(out, grid), "the task must actually require filling"

    for level, rect in enumerate(order):
        unique_pixels = _unique_corner_pixels(grid, rect, rects)
        assert unique_pixels, "each rectangle must keep at least one interior witness pixel"
        expected = int(Color.BLACK) if level == 0 else int(order[level - 1]["color"])
        assert any(int(out[r, c]) == expected for r, c in unique_pixels), "each layer's fill rule must be visible"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the stacking order of intersecting hollow rectangles and fill each interior with the next lower outline color.

    Input structure:
    - A black background contains 4-5 axis-aligned hollow rectangles with distinct outline colors.
    - Rectangle borders are one pixel thick; lower borders may be hidden at crossings, so each visible color can have holes.
    - Every pair of rectangles crosses, and each crossing shows only the color of the higher rectangle at that point.

    Rule:
    - Reconstruct each rectangle from the bounding box of its visible border color.
    - Compare every pair of reconstructed borders at their crossing pixels; the visible color there identifies which rectangle is higher.
    - Sort the rectangles from bottom to top using those pairwise comparisons.
    - Keep all visible outline pixels unchanged, and fill each rectangle's black interior with the outline color of the rectangle directly below it in the stack; the bottom rectangle stays black.
    - When interiors overlap, the higher rectangle's fill wins because it is painted later in stack order.

    Key insight:
    - Even with hidden crossing pixels, each color still reveals one hollow rectangle bbox, and the crossing colors form a complete pairwise ordering.
    - The fill color is not intrinsic to a rectangle: it is defined by its immediate lower neighbor in the inferred stack.
    """
    g = np.asarray(grid, dtype=int)
    rects = _visible_rectangles(g)
    order = _infer_order(g, rects)
    return _render_output(g, order)


def _build_rectangles(n_rects: int) -> list[dict]:
    v_pool = random.choice(SPAN_FAMILIES)
    h_pool = random.choice(SPAN_FAMILIES)
    v_spans = sorted(random.sample(v_pool, n_rects), reverse=True)
    h_spans = sorted(random.sample(h_pool, n_rects))
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    center_r = top_margin + v_spans[0]
    center_c = left_margin + h_spans[-1]
    height = center_r + v_spans[0] + bottom_margin + 1
    width = center_c + h_spans[-1] + right_margin + 1

    colors = random.sample(PALETTE, n_rects)
    rects: list[dict] = []
    for idx in range(n_rects):
        rects.append(
            {
                "color": int(colors[idx]),
                "bbox": (
                    int(center_r - v_spans[idx]),
                    int(center_c - h_spans[idx]),
                    int(center_r + v_spans[idx]),
                    int(center_c + h_spans[idx]),
                ),
            }
        )

    assert max(height, width) <= 30
    return rects


def _draw_input(rects: list[dict]) -> np.ndarray:
    max_r = max(rect["bbox"][2] for rect in rects) + 1
    max_c = max(rect["bbox"][3] for rect in rects) + 1
    grid = np.zeros((max_r, max_c), dtype=int)

    order = rects[:]
    random.shuffle(order)
    for rect in order:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        color = int(rect["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.choice(RECT_COUNT_CHOICES)
    rects = _build_rectangles(n_rects)
    grid = _draw_input(rects)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GEN_ATTEMPTS} attempts")
