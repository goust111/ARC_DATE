from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components


BACKGROUND = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_TO_CW_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_cw(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(patch, k=(-int(quarter_turns)) % 4).copy()


def _dominant_color(patch: np.ndarray) -> int:
    colors = [int(x) for x in patch.flatten() if int(x) not in {BACKGROUND, GREY}]
    counts = Counter(colors)
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "expected a unique dominant template color"
    return int(winners[0])


def _find_template_and_indicators(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    mask = (grid != BACKGROUND) & (grid != GREY)
    labeled = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True, colors=[1], return_format="dict")

    template_bbox = None
    indicators: list[tuple[int, int, int]] = []
    for comp in labeled:
        coords = comp["coords"]
        area = int(comp["area"])
        b = _bbox_from_coords(coords)
        patch = crop_bbox(grid, b)
        colors = {int(x) for x in patch.flatten() if int(x) != BACKGROUND}
        if area == 1:
            rr, cc = map(int, coords[0])
            indicators.append((rr, cc, int(grid[rr, cc])))
            continue
        assert len(colors) >= 2, "template should be multicolor"
        assert template_bbox is None, "expected a unique template object"
        template_bbox = b

    assert template_bbox is not None, "template not found"
    template = crop_bbox(grid, template_bbox)
    return template, indicators


def _indicator_side(indicator: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> str | None:
    r, c = indicator
    r0, c0, r1, c1 = rect_bbox
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    return None


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of the template into grey slots using attached indicator pixels.

    Input structure:
    - A black background contains one connected multicolor template made of non-grey pixels.
    - Several solid grey rectangles appear elsewhere, each with a single non-grey indicator
      pixel attached just outside exactly one side.
    - Indicator colors do not belong to the template and each indicator is a single pixel.

    Rule:
    - Identify the unique non-single-pixel template object and each solid grey target rectangle.
    - Read the side of the attached indicator: top means 0 degrees clockwise, right 90,
      bottom 180, and left 270.
    - Rotate a copy of the template accordingly, replace its most frequent color by the
      indicator color, stamp it over the grey rectangle, and erase the indicator pixel to black.
    - Keep the original template unchanged.

    Key insight:
    - Each indicator encodes two parameters at once: its side determines the template
      rotation and its color determines the recoloring of the template's dominant color.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    template, indicators = _find_template_and_indicators(grid)
    dominant = _dominant_color(template)

    grey_rects = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    assert grey_rects, "expected grey target rectangles"

    out = grid.copy()
    used_indicators: set[tuple[int, int]] = set()

    for rect in grey_rects:
        rect_bbox = rect["bbox"]
        r0, c0, r1, c1 = rect_bbox
        rect_h = bbox_h(rect_bbox)
        rect_w = bbox_w(rect_bbox)
        assert int(rect["area"]) == rect_h * rect_w, "grey targets must be solid rectangles"

        matches = []
        for ir, ic, color in indicators:
            side = _indicator_side((ir, ic), rect_bbox)
            if side is not None:
                matches.append((side, ir, ic, color))
        assert len(matches) == 1, "each grey rectangle should have exactly one indicator"

        side, ir, ic, indicator_color = matches[0]
        rotated = _rotate_cw(template, SIDE_TO_CW_TURNS[side])
        assert rotated.shape == (rect_h, rect_w), "target size must match rotated template"

        recolored = rotated.copy()
        recolored[rotated == dominant] = int(indicator_color)
        out[r0 : r1 + 1, c0 : c1 + 1] = recolored
        out[ir, ic] = BACKGROUND
        used_indicators.add((ir, ic))

    for ir, ic, _color in indicators:
        assert (ir, ic) in used_indicators, "every indicator should belong to a grey rectangle"
    return out


def _choice_middle(length: int) -> int:
    if length % 2 == 1:
        return length // 2
    return random.choice([length // 2 - 1, length // 2])


def _make_template() -> np.ndarray:
    for _ in range(300):
        h, w = random.choice([(3, 4), (4, 3)])
        dominant, accent1, accent2 = random.sample(PALETTE, 3)
        patch = np.full((h, w), dominant, dtype=int)
        cells = [(r, c) for r in range(h) for c in range(w)]
        n_accents = random.randint(3, min(5, h * w - 2))
        accent_cells = random.sample(cells, n_accents)
        third_color = random.random() < 0.5
        for i, (r, c) in enumerate(accent_cells):
            if third_color and i == n_accents - 1:
                patch[r, c] = accent2
            else:
                patch[r, c] = accent1
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        if len({int(x) for x in patch.flatten()}) < 2:
            continue
        return patch
    raise RuntimeError("failed to build asymmetric template")


def _target_rect_shape(template_shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = map(int, template_shape)
    if side in {"top", "bottom"}:
        return h, w
    return w, h


def _footprint_shape(template_shape: tuple[int, int], side: str) -> tuple[int, int]:
    rect_h, rect_w = _target_rect_shape(template_shape, side)
    if side in {"top", "bottom"}:
        return rect_h + 1, rect_w
    return rect_h, rect_w + 1


def _render_target(
    grid: np.ndarray,
    footprint_top_left: tuple[int, int],
    template_shape: tuple[int, int],
    side: str,
    color: int,
) -> None:
    fr, fc = map(int, footprint_top_left)
    rect_h, rect_w = _target_rect_shape(template_shape, side)

    if side == "top":
        rr, cc = fr + 1, fc
        ir, ic = fr, fc + _choice_middle(rect_w)
    elif side == "bottom":
        rr, cc = fr, fc
        ir, ic = fr + rect_h, fc + _choice_middle(rect_w)
    elif side == "left":
        rr, cc = fr, fc + 1
        ir, ic = fr + _choice_middle(rect_h), fc
    else:
        rr, cc = fr, fc
        ir, ic = fr + _choice_middle(rect_h), fc + rect_w

    grid[rr : rr + rect_h, cc : cc + rect_w] = GREY
    grid[ir, ic] = int(color)


def _place_items(
    grid_shape: tuple[int, int],
    item_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    H, W = map(int, grid_shape)
    occupied = np.zeros((H, W), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(item_shapes)
    order = sorted(range(len(item_shapes)), key=lambda i: item_shapes[i][0] * item_shapes[i][1], reverse=True)

    def search(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        h, w = item_shapes[idx]
        candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(candidates)
        for r, c in candidates:
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(H, r + h + 1)
            c1 = min(W, c + w + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r:r + h, c:c + w] = True
            placements[idx] = (r, c)
            if search(i + 1):
                return True
            placements[idx] = None
            occupied[r:r + h, c:c + w] = False
        return False

    if not search(0):
        return None
    return [pos for pos in placements if pos is not None]


def _build_candidate() -> np.ndarray | None:
    template = _make_template()
    template_colors = sorted({int(x) for x in template.flatten()})

    target_count = random.randint(2, 4)
    sides = random.sample(list(SIDE_TO_CW_TURNS), target_count)
    indicator_palette = [color for color in PALETTE if color not in template_colors]
    indicator_colors = random.sample(indicator_palette, target_count)

    item_shapes = [template.shape]
    item_shapes.extend(_footprint_shape(template.shape, side) for side in sides)

    for _ in range(120):
        H = random.randint(16, 24)
        W = random.randint(16, 24)
        placements = _place_items((H, W), item_shapes)
        if placements is None:
            continue

        grid = np.zeros((H, W), dtype=int)
        tr, tc = placements[0]
        grid[tr : tr + template.shape[0], tc : tc + template.shape[1]] = template

        for (fr, fc), side, color in zip(placements[1:], sides, indicator_colors):
            _render_target(grid, (fr, fc), template.shape, side, color)

        try:
            solved = main(grid)
        except Exception:
            continue

        expected = grid.copy()
        dominant = _dominant_color(template)
        for (fr, fc), side, color in zip(placements[1:], sides, indicator_colors):
            rect_h, rect_w = _target_rect_shape(template.shape, side)
            if side == "top":
                rr, cc = fr + 1, fc
                ir, ic = fr, None
            elif side == "bottom":
                rr, cc = fr, fc
                ir, ic = fr + rect_h, None
            elif side == "left":
                rr, cc = fr, fc + 1
                ir, ic = None, fc
            else:
                rr, cc = fr, fc
                ir, ic = None, fc + rect_w

            rotated = _rotate_cw(template, SIDE_TO_CW_TURNS[side])
            recolored = rotated.copy()
            recolored[rotated == dominant] = int(color)
            expected[rr : rr + rect_h, cc : cc + rect_w] = recolored

            if side in {"top", "bottom"}:
                cols = np.where(grid[ir, :] == color)[0]
                assert len(cols) == 1
                expected[ir, int(cols[0])] = BACKGROUND
            else:
                rows = np.where(grid[:, ic] == color)[0]
                assert len(rows) == 1
                expected[int(rows[0]), ic] = BACKGROUND

        if not np.array_equal(solved, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
