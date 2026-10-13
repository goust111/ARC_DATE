from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, find_components, get_bg, paste


GEN_SIDE_OPTIONS = (4,)
GEN_TEMPLATE_AREA_RANGE = (4, 7)
GEN_EXTRA_PIXEL_RANGE = (0, 3)
GEN_MAX_ATTEMPTS = 400
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    if w == 2 * h:
        return grid[:, :h].copy(), grid[:, h:].copy()
    if h == 2 * w:
        return grid[:w, :].copy(), grid[w:, :].copy()
    raise AssertionError("expected two equal square panels arranged side-by-side or top-to-bottom")


def _panel_info(panel: np.ndarray, bg: int) -> dict:
    fg_mask = panel != bg
    fg_box = bbox(fg_mask)
    assert fg_box is not None, "panel must contain foreground"
    comps = find_components(panel, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    return {
        "colors": colors_present(panel, exclude={bg}),
        "area": int(np.count_nonzero(fg_mask)),
        "bbox": fg_box,
        "max_component_area": max(int(comp["area"]) for comp in comps),
        "lr_symmetric": np.array_equal(fg_mask, np.fliplr(fg_mask)),
        "ud_symmetric": np.array_equal(fg_mask, np.flipud(fg_mask)),
    }


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    panel_a, panel_b = _split_panels(g)
    assert panel_a.shape == panel_b.shape
    n = int(panel_a.shape[0])
    assert panel_a.shape[1] == n and n % 2 == 0

    infos = [_panel_info(panel_a, bg), _panel_info(panel_b, bg)]
    template_candidates = [
        idx
        for idx, info in enumerate(infos)
        if len(info["colors"]) == 1
        and info["area"] >= 2
        and not info["lr_symmetric"]
        and not info["ud_symmetric"]
    ]
    if len(template_candidates) != 1:
        template_candidates = [
            idx for idx, info in enumerate(infos) if len(info["colors"]) == 1 and info["area"] >= 2
        ]
    assert len(template_candidates) == 1, "template panel must be uniquely identifiable"

    template_idx = template_candidates[0]
    source_idx = 1 - template_idx
    source_panel = (panel_a, panel_b)[source_idx]
    template_panel = (panel_a, panel_b)[template_idx]
    assert len(colors_present(source_panel, exclude={bg})) >= 1
    return source_panel, template_panel, n, bg


def _oriented_mask(template_mask: np.ndarray, r: int, c: int, half: int) -> np.ndarray:
    stamp = template_mask
    if c >= half:
        stamp = np.fliplr(stamp)
    if r >= half:
        stamp = np.flipud(stamp)
    return stamp


def _render_output(
    source_panel: np.ndarray,
    template_mask: np.ndarray,
    bg: int,
    *,
    use_quadrant_orientation: bool,
) -> np.ndarray:
    n = int(source_panel.shape[0])
    out = np.full((n * n, n * n), bg, dtype=int)
    half = n // 2

    for r, c in np.argwhere(source_panel != bg):
        color = int(source_panel[r, c])
        stamp = (
            _oriented_mask(template_mask, int(r), int(c), half)
            if use_quadrant_orientation
            else template_mask
        )
        block = out[int(r) * n : (int(r) + 1) * n, int(c) * n : (int(c) + 1) * n]
        block[stamp] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored source pixel into a template stamp whose flip depends on the output quadrant.

    Input structure:
    - The input contains two equal square panels, placed either side-by-side or top-to-bottom.
    - One panel is a sparse source panel with colored foreground pixels on a shared background.
    - The other panel is a template panel whose foreground uses one marker color and is not mirror-symmetric in either axis.

    Rule:
    - Split the input into the two square panels and identify the single-color one as the template panel.
    - Create an N^2 by N^2 output, viewed as an N by N grid of N by N blocks.
    - For every non-background source pixel at position (r, c), paint the template mask into block (r, c) using that pixel's color.
    - The stamp orientation depends on the block quadrant: upper-left keeps the template, upper-right flips left-right, lower-left flips up-down, and lower-right flips both axes.

    Key insight:
    - The output is not made by repeating one fixed stamp: every stamped copy is mirrored according to its own destination quadrant.
    """
    source_panel, template_panel, _n, bg = _extract_panels(grid)
    template_mask = template_panel != bg
    return _render_output(source_panel, template_mask, bg, use_quadrant_orientation=True)


def _grow_connected_mask(side: int, target_area: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    start = (random.randrange(side), random.randrange(side))
    mask[start] = True

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in FOUR_NEIGHBORS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < side and 0 <= cc < side and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    return mask


def _build_template_panel(side: int, bg: int, marker: int) -> np.ndarray:
    for _ in range(120):
        target_area = random.randint(*GEN_TEMPLATE_AREA_RANGE)
        raw_mask = _grow_connected_mask(side, target_area)
        raw_box = bbox(raw_mask)
        assert raw_box is not None
        patch = crop_bbox(raw_mask.astype(int), raw_box).astype(bool)
        if min(map(int, patch.shape)) < 2:
            continue

        canvas = np.zeros((side, side), dtype=int)
        r0 = random.randint(0, side - int(patch.shape[0]))
        c0 = random.randint(0, side - int(patch.shape[1]))
        paste(canvas, patch.astype(int), r0, c0)
        mask = canvas.astype(bool)

        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue

        panel = np.full((side, side), bg, dtype=int)
        panel[mask] = marker
        return panel

    raise RuntimeError("failed to build an asymmetric template panel")


def _quadrant_cells(side: int) -> list[list[tuple[int, int]]]:
    half = side // 2
    return [
        [(r, c) for r in range(0, half) for c in range(0, half)],
        [(r, c) for r in range(0, half) for c in range(half, side)],
        [(r, c) for r in range(half, side) for c in range(0, half)],
        [(r, c) for r in range(half, side) for c in range(half, side)],
    ]


def _build_source_panel(side: int, bg: int, source_colors: list[int]) -> np.ndarray:
    panel = np.full((side, side), bg, dtype=int)
    used_positions: set[tuple[int, int]] = set()

    for color, cells in zip(source_colors[:4], _quadrant_cells(side)):
        pos = random.choice(cells)
        used_positions.add(pos)
        panel[pos] = color

    extra_count = random.randint(*GEN_EXTRA_PIXEL_RANGE)
    free_positions = [(r, c) for r in range(side) for c in range(side) if (r, c) not in used_positions]
    random.shuffle(free_positions)
    palette = source_colors[:]
    for pos in free_positions[:extra_count]:
        panel[pos] = random.choice(palette)

    return panel


def _combine_panels(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        return np.concatenate((first, second), axis=1)
    return np.concatenate((first, second), axis=0)


def _semantic_checks(grid: np.ndarray) -> None:
    source_panel, template_panel, side, bg = _extract_panels(grid)
    template_mask = template_panel != bg
    source_colors = colors_present(source_panel, exclude={bg})

    assert side * side <= 30
    assert len(source_colors) >= 2
    assert len(colors_present(template_panel, exclude={bg})) == 1
    assert not np.array_equal(template_mask, np.fliplr(template_mask))
    assert not np.array_equal(template_mask, np.flipud(template_mask))
    assert not np.array_equal(source_panel, np.fliplr(source_panel))
    assert not np.array_equal(source_panel, np.flipud(source_panel))

    half = side // 2
    quadrants = (
        source_panel[:half, :half],
        source_panel[:half, half:],
        source_panel[half:, :half],
        source_panel[half:, half:],
    )
    assert all(np.any(block != bg) for block in quadrants)

    correct = main(grid)
    naive = _render_output(source_panel, template_mask, bg, use_quadrant_orientation=False)
    assert not np.array_equal(correct, naive), "quadrant-specific flips must matter"


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(GEN_SIDE_OPTIONS)
    palette = random.sample(Color.ALL_COLORS, 6)
    bg, marker = palette[0], palette[1]
    source_colors = palette[2:]

    source_panel = _build_source_panel(side, bg, source_colors)
    template_panel = _build_template_panel(side, bg, marker)

    panels = [source_panel, template_panel]
    if random.random() < 0.5:
        panels.reverse()
    grid = _combine_panels(panels[0], panels[1])

    try:
        _semantic_checks(grid)
        out = main(grid)
    except Exception:
        return None

    if out.shape != (side * side, side * side):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-stamp input")
