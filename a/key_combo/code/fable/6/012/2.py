from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GRID_SIZE = 30
MAX_OUTPUT_SIZE = MAX_GRID_SIZE
GEN_MAX_ATTEMPTS = 1000


def _get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    height, width = map(int, grid.shape)
    candidates: list[int] = []
    for c in range(width):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    sep_col = _find_separator_col(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size > 0 and right.size > 0, "separator must split the grid into two panels"
    return left.copy(), right.copy(), int(bg)


def _extract_color_and_hole_mask(panel: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    colors = sorted(int(v) for v in np.unique(panel) if int(v) != bg)
    assert len(colors) == 1, "each panel must contain exactly one rectangle color"
    color = int(colors[0])

    rect_mask = panel == color
    rect_box = _bbox(rect_mask)
    assert rect_box is not None, "panel rectangle must be non-empty"
    r0, c0, r1, c1 = rect_box
    rect_crop = panel[r0 : r1 + 1, c0 : c1 + 1]

    hole_mask = rect_crop == bg
    hole_box = _bbox(hole_mask)
    assert hole_box is not None, "rectangle must contain at least one hole"
    hr0, hc0, hr1, hc1 = hole_box
    return color, hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1].copy()


def _compose_output(outer_mask: np.ndarray, inner_mask: np.ndarray, base_color: int, hole_color: int) -> np.ndarray:
    outer_h, outer_w = map(int, outer_mask.shape)
    inner_h, inner_w = map(int, inner_mask.shape)
    out_h = outer_h * inner_h
    out_w = outer_w * inner_w
    assert out_h <= MAX_OUTPUT_SIZE and out_w <= MAX_OUTPUT_SIZE, "output exceeds ARC limits"

    out = np.full((out_h, out_w), int(base_color), dtype=int)
    tile = np.where(inner_mask, int(hole_color), int(base_color)).astype(int)
    for r, c in np.argwhere(outer_mask):
        rr = int(r) * inner_h
        cc = int(c) * inner_w
        out[rr : rr + inner_h, cc : cc + inner_w] = tile
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Build the Kronecker-style output from the hidden hole masks of two rectangles.

    The input contains two panels separated by a full-height vertical line. Each panel
    holds one solid rectangle with background-colored holes cut out of it. The left
    hole mask provides the outer stencil and the right hole mask provides the inner
    tile. The rectangle colors become the output base and foreground colors.
    """
    left, right, bg = _split_panels(np.asarray(input_grid, dtype=int))
    base_color, outer_mask = _extract_color_and_hole_mask(left, bg)
    hole_color, inner_mask = _extract_color_and_hole_mask(right, bg)
    return _compose_output(outer_mask, inner_mask, base_color, hole_color)


def _random_hole_mask(height: int, width: int) -> np.ndarray:
    assert height >= 2 and width >= 2
    mask = np.zeros((height, width), dtype=bool)

    # Force the bbox to be the full requested size while keeping the shape varied.
    mask[0, random.randrange(width)] = True
    mask[height - 1, random.randrange(width)] = True
    mask[random.randrange(height), 0] = True
    mask[random.randrange(height), width - 1] = True

    density = random.uniform(0.25, 0.75)
    for r in range(height):
        for c in range(width):
            if random.random() < density:
                mask[r, c] = True

    if not np.any(~mask):
        mask[random.randrange(height), random.randrange(width)] = False
    return mask


def _embed_rectangle(
    panel: np.ndarray,
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    color: int,
    bg: int,
    hole_top: int,
    hole_left: int,
    hole_mask: np.ndarray,
) -> None:
    panel[top : top + rect_h, left : left + rect_w] = int(color)
    h, w = map(int, hole_mask.shape)
    hole_slice = panel[
        top + hole_top : top + hole_top + h,
        left + hole_left : left + hole_left + w,
    ]
    hole_slice[hole_mask] = int(bg)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_MAX_ATTEMPTS):
        outer_h = random.randint(2, 5)
        outer_w = random.randint(2, 5)
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 5)
        if outer_h * inner_h > MAX_OUTPUT_SIZE or outer_w * inner_w > MAX_OUTPUT_SIZE:
            continue

        outer_mask = _random_hole_mask(outer_h, outer_w)
        inner_mask = _random_hole_mask(inner_h, inner_w)

        bg, sep, base_color, hole_color = random.sample(range(10), 4)

        outer_pad = (
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
        )
        inner_pad = (
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
            random.randint(1, 3),
        )
        left_rect_h = outer_h + outer_pad[0] + outer_pad[1]
        left_rect_w = outer_w + outer_pad[2] + outer_pad[3]
        right_rect_h = inner_h + inner_pad[0] + inner_pad[1]
        right_rect_w = inner_w + inner_pad[2] + inner_pad[3]

        panel_h = max(left_rect_h, right_rect_h) + random.randint(2, 6)
        left_panel_w = left_rect_w + random.randint(2, 6)
        right_panel_w = right_rect_w + random.randint(2, 6)
        grid_w = left_panel_w + 1 + right_panel_w
        if panel_h > MAX_GRID_SIZE or grid_w > MAX_GRID_SIZE:
            continue

        grid = np.full((panel_h, grid_w), int(bg), dtype=int)
        grid[:, left_panel_w] = int(sep)

        left_top = random.randint(1, panel_h - left_rect_h - 1)
        left_left = random.randint(1, left_panel_w - left_rect_w - 1)
        right_top = random.randint(1, panel_h - right_rect_h - 1)
        right_left = random.randint(1, right_panel_w - right_rect_w - 1)

        _embed_rectangle(
            grid[:, :left_panel_w],
            left_top,
            left_left,
            left_rect_h,
            left_rect_w,
            base_color,
            bg,
            outer_pad[0],
            outer_pad[2],
            outer_mask,
        )
        _embed_rectangle(
            grid[:, left_panel_w + 1 :],
            right_top,
            right_left,
            right_rect_h,
            right_rect_w,
            hole_color,
            bg,
            inner_pad[0],
            inner_pad[2],
            inner_mask,
        )

        expected = _compose_output(outer_mask, inner_mask, base_color, hole_color)
        actual = main(grid)
        if np.array_equal(actual, expected) and grid.shape[0] <= MAX_GRID_SIZE and grid.shape[1] <= MAX_GRID_SIZE:
            return grid, expected

    raise RuntimeError("failed to generate a valid ARC case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]


if __name__ == "__main__":
    for _ in range(8):
        generated_input, expected_output = _generate_case()
        actual_output = main(generated_input)
        assert isinstance(generated_input, np.ndarray) and generated_input.ndim == 2
        assert isinstance(actual_output, np.ndarray) and actual_output.ndim == 2
        assert actual_output.shape == expected_output.shape
        assert actual_output.shape[0] <= MAX_OUTPUT_SIZE and actual_output.shape[1] <= MAX_OUTPUT_SIZE
        assert np.array_equal(actual_output, expected_output)
