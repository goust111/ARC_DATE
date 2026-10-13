from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_is_color(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _visible_layers(grid: np.ndarray) -> list[dict[str, int]]:
    layers: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        r0, c0, r1, c1 = _bbox(grid == color)
        assert _boundary_is_color(grid, color, r0, c0, r1, c1), (
            "each color must expose the full outer boundary of its hidden rectangle"
        )
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        layers.append(
            {
                "color": color,
                "height": height,
                "width": width,
                "area": height * width,
            }
        )

    assert len(layers) >= 2, "expected multiple colored rectangles"
    areas = [layer["area"] for layer in layers]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    layers.sort(key=lambda layer: layer["area"])

    outer_h = layers[-1]["height"]
    outer_w = layers[-1]["width"]
    for small, large in zip(layers, layers[1:]):
        assert large["height"] >= small["height"] + 4
        assert large["width"] >= small["width"] + 4
    for layer in layers:
        assert (outer_h - layer["height"]) % 2 == 0
        assert (outer_w - layer["width"]) % 2 == 0
    return layers


def _draw_concentric(layers: list[dict[str, int]]) -> np.ndarray:
    outer_h = layers[-1]["height"]
    outer_w = layers[-1]["width"]
    assert outer_h <= 30 and outer_w <= 30
    out = np.full((outer_h, outer_w), BACKGROUND, dtype=int)

    for idx in range(len(layers) - 1, -1, -1):
        layer = layers[idx]
        color = layer["color"]
        height = layer["height"]
        width = layer["width"]
        r0 = (outer_h - height) // 2
        c0 = (outer_w - width) // 2

        if idx != len(layers) - 1:
            gr0, gc0 = r0 - 1, c0 - 1
            gr1, gc1 = r0 + height, c0 + width
            assert 0 <= gr0 <= gr1 < outer_h
            assert 0 <= gc0 <= gc1 < outer_w
            out[gr0 : gr1 + 1, gc0 : gc1 + 1] = BACKGROUND

        out[r0 : r0 + height, c0 : c0 + width] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapped colored rectangles and redraw them as ordered concentric frames.

    Input structure:
    - The background is black.
    - Each non-background color belongs to one solid rectangle from an overlapped stack.
    - Smaller rectangles lie above larger rectangles, so larger rectangles are visible only as edge frames.
    - For every color, the visible pixels expose the outer boundary of that color's full hidden rectangle.

    Rule:
    - Recover each full rectangle by taking the bounding box of its visible pixels.
    - Sort the recovered rectangles by full area from small to large.
    - On a black output canvas the size of the largest recovered rectangle, draw the rectangles centered together.
    - Draw larger rectangles first, clear a one-pixel black gap around each smaller rectangle, then draw that smaller rectangle.

    Key insight:
    - The input is an occlusion encoding: a color's frame gives the full rectangle boundary, and the output reorganizes those recovered rectangle sizes into a regular concentric nesting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    layers = _visible_layers(g)
    return _draw_concentric(layers)


def _sample_dimensions(count: int) -> list[tuple[int, int]] | None:
    height = random.randint(2, 6)
    width = random.randint(2, 7)
    dims = [(height, width)]

    for _ in range(count - 1):
        height += random.choice((4, 6, 8))
        width += random.choice((4, 6, 8))
        if height > 26 or width > 26:
            return None
        dims.append((height, width))

    return dims


def _place_stack(
    grid_shape: tuple[int, int],
    dims_small_to_large: list[tuple[int, int]],
    colors_small_to_large: list[int],
) -> np.ndarray | None:
    grid_h, grid_w = grid_shape
    max_h, max_w = dims_small_to_large[-1]
    if max_h > grid_h or max_w > grid_w:
        return None

    top = random.randint(0, grid_h - max_h)
    left = random.randint(0, grid_w - max_w)

    bboxes_large_to_small: list[tuple[int, int, int, int]] = [(top, left, max_h, max_w)]
    prev_r, prev_c, prev_h, prev_w = bboxes_large_to_small[0]

    for height, width in reversed(dims_small_to_large[:-1]):
        centered_r = prev_r + (prev_h - height) // 2
        centered_c = prev_c + (prev_w - width) // 2
        slack_r = (prev_h - height) // 2 - 1
        slack_c = (prev_w - width) // 2 - 1
        jitter_r = random.randint(-min(2, slack_r), min(2, slack_r))
        jitter_c = random.randint(-min(2, slack_c), min(2, slack_c))
        r0 = centered_r + jitter_r
        c0 = centered_c + jitter_c

        if not (prev_r < r0 and r0 + height - 1 < prev_r + prev_h - 1):
            return None
        if not (prev_c < c0 and c0 + width - 1 < prev_c + prev_w - 1):
            return None

        bboxes_large_to_small.append((r0, c0, height, width))
        prev_r, prev_c, prev_h, prev_w = r0, c0, height, width

    grid = np.full(grid_shape, BACKGROUND, dtype=int)
    colors_large_to_small = list(reversed(colors_small_to_large))
    for (r0, c0, height, width), color in zip(bboxes_large_to_small, colors_large_to_small):
        grid[r0 : r0 + height, c0 : c0 + width] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 6)
    dims = _sample_dimensions(count)
    if dims is None:
        return None

    max_h, max_w = dims[-1]
    if max_h + 2 > 30 or max_w + 2 > 30:
        return None

    grid_h = random.randint(max_h + 2, 30)
    grid_w = random.randint(max_w + 2, 30)
    colors = random.sample(COLORS, count)
    grid = _place_stack((grid_h, grid_w), dims, colors)
    if grid is None:
        return None

    try:
        expected_layers = [
            {"color": color, "height": height, "width": width, "area": height * width}
            for (height, width), color in zip(dims, colors)
        ]
        expected = _draw_concentric(expected_layers)
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
