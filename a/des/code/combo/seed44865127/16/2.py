from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.sum(counts == max_count)) == 1, "expected a unique dominant background color"
    return int(colors[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _single_non_bg_color(region: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(region) if int(color) != int(bg)]
    assert len(colors) == 1, "expected exactly one foreground color in each half"
    return colors[0]


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, int, int, np.ndarray, tuple[int, int, int, int]]:
    bg = _dominant_color(grid)

    sep_rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == bg)]
    assert len(sep_rows) == 1, "expected exactly one full background separator row"
    sep = sep_rows[0]
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into two nonempty halves"

    top = grid[:sep, :]
    bottom = grid[sep + 1 :, :]
    assert not np.any(np.all(top == bg, axis=1)), "top half should contain the mold on every row"
    assert not np.any(np.all(bottom == bg, axis=1)), "bottom half should contain the material on every row"

    mold_color = _single_non_bg_color(top, bg)
    material_color = _single_non_bg_color(bottom, bg)
    assert mold_color != material_color, "mold and material should use different colors"

    mold_box = _bbox(top == mold_color)
    material_box_local = _bbox(bottom == material_color)
    mr0, mc0, mr1, mc1 = mold_box
    br0, bc0, br1, bc1 = material_box_local

    mold_mask = top[mr0 : mr1 + 1, mc0 : mc1 + 1] == mold_color
    material_region = bottom[br0 : br1 + 1, bc0 : bc1 + 1]
    assert np.all(material_region == material_color), "material must be a solid rectangle"
    assert mold_mask.shape[1] == material_region.shape[1], "mold and material must have the same width"

    material_box = (sep + 1 + br0, bc0, sep + 1 + br1, bc1)
    return bg, sep, mold_color, material_color, mold_mask, material_box


def _apply_imprint(grid: np.ndarray) -> np.ndarray:
    bg, _sep, _mold_color, _material_color, mold_mask, material_box = _parse_scene(grid)
    r0, c0, r1, c1 = material_box
    material_h = r1 - r0 + 1
    flipped = np.flipud(mold_mask)
    imprint = flipped[:material_h, :]

    out = grid.copy()
    region = out[r0 : r0 + imprint.shape[0], c0 : c1 + 1]
    region[imprint] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Imprint the vertically flipped mold pattern into the solid material block.

    Input structure
    - The dominant color is background.
    - The grid is split by one full background row.
    - Above the separator, one foreground color forms the mold pattern.
    - Below the separator, another foreground color forms a solid material rectangle.

    Rule
    - Vertically flip the mold pattern inside its bbox.
    - Align that flipped pattern to the material's top surface.
    - Wherever the flipped mold has foreground pixels, carve the material back to background.
    - Leave the mold and separator unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    return _apply_imprint(g)


# --------------- generator ---------------

GEN_RETRIES = 300
MAX_DIM = 30
PALETTE = tuple(range(10))
MOLD_H_RANGE = (3, 7)
MOLD_W_RANGE = (4, 10)
EXTRA_MATERIAL_DEPTH_RANGE = (0, 4)
EXTRA_WIDTH_RANGE = (4, 12)
HEIGHT_MODES = ("independent", "runs", "walk")


def _height_sequence(mold_h: int, mold_w: int) -> list[int] | None:
    for _ in range(100):
        mode = random.choice(HEIGHT_MODES)
        if mode == "independent":
            heights = [random.randint(1, mold_h) for _ in range(mold_w)]
        elif mode == "runs":
            segment_count = random.randint(2, min(4, mold_w))
            cuts = sorted(random.sample(range(1, mold_w), segment_count - 1))
            bounds = [0] + cuts + [mold_w]
            heights = []
            last = None
            for start, end in zip(bounds, bounds[1:]):
                choices = [h for h in range(1, mold_h + 1) if h != last] or list(range(1, mold_h + 1))
                value = random.choice(choices)
                heights.extend([value] * (end - start))
                last = value
        else:
            heights = [random.randint(1, mold_h)]
            for _ in range(mold_w - 1):
                step = random.choice((-2, -1, 0, 1, 2))
                nxt = min(mold_h, max(1, heights[-1] + step))
                heights.append(nxt)

        if max(heights) < mold_h:
            heights[random.randrange(mold_w)] = mold_h
        if min(heights) == mold_h:
            idx = random.randrange(mold_w)
            heights[idx] = random.randint(1, mold_h - 1)
        if len(set(heights)) == 1:
            idx = random.randrange(mold_w)
            heights[idx] = mold_h if heights[idx] != mold_h else max(1, mold_h - 1)

        if max(heights) == mold_h and min(heights) < mold_h:
            return heights
    return None


def _heights_to_mask(heights: list[int], mold_h: int) -> np.ndarray:
    mold_w = len(heights)
    mask = np.zeros((mold_h, mold_w), dtype=bool)
    for c, height in enumerate(heights):
        mask[mold_h - height :, c] = True
    assert np.all(mask.any(axis=1)), "mold bbox must span every row"
    assert np.any(~mask), "mold should contain recessed background cells"
    return mask


def _min_canvas_width(total_h: int, mold_w: int, material_h: int, mold_pixels: int) -> int:
    material_area = mold_w * material_h
    return max(mold_w + 1, (2 * material_area + mold_pixels) // total_h + 1)


def _render_input(
    bg: int,
    mold_color: int,
    material_color: int,
    mold_mask: np.ndarray,
    width: int,
    mold_left: int,
    material_left: int,
    material_h: int,
) -> np.ndarray:
    mold_h, mold_w = mold_mask.shape
    grid = np.full((mold_h + 1 + material_h, width), bg, dtype=int)
    grid[:mold_h, mold_left : mold_left + mold_w][mold_mask] = mold_color
    grid[mold_h + 1 :, material_left : material_left + mold_w] = material_color
    return grid


def _direct_imprint(grid: np.ndarray) -> np.ndarray:
    bg, _sep, _mold_color, _material_color, mold_mask, material_box = _parse_scene(grid)
    r0, c0, r1, c1 = material_box
    material_h = r1 - r0 + 1
    imprint = mold_mask[:material_h, :]

    out = grid.copy()
    region = out[r0 : r0 + imprint.shape[0], c0 : c1 + 1]
    region[imprint] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    mold_h = random.randint(*MOLD_H_RANGE)
    mold_w = random.randint(*MOLD_W_RANGE)
    material_h = mold_h + random.randint(*EXTRA_MATERIAL_DEPTH_RANGE)
    if mold_h + 1 + material_h > MAX_DIM:
        return None

    heights = _height_sequence(mold_h, mold_w)
    if heights is None:
        return None
    mold_mask = _heights_to_mask(heights, mold_h)

    total_h = mold_h + 1 + material_h
    min_width = _min_canvas_width(total_h, mold_w, material_h, int(mold_mask.sum()))
    max_width = min(MAX_DIM, mold_w + EXTRA_WIDTH_RANGE[1])
    min_width = max(min_width, mold_w + EXTRA_WIDTH_RANGE[0])
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    colors = random.sample(PALETTE, 3)
    bg, mold_color, material_color = colors
    mold_left = random.randint(0, width - mold_w)
    material_left = random.randint(0, width - mold_w)

    grid = _render_input(bg, mold_color, material_color, mold_mask, width, mold_left, material_left, material_h)

    bg_count = int(np.count_nonzero(grid == bg))
    material_count = int(np.count_nonzero(grid == material_color))
    mold_count = int(np.count_nonzero(grid == mold_color))
    if not (bg_count > material_count and bg_count > mold_count):
        return None

    expected = _apply_imprint(grid)
    direct = _direct_imprint(grid)
    if np.array_equal(expected, grid):
        return None
    if np.array_equal(expected, direct):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid imprint puzzle input")
