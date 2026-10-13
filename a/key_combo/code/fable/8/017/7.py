from __future__ import annotations

import random

import numpy as np


OUTPUT_BACKGROUND = 0
OUTPUT_STAMP_COLOR = 3

_INPUT_COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
_HEIGHT_RANGE = (2, 5)
_WIDTH_RANGE = (2, 5)
_GEN_ATTEMPTS = 300


def _find_separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    candidates: list[int] = []
    for r in range(height):
        color = int(grid[r, 0])
        if not np.all(grid[r, :] == color):
            continue
        if int(np.count_nonzero(grid == color)) != width:
            continue
        candidates.append(r)

    assert len(candidates) == 1, "expected exactly one unique separator row"
    sep_r = candidates[0]
    assert 0 < sep_r < height - 1, "separator row must split nonempty top and bottom masks"
    assert sep_r * 2 + 1 == height, "separator row must divide the grid into equal-height halves"
    return int(sep_r)


def _mask_colors(half: np.ndarray, bg: int) -> int:
    colors = set(map(int, np.unique(half)))
    assert bg in colors, "background color must appear in each half"
    fg = colors - {bg}
    assert len(fg) == 1, "each half must contain exactly one foreground color"
    return int(next(iter(fg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the XOR disagreement mask into the block slots selected by its own cells.

    Input structure:
    - A single full separator row has a color that appears nowhere else in the grid.
    - The separator splits the grid into top and bottom masks of equal shape.
    - Each half uses one shared background color and exactly one foreground color.

    Rule:
    - Align the two halves and mark each cell where exactly one half has foreground.
    - Treat that disagreement mask as an h x w stamp, where h and w are the half dimensions.
    - Create a black output of shape (h*h, w*w), divided into h x w block slots, and copy the whole disagreement mask in green into every slot whose selector cell is true.

    Key insight:
    - The visible input masks are only ingredients; the real stamp and the slot selector are the same hidden XOR-derived mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_r = _find_separator_row(g)
    top = g[:sep_r, :]
    bottom = g[sep_r + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "top and bottom halves must match"

    common_colors = set(map(int, np.unique(top))) & set(map(int, np.unique(bottom)))
    assert len(common_colors) == 1, "expected exactly one shared background color"
    bg = int(next(iter(common_colors)))
    top_fg = _mask_colors(top, bg)
    bottom_fg = _mask_colors(bottom, bg)

    diff = (top == top_fg) ^ (bottom == bottom_fg)
    h, w = map(int, diff.shape)

    stamp = np.where(diff, OUTPUT_STAMP_COLOR, OUTPUT_BACKGROUND)
    out = np.full((h * h, w * w), OUTPUT_BACKGROUND, dtype=int)
    for r, c in np.argwhere(diff):
        rr = int(r) * h
        cc = int(c) * w
        out[rr : rr + h, cc : cc + w] = stamp
    return out


def _expected_output(diff: np.ndarray) -> np.ndarray:
    h, w = map(int, diff.shape)
    stamp = np.where(diff, OUTPUT_STAMP_COLOR, OUTPUT_BACKGROUND)
    out = np.full((h * h, w * w), OUTPUT_BACKGROUND, dtype=int)
    for r, c in np.argwhere(diff):
        rr = int(r) * h
        cc = int(c) * w
        out[rr : rr + h, cc : cc + w] = stamp
    return out


def _sample_category_grid(h: int, w: int) -> np.ndarray:
    total = h * w
    assert total >= 4

    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)

    categories = np.zeros((h, w), dtype=int)
    # 0=shared background, 1=shared foreground, 2=top-only foreground, 3=bottom-only foreground.
    for cat, (r, c) in enumerate(coords[:4]):
        categories[r, c] = cat

    branch_weights = random.choice(
        (
            (0.34, 0.20, 0.23, 0.23),
            (0.20, 0.30, 0.25, 0.25),
            (0.24, 0.16, 0.30, 0.30),
        )
    )
    extra_choices = random.choices(range(4), weights=branch_weights, k=total - 4)
    for (r, c), cat in zip(coords[4:], extra_choices):
        categories[r, c] = int(cat)

    return categories


def _render_input_from_categories(
    categories: np.ndarray,
    bg: int,
    top_fg: int,
    bottom_fg: int,
    sep: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, categories.shape)
    top = np.full((h, w), bg, dtype=int)
    bottom = np.full((h, w), bg, dtype=int)

    top[(categories == 1) | (categories == 2)] = top_fg
    bottom[(categories == 1) | (categories == 3)] = bottom_fg

    grid = np.full((2 * h + 1, w), sep, dtype=int)
    grid[:h, :] = top
    grid[h + 1 :, :] = bottom

    diff = (categories == 2) | (categories == 3)
    return grid, diff


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*_HEIGHT_RANGE)
    w = random.randint(*_WIDTH_RANGE)
    colors = random.sample(_INPUT_COLOR_POOL, 4)
    bg, top_fg, bottom_fg, sep = map(int, colors)

    categories = _sample_category_grid(h, w)
    grid, diff = _render_input_from_categories(categories, bg, top_fg, bottom_fg, sep)
    expected = _expected_output(diff)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_ATTEMPTS} attempts")
