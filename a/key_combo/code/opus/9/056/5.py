from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FOREGROUND = 2
LEGEND = 5

SEED_SIZES = (2, 3)
MIN_CANVAS = 10
MAX_CANVAS = 15
GEN_ATTEMPTS = 400


def _parse_seed_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey = np.argwhere(g == LEGEND)
    assert grey.size > 0, "expected a grey L-shaped legend"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    assert int(r0) == 0 and int(c0) == 0, "legend must be anchored at the top-left corner"
    assert int(r1) == int(c1), "legend bbox must be square"

    side = int(r1) + 1
    assert side >= 2, "legend must contain a non-empty seed plus border"
    seed_size = side - 1

    legend_box = g[:side, :side]
    expected_grey = np.zeros((side, side), dtype=bool)
    expected_grey[-1, :] = True
    expected_grey[:, -1] = True
    assert np.array_equal(legend_box == LEGEND, expected_grey), "grey cells must form one solid L border"
    assert np.all(g[side:, :] == BACKGROUND), "cells below the legend must be background"
    assert np.all(g[:side, side:] == BACKGROUND), "cells right of the legend must be background"

    seed = legend_box[:seed_size, :seed_size]
    assert np.all((seed == BACKGROUND) | (seed == FOREGROUND)), "seed may contain only black and red"
    assert np.any(seed == FOREGROUND), "seed must contain foreground"
    assert np.any(seed == BACKGROUND), "seed must contain background"
    return seed == FOREGROUND


def _settle_columns(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for c in range(w):
        count = int(mask[:, c].sum())
        if count:
            out[h - count :, c] = True
    return out


def _recursive_expand(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    seed_h, seed_w = mask.shape
    assert seed_h == seed_w, "this puzzle uses a square seed"
    rows = np.arange(int(shape[0]))[:, None]
    cols = np.arange(int(shape[1]))[None, :]

    inner = mask[rows % seed_h, cols % seed_w]
    meta = mask[(rows // seed_h) % seed_h, (cols // seed_w) % seed_w]

    out = np.full(shape, BACKGROUND, dtype=int)
    out[inner & ~meta] = LEGEND
    out[inner & meta] = FOREGROUND
    return out


def _periodic_seed(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    seed_h, seed_w = mask.shape
    rows = np.arange(int(shape[0]))[:, None]
    cols = np.arange(int(shape[1]))[None, :]
    out = np.full(shape, BACKGROUND, dtype=int)
    out[mask[rows % seed_h, cols % seed_w]] = FOREGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Settle the top-left seed by column gravity, then expand that settled mask into a recursive self-similar pattern.

    Input structure:
    - The grid is mostly black background.
    - A small square legend sits in the top-left corner; its last row and last column are solid grey, forming an L border.
    - Removing that grey border leaves a red/black seed region, while every cell outside the legend is black.

    Rule:
    - Extract the seed region inside the grey L border.
    - Apply independent downward gravity inside that seed only: in each seed column, red cells fall to the lowest available seed cells.
    - Use the settled red/black seed as a binary mask for a two-level self-similar expansion:
      settled-red positions become red copies of the settled mask, settled-black positions become grey copies of the settled mask's foreground.
    - Tile and crop that recursive expansion to the original grid size, and discard the input legend border.

    Key insight:
    - The visible top-left seed is not the true fractal template; the seed must first be transformed by local column gravity, and only the settled mask drives the final recursive pattern.
    """
    g = np.asarray(grid, dtype=int)
    seed_mask = _parse_seed_mask(g)
    settled = _settle_columns(seed_mask)
    return _recursive_expand(settled, g.shape)


def _render_input(seed_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    seed_size = int(seed_mask.shape[0])
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[:seed_size, :seed_size] = np.where(seed_mask, FOREGROUND, BACKGROUND)
    grid[seed_size, : seed_size + 1] = LEGEND
    grid[:seed_size, seed_size] = LEGEND
    return grid


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.reshape(-1))


def _row_span(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(mask, axis=1)))


def _col_span(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(mask, axis=0)))


def _canvas_options(seed_size: int) -> list[tuple[int, int]]:
    minimum = max(MIN_CANVAS, seed_size * seed_size + 1)
    return [
        (h, w)
        for h in range(minimum, MAX_CANVAS + 1)
        for w in range(minimum, MAX_CANVAS + 1)
        if h <= 30 and w <= 30
    ]


def _semantic_shapes(raw_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    settled = _settle_columns(raw_mask)
    shapes: list[tuple[int, int]] = []
    for shape in _canvas_options(int(raw_mask.shape[0])):
        settled_out = _recursive_expand(settled, shape)
        raw_out = _recursive_expand(raw_mask, shape)
        tiled_out = _periodic_seed(settled, shape)
        if np.array_equal(settled_out, raw_out):
            continue
        if np.array_equal(settled_out, tiled_out):
            continue
        if FOREGROUND not in settled_out or LEGEND not in settled_out:
            continue
        if int(np.count_nonzero(settled_out)) <= int(raw_mask.sum()) * 2:
            continue
        shapes.append(shape)
    return tuple(shapes)


def _build_seed_catalog(seed_size: int) -> tuple[tuple[np.ndarray, tuple[tuple[int, int], ...]], ...]:
    area = seed_size * seed_size
    groups: dict[tuple[int, ...], list[np.ndarray]] = {}

    for bits in range(1 << area):
        raw = np.array([(bits >> idx) & 1 for idx in range(area)], dtype=bool).reshape(seed_size, seed_size)
        red_count = int(raw.sum())
        if not 2 <= red_count <= 4:
            continue
        if _row_span(raw) < 2 or _col_span(raw) < 2:
            continue

        settled = _settle_columns(raw)
        if np.array_equal(raw, settled):
            continue
        if not np.any(~settled):
            continue

        key = _mask_signature(settled)
        groups.setdefault(key, []).append(raw.copy())

    catalog: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=_mask_signature)
        for raw in candidates:
            valid_shapes = _semantic_shapes(raw)
            if valid_shapes:
                catalog.append((raw, valid_shapes))
                break
    assert catalog, f"no valid seed catalog for size {seed_size}"
    return tuple(catalog)


SEED_CATALOG = {seed_size: _build_seed_catalog(seed_size) for seed_size in SEED_SIZES}


def _pick_shape(seed_size: int, shapes: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    exact = [shape for shape in shapes if shape[0] % seed_size == 0 and shape[1] % seed_size == 0]
    cropped = [shape for shape in shapes if shape not in exact]
    if exact and cropped:
        pool = exact if random.random() < 0.5 else cropped
    else:
        pool = list(shapes)
    return random.choice(pool)


def _validate_example(grid: np.ndarray) -> bool:
    try:
        raw = _parse_seed_mask(grid)
    except AssertionError:
        return False

    settled = _settle_columns(raw)
    if np.array_equal(raw, settled):
        return False
    if int(raw.sum()) < 2 or int(raw.sum()) > 4:
        return False
    if _row_span(raw) < 2 or _col_span(raw) < 2:
        return False

    expected = _recursive_expand(settled, grid.shape)
    raw_based = _recursive_expand(raw, grid.shape)
    tiled = _periodic_seed(settled, grid.shape)

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, raw_based):
        return False
    if np.array_equal(actual, tiled):
        return False
    if FOREGROUND not in actual or LEGEND not in actual:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed_size = random.choice(SEED_SIZES)
        raw_mask, valid_shapes = random.choice(SEED_CATALOG[seed_size])
        shape = _pick_shape(seed_size, valid_shapes)
        grid = _render_input(raw_mask, shape)
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 5")
