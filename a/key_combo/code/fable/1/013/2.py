from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, paste


Span = tuple[int, int]

_RNG = np.random.default_rng()
_PALETTE = [color for color in range(10) if color != Color.GREY]
_GEN_RETRY_LIMIT = 400


def _segments_from_separators(length: int, separators: np.ndarray) -> list[Span]:
    separators = np.asarray(separators, dtype=int)
    assert separators.size > 0, "expected internal separator lines on both axes"
    cuts = [-1, *map(int, separators), int(length)]
    spans: list[Span] = []
    for left, right in zip(cuts, cuts[1:]):
        start = int(left) + 1
        stop = int(right) - 1
        assert start <= stop, "separator lines must leave non-empty tile spans"
        spans.append((start, stop))
    sizes = {stop - start + 1 for start, stop in spans}
    assert len(sizes) == 1, "tile bands must all have the same size"
    return spans


def _extract_factors(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    sep_rows = np.flatnonzero(np.all(g == Color.GREY, axis=1))
    sep_cols = np.flatnonzero(np.all(g == Color.GREY, axis=0))

    row_spans = _segments_from_separators(g.shape[0], sep_rows)
    col_spans = _segments_from_separators(g.shape[1], sep_cols)

    tiles: dict[bytes, np.ndarray] = {}
    layout = np.zeros((len(row_spans), len(col_spans)), dtype=bool)
    keys = np.empty(layout.shape, dtype=object)

    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            key = patch.tobytes()
            tiles.setdefault(key, patch)
            keys[i, j] = key

    assert len(tiles) == 2, "expected exactly one empty tile template and one occupied tile template"

    monochrome_tiles: list[tuple[bytes, int]] = []
    for key, patch in tiles.items():
        patch_colors = colors_present(patch, exclude={Color.GREY})
        if len(patch_colors) == 1:
            monochrome_tiles.append((key, int(patch_colors[0])))

    assert len(monochrome_tiles) == 1, "expected one unique monochrome empty tile template"
    empty_key, bg = monochrome_tiles[0]
    occupied_keys = [key for key in tiles if key != empty_key]
    assert len(occupied_keys) == 1

    occupied_patch = tiles[occupied_keys[0]]
    fg_colors = colors_present(occupied_patch, exclude={bg, Color.GREY})
    assert len(fg_colors) == 1, "occupied tile must contain exactly one foreground color"
    fg = int(fg_colors[0])

    pattern_mask = occupied_patch == fg
    assert np.any(pattern_mask) and np.any(~pattern_mask), "tile pattern must be non-empty and non-solid"

    for i in range(layout.shape[0]):
        for j in range(layout.shape[1]):
            layout[i, j] = keys[i, j] != empty_key

    assert np.any(layout) and np.any(~layout), "layout must contain both occupied and empty tiles"
    return layout, pattern_mask, bg, fg


def _swap_kron(pattern_mask: np.ndarray, layout_mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    mask = np.kron(pattern_mask.astype(int), layout_mask.astype(int)).astype(bool)
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = int(fg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the layout mask and the repeated tile pattern in a separator-sliced Kronecker product scene.

    Input structure:
    - Grey full-row and full-column separator lines partition the grid into an
      R-by-C array of equal-sized tiles.
    - Every occupied tile contains the same single-color pattern at the same
      in-tile coordinates, while every empty tile is pure background.
    - The occupied-tile distribution is a layout mask A and the shared in-tile
      figure is a pattern mask B.

    Rule:
    - Remove the separator lines conceptually and recover A from tile occupancy
      and B from the common occupied tile.
    - Produce an output of shape (r*R, c*C) with no separators.
    - Treat B as the macro layout: each foreground cell of B expands to one
      R-by-C block containing A, while each background cell expands to an
      all-background block.

    Key insight:
    - The input already encodes A ⊗ B; the task is to keep the same two binary
      factors but exchange their roles, yielding B ⊗ A.
    """
    layout_mask, pattern_mask, bg, fg = _extract_factors(grid)
    return _swap_kron(pattern_mask, layout_mask, bg, fg)


def _mask_spans_both_axes(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return (
        coords.shape[0] >= 2
        and len(np.unique(coords[:, 0])) >= 2
        and len(np.unique(coords[:, 1])) >= 2
    )


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _random_scattered_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    chosen = rng.choice(h * w, size=target, replace=False)
    mask.flat[chosen] = True
    return mask


def _sample_mask(
    rng: np.random.Generator,
    h: int,
    w: int,
    *,
    connected_bias: float,
    min_count: int,
    max_count: int,
) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(min_count, max_count + 1))
        if rng.random() < connected_bias:
            mask = _random_connected_mask(rng, h, w, target)
        else:
            mask = _random_scattered_mask(rng, h, w, target)
        if _mask_spans_both_axes(mask):
            return mask
    raise RuntimeError("failed to sample a non-degenerate mask")


def _sample_factors(
    rng: np.random.Generator, R: int, C: int, r: int, c: int
) -> tuple[np.ndarray, np.ndarray]:
    layout_area = R * C
    pattern_area = r * c

    layout_min = max(2, int(np.ceil(0.28 * layout_area)))
    layout_max = min(layout_area - 1, int(np.floor(0.72 * layout_area)))
    pattern_min = max(2, int(np.ceil(0.24 * pattern_area)))
    pattern_max = min(pattern_area - 1, int(np.floor(0.6 * pattern_area)))

    for _ in range(250):
        layout_mask = _sample_mask(
            rng,
            R,
            C,
            connected_bias=0.35,
            min_count=layout_min,
            max_count=layout_max,
        )
        pattern_mask = _sample_mask(
            rng,
            r,
            c,
            connected_bias=0.75,
            min_count=pattern_min,
            max_count=pattern_max,
        )
        if layout_mask.shape == pattern_mask.shape and np.array_equal(layout_mask, pattern_mask):
            continue
        if np.array_equal(
            np.kron(layout_mask.astype(int), pattern_mask.astype(int)),
            np.kron(pattern_mask.astype(int), layout_mask.astype(int)),
        ):
            continue
        return layout_mask, pattern_mask
    raise RuntimeError("failed to sample compatible layout and pattern masks")


def _build_input(layout_mask: np.ndarray, pattern_mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    R, C = layout_mask.shape
    r, c = pattern_mask.shape
    H = R * r + (R - 1)
    W = C * c + (C - 1)
    grid = np.full((H, W), int(bg), dtype=int)

    for split in range(1, R):
        grid[split * (r + 1) - 1, :] = Color.GREY
    for split in range(1, C):
        grid[:, split * (c + 1) - 1] = Color.GREY

    tile_patch = np.full((r, c), int(bg), dtype=int)
    tile_patch[pattern_mask] = int(fg)
    for i in range(R):
        for j in range(C):
            if layout_mask[i, j]:
                paste(grid, tile_patch, i * (r + 1), j * (c + 1))
    return grid


def _sample_dimensions(rng: np.random.Generator) -> tuple[int, int, int, int]:
    choices: list[tuple[int, int, int, int]] = []
    for R in range(2, 6):
        for C in range(2, 6):
            for r in range(2, 6):
                for c in range(2, 6):
                    in_h = R * r + (R - 1)
                    in_w = C * c + (C - 1)
                    out_h = R * r
                    out_w = C * c
                    if max(in_h, in_w, out_h, out_w) <= 30:
                        choices.append((R, C, r, c))
    return choices[int(rng.integers(0, len(choices)))]


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    R, C, r, c = _sample_dimensions(rng)
    bg, fg = rng.choice(_PALETTE, size=2, replace=False).tolist()
    layout_mask, pattern_mask = _sample_factors(rng, R, C, r, c)

    inp = _build_input(layout_mask, pattern_mask, int(bg), int(fg))
    expected = _swap_kron(pattern_mask, layout_mask, int(bg), int(fg))

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        candidate = _generate_candidate(_RNG)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid sample after {_GEN_RETRY_LIMIT} attempts")
