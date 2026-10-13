from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, paste


_RNG = np.random.default_rng()

_GLYPH_POOL = (
    np.array([[9, 9, 9], [9, 9, 9], [9, 9, 9]], dtype=int),
    np.array([[9, 9, 9], [9, 0, 9], [9, 9, 9]], dtype=int),
    np.array([[0, 9, 0], [9, 9, 9], [0, 9, 0]], dtype=int),
    np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=int),
    np.array([[9, 0, 0], [9, 0, 0], [9, 9, 9]], dtype=int),
)
_ASYMMETRIC_GLYPHS = {3, 4}
_REGION_COLORS = np.array([Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW], dtype=int)


def _separator_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox(grid == Color.GREY)
    assert box is not None, "expected a grey separator lattice"
    return tuple(map(int, box))


def _compress_regions(grid: np.ndarray, sep_box: tuple[int, int, int, int]) -> np.ndarray:
    block = crop_bbox(grid, sep_box)
    sep_rows = [r for r in range(block.shape[0]) if np.all(block[r, :] == Color.GREY)]
    sep_cols = [c for c in range(block.shape[1]) if np.all(block[:, c] == Color.GREY)]

    assert len(sep_rows) >= 2 and len(sep_cols) >= 2
    assert sep_rows[0] == 0 and sep_rows[-1] == block.shape[0] - 1
    assert sep_cols[0] == 0 and sep_cols[-1] == block.shape[1] - 1

    row_spans = [(sep_rows[i] + 1, sep_rows[i + 1]) for i in range(len(sep_rows) - 1)]
    col_spans = [(sep_cols[i] + 1, sep_cols[i + 1]) for i in range(len(sep_cols) - 1)]
    row_spans = [(r0, r1) for r0, r1 in row_spans if r0 < r1]
    col_spans = [(c0, c1) for c0, c1 in col_spans if c0 < c1]

    compact = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            patch = block[r0:r1, c0:c1]
            colors = np.unique(patch)
            assert colors.size == 1 and int(colors[0]) != Color.GREY
            compact[i, j] = int(colors[0])
    return compact


def _legend_mapping(
    grid: np.ndarray,
    sep_box: tuple[int, int, int, int],
    used_colors: list[int],
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = sep_box
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    maroon_components = [
        comp
        for comp in find_components(
            grid,
            bg=Color.BLACK,
            connectivity=8,
            monochromatic=True,
            colors=[Color.MAROON],
            return_format="dict",
        )
        if int(comp["bbox"][1]) > c1
    ]
    assert maroon_components, "expected legend glyphs to the right of the separator lattice"

    mapping: dict[int, np.ndarray] = {}
    for color in used_colors:
        indicator_locs = np.argwhere((grid == int(color)) & outside)
        assert indicator_locs.shape[0] == 1, "each used color needs exactly one legend indicator pixel"
        ir, ic = map(int, indicator_locs[0])

        candidates = []
        for comp in maroon_components:
            gr0, gc0, gr1, gc1 = map(int, comp["bbox"])
            if gc0 <= ic:
                continue
            if not (gr0 <= ir <= gr1):
                continue
            if (gr1 - gr0 + 1, gc1 - gc0 + 1) != (3, 3):
                continue
            candidates.append(comp)

        assert candidates, "legend indicator must align with a 3x3 maroon glyph"
        chosen = min(candidates, key=lambda comp: int(comp["bbox"][1]) - ic)
        glyph = crop_bbox(grid, tuple(map(int, chosen["bbox"])))
        assert glyph.shape == (3, 3)
        assert set(map(int, np.unique(glyph))).issubset({Color.BLACK, Color.MAROON})
        mapping[int(color)] = glyph

    return mapping


def _stamp_symbols(symbols: np.ndarray, mapping: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((symbols.shape[0] * 3, symbols.shape[1] * 3), Color.BLACK, dtype=int)
    for r, c in np.ndindex(symbols.shape):
        paste(out, mapping[int(symbols[r, c])], 3 * r, 3 * c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-partitioned color grid, mirror the symbol matrix, and decode it into 3x3 glyph tiles.

    Input structure:
    - The left side contains a grey separator lattice that encloses a rectangular grid of uniformly colored regions.
    - Each enclosed region is filled with one color, and the right side holds a legend made of single-pixel color indicators plus nearby 3x3 maroon glyphs on black.
    - The legend provides one glyph entry for every region color that appears in the separator lattice.

    Rule:
    - Find the grey partition, and replace each enclosed uniform region by one pixel of its color to obtain a compact symbol matrix.
    - Extend that symbol matrix by concatenating its left-to-right mirror on the right.
    - Read the legend mapping from region colors to 3x3 glyphs, then stamp each symbol with its glyph to build the final output with no gaps.

    Key insight:
    - The reflection happens before decoding: columns are mirrored at the compressed symbol level, so the glyph order reverses on the right while each individual glyph keeps its original orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_box = _separator_bbox(g)
    compact = _compress_regions(g, sep_box)
    mirrored = np.hstack([compact, np.fliplr(compact)])
    used_colors = sorted(set(map(int, compact.flatten())))
    glyphs = _legend_mapping(g, sep_box, used_colors)
    return _stamp_symbols(mirrored, glyphs)


def _sample_used_colors() -> list[int]:
    count = 4 if _RNG.random() < 0.6 else 3
    return [int(c) for c in _RNG.choice(_REGION_COLORS, size=count, replace=False)]


def _sample_glyph_mapping(used_colors: list[int]) -> dict[int, np.ndarray]:
    asymmetric = int(_RNG.choice(sorted(_ASYMMETRIC_GLYPHS)))
    available = [i for i in range(len(_GLYPH_POOL)) if i != asymmetric]
    chosen = [asymmetric]
    chosen.extend(int(i) for i in _RNG.choice(available, size=len(used_colors) - 1, replace=False))
    _RNG.shuffle(chosen)
    return {color: _GLYPH_POOL[idx].copy() for color, idx in zip(used_colors, chosen)}


def _sample_compact_matrix(rows: int, cols: int, used_colors: list[int]) -> np.ndarray:
    while True:
        flat = _RNG.choice(used_colors, size=rows * cols, replace=True).astype(int)
        flat[: len(used_colors)] = np.array(used_colors, dtype=int)
        flat = flat[_RNG.permutation(flat.size)]
        compact = flat.reshape(rows, cols)
        if np.array_equal(compact, np.fliplr(compact)):
            continue
        return compact


def _partition_block(compact: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    block = np.full((height, width), Color.GREY, dtype=int)

    r = 1
    for i, h in enumerate(row_heights):
        c = 1
        for j, w in enumerate(col_widths):
            block[r : r + h, c : c + w] = int(compact[i, j])
            c += w + 1
        r += h + 1
    return block


def _build_legend(mapping: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    height = 4 * (len(order) - 1) + 3
    width = 5
    legend = np.zeros((height, width), dtype=int)
    for idx, color in enumerate(order):
        top = 4 * idx
        legend[top + 1, 0] = int(color)
        legend[top : top + 3, 2:5] = mapping[int(color)]
    return legend


def _compose_input(partition: np.ndarray, legend: np.ndarray) -> np.ndarray:
    top_partition = int(_RNG.integers(1, 3))
    top_legend = int(_RNG.integers(1, 3))
    left_margin = int(_RNG.integers(1, 3))
    right_margin = int(_RNG.integers(1, 3))
    mid_gap = int(_RNG.integers(2, 4))
    bottom_margin = int(_RNG.integers(1, 3))

    height = max(top_partition + partition.shape[0], top_legend + legend.shape[0]) + bottom_margin
    width = left_margin + partition.shape[1] + mid_gap + legend.shape[1] + right_margin
    if width <= height:
        width = min(30, height + int(_RNG.integers(1, 3)))
    grid = np.zeros((height, width), dtype=int)

    paste(grid, partition, top_partition, left_margin)
    paste(grid, legend, top_legend, left_margin + partition.shape[1] + mid_gap)
    return grid


def _expected_output(compact: np.ndarray, mapping: dict[int, np.ndarray]) -> np.ndarray:
    mirrored = np.hstack([compact, np.fliplr(compact)])
    return _stamp_symbols(mirrored, mapping)


def _pixel_level_mirror_output(compact: np.ndarray, mapping: dict[int, np.ndarray]) -> np.ndarray:
    left = _stamp_symbols(compact, mapping)
    return np.hstack([left, np.fliplr(left)])


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows = int(_RNG.integers(3, 5))
        cols = int(_RNG.integers(2, 4))
        used_colors = _sample_used_colors()
        mapping = _sample_glyph_mapping(used_colors)
        compact = _sample_compact_matrix(rows, cols, used_colors)

        row_heights = [int(_RNG.integers(2, 4)) for _ in range(rows)]
        col_widths = [int(_RNG.integers(3, 5)) for _ in range(cols)]
        partition = _partition_block(compact, row_heights, col_widths)
        legend_order = [int(c) for c in _RNG.permutation(used_colors)]
        legend = _build_legend(mapping, legend_order)
        grid = _compose_input(partition, legend)

        if max(grid.shape) > 30:
            continue

        expected = _expected_output(compact, mapping)
        if max(expected.shape) > 30:
            continue
        if np.array_equal(expected, _pixel_level_mirror_output(compact, mapping)):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
