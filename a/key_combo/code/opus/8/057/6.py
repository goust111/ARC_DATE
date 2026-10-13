from __future__ import annotations

from typing import Any

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

CELL_SIZE = 3
RIGHT_COLS = 3
RIGHT_ROW_OPTIONS = (2, 3)
LEFT_WIDTH_OPTIONS = (5, 6, 7)
MAX_ATTEMPTS = 400

MARKER_TO_PAIR = {
    BLUE: (RED, ORANGE),
    RED: (YELLOW, MAROON),
    GREEN: (TEAL, GREY),
    PINK: (BLUE, GREEN),
}
MARKER_COLORS = tuple(MARKER_TO_PAIR)


def _non_separator_intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = [-1, *sorted(int(s) for s in separators), int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _full_grey_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]


def _full_grey_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]


def _extract_mask(left_panel: np.ndarray) -> np.ndarray:
    coords = np.argwhere(left_panel == GREY)
    assert len(coords) >= 2, "expected a grey binary mask in the left panel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = left_panel[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert set(map(int, np.unique(patch))) <= {BG, GREY}, "left mask must use only black and grey"
    mask = patch == GREY
    assert mask.shape == (CELL_SIZE, CELL_SIZE), "mask must match the cell size"
    fg = int(mask.sum())
    assert fg >= 2 and mask.size - fg >= 2, "mask needs both foreground and background area"
    return mask


def _parse_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_grey_cols = _full_grey_columns(g)
    assert full_grey_cols, "expected at least one full-height grey separator column"
    divider = int(full_grey_cols[0])
    assert 0 < divider < g.shape[1] - 1, "divider must split non-empty left and right regions"

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    mask = _extract_mask(left)
    n = mask.shape[0]
    center = (n // 2, n // 2)

    sep_rows = _full_grey_rows(right)
    sep_cols = _full_grey_columns(right)
    row_spans = _non_separator_intervals(right.shape[0], sep_rows)
    col_spans = _non_separator_intervals(right.shape[1], sep_cols)
    assert len(row_spans) in RIGHT_ROW_OPTIONS, "right region must be a 2x3 or 3x3 cell lattice"
    assert len(col_spans) == RIGHT_COLS, "right region must have three columns of cells"
    assert all(r1 - r0 == n for r0, r1 in row_spans), "cell heights must match the mask"
    assert all(c1 - c0 == n for c0, c1 in col_spans), "cell widths must match the mask"

    markers: dict[tuple[int, int], int] = {}
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            window = right[r0:r1, c0:c1]
            assert window.shape == (n, n)
            assert not bool(np.any(window == GREY)), "grey is reserved for separators in the input lattice"
            coords = np.argwhere(window != BG)
            assert len(coords) == 1, "each cell must contain exactly one marker"
            marker_r, marker_c = map(int, coords[0])
            assert (marker_r, marker_c) == center, "marker must sit at the cell center"
            color = int(window[marker_r, marker_c])
            assert color in MARKER_TO_PAIR, "unexpected marker color"
            markers[(rr, cc)] = color

    return {
        "grid": g,
        "divider": divider,
        "left": left,
        "right": right,
        "mask": mask,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "markers": markers,
    }


def _render_right(mask: np.ndarray, right: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], markers: dict[tuple[int, int], int]) -> np.ndarray:
    out = right.copy()
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            fg, bg = MARKER_TO_PAIR[markers[(rr, cc)]]
            tile = np.where(mask, fg, bg).astype(int)
            out[r0:r1, c0:c1] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the left binary mask into each right-side cell using the center marker's color pair.

    Input structure
    - A full-height grey column splits the grid into a left mask panel and a right cell lattice.
    - The left panel contains one black-and-grey binary mask; the right panel is partitioned by grey separator rows and columns into equal 3x3 cells.
    - Each right-side cell contains exactly one centered marker pixel chosen from blue, red, green, or pink.

    Rule
    - Extract the left binary mask, treating grey cells as foreground and black cells as background.
    - For each right-side cell, read its centered marker color and map it to a foreground/background pair:
      blue -> red/orange, red -> yellow/maroon, green -> teal/grey, pink -> blue/green.
    - Replace that entire cell by the left mask recolored with the chosen pair, while keeping the right-side grey separator lines unchanged.
    - Output only the transformed right region, excluding the left panel and the main divider column.

    Key insight
    - The left mask supplies one shared spatial template, but each right-side cell chooses its own two output colors from its marker, so shape and color must be decoded independently and combined per cell.
    """
    parsed = _parse_grid(grid)
    return _render_right(
        parsed["mask"],
        parsed["right"],
        parsed["row_spans"],
        parsed["col_spans"],
        parsed["markers"],
    )


def _is_strongly_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.flipud(mask),
        np.fliplr(mask),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        mask.T,
    ]
    return all(not np.array_equal(mask, other) for other in variants)


def _sample_mask(rng: np.random.Generator) -> np.ndarray:
    while True:
        mask = rng.integers(0, 2, size=(CELL_SIZE, CELL_SIZE), endpoint=False).astype(bool)
        fg = int(mask.sum())
        if fg < 2 or mask.size - fg < 2:
            continue
        if not _is_strongly_asymmetric(mask):
            continue
        return mask


def _sample_markers(rng: np.random.Generator, cell_count: int) -> list[int]:
    markers = list(MARKER_COLORS)
    while len(markers) < cell_count:
        markers.append(int(rng.choice(MARKER_COLORS)))
    rng.shuffle(markers)
    return markers


def _build_input(mask: np.ndarray, row_cells: int, left_width: int, mask_top: int, mask_left: int, markers: list[int]) -> np.ndarray:
    right_h = row_cells * CELL_SIZE + (row_cells - 1)
    right_w = RIGHT_COLS * CELL_SIZE + (RIGHT_COLS - 1)
    total_w = left_width + 1 + right_w

    grid = np.zeros((right_h, total_w), dtype=int)
    divider = left_width
    grid[:, divider] = GREY
    grid[mask_top : mask_top + CELL_SIZE, mask_left : mask_left + CELL_SIZE][mask] = GREY

    right0 = divider + 1
    for k in range(1, row_cells):
        grid[k * (CELL_SIZE + 1) - 1, right0:] = GREY
    for k in range(1, RIGHT_COLS):
        grid[:, right0 + k * (CELL_SIZE + 1) - 1] = GREY

    index = 0
    for rr in range(row_cells):
        r0 = rr * (CELL_SIZE + 1)
        for cc in range(RIGHT_COLS):
            c0 = right0 + cc * (CELL_SIZE + 1)
            grid[r0 + CELL_SIZE // 2, c0 + CELL_SIZE // 2] = markers[index]
            index += 1
    return grid


def _wrong_render_global_pair(parsed: dict[str, Any]) -> np.ndarray:
    first_marker = parsed["markers"][(0, 0)]
    wrong_markers = {key: first_marker for key in parsed["markers"]}
    return _render_right(parsed["mask"], parsed["right"], parsed["row_spans"], parsed["col_spans"], wrong_markers)


def _wrong_render_inverted_mask(parsed: dict[str, Any]) -> np.ndarray:
    return _render_right(~parsed["mask"], parsed["right"], parsed["row_spans"], parsed["col_spans"], parsed["markers"])


def _wrong_render_solid_foreground(parsed: dict[str, Any]) -> np.ndarray:
    out = parsed["right"].copy()
    for rr, (r0, r1) in enumerate(parsed["row_spans"]):
        for cc, (c0, c1) in enumerate(parsed["col_spans"]):
            fg, _bg = MARKER_TO_PAIR[parsed["markers"][(rr, cc)]]
            out[r0:r1, c0:c1] = fg
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False

        parsed = _parse_grid(g)
        mask = parsed["mask"]
        if not _is_strongly_asymmetric(mask):
            return False

        marker_values = list(parsed["markers"].values())
        unique_markers = set(marker_values)
        if unique_markers != set(MARKER_COLORS):
            return False

        out = main(g)
        expected_shape = parsed["right"].shape
        if out.shape != expected_shape or max(out.shape) > 30:
            return False

        if np.array_equal(out, parsed["right"]):
            return False
        if np.array_equal(out, _wrong_render_global_pair(parsed)):
            return False
        if np.array_equal(out, _wrong_render_inverted_mask(parsed)):
            return False
        if np.array_equal(out, _wrong_render_solid_foreground(parsed)):
            return False

        seen_tiles: set[bytes] = set()
        for rr, (r0, r1) in enumerate(parsed["row_spans"]):
            for cc, (c0, c1) in enumerate(parsed["col_spans"]):
                tile = out[r0:r1, c0:c1]
                fg, bg = MARKER_TO_PAIR[parsed["markers"][(rr, cc)]]
                if not np.any(tile == fg) or not np.any(tile == bg):
                    return False
                seen_tiles.add(tile.tobytes())
        if len(seen_tiles) < 4:
            return False

        if len({parsed["markers"][(rr, cc)] for rr in range(len(parsed["row_spans"])) for cc in range(len(parsed["col_spans"]))}) < 4:
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        row_cells = int(rng.choice(RIGHT_ROW_OPTIONS))
        left_width = int(rng.choice(LEFT_WIDTH_OPTIONS))
        right_h = row_cells * CELL_SIZE + (row_cells - 1)

        mask = _sample_mask(rng)
        mask_top = int(rng.integers(0, right_h - CELL_SIZE + 1))
        mask_left = int(rng.integers(0, left_width - CELL_SIZE))

        markers = _sample_markers(rng, row_cells * RIGHT_COLS)
        grid = _build_input(mask, row_cells, left_width, mask_top, mask_left, markers)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
