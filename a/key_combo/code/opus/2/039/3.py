from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MAGENTA = 6
LABEL_COLORS = (BLUE, RED, YELLOW)

PLUS_GLYPH = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
HBAR_GLYPH = frozenset({(0, 0), (0, 1), (0, 2)})
VBAR_GLYPH = frozenset({(0, 0), (1, 0), (2, 0)})


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    assert len(separators) == 2
    a, b = sorted(int(x) for x in separators)
    bands = [(0, a - 1), (a + 1, b - 1), (b + 1, size - 1)]
    assert all(start <= end for start, end in bands)
    return bands


def _single_label(region: np.ndarray) -> int | None:
    coords = np.argwhere((region != BG) & (region != GREY))
    if coords.size == 0:
        return None
    assert coords.shape[0] == 1, "label regions must contain exactly one non-background pixel"
    r, c = map(int, coords[0])
    return int(region[r, c])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _normalize_shape(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def _decode_comparator(region: np.ndarray) -> str:
    mask = region == MAGENTA
    comps = _components4(mask)
    assert len(comps) == 1, "expected one central glyph"
    shape = _normalize_shape(comps[0])
    if shape == PLUS_GLYPH:
        return "equal"
    if shape in {HBAR_GLYPH, VBAR_GLYPH}:
        return "not_equal"
    raise AssertionError("unknown central glyph")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the central magenta glyph as an equality-vs-inequality comparator and fill the 2x2 body cells.

    Input structure:
    - A black background is divided by two full grey separator rows and two full grey separator columns into a 3x3 table.
    - The top-left region is empty; the other two header regions on the top row contain single-pixel column labels, and the other two header regions on the left column contain single-pixel row labels.
    - The bottom-right 2x2 regions are the body cells, and the upper-left one of those four body cells (the global center region) contains a small magenta glyph.

    Rule:
    - Decode the center glyph: a 5-pixel plus means compare by equality, while a 3-pixel horizontal or vertical bar means compare by inequality.
    - Erase the magenta glyph back to black.
    - For each body cell, compare that cell's row-label color with its column-label color.
    - In equality mode, fill the whole body cell when the two labels are equal; in inequality mode, fill the whole body cell when the two labels are different, using the row-label color.
    - Leave separator lines and header label pixels unchanged.

    Key insight:
    - The center glyph is not a color code; it selects the boolean comparison used to decide which table regions get filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    sep_rows = _full_line_indices(g, GREY, axis=0)
    sep_cols = _full_line_indices(g, GREY, axis=1)
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected a 3x3 partition grid"

    row_bands = _bands(sep_rows, g.shape[0])
    col_bands = _bands(sep_cols, g.shape[1])

    row_labels: list[int] = []
    col_labels: list[int] = []

    for band_idx in (1, 2):
        r0, r1 = row_bands[band_idx]
        c0, c1 = col_bands[0]
        label = _single_label(g[r0 : r1 + 1, c0 : c1 + 1])
        assert label in LABEL_COLORS
        row_labels.append(int(label))

    for band_idx in (1, 2):
        r0, r1 = row_bands[0]
        c0, c1 = col_bands[band_idx]
        label = _single_label(g[r0 : r1 + 1, c0 : c1 + 1])
        assert label in LABEL_COLORS
        col_labels.append(int(label))

    center_r0, center_r1 = row_bands[1]
    center_c0, center_c1 = col_bands[1]
    comparator = _decode_comparator(g[center_r0 : center_r1 + 1, center_c0 : center_c1 + 1])

    out = g.copy()
    out[center_r0 : center_r1 + 1, center_c0 : center_c1 + 1][
        out[center_r0 : center_r1 + 1, center_c0 : center_c1 + 1] == MAGENTA
    ] = BG

    for body_r in (1, 2):
        for body_c in (1, 2):
            row_color = row_labels[body_r - 1]
            col_color = col_labels[body_c - 1]
            if comparator == "equal":
                should_fill = row_color == col_color
            else:
                should_fill = row_color != col_color
            if not should_fill:
                continue

            r0, r1 = row_bands[body_r]
            c0, c1 = col_bands[body_c]
            out[r0 : r1 + 1, c0 : c1 + 1] = row_color

    return out


def _random_point(rng: random.Random, band_r: tuple[int, int], band_c: tuple[int, int]) -> tuple[int, int]:
    return rng.randint(band_r[0], band_r[1]), rng.randint(band_c[0], band_c[1])


def _choose_labels(rng: random.Random) -> tuple[list[int], list[int]]:
    while True:
        row_labels = [rng.choice(LABEL_COLORS), rng.choice(LABEL_COLORS)]
        col_labels = [rng.choice(LABEL_COLORS), rng.choice(LABEL_COLORS)]
        equal_count = sum(r == c for r in row_labels for c in col_labels)
        if 0 < equal_count < 4 and len(set(row_labels + col_labels)) >= 2:
            return row_labels, col_labels


def _place_glyph(
    grid: np.ndarray,
    rng: random.Random,
    band_r: tuple[int, int],
    band_c: tuple[int, int],
    mode: str,
) -> None:
    top = rng.randint(band_r[0], band_r[1] - 2)
    left = rng.randint(band_c[0], band_c[1] - 2)
    if mode == "equal":
        shape = PLUS_GLYPH
    else:
        shape = rng.choice((HBAR_GLYPH, VBAR_GLYPH))
    for dr, dc in shape:
        grid[top + dr, left + dc] = MAGENTA


def generate_input() -> np.ndarray:
    rng = random.Random()

    cell_h = rng.randint(3, 9)
    cell_w = rng.randint(3, 9)
    h = 3 * cell_h + 2
    w = 3 * cell_w + 2
    grid = np.full((h, w), BG, dtype=int)

    sep_rows = [cell_h, 2 * cell_h + 1]
    sep_cols = [cell_w, 2 * cell_w + 1]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    row_bands = _bands(sep_rows, h)
    col_bands = _bands(sep_cols, w)

    row_labels, col_labels = _choose_labels(rng)

    for idx, color in enumerate(col_labels, start=1):
        r, c = _random_point(rng, row_bands[0], col_bands[idx])
        grid[r, c] = color

    for idx, color in enumerate(row_labels, start=1):
        r, c = _random_point(rng, row_bands[idx], col_bands[0])
        grid[r, c] = color

    mode = rng.choice(("equal", "not_equal"))
    _place_glyph(grid, rng, row_bands[1], col_bands[1], mode)

    return grid
