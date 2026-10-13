from __future__ import annotations

import random

import numpy as np


BG = 5
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
SEPARATOR = 8

PALETTE = (BLUE, RED, GREEN, ORANGE, PINK)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best: tuple[int, int, int, int] | None = None
    best_area = -1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                area += 1
                r0 = min(r0, rr)
                c0 = min(c0, cc)
                r1 = max(r1, rr)
                c1 = max(c1, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if area > best_area:
                best_area = area
                best = (r0, c0, r1, c1)
    assert best is not None, "expected a separator lattice"
    return best


def _visible_colors(line: np.ndarray, bg: int, separator: int) -> frozenset[int]:
    return frozenset(int(v) for v in line if int(v) not in (bg, separator))


def _axis_bands(grid: np.ndarray, bbox: tuple[int, int, int, int], axis: int, bg: int, separator: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    signatures: list[tuple[int, frozenset[int]]] = []
    if axis == 0:
        for r in range(r0, r1 + 1):
            colors = _visible_colors(grid[r, c0 : c1 + 1], bg, separator)
            if colors:
                signatures.append((r, colors))
    else:
        for c in range(c0, c1 + 1):
            colors = _visible_colors(grid[r0 : r1 + 1, c], bg, separator)
            if colors:
                signatures.append((c, colors))

    assert signatures, "expected visible cell bands"

    bands: list[tuple[int, int]] = []
    start = prev = signatures[0][0]
    current = signatures[0][1]
    for idx, colors in signatures[1:]:
        if idx != prev + 1 or colors != current:
            bands.append((start, prev))
            start = idx
            current = colors
        prev = idx
    bands.append((start, prev))
    return bands


def _cell_color(
    grid: np.ndarray,
    rows: tuple[int, int],
    cols: tuple[int, int],
    bg: int,
    separator: int,
) -> int:
    r0, r1 = rows
    c0, c1 = cols
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = sorted(int(v) for v in np.unique(region) if int(v) not in (bg, separator))
    assert len(colors) == 1, "each coarse cell must decode to one foreground color"
    return int(colors[0])


def _compress_lattice(grid: np.ndarray, bg: int, separator: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _largest_component_bbox(grid == separator)
    row_bands = _axis_bands(grid, bbox, axis=0, bg=bg, separator=separator)
    col_bands = _axis_bands(grid, bbox, axis=1, bg=bg, separator=separator)
    coarse = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for rr, row_band in enumerate(row_bands):
        for cc, col_band in enumerate(col_bands):
            coarse[rr, cc] = _cell_color(grid, row_band, col_band, bg, separator)
    return coarse, bbox


def _yellow_span(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    cols = [c for c in range(w) if int(grid[h - 1, c]) == YELLOW]
    assert cols, "expected a yellow bottom segment"
    assert cols == list(range(cols[0], cols[-1] + 1)), "yellow segment must be contiguous"
    return h - 1, int(cols[0]), int(cols[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the upper lattice, read its bottom-row colors as a cycle, and propagate matching stripes upward from the bottom yellow segment.

    Input structure:
    - Grey background surrounds one upper rectangular lattice built from teal separator lines and monochrome cell blocks.
    - Each cell block uses one foreground color from a small palette and may contain grey holes, while rows and columns inside the same band share the same visible foreground-color set.
    - The lower area is otherwise grey and contains one contiguous yellow segment on the bottom edge.

    Rule:
    - Detect the teal lattice, group its non-separator rows and columns into consecutive bands by visible foreground-color sets, and collapse each band intersection to its unique foreground color to obtain a coarse grid.
    - Read the coarse grid's bottom row from left to right as the cyclic stripe palette.
    - Keep the lattice and the yellow segment unchanged, then paint one-row horizontal stripes only within the yellow segment's column span on every other row above the segment, cycling through the palette.
    - Leave the one-row gaps between stripes grey and stop before the stripes would touch or enter the lattice bbox.

    Key insight:
    - The stripe colors are not preset: they are encoded by the compressed lattice's bottom row, so the lattice must be decoded before the lower propagation can be determined.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    coarse, lattice_bbox = _compress_lattice(g, bg=bg, separator=SEPARATOR)
    palette = [int(v) for v in coarse[-1]]
    seg_row, c0, c1 = _yellow_span(g)
    lattice_bottom = int(lattice_bbox[2])

    out = g.copy()
    for idx, r in enumerate(range(seg_row - 2, lattice_bottom, -2)):
        out[r, c0 : c1 + 1] = int(palette[idx % len(palette)])
        gap_row = r + 1
        if lattice_bottom < gap_row < seg_row:
            out[gap_row, c0 : c1 + 1] = int(bg)
    return out


def _line_color_set(arr: np.ndarray, axis: int, idx: int) -> frozenset[int]:
    line = arr[idx, :] if axis == 0 else arr[:, idx]
    return frozenset(int(v) for v in line)


def _valid_coarse(coarse: np.ndarray) -> bool:
    if coarse.shape[0] < 3 or coarse.shape[1] < 3:
        return False
    if len(set(int(v) for v in coarse[-1])) < 2:
        return False
    if len(set(int(v) for v in coarse.flat)) < 3:
        return False
    for r in range(1, coarse.shape[0]):
        if _line_color_set(coarse, axis=0, idx=r) == _line_color_set(coarse, axis=0, idx=r - 1):
            return False
    for c in range(1, coarse.shape[1]):
        if _line_color_set(coarse, axis=1, idx=c) == _line_color_set(coarse, axis=1, idx=c - 1):
            return False
    return True


def _sample_coarse(rng: random.Random) -> np.ndarray:
    while True:
        rows = rng.randint(3, 5)
        cols = rng.randint(3, 5)
        coarse = np.array(
            [[rng.choice(PALETTE) for _ in range(cols)] for _ in range(rows)],
            dtype=int,
        )
        if _valid_coarse(coarse):
            return coarse


def _fill_cell_with_holes(
    lattice: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    rng: random.Random,
    force_holes: bool,
) -> None:
    lattice[r0 : r0 + h, c0 : c0 + w] = int(color)
    keep_r = r0 + rng.randrange(h)
    keep_c = c0 + rng.randrange(w)
    hole_prob = rng.choice((0.0, 0.0, 0.1, 0.15, 0.2, 0.25))
    made_hole = False
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            if rr == keep_r or cc == keep_c:
                continue
            if rng.random() < hole_prob:
                lattice[rr, cc] = BG
                made_hole = True
    if force_holes and not made_hole and h * w >= 6:
        candidates = [
            (rr, cc)
            for rr in range(r0, r0 + h)
            for cc in range(c0, c0 + w)
            if rr != keep_r and cc != keep_c
        ]
        if candidates:
            rr, cc = rng.choice(candidates)
            lattice[rr, cc] = BG


def _render_lattice(
    coarse: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    rng: random.Random,
) -> np.ndarray:
    rows, cols = coarse.shape
    lattice_h = sum(row_heights) + rows + 1
    lattice_w = sum(col_widths) + cols + 1
    lattice = np.full((lattice_h, lattice_w), SEPARATOR, dtype=int)
    hole_cells = rng.sample(
        [(r, c) for r in range(rows) for c in range(cols)],
        k=rng.randint(1, max(1, rows * cols // 2)),
    )
    hole_cells = set(hole_cells)

    r = 1
    for rr in range(rows):
        c = 1
        for cc in range(cols):
            _fill_cell_with_holes(
                lattice,
                r,
                c,
                row_heights[rr],
                col_widths[cc],
                int(coarse[rr, cc]),
                rng,
                force_holes=(rr, cc) in hole_cells,
            )
            c += col_widths[cc] + 1
        r += row_heights[rr] + 1
    return lattice


def _manual_output(
    grid: np.ndarray,
    coarse: np.ndarray,
    lattice_bbox: tuple[int, int, int, int],
    span: tuple[int, int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    seg_row, c0, c1 = span
    palette = [int(v) for v in coarse[-1]]
    lattice_bottom = int(lattice_bbox[2])
    for idx, r in enumerate(range(seg_row - 2, lattice_bottom, -2)):
        out[r, c0 : c1 + 1] = int(palette[idx % len(palette)])
        gap_row = r + 1
        if lattice_bottom < gap_row < seg_row:
            out[gap_row, c0 : c1 + 1] = BG
    return out


def _validate_instance(
    grid: np.ndarray,
    coarse: np.ndarray,
    lattice_bbox: tuple[int, int, int, int],
    span: tuple[int, int, int],
) -> bool:
    try:
        decoded, bbox = _compress_lattice(grid, bg=BG, separator=SEPARATOR)
    except Exception:
        return False
    if bbox != lattice_bbox or not np.array_equal(decoded, coarse):
        return False

    seg_row, c0, c1 = span
    if seg_row != grid.shape[0] - 1 or not (3 <= c1 - c0 + 1 <= 5):
        return False
    if c0 < 1 or c1 > grid.shape[1] - 2:
        return False

    lattice_bottom = lattice_bbox[2]
    input_bg_gap = grid.shape[0] - lattice_bottom - 2
    if input_bg_gap < 2:
        return False

    stripe_rows = list(range(seg_row - 2, lattice_bottom, -2))
    if len(stripe_rows) < 2:
        return False

    expected = _manual_output(grid, coarse, lattice_bbox, span)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if not np.array_equal(actual[: lattice_bottom + 1], grid[: lattice_bottom + 1]):
        return False
    if not np.array_equal(actual[seg_row], grid[seg_row]):
        return False

    palette = [int(v) for v in coarse[-1]]
    for idx, r in enumerate(stripe_rows):
        if not np.all(actual[r, c0 : c1 + 1] == palette[idx % len(palette)]):
            return False
        if np.any(actual[r, :c0] != BG) or np.any(actual[r, c1 + 1 :] != BG):
            return False
        gap_row = r + 1
        if lattice_bottom < gap_row < seg_row and not np.all(actual[gap_row, c0 : c1 + 1] == BG):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse = _sample_coarse(rng)
    row_heights = [rng.randint(2, 4) for _ in range(coarse.shape[0])]
    col_widths = [rng.randint(2, 4) for _ in range(coarse.shape[1])]
    lattice = _render_lattice(coarse, row_heights, col_widths, rng)

    top_pad = rng.randint(1, 3)
    left_pad = rng.randint(1, 4)
    right_pad = rng.randint(1, 4)
    lattice_bottom = top_pad + lattice.shape[0] - 1

    palette_len = coarse.shape[1]
    max_stripes = palette_len + 2
    stripe_count = rng.randint(2, max_stripes)
    if rng.random() < 0.5 and max_stripes > palette_len:
        stripe_count = rng.randint(palette_len + 1, max_stripes)
    clearance = rng.randint(1, 3)
    height = lattice_bottom + clearance + 2 * stripe_count + 1
    if height > 30:
        return None
    if not (0.42 <= (lattice_bottom + 1) / height <= 0.68):
        return None

    min_width = left_pad + lattice.shape[1] + right_pad
    width = max(min_width, rng.randint(18, 28))
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[top_pad : top_pad + lattice.shape[0], left_pad : left_pad + lattice.shape[1]] = lattice

    seg_len = rng.randint(3, 5)
    if width - seg_len - 2 < 1:
        return None
    seg_start = rng.randint(1, width - seg_len - 1)
    grid[height - 1, seg_start : seg_start + seg_len] = YELLOW

    lattice_bbox = (
        top_pad,
        left_pad,
        top_pad + lattice.shape[0] - 1,
        left_pad + lattice.shape[1] - 1,
    )
    span = (height - 1, seg_start, seg_start + seg_len - 1)

    if not _validate_instance(grid, coarse, lattice_bbox, span):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
