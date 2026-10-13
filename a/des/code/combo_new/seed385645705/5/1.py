import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE_ORDER = ("top", "left", "right", "bottom")
MAX_GENERATION_ATTEMPTS = 300

FRAME_SIZE_MIN = 4
FRAME_SIZE_MAX = 7
ROW_GAP_RANGE = (2, 4)
COL_GAP_RANGE = (2, 4)
PALETTE_TO_FRAMES_GAP = (2, 5)
OUTER_MARGIN_RANGE = (0, 2)
PALETTE_COL_RANGE = (0, 2)
PALETTE_GAP_RANGE = (0, 1)
COLOR_POOL = tuple(c for c in range(10) if c not in {BG_COLOR, FRAME_COLOR})


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _palette_colors(grid: np.ndarray, bg: int, frame_color: int) -> list[int]:
    coords = np.argwhere((grid != bg) & (grid != frame_color))
    assert len(coords) == 4, "expected exactly four palette pixels"

    cols, counts = np.unique(coords[:, 1], return_counts=True)
    palette_col = int(cols[np.argmax(counts)])
    palette_coords = coords[coords[:, 1] == palette_col]
    assert len(palette_coords) == 4, "palette must be a 4-pixel vertical column"

    palette_coords = palette_coords[np.argsort(palette_coords[:, 0])]
    colors = [int(grid[r, c]) for r, c in palette_coords]
    assert len(set(colors)) == 4, "palette colors must be distinct"
    return colors


def _frame_gap_side(grid: np.ndarray, frame_color: int, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 3, "frame bbox must have an interior"

    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert not np.any((patch == frame_color) & ~border), "frame color must stay on the border"
    missing = np.argwhere(border & (patch != frame_color))
    assert len(missing) == 1, "frame border must have exactly one gap"
    mr, mc = map(int, missing[0])

    if mr == 0 and 0 < mc < w - 1:
        side = "top"
    elif mr == h - 1 and 0 < mc < w - 1:
        side = "bottom"
    elif mc == 0 and 0 < mr < h - 1:
        side = "left"
    elif mc == w - 1 and 0 < mr < h - 1:
        side = "right"
    else:
        raise AssertionError("gap must lie on exactly one non-corner side cell")

    return side, (r0 + mr, c0 + mc)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected non-background objects"

    frame_color = max(colors, key=lambda c: int(np.count_nonzero(g == c)))
    palette = _palette_colors(g, bg, frame_color)
    fill_by_side = dict(zip(PALETTE_ORDER, palette))

    out = g.copy()
    for coords in _connected_components(g == frame_color):
        bbox = _bbox(coords)
        side, gap_cell = _frame_gap_side(g, frame_color, bbox)
        fill_color = int(fill_by_side[side])
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        out[gap_cell] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow gray frame using the palette color keyed by its border-gap direction.

    Input structure:
    - The background is black.
    - A vertical palette on the left consists of four distinct 1x1 colored pixels read from top to bottom, optionally separated by black rows.
    - On the right are several gray one-pixel-thick rectangular hollow frames on black background.
    - Each frame has exactly one missing gray border pixel on one non-corner side cell: top, left, right, or bottom.

    Rule:
    - Read the palette top-to-bottom as a fixed direction mapping in the order top, left, right, bottom.
    - For each gray frame, locate the unique missing border cell and determine which side contains it.
    - Fill the entire interior of that frame with the mapped palette color, and also fill the missing border cell with the same color.
    - Keep the palette pixels and all remaining gray border pixels unchanged.

    Key insight:
    The frame is not keyed by its size or position; the only meaningful cue is which side of the border contains the single-pixel gap, and the palette order encodes the direction-to-color lookup.
    """
    return _solve(grid)


def _gap_cell(rng: random.Random, r: int, c: int, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return r, c + rng.randint(1, w - 2)
    if side == "bottom":
        return r + h - 1, c + rng.randint(1, w - 2)
    if side == "left":
        return r + rng.randint(1, h - 2), c
    if side == "right":
        return r + rng.randint(1, h - 2), c + w - 1
    raise ValueError(f"unknown side: {side}")


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int, side: str, rng: random.Random) -> None:
    grid[r, c : c + w] = FRAME_COLOR
    grid[r + h - 1, c : c + w] = FRAME_COLOR
    grid[r : r + h, c] = FRAME_COLOR
    grid[r : r + h, c + w - 1] = FRAME_COLOR
    gr, gc = _gap_cell(rng, r, c, h, w, side)
    grid[gr, gc] = BG_COLOR


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_colors = rng.sample(COLOR_POOL, 4)
    directions = list(PALETTE_ORDER)
    rng.shuffle(directions)
    frame_sizes = [
        (
            rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
            rng.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
        )
        for _ in range(4)
    ]

    row_gap = rng.randint(*ROW_GAP_RANGE)
    col_gap = rng.randint(*COL_GAP_RANGE)
    palette_gap = rng.randint(*PALETTE_TO_FRAMES_GAP)
    top_margin = rng.randint(*OUTER_MARGIN_RANGE)
    right_margin = rng.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = rng.randint(*OUTER_MARGIN_RANGE)
    palette_col = rng.randint(*PALETTE_COL_RANGE)

    row_heights = [
        max(frame_sizes[0][0], frame_sizes[1][0]),
        max(frame_sizes[2][0], frame_sizes[3][0]),
    ]
    col_widths = [
        max(frame_sizes[0][1], frame_sizes[2][1]),
        max(frame_sizes[1][1], frame_sizes[3][1]),
    ]

    cluster_left = palette_col + 1 + palette_gap
    cluster_width = col_widths[0] + col_gap + col_widths[1]
    cluster_height = row_heights[0] + row_gap + row_heights[1]
    min_w = cluster_left + cluster_width + right_margin
    min_h = top_margin + cluster_height + bottom_margin

    palette_row_gaps = [rng.randint(*PALETTE_GAP_RANGE) for _ in range(3)]
    palette_height = 4 + sum(palette_row_gaps)
    base_h = max(min_h, palette_height)

    if min_w > 30 or base_h > 30:
        return None

    h = base_h + rng.randint(0, 30 - base_h)
    w = min_w + rng.randint(0, 30 - min_w)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    palette_top = rng.randint(0, h - palette_height)
    row = palette_top
    for i, color in enumerate(palette_colors):
        grid[row, palette_col] = color
        if i < 3:
            row += 1 + palette_row_gaps[i]

    cell_tops = [top_margin, top_margin, top_margin + row_heights[0] + row_gap, top_margin + row_heights[0] + row_gap]
    cell_lefts = [cluster_left, cluster_left + col_widths[0] + col_gap, cluster_left, cluster_left + col_widths[0] + col_gap]

    for index, side in enumerate(directions):
        cell_h = row_heights[0] if index < 2 else row_heights[1]
        cell_w = col_widths[0] if index in (0, 2) else col_widths[1]
        fh, fw = frame_sizes[index]
        r = cell_tops[index] + rng.randint(0, cell_h - fh)
        c = cell_lefts[index] + rng.randint(0, cell_w - fw)
        _draw_frame(grid, r, c, fh, fw, side, rng)

    try:
        _solve(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
