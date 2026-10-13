from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

RECT_EXTENT_RANGE = (2, 5)
MARGIN_EXTRA_RANGE = (3, 6)
GENERATION_ATTEMPTS = 300
BLOB_ATTEMPTS = 200


def _gray_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one gray rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "gray cells must form a solid rectangle"
    assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1), "gray rectangle must have no holes"
    return int(r0), int(c0), int(r1), int(c1)


def _sector_info(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> list[tuple[int, int]]:
    sectors = [
        grid[:r0, :c0],
        grid[:r0, c1 + 1 :],
        grid[r1 + 1 :, :c0],
        grid[r1 + 1 :, c1 + 1 :],
    ]
    info: list[tuple[int, int]] = []
    for sector in sectors:
        colors = [int(color) for color in np.unique(sector) if int(color) != BLACK]
        assert len(colors) == 1, "expected exactly one color pile in each corner sector"
        color = colors[0]
        assert color not in {GREY, PINK}, "scatter colors must stay distinct from gray and pink"
        info.append((color, int(np.count_nonzero(sector == color))))
    return info


def _find_cross(
    rect_bbox: tuple[int, int, int, int], counts: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    candidates: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        top = r - r0
        bottom = r1 - r
        for c in range(c0 + 1, c1):
            left = c - c0
            right = c1 - c
            areas = (
                top * left,
                top * right,
                bottom * left,
                bottom * right,
            )
            if areas == counts:
                candidates.append((r, c))
    assert len(candidates) == 1, "expected one unique invisible cross"
    return candidates[0]


def _render_output(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    cross: tuple[int, int],
    colors: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = rect_bbox
    r, c = cross
    tl, tr, bl, br = colors

    out = np.full(shape, BLACK, dtype=int)
    out[r0:r, c0:c] = tl
    out[r0:r, c + 1 : c1 + 1] = tr
    out[r + 1 : r1 + 1, c0:c] = bl
    out[r + 1 : r1 + 1, c + 1 : c1 + 1] = br
    out[r, c0 : c1 + 1] = PINK
    out[r0 : r1 + 1, c] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the invisible split cross inside the gray rectangle from four corner pile counts, then recolor the rooms.

    Input structure:
    - The background is black.
    - One solid gray rectangle appears with no internal marks.
    - Four other colors appear as scattered pixel piles in the four outer corner regions around that rectangle, one color pile per region.

    Rule:
    - Count the pixels in the top-left, top-right, bottom-left, and bottom-right corner piles.
    - Find the unique internal row and column inside the gray rectangle whose four quadrant areas match those four counts in the same corner order.
    - Erase the piles, paint the chosen row and column pink inside the rectangle, and fill each quadrant with the color of its matching pile.

    Key insight:
    - The split cross is never drawn directly; the hidden row and column are recovered by solving the four area equations induced by the corner-pile counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect_bbox = _gray_bbox(g)
    r0, c0, r1, c1 = rect_bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "gray rectangle must admit an internal cross"

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    assert not np.any(outside == GREY), "gray should appear only in the rectangle"
    assert not np.any(outside == PINK), "pink is reserved for the output cross"

    side_bands = (
        g[:r0, c0 : c1 + 1],
        g[r1 + 1 :, c0 : c1 + 1],
        g[r0 : r1 + 1, :c0],
        g[r0 : r1 + 1, c1 + 1 :],
    )
    assert all(np.all(band == BLACK) for band in side_bands), "scatter piles must stay in the four corner sectors"

    sector_info = _sector_info(g, r0, c0, r1, c1)
    colors = tuple(color for color, _count in sector_info)
    counts = tuple(count for _color, count in sector_info)
    assert len(set(colors)) == 4, "expected four distinct scatter colors"

    cross = _find_cross(rect_bbox, counts)
    return _render_output(g.shape, rect_bbox, cross, colors)


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _random_blob(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(BLOB_ATTEMPTS):
        coords = {(0, 0)}
        while len(coords) < area:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in coords:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coords or (nr, nc) in seen:
                        continue
                    trial = coords | {(nr, nc)}
                    rows = [rr for rr, _cc in trial]
                    cols = [cc for _rr, cc in trial]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) != area:
            continue

        mask = _normalize(coords)
        for _ in range(random.randrange(4)):
            mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask.copy()
    return None


def _paint_blob(
    grid: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    color: int,
    area: int,
) -> bool:
    zone_h = row_hi - row_lo
    zone_w = col_hi - col_lo
    if zone_h <= 0 or zone_w <= 0 or zone_h * zone_w < area:
        return False
    blob = _random_blob(area, zone_h, zone_w)
    if blob is None:
        return False
    dh = zone_h - blob.shape[0]
    dw = zone_w - blob.shape[1]
    r0 = row_lo + random.randint(0, dh)
    c0 = col_lo + random.randint(0, dw)
    patch = grid[r0 : r0 + blob.shape[0], c0 : c0 + blob.shape[1]]
    assert not np.any((patch != BLACK) & blob)
    patch[blob] = color
    return True


def _candidate_counts(top: int, bottom: int, left: int, right: int) -> tuple[int, int, int, int]:
    return (
        top * left,
        top * right,
        bottom * left,
        bottom * right,
    )


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(*RECT_EXTENT_RANGE)
    bottom = random.randint(*RECT_EXTENT_RANGE)
    left = random.randint(*RECT_EXTENT_RANGE)
    right = random.randint(*RECT_EXTENT_RANGE)

    rect_h = top + 1 + bottom
    rect_w = left + 1 + right
    counts = _candidate_counts(top, bottom, left, right)

    top_margin = top + random.randint(*MARGIN_EXTRA_RANGE)
    bottom_margin = bottom + random.randint(*MARGIN_EXTRA_RANGE)
    left_margin = left + random.randint(*MARGIN_EXTRA_RANGE)
    right_margin = right + random.randint(*MARGIN_EXTRA_RANGE)

    height = top_margin + rect_h + bottom_margin
    width = left_margin + rect_w + right_margin
    if height > 30 or width > 30:
        return None

    rect_r0 = top_margin
    rect_c0 = left_margin
    rect_r1 = rect_r0 + rect_h - 1
    rect_c1 = rect_c0 + rect_w - 1
    rect_bbox = (rect_r0, rect_c0, rect_r1, rect_c1)
    cross = (rect_r0 + top, rect_c0 + left)

    colors = random.sample(PALETTE, 4)
    inp = np.full((height, width), BLACK, dtype=int)
    inp[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = GREY

    placements = [
        (0, rect_r0 - 1, 0, rect_c0 - 1),
        (0, rect_r0 - 1, rect_c1 + 2, width),
        (rect_r1 + 2, height, 0, rect_c0 - 1),
        (rect_r1 + 2, height, rect_c1 + 2, width),
    ]
    for color, area, (row_lo, row_hi, col_lo, col_hi) in zip(colors, counts, placements, strict=True):
        if not _paint_blob(inp, row_lo, row_hi, col_lo, col_hi, color, area):
            return None

    expected = _render_output(inp.shape, rect_bbox, cross, tuple(colors))

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-cross input")
