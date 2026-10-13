import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_lookup_table(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww != 4 or hh < 4 or (hh - 2) % 2 != 0:
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        frame = np.zeros((hh, ww), dtype=bool)
        frame[rows - r0, cols - c0] = True
        expected = np.zeros((hh, ww), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(frame, expected):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one corner lookup table"
    return candidates[0]


def _table_maps(grid: np.ndarray, bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape[1] == 2, "lookup table must have exactly two columns"
    left = inner[:, 0].astype(int)
    right = inner[:, 1].astype(int)
    assert np.all(left != BG) and np.all(left != GREY)
    assert np.all(right != BG) and np.all(right != GREY)
    assert len(set(map(int, left))) == len(left), "left-column keys must be unique"
    return _column_maps(left.tolist(), right.tolist())


def _column_maps(left: list[int], right: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    row_of = {int(color): idx for idx, color in enumerate(left)}
    assert all(int(color) in row_of for color in right), "vertical lookup colors must reappear as keys"
    direct = {int(src): int(dst) for src, dst in zip(left, right)}
    vertical = {}
    for src, mid in direct.items():
        mid_row = row_of[mid]
        partner_row = mid_row ^ 1
        vertical[src] = int(right[partner_row])
    return direct, vertical


def _pair_coords(grid: np.ndarray, table_bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    work = grid.copy()
    r0, c0, r1, c1 = table_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = BG
    pairs: dict[int, np.ndarray] = {}
    for color in sorted(set(map(int, work.flat)) - {BG, GREY}):
        coords = np.argwhere(work == color)
        assert coords.shape[0] == 2, "each active color must occur exactly twice"
        pairs[int(color)] = coords.astype(int)
    assert pairs, "expected at least one pixel pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each same-color pixel pair with a lookup-selected line color while preserving the corner table.

    Input structure:
    - The background is black.
    - Outside a small grey-bordered lookup table in one corner, each active non-grey color appears exactly twice as isolated pixels.
    - The lookup table is a 2-column palette inside a one-cell-thick grey frame; consecutive row pairs form swap blocks.
    - Every same-color pixel pair is aligned either horizontally or vertically.

    Rule:
    - Locate the corner lookup table and read its left column as keys and right column as direct fill colors.
    - For a horizontal pixel pair, use the pair color's direct fill from the same row.
    - For a vertical pixel pair, first take the direct fill color, then find that color as a left-column key and use the other row's right-column color from that key's two-row swap block.
    - Fill the full axis-aligned segment between the two pixels with the selected color, then restore both endpoints to their original pair color and leave the lookup table unchanged.

    Key insight:
    - The colored pixels are paired endpoints, while the grey corner object is a two-step legend: horizontal pairs use one lookup, but vertical pairs use a lookup followed by a swap to the partner row of the second key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    table_bbox = _find_lookup_table(g)
    direct, vertical = _table_maps(g, table_bbox)
    pairs = _pair_coords(g, table_bbox)

    out = g.copy()
    for color, coords in pairs.items():
        (r0, c0), (r1, c1) = coords
        if r0 == r1:
            fill = direct[color]
        else:
            assert c0 == c1, "pairs must be horizontal or vertical"
            fill = vertical[color]
        rr0, rr1 = sorted((int(r0), int(r1)))
        cc0, cc1 = sorted((int(c0), int(c1)))
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = fill
        out[int(r0), int(c0)] = color
        out[int(r1), int(c1)] = color

    tr0, tc0, tr1, tc1 = table_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    return out


def _make_table_rows(row_count: int) -> tuple[list[int], list[int]]:
    left = random.sample(COLOR_POOL, row_count)
    pair_count = row_count // 2
    groups = [left[2 * i : 2 * i + 2] for i in range(pair_count)]
    shift = random.randint(1, pair_count - 1)
    right: list[int] = []
    for idx in range(pair_count):
        target = list(groups[(idx + shift) % pair_count])
        if random.random() < 0.5:
            target.reverse()
        right.extend(target)
    return left, right


def _table_bbox(h: int, w: int, row_count: int, corner: str) -> tuple[int, int, int, int]:
    th = row_count + 2
    tw = 4
    if corner == "tl":
        return 0, 0, th - 1, tw - 1
    if corner == "tr":
        return 0, w - tw, th - 1, w - 1
    if corner == "bl":
        return h - th, 0, h - 1, tw - 1
    return h - th, w - tw, h - 1, w - 1


def _mark_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_pairs(
    h: int,
    w: int,
    colors: list[int],
    orientations: list[str],
    table_bbox: tuple[int, int, int, int],
) -> dict[int, tuple[int, int, int, int]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    _mark_reserved(reserved, table_bbox, pad=1)

    specs = []
    for color, orientation in zip(colors, orientations):
        axis = w if orientation == "h" else h
        max_len = min(10, axis)
        length = random.randint(3, max_len)
        specs.append((length, color, orientation))
    specs.sort(reverse=True)

    placements: dict[int, tuple[int, int, int, int]] = {}
    for length, color, orientation in specs:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if orientation == "h":
                r = random.randrange(h)
                c0 = random.randrange(w - length + 1)
                bbox = (r, c0, r, c0 + length - 1)
            else:
                r0 = random.randrange(h - length + 1)
                c = random.randrange(w)
                bbox = (r0, c, r0 + length - 1, c)
            if not _can_place(reserved, bbox, pad=1):
                continue
            placements[color] = bbox
            _mark_reserved(reserved, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None
    return placements


def _gen_attempt() -> np.ndarray | None:
    pair_groups = random.randint(2, 4)
    row_count = 2 * pair_groups
    min_h = max(14, row_count + 6)
    min_w = 14
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    corner = random.choice(("tl", "tr", "bl", "br"))
    table_bbox = _table_bbox(h, w, row_count, corner)

    left, right = _make_table_rows(row_count)
    colors = list(left)
    orientations = ["h"] * (row_count // 2) + ["v"] * (row_count // 2)
    random.shuffle(orientations)
    _direct, vertical = _column_maps(left, right)
    if any(orientation == "v" and vertical[color] == color for color, orientation in zip(colors, orientations)):
        return None
    placements = _place_pairs(h, w, colors, orientations, table_bbox)
    if placements is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = table_bbox
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 + 1 : r1, c0 + 1] = np.array(left, dtype=int)
    grid[r0 + 1 : r1, c0 + 2] = np.array(right, dtype=int)

    for color, bbox in placements.items():
        pr0, pc0, pr1, pc1 = bbox
        grid[pr0, pc0] = color
        grid[pr1, pc1] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
