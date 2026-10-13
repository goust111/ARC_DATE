from __future__ import annotations

import random

import numpy as np


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected at least one active cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _false_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(list(map(bool, flags)) + [True]):
        if not flag and start is None:
            start = idx
        elif flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def _extract_candidate(grid: np.ndarray, bg: int, sep: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of(grid != bg)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_sep = np.all(sub == sep, axis=1)
    col_sep = np.all(sub == sep, axis=0)
    assert not row_sep[0] and not row_sep[-1], "separator rows cannot be at the outer boundary"
    assert not col_sep[0] and not col_sep[-1], "separator cols cannot be at the outer boundary"

    row_tiles = _false_runs(row_sep)
    col_tiles = _false_runs(col_sep)
    assert row_tiles and col_tiles
    assert len(row_tiles) > 1 or len(col_tiles) > 1, "expected multiple tiles"

    row_heights = {rr1 - rr0 + 1 for rr0, rr1 in row_tiles}
    col_widths = {cc1 - cc0 + 1 for cc0, cc1 in col_tiles}
    assert len(row_heights) == 1 and len(col_widths) == 1, "tiles must share one size"
    tile_h = row_heights.pop()
    tile_w = col_widths.pop()

    out = np.full((tile_h, tile_w), bg, dtype=int)
    seen_colors: list[int] = []
    for rr0, rr1 in row_tiles:
        for cc0, cc1 in col_tiles:
            patch = sub[rr0 : rr1 + 1, cc0 : cc1 + 1]
            assert patch.shape == (tile_h, tile_w)
            assert not np.any(patch == sep), "separator color cannot enter tile interiors"

            colors = set(map(int, np.unique(patch))) - {bg}
            assert len(colors) == 1, "each tile must use exactly one foreground color"
            color = next(iter(colors))
            seen_colors.append(color)

            mask = patch != bg
            assert mask.any(), "every tile must contribute at least one pixel"
            assert not np.any(out[mask] != bg), "tile contributions must be complementary"
            out[mask] = patch[mask]

    assert len(set(seen_colors)) == len(seen_colors), "tile colors must be distinct"
    assert np.any(out != bg), "output cannot be empty"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay equal-sized monochrome tiles separated by full gray lines and recover the complete tile.

    Input structure
    - Background is black.
    - Several equal-sized rectangular tiles are arranged in a row/column grid.
    - Full separator rows/columns use one structural color.
    - Inside each tile, one foreground color marks a subset of a shared shape and black marks missing cells.

    Rule
    - Identify the separator color and partition the grid into equal tile rectangles.
    - For each local tile position, exactly one tile contributes a foreground pixel.
    - Output one tile-sized grid formed by overlaying all contributions.

    Key insight
    - The tiles are complementary slices of one hidden shape, so the solution is obtained by aligned superposition.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0 if np.any(g == 0) else int(np.bincount(g.ravel()).argmax())

    candidates: list[np.ndarray] = []
    for sep in sorted(set(map(int, np.unique(g))) - {bg}):
        try:
            candidates.append(_extract_candidate(g, bg, sep))
        except AssertionError:
            continue

    assert len(candidates) == 1, f"expected one valid separator color, found {len(candidates)}"
    return candidates[0]


# --------------- generator ---------------

BG = 0
SEP = 5
TILE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

LAYOUT_CHOICES = ((1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2))
TILE_H_RANGE = range(5, 9)
TILE_W_RANGE = range(5, 9)
SEP_WIDTH_CHOICES = (1, 2)
PADDING_CHOICES = (1, 2, 3)
MAX_SIDE = 28

MASK_MODES = ("cross", "el", "frame", "fork", "polyline")
ASSIGN_MODES = ("cyclic", "shuffle")

MIN_DENSITY = 0.25
MAX_DENSITY = 0.65
GEN_RETRIES = 300
MASK_RETRIES = 120
LAYOUT_RETRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    path = [(r, c)]
    while (r, c) != end:
        step_options: list[tuple[int, int]] = []
        if r != end[0]:
            step_options.append((r + (1 if end[0] > r else -1), c))
        if c != end[1]:
            step_options.append((r, c + (1 if end[1] > c else -1)))
        r, c = random.choice(step_options)
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for rr, cc in _neighbors4(int(r), int(c), h, w):
            if not mask[rr, cc]:
                cells.add((rr, cc))
    return list(cells)


def _grow_mask(mask: np.ndarray, target_area: int) -> np.ndarray:
    out = mask.copy()
    while int(out.sum()) < target_area:
        border = _frontier(out)
        if not border:
            break
        rr, cc = random.choice(border)
        out[rr, cc] = True
    return out


def _mask_cross(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    thickness = 1 if min(h, w) <= 6 else random.choice((1, 2))
    r0 = random.randint(0, h - thickness)
    c0 = random.randint(0, w - thickness)
    mask[:, c0 : c0 + thickness] = True
    mask[r0 : r0 + thickness, :] = True
    return _random_transform(mask)


def _mask_el(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    v_thick = 1 if w <= 6 else random.choice((1, 2))
    h_thick = 1 if h <= 6 else random.choice((1, 2))
    mask[:, :v_thick] = True
    mask[h - h_thick :, :] = True
    return _random_transform(mask)


def _mask_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mask_fork(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    trunk_w = 1 if w <= 6 else random.choice((1, 2))
    trunk_c = random.randint(0, w - trunk_w)
    main_r = random.randint(0, h - 1)
    mask[:, trunk_c : trunk_c + trunk_w] = True
    mask[main_r, :] = True

    extra_rows = [r for r in range(h) if r != main_r]
    random.shuffle(extra_rows)
    for row in extra_rows[: random.choice((1, 2))]:
        if random.random() < 0.5:
            mask[row, : trunk_c + trunk_w] = True
        else:
            mask[row, trunk_c:] = True
    return _random_transform(mask)


def _mask_polyline(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = (random.randint(1, h - 2), random.randint(1, w - 2))
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(anchors)
    for anchor in anchors:
        for rr, cc in _random_path(anchor, center):
            mask[rr, cc] = True
    return _random_transform(mask)


def _make_mask(h: int, w: int, n_tiles: int) -> np.ndarray | None:
    min_area = max(2 * n_tiles, int(np.ceil(MIN_DENSITY * h * w)))
    max_area = int(np.floor(MAX_DENSITY * h * w))

    if min_area > max_area:
        return None

    builders = {
        "cross": _mask_cross,
        "el": _mask_el,
        "frame": _mask_frame,
        "fork": _mask_fork,
        "polyline": _mask_polyline,
    }

    for _ in range(MASK_RETRIES):
        mask = builders[random.choice(MASK_MODES)](h, w)
        area = int(mask.sum())
        if area > max_area:
            continue

        target_area = random.randint(max(min_area, area), max_area)
        mask = _grow_mask(mask, target_area)
        coords = np.argwhere(mask)
        if int(mask.sum()) < min_area or int(mask.sum()) > max_area:
            continue
        if not _touches_all_sides(mask):
            continue
        if len(set(map(int, coords[:, 0]))) < 3 or len(set(map(int, coords[:, 1]))) < 3:
            continue
        if np.all(mask):
            continue
        return mask
    return None


def _balanced_sizes(total: int, parts: int) -> list[int]:
    base = total // parts
    sizes = [base] * parts
    for idx in random.sample(range(parts), total % parts):
        sizes[idx] += 1
    assert min(sizes) >= 1
    return sizes


def _partition_coords(mask: np.ndarray, n_tiles: int) -> list[list[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) >= n_tiles

    mode = random.choice(ASSIGN_MODES)
    groups: list[list[tuple[int, int]]] = [[] for _ in range(n_tiles)]

    if mode == "cyclic":
        a, b = random.choice(((1, 1), (1, 2), (2, 1), (1, -1), (2, -1)))
        ordered = sorted(coords, key=lambda rc: (a * rc[0] + b * rc[1], rc[0], rc[1]))
        slots = list(range(n_tiles))
        random.shuffle(slots)
        for idx, coord in enumerate(ordered):
            groups[slots[idx % n_tiles]].append(coord)
        return groups

    shuffled = coords[:]
    random.shuffle(shuffled)
    sizes = _balanced_sizes(len(shuffled), n_tiles)
    pos = 0
    for idx, size in enumerate(sizes):
        groups[idx] = shuffled[pos : pos + size]
        pos += size
    return groups


def _sample_layout() -> tuple[int, int, int, int, int, int, int, int] | None:
    for _ in range(LAYOUT_RETRIES):
        rows, cols = random.choice(LAYOUT_CHOICES)
        tile_h = random.choice(tuple(TILE_H_RANGE))
        tile_w = random.choice(tuple(TILE_W_RANGE))
        sep_w = random.choice(SEP_WIDTH_CHOICES)
        pad_t = random.choice(PADDING_CHOICES)
        pad_b = random.choice(PADDING_CHOICES)
        pad_l = random.choice(PADDING_CHOICES)
        pad_r = random.choice(PADDING_CHOICES)

        full_h = rows * tile_h + (rows - 1) * sep_w + pad_t + pad_b
        full_w = cols * tile_w + (cols - 1) * sep_w + pad_l + pad_r
        if full_h <= MAX_SIDE and full_w <= MAX_SIDE:
            return rows, cols, tile_h, tile_w, sep_w, pad_t, pad_l, full_h, full_w
    return None


def _build_input(
    patches: list[np.ndarray],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    sep_w: int,
    pad_t: int,
    pad_l: int,
    full_h: int,
    full_w: int,
) -> np.ndarray:
    grid = np.full((full_h, full_w), BG, dtype=int)

    arr_r0 = pad_t
    arr_c0 = pad_l
    arr_r1 = arr_r0 + rows * tile_h + (rows - 1) * sep_w - 1
    arr_c1 = arr_c0 + cols * tile_w + (cols - 1) * sep_w - 1

    for split_r in range(1, rows):
        start = arr_r0 + split_r * tile_h + (split_r - 1) * sep_w
        grid[start : start + sep_w, arr_c0 : arr_c1 + 1] = SEP
    for split_c in range(1, cols):
        start = arr_c0 + split_c * tile_w + (split_c - 1) * sep_w
        grid[arr_r0 : arr_r1 + 1, start : start + sep_w] = SEP

    idx = 0
    for row in range(rows):
        for col in range(cols):
            top = arr_r0 + row * (tile_h + sep_w)
            left = arr_c0 + col * (tile_w + sep_w)
            grid[top : top + tile_h, left : left + tile_w] = patches[idx]
            idx += 1

    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    rows, cols, tile_h, tile_w, sep_w, pad_t, pad_l, full_h, full_w = layout
    n_tiles = rows * cols
    colors = random.sample(TILE_COLORS, n_tiles)

    mask = _make_mask(tile_h, tile_w, n_tiles)
    if mask is None:
        return None

    groups = _partition_coords(mask, n_tiles)
    if any(len(group) == 0 for group in groups):
        return None

    expected = np.full((tile_h, tile_w), BG, dtype=int)
    patches: list[np.ndarray] = []
    for color, group in zip(colors, groups):
        patch = np.full((tile_h, tile_w), BG, dtype=int)
        rr = [r for r, _c in group]
        cc = [c for _r, c in group]
        patch[rr, cc] = color
        expected[rr, cc] = color
        patches.append(patch)

    grid = _build_input(patches, rows, cols, tile_h, tile_w, sep_w, pad_t, pad_l, full_h, full_w)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-tile input")
