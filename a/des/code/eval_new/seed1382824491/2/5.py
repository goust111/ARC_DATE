import numpy as np


BG = 0
DIVIDER = 8
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
RNG = np.random.default_rng()
GEN_RETRIES = 400
SHAPE_RETRIES = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_masks(component_mask: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    bg_mask = ~component_mask
    for coords in _components(bg_mask):
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == component_mask.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == component_mask.shape[1] - 1)
        )
        if not touches_edge:
            holes.append(_coords_to_mask(coords))
    return holes


def _matches_under_rotation(piece: np.ndarray, cavity: np.ndarray) -> bool:
    for k in range(4):
        rotated = np.rot90(piece, k)
        if rotated.shape == cavity.shape and np.array_equal(rotated, cavity):
            return True
    return False


def _palette_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == DIVIDER)]
    assert rows, "expected at least one full-width divider row"
    return int(rows[-1])


def _region_spans(grid: np.ndarray, palette_sep: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    region_sep_rows = [r for r in range(palette_sep) if np.all(grid[r] == DIVIDER)]
    divider_cols = [c for c in range(w) if np.all(grid[: palette_sep + 1, c] == DIVIDER)]

    row_bounds = [-1] + region_sep_rows + [palette_sep]
    col_bounds = [-1] + divider_cols + [w]

    spans: list[tuple[int, int, int, int]] = []
    for i in range(len(row_bounds) - 1):
        r0 = row_bounds[i] + 1
        r1 = row_bounds[i + 1]
        if r0 >= r1:
            continue
        for j in range(len(col_bounds) - 1):
            c0 = col_bounds[j] + 1
            c1 = col_bounds[j + 1]
            if c0 < c1:
                spans.append((r0, r1, c0, c1))
    assert spans, "expected at least one puzzle region"
    return spans


def _palette_colors(grid: np.ndarray, palette_sep: int) -> list[int]:
    palette_area = grid[palette_sep + 1 :]
    coords = np.argwhere((palette_area != BG) & (palette_area != DIVIDER))
    assert coords.size > 0, "expected palette cells below the last divider row"
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    return [int(palette_area[r, c]) for r, c in coords]


def _region_matches(region: np.ndarray) -> bool:
    comps = _components(region != BG)
    assert len(comps) == 2, "each region must contain exactly two components"

    cavity = None
    piece = None
    for coords in comps:
        mask = _coords_to_mask(coords)
        holes = _hole_masks(mask)
        if len(holes) == 1:
            assert cavity is None, "expected a unique hollow frame"
            cavity = holes[0]
        else:
            assert len(holes) == 0, "solid piece must not contain holes"
            assert piece is None, "expected a unique solid piece"
            piece = mask

    assert cavity is not None and piece is not None
    return _matches_under_rotation(piece, cavity)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each table cell by testing whether its solid piece matches the hollow frame cavity.

    Input structure:
    - Background is black.
    - Azure one-cell-wide horizontal and vertical divider lines split the upper part of the grid into rectangular regions.
    - Each region contains exactly two 4-connected sprites: one hollow frame with a single enclosed black cavity, and one separate solid piece.
    - Below the last horizontal divider is a palette row of single colored cells ordered from left to right.

    Rule:
    - For each region, extract the hollow frame's enclosed cavity as a tight binary mask and extract the solid piece as its own tight binary mask.
    - If the solid piece equals the cavity under a rotation of 0, 90, 180, or 270 degrees, treat the region as a match; otherwise it is a mismatch.
    - Fill every black cell in a matching region with the palette's first color and every black cell in a mismatching region with the palette's second color.
    - Keep the frame, the solid piece, all azure divider lines, and the palette row unchanged.

    Key insight:
    - The visible task is a key-versus-lock test: the meaningful comparison is between the solid piece and the hollow frame's enclosed cavity, and the binary outcome is encoded by the first two palette positions rather than by the piece colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    palette_sep = _palette_separator_row(g)
    palette = _palette_colors(g, palette_sep)
    assert len(palette) >= 2, "palette must contain at least two colors"

    out = g.copy()
    for r0, r1, c0, c1 in _region_spans(g, palette_sep):
        fill_color = palette[0] if _region_matches(g[r0:r1, c0:c1]) else palette[1]
        patch = out[r0:r1, c0:c1]
        patch[patch == BG] = fill_color
    return out


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _random_polyomino(
    rng: np.random.Generator,
    min_area: int = 2,
    max_area: int = 6,
    exact_area: int | None = None,
) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if exact_area is None:
            h = _randint(rng, 2, 4)
            w = _randint(rng, 2, 4)
            area = _randint(rng, min_area, min(max_area, h * w))
        else:
            area = exact_area
            boxes = [(h, w) for h in range(2, 5) for w in range(2, 5) if h * w >= area]
            if not boxes:
                continue
            h, w = boxes[_randint(rng, 0, len(boxes) - 1)]

        mask = np.zeros((h, w), dtype=bool)
        start = (_randint(rng, 0, h - 1), _randint(rng, 0, w - 1))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[_randint(rng, 0, len(frontier) - 1)]
            mask[nr, nc] = True
        if int(mask.sum()) == area:
            return _tight_mask(mask)
    raise RuntimeError("failed to generate a connected polyomino")


def _frame_from_cavity(cavity: np.ndarray) -> np.ndarray:
    pad = np.pad(cavity, 1)
    dilated = pad.copy()
    for r, c in np.argwhere(pad):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                dilated[r + dr, c + dc] = True
    return dilated & ~pad


def _valid_frame(frame: np.ndarray, cavity: np.ndarray) -> bool:
    return len(_components(frame)) == 1 and len(_hole_masks(frame)) == 1 and np.array_equal(_hole_masks(frame)[0], cavity)


def _nonmatching_piece(rng: np.random.Generator, cavity: np.ndarray) -> np.ndarray | None:
    cavity_area = int(cavity.sum())
    different_areas = [area for area in range(2, 7) if area != cavity_area]

    if different_areas and rng.random() < 0.6:
        piece = _random_polyomino(rng, min_area=2, max_area=6, exact_area=different_areas[_randint(rng, 0, len(different_areas) - 1)])
        return np.rot90(piece, _randint(rng, 0, 3))

    for _ in range(SHAPE_RETRIES):
        piece = _random_polyomino(rng, min_area=2, max_area=6, exact_area=cavity_area)
        piece = np.rot90(piece, _randint(rng, 0, 3))
        if not _matches_under_rotation(piece, cavity):
            return piece
    return None


def _place_pair(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    frame: np.ndarray,
    piece: np.ndarray,
    frame_color: int,
    piece_color: int,
) -> np.ndarray | None:
    fh, fw = frame.shape
    ph, pw = piece.shape
    options: list[str] = []
    for side in ("left", "right", "top", "bottom"):
        if side in ("left", "right"):
            if max(fh, ph) + 2 <= region_h and fw + pw + 3 <= region_w:
                options.append(side)
        else:
            if fh + ph + 3 <= region_h and max(fw, pw) + 2 <= region_w:
                options.append(side)
    if not options:
        return None

    side = options[_randint(rng, 0, len(options) - 1)]
    gap = _randint(rng, 1, 2)

    if side in ("left", "right"):
        total_h = max(fh, ph)
        total_w = fw + pw + gap
        if total_h + 2 > region_h or total_w + 2 > region_w:
            return None
        frame_r = _randint(rng, 0, total_h - fh)
        piece_r = _randint(rng, 0, total_h - ph)
        if side == "left":
            piece_c = 0
            frame_c = pw + gap
        else:
            frame_c = 0
            piece_c = fw + gap
        frame_pos = (frame_r, frame_c)
        piece_pos = (piece_r, piece_c)
    else:
        total_h = fh + ph + gap
        total_w = max(fw, pw)
        if total_h + 2 > region_h or total_w + 2 > region_w:
            return None
        frame_c = _randint(rng, 0, total_w - fw)
        piece_c = _randint(rng, 0, total_w - pw)
        if side == "top":
            piece_r = 0
            frame_r = ph + gap
        else:
            frame_r = 0
            piece_r = fh + gap
        frame_pos = (frame_r, frame_c)
        piece_pos = (piece_r, piece_c)

    origin_r = _randint(rng, 1, region_h - total_h - 1)
    origin_c = _randint(rng, 1, region_w - total_w - 1)
    region = np.full((region_h, region_w), BG, dtype=int)

    fr, fc = frame_pos
    pr, pc = piece_pos
    frame_view = region[origin_r + fr : origin_r + fr + fh, origin_c + fc : origin_c + fc + fw]
    piece_view = region[origin_r + pr : origin_r + pr + ph, origin_c + pc : origin_c + pc + pw]
    frame_view[frame] = frame_color
    piece_view[piece] = piece_color

    if len(_components(region != BG)) != 2:
        return None
    return region


def _make_region(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    match: bool,
    fill_color: int,
    shape_colors: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_RETRIES):
        cavity = _random_polyomino(rng, min_area=2, max_area=6)
        frame = _frame_from_cavity(cavity)
        if not _valid_frame(frame, cavity):
            continue

        piece = np.rot90(cavity, _randint(rng, 0, 3)) if match else _nonmatching_piece(rng, cavity)
        if piece is None:
            continue
        frame_color, piece_color = rng.choice(shape_colors, size=2, replace=False).tolist()
        region = _place_pair(rng, region_h, region_w, frame, piece, int(frame_color), int(piece_color))
        if region is None:
            continue
        out = region.copy()
        out[out == BG] = fill_color
        return region, out
    return None


def _layout_dimensions(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    row_count, col_count = [(1, 2), (1, 3), (2, 2), (2, 3)][_randint(rng, 0, 3)]
    for _ in range(SHAPE_RETRIES):
        max_h = 12 if row_count == 1 else 10
        max_w = 11 if col_count == 2 else 9
        heights = [_randint(rng, 7, max_h) for _ in range(row_count)]
        widths = [_randint(rng, 8, max_w) for _ in range(col_count)]
        total_h = sum(heights) + (row_count - 1) + 2
        total_w = sum(widths) + (col_count - 1)
        if total_h <= 30 and total_w <= 30:
            return heights, widths
    raise RuntimeError("failed to sample valid region dimensions")


def _palette_row(rng: np.random.Generator, width: int) -> tuple[np.ndarray, list[int]]:
    length = _randint(rng, 3, min(5, width))
    colors = rng.choice(COLOR_POOL, size=length, replace=False).tolist()
    cols = sorted(rng.choice(width, size=length, replace=False).tolist())
    row = np.full(width, BG, dtype=int)
    for col, color in zip(cols, colors):
        row[int(col)] = int(color)
    return row, [int(color) for color in colors]


def _gen_attempt() -> np.ndarray | None:
    heights, widths = _layout_dimensions(RNG)
    row_count = len(heights)
    col_count = len(widths)
    region_count = row_count * col_count

    match_count = _randint(RNG, 1, region_count - 1)
    flags = np.array([True] * match_count + [False] * (region_count - match_count), dtype=bool)
    RNG.shuffle(flags)

    h = sum(heights) + (row_count - 1) + 2
    w = sum(widths) + (col_count - 1)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    row_starts: list[int] = []
    cursor = 0
    for idx, height in enumerate(heights):
        row_starts.append(cursor)
        cursor += height
        if idx < row_count - 1:
            grid[cursor, :] = DIVIDER
            expected[cursor, :] = DIVIDER
            cursor += 1
    palette_sep = cursor
    grid[palette_sep, :] = DIVIDER
    expected[palette_sep, :] = DIVIDER

    col_starts: list[int] = []
    cursor = 0
    for idx, width in enumerate(widths):
        col_starts.append(cursor)
        cursor += width
        if idx < col_count - 1:
            grid[: palette_sep + 1, cursor] = DIVIDER
            expected[: palette_sep + 1, cursor] = DIVIDER
            cursor += 1

    palette_row, palette_colors = _palette_row(RNG, w)
    grid[palette_sep + 1, :] = palette_row
    expected[palette_sep + 1, :] = palette_row

    shape_colors = [color for color in COLOR_POOL if color not in palette_colors]
    if len(shape_colors) < 2:
        return None

    index = 0
    for rr, region_h in enumerate(heights):
        for cc, region_w in enumerate(widths):
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            fill = palette_colors[0] if bool(flags[index]) else palette_colors[1]
            region_pair = _make_region(RNG, region_h, region_w, bool(flags[index]), fill, shape_colors)
            if region_pair is None:
                return None
            region_in, region_out = region_pair
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = region_in
            expected[r0 : r0 + region_h, c0 : c0 + region_w] = region_out
            index += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock palette grid")
