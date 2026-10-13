import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _mode(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    assert flat.size > 0
    return int(np.bincount(flat).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_structure_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    best_color = colors[0]
    best_key = (-1, -1, -1, -1)
    for color in colors:
        row_hits = int(np.sum(np.all(grid == color, axis=1)))
        col_hits = int(np.sum(np.all(grid == color, axis=0)))
        pixel_hits = int(np.count_nonzero(grid == color))
        key = (row_hits + col_hits, row_hits, col_hits, pixel_hits)
        if key > best_key:
            best_key = key
            best_color = color

    if best_key[0] == 0:
        best_color = max(colors, key=lambda color: int(np.count_nonzero(grid == color)))
    return int(best_color)


def _parse_bottom_palette(grid: np.ndarray, sep_row: int, grey: int, bg: int) -> dict[int, np.ndarray]:
    band = grid[sep_row + 1 :, :]
    cell_total = int(band.shape[0])
    assert cell_total in (3, 4)

    active_cols = np.flatnonzero(np.any(band != bg, axis=0))
    assert active_cols.size > 0
    left = int(active_cols[0])
    right = int(active_cols[-1])

    spans: list[tuple[int, int]] = []
    start = left
    for c in range(left, right + 1):
        if np.all(band[:, c] == grey):
            assert start <= c - 1
            spans.append((start, c - 1))
            start = c + 1
    assert start <= right
    spans.append((start, right))

    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in spans:
        sub = band[:, c0 : c1 + 1]
        assert sub.shape == (cell_total, cell_total)
        marker = int(sub[0, 0])
        assert marker not in (bg, grey)
        assert marker not in mapping
        assert np.all(sub[0, 1:] == bg)
        assert np.all(sub[1:, 0] == bg)
        pattern = sub[1:, 1:].copy()
        colors = set(map(int, np.unique(pattern)))
        assert pattern.shape == (cell_total - 1, cell_total - 1)
        assert bg not in colors and grey not in colors and len(colors) == 2
        mapping[marker] = pattern
    assert mapping
    return mapping


def _parse_right_palette(grid: np.ndarray, sep_col: int, grey: int, bg: int) -> dict[int, np.ndarray]:
    band = grid[:, sep_col + 1 :]
    cell_total = int(band.shape[1])
    assert cell_total in (3, 4)

    active_rows = np.flatnonzero(np.any(band != bg, axis=1))
    assert active_rows.size > 0
    top = int(active_rows[0])
    bottom = int(active_rows[-1])

    spans: list[tuple[int, int]] = []
    start = top
    for r in range(top, bottom + 1):
        if np.all(band[r, :] == grey):
            assert start <= r - 1
            spans.append((start, r - 1))
            start = r + 1
    assert start <= bottom
    spans.append((start, bottom))

    mapping: dict[int, np.ndarray] = {}
    for r0, r1 in spans:
        sub = band[r0 : r1 + 1, :]
        assert sub.shape == (cell_total, cell_total)
        marker = int(sub[0, 0])
        assert marker not in (bg, grey)
        assert marker not in mapping
        assert np.all(sub[0, 1:] == bg)
        assert np.all(sub[1:, 0] == bg)
        pattern = sub[1:, 1:].copy()
        colors = set(map(int, np.unique(pattern)))
        assert pattern.shape == (cell_total - 1, cell_total - 1)
        assert bg not in colors and grey not in colors and len(colors) == 2
        mapping[marker] = pattern
    assert mapping
    return mapping


def _palette_info(grid: np.ndarray, grey: int, bg: int):
    h, w = grid.shape
    candidates = []

    for sep_row in np.flatnonzero(np.all(grid == grey, axis=1)):
        sep_row = int(sep_row)
        if h - sep_row - 1 not in (3, 4):
            continue
        try:
            mapping = _parse_bottom_palette(grid, sep_row, grey, bg)
        except AssertionError:
            continue
        palette_mask = np.zeros_like(grid, dtype=bool)
        palette_mask[sep_row:, :] = True
        candidates.append(("bottom", sep_row, mapping, palette_mask))

    for sep_col in np.flatnonzero(np.all(grid == grey, axis=0)):
        sep_col = int(sep_col)
        if w - sep_col - 1 not in (3, 4):
            continue
        try:
            mapping = _parse_right_palette(grid, sep_col, grey, bg)
        except AssertionError:
            continue
        palette_mask = np.zeros_like(grid, dtype=bool)
        palette_mask[:, sep_col:] = True
        candidates.append(("right", sep_col, mapping, palette_mask))

    assert candidates, "failed to locate palette strip"
    candidates.sort(key=lambda item: (len(item[2]), item[1]))
    return candidates[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region by tiling the palette pattern keyed by its seed color.

    Input structure:
    - Black background with one palette strip at the bottom or right, separated from the main area by a full grey line.
    - The main area contains several 1-cell-thick grey closed borders around rectangular or L-shaped interiors.
    - Each enclosed interior contains exactly one non-black, non-grey seed pixel on an otherwise black region.
    - Every palette cell is a square block whose top-left pixel is a seed-color marker and whose lower-right 2x2 or 3x3 square is the fill pattern.

    Rule:
    - Detect the palette strip, split it into cells, and map each marker color to its associated 2x2 or 3x3 pattern.
    - For every grey-enclosed interior, read its unique seed color and select the matching palette pattern.
    - Tile that pattern over the whole interior starting from the interior's top-left cell, replacing the seed and black cells while keeping the grey border unchanged.
    - Clear the entire palette strip, including its separator line, to black.

    Key insight:
    - The seed pixel is only a lookup key; the real content comes from decoding the palette cell and repeating that decoded pattern across the whole enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode(g)
    grey = _find_structure_color(g, bg)
    orientation, split_idx, mapping, palette_mask = _palette_info(g, grey, bg)

    out = g.copy()
    out[palette_mask] = bg

    if orientation == "bottom":
        play = g[:split_idx, :]
        row_offset = 0
        col_offset = 0
    else:
        play = g[:, :split_idx]
        row_offset = 0
        col_offset = 0

    for comp in _components(play != grey):
        if any(r == 0 or c == 0 or r == play.shape[0] - 1 or c == play.shape[1] - 1 for r, c in comp):
            continue

        colors = {int(play[r, c]) for r, c in comp if int(play[r, c]) != bg}
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        seed_color = colors.pop()
        assert seed_color in mapping, "missing palette entry for enclosed seed"
        pattern = mapping[seed_color]
        ph, pw = pattern.shape
        top = min(r for r, _c in comp)
        left = min(c for _r, c in comp)

        for r, c in comp:
            out[row_offset + r, col_offset + c] = int(pattern[(r - top) % ph, (c - left) % pw])

    return out


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    assert low <= high
    return int(rng.integers(low, high + 1))


def _pattern_mask(size: int, kind: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    if kind == 0:
        return (rr + cc) % 2 == 0
    if kind == 1:
        return rr % 2 == 0
    return cc % 2 == 0


def _make_pattern(size: int, seed_color: int, rng: np.random.Generator) -> np.ndarray:
    colors = [color for color in COLOR_POOL if color != seed_color]
    c0, c1 = rng.choice(colors, size=2, replace=False).tolist()
    mask = _pattern_mask(size, _randint(rng, 0, 2))
    pattern = np.full((size, size), int(c1), dtype=int)
    pattern[mask] = int(c0)
    return pattern


def _make_rectangle(min_dim: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, min_dim, 8)
    w = _randint(rng, min_dim, 8)
    return np.ones((h, w), dtype=bool)


def _make_l_shape(min_dim: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, min_dim + 1, 8)
    w = _randint(rng, min_dim + 1, 8)
    arm_h = _randint(rng, 2, h - 1)
    arm_w = _randint(rng, 2, w - 1)

    shape = np.zeros((h, w), dtype=bool)
    shape[:arm_h, :] = True
    shape[:, :arm_w] = True

    orientation = _randint(rng, 0, 2)
    if orientation == 1:
        shape = shape[::-1, :]
    elif orientation == 2:
        shape = shape[:, ::-1]
    return shape


def _make_region_shape(pattern_size: int, rng: np.random.Generator) -> np.ndarray:
    min_dim = pattern_size + 1
    if _randint(rng, 0, 1) == 0:
        return _make_rectangle(min_dim, rng)
    return _make_l_shape(min_dim, rng)


def _frame_masks(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(interior.astype(bool), 1)
    border = np.zeros_like(padded, dtype=bool)
    for r, c in np.argwhere(padded):
        r = int(r)
        c = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and not padded[nr, nc]:
                border[nr, nc] = True
    return padded, border


def _place_regions(main_area: np.ndarray, seed_colors: list[int], pattern_size: int, rng: np.random.Generator) -> bool:
    h, w = main_area.shape
    occupied = np.zeros((h, w), dtype=bool)

    jobs = []
    for seed_color in seed_colors:
        interior = _make_region_shape(pattern_size, rng)
        interior_mask, border_mask = _frame_masks(interior)
        jobs.append((int(seed_color), interior_mask, border_mask))

    jobs.sort(key=lambda item: int(np.count_nonzero(item[1] | item[2])), reverse=True)

    for seed_color, interior_mask, border_mask in jobs:
        ch, cw = interior_mask.shape
        positions = [(r, c) for r in range(h - ch + 1) for c in range(w - cw + 1)]
        rng.shuffle(positions)
        placed = False

        for r0, c0 in positions:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ch + 1)
            cc1 = min(w, c0 + cw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue

            local = main_area[r0 : r0 + ch, c0 : c0 + cw]
            assert np.all(local[border_mask] == BG)
            local[border_mask] = GREY

            seed_choices = np.argwhere(interior_mask)
            sr, sc = seed_choices[_randint(rng, 0, len(seed_choices) - 1)]
            local[int(sr), int(sc)] = seed_color

            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _build_palette_entries(count: int, pattern_size: int, rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    seed_colors = rng.choice(COLOR_POOL, size=count, replace=False).tolist()
    entries = [(int(seed), _make_pattern(pattern_size, int(seed), rng)) for seed in seed_colors]
    if count > 1 and len({pattern.tobytes() for _seed, pattern in entries}) == 1:
        seed = int(entries[-1][0])
        entries[-1] = (seed, _make_pattern(pattern_size, seed, rng))
    return entries


def _draw_bottom_palette(
    grid: np.ndarray,
    sep_row: int,
    entries: list[tuple[int, np.ndarray]],
    cell_total: int,
    rng: np.random.Generator,
) -> None:
    palette_span = len(entries) * cell_total + max(0, len(entries) - 1)
    left = _randint(rng, 0, grid.shape[1] - palette_span)
    grid[sep_row, :] = GREY

    col = left
    for index, (seed_color, pattern) in enumerate(entries):
        sub = grid[sep_row + 1 : sep_row + 1 + cell_total, col : col + cell_total]
        sub[0, 0] = int(seed_color)
        sub[1:, 1:] = pattern
        col += cell_total
        if index != len(entries) - 1:
            grid[sep_row + 1 :, col] = GREY
            col += 1


def _draw_right_palette(
    grid: np.ndarray,
    sep_col: int,
    entries: list[tuple[int, np.ndarray]],
    cell_total: int,
    rng: np.random.Generator,
) -> None:
    palette_span = len(entries) * cell_total + max(0, len(entries) - 1)
    top = _randint(rng, 0, grid.shape[0] - palette_span)
    grid[:, sep_col] = GREY

    row = top
    for index, (seed_color, pattern) in enumerate(entries):
        sub = grid[row : row + cell_total, sep_col + 1 : sep_col + 1 + cell_total]
        sub[0, 0] = int(seed_color)
        sub[1:, 1:] = pattern
        row += cell_total
        if index != len(entries) - 1:
            grid[row, sep_col + 1 :] = GREY
            row += 1


def _valid_separator_layout(grid: np.ndarray, orientation: str, split_idx: int) -> bool:
    full_rows = set(map(int, np.flatnonzero(np.all(grid == GREY, axis=1))))
    full_cols = set(map(int, np.flatnonzero(np.all(grid == GREY, axis=0))))
    if orientation == "bottom":
        return full_rows == {split_idx} and not full_cols
    return full_cols == {split_idx} and not full_rows


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "bottom" if _randint(rng, 0, 1) == 0 else "right"
    pattern_size = 2 if _randint(rng, 0, 1) == 0 else 3
    cell_total = pattern_size + 1
    count = _randint(rng, 2, 4)
    entries = _build_palette_entries(count, pattern_size, rng)
    palette_span = count * cell_total + (count - 1)

    stack_low = max(14, palette_span, 5 * count + 6)
    cross_low = max(12, 4 * count + 2)

    if orientation == "bottom":
        max_main_h = 30 - 1 - cell_total
        if cross_low > max_main_h:
            return None
        main_h = _randint(rng, cross_low, max_main_h)
        total_w = _randint(rng, stack_low, 30)
        grid = np.full((main_h + 1 + cell_total, total_w), BG, dtype=int)
        if not _place_regions(grid[:main_h, :], [seed for seed, _pattern in entries], pattern_size, rng):
            return None
        _draw_bottom_palette(grid, main_h, entries, cell_total, rng)
        split_idx = main_h
    else:
        max_main_w = 30 - 1 - cell_total
        if cross_low > max_main_w:
            return None
        total_h = _randint(rng, stack_low, 30)
        main_w = _randint(rng, cross_low, max_main_w)
        grid = np.full((total_h, main_w + 1 + cell_total), BG, dtype=int)
        if not _place_regions(grid[:, :main_w], [seed for seed, _pattern in entries], pattern_size, rng):
            return None
        _draw_right_palette(grid, main_w, entries, cell_total, rng)
        split_idx = main_w

    if not _valid_separator_layout(grid, orientation, split_idx):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-fill puzzle")
