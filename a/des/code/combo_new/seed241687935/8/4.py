import numpy as np

from functools import lru_cache


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# Generator configuration.
BOARD_MIN = 8
BOARD_MAX = 11
RECT_MIN = 3
RECT_MAX = 5
GROUP_COUNT_MIN = 2
GROUP_COUNT_MAX = 4
PIECE_COUNT_OPTIONS = (2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_extent(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _canonical_shapes(shapes) -> tuple[tuple[tuple[int, int], ...], ...]:
    normed = [_normalize_shape(shape) for shape in shapes]
    return tuple(sorted(normed, key=lambda shp: (-len(shp), shp)))


@lru_cache(maxsize=None)
def _can_tile_holey_rectangle(
    shapes_key: tuple[tuple[tuple[int, int], ...], ...],
    height: int,
    width: int,
    hole_r: int,
    hole_c: int,
) -> bool:
    hole = (int(hole_r), int(hole_c))
    cells = []
    bit_of = {}
    bit = 1
    for r in range(height):
        for c in range(width):
            if (r, c) == hole:
                continue
            cells.append((r, c))
            bit_of[(r, c)] = bit
            bit <<= 1
    full_mask = bit - 1

    placements = []
    for shape in shapes_key:
        sh, sw = _shape_extent(shape)
        opts = set()
        for dr in range(height - sh + 1):
            for dc in range(width - sw + 1):
                placed = [(r + dr, c + dc) for r, c in shape]
                if hole in placed:
                    continue
                mask = 0
                for cell in placed:
                    mask |= bit_of[cell]
                opts.add(mask)
        if not opts:
            return False
        placements.append(tuple(sorted(opts)))

    placements = tuple(placements)

    @lru_cache(maxsize=None)
    def dfs(remaining: tuple[int, ...], used_mask: int) -> bool:
        if used_mask == full_mask:
            return not remaining
        if not remaining:
            return False

        union_mask = 0
        viable_by_piece: dict[int, tuple[int, ...]] = {}
        for idx in remaining:
            viable = tuple(mask for mask in placements[idx] if (mask & used_mask) == 0)
            if not viable:
                return False
            viable_by_piece[idx] = viable
            piece_union = 0
            for mask in viable:
                piece_union |= mask
            union_mask |= piece_union
        if (union_mask | used_mask) != full_mask:
            return False

        needed = full_mask & ~used_mask
        pivot = needed & -needed

        candidate_groups = []
        for idx in remaining:
            covers = tuple(mask for mask in viable_by_piece[idx] if mask & pivot)
            if covers:
                candidate_groups.append((len(covers), idx, covers))
        if not candidate_groups:
            return False
        candidate_groups.sort()

        for _count, idx, covers in candidate_groups:
            next_remaining = tuple(j for j in remaining if j != idx)
            for mask in covers:
                if dfs(next_remaining, used_mask | mask):
                    return True
        return False

    return dfs(tuple(range(len(placements))), 0)


def _all_group_solutions(
    shapes,
    board_shape: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    shapes_key = _canonical_shapes(shapes)
    total_area = sum(len(shape) for shape in shapes_key) + 1
    max_board_h = None if board_shape is None else int(board_shape[0])
    max_board_w = None if board_shape is None else int(board_shape[1])
    solutions = []

    for h in range(RECT_MIN, total_area + 1):
        if total_area % h != 0:
            continue
        w = total_area // h
        if w < RECT_MIN:
            continue
        if max_board_h is not None and h > max_board_h:
            continue
        if max_board_w is not None and w > max_board_w:
            continue
        if any(_shape_extent(shape)[0] > h or _shape_extent(shape)[1] > w for shape in shapes_key):
            continue

        hole_rows = range(1, h - 1)
        hole_cols = range(1, w - 1)
        if max_board_h is not None:
            hole_rows = [r for r in hole_rows if r + h <= max_board_h]
        if max_board_w is not None:
            hole_cols = [c for c in hole_cols if c + w <= max_board_w]

        for hole_r in hole_rows:
            for hole_c in hole_cols:
                if _can_tile_holey_rectangle(shapes_key, int(h), int(w), int(hole_r), int(hole_c)):
                    solutions.append((int(h), int(w), int(hole_r), int(hole_c)))
    return solutions


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dst[top : top + h, left : left + w] = src


def _stack_outputs(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros((1, 1), dtype=int)
    height = sum(int(part.shape[0]) for part in parts) + len(parts) - 1
    width = max(int(part.shape[1]) for part in parts)
    out = np.zeros((height, width), dtype=int)
    row = 0
    for idx, part in enumerate(parts):
        _paste(out, part, row, 0)
        row += int(part.shape[0])
        if idx != len(parts) - 1:
            row += 1
    return out


def _find_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert grey_cols, "expected at least one full grey separator column"
    for c in grey_cols:
        if np.any(grid[:, :c] != BLACK) and np.any(grid[:, c + 1 :] != BLACK):
            return int(c)
    return int(grey_cols[0])


def _extract_board(grid: np.ndarray, sep_col: int) -> np.ndarray:
    left = grid[:, :sep_col]
    coords = np.argwhere(left != BLACK)
    assert coords.size, "left side must contain a colored pattern board"
    r0, c0, r1, c1 = _bbox(coords)
    board = left[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(board != BLACK) and np.all(board != GREY)
    return board.astype(int)


def _extract_groups(grid: np.ndarray, sep_col: int):
    right = grid[:, sep_col + 1 :]
    colors = sorted(set(map(int, right.ravel())) - {BLACK, GREY})
    groups = []
    for color in colors:
        comps = _components(right == color)
        assert comps, f"missing pieces for color {color}"
        first = min((int(r), int(c)) for comp in comps for r, c in comp)
        shapes = [_normalize_shape(comp) for comp in comps]
        groups.append((first, int(color), shapes))
    groups.sort(key=lambda item: item[0])
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color fragments into holey rectangles and use each hole as a crop coordinate on the shared pattern board.

    Input structure:
    - Black is the background and one full grey column splits the grid into a left board region and a right fragment region.
    - The left region contains one solid rectangular pattern board made only of non-black, non-grey colors.
    - The right region contains several monochrome 4-connected fragment objects; fragments with the same color belong to one group.
    - Within each color group, the fragments can be translated without rotation or reflection so their union becomes one rectangle missing exactly one internal 1x1 cell.

    Rule:
    - For each right-side color group, find the unique rectangle dimensions and translations that pack its fragments into a rectangle with one internal missing cell, then record that hole position (r, c) inside the reconstructed rectangle.
    - Use (r, c) as the top-left coordinate of a crop on the left pattern board, with the crop size equal to the reconstructed rectangle size.
    - Output those crops in the order their fragment colors first appear when scanning the right region from top to bottom, using one black separator row between crops and left-aligning them on the common output canvas.

    Key insight:
    - The missing cell inside each reconstructed rectangle is not a flaw to repair; it is a coordinate pointer telling which sub-rectangle to cut from the shared pattern board.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_col = _find_separator_col(g)
    board = _extract_board(g, sep_col)
    groups = _extract_groups(g, sep_col)
    assert groups, "right side must contain at least one fragment group"

    pieces = []
    board_shape = tuple(map(int, board.shape))
    for _first, _color, shapes in groups:
        solutions = _all_group_solutions(shapes, board_shape=board_shape)
        assert len(solutions) == 1, "fragment group must assemble into a unique holey rectangle"
        h, w, hole_r, hole_c = solutions[0]
        pieces.append(board[hole_r : hole_r + h, hole_c : hole_c + w].copy())

    return _stack_outputs(pieces)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _partition_holey_rectangle(
    height: int,
    width: int,
    hole: tuple[int, int],
    piece_count: int,
):
    occupied = {(r, c) for r in range(height) for c in range(width) if (r, c) != hole}
    cell_list = sorted(occupied)
    min_piece_size = 2 if len(occupied) >= 2 * piece_count else 1

    for _ in range(120):
        seed_indices = RNG.choice(len(cell_list), size=piece_count, replace=False)
        pieces = [set([cell_list[int(idx)]]) for idx in seed_indices]
        unassigned = set(occupied)
        for idx in seed_indices:
            unassigned.remove(cell_list[int(idx)])

        ok = True
        while any(len(piece) < min_piece_size for piece in pieces):
            order = list(range(piece_count))
            RNG.shuffle(order)
            progressed = False
            for idx in order:
                if len(pieces[idx]) >= min_piece_size:
                    continue
                frontier = []
                for cell in unassigned:
                    if any((nr, nc) in pieces[idx] for nr, nc in _neighbors4(*cell)):
                        frontier.append(cell)
                if not frontier:
                    ok = False
                    break
                cell = frontier[int(RNG.integers(0, len(frontier)))]
                pieces[idx].add(cell)
                unassigned.remove(cell)
                progressed = True
            if not ok or not progressed:
                break
        if not ok:
            continue

        while unassigned:
            frontier = []
            for cell in unassigned:
                adjacent = [idx for idx, piece in enumerate(pieces) if any((nr, nc) in piece for nr, nc in _neighbors4(*cell))]
                if adjacent:
                    frontier.append((cell, adjacent))
            if not frontier:
                ok = False
                break
            cell, adjacent = frontier[int(RNG.integers(0, len(frontier)))]
            pick = int(adjacent[int(RNG.integers(0, len(adjacent)))])
            pieces[pick].add(cell)
            unassigned.remove(cell)
        if not ok:
            continue
        if not all(len(piece) >= min_piece_size and _is_connected(piece) for piece in pieces):
            continue
        return [_normalize_shape(piece) for piece in pieces]
    return None


def _build_board(height: int, width: int) -> np.ndarray:
    palette_size = int(RNG.integers(4, min(7, len(COLOR_POOL)) + 1))
    palette = RNG.choice(np.array(COLOR_POOL, dtype=int), size=palette_size, replace=False)
    rr = np.arange(height)[:, None]
    cc = np.arange(width)[None, :]
    a = int(RNG.integers(1, 4))
    b = int(RNG.integers(1, 4))
    c = int(RNG.integers(1, 4))
    d = int(RNG.integers(2, 5))
    board = palette[(a * rr + b * cc + ((rr + c * cc) // d)) % palette_size]

    for _ in range(int(RNG.integers(1, 4))):
        r0 = int(RNG.integers(0, height))
        c0 = int(RNG.integers(0, width))
        rh = int(RNG.integers(1, min(4, height - r0) + 1))
        cw = int(RNG.integers(1, min(4, width - c0) + 1))
        board[r0 : r0 + rh, c0 : c0 + cw] = int(palette[int(RNG.integers(0, palette_size))])

    return board.astype(int)


def _crop_signature(arr: np.ndarray) -> tuple:
    return (int(arr.shape[0]), int(arr.shape[1]), tuple(map(int, arr.ravel())))


def _gen_attempt() -> np.ndarray | None:
    group_count = int(RNG.integers(GROUP_COUNT_MIN, GROUP_COUNT_MAX + 1))
    colors = [int(v) for v in RNG.choice(np.array(COLOR_POOL, dtype=int), size=group_count, replace=False)]

    used_holes = set()
    groups = []

    for color in colors:
        built = None
        for _ in range(120):
            height = int(RNG.integers(RECT_MIN, RECT_MAX + 1))
            width = int(RNG.integers(RECT_MIN, RECT_MAX + 1))
            hole_r = int(RNG.integers(1, height - 1))
            hole_c = int(RNG.integers(1, width - 1))
            if (hole_r, hole_c) in used_holes:
                continue
            piece_count = int(PIECE_COUNT_OPTIONS[int(RNG.integers(0, len(PIECE_COUNT_OPTIONS)))])
            piece_count = min(piece_count, (height * width - 1) // 2)
            piece_count = max(piece_count, 2)
            shapes = _partition_holey_rectangle(height, width, (hole_r, hole_c), piece_count)
            if shapes is None:
                continue
            solutions = _all_group_solutions(shapes)
            if solutions != [(height, width, hole_r, hole_c)]:
                continue
            built = {
                "color": color,
                "height": height,
                "width": width,
                "hole": (hole_r, hole_c),
                "shapes": shapes,
            }
            break
        if built is None:
            return None
        used_holes.add(built["hole"])
        groups.append(built)

    min_board_h = max(BOARD_MIN, max(int(group["hole"][0]) + int(group["height"]) for group in groups))
    min_board_w = max(BOARD_MIN, max(int(group["hole"][1]) + int(group["width"]) for group in groups))
    if min_board_h > BOARD_MAX or min_board_w > BOARD_MAX:
        return None

    board_h = int(RNG.integers(min_board_h, BOARD_MAX + 1))
    board_w = int(RNG.integers(min_board_w, BOARD_MAX + 1))
    board = _build_board(board_h, board_w)

    crops = []
    for group in groups:
        hole_r, hole_c = group["hole"]
        height = int(group["height"])
        width = int(group["width"])
        crops.append(board[hole_r : hole_r + height, hole_c : hole_c + width].copy())
    if len({_crop_signature(crop) for crop in crops}) != len(crops):
        return None
    expected = _stack_outputs(crops)

    layout_groups = []
    cursor_y = 0
    max_span_w = 0
    for group in groups:
        shapes = list(group["shapes"])
        RNG.shuffle(shapes)
        piece_sizes = [_shape_extent(shape) for shape in shapes]
        gaps = [int(RNG.integers(1, 3)) for _ in range(max(0, len(shapes) - 1))]

        x_cursor = 0
        piece_layouts = []
        anchor_index = int(RNG.integers(0, len(shapes)))
        band_h = max(h for h, _ in piece_sizes) + int(RNG.integers(1, 3))
        for idx, shape in enumerate(shapes):
            ph, pw = piece_sizes[idx]
            y_off = 0 if idx == anchor_index else int(RNG.integers(0, band_h - ph + 1))
            piece_layouts.append((shape, y_off, x_cursor))
            x_cursor += pw
            if idx < len(gaps):
                x_cursor += gaps[idx]
        span_w = x_cursor
        max_span_w = max(max_span_w, span_w)
        layout_groups.append(
            {
                "color": int(group["color"]),
                "band_top": int(cursor_y),
                "band_h": int(band_h),
                "span_w": int(span_w),
                "pieces": piece_layouts,
            }
        )
        cursor_y += band_h + int(RNG.integers(1, 3))

    right_h = int(layout_groups[-1]["band_top"] + layout_groups[-1]["band_h"])
    right_w = int(max_span_w + RNG.integers(0, 3))
    board_left = int(RNG.integers(0, 2))
    grid_w = board_left + board_w + 1 + right_w
    if grid_w > 30:
        return None

    grid_h = max(board_h, right_h) + int(RNG.integers(0, 3))
    if grid_h > 30:
        return None

    board_top = int(RNG.integers(0, grid_h - board_h + 1))
    right_top = int(RNG.integers(0, grid_h - right_h + 1))
    sep_col = board_left + board_w

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[:, sep_col] = GREY
    _paste(grid, board, board_top, board_left)

    for group in layout_groups:
        color = int(group["color"])
        x_shift = int(RNG.integers(0, right_w - int(group["span_w"]) + 1))
        for shape, y_off, x_off in group["pieces"]:
            top = right_top + int(group["band_top"]) + int(y_off)
            left = sep_col + 1 + x_shift + int(x_off)
            for r, c in shape:
                grid[top + r, left + c] = color

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
