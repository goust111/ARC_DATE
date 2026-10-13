import random
from collections import Counter, deque

import numpy as np


BG = 0
SEP = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _segments_from_full_lines(full_mask: np.ndarray) -> list[slice]:
    segments: list[slice] = []
    start: int | None = None
    for idx, is_full in enumerate(full_mask.tolist() + [True]):
        if not is_full and start is None:
            start = idx
        elif is_full and start is not None:
            segments.append(slice(start, idx))
            start = None
    return segments


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _top_level_regions(grid: np.ndarray) -> list[np.ndarray]:
    row_segments = _segments_from_full_lines(np.all(grid == SEP, axis=1))
    col_segments = _segments_from_full_lines(np.all(grid == SEP, axis=0))
    assert row_segments and col_segments, "missing separator partition"
    return [grid[rs, cs].copy() for rs in row_segments for cs in col_segments]


def _parse_order_legend(region: np.ndarray) -> np.ndarray | None:
    row_full = np.all(region == SEP, axis=1)
    col_full = np.all(region == SEP, axis=0)
    if not row_full.any() and not col_full.any():
        return None

    row_segments = _segments_from_full_lines(row_full)
    col_segments = _segments_from_full_lines(col_full)
    if len(row_segments) * len(col_segments) <= 1:
        return None

    order_rows: list[list[int]] = []
    flat: list[int] = []
    for rs in row_segments:
        row_colors: list[int] = []
        for cs in col_segments:
            cell = region[rs, cs]
            if cell.size == 0 or np.any(cell == SEP) or np.any(cell == BG):
                return None
            colors = np.unique(cell)
            if len(colors) != 1:
                return None
            color = int(colors[0])
            row_colors.append(color)
            flat.append(color)
        order_rows.append(row_colors)

    if len(set(flat)) != len(flat):
        return None
    return np.array(order_rows, dtype=int)


def _rotation_from_marker(points: list[tuple[int, int]]) -> int | None:
    pset = set(points)
    for r, c in points:
        dirs: set[str] = set()
        if (r - 1, c) in pset:
            dirs.add("U")
        if (r + 1, c) in pset:
            dirs.add("D")
        if (r, c - 1) in pset:
            dirs.add("L")
        if (r, c + 1) in pset:
            dirs.add("R")
        if len(dirs) != 2:
            continue
        if dirs == {"U", "R"}:
            return 0
        if dirs == {"D", "R"}:
            return 3
        if dirs == {"D", "L"}:
            return 2
        if dirs == {"U", "L"}:
            return 1
    return None


def _parse_rotation_legend(region: np.ndarray) -> dict[int, int] | None:
    if np.any(region == SEP):
        return None

    mask = region != BG
    comps = _components(mask)
    if len(comps) < 2:
        return None

    mapping: dict[int, int] = {}
    for comp in comps:
        if len(comp) != 3:
            return None
        colors = {int(region[r, c]) for r, c in comp}
        if len(colors) != 1:
            return None
        color = next(iter(colors))
        rotation = _rotation_from_marker(comp)
        if rotation is None or color in mapping:
            return None
        mapping[color] = rotation
    return mapping


def _extract_stamp(region: np.ndarray) -> tuple[int, np.ndarray] | None:
    mask = (region != BG) & (region != SEP)
    comps = _components(mask)
    if len(comps) != 1:
        return None
    comp = comps[0]
    if len(comp) <= 3:
        return None
    r0, c0, r1, c1 = _bbox(comp)
    crop = region[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = crop[(crop != BG) & (crop != SEP)]
    if colors.size == 0:
        return None
    color = Counter(map(int, colors.tolist())).most_common(1)[0][0]
    return color, crop


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, np.ndarray]]:
    regions = _top_level_regions(grid)

    order: np.ndarray | None = None
    rotations: dict[int, int] | None = None
    order_index = -1
    rotation_index = -1

    for idx, region in enumerate(regions):
        parsed_order = _parse_order_legend(region)
        if parsed_order is not None:
            assert order is None, "multiple order legends found"
            order = parsed_order
            order_index = idx
            continue

        parsed_rotations = _parse_rotation_legend(region)
        if parsed_rotations is not None:
            assert rotations is None, "multiple rotation legends found"
            rotations = parsed_rotations
            rotation_index = idx

    assert order is not None, "order legend not found"
    assert rotations is not None, "rotation legend not found"

    stamps: dict[int, np.ndarray] = {}
    for idx, region in enumerate(regions):
        if idx in (order_index, rotation_index):
            continue
        stamp = _extract_stamp(region)
        assert stamp is not None, "failed to parse stamp region"
        color, crop = stamp
        assert color not in stamps, "stamp colors must be unique"
        stamps[color] = crop

    order_colors = {int(v) for v in order.reshape(-1)}
    assert order_colors == set(rotations), "rotation legend colors must match order legend"
    assert order_colors == set(stamps), "stamp colors must match order legend"
    return order, rotations, stamps


def _assemble_output(
    order: np.ndarray, rotations: dict[int, int], stamps: dict[int, np.ndarray]
) -> np.ndarray:
    tiles = [
        [np.rot90(stamps[int(color)], rotations[int(color)]) for color in row]
        for row in order
    ]
    row_heights = [max(tile.shape[0] for tile in row) for row in tiles]
    col_widths = [
        max(tiles[r][c].shape[1] for r in range(len(tiles))) for c in range(len(tiles[0]))
    ]

    out = np.full((sum(row_heights), sum(col_widths)), BG, dtype=int)
    r_off = 0
    for r, row in enumerate(tiles):
        c_off = 0
        for c, tile in enumerate(row):
            h, w = tile.shape
            out[r_off : r_off + h, c_off : c_off + w] = tile
            c_off += col_widths[c]
        r_off += row_heights[r]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read stamp identity, rotation, and order from separate regions and assemble the rotated stamps.
    """
    g = np.asarray(grid, dtype=int)
    order, rotations, stamps = _analyze_grid(g)
    return _assemble_output(order, rotations, stamps)


def _marker_block(color: int, rotation: int) -> np.ndarray:
    block = np.zeros((2, 2), dtype=int)
    cells_by_rotation = {
        0: [(0, 0), (1, 0), (1, 1)],
        1: [(0, 1), (1, 0), (1, 1)],
        2: [(0, 0), (0, 1), (1, 1)],
        3: [(0, 0), (0, 1), (1, 0)],
    }
    for r, c in cells_by_rotation[rotation]:
        block[r, c] = color
    return block


def _mask_signature(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _random_stamp_mask(size: int, rng: random.Random) -> np.ndarray | None:
    target_lo = size + 2
    target_hi = size * size - 2
    for _ in range(300):
        target = rng.randint(target_lo, target_hi)
        start = (rng.randrange(size), rng.randrange(size))
        blob = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < size and 0 <= nc < size:
                frontier.add((nr, nc))

        while frontier and len(blob) < target:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            blob.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in blob:
                    frontier.add((nr, nc))

        if len(blob) != target:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in blob:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    return None


def _choose_layout(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int], int]:
    options = [
        ((2, 3), (2, 2), 4),
        ((3, 2), (2, 2), 4),
        ((2, 4), (2, 3), 6),
        ((2, 4), (3, 2), 6),
        ((4, 2), (2, 3), 6),
        ((4, 2), (3, 2), 6),
    ]
    return rng.choice(options)


def _make_order_panel(
    colors: list[int], order_shape: tuple[int, int], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    order_rows, order_cols = order_shape
    cell = 3
    panel = np.full(
        (order_rows * cell + (order_rows - 1), order_cols * cell + (order_cols - 1)),
        SEP,
        dtype=int,
    )
    perm = colors[:]
    rng.shuffle(perm)
    order = np.array(perm, dtype=int).reshape(order_rows, order_cols)
    for r in range(order_rows):
        for c in range(order_cols):
            r0 = r * (cell + 1)
            c0 = c * (cell + 1)
            panel[r0 : r0 + cell, c0 : c0 + cell] = order[r, c]
    return panel, order


def _make_rotation_panel(
    colors: list[int], rotations: dict[int, int], rng: random.Random
) -> np.ndarray:
    cols_options = [2]
    if len(colors) >= 6:
        cols_options.append(3)
    marker_cols = rng.choice(cols_options)
    marker_rows = (len(colors) + marker_cols - 1) // marker_cols
    panel = np.full(
        (marker_rows * 2 + (marker_rows - 1), marker_cols * 2 + (marker_cols - 1)),
        BG,
        dtype=int,
    )
    perm = colors[:]
    rng.shuffle(perm)
    for idx, color in enumerate(perm):
        rr = idx // marker_cols
        cc = idx % marker_cols
        r0 = rr * 3
        c0 = cc * 3
        block = _marker_block(color, rotations[color])
        panel[r0 : r0 + 2, c0 : c0 + 2][block != BG] = color
    return panel


def _pick_special_cells(
    grid_shape: tuple[int, int], rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    rows, cols = grid_shape
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    order_cell = rng.choice(cells)
    rotation_choices = [cell for cell in cells if cell[0] != order_cell[0] and cell[1] != order_cell[1]]
    rotation_cell = rng.choice(rotation_choices)
    return order_cell, rotation_cell


def _cell_starts(lengths: list[int]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for idx, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if idx != len(lengths) - 1:
            cursor += 1
    return starts


def _build_grid(
    row_heights: list[int], col_widths: list[int]
) -> tuple[np.ndarray, list[int], list[int]]:
    total_h = sum(row_heights) + len(row_heights) - 1
    total_w = sum(col_widths) + len(col_widths) - 1
    grid = np.full((total_h, total_w), BG, dtype=int)

    row_starts = _cell_starts(row_heights)
    col_starts = _cell_starts(col_widths)
    for r, height in enumerate(row_heights[:-1]):
        sep_row = row_starts[r] + height
        grid[sep_row, :] = SEP
    for c, width in enumerate(col_widths[:-1]):
        sep_col = col_starts[c] + width
        grid[:, sep_col] = SEP
    return grid, row_starts, col_starts


def _paint_stamp(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    stamp: np.ndarray,
    rng: random.Random,
) -> None:
    sh, sw = stamp.shape
    r0 = top + rng.randint(0, height - sh)
    c0 = left + rng.randint(0, width - sw)
    grid[r0 : r0 + sh, c0 : c0 + sw][stamp != BG] = stamp[stamp != BG]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_shape, order_shape, stamp_count = _choose_layout(rng)
    top_rows, top_cols = top_shape
    order_cell, rotation_cell = _pick_special_cells(top_shape, rng)

    colors = rng.sample(list(STAMP_COLORS), stamp_count)
    stamp_size = rng.randint(3, 5)

    stamps: dict[int, np.ndarray] = {}
    seen_masks: set[bytes] = set()
    for color in colors:
        mask = _random_stamp_mask(stamp_size, rng)
        if mask is None:
            return None
        sig = _mask_signature(mask)
        retry = 0
        while sig in seen_masks:
            mask = _random_stamp_mask(stamp_size, rng)
            if mask is None:
                return None
            sig = _mask_signature(mask)
            retry += 1
            if retry > 50:
                return None
        seen_masks.add(sig)
        stamp = np.full((stamp_size, stamp_size), BG, dtype=int)
        stamp[mask] = color
        stamps[color] = stamp

    rotations = {color: rng.randrange(4) for color in colors}
    if len(set(rotations.values())) == 1:
        chosen = rng.choice(colors)
        rotations[chosen] = (rotations[chosen] + 1) % 4

    order_panel, order = _make_order_panel(colors, order_shape, rng)
    rotation_panel = _make_rotation_panel(colors, rotations, rng)

    row_heights = [rng.randint(stamp_size + 1, stamp_size + 3) for _ in range(top_rows)]
    col_widths = [rng.randint(stamp_size + 1, stamp_size + 3) for _ in range(top_cols)]

    row_heights[order_cell[0]] = order_panel.shape[0]
    col_widths[order_cell[1]] = order_panel.shape[1]
    row_heights[rotation_cell[0]] = rotation_panel.shape[0]
    col_widths[rotation_cell[1]] = rotation_panel.shape[1]

    if order_panel.shape[0] <= rotation_panel.shape[0] and order_panel.shape[1] <= rotation_panel.shape[1]:
        return None

    grid, row_starts, col_starts = _build_grid(row_heights, col_widths)

    order_top = row_starts[order_cell[0]]
    order_left = col_starts[order_cell[1]]
    grid[
        order_top : order_top + order_panel.shape[0],
        order_left : order_left + order_panel.shape[1],
    ] = order_panel

    rot_top = row_starts[rotation_cell[0]]
    rot_left = col_starts[rotation_cell[1]]
    grid[
        rot_top : rot_top + rotation_panel.shape[0],
        rot_left : rot_left + rotation_panel.shape[1],
    ] = rotation_panel

    stamp_cells = [
        (r, c)
        for r in range(top_rows)
        for c in range(top_cols)
        if (r, c) not in (order_cell, rotation_cell)
    ]
    rng.shuffle(stamp_cells)
    for color, cell in zip(colors, stamp_cells):
        r, c = cell
        top = row_starts[r]
        left = col_starts[c]
        height = row_heights[r]
        width = col_widths[c]
        if height < stamp_size or width < stamp_size:
            return None
        _paint_stamp(grid, top, left, height, width, stamps[color], rng)

    expected = _assemble_output(order, rotations, stamps)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random puzzle instance with separate order and rotation legends.
    """
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
