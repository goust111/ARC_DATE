import random
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
SEED_COLORS = (RED, BLUE, GREEN)
DIRS = ("up", "down", "left", "right")


def _grey_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == GREY)
    assert coords.size > 0, "expected grey room walls"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_dictionary(g: np.ndarray, maze_bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    h, w = g.shape
    r0, c0, r1, c1 = maze_bbox
    patterns: dict[int, np.ndarray] = {}

    for r in range(h - 3):
        for c in range(w - 3):
            color = int(g[r, c])
            if color not in SEED_COLORS:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue

            block = g[r : r + 4, c : c + 4]
            if np.any(block[0, 1:] != BLACK):
                continue
            if np.any(block[1:, 0] != BLACK):
                continue
            if not set(map(int, np.unique(block[1:, 1:]))) <= {BLACK, ORANGE}:
                continue

            mask = block[1:, 1:] == ORANGE
            if not bool(mask.any()):
                continue
            patterns[color] = mask.copy()

    assert set(patterns) == set(SEED_COLORS), "dictionary must provide red, blue, and green patterns"
    return patterns


def _line_indices(sub: np.ndarray) -> tuple[list[int], list[int]]:
    row_counts = np.sum(sub == GREY, axis=1)
    col_counts = np.sum(sub == GREY, axis=0)
    rows = [int(i) for i, count in enumerate(row_counts) if int(count) > sub.shape[1] // 2]
    cols = [int(i) for i, count in enumerate(col_counts) if int(count) > sub.shape[0] // 2]
    assert len(rows) >= 2 and len(cols) >= 2, "expected a rectangular wall grid"
    return rows, cols


def _gap_info(g: np.ndarray, top: int, bottom: int, left: int, right: int) -> tuple[str, int, int]:
    walls = {
        "up": [(top, c) for c in range(left + 1, right)],
        "down": [(bottom, c) for c in range(left + 1, right)],
        "left": [(r, left) for r in range(top + 1, bottom)],
        "right": [(r, right) for r in range(top + 1, bottom)],
    }
    hits = []
    for direction, coords in walls.items():
        gaps = [(r, c) for r, c in coords if int(g[r, c]) == BLACK]
        if gaps:
            assert len(gaps) == 1, "each wall can contain at most one gap"
            hits.append((direction, gaps[0][0], gaps[0][1]))
    assert len(hits) == 1, "each room must have exactly one wall gap"
    return hits[0]


def _rotate_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return mask
    if direction == "right":
        return np.rot90(mask, 3)
    if direction == "down":
        return np.rot90(mask, 2)
    if direction == "left":
        return np.rot90(mask, 1)
    raise AssertionError(direction)


def _placement_origin(direction: str, gap_r: int, gap_c: int) -> tuple[int, int]:
    if direction == "up":
        return gap_r - 3, gap_c - 1
    if direction == "down":
        return gap_r + 1, gap_c - 1
    if direction == "left":
        return gap_r - 1, gap_c - 3
    if direction == "right":
        return gap_r - 1, gap_c + 1
    raise AssertionError(direction)


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = out.shape
    assert 0 <= r0 and 0 <= c0 and r0 + 3 <= h and c0 + 3 <= w, "pattern must stay inside the grid"
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        assert int(out[rr, cc]) == BLACK, "grown patterns should not overlap existing objects"
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room's color-coded dictionary pattern outward through the room's only wall gap.

    Input structure:
    - The background is black.
    - Grey one-cell wall lines form a rectangular room grid; each room contains exactly one red, blue, or green seed pixel.
    - Every room has exactly one one-cell gap on one of its four walls.
    - A dictionary area sits below or to the right of the rooms; each 4x4 dictionary block has a colored top-left key pixel and a 3x3 orange-on-black binary pattern.

    Rule:
    - Find each room, locate its unique wall gap, and treat the gap direction as up, down, left, or right.
    - Use the seed color to select that color's 3x3 dictionary pattern, then rotate the pattern so its original top faces the gap direction.
    - Place the rotated pattern just outside the gap with its center row or center column aligned to the gap cell, repainting the orange cells with the seed color.
    - Clear the dictionary area to black while leaving all seeds and grey walls unchanged.

    Key insight:
    The same room gap determines both which side receives the output pattern and how the color-linked dictionary pattern must be rotated before placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    maze_bbox = _grey_bbox(g)
    patterns = _extract_dictionary(g, maze_bbox)

    r0, c0, r1, c1 = maze_bbox
    maze = g[r0 : r1 + 1, c0 : c1 + 1]
    line_rows, line_cols = _line_indices(maze)

    out = g.copy()
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    out[outside] = BLACK

    for i in range(len(line_rows) - 1):
        for j in range(len(line_cols) - 1):
            top = r0 + line_rows[i]
            bottom = r0 + line_rows[i + 1]
            left = c0 + line_cols[j]
            right = c0 + line_cols[j + 1]

            interior = g[top + 1 : bottom, left + 1 : right]
            seeds = [(int(r), int(c)) for r, c in np.argwhere(np.isin(interior, SEED_COLORS))]
            assert len(seeds) == 1, "each room must contain exactly one seed"
            seed_r, seed_c = seeds[0]
            seed_color = int(interior[seed_r, seed_c])

            direction, gap_r, gap_c = _gap_info(g, top, bottom, left, right)
            rotated = _rotate_mask(patterns[seed_color], direction)
            place_r, place_c = _placement_origin(direction, gap_r, gap_c)
            _paint_mask(out, rotated, seed_color, place_r, place_c)

    return out


def _all_pattern_masks() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        cells = np.array([(bits >> k) & 1 for k in range(9)], dtype=int).reshape(3, 3).astype(bool)
        count = int(cells.sum())
        if not 3 <= count <= 6:
            continue
        rots = {tuple(np.rot90(cells, k).ravel().tolist()) for k in range(4)}
        if len(rots) < 4:
            continue
        masks.append(cells)
    return masks


_PATTERN_POOL = _all_pattern_masks()


def _sample_patterns() -> dict[int, np.ndarray]:
    chosen: list[np.ndarray] = []
    pool = _PATTERN_POOL[:]
    random.shuffle(pool)

    for mask in pool:
        key = tuple(mask.ravel().tolist())
        key_rots = {tuple(np.rot90(mask, k).ravel().tolist()) for k in range(4)}
        if any(key == tuple(prev.ravel().tolist()) or key in {tuple(np.rot90(prev, k).ravel().tolist()) for k in range(4)} for prev in chosen):
            continue
        if any(key_rots & {tuple(np.rot90(prev, k).ravel().tolist()) for k in range(4)} for prev in chosen):
            continue
        chosen.append(mask)
        if len(chosen) == 3:
            break

    assert len(chosen) == 3
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    return {color: mask.copy() for color, mask in zip(colors, chosen, strict=True)}


def _room_sizes(count: int) -> list[int]:
    if count == 3:
        return [5, 5, 5]
    return [random.randint(5, 6) for _ in range(count)]


def _line_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in sizes:
        cur += size + 1
        pos.append(cur)
    return pos


def _eligible_dirs(n_rows: int, n_cols: int, i: int, j: int) -> list[str]:
    dirs = []
    if i == 0:
        dirs.append("up")
    if i == n_rows - 1:
        dirs.append("down")
    if j == 0:
        dirs.append("left")
    if j == n_cols - 1:
        dirs.append("right")
    return dirs


def _assign_gap_dirs(n_rows: int, n_cols: int) -> dict[tuple[int, int], str]:
    rooms = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    for _ in range(200):
        assignment = {}
        used = set()
        random.shuffle(rooms)
        for room in rooms:
            dirs = _eligible_dirs(n_rows, n_cols, room[0], room[1])
            assignment[room] = random.choice(dirs)
            used.add(assignment[room])
        if used == set(DIRS):
            return assignment
    raise AssertionError("failed to cover all four gap directions")


def _gap_coord(direction: str, top: int, bottom: int, left: int, right: int) -> tuple[int, int]:
    if direction in {"up", "down"}:
        gap_c = random.randint(left + 2, right - 2)
        return (top if direction == "up" else bottom), gap_c
    gap_r = random.randint(top + 2, bottom - 2)
    return gap_r, (left if direction == "left" else right)


def _dict_block(color: int, mask: np.ndarray) -> np.ndarray:
    block = np.zeros((4, 4), dtype=int)
    block[0, 0] = int(color)
    block[1:, 1:][mask] = ORANGE
    return block


def _place_dictionary(
    g: np.ndarray,
    maze_bbox: tuple[int, int, int, int],
    patterns: dict[int, np.ndarray],
    side: str,
) -> None:
    r0, c0, r1, c1 = maze_bbox
    items = list(patterns.items())
    random.shuffle(items)
    blocks = [_dict_block(color, mask) for color, mask in items]
    gap = random.randint(1, 2)

    if side == "bottom":
        total_w = 12 + 2 * gap
        row = random.randint(r1 + 5, g.shape[0] - 4)
        start = random.randint(0, g.shape[1] - total_w)
        col = start
        for block in blocks:
            g[row : row + 4, col : col + 4] = np.maximum(g[row : row + 4, col : col + 4], block)
            col += 4 + gap
        return

    total_h = 12 + 2 * gap
    col = random.randint(c1 + 5, g.shape[1] - 4)
    start = random.randint(0, g.shape[0] - total_h)
    row = start
    for block in blocks:
        g[row : row + 4, col : col + 4] = np.maximum(g[row : row + 4, col : col + 4], block)
        row += 4 + gap


def _build_candidate() -> np.ndarray:
    wide = bool(random.getrandbits(1))
    if wide:
        n_rows = 2
        n_cols = random.randint(2, 3)
    else:
        n_rows = random.randint(2, 3)
        n_cols = 2

    row_sizes = _room_sizes(n_rows)
    col_sizes = _room_sizes(n_cols)
    row_lines = _line_positions(row_sizes)
    col_lines = _line_positions(col_sizes)
    maze_h = row_lines[-1] + 1
    maze_w = col_lines[-1] + 1

    dict_side = random.choice(("bottom", "right"))
    min_h = maze_h + 6 + (5 if dict_side == "bottom" else 0)
    min_w = maze_w + 6 + (5 if dict_side == "right" else 0)
    h = random.randint(min_h, min(30, min_h + 2))
    w = random.randint(min_w, min(30, min_w + 2))

    top = random.randint(3, h - maze_h - (8 if dict_side == "bottom" else 3))
    left = random.randint(3, w - maze_w - (8 if dict_side == "right" else 3))

    g = np.zeros((h, w), dtype=int)
    maze_r0, maze_c0 = top, left
    maze_r1 = top + maze_h - 1
    maze_c1 = left + maze_w - 1

    for rr in row_lines:
        g[maze_r0 + rr, maze_c0 : maze_c1 + 1] = GREY
    for cc in col_lines:
        g[maze_r0 : maze_r1 + 1, maze_c0 + cc] = GREY

    gap_dirs = _assign_gap_dirs(n_rows, n_cols)
    patterns = _sample_patterns()

    for i in range(n_rows):
        for j in range(n_cols):
            top_wall = maze_r0 + row_lines[i]
            bottom_wall = maze_r0 + row_lines[i + 1]
            left_wall = maze_c0 + col_lines[j]
            right_wall = maze_c0 + col_lines[j + 1]

            direction = gap_dirs[(i, j)]
            gap_r, gap_c = _gap_coord(direction, top_wall, bottom_wall, left_wall, right_wall)
            g[gap_r, gap_c] = BLACK

            seed_color = random.choice(SEED_COLORS)
            seed_r = random.randint(top_wall + 1, bottom_wall - 1)
            seed_c = random.randint(left_wall + 1, right_wall - 1)
            g[seed_r, seed_c] = seed_color

    _place_dictionary(g, (maze_r0, maze_c0, maze_r1, maze_c1), patterns, dict_side)
    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_candidate()
        try:
            out = main(grid)
        except Exception:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
