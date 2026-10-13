import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
TOKEN = 9
INDICATOR_COLORS = (1, 2, 3, 4)
BOARD_SIZE = 17
BLOCK_SIZE = 5
STRIDE = 6
TOKEN_SIZE = 3
MAX_RETRIES = 600

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SOLID_PATCH = {(r, c) for r in range(3) for c in range(3)}
X_PATCH = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}
DIAMOND_PATCH = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
CARDINAL_PATCH = {(0, 1), (1, 0), (1, 2), (2, 1)}

SHAPE_TO_PATCH = {
    "solid": SOLID_PATCH,
    "x": X_PATCH,
    "diamond": DIAMOND_PATCH,
}
NEXT_SHAPE = {"solid": "diamond", "diamond": "x", "x": "solid"}
ADDRESS_MAP = {"diamond": (0, 2), "x": (1, 1), "solid": (2, 0)}


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _classify_token_at(grid: np.ndarray, r: int, c: int) -> str | None:
    if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
        return None
    if int(grid[r, c]) != TOKEN:
        return None
    patch = {
        (rr, cc)
        for rr in range(3)
        for cc in range(3)
        if int(grid[r - 1 + rr, c - 1 + cc]) == TOKEN
    }
    if patch == SOLID_PATCH:
        return "solid"
    if patch == DIAMOND_PATCH:
        return "diamond"
    extra = patch - X_PATCH
    if X_PATCH.issubset(patch) and extra.issubset(CARDINAL_PATCH) and len(extra) <= 1:
        return "x"
    return None


def _find_token(grid: np.ndarray) -> tuple[tuple[int, int], str]:
    found: list[tuple[tuple[int, int], str]] = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            shape = _classify_token_at(grid, r, c)
            if shape is not None:
                found.append(((r, c), shape))
    assert len(found) == 1, f"expected exactly one token center, got {found}"
    return found[0]


def _find_indicator(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    coords = np.argwhere((grid != BG) & (grid != GREY) & (grid != TOKEN))
    assert coords.shape[0] == 1, f"expected one indicator pixel, got {coords.shape[0]}"
    r, c = map(int, coords[0])
    color = int(grid[r, c])
    assert color in INDICATOR_COLORS
    assert sum(
        1
        for nr, nc in _neighbors4(r, c)
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == TOKEN
    ) == 1
    return (r, c), color


def _find_board_origin(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r0 in range(h - BOARD_SIZE + 1):
        for c0 in range(w - BOARD_SIZE + 1):
            sub = grid[r0 : r0 + BOARD_SIZE, c0 : c0 + BOARD_SIZE]
            if not np.all(sub[5, :] == GREY):
                continue
            if not np.all(sub[11, :] == GREY):
                continue
            if not np.all(sub[:, 5] == GREY):
                continue
            if not np.all(sub[:, 11] == GREY):
                continue
            non_sep = sub.copy()
            non_sep[5, :] = BG
            non_sep[11, :] = BG
            non_sep[:, 5] = BG
            non_sep[:, 11] = BG
            if not np.all((non_sep == BG) | (non_sep == GREY)):
                continue
            candidates.append((r0, c0))
    assert len(candidates) == 1, f"expected one board, got {candidates}"
    return candidates[0]


def _target_block(grid: np.ndarray, board_origin: tuple[int, int], original_shape: str) -> np.ndarray:
    transformed = NEXT_SHAPE[original_shape]
    br, bc = ADDRESS_MAP[transformed]
    r0 = board_origin[0] + br * STRIDE
    c0 = board_origin[1] + bc * STRIDE
    return grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 3×3 token into a block address, crop that 5×5 block, and recolor it using the wire endpoint color.

    Input structure:
    - A black background contains one 3×3 array of 5×5 blocks separated by full grey divider rows and columns.
    - Each 5×5 block contains a small grey foreground pattern, and a separate maroon 3×3 token sits to the right of or below the block array.
    - A maroon orthogonal wire starts at the token center and reaches a single colored indicator pixel outside the block array.

    Rule:
    - Identify whether the token is solid, diamond, or X-shaped, then advance it by one step in the cycle solid → diamond → X → solid.
    - Use the transformed shape as an address: diamond selects the upper-right block, X selects the center block, and solid selects the lower-left block.
    - Crop that 5×5 block and recolor every grey foreground cell inside it to the indicator pixel's color, leaving black cells unchanged.

    Key insight:
    - The token is not copied directly; its post-transform shape only chooses which block to crop, while the attached wire independently supplies the recolor parameter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    board_origin = _find_board_origin(g)
    (_token_center, token_shape) = _find_token(g)
    (_indicator_pos, indicator_color) = _find_indicator(g)

    out = _target_block(g, board_origin, token_shape)
    out[out == GREY] = indicator_color
    return out


def _shape_patch(shape: str) -> np.ndarray:
    patch = np.zeros((TOKEN_SIZE, TOKEN_SIZE), dtype=int)
    for r, c in SHAPE_TO_PATCH[shape]:
        patch[r, c] = TOKEN
    return patch


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_pattern(rng: random.Random) -> np.ndarray | None:
    area = rng.randint(3, 8)
    start = (rng.randrange(BLOCK_SIZE), rng.randrange(BLOCK_SIZE))
    cells = {start}
    while len(cells) < area:
        frontier = {
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE and (nr, nc) not in cells
        }
        if not frontier:
            return None
        cells.add(rng.choice(sorted(frontier)))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    if any(sum(1 for rr, _ in cells if rr == r) == BLOCK_SIZE for r in range(BLOCK_SIZE)):
        return None
    if any(sum(1 for _, cc in cells if cc == c) == BLOCK_SIZE for c in range(BLOCK_SIZE)):
        return None

    arr = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    for r, c in cells:
        arr[r, c] = GREY
    comps = _components(arr == GREY)
    if len(comps) != 1:
        return None
    return arr


def _generate_patterns(rng: random.Random) -> list[np.ndarray] | None:
    patterns: list[np.ndarray] = []
    seen = set()
    seen_shapes = set()
    while len(patterns) < 9:
        arr = _random_pattern(rng)
        if arr is None:
            return None
        key = arr.tobytes()
        shape_key = _normalize_cells({tuple(map(int, rc)) for rc in np.argwhere(arr == GREY)})
        if key in seen or shape_key in seen_shapes:
            continue
        seen.add(key)
        seen_shapes.add(shape_key)
        patterns.append(arr)
    return patterns


def _draw_board(grid: np.ndarray, board_r0: int, board_c0: int, patterns: list[np.ndarray]) -> None:
    grid[board_r0 + 5, board_c0 : board_c0 + BOARD_SIZE] = GREY
    grid[board_r0 + 11, board_c0 : board_c0 + BOARD_SIZE] = GREY
    grid[board_r0 : board_r0 + BOARD_SIZE, board_c0 + 5] = GREY
    grid[board_r0 : board_r0 + BOARD_SIZE, board_c0 + 11] = GREY
    for idx, patch in enumerate(patterns):
        br, bc = divmod(idx, 3)
        r0 = board_r0 + br * STRIDE
        c0 = board_c0 + bc * STRIDE
        mask = patch == GREY
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE][mask] = GREY


def _build_straight(center: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cr, cc = center
    dr, dc = direction
    path = [(cr + step * dr, cc + step * dc) for step in range(length + 1)]
    indicator = (cr + (length + 1) * dr, cc + (length + 1) * dc)
    return path, indicator


def _build_turn(
    center: tuple[int, int],
    dir1: tuple[int, int],
    len1: int,
    dir2: tuple[int, int],
    len2: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cr, cc = center
    d1r, d1c = dir1
    d2r, d2c = dir2
    path = [(cr, cc)]
    for step in range(1, len1 + 1):
        path.append((cr + step * d1r, cc + step * d1c))
    bend = path[-1]
    for step in range(1, len2 + 1):
        path.append((bend[0] + step * d2r, bend[1] + step * d2c))
    indicator = (bend[0] + (len2 + 1) * d2r, bend[1] + (len2 + 1) * d2c)
    return path, indicator


def _path_candidates(
    grid_shape: tuple[int, int],
    center: tuple[int, int],
    side: str,
    rng: random.Random,
) -> list[tuple[list[tuple[int, int]], tuple[int, int], str]]:
    del rng
    if side == "right":
        primary_dirs = [(0, 1), (-1, 0), (1, 0)]
    else:
        primary_dirs = [(1, 0), (0, -1), (0, 1)]
    candidates = []
    for dir1 in primary_dirs:
        for length in range(2, 7):
            path, indicator = _build_straight(center, dir1, length)
            candidates.append((path, indicator, "straight"))
        if side == "right":
            turns = [(0, 1)] if dir1 in {(-1, 0), (1, 0)} else [(-1, 0), (1, 0)]
        else:
            turns = [(1, 0)] if dir1 in {(0, -1), (0, 1)} else [(0, -1), (0, 1)]
        for dir2 in turns:
            for len1 in range(1, 5):
                for len2 in range(1, 5):
                    path, indicator = _build_turn(center, dir1, len1, dir2, len2)
                    candidates.append((path, indicator, "turn"))
    good = []
    for path, indicator, style in candidates:
        pts = path + [indicator]
        if len(set(pts)) != len(pts):
            continue
        if not all(_in_bounds(grid_shape, r, c) for r, c in pts):
            continue
        good.append((path, indicator, style))
    return good


def _place_token_and_wire(
    grid: np.ndarray,
    board_origin: tuple[int, int],
    shape: str,
    side: str,
    indicator_color: int,
    rng: random.Random,
) -> dict | None:
    board_r0, board_c0 = board_origin
    h, w = grid.shape
    if side == "right":
        col_choices = list(range(board_c0 + BOARD_SIZE + 1, w - 2))
        row_choices = list(range(0, h - 2))
    else:
        col_choices = list(range(0, w - 2))
        row_choices = list(range(board_r0 + BOARD_SIZE + 1, h - 2))
    token_positions = [(r0, c0) for r0 in row_choices for c0 in col_choices]
    rng.shuffle(token_positions)
    patch = _shape_patch(shape)

    for token_r0, token_c0 in token_positions:
        token_box = {
            (token_r0 + r, token_c0 + c)
            for r in range(TOKEN_SIZE)
            for c in range(TOKEN_SIZE)
        }
        if any(int(grid[r, c]) != BG for r, c in token_box):
            continue
        center = (token_r0 + 1, token_c0 + 1)
        candidates = _path_candidates(grid.shape, center, side, rng)
        rng.shuffle(candidates)
        for path, indicator, style in candidates:
            path_set = set(path)
            if any(
                int(grid[r, c]) != BG and (r, c) not in token_box
                for r, c in path_set
            ):
                continue
            if int(grid[indicator]) != BG:
                continue
            if any(
                int(grid[nr, nc]) != BG
                for nr, nc in _neighbors4(*indicator)
                if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in {path[-1]}
            ):
                continue
            grid[token_r0 : token_r0 + TOKEN_SIZE, token_c0 : token_c0 + TOKEN_SIZE] = np.maximum(
                grid[token_r0 : token_r0 + TOKEN_SIZE, token_c0 : token_c0 + TOKEN_SIZE],
                patch,
            )
            for r, c in path_set:
                grid[r, c] = TOKEN
            grid[indicator] = indicator_color
            return {
                "token_r0": token_r0,
                "token_c0": token_c0,
                "center": center,
                "style": style,
                "indicator": indicator,
            }
    return None


def _expected_from_meta(patterns: list[np.ndarray], original_shape: str, indicator_color: int) -> np.ndarray:
    tr, tc = ADDRESS_MAP[NEXT_SHAPE[original_shape]]
    patch = patterns[tr * 3 + tc].copy()
    patch[patch == GREY] = indicator_color
    return patch


def _manual_semantic_checks(
    grid: np.ndarray,
    patterns: list[np.ndarray],
    board_origin: tuple[int, int],
    original_shape: str,
    indicator_color: int,
) -> bool:
    transformed = NEXT_SHAPE[original_shape]
    expected = _expected_from_meta(patterns, original_shape, indicator_color)

    try:
        if _find_board_origin(grid) != board_origin:
            return False
        (_center, found_shape) = _find_token(grid)
        if found_shape != original_shape:
            return False
        (_indicator, found_color) = _find_indicator(grid)
        if found_color != indicator_color:
            return False
    except AssertionError:
        return False

    all_outputs = []
    for named_shape in ("diamond", "x", "solid"):
        br, bc = ADDRESS_MAP[named_shape]
        patch = patterns[br * 3 + bc].copy()
        patch[patch == GREY] = indicator_color
        all_outputs.append(patch)
    if len({patch.tobytes() for patch in all_outputs}) != 3:
        return False

    wrong_identity = patterns[ADDRESS_MAP[original_shape][0] * 3 + ADDRESS_MAP[original_shape][1]].copy()
    wrong_identity[wrong_identity == GREY] = indicator_color
    if np.array_equal(expected, wrong_identity):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    target_block = _target_block(grid, board_origin, original_shape)
    if np.array_equal(target_block, expected):
        return False
    if transformed == original_shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(["right", "bottom"])
    board_r0 = rng.randint(0, 2)
    board_c0 = rng.randint(0, 2)
    if side == "right":
        right_pad = rng.randint(5, 9)
        bottom_pad = rng.randint(0, 4)
    else:
        right_pad = rng.randint(0, 4)
        bottom_pad = rng.randint(5, 9)
    h = board_r0 + BOARD_SIZE + bottom_pad
    w = board_c0 + BOARD_SIZE + right_pad
    if max(h, w) > 30:
        return None

    patterns = _generate_patterns(rng)
    if patterns is None:
        return None

    original_shape = rng.choice(["solid", "diamond", "x"])
    indicator_color = rng.choice(INDICATOR_COLORS)
    board_origin = (board_r0, board_c0)

    grid = np.zeros((h, w), dtype=int)
    _draw_board(grid, board_r0, board_c0, patterns)
    placement = _place_token_and_wire(grid, board_origin, original_shape, side, indicator_color, rng)
    if placement is None:
        return None

    if not _manual_semantic_checks(grid, patterns, board_origin, original_shape, indicator_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
