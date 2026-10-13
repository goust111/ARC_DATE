import random

import numpy as np


BG = 8
SEPARATOR = 9
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7]
BLOCK_SIZE = 3
MAX_TAIL_LEN = 5
GEN_RETRIES = 300
REGION_COUNT_RANGE = (2, 3)
REGION_SIZE_RANGE = (8, 13)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _separator_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[tuple[int, int], int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not row_mask.any() and not col_mask.any():
            continue
        coords = np.argwhere(grid == color)
        if all(bool(row_mask[r] or col_mask[c]) for r, c in coords):
            score = (int(row_mask.sum() + col_mask.sum()), int(coords.shape[0]))
            candidates.append((score, int(color)))
    assert candidates, "expected a separator color occupying full rows or columns"
    candidates.sort()
    return candidates[-1][1]


def _non_separator_intervals(size: int, separator_indices: np.ndarray) -> list[tuple[int, int]]:
    bounds = [-1] + [int(i) for i in separator_indices] + [int(size)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            intervals.append((int(left + 1), int(right - 1)))
    return intervals


def _is_consecutive(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    ordered = np.sort(values.astype(int))
    expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1)
    return np.array_equal(ordered, expected)


def _region_object(subgrid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    colors = [int(c) for c in np.unique(subgrid) if int(c) != bg]
    assert len(colors) == 2, "each region must contain one block color and one tail color"

    block_color = None
    block_top = None
    block_bbox = None
    tail_coords = None
    for color in colors:
        coords = np.argwhere(subgrid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (
            int(coords.shape[0]) == BLOCK_SIZE * BLOCK_SIZE
            and int(r1 - r0 + 1) == BLOCK_SIZE
            and int(c1 - c0 + 1) == BLOCK_SIZE
            and np.all(subgrid[r0 : r1 + 1, c0 : c1 + 1] == color)
        ):
            assert block_color is None, "region cannot contain multiple 3x3 blocks"
            block_color = int(color)
            block_top = (int(r0), int(c0))
            block_bbox = (int(r0), int(c0), int(r1), int(c1))
        else:
            tail_coords = coords.astype(int)

    assert block_color is not None and block_top is not None and block_bbox is not None
    assert tail_coords is not None and int(tail_coords.shape[0]) >= 1

    rows = tail_coords[:, 0]
    cols = tail_coords[:, 1]
    straight = (np.all(rows == rows[0]) and _is_consecutive(cols)) or (
        np.all(cols == cols[0]) and _is_consecutive(rows)
    )
    assert straight, "tail must be a straight contiguous line"

    r0, c0, r1, c1 = block_bbox
    block_cells = {
        (r, c)
        for r in range(int(r0), int(r1) + 1)
        for c in range(int(c0), int(c1) + 1)
    }
    adjacent_tail = []
    for r, c in tail_coords:
        for dr, dc in DIRS:
            if (int(r + dr), int(c + dc)) in block_cells:
                adjacent_tail.append((int(r), int(c)))
                break
    assert len(adjacent_tail) == 1, "tail must touch the block at exactly one cell"
    tr, tc = adjacent_tail[0]

    if tc == c0 - 1 and r0 <= tr <= r1:
        tail_dir = LEFT
    elif tc == c1 + 1 and r0 <= tr <= r1:
        tail_dir = RIGHT
    elif tr == r0 - 1 and c0 <= tc <= c1:
        tail_dir = UP
    elif tr == r1 + 1 and c0 <= tc <= c1:
        tail_dir = DOWN
    else:
        raise AssertionError("tail must attach orthogonally to one side of the block")

    move_dir = (-tail_dir[0], -tail_dir[1])
    return block_color, block_top, move_dir, int(tail_coords.shape[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 3x3 block opposite to its tail direction by the tail length,
    clipping the translated block at separator-defined region boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    separator = _separator_color(g, bg)
    row_mask = np.all(g == separator, axis=1)
    col_mask = np.all(g == separator, axis=0)

    row_regions = _non_separator_intervals(g.shape[0], np.flatnonzero(row_mask))
    col_regions = _non_separator_intervals(g.shape[1], np.flatnonzero(col_mask))
    assert row_regions and col_regions, "separator grid must define at least one region"

    out = g.copy()
    for rr0, rr1 in row_regions:
        for cc0, cc1 in col_regions:
            sub = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if np.all(sub == bg):
                continue

            block_color, (br0, bc0), (dr, dc), tail_len = _region_object(sub, bg)
            region_out = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
            region_out[sub != bg] = bg

            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    nr = int(br0 + r + dr * tail_len)
                    nc = int(bc0 + c + dc * tail_len)
                    if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                        region_out[nr, nc] = block_color

    return out


def _tail_cells(block_top: tuple[int, int], move_dir: tuple[int, int], tail_len: int) -> list[tuple[int, int]]:
    br, bc = block_top
    center = (int(br + 1), int(bc + 1))
    tail_dir = (-move_dir[0], -move_dir[1])
    return [
        (int(center[0] + tail_dir[0] * step), int(center[1] + tail_dir[1] * step))
        for step in range(2, int(tail_len) + 2)
    ]


def _kept_block_pixels(
    height: int, width: int, block_top: tuple[int, int], move_dir: tuple[int, int], tail_len: int
) -> int:
    br, bc = block_top
    dr, dc = move_dir
    kept = 0
    for r in range(BLOCK_SIZE):
        for c in range(BLOCK_SIZE):
            nr = int(br + r + dr * tail_len)
            nc = int(bc + c + dc * tail_len)
            if 0 <= nr < height and 0 <= nc < width:
                kept += 1
    return kept


def _placement_options(height: int, width: int, move_dir: tuple[int, int]) -> dict[bool, dict[int, list[tuple[int, int]]]]:
    options: dict[bool, dict[int, list[tuple[int, int]]]] = {False: {}, True: {}}
    for tail_len in range(1, MAX_TAIL_LEN + 1):
        for br in range(height - BLOCK_SIZE + 1):
            for bc in range(width - BLOCK_SIZE + 1):
                tail = _tail_cells((br, bc), move_dir, tail_len)
                if not all(0 <= r < height and 0 <= c < width for r, c in tail):
                    continue
                kept = _kept_block_pixels(height, width, (br, bc), move_dir, tail_len)
                if kept == BLOCK_SIZE * BLOCK_SIZE:
                    options[False].setdefault(tail_len, []).append((int(br), int(bc)))
                elif kept >= BLOCK_SIZE:
                    options[True].setdefault(tail_len, []).append((int(br), int(bc)))
    return options


def _choose_placement(
    rng: random.Random, height: int, width: int, move_dir: tuple[int, int], clipped: bool
) -> tuple[int, tuple[int, int]]:
    options = _placement_options(height, width, move_dir)[clipped]
    assert options, "region must support the requested branch"
    tail_len = rng.choice(sorted(options))
    block_top = rng.choice(options[tail_len])
    return int(tail_len), (int(block_top[0]), int(block_top[1]))


def _paint_region(
    region: np.ndarray,
    move_dir: tuple[int, int],
    clipped: bool,
    block_color: int,
    tail_color: int,
    rng: random.Random,
) -> dict[str, object]:
    height, width = map(int, region.shape)
    tail_len, block_top = _choose_placement(rng, height, width, move_dir, clipped)

    br, bc = block_top
    region[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE] = int(block_color)
    for r, c in _tail_cells(block_top, move_dir, tail_len):
        region[r, c] = int(tail_color)

    kept = _kept_block_pixels(height, width, block_top, move_dir, tail_len)
    return {
        "move_dir": move_dir,
        "clipped": bool(clipped),
        "tail_len": int(tail_len),
        "kept": int(kept),
    }


def _build_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = SEPARATOR
    grid[-1, :] = SEPARATOR
    grid[:, 0] = SEPARATOR
    grid[:, -1] = SEPARATOR

    row_bounds = []
    cursor = 1
    for size in row_sizes:
        row_bounds.append((int(cursor), int(cursor + size - 1)))
        cursor += size
        grid[cursor, :] = SEPARATOR
        cursor += 1

    col_bounds = []
    cursor = 1
    for size in col_sizes:
        col_bounds.append((int(cursor), int(cursor + size - 1)))
        cursor += size
        grid[:, cursor] = SEPARATOR
        cursor += 1

    regions = [
        (rr0, rr1, cc0, cc1)
        for rr0, rr1 in row_bounds
        for cc0, cc1 in col_bounds
    ]
    return grid, regions


def _dir_key(move_dir: tuple[int, int]) -> str:
    return f"{move_dir[0]},{move_dir[1]}"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(*REGION_COUNT_RANGE)
    col_count = rng.randint(*REGION_COUNT_RANGE)
    if row_count * col_count < 4:
        return None

    row_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(row_count)]
    col_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(col_count)]
    grid, regions = _build_grid(row_sizes, col_sizes)

    directions = DIRS.copy()
    while len(directions) < len(regions):
        directions.append(rng.choice(DIRS))
    rng.shuffle(directions)

    clipped_flags = [True, False]
    while len(clipped_flags) < len(regions):
        clipped_flags.append(bool(rng.getrandbits(1)))
    rng.shuffle(clipped_flags)

    region_colors = PALETTE.copy()
    rng.shuffle(region_colors)
    meta = []
    for idx, (rr0, rr1, cc0, cc1) in enumerate(regions):
        region = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
        move_dir = directions[idx]
        clipped = clipped_flags[idx]

        block_color = region_colors[idx % len(region_colors)]
        tail_choices = [color for color in PALETTE if color != block_color]
        tail_color = rng.choice(tail_choices)
        try:
            info = _paint_region(region, move_dir, clipped, block_color, tail_color, rng)
        except AssertionError:
            return None
        meta.append(info)

    if not any(item["clipped"] for item in meta):
        return None
    if not any(not item["clipped"] for item in meta):
        return None
    if { _dir_key(item["move_dir"]) for item in meta } != { _dir_key(d) for d in DIRS }:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
