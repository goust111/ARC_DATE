import random
from collections import Counter

import numpy as np


BG = 0
TOWER = 1
RECEIVER = 3
WALL = 5
PALETTE_POOL = (2, 4, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULARS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_palette(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    special = {bg, TOWER, RECEIVER, WALL}
    for col in range(width - 1, -1, -1):
        rows = np.flatnonzero(grid[:, col] != bg)
        if len(rows) != 4:
            continue
        if not np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + 4)):
            continue
        colors = [int(grid[int(r), col]) for r in rows]
        if len(set(colors)) != 4:
            continue
        if any(color in special for color in colors):
            continue
        return int(col), [int(r) for r in rows], colors
    raise AssertionError("expected one right-side palette column with four stacked colors")


def _ray_counts(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    tower_mask = g == TOWER
    wall_mask = g == WALL
    counts = np.zeros((height, width), dtype=int)

    for r_raw, c_raw in np.argwhere(tower_mask):
        r0 = int(r_raw)
        c0 = int(c_raw)
        for dr, dc in DIRS:
            r = int(r0 + dr)
            c = int(c0 + dc)
            while 0 <= r < height and 0 <= c < width:
                if bool(wall_mask[r, c]) or bool(tower_mask[r, c]):
                    break
                counts[r, c] += 1
                r += dr
                c += dc
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each receiver by the number of unobstructed tower rays that cross it.

    Input structure:
    - Background is black.
    - Blue singleton pixels are signal towers and green singleton pixels are receivers.
    - Gray pixels form 4-connected wall components that block straight rays.
    - A palette column on the right contains four stacked colors C1..C4 for hit counts 1..4+.

    Rule:
    - Every blue tower emits invisible rays upward, downward, leftward, and rightward.
    - A ray advances cell by cell until it would leave the grid or enter a gray wall
      pixel or another blue tower; the blocking cell is not crossed.
    - Count how many distinct rays pass through each green receiver.
    - Replace a receiver hit N times with palette color Cmin(N, 4); receivers with
      zero hits stay green. Towers, walls, background, and the palette stay unchanged.

    Key insight:
    - The visible output is only a recoloring of receivers, but the hidden computation
      is a four-direction ray cast whose counts are decoded by the right-side palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    bg = _most_common_color(g)
    _palette_col, _palette_rows, palette = _find_palette(g, bg)

    counts = _ray_counts(g)
    out = g.copy()
    for r_raw, c_raw in np.argwhere(g == RECEIVER):
        r = int(r_raw)
        c = int(c_raw)
        hits = int(counts[r, c])
        if hits > 0:
            out[r, c] = int(palette[min(hits, 4) - 1])
    return out


def _receiver_slots(height: int, play_width: int) -> tuple[list[int], list[int]]:
    rows = list(range(3, height - 3, 2))
    cols = list(range(3, play_width - 3, 2))
    return rows, cols


def _tower_candidates(height: int, play_width: int, r: int, c: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == LEFT:
        return [(r, cc) for cc in range(0, c, 2) if c - cc >= 3]
    if direction == RIGHT:
        return [(r, cc) for cc in range(c + 1, play_width, 2) if cc - c >= 3]
    if direction == UP:
        return [(rr, c) for rr in range(0, r, 2) if r - rr >= 3]
    if direction == DOWN:
        return [(rr, c) for rr in range(r + 1, height, 2) if rr - r >= 3]
    raise AssertionError("unknown direction")


def _decoy_anchor(r: int, c: int, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return int(r + dr), int(c + dc)


def _place_branch_wall(
    grid: np.ndarray,
    anchor: tuple[int, int],
    direction: tuple[int, int],
    play_width: int,
) -> list[tuple[int, int]] | None:
    height = grid.shape[0]
    if int(grid[anchor]) != BG:
        return None

    perps = list(PERPENDICULARS[direction])
    random.shuffle(perps)
    for dr, dc in perps:
        rr = int(anchor[0] + dr)
        cc = int(anchor[1] + dc)
        if not (0 <= rr < height and 0 <= cc < play_width):
            continue
        if int(grid[rr, cc]) != BG:
            continue
        cells = [anchor, (rr, cc)]
        for wr, wc in cells:
            grid[wr, wc] = WALL
        return cells

    grid[anchor] = WALL
    return [anchor]


def _choose_decoys(opportunities: list[tuple[int, tuple[int, int]]]) -> list[tuple[int, tuple[int, int]]]:
    if not opportunities:
        return []

    horiz = [item for item in opportunities if item[1] in (LEFT, RIGHT)]
    vert = [item for item in opportunities if item[1] in (UP, DOWN)]

    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, tuple[int, int]]] = set()

    if horiz and vert:
        for pool in (horiz, vert):
            item = random.choice(pool)
            if item not in used:
                chosen.append(item)
                used.add(item)
    else:
        item = random.choice(opportunities)
        chosen.append(item)
        used.add(item)

    remaining = [item for item in opportunities if item not in used]
    max_total = min(3, len(opportunities))
    target_total = random.randint(len(chosen), max_total)
    if target_total > len(chosen):
        chosen.extend(random.sample(remaining, target_total - len(chosen)))
    return chosen


def _safe_free_wall_cell(
    grid: np.ndarray,
    r: int,
    c: int,
    play_width: int,
    receiver_rows: set[int],
    receiver_cols: set[int],
) -> bool:
    return (
        0 <= r < grid.shape[0]
        and 0 <= c < play_width
        and int(grid[r, c]) == BG
        and r not in receiver_rows
        and c not in receiver_cols
    )


def _try_place_free_wall(
    grid: np.ndarray,
    play_width: int,
    receiver_rows: set[int],
    receiver_cols: set[int],
) -> bool:
    height = grid.shape[0]
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(play_width)
        if _safe_free_wall_cell(grid, r, c, play_width, receiver_rows, receiver_cols)
    ]
    if not candidates:
        return False

    random.shuffle(candidates)
    for start in candidates[:40]:
        target_size = random.randint(2, 4)
        cells = {start}
        frontier = [start]

        while frontier and len(cells) < target_size:
            base = random.choice(frontier)
            nbrs = []
            for dr, dc in DIRS:
                rr = int(base[0] + dr)
                cc = int(base[1] + dc)
                if not _safe_free_wall_cell(grid, rr, cc, play_width, receiver_rows, receiver_cols):
                    continue
                if (rr, cc) in cells:
                    continue
                nbrs.append((rr, cc))
            if not nbrs:
                frontier.remove(base)
                continue
            chosen = random.choice(nbrs)
            cells.add(chosen)
            frontier.append(chosen)

        if len(cells) >= 2:
            for r, c in cells:
                grid[r, c] = WALL
            return True

    return False


def _expected_receiver_color(palette: list[int], hits: int) -> int:
    return RECEIVER if hits == 0 else int(palette[min(hits, 4) - 1])


def _gen_attempt() -> np.ndarray | None:
    extra_receivers = random.randint(0, 2)
    targets = [0, 1, 2, 3, 4] + [random.randint(0, 4) for _ in range(extra_receivers)]
    random.shuffle(targets)
    receiver_count = len(targets)

    min_dim = 2 * receiver_count + 5
    height = random.randint(max(15, min_dim), 24)
    play_width = random.randint(max(15, min_dim), 22)
    width = play_width + 2  # separator + palette

    grid = np.full((height, width), BG, dtype=int)
    palette = random.sample(PALETTE_POOL, 4)
    palette_top = random.randint(0, height - 4)
    grid[palette_top : palette_top + 4, width - 1] = np.array(palette, dtype=int)

    row_slots, col_slots = _receiver_slots(height, play_width)
    if len(row_slots) < receiver_count or len(col_slots) < receiver_count:
        return None

    rows = random.sample(row_slots, receiver_count)
    cols = random.sample(col_slots, receiver_count)
    random.shuffle(rows)
    random.shuffle(cols)

    receiver_rows = set(rows)
    receiver_cols = set(cols)
    records: list[dict[str, object]] = []

    for target, r, c in zip(targets, rows, cols):
        grid[r, c] = RECEIVER
        active_dirs = tuple(random.sample(DIRS, target))
        records.append(
            {
                "pos": (int(r), int(c)),
                "target": int(target),
                "dirs": active_dirs,
            }
        )

    for record in records:
        r, c = record["pos"]
        for direction in record["dirs"]:
            candidates = _tower_candidates(height, play_width, r, c, direction)
            random.shuffle(candidates)
            placed = False
            for rr, cc in candidates:
                if int(grid[rr, cc]) != BG:
                    continue
                grid[rr, cc] = TOWER
                placed = True
                break
            if not placed:
                return None

    opportunities: list[tuple[int, tuple[int, int]]] = []
    for index, record in enumerate(records):
        active = set(record["dirs"])
        for direction in DIRS:
            if direction not in active:
                opportunities.append((index, direction))

    decoys = _choose_decoys(opportunities)
    if not decoys:
        return None

    for index, direction in decoys:
        r, c = records[index]["pos"]
        anchor = _decoy_anchor(r, c, direction)
        if not (0 <= anchor[0] < height and 0 <= anchor[1] < play_width):
            return None
        if int(grid[anchor]) != BG:
            return None

        candidates = _tower_candidates(height, play_width, r, c, direction)
        candidates = [
            (rr, cc)
            for rr, cc in candidates
            if (direction == LEFT and cc < anchor[1])
            or (direction == RIGHT and cc > anchor[1])
            or (direction == UP and rr < anchor[0])
            or (direction == DOWN and rr > anchor[0])
        ]
        random.shuffle(candidates)
        placed = False
        for rr, cc in candidates:
            if int(grid[rr, cc]) != BG:
                continue
            grid[rr, cc] = TOWER
            placed = True
            break
        if not placed:
            return None

        if _place_branch_wall(grid, anchor, direction, play_width) is None:
            return None

    extra_walls = random.randint(0, 1)
    for _ in range(extra_walls):
        _try_place_free_wall(grid, play_width, receiver_rows, receiver_cols)

    hits = _ray_counts(grid)
    out = main(grid)

    for record in records:
        r, c = record["pos"]
        target = int(record["target"])
        if int(hits[r, c]) != target:
            return None
        if int(out[r, c]) != _expected_receiver_color(palette, target):
            return None

    if np.count_nonzero(grid == WALL) < 2:
        return None
    if np.count_nonzero(grid == TOWER) < 4:
        return None

    receiver_hit_counts = Counter(
        int(hits[r, c]) for r, c in [record["pos"] for record in records]
    )
    if receiver_hit_counts[0] == 0:
        return None
    if receiver_hit_counts[4] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
