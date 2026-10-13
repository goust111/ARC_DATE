from __future__ import annotations

import random

import numpy as np

BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MARKER = 6

_TILE_SHAPES = ((2, 3), (3, 2), (3, 3), (4, 2), (2, 4))
_MIN_SIZE = 12
_MAX_SIZE = 16
_MAX_ATTEMPTS = 500


def _settle_blue(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[out == BLUE] = BG

    for c in range(w):
        r = 0
        while r < h:
            while r < h and int(out[r, c]) != BG:
                r += 1
            start = r
            while r < h and int(out[r, c]) == BG:
                r += 1
            end = r - 1
            if start <= end:
                count = int(np.count_nonzero(g[start : end + 1, c] == BLUE))
                if count:
                    out[end - count + 1 : end + 1, c] = BLUE
    return out


def _consistent_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r, c in np.argwhere((grid == RED) | (grid == GREEN)):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(grid[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_unique_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    visible_colors = set(map(int, g[(g == RED) | (g == GREEN)].ravel()))
    assert visible_colors == {RED, GREEN}, "expected visible red/green pattern evidence"

    h, w = g.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _consistent_tile(g, row_period, col_period)
            if tile is not None and set(map(int, tile.ravel())) == {RED, GREEN}:
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "periodic pattern must be inferable"
    min_area = min(area for area, _rp, _cp, _tile in candidates)
    minimal = [
        (row_period, col_period, tile)
        for area, row_period, col_period, tile in candidates
        if area == min_area
    ]
    assert len(minimal) == 1, "minimum-area period must be unique"
    return minimal[0]


def _complete_markers(grid: np.ndarray) -> np.ndarray:
    row_period, col_period, tile = _infer_unique_tile(grid)
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == MARKER):
        rr = int(r) % row_period
        cc = int(c) % col_period
        out[int(r), int(c)] = int(tile[rr, cc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop blue water straight down, then complete the magenta holes from the settled red/green periodic tile.

    Input structure:
    - The grid uses black background with blue water pixels, grey obstacle pixels, and a rectangular red/green periodic patch.
    - Some cells inside that periodic patch are replaced by magenta missing-value markers.
    - Blue pixels start in background cells and share the same global (row, col) coordinate system as the pattern.

    Rule:
    - First simulate vertical gravity on every blue pixel within its column: it falls through black cells and stops just above the first non-black blocker or the bottom boundary, stacking with other settled blue pixels.
    - After settling, ignore blue and magenta cells and infer the unique minimum-area rectangular row/column period of the remaining red/green cells using absolute coordinates modulo that period.
    - Replace every magenta marker by the inferred tile color at the same global residue, keep settled blue pixels, and leave all other cells unchanged.

    Key insight:
    - The same coordinate frame drives both steps: blue pixels must be settled in place first, while the red/green evidence reveals a globally anchored periodic tile that supplies the missing marker colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    settled = _settle_blue(g)
    return _complete_markers(settled)


def _min_period_area(tile: np.ndarray) -> int:
    h, w = tile.shape
    areas = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                areas.append(row_period * col_period)
    return min(areas)


def _random_tile(shape: tuple[int, int]) -> np.ndarray | None:
    row_period, col_period = shape
    area = row_period * col_period
    for _ in range(200):
        red_fraction = random.uniform(0.3, 0.7)
        tile = np.where(
            np.random.random((row_period, col_period)) < red_fraction,
            RED,
            GREEN,
        ).astype(int)
        if set(map(int, tile.ravel())) != {RED, GREEN}:
            continue
        if _min_period_area(tile) != area:
            continue
        return tile
    return None


def _paint_periodic_patch(
    grid: np.ndarray,
    tile: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> None:
    row_period, col_period = tile.shape
    rows = (np.arange(top, top + height)[:, None]) % row_period
    cols = (np.arange(left, left + width)[None, :]) % col_period
    grid[top : top + height, left : left + width] = tile[rows, cols]


def _choose_patch_dims(full: int, period: int, side_margin: int) -> int:
    low = period * 2
    high = min(full - side_margin, period * 3)
    return random.randint(low, high)


def _add_markers(
    grid: np.ndarray,
    patch_top: int,
    patch_left: int,
    patch_height: int,
    patch_width: int,
    tile_shape: tuple[int, int],
) -> bool:
    row_period, col_period = tile_shape
    residue_to_coords: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(patch_top, patch_top + patch_height):
        for c in range(patch_left, patch_left + patch_width):
            residue_to_coords.setdefault((r % row_period, c % col_period), []).append((r, c))

    preserve = {random.choice(coords) for coords in residue_to_coords.values()}
    remaining = [
        coord
        for coords in residue_to_coords.values()
        for coord in coords
        if coord not in preserve
    ]
    if len(remaining) < 5:
        return False

    marker_count = random.randint(3, 5)
    red_choices = [coord for coord in remaining if int(grid[coord]) == RED]
    green_choices = [coord for coord in remaining if int(grid[coord]) == GREEN]
    if not red_choices or not green_choices:
        return False

    markers: set[tuple[int, int]] = {
        random.choice(red_choices),
        random.choice(green_choices),
    }
    pool = [coord for coord in remaining if coord not in markers]
    if len(pool) < marker_count - len(markers):
        return False
    markers.update(random.sample(pool, marker_count - len(markers)))

    for r, c in markers:
        grid[r, c] = MARKER
    return True


def _place_segment_obstacle(grid: np.ndarray, row: int) -> bool:
    h, w = grid.shape
    for _ in range(50):
        length = 1 if random.random() < 0.45 else random.randint(2, min(3, w))
        left = random.randint(0, w - length)
        if np.all(grid[row, left : left + length] == BG):
            grid[row, left : left + length] = GREY
            return True
    return False


def _add_obstacles(grid: np.ndarray) -> bool:
    h, _w = grid.shape
    bottom_rows = [h - 1]
    if random.random() < 0.5:
        bottom_rows.append(h - 2)

    middle_low = max(h // 2, 3)
    middle_high = max(middle_low, h - 4)
    middle_rows = [random.randint(middle_low, middle_high)]
    if random.random() < 0.5:
        middle_rows.append(random.randint(middle_low, middle_high))

    placed = 0
    for row in bottom_rows + middle_rows:
        if _place_segment_obstacle(grid, row):
            placed += 1
    return placed >= 2 and int(np.count_nonzero(grid == GREY)) >= 2


def _zero_segments(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    upper_limit = (h - 1) // 2
    segments: list[dict[str, object]] = []
    for c in range(w):
        r = 0
        while r < h:
            while r < h and int(grid[r, c]) != BG:
                r += 1
            start = r
            while r < h and int(grid[r, c]) == BG:
                r += 1
            end = r - 1
            if start <= end:
                sources = [row for row in range(start, min(end, upper_limit) + 1)]
                if sources:
                    segments.append({"col": c, "start": start, "end": end, "sources": sources})
    return segments


def _place_blue_sources(grid: np.ndarray, require_stuck: bool) -> tuple[bool, dict[str, int]]:
    segments = _zero_segments(grid)
    if not segments:
        return False, {}

    blue_count = random.randint(4, 6)
    counts = [0] * len(segments)
    used_sources = [set() for _ in segments]
    placements: list[tuple[int, int, int]] = []
    moved = 0
    stuck = 0

    def options(kind: str | None) -> list[tuple[int, int, int]]:
        opts: list[tuple[int, int, int]] = []
        for idx, seg in enumerate(segments):
            start = int(seg["start"])
            end = int(seg["end"])
            final_row = end - counts[idx]
            if final_row < start:
                continue
            for source_row in seg["sources"]:
                if source_row in used_sources[idx] or source_row > final_row:
                    continue
                is_moving = source_row < final_row
                if kind == "moving" and not is_moving:
                    continue
                if kind == "stuck" and source_row != final_row:
                    continue
                opts.append((idx, source_row, final_row))
        return opts

    def commit(choice: tuple[int, int, int]) -> None:
        nonlocal moved, stuck
        idx, source_row, final_row = choice
        counts[idx] += 1
        used_sources[idx].add(source_row)
        placements.append((source_row, int(segments[idx]["col"]), final_row))
        if source_row < final_row:
            moved += 1
        else:
            stuck += 1

    moving_opts = options("moving")
    if not moving_opts:
        return False, {}
    commit(random.choice(moving_opts))

    while len(placements) < blue_count:
        if require_stuck and stuck == 0:
            stuck_opts = options("stuck")
            if not stuck_opts:
                return False, {}
            commit(random.choice(stuck_opts))
            continue

        kind = "moving" if random.random() < 0.7 else None
        opts = options(kind)
        if not opts:
            opts = options(None)
        if not opts:
            return False, {}
        commit(random.choice(opts))

    if moved == 0 or (require_stuck and stuck == 0):
        return False, {}

    for source_row, col, _final_row in placements:
        grid[source_row, col] = BLUE

    return True, {"moved": moved, "stuck": stuck}


def _count_blue_motion(input_grid: np.ndarray, settled_grid: np.ndarray) -> int:
    return int(
        len(set(map(tuple, np.argwhere(input_grid == BLUE))) ^ set(map(tuple, np.argwhere(settled_grid == BLUE))))
        // 2
    )


def _gen_attempt() -> np.ndarray | None:
    tile_shape = random.choice(_TILE_SHAPES)
    tile = _random_tile(tile_shape)
    if tile is None:
        return None

    row_period, col_period = tile_shape
    height = random.randint(max(_MIN_SIZE, row_period * 2 + 4), _MAX_SIZE)
    width = random.randint(max(_MIN_SIZE, col_period * 2 + 3), _MAX_SIZE)
    patch_height = _choose_patch_dims(height, row_period, 3)
    patch_width = _choose_patch_dims(width, col_period, 2)
    patch_top = random.randint(2, height - patch_height - 1)
    patch_left = random.randint(1, width - patch_width - 1)

    grid = np.zeros((height, width), dtype=int)
    _paint_periodic_patch(grid, tile, patch_top, patch_left, patch_height, patch_width)
    if not _add_markers(grid, patch_top, patch_left, patch_height, patch_width, tile_shape):
        return None
    if not _add_obstacles(grid):
        return None

    require_stuck = random.random() < 0.55
    ok, blue_meta = _place_blue_sources(grid, require_stuck=require_stuck)
    if not ok:
        return None

    input_grid = grid
    settled = _settle_blue(input_grid)
    output_grid = main(input_grid)

    try:
        row_period_out, col_period_out, tile_out = _infer_unique_tile(settled)
    except AssertionError:
        return None

    if (row_period_out, col_period_out) != tile.shape:
        if not np.array_equal(tile_out, tile):
            return None
    if not np.array_equal(output_grid, _complete_markers(settled)):
        return None
    if np.count_nonzero(input_grid == MARKER) not in (3, 4, 5):
        return None
    if np.count_nonzero(input_grid == BLUE) not in (4, 5, 6):
        return None
    if np.any(np.argwhere(input_grid == BLUE)[:, 0] > (height - 1) // 2):
        return None
    if _count_blue_motion(input_grid, settled) <= 0:
        return None
    if require_stuck and blue_meta["stuck"] <= 0:
        return None
    if np.any((settled == BLUE) & (input_grid == MARKER)):
        return None

    gravity_only = settled
    completion_only = _complete_markers(input_grid)
    if np.array_equal(gravity_only, output_grid):
        return None
    if np.array_equal(completion_only, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-period-completion input")
