import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, GREY))

GEN_ATTEMPTS = 400
MAX_GRID_H = 34
MAX_GRID_W = 32


def _black_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _pipe_rows(grid: np.ndarray, bottom_row: int, gap0: int, gap1: int) -> list[int] | None:
    h, w = grid.shape
    left_wall = gap0 - 1
    right_wall = gap1 + 1
    if left_wall < 0 or right_wall >= w:
        return None

    rows: list[int] = []
    for r in range(bottom_row + 1, h):
        interior = grid[r, gap0 : gap1 + 1]
        if np.any(interior == GREY):
            break
        if int(grid[r, left_wall]) != GREY or int(grid[r, right_wall]) != GREY:
            return None
        if np.any(interior != BLACK):
            return None
        rows.append(r)

    if not rows:
        return None
    return rows


def _pool_spec(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> dict | None:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    if int(grid[r1, c0]) != GREY or int(grid[r1, c1]) != GREY:
        return None
    if np.any(grid[r0, c0 : c1 + 1] != GREY):
        return None
    if np.any(grid[r0 : r1 + 1, c0] != GREY) or np.any(grid[r0 : r1 + 1, c1] != GREY):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == GREY):
        return None
    colors = np.unique(interior[(interior != BLACK) & (interior != GREY)])
    if len(colors) > 1:
        return None

    bottom = grid[r1, c0 + 1 : c1]
    if bottom.size == 0 or np.any((bottom != BLACK) & (bottom != GREY)):
        return None
    runs = _black_runs(bottom == BLACK)
    if len(runs) != 1:
        return None
    rel0, rel1 = runs[0]
    if rel1 - rel0 + 1 not in (1, 2):
        return None

    gap0 = c0 + 1 + rel0
    gap1 = c0 + 1 + rel1
    pipe_rows = _pipe_rows(grid, r1, gap0, gap1)
    if pipe_rows is None:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap0, gap1),
        "pipe_rows": pipe_rows,
    }


def _find_pools(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    pools: dict[tuple[int, int, int, int], dict] = {}

    for r0 in range(h - 3):
        grey_cols = np.where(grid[r0] == GREY)[0]
        if grey_cols.size < 4:
            continue
        for idx, c0 in enumerate(grey_cols[:-3]):
            for c1 in grey_cols[idx + 3 :]:
                if np.any(grid[r0, c0 : c1 + 1] != GREY):
                    continue
                for r1 in range(r0 + 3, h):
                    spec = _pool_spec(grid, int(r0), int(c0), int(r1), int(c1))
                    if spec is not None:
                        pools[(int(r0), int(c0), int(r1), int(c1))] = spec

    assert pools, "expected at least one valid pool"
    return [pools[key] for key in sorted(pools)]


def _settle_coords(r0: int, c0: int, r1: int, c1: int, gap0: int, gap1: int) -> list[tuple[int, int]]:
    cols = list(range(gap0, gap1 + 1))
    offset = 1
    while True:
        added = False
        left = gap0 - offset
        right = gap1 + offset
        if left >= c0 + 1:
            cols.append(left)
            added = True
        if right <= c1 - 1:
            cols.append(right)
            added = True
        if not added:
            break
        offset += 1

    coords: list[tuple[int, int]] = []
    for c in cols:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c))
    return coords


def _pipe_fill_coords(pipe_rows: list[int], gap0: int, gap1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r in pipe_rows:
        for c in range(gap0, gap1 + 1):
            coords.append((r, c))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each pool's colored cells, restack them above the outlet, and pour the
    same number of cells into the pipe below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    pools = _find_pools(g)
    out = g.copy()

    for pool in pools:
        r0, c0, r1, c1 = pool["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[(interior != BLACK) & (interior != GREY)] = BLACK

    for pool in pools:
        r0, c0, r1, c1 = pool["bbox"]
        gap0, gap1 = pool["gap"]
        source = g[r0 + 1 : r1, c0 + 1 : c1]
        colored = source[(source != BLACK) & (source != GREY)]
        if colored.size == 0:
            continue

        color = int(colored[0])
        count = int(colored.size)

        for r, c in _settle_coords(r0, c0, r1, c1, gap0, gap1)[:count]:
            out[r, c] = color
        for r, c in _pipe_fill_coords(pool["pipe_rows"], gap0, gap1)[:count]:
            out[r, c] = color

    return out


def _sample_basin(rng: random.Random, prev: dict | None) -> dict | None:
    outlet_w = rng.choice((1, 2))
    min_w = 6 if outlet_w == 1 else 7
    w = rng.randint(min_w, 10)
    h = rng.randint(5, 8)
    gap_rel = rng.randint(2, w - outlet_w - 3)

    if prev is None:
        c0 = 0
        r0 = 0
    else:
        lo = prev["gap1"] + 3 - w
        hi = prev["gap0"] - 2
        if lo > hi:
            return None
        c0 = rng.randint(lo, hi)
        r0 = prev["r1"] + prev["pipe_to_next"] + 1

    gap0 = c0 + gap_rel
    gap1 = gap0 + outlet_w - 1
    return {
        "r0": r0,
        "c0": c0,
        "r1": r0 + h - 1,
        "w": w,
        "h": h,
        "gap0": gap0,
        "gap1": gap1,
        "outlet_w": outlet_w,
    }


def _make_channel(rng: random.Random) -> dict | None:
    level_count = rng.choice((1, 2))
    basins: list[dict] = []

    for level in range(level_count):
        basin = _sample_basin(rng, basins[-1] if basins else None)
        if basin is None:
            return None
        if level + 1 < level_count:
            basin["pipe_to_next"] = rng.randint(2, 6)
        basins.append(basin)

    shift = -min(b["c0"] for b in basins)
    for basin in basins:
        basin["c0"] += shift
        basin["gap0"] += shift
        basin["gap1"] += shift

    width = max(b["c0"] + b["w"] for b in basins)
    body_height = max(b["r1"] for b in basins) + 1
    return {"basins": basins, "width": width, "body_height": body_height}


def _draw_pool(grid: np.ndarray, basin: dict, row_off: int, col_off: int) -> None:
    r0 = row_off + basin["r0"]
    c0 = col_off + basin["c0"]
    r1 = row_off + basin["r1"]
    c1 = c0 + basin["w"] - 1
    gap0 = col_off + basin["gap0"]
    gap1 = col_off + basin["gap1"]

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r1, gap0 : gap1 + 1] = BLACK


def _draw_pipe(grid: np.ndarray, basin: dict, row_off: int, col_off: int, pipe_len: int) -> None:
    start = row_off + basin["r1"] + 1
    stop = start + pipe_len
    left_wall = col_off + basin["gap0"] - 1
    right_wall = col_off + basin["gap1"] + 1
    grid[start:stop, left_wall] = GREY
    grid[start:stop, right_wall] = GREY


def _scatter_input_pixels(grid: np.ndarray, basin: dict, row_off: int, col_off: int, rng: random.Random) -> None:
    cells = [
        (row_off + r, col_off + c)
        for r in range(basin["r0"] + 1, basin["r1"])
        for c in range(basin["c0"] + 1, basin["c0"] + basin["w"] - 1)
    ]
    for r, c in rng.sample(cells, basin["count"]):
        grid[r, c] = basin["color"]


def _expected_bboxes(channels: list[dict]) -> set[tuple[int, int, int, int]]:
    boxes: set[tuple[int, int, int, int]] = set()
    for channel in channels:
        row_off = channel["row_off"]
        col_off = channel["col_off"]
        for basin in channel["basins"]:
            boxes.add(
                (
                    row_off + basin["r0"],
                    col_off + basin["c0"],
                    row_off + basin["r1"],
                    col_off + basin["c0"] + basin["w"] - 1,
                )
            )
    return boxes


def _assign_counts_and_colors(grid_h: int, channels: list[dict], rng: random.Random) -> bool:
    basin_refs: list[tuple[dict, dict]] = []
    for channel in channels:
        for basin in channel["basins"]:
            basin_refs.append((channel, basin))
    if len(basin_refs) > len(COLOR_POOL):
        return False

    colors = rng.sample(list(COLOR_POOL), len(basin_refs))
    rng.shuffle(colors)

    for color, (channel, basin) in zip(colors, basin_refs):
        basin["color"] = color
        if "pipe_to_next" in basin:
            pipe_len = basin["pipe_to_next"]
        else:
            abs_bottom = channel["row_off"] + basin["r1"]
            pipe_len = grid_h - 1 - abs_bottom
        if pipe_len < 2:
            return False

        basin["pipe_len"] = pipe_len
        area = (basin["h"] - 2) * (basin["w"] - 2)
        area_cap = area if area < 6 or rng.random() < 0.2 else area - 1
        pipe_cap = pipe_len * basin["outlet_w"]
        overflow_possible = area_cap > pipe_cap

        if overflow_possible and rng.random() < 0.5:
            basin["count"] = rng.randint(pipe_cap + 1, area_cap)
        else:
            basin["count"] = rng.randint(1, min(area_cap, pipe_cap if pipe_cap > 0 else area_cap))

    return True


def _render_scene(grid_h: int, grid_w: int, channels: list[dict], rng: random.Random) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    for channel in channels:
        for basin in channel["basins"]:
            _draw_pool(grid, basin, channel["row_off"], channel["col_off"])
    for channel in channels:
        for basin in channel["basins"]:
            _draw_pipe(grid, basin, channel["row_off"], channel["col_off"], basin["pipe_len"])
    for channel in channels:
        for basin in channel["basins"]:
            _scatter_input_pixels(grid, basin, channel["row_off"], channel["col_off"], rng)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    channel_count = rng.choice((1, 2))
    channels: list[dict] = []
    for _ in range(channel_count):
        channel = _make_channel(rng)
        if channel is None:
            return None
        channels.append(channel)

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    x = left_margin
    for idx, channel in enumerate(channels):
        channel["col_off"] = x
        x += channel["width"]
        if idx + 1 < len(channels):
            x += rng.randint(3, 6)
    grid_w = x + right_margin

    for channel in channels:
        channel["row_off"] = rng.randint(1, 4)
    bottom_extra = rng.randint(3, 8)
    grid_h = max(channel["row_off"] + channel["body_height"] for channel in channels) + bottom_extra

    if grid_h > MAX_GRID_H or grid_w > MAX_GRID_W:
        return None
    if not _assign_counts_and_colors(grid_h, channels, rng):
        return None

    grid = _render_scene(grid_h, grid_w, channels, rng)
    pools = _find_pools(grid)
    if _expected_bboxes(channels) != {tuple(pool["bbox"]) for pool in pools}:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pool-and-pipe input")
