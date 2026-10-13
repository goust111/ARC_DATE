import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MAROON = 9

REGION_LAYOUTS = ((2, 2), (2, 3), (3, 2))
REGION_HEIGHT_RANGE = (6, 8)
REGION_WIDTH_RANGE = (4, 7)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in _components_mask(grid != RED):
        r0, c0, r1, c1 = _bbox(coords)
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(box == RED), "non-wall component must occupy a wall-free rectangle"
        assert coords.shape[0] == box.size, "each region must be rectangular"
        regions.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    regions.sort(key=lambda reg: reg["bbox"])
    return regions


def _terrain_heights(grid: np.ndarray, region: dict) -> list[int]:
    r0, c0, r1, c1 = region["bbox"]
    heights = []
    for c in range(c0, c1 + 1):
        col = grid[r0 : r1 + 1, c]
        height = 0
        idx = len(col) - 1
        while idx >= 0 and int(col[idx]) == GREY:
            height += 1
            idx -= 1
        assert height > 0, "each region column must contain bottom terrain"
        assert np.all(col[-height:] == GREY), "terrain must be a contiguous bottom segment"
        assert np.all(col[:-height] != GREY), "terrain cannot float above empty cells"
        heights.append(height)
    return heights


def _region_info(grid: np.ndarray, region: dict) -> dict:
    r0, c0, r1, c1 = region["bbox"]
    heights = _terrain_heights(grid, region)
    direction = 1 if heights[0] < heights[-1] else -1
    assert heights[0] != heights[-1], "terrain slope cannot be flat"
    diffs = np.diff(heights)
    if direction == 1:
        assert np.all(diffs >= 0) and np.any(diffs > 0), "right-sliding terrain must be nondecreasing"
    else:
        assert np.all(diffs <= 0) and np.any(diffs < 0), "left-sliding terrain must be nonincreasing"
    movers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if int(grid[r, c]) == YELLOW
    ]
    return {
        "bbox": (r0, c0, r1, c1),
        "heights": heights,
        "direction": direction,
        "peak": max(heights),
        "movers": movers,
    }


def _slide_region(out: np.ndarray, grid: np.ndarray, region: dict, direction: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region["bbox"]
    movers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if int(grid[r, c]) == YELLOW
    ]
    for r, c in movers:
        out[r, c] = BLACK
    movers.sort(key=lambda rc: rc[1], reverse=direction == 1)
    settled = []
    for r, c in movers:
        cc = c
        while True:
            nc = cc + direction
            if nc < c0 or nc > c1 or int(out[r, nc]) != BLACK:
                break
            cc = nc
        out[r, cc] = YELLOW
        settled.append((r, cc))
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each region's yellow pixels horizontally according to its terrain slope, then recolor only the peak-highest and peak-lowest regions.

    Input structure:
    - Black background is split by one-cell-thick red walls into 4-6 rectangular regions.
    - Every region contains a grey bottom terrain profile with one contiguous grey stack per column.
    - Yellow single-pixel movers sit above the terrain inside their own region.

    Rule:
    - Treat red pixels as separators and identify every rectangular non-red region.
    - In each region, compare the leftmost and rightmost terrain heights: if left < right, all yellow pixels slide right; otherwise they slide left.
    - Resolve movers one by one in slide order, stopping before region edges, grey terrain, or already settled movers.
    - After all slides, compute each region's terrain peak height; recolor the movers in the unique global highest-peak region to maroon and those in the unique global lowest-peak region to blue.

    Key insight:
    - The same terrain profile simultaneously defines the local motion direction, creates the collision geometry, and decides which finished movers are recolored by global peak comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    regions = _find_regions(g)

    results = []
    for region in regions:
        info = _region_info(g, region)
        settled = _slide_region(out, g, region, info["direction"])
        results.append({**info, "settled": settled})

    peaks = [res["peak"] for res in results]
    assert peaks.count(max(peaks)) == 1 and peaks.count(min(peaks)) == 1, "peak extremes must be unique"
    max_idx = peaks.index(max(peaks))
    min_idx = peaks.index(min(peaks))
    assert max_idx != min_idx

    for idx, res in enumerate(results):
        color = YELLOW
        if idx == max_idx:
            color = MAROON
        elif idx == min_idx:
            color = BLUE
        for r, c in res["settled"]:
            out[r, c] = color
    return out


def _build_monotone_heights(width: int, peak: int, direction: int, rng: random.Random) -> list[int]:
    low = rng.randint(1, peak - 1)
    middle = [rng.randint(low, peak) for _ in range(max(0, width - 2))]
    heights = [low] + sorted(middle) + [peak]
    if direction == -1:
        heights.reverse()
    return heights


def _terrain_patch(height: int, width: int, heights: list[int]) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    for c, col_height in enumerate(heights):
        patch[height - col_height :, c] = GREY
    return patch


def _slide_local(patch: np.ndarray, movers: list[tuple[int, int]], direction: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = patch.copy()
    for r, c in movers:
        out[r, c] = BLACK
    order = sorted(movers, key=lambda rc: rc[1], reverse=direction == 1)
    settled = []
    for r, c in order:
        cc = c
        while True:
            nc = cc + direction
            if nc < 0 or nc >= patch.shape[1] or int(out[r, nc]) != BLACK:
                break
            cc = nc
        out[r, cc] = YELLOW
        settled.append((r, cc))
    return out, settled


def _local_stop_reason(patch: np.ndarray, r: int, c: int, direction: int) -> str:
    nc = c + direction
    if nc < 0 or nc >= patch.shape[1]:
        return "boundary"
    value = int(patch[r, nc])
    if value == GREY:
        return "terrain"
    if value == YELLOW:
        return "mover"
    return "none"


def _place_movers(patch: np.ndarray, direction: int, rng: random.Random, force_stack: bool, force_boundary: bool) -> np.ndarray | None:
    height, width = patch.shape
    row_open = {r: [c for c in range(width) if int(patch[r, c]) == BLACK] for r in range(height)}
    barrier_rows = [r for r, cols in row_open.items() if 1 < len(cols) < width]
    full_rows = [r for r, cols in row_open.items() if len(cols) == width]
    usable_rows = [r for r, cols in row_open.items() if len(cols) >= 2]
    if not barrier_rows or not full_rows:
        return None

    if force_stack:
        stack_rows = [r for r in barrier_rows if len(row_open[r]) >= 4]
        if not stack_rows:
            return None
        barrier_row = rng.choice(stack_rows)
        barrier_count = 2
    else:
        barrier_row = rng.choice(barrier_rows)
        barrier_count = 1

    min_total = max(2, barrier_count + (1 if force_boundary else 0))
    total = rng.randint(min_total, 4)
    row_counts: dict[int, int] = {barrier_row: barrier_count}
    used_rows = {barrier_row}

    if force_boundary or (total > barrier_count and rng.random() < 0.75):
        boundary_choices = [r for r in full_rows if r not in used_rows]
        if not boundary_choices:
            boundary_choices = full_rows[:]
        if boundary_choices:
            boundary_row = rng.choice(boundary_choices)
            row_counts[boundary_row] = row_counts.get(boundary_row, 0) + 1
            used_rows.add(boundary_row)

    while sum(row_counts.values()) < total:
        extra_choices = [r for r in usable_rows if r not in used_rows]
        if not extra_choices:
            return None
        extra_row = rng.choice(extra_choices)
        row_counts[extra_row] = 1
        used_rows.add(extra_row)

    movers = []
    for row, count in row_counts.items():
        cols = row_open[row]
        if len(cols) < 2 * count:
            return None
        if direction == 1:
            init_cols = cols[:count]
        else:
            init_cols = cols[-count:]
        movers.extend((row, c) for c in init_cols)

    if len(movers) != len(set(movers)):
        return None

    with_movers = patch.copy()
    for r, c in movers:
        with_movers[r, c] = YELLOW

    slid_patch, settled = _slide_local(with_movers, movers, direction)
    if set(movers) == set(settled):
        return None
    if force_boundary and not any(_local_stop_reason(slid_patch, r, c, direction) == "boundary" for r, c in settled):
        return None
    if not any(_local_stop_reason(slid_patch, r, c, direction) == "terrain" for r, c in settled):
        return None
    return with_movers


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    regions = _find_regions(g)
    assert 4 <= len(regions) <= 6

    infos = [_region_info(g, region) for region in regions]
    directions = [info["direction"] for info in infos]
    peaks = [info["peak"] for info in infos]
    assert set(directions) == {-1, 1}, "both slope directions must appear"
    assert peaks.count(max(peaks)) == 1 and peaks.count(min(peaks)) == 1 and max(peaks) > min(peaks)
    assert all(2 <= len(info["movers"]) <= 4 for info in infos)

    out = main(g)
    moved_any = False
    terrain_stop = False
    boundary_stop = False
    stack_seen = False
    max_idx = peaks.index(max(peaks))
    min_idx = peaks.index(min(peaks))

    for idx, info in enumerate(infos):
        r0, c0, r1, c1 = info["bbox"]
        final_cells = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(out[r, c]) in (BLUE, YELLOW, MAROON)
        ]
        final_colors = [int(out[r, c]) for r, c in final_cells]
        assert len(final_cells) == len(info["movers"])
        if set(final_cells) != set(info["movers"]):
            moved_any = True

        expected = YELLOW
        if idx == max_idx:
            expected = MAROON
        elif idx == min_idx:
            expected = BLUE
        assert final_colors and all(color == expected for color in final_colors)

        row_counts = {}
        for r, _ in final_cells:
            row_counts[r] = row_counts.get(r, 0) + 1
        if any(count >= 2 for count in row_counts.values()):
            stack_seen = True

        for r, c in final_cells:
            nc = c + info["direction"]
            if nc < c0 or nc > c1:
                boundary_stop = True
                continue
            neighbor = int(out[r, nc])
            if neighbor == GREY:
                terrain_stop = True
            elif neighbor in (BLUE, YELLOW, MAROON):
                stack_seen = True

    assert moved_any, "at least one mover must actually slide"
    assert terrain_stop, "some mover must stop against terrain"
    assert boundary_stop or stack_seen, "examples should also expose boundary or stacking stops"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_rows, region_cols = rng.choice(REGION_LAYOUTS)
    row_sizes = [rng.randint(*REGION_HEIGHT_RANGE) for _ in range(region_rows)]
    col_sizes = [rng.randint(*REGION_WIDTH_RANGE) for _ in range(region_cols)]
    height = sum(row_sizes) + region_rows + 1
    width = sum(col_sizes) + region_cols + 1
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    wall_rows = [0]
    cur = 0
    for size in row_sizes:
        cur += size + 1
        wall_rows.append(cur)
    wall_cols = [0]
    cur = 0
    for size in col_sizes:
        cur += size + 1
        wall_cols.append(cur)
    for r in wall_rows:
        grid[r, :] = RED
    for c in wall_cols:
        grid[:, c] = RED

    region_specs = []
    for rr in range(region_rows):
        for cc in range(region_cols):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            region_specs.append({"bbox": (r0, c0, r1, c1), "height": r1 - r0 + 1, "width": c1 - c0 + 1})

    directions = [1, -1] + [rng.choice((-1, 1)) for _ in range(len(region_specs) - 2)]
    rng.shuffle(directions)

    for _ in range(100):
        peaks = [rng.randint(2, spec["height"] - 2) for spec in region_specs]
        if peaks.count(max(peaks)) == 1 and peaks.count(min(peaks)) == 1 and max(peaks) > min(peaks):
            break
    else:
        return None

    terrain_patches = []
    for spec, direction, peak in zip(region_specs, directions, peaks):
        heights = _build_monotone_heights(spec["width"], peak, direction, rng)
        patch = _terrain_patch(spec["height"], spec["width"], heights)
        terrain_patches.append({"patch": patch, "direction": direction, "heights": heights, "peak": peak})

    stack_candidates = [
        idx
        for idx, item in enumerate(terrain_patches)
        if item["patch"].shape[1] >= 5
        and any(3 < sum(item["patch"][r] == BLACK) < item["patch"].shape[1] for r in range(item["patch"].shape[0]))
    ]
    if not stack_candidates:
        return None
    stack_idx = rng.choice(stack_candidates)
    boundary_idx = rng.choice([idx for idx in range(len(region_specs)) if idx != stack_idx])

    for idx, (spec, item) in enumerate(zip(region_specs, terrain_patches)):
        patch = _place_movers(
            item["patch"],
            item["direction"],
            rng,
            force_stack=idx == stack_idx,
            force_boundary=idx == boundary_idx,
        )
        if patch is None:
            return None
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    if rng.random() < 0.5:
        grid = np.fliplr(grid).copy()

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
