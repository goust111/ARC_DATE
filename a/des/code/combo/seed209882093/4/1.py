import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7

GEN_RETRIES = 400
LEGEND_SIZES = (3, 4)
REGION_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (4, 7)
INNER_W_RANGE = (4, 8)
GRID_MIN = 24
GRID_MAX = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        if len(pts) != 2 * (hh + ww) - 4:
            continue
        if np.any(grid[r0, c0:c1 + 1] != GREY) or np.any(grid[r1, c0:c1 + 1] != GREY):
            continue
        if np.any(grid[r0:r1 + 1, c0] != GREY) or np.any(grid[r0:r1 + 1, c1] != GREY):
            continue
        if hh > 2 and ww > 2 and np.any(grid[r0 + 1:r1, c0 + 1:c1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _seed_color(grid: np.ndarray, frame: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame
    interior = grid[r0 + 1:r1, c0 + 1:c1]
    pts = np.argwhere((interior != BLACK) & (interior != GREY))
    assert len(pts) == 1, "each frame must contain exactly one seed pixel"
    rr, cc = map(int, pts[0])
    return int(interior[rr, cc])


def _legend_candidate(patch: np.ndarray) -> bool:
    n = int(patch.shape[0])
    if patch.shape != (n, n) or n not in LEGEND_SIZES:
        return False
    label = int(patch[0, 0])
    if label in (BLACK, GREY):
        return False
    if np.any(patch == GREY):
        return False

    body = patch.copy()
    body[0, 0] = BLACK
    if not np.all((body == BLACK) | (body == ORANGE)):
        return False
    if not np.any(body == ORANGE):
        return False

    # Make the full legend extent observable so 3x3 and 4x4 blocks are distinguishable.
    if not (np.any(patch[-1, :] == ORANGE) or np.any(patch[:, -1] == ORANGE)):
        return False
    if n == 4 and not (np.any(patch[3, :3] == ORANGE) or np.any(patch[:3, 3] == ORANGE)):
        return False
    return True


def _scan_top_legends(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    legends: dict[int, np.ndarray] = {}
    c = 0
    while c <= w - 3:
        found: tuple[int, np.ndarray] | None = None
        if int(grid[0, c]) not in (BLACK, GREY):
            for n in (4, 3):
                if n > h or c + n > w:
                    continue
                patch = grid[:n, c:c + n]
                if not _legend_candidate(patch):
                    continue
                if c > 0 and np.any(grid[:n, c - 1] != BLACK):
                    continue
                if c + n < w and np.any(grid[:n, c + n] != BLACK):
                    continue
                if n < h and np.any(grid[n, c:c + n] != BLACK):
                    continue
                found = (n, patch.copy())
                break
        if found is None:
            c += 1
            continue
        n, patch = found
        key = int(patch[0, 0])
        assert key not in legends, "duplicate legend key on top edge"
        mask = patch == ORANGE
        mask[0, 0] = False
        legends[key] = mask
        c += n + 1
    return legends


def _scan_left_legends(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    legends: dict[int, np.ndarray] = {}
    r = 0
    while r <= h - 3:
        found: tuple[int, np.ndarray] | None = None
        if int(grid[r, 0]) not in (BLACK, GREY):
            for n in (4, 3):
                if n > w or r + n > h:
                    continue
                patch = grid[r:r + n, :n]
                if not _legend_candidate(patch):
                    continue
                if r > 0 and np.any(grid[r - 1, :n] != BLACK):
                    continue
                if r + n < h and np.any(grid[r + n, :n] != BLACK):
                    continue
                if n < w and np.any(grid[r:r + n, n] != BLACK):
                    continue
                found = (n, patch.copy())
                break
        if found is None:
            r += 1
            continue
        n, patch = found
        key = int(patch[0, 0])
        assert key not in legends, "duplicate legend key on left edge"
        mask = patch == ORANGE
        mask[0, 0] = False
        legends[key] = mask
        r += n + 1
    return legends


def _extract_legends(grid: np.ndarray, seed_colors: set[int]) -> dict[int, np.ndarray]:
    top = _scan_top_legends(grid)
    left = _scan_left_legends(grid)
    for legends in (top, left):
        if legends and seed_colors.issubset(legends.keys()):
            return legends
    raise AssertionError("failed to find a legend set covering all seed colors")


def _tile_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    out = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            out[r, c] = bool(pattern[r % ph, c % pw])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    seed_colors = {_seed_color(g, frame) for frame in frames}
    legends = _extract_legends(g, seed_colors)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        seed = _seed_color(g, (r0, c0, r1, c1))
        pattern = legends[seed]
        fill = _tile_pattern(r1 - r0 - 1, c1 - c0 - 1, pattern)
        patch = out[r0 + 1:r1, c0 + 1:c1]
        patch[:] = BLACK
        patch[fill] = seed
    return out


def _random_pattern(size: int, rng: random.Random) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    min_on = 2 if size == 3 else 3
    max_on = len(cells) - 1
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        density = rng.uniform(0.25, 0.7)
        for r, c in cells:
            if rng.random() < density:
                mask[r, c] = True
        count = int(mask.sum())
        if count < min_on or count > max_on:
            continue
        if not (np.any(mask[-1, :]) or np.any(mask[:, -1])):
            continue
        if size == 4 and not (np.any(mask[3, :3]) or np.any(mask[:3, 3])):
            continue
        return mask
    raise RuntimeError("failed to create a non-degenerate legend pattern")


def _legend_patch(seed: int, pattern: np.ndarray) -> np.ndarray:
    patch = np.full(pattern.shape, BLACK, dtype=int)
    patch[pattern] = ORANGE
    patch[0, 0] = seed
    return patch


def _place_top_legends(grid: np.ndarray, seeds: list[int], patterns: list[np.ndarray], rng: random.Random) -> None:
    size = int(patterns[0].shape[0])
    total_w = len(seeds) * size + (len(seeds) - 1)
    start_c = rng.randint(0, grid.shape[1] - total_w)
    order = list(range(len(seeds)))
    rng.shuffle(order)
    for idx, slot in enumerate(order):
        c0 = start_c + idx * (size + 1)
        grid[:size, c0:c0 + size] = _legend_patch(seeds[slot], patterns[slot])


def _place_left_legends(grid: np.ndarray, seeds: list[int], patterns: list[np.ndarray], rng: random.Random) -> None:
    size = int(patterns[0].shape[0])
    total_h = len(seeds) * size + (len(seeds) - 1)
    start_r = rng.randint(0, grid.shape[0] - total_h)
    order = list(range(len(seeds)))
    rng.shuffle(order)
    for idx, slot in enumerate(order):
        r0 = start_r + idx * (size + 1)
        grid[r0:r0 + size, :size] = _legend_patch(seeds[slot], patterns[slot])


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occ.shape[0], r0 + h + 1)
    cc1 = min(occ.shape[1], c0 + w + 1)
    return not np.any(occ[rr0:rr1, cc0:cc1])


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occ.shape[0], r0 + h + 1)
    cc1 = min(occ.shape[1], c0 + w + 1)
    occ[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, seed: int, rng: random.Random) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[r0:r0 + outer_h, c0] = GREY
    grid[r0:r0 + outer_h, c0 + outer_w - 1] = GREY
    grid[r0, c0:c0 + outer_w] = GREY
    grid[r0 + outer_h - 1, c0:c0 + outer_w] = GREY
    sr = rng.randint(r0 + 1, r0 + inner_h)
    sc = rng.randint(c0 + 1, c0 + inner_w)
    grid[sr, sc] = seed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(*REGION_COUNT_RANGE)
    legend_size = rng.choice(LEGEND_SIZES)
    orientation = rng.choice(("top", "left"))

    legend_span = region_count * legend_size + (region_count - 1)
    if orientation == "top":
        min_h = legend_size + 10
        min_w = legend_span + 2
    else:
        min_h = legend_span + 2
        min_w = legend_size + 10

    h = rng.randint(max(GRID_MIN, min_h), GRID_MAX)
    w = rng.randint(max(GRID_MIN, min_w), GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    seed_pool = [1, 2, 3, 4, 6, 8, 9]
    seeds = rng.sample(seed_pool, region_count)
    patterns = [_random_pattern(legend_size, rng) for _ in range(region_count)]

    if orientation == "top":
        _place_top_legends(grid, seeds, patterns, rng)
        row_min, col_min = legend_size + 2, 1
    else:
        _place_left_legends(grid, seeds, patterns, rng)
        row_min, col_min = 1, legend_size + 2

    occ = np.zeros((h, w), dtype=bool)
    if orientation == "top":
        occ[:legend_size + 2, :] = True
    else:
        occ[:, :legend_size + 2] = True

    dims = [(rng.randint(*INNER_H_RANGE), rng.randint(*INNER_W_RANGE)) for _ in range(region_count)]
    order = sorted(range(region_count), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    placements: list[tuple[int, int]] = [(-1, -1)] * region_count

    for idx in order:
        inner_h, inner_w = dims[idx]
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if row_min > h - outer_h - 1 or col_min > w - outer_w - 1:
            return None
        candidates: list[tuple[int, int]] = []
        for r0 in range(row_min, h - outer_h):
            for c0 in range(col_min, w - outer_w):
                if _can_place(occ, r0, c0, outer_h, outer_w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        placements[idx] = rng.choice(candidates)
        _mark_occ(occ, placements[idx][0], placements[idx][1], outer_h, outer_w)

    for idx, seed in enumerate(seeds):
        r0, c0 = placements[idx]
        inner_h, inner_w = dims[idx]
        _draw_frame(grid, r0, c0, inner_h, inner_w, seed, rng)

    try:
        frames = _find_frames(grid)
        if len(frames) != region_count:
            return None
        found_seeds = {_seed_color(grid, frame) for frame in frames}
        if found_seeds != set(seeds):
            return None
        legends = _extract_legends(grid, found_seeds)
        if set(legends) != found_seeds:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
