from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WATER = 1
TERRAIN = 5
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _top_reachable_empty(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for c in range(w):
        if int(grid[0, c]) == BG:
            reachable[0, c] = True
            q.append((0, c))
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not reachable[rr, cc] and int(grid[rr, cc]) == BG:
                reachable[rr, cc] = True
                q.append((rr, cc))
    return reachable


def _fill_from_deepest(reachable: np.ndarray, volume: int) -> np.ndarray:
    h, _w = reachable.shape
    filled = np.zeros_like(reachable)
    remaining = int(volume)
    for r in range(h - 1, -1, -1):
        count = int(reachable[r].sum())
        if count == 0:
            continue
        if remaining == 0:
            break
        if remaining < count:
            raise AssertionError("water volume must match a whole number of reachable rows")
        filled[r] = reachable[r]
        remaining -= count
    if remaining != 0:
        raise AssertionError("water volume exceeds reachable capacity")
    return filled


def _row_runs(mask: np.ndarray) -> int:
    runs = 0
    inside = False
    for value in mask.tolist():
        if value and not inside:
            runs += 1
            inside = True
        elif not value:
            inside = False
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the suspended water drops and pour that much water into the deepest sky-reachable empty layers while leaving sealed cavities dry.

    Input structure:
    - Background is black (0); gray (5) terrain forms several open basins plus internal chambers.
    - Some chambers are fully sealed by terrain, while others are connected to the sky only through bent side tunnels.
    - Blue (1) water-drop pixels are scattered in the upper sky.

    Rule:
    - Remove all blue drops and let V be their total count.
    - Candidate cells are exactly the black cells that are 4-connected to the top edge after the drops are cleared.
    - Fill those candidates with V blue cells using one global waterline: fill the deepest reachable row completely, then the next deepest reachable row, and so on. The construction guarantees V matches an integer number of full reachable rows.
    - Terrain stays gray; every non-reachable cavity stays black and dry.

    Key insight:
    - The drop locations do not matter at all; only their total count matters, and the deepest visible empty cells may still be untouchable sealed cavities that must be excluded before setting the global waterline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cleared = g.copy()
    volume = int(np.sum(cleared == WATER))
    cleared[cleared == WATER] = BG

    reachable = _top_reachable_empty(cleared)
    filled = _fill_from_deepest(reachable, volume)

    out = cleared.copy()
    out[filled] = WATER
    return out


def _segment_widths(width: int, n_segments: int) -> list[int] | None:
    widths = [4, 4, 5, 4, 4] if n_segments == 5 else [3] * n_segments
    extra = width - sum(widths)
    if extra < 0:
        return None
    for _ in range(extra):
        widths[random.randrange(n_segments)] += 1
    return widths


def _make_surface(height: int, width: int, sky_rows: int) -> tuple[list[int], list[tuple[int, int, int]]]:
    n_segments = 5
    widths = _segment_widths(width, n_segments)
    if widths is None:
        raise ValueError("grid too narrow for terrain segments")

    ridge_heights = [random.randint(sky_rows + 2, sky_rows + 5) for _ in range(3)]
    valley_base_low = sky_rows + 8
    valley_base_high = height - 8
    if valley_base_low > valley_base_high:
        valley_base_low = valley_base_high
    valley_base = random.randint(valley_base_low, valley_base_high)
    valley_heights = [
        min(height - 7, valley_base + random.randint(0, 2)),
        min(height - 7, valley_base + random.randint(0, 2)),
    ]
    segment_heights = [
        ridge_heights[0],
        max(valley_heights[0], ridge_heights[0] + 4, ridge_heights[1] + 4),
        ridge_heights[1],
        max(valley_heights[1], ridge_heights[1] + 4, ridge_heights[2] + 4),
        ridge_heights[2],
    ]
    segments: list[tuple[int, int, int]] = []
    surface: list[int] = []
    c0 = 0
    for seg_w, seg_h in zip(widths, segment_heights):
        c1 = c0 + seg_w - 1
        segments.append((c0, c1, seg_h))
        surface.extend([seg_h] * seg_w)
        c0 = c1 + 1
    return surface, segments


def _fill_terrain(grid: np.ndarray, surface: list[int]) -> None:
    h, w = grid.shape
    for c in range(w):
        grid[int(surface[c]) : h, c] = TERRAIN


def _carve_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    grid[r0 : r0 + h, c0 : c0 + w] = BG
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _pick_ridge_segments(segments: list[tuple[int, int, int]]) -> list[int]:
    return [i for i, (c0, c1, _h) in enumerate(segments) if i % 2 == 0 and c1 - c0 + 1 >= 5]


def _carve_open_cave(
    grid: np.ndarray, surface: list[int], segments: list[tuple[int, int, int]]
) -> dict[str, object] | None:
    h, _w = grid.shape
    ridge_id = 2
    c0, c1, ridge_top = segments[ridge_id]
    ridge_w = c1 - c0 + 1
    if ridge_w < 5:
        return None

    side, valley_id = random.choice((("left", 1), ("right", 3)))
    vc0, vc1, _valley_top = segments[valley_id]
    cave_w = random.randint(3, min(5, ridge_w - 1))
    cave_h = random.randint(2, 4)
    cave_c0 = random.randint(c0 + 1, c1 - cave_w + 1)
    nearby_valley = max(segments[1][2], segments[3][2])
    cave_r0_min = max(ridge_top + 4, nearby_valley + 2)
    cave_r0_max = h - cave_h - 6
    if cave_r0_min > cave_r0_max:
        return None
    cave_r0 = random.randint(cave_r0_min, cave_r0_max)

    if side == "left":
        exit_col = random.randint(vc0 + 1, vc1)
        start_col = cave_c0
    else:
        exit_col = random.randint(vc0, vc1 - 1)
        start_col = cave_c0 + cave_w - 1

    tunnel_row = random.randint(cave_r0, cave_r0 + cave_h - 1)
    shaft_top = int(surface[exit_col])
    if shaft_top >= tunnel_row:
        return None

    chamber = _carve_rect(grid, cave_r0, cave_c0, cave_h, cave_w)
    step = -1 if exit_col < start_col else 1
    tunnel_cells: set[tuple[int, int]] = set()
    for cc in range(start_col, exit_col + step, step):
        grid[tunnel_row, cc] = BG
        tunnel_cells.add((tunnel_row, cc))
    for rr in range(shaft_top, tunnel_row + 1):
        grid[rr, exit_col] = BG
        tunnel_cells.add((rr, exit_col))

    return {
        "ridge_id": ridge_id,
        "valley_id": valley_id,
        "chamber": chamber,
        "tunnel": tunnel_cells,
        "exit_col": exit_col,
    }


def _carve_sealed_cavity(
    grid: np.ndarray, segments: list[tuple[int, int, int]], forbidden_segment: int | None
) -> set[tuple[int, int]] | None:
    h, _w = grid.shape
    ridge_ids = [i for i in (0, 4) if i != forbidden_segment and i < len(segments)]
    random.shuffle(ridge_ids)
    for ridge_id in ridge_ids:
        c0, c1, ridge_top = segments[ridge_id]
        ridge_w = c1 - c0 + 1
        if ridge_w < 4:
            continue
        cavity_w = random.randint(3, min(5, ridge_w - 1))
        cavity_h = random.randint(2, 4)
        bottom = random.randint(h - 4, h - 2)
        r0 = bottom - cavity_h + 1
        if r0 <= ridge_top + 2:
            continue
        c_start = random.randint(c0 + 1, c1 - cavity_w + 1)
        return _carve_rect(grid, r0, c_start, cavity_h, cavity_w)
    return None


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    reachable = _top_reachable_empty(grid)
    empty = grid == BG
    unreachable = empty & ~reachable
    row_counts = {r: int(reachable[r].sum()) for r in range(grid.shape[0]) if int(reachable[r].sum()) > 0}
    return reachable, unreachable, row_counts


def _filled_mask_for_level(reachable: np.ndarray, level: int) -> np.ndarray:
    h, _w = reachable.shape
    row_ids = np.arange(h)[:, None]
    return reachable & (row_ids >= int(level))


def _choose_level(
    reachable: np.ndarray,
    chamber: set[tuple[int, int]],
    sky_capacity: int,
) -> tuple[int, np.ndarray] | None:
    h, _w = reachable.shape
    rows = [r for r in range(h) if int(reachable[r].sum()) > 0]
    candidates: list[tuple[int, np.ndarray]] = []
    for level in rows:
        filled = _filled_mask_for_level(reachable, level)
        volume = int(filled.sum())
        if volume < 8 or volume > min(sky_capacity, 60):
            continue
        if len({r for r, _c in np.argwhere(filled)}) < 2:
            continue
        if not any(filled[r, c] for r, c in chamber):
            continue
        if max(_row_runs(filled[r]) for r in range(h) if filled[r].any()) < 2:
            continue
        candidates.append((level, filled.copy()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(item[1].sum()))
    window = candidates[: min(4, len(candidates))]
    return random.choice(window)


def _place_droplets(grid: np.ndarray, volume: int, sky_rows: int, filled: np.ndarray) -> np.ndarray | None:
    h, w = grid.shape
    out = grid.copy()
    filled_cols = {int(c) for _r, c in np.argwhere(filled)}

    stripe_side = random.choice(("left", "right"))
    stripe_w = max(4, min(w, (volume + sky_rows - 1) // sky_rows + random.randint(1, 3)))
    if stripe_side == "left":
        stripe_cols = list(range(stripe_w))
    else:
        stripe_cols = list(range(w - stripe_w, w))

    sky_cells = [(r, c) for r in range(sky_rows) for c in stripe_cols if int(out[r, c]) == BG]
    if len(sky_cells) < volume:
        sky_cells = [(r, c) for r in range(sky_rows) for c in range(w) if int(out[r, c]) == BG]
    if len(sky_cells) < volume:
        return None

    chosen = random.sample(sky_cells, volume)
    for r, c in chosen:
        out[r, c] = WATER

    source_cols = {c for _r, c in chosen}
    if filled_cols.issubset(source_cols):
        return None
    return out


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    sky_rows: int,
    chamber: set[tuple[int, int]],
    sealed: set[tuple[int, int]],
) -> bool:
    cleared = inp.copy()
    cleared[cleared == WATER] = BG
    reachable, unreachable, _row_counts = _analyze(cleared)
    filled = expected == WATER

    if not np.array_equal(main(inp), expected):
        return False
    if int(filled.sum()) != int(np.sum(inp == WATER)):
        return False
    if not any(filled[r, c] for r, c in chamber):
        return False
    if any(filled[r, c] for r, c in sealed):
        return False
    if not sealed:
        return False
    if int(max(r for r, _c in sealed)) <= int(max(r for r, _c in chamber)):
        return False
    unreachable_rows = [r for r, _c in np.argwhere(unreachable)]
    reachable_rows = [r for r, _c in np.argwhere(reachable)]
    if not unreachable_rows or not reachable_rows:
        return False
    if max(unreachable_rows) <= max(reachable_rows):
        return False
    if not np.all(np.argwhere(inp == WATER)[:, 0] < sky_rows):
        return False
    if max(_row_runs(filled[r]) for r in range(filled.shape[0]) if filled[r].any()) < 2:
        return False
    for r in range(filled.shape[0] - 1, -1, -1):
        cnt = int(reachable[r].sum())
        if cnt == 0:
            continue
        if filled[r].any():
            if int(filled[r].sum()) != cnt:
                return False
        else:
            upper = filled[:r].sum()
            if upper > 0:
                return False
            break
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    width_min = 21
    w = random.randint(width_min, 24)
    sky_rows = random.randint(4, 6)

    grid = np.full((h, w), BG, dtype=int)
    try:
        surface, segments = _make_surface(h, w, sky_rows)
    except ValueError:
        return None
    _fill_terrain(grid, surface)

    open_cave = _carve_open_cave(grid, surface, segments)
    if open_cave is None:
        return None

    sealed = _carve_sealed_cavity(grid, segments, int(open_cave["ridge_id"]))
    if sealed is None:
        return None

    reachable, unreachable, _row_counts = _analyze(grid)
    if int(reachable.sum()) < 12 or int(unreachable.sum()) < 4:
        return None

    choice = _choose_level(reachable, set(open_cave["chamber"]), sky_rows * w)
    if choice is None:
        return None
    _level, filled = choice

    expected = grid.copy()
    expected[filled] = WATER
    volume = int(filled.sum())

    inp = _place_droplets(grid, volume, sky_rows, filled)
    if inp is None:
        return None

    if not _semantic_ok(inp, expected, sky_rows, set(open_cave["chamber"]), sealed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 密腔水位 input")
