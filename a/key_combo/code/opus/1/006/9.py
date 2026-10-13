from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
SEED = 4
WALL = 8
PATTERN_COLORS = (1, 2, 3, 5, 6, 7, 9)
LAYOUT_OPTIONS = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2))
CHAMBER_SIZE_RANGE = (4, 7)
TILE_SIZE_RANGE = (2, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _find_unique_seed(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == SEED)
    assert seeds.shape[0] == 1, "expected exactly one yellow seed"
    return tuple(map(int, seeds[0]))


def _reachable_mask(grid: np.ndarray) -> np.ndarray:
    height, width = map(int, grid.shape)
    seed = _find_unique_seed(grid)
    seen = np.zeros((height, width), dtype=bool)
    q = deque([seed])
    seen[seed] = True

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if seen[nr, nc] or int(grid[nr, nc]) == WALL:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            if np.array_equal(tile, tile[rows % tile_h, cols % tile_w]):
                return tile_h, tile_w
    return h, w


def _infer_unique_tile(
    grid: np.ndarray, domain_mask: np.ndarray, evidence_mask: np.ndarray
) -> tuple[int, int, dict[tuple[int, int], int]]:
    domain_positions = [tuple(map(int, pos)) for pos in np.argwhere(domain_mask)]
    evidence = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(evidence_mask)
    ]
    assert domain_positions, "expected a non-empty domain"
    assert evidence, "expected visible evidence for the wallpaper"

    height, width = map(int, grid.shape)
    best_area: int | None = None
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []

    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            area = tile_h * tile_w
            if best_area is not None and area > best_area:
                continue

            phase_colors: dict[tuple[int, int], int] = {}
            valid = True
            for r, c, color in evidence:
                phase = (r % tile_h, c % tile_w)
                prev = phase_colors.get(phase)
                if prev is None:
                    phase_colors[phase] = color
                elif prev != color:
                    valid = False
                    break
            if not valid:
                continue

            required_phases = {(r % tile_h, c % tile_w) for r, c in domain_positions}
            if not required_phases.issubset(phase_colors):
                continue

            if best_area is None or area < best_area:
                best_area = area
                candidates = []
            candidates.append((tile_h, tile_w, phase_colors))

    assert candidates, "expected at least one valid periodic tile"
    signatures = {
        (tile_h, tile_w, tuple(sorted(phase_colors.items())))
        for tile_h, tile_w, phase_colors in candidates
    }
    assert len(signatures) == 1, "expected a unique minimal periodic tile"
    return candidates[0]


def _apply_tile_fill(
    grid: np.ndarray,
    domain_mask: np.ndarray,
    tile_h: int,
    tile_w: int,
    phase_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for r, c in np.argwhere(domain_mask):
        rr, cc = int(r), int(c)
        fill = int(phase_colors[(rr % tile_h, cc % tile_w)])
        if int(grid[rr, cc]) in (BLACK, SEED):
            out[rr, cc] = fill
        else:
            assert int(grid[rr, cc]) == fill, "evidence pixels must agree with the inferred tile"
    return out


def _visible_evidence_mask(grid: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    return region_mask & (grid != BLACK) & (grid != WALL) & (grid != SEED)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the seed-reachable wallpaper region by inferring its minimal repeating tile and filling only that corridor component.

    Input structure:
    - Cyan wall cells partition the grid into several 4-connected corridor regions made of black gaps and scattered colored wallpaper pixels.
    - A single yellow seed lies in exactly one corridor region.
    - Each corridor region follows its own periodic wallpaper, but only some cells of each wallpaper are visible.

    Rule:
    - Flood-fill 4-connectedly from the yellow seed through every non-cyan cell to mark the reachable corridor region.
    - Use only the reachable region's non-black, non-cyan, non-yellow pixels as evidence for the wallpaper.
    - Infer the unique minimal periodic tile aligned to global coordinates, then fill the reachable region's black holes with the tile colors and replace the seed by its tile color.
    - Leave cyan walls and every unreachable region unchanged.

    Key insight:
    - Reachability is not just the fill target: it also defines which visible pixels are allowed to participate in periodic inference, preventing disconnected distractor regions from corrupting the tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reachable = _reachable_mask(g)
    evidence_mask = _visible_evidence_mask(g, reachable)
    tile_h, tile_w, phase_colors = _infer_unique_tile(g, reachable, evidence_mask)
    return _apply_tile_fill(g, reachable, tile_h, tile_w, phase_colors)


def _random_tile(tile_h: int, tile_w: int, forbid: np.ndarray | None = None) -> np.ndarray:
    max_colors = min(4, len(PATTERN_COLORS), tile_h * tile_w)
    for _ in range(300):
        n_colors = random.randint(2, max_colors)
        palette = random.sample(PATTERN_COLORS, n_colors)
        tile = np.array(
            [[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        if len(set(map(int, tile.flat))) < 2:
            continue
        if forbid is not None and np.array_equal(tile, forbid):
            continue
        if _minimal_period(tile) != (tile_h, tile_w):
            continue
        return tile
    raise RuntimeError("failed to sample a full-period tile")


def _sample_partition() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    n_rows, n_cols = random.choice(LAYOUT_OPTIONS)
    row_sizes = [random.randint(*CHAMBER_SIZE_RANGE) for _ in range(n_rows)]
    col_sizes = [random.randint(*CHAMBER_SIZE_RANGE) for _ in range(n_cols)]

    height = sum(row_sizes) + n_rows + 1
    width = sum(col_sizes) + n_cols + 1
    grid = np.full((height, width), WALL, dtype=int)

    chambers: list[tuple[int, int, int, int]] = []
    r0 = 1
    for chamber_h in row_sizes:
        c0 = 1
        for chamber_w in col_sizes:
            r1 = r0 + chamber_h
            c1 = c0 + chamber_w
            grid[r0:r1, c0:c1] = BLACK
            chambers.append((r0, r1, c0, c1))
            c0 = c1 + 1
        r0 = r1 + 1
    return grid, chambers


def _phase_buckets(
    bbox: tuple[int, int, int, int], tile_h: int, tile_w: int
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    r0, r1, c0, c1 = bbox
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(r0, r1):
        for c in range(c0, c1):
            buckets.setdefault((r % tile_h, c % tile_w), []).append((r, c))
    return buckets


def _sample_visible_cells(
    bbox: tuple[int, int, int, int], tile_h: int, tile_w: int
) -> set[tuple[int, int]]:
    buckets = _phase_buckets(bbox, tile_h, tile_w)
    visible = {random.choice(cells) for cells in buckets.values()}

    r0, r1, c0, c1 = bbox
    all_cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    remaining = [cell for cell in all_cells if cell not in visible]
    max_extra = min(len(remaining) - 1, max(0, len(all_cells) // 4))
    if max_extra > 0:
        extra_count = random.randint(0, max_extra)
        visible.update(random.sample(remaining, extra_count))
    return visible


def _sample_seed_cell(
    bbox: tuple[int, int, int, int], visible: set[tuple[int, int]]
) -> tuple[int, int]:
    r0, r1, c0, c1 = bbox
    hidden = [(r, c) for r in range(r0, r1) for c in range(c0, c1) if (r, c) not in visible]
    assert hidden, "need at least one hidden cell for the seed"

    boundary_hidden = [
        (r, c)
        for r, c in hidden
        if r in (r0, r1 - 1) or c in (c0, c1 - 1)
    ]
    return random.choice(boundary_hidden or hidden)


def _paint_chamber(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    visible: set[tuple[int, int]],
) -> None:
    tile_h, tile_w = map(int, tile.shape)
    for r, c in visible:
        grid[r, c] = int(tile[r % tile_h, c % tile_w])


def _expected_output(
    grid: np.ndarray, bbox: tuple[int, int, int, int], tile: np.ndarray
) -> np.ndarray:
    out = grid.copy()
    r0, r1, c0, c1 = bbox
    tile_h, tile_w = map(int, tile.shape)
    for r in range(r0, r1):
        for c in range(c0, c1):
            out[r, c] = int(tile[r % tile_h, c % tile_w])
    return out


def _count_regions(grid: np.ndarray) -> int:
    open_mask = grid != WALL
    seen = np.zeros_like(open_mask, dtype=bool)
    count = 0
    for start in np.argwhere(open_mask):
        r, c = map(int, start)
        if seen[r, c]:
            continue
        count += 1
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc, *grid.shape):
                if open_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _try_infer(
    grid: np.ndarray, domain_mask: np.ndarray, evidence_mask: np.ndarray
) -> tuple[int, int, dict[tuple[int, int], int]] | None:
    try:
        return _infer_unique_tile(grid, domain_mask, evidence_mask)
    except AssertionError:
        return None


def _build_instance():
    grid, chambers = _sample_partition()
    min_h = min(r1 - r0 for r0, r1, _c0, _c1 in chambers)
    min_w = min(c1 - c0 for _r0, _r1, c0, c1 in chambers)
    min_area = min((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in chambers)
    size_choices = [
        (tile_h, tile_w)
        for tile_h in range(TILE_SIZE_RANGE[0], min(TILE_SIZE_RANGE[1], min_h) + 1)
        for tile_w in range(TILE_SIZE_RANGE[0], min(TILE_SIZE_RANGE[1], min_w) + 1)
        if tile_h * tile_w < min_area
    ]
    assert size_choices, "expected at least one tile size smaller than every chamber"
    tile_h, tile_w = random.choice(size_choices)
    base_tile = _random_tile(tile_h, tile_w)

    seed_index = random.randrange(len(chambers))
    chamber_tiles = []
    visible_sets = []
    for idx, chamber in enumerate(chambers):
        tile = base_tile if idx == seed_index else _random_tile(tile_h, tile_w, forbid=base_tile)
        visible = _sample_visible_cells(chamber, tile_h, tile_w)
        chamber_tiles.append(tile)
        visible_sets.append(visible)
        _paint_chamber(grid, chamber, tile, visible)

    seed_cell = _sample_seed_cell(chambers[seed_index], visible_sets[seed_index])
    grid[seed_cell] = SEED
    expected = _expected_output(grid, chambers[seed_index], base_tile)

    meta = {
        "chambers": chambers,
        "seed_index": seed_index,
        "base_tile": base_tile,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "expected": expected,
    }
    return grid, meta


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    expected = meta["expected"]
    chambers = meta["chambers"]
    seed_index = int(meta["seed_index"])
    base_tile = np.asarray(meta["base_tile"], dtype=int)
    tile_h, tile_w = map(int, base_tile.shape)

    if _count_regions(grid) < 2:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False

    reachable = _reachable_mask(grid)
    if not np.array_equal(out[~reachable], grid[~reachable]):
        return False
    if not np.any((~reachable) & (grid == BLACK)):
        return False
    if not np.any((~reachable) & (grid != BLACK) & (grid != WALL)):
        return False

    seed_bbox = chambers[seed_index]
    r0, r1, c0, c1 = seed_bbox
    seed_region = np.zeros_like(grid, dtype=bool)
    seed_region[r0:r1, c0:c1] = True
    if not np.array_equal(reachable, seed_region):
        return False

    reachable_evidence = _visible_evidence_mask(grid, reachable)
    inferred = _try_infer(grid, reachable, reachable_evidence)
    if inferred is None:
        return False
    inf_h, inf_w, inf_phase_colors = inferred
    if (inf_h, inf_w) != (tile_h, tile_w):
        return False
    for rr in range(tile_h):
        for cc in range(tile_w):
            if inf_phase_colors.get((rr, cc)) != int(base_tile[rr, cc]):
                return False

    global_evidence = (grid != BLACK) & (grid != WALL) & (grid != SEED)
    global_on_seed = _try_infer(grid, reachable, global_evidence)
    if global_on_seed is not None:
        wrong = _apply_tile_fill(grid, reachable, *global_on_seed)
        if np.array_equal(wrong, expected):
            return False

    global_on_all = _try_infer(grid, grid != WALL, global_evidence)
    if global_on_all is not None:
        wrong = _apply_tile_fill(grid, grid != WALL, *global_on_all)
        if np.array_equal(wrong, expected):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, meta = _build_instance()
        if _semantic_checks(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid flood-gated periodic wallpaper puzzle")
