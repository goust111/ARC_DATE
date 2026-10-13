import random

import numpy as np


GEN_WIDTH = 8
OUT_WIDTH = 16
BG = 0
RETRY_LIMIT = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _foreground_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _minimal_horizontal_period(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    for period in range(1, w + 1):
        if np.array_equal(g[:, period:], g[:, :-period]):
            return period
    return w


def _largest_component_info(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[int]]:
    comps = _foreground_components(np.asarray(grid) != BG)
    assert comps, "expected foreground"
    sizes = [len(comp) for comp in comps]
    max_size = max(sizes)
    winners = [comp for comp in comps if len(comp) == max_size]
    assert len(winners) == 1, "expected a unique largest component"
    largest = winners[0]
    rows = {r for r, _ in largest}
    return largest, rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the horizontal period and extend only the rows covered by the largest foreground component.

    Input structure:
    - The background is black.
    - Non-background cells form 2-4 separate 8-connected foreground components.
    - All foreground rows follow one shared horizontal repeating period, visible inside the 8-column input.
    - Exactly one foreground component is strictly largest, and its covered rows are a strict subset of all rows.

    Rule:
    - Find all 8-connected foreground components and keep the unique largest one M.
    - Let R be the set of rows touched by M.
    - On rows R, detect the shortest horizontal period p from the original input pattern.
    - Build a same-height output of width 16.
    - For rows in R, continue the period to width 16 but keep only the phase positions that belong to M; all other phase positions become background.
    - For rows outside R, output only background.

    Key insight:
    - The largest component does not survive as a cropped object; it acts as a row-selective periodic mask, so both component analysis and period continuation are required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] == GEN_WIDTH

    largest, rows = _largest_component_info(g)
    row_list = sorted(rows)
    period = _minimal_horizontal_period(g[row_list, :])

    keep_phase = np.zeros((g.shape[0], period), dtype=bool)
    phase_color = np.zeros((g.shape[0], period), dtype=int)
    for r, c in largest:
        phase = c % period
        if keep_phase[r, phase]:
            assert phase_color[r, phase] == int(g[r, c]), "period/color inconsistency"
        keep_phase[r, phase] = True
        phase_color[r, phase] = int(g[r, c])

    out = np.zeros((g.shape[0], OUT_WIDTH), dtype=int)
    for r in row_list:
        for c in range(OUT_WIDTH):
            phase = c % period
            if keep_phase[r, phase]:
                out[r, c] = phase_color[r, phase]
    return out


def _repeat_tile(tile: np.ndarray, width: int = GEN_WIDTH) -> np.ndarray:
    cols = np.arange(width) % tile.shape[1]
    return tile[:, cols]


def _component_connected(mask: np.ndarray) -> bool:
    comps = _foreground_components(mask)
    return len(comps) == 1


def _build_random_connected_tile(
    tile_h: int,
    period: int,
    area_lo: int,
    area_hi: int,
    rng: random.Random,
) -> np.ndarray | None:
    tile_area = tile_h * period
    if tile_area < 2:
        return None
    area_lo = max(2, min(area_lo, tile_area - 1))
    area_hi = max(area_lo, min(area_hi, tile_area - 1))
    for _ in range(80):
        target = rng.randint(area_lo, area_hi)
        start = (rng.randrange(tile_h), rng.randrange(period))
        cells = {start}
        frontier = {start}
        while len(cells) < target and frontier:
            base = rng.choice(tuple(frontier))
            nbrs = []
            for nr, nc in _neighbors8(*base):
                if 0 <= nr < tile_h and 0 <= nc < period and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = rng.choice(nbrs)
            cells.add(nxt)
            frontier.add(nxt)
        tile = np.zeros((tile_h, period), dtype=bool)
        for r, c in cells:
            tile[r, c] = True
        if not tile.any():
            continue
        if not _component_connected(tile):
            continue
        used_rows = np.flatnonzero(tile.any(axis=1))
        if used_rows.size < 2:
            continue
        if tile[:, 0].sum() == 0 or tile[:, -1].sum() == 0:
            continue
        if tile.all():
            continue
        return tile
    return None


def _forbidden_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _random_interval_candidates(
    height: int,
    rows_r: set[int],
    relation: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start in range(height):
        for tile_h in range(2, 5):
            end = start + tile_h
            if end > height:
                continue
            row_set = set(range(start, end))
            overlap = row_set & rows_r
            if relation == "disjoint" and not overlap:
                candidates.append((start, tile_h))
    rng.shuffle(candidates)
    return candidates


def _tile_to_full_mask(tile: np.ndarray, start_row: int, height: int) -> np.ndarray:
    full = np.zeros((height, GEN_WIDTH), dtype=bool)
    full[start_row : start_row + tile.shape[0], :] = _repeat_tile(tile)
    return full


def _place_component(
    height: int,
    period: int,
    occupied: np.ndarray,
    rows_r: set[int],
    relation: str,
    max_area: int,
    rng: random.Random,
) -> np.ndarray | None:
    forbidden = _forbidden_mask(occupied)
    intervals = _random_interval_candidates(height, rows_r, relation, rng)
    for start, tile_h in intervals:
        tile_area_cap = tile_h * period - 1
        area_hi = min(tile_area_cap, max(2, max_area))
        area_lo = 2
        for _ in range(60):
            tile = _build_random_connected_tile(tile_h, period, area_lo, area_hi, rng)
            if tile is None:
                continue
            full = _tile_to_full_mask(tile, start, height)
            area = int(full.sum())
            if area >= max_area:
                continue
            if np.any(full & forbidden):
                continue
            if not _component_connected(full):
                continue
            row_set = {r for r in range(height) if full[r].any()}
            overlap = row_set & rows_r
            if relation == "disjoint" and overlap:
                continue
            return full
    return None


def _component_phase_map(grid: np.ndarray, comp: list[tuple[int, int]], period: int) -> np.ndarray:
    phase = np.zeros((grid.shape[0], period), dtype=int)
    for r, c in comp:
        phase[r, c % period] = int(grid[r, c])
    return phase


def _expected_output_from_spec(grid: np.ndarray) -> tuple[np.ndarray, int, set[int], list[tuple[int, int]]]:
    largest, rows_r = _largest_component_info(grid)
    period = _minimal_horizontal_period(grid[sorted(rows_r), :])
    phase = _component_phase_map(grid, largest, period)
    out = np.zeros((grid.shape[0], OUT_WIDTH), dtype=int)
    for r in rows_r:
        for c in range(OUT_WIDTH):
            out[r, c] = phase[r, c % period]
    return out, period, rows_r, largest


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.shape[1] == GEN_WIDTH and g.ndim == 2
    comps = _foreground_components(g != BG)
    assert 2 <= len(comps) <= 4

    sizes = [len(comp) for comp in comps]
    max_size = max(sizes)
    assert sizes.count(max_size) == 1

    out, period, rows_r, largest = _expected_output_from_spec(g)
    assert period in (3, 4)
    assert rows_r and len(rows_r) < g.shape[0]

    all_fg_rows = {r for r in range(g.shape[0]) if np.any(g[r] != BG)}
    assert rows_r < all_fg_rows, "largest rows must be a strict subset of foreground rows"

    selected = g[sorted(rows_r), :]
    assert _minimal_horizontal_period(selected) == period

    largest_phase = np.where(_component_phase_map(g, largest, period) != BG, 1, 0)
    assert _minimal_horizontal_period(largest_phase) == period

    outside_rows = []
    largest_set = set(largest)
    for r in range(g.shape[0]):
        if r not in rows_r and np.any(g[r] != BG):
            outside_rows.append(r)
    assert outside_rows, "need non-selected foreground rows to clear"

    allrow_periodic = np.zeros_like(out)
    for r in range(g.shape[0]):
        for c in range(OUT_WIDTH):
            allrow_periodic[r, c] = int(g[r, c % period])
    assert not np.array_equal(out, allrow_periodic), "row selection must matter"

    assert np.array_equal(main(g), out)


def _generate_main_component(height: int, period: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        tile_h = rng.randint(3, min(5, height - 1))
        start = rng.randint(0, height - tile_h)
        tile = _build_random_connected_tile(tile_h, period, tile_h + 1, tile_h * period - 2, rng)
        if tile is None:
            continue
        repeated = _repeat_tile(tile)
        if not _component_connected(repeated):
            continue
        if _minimal_horizontal_period(repeated.astype(int)) != period:
            continue
        full = np.zeros((height, GEN_WIDTH), dtype=bool)
        full[start : start + tile_h, :] = repeated
        rows_r = {r for r in range(height) if full[r].any()}
        if not rows_r or len(rows_r) == height:
            continue
        return full
    return None


def _paint_component(grid: np.ndarray, mask: np.ndarray, color: int) -> None:
    grid[mask] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(8, 12)
    period = rng.choice((3, 4))
    main_mask = _generate_main_component(height, period, rng)
    if main_mask is None:
        return None

    rows_r = {r for r in range(height) if main_mask[r].any()}
    main_area = int(main_mask.sum())
    target_components = rng.choice((2, 3, 4))

    masks = [main_mask]
    occupied = main_mask.copy()
    while len(masks) < 2:
        outside_mask = _place_component(
            height=height,
            period=period,
            occupied=occupied,
            rows_r=rows_r,
            relation="disjoint",
            max_area=max(3, main_area - 1),
            rng=rng,
        )
        if outside_mask is None:
            return None
        masks.append(outside_mask)
        occupied |= outside_mask

    while len(masks) < target_components:
        extra = _place_component(
            height=height,
            period=period,
            occupied=occupied,
            rows_r=rows_r,
            relation="disjoint",
            max_area=max(3, main_area - 2),
            rng=rng,
        )
        if extra is None:
            return None
        masks.append(extra)
        occupied |= extra

    comps = _foreground_components(occupied)
    if len(comps) != target_components:
        return None
    comp_areas = sorted((len(comp) for comp in comps), reverse=True)
    if len(comp_areas) > 1 and comp_areas[0] <= comp_areas[1]:
        return None
    if comp_areas[0] != main_area:
        return None

    colors = rng.sample(range(1, 10), len(masks))
    grid = np.zeros((height, GEN_WIDTH), dtype=int)
    for idx, mask in enumerate(masks):
        _paint_component(grid, mask, colors[idx])

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
