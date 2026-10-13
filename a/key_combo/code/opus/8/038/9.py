from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
TEAL = 8
BOMB_COLORS = (1, 2, 3, 4, 6, 7, 9)

SPRITE_MASKS = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
)

GEN_RETRIES = 300
START_SEARCH_TRIES = 60
OBSTACLE_SEARCH_TRIES = 220


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    coords = np.argwhere(grid == int(color))
    remaining = {tuple(map(int, rc)) for rc in coords}
    comps: list[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        part = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                nxt = (nr, nc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    part.append(nxt)
        comps.append(np.array(sorted(part), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _sprite_info(grid: np.ndarray) -> dict:
    comps = _components_of_color(grid, TEAL)
    assert comps, "expected teal sprite instances"

    instances: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = local == TEAL
        assert np.array_equal(comp - np.array([r0, c0]), np.argwhere(mask)), "component/mask mismatch"
        instances.append((r0, c0, mask))

    ref_mask = instances[0][2]
    assert ref_mask.size > 0
    for _r0, _c0, mask in instances[1:]:
        assert mask.shape == ref_mask.shape and np.array_equal(mask, ref_mask), "all sprites must share one mask"

    rows = sorted({int(r0) for r0, _c0, _mask in instances})
    cols = sorted({int(c0) for _r0, c0, _mask in instances})
    occupied = {(int(r0), int(c0)) for r0, c0, _mask in instances}
    assert len(rows) >= 2 and len(cols) >= 2, "expected a lattice of sprites"
    assert any(all((r0, c0) in occupied for c0 in cols) for r0 in rows), "expected a complete sprite row"
    assert any(all((r0, c0) in occupied for r0 in rows) for c0 in cols), "expected a complete sprite column"

    missing = [(r0, c0) for r0 in rows for c0 in cols if (r0, c0) not in occupied]
    assert missing, "expected at least one missing lattice slot"
    return {
        "mask": ref_mask,
        "rows": rows,
        "cols": cols,
        "occupied": occupied,
        "missing": missing,
    }


def _bombs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BG, GREY, TEAL):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each color-bomb must be a singleton pixel"
        r, c = map(int, coords[0])
        out.append((color, r, c))
    assert out, "expected at least one color-bomb"
    return out


def _obstacles(grid: np.ndarray) -> list[tuple[int, int, int]]:
    comps = _components_of_color(grid, GREY)
    out = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert r0 == r1, "obstacles must be horizontal bars"
        out.append((r0, c0, c1 - c0 + 1))
    return sorted(out)


def _stop_row(info: dict) -> int:
    return int(min(info["rows"]) - 1)


def _simulate_drop(base: np.ndarray, start: tuple[int, int], stop_row: int) -> tuple[int, int]:
    """
    Simulate one color-bomb until it reaches the row just above the sprite lattice,
    or gets stranded on an obstacle higher up.
    """
    H, W = base.shape
    r, c = map(int, start)
    assert 0 <= r <= stop_row < H
    assert 0 <= c < W and int(base[r, c]) == BG

    for _ in range(H * W * 4):
        if r == stop_row:
            return int(r), int(c)

        if r + 1 <= stop_row and int(base[r + 1, c]) == BG:
            r += 1
            continue

        left = c
        while left - 1 >= 0 and int(base[r, left - 1]) == BG:
            left -= 1
        right = c
        while right + 1 < W and int(base[r, right + 1]) == BG:
            right += 1

        drops = [x for x in range(left, right + 1) if r + 1 <= stop_row and int(base[r + 1, x]) == BG]
        if not drops:
            return int(r), int(c)

        best_dist = min(abs(x - c) for x in drops)
        best_col = min(x for x in drops if abs(x - c) == best_dist)
        c += -1 if best_col < c else 1

    raise AssertionError("drop simulation did not converge")


def _slot_bands(info: dict) -> list[dict]:
    mask = info["mask"]
    width = int(mask.shape[1])
    grouped: dict[int, list[tuple[int, int]]] = {}
    for r0, c0 in info["missing"]:
        grouped.setdefault(int(c0), []).append((int(r0), int(c0)))
    bands = []
    for c0, slots in sorted(grouped.items()):
        bands.append({"c0": c0, "c1": c0 + width - 1, "slots": sorted(slots)})
    return bands


def _band_index_for_col(col: int, bands: list[dict]) -> int | None:
    hits = [i for i, band in enumerate(bands) if int(band["c0"]) <= int(col) <= int(band["c1"])]
    if len(hits) != 1:
        return None
    return int(hits[0])


def _expected_output_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _sprite_info(g)
    bands = _slot_bands(info)
    bombs = _bombs(g)

    base = g.copy()
    for _color, r, c in bombs:
        base[r, c] = BG

    stop_row = _stop_row(info)
    band_colors: dict[int, int] = {}
    for color, r, c in bombs:
        _end_r, end_c = _simulate_drop(base, (r, c), stop_row)
        band_idx = _band_index_for_col(end_c, bands)
        assert band_idx is not None, "each settled bomb must align with one missing lattice column"
        assert band_idx not in band_colors, "two bombs cannot claim the same missing lattice column"
        band_colors[band_idx] = int(color)

    assert len(band_colors) == len(bands), "missing lattice columns must be covered bijectively"

    out = base.copy()
    mask = info["mask"]
    for idx, band in enumerate(bands):
        color = band_colors[idx]
        for r0, c0 in band["slots"]:
            _paint_mask(out, mask, r0, c0, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each missing sprite slot with the color assigned by a falling singleton pixel.

    Input structure:
    - Black background contains identical teal sprites arranged on a row/column lattice, with some lattice slots missing.
    - Grey horizontal bars sit above the lattice, and distinct single colored pixels float above them.
    - The sprite copies share one bbox, so each missing lattice column is a visible horizontal band of columns.

    Rule:
    - Detect the common teal sprite mask and the Cartesian-product lattice of sprite top-left positions; the absent lattice slots are the gaps to fill.
    - Remove the colored singleton pixels from the scene, then simulate each one independently: it falls through black cells, and when blocked it slides along its current row toward the nearest column where it can descend next, preferring the left side on ties.
    - The pixel stops as soon as it reaches the row immediately above the top sprite row, or earlier if an obstacle leaves it with no reachable drop; its final column selects the unique missing lattice column band covering that column.
    - Copy the sprite mask into every missing slot of that selected band using the pixel's own color, erase the singleton pixels, and leave the original teal sprites plus grey bars unchanged.

    Key insight:
    - The floating colored pixels are not decorative: gravity and obstacle routing decide which color is assigned to each missing lattice column before the sprite completion happens.
    """
    return _expected_output_from_input(np.asarray(grid, dtype=int))


def _build_lattice(mask: np.ndarray, rows: list[int], cols: list[int], missing: set[tuple[int, int]]) -> np.ndarray:
    height = rows[-1] + mask.shape[0] + random.randint(1, 3)
    width = cols[-1] + mask.shape[1] + random.randint(1, 3)
    grid = np.zeros((height, width), dtype=int)
    for r0 in rows:
        for c0 in cols:
            if (r0, c0) in missing:
                continue
            _paint_mask(grid, mask, r0, c0, TEAL)
    return grid


def _choose_layout() -> tuple[np.ndarray, list[int], list[int], list[tuple[int, int]]]:
    mask = random.choice(SPRITE_MASKS)
    mh, mw = mask.shape

    n_rows = random.randint(3, 4)
    n_cols = random.randint(3, 5)
    row_step = random.randint(mh + 2, mh + 4)
    col_step = random.randint(mw + 2, mw + 4)
    top = random.randint(8, 11)
    left = random.randint(1, 3)

    rows = [top + i * row_step for i in range(n_rows)]
    cols = [left + j * col_step for j in range(n_cols)]

    anchor_row = random.choice(rows)
    miss_count = random.randint(2, min(4, n_cols - 1))
    miss_cols = sorted(random.sample(cols, miss_count))
    missing = []
    other_rows = [r0 for r0 in rows if r0 != anchor_row]
    for c0 in miss_cols:
        missing.append((random.choice(other_rows), c0))
    return mask, rows, cols, missing


def _try_route_colors(base_grid: np.ndarray, missing: list[tuple[int, int]], mask: np.ndarray, rows: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    stop_row = rows[0] - 1
    bomb_row = random.randint(1, 2)
    if stop_row - bomb_row < 4:
        return None

    width = mask.shape[1]
    missing_sorted = sorted(missing, key=lambda rc: rc[1])
    bands = [{"c0": c0, "c1": c0 + width - 1} for _r0, c0 in missing_sorted]

    start_candidates = [c for c in range(base_grid.shape[1]) if int(base_grid[bomb_row, c]) == BG]
    if len(start_candidates) < len(missing_sorted):
        return None

    no_obstacle = base_grid.copy()
    obstacle_rows = list(range(bomb_row + 2, stop_row))
    if not obstacle_rows:
        return None

    def sample_bar_rows(count: int) -> list[int] | None:
        choices = obstacle_rows[:]
        random.shuffle(choices)
        picked: list[int] = []
        for row in choices:
            if all(abs(row - other) >= 2 for other in picked):
                picked.append(row)
                if len(picked) == count:
                    return sorted(picked)
        return None

    for _ in range(START_SEARCH_TRIES):
        start_cols = sorted(random.sample(start_candidates, len(missing_sorted)))
        no_ids = []
        for c in start_cols:
            _end_r, end_c = _simulate_drop(no_obstacle, (bomb_row, c), stop_row)
            no_ids.append(_band_index_for_col(end_c, bands))

        for _ in range(OBSTACLE_SEARCH_TRIES):
            test = base_grid.copy()
            n_bars = 1 if random.random() < 0.45 else 2
            if len(obstacle_rows) < n_bars:
                continue
            bar_rows = sample_bar_rows(n_bars)
            if bar_rows is None:
                continue
            bars: list[tuple[int, int, int]] = []
            for r0 in bar_rows:
                width_bar = random.randint(3, 5)
                c0 = random.randint(0, test.shape[1] - width_bar)
                test[r0, c0 : c0 + width_bar] = GREY
                bars.append((r0, c0, width_bar))

            final_ids = []
            for c in start_cols:
                _end_r, end_c = _simulate_drop(test, (bomb_row, c), stop_row)
                final_ids.append(_band_index_for_col(end_c, bands))

            if any(idx is None for idx in final_ids):
                continue
            if sorted(int(x) for x in final_ids) != list(range(len(missing_sorted))):
                continue
            if final_ids == no_ids:
                continue

            colors = random.sample(BOMB_COLORS, len(start_cols))
            bombs = [(int(color), bomb_row, int(c)) for color, c in zip(colors, start_cols, strict=True)]
            return test, bombs, bars

    return None


def _independent_validate(inp: np.ndarray, out: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= 30 and w <= 30
    assert out.ndim == 2 and out.shape == g.shape

    info = _sprite_info(g)
    bands = _slot_bands(info)
    bombs = _bombs(g)
    obstacles = _obstacles(g)
    assert 2 <= len(bands) <= 4, "generator should create 2-4 missing lattice columns"
    assert len(bombs) == len(bands), "one colored pixel per missing lattice column"
    assert 1 <= len(obstacles) <= 2, "generator should create 1-2 obstacle bars"

    stop_row = _stop_row(info)
    assert all(r < stop_row for _color, r, _c in bombs)
    assert all(r < stop_row for r, _c0, _w in obstacles)

    base = g.copy()
    for _color, r, c in bombs:
        base[r, c] = BG

    color_to_band: dict[int, int] = {}
    for color, r, c in bombs:
        _end_r, end_c = _simulate_drop(base, (r, c), stop_row)
        band_idx = _band_index_for_col(end_c, bands)
        assert band_idx is not None
        assert band_idx not in color_to_band.values()
        color_to_band[int(color)] = int(band_idx)

    no_obstacle = base.copy()
    for r0, c0, width in obstacles:
        no_obstacle[r0, c0 : c0 + width] = BG

    changed = False
    for color, r, c in bombs:
        _end_r0, end_c0 = _simulate_drop(no_obstacle, (r, c), stop_row)
        no_band = _band_index_for_col(end_c0, bands)
        if no_band != color_to_band[color]:
            changed = True
            break
    assert changed, "at least one bomb must be rerouted by the grey obstacles"

    expected = base.copy()
    mask = info["mask"]
    band_colors = {band_idx: color for color, band_idx in color_to_band.items()}
    for idx, band in enumerate(bands):
        color = band_colors[idx]
        for r0, c0 in band["slots"]:
            _paint_mask(expected, mask, r0, c0, color)

    assert np.array_equal(out, expected), "output must match the routed color completion"


def _generate_attempt() -> np.ndarray | None:
    mask, rows, cols, missing = _choose_layout()
    grid = _build_lattice(mask, rows, cols, set(missing))
    if max(grid.shape) > 30:
        return None

    routed = _try_route_colors(grid, missing, mask, rows)
    if routed is None:
        return None

    with_obstacles, bombs, _bars = routed
    inp = with_obstacles.copy()
    for color, r, c in bombs:
        assert int(inp[r, c]) == BG
        inp[r, c] = int(color)

    out = main(inp)
    _independent_validate(inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _generate_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid routed-color completion puzzle")
