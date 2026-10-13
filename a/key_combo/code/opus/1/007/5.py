from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BACKGROUND = 0
OBJECT_COLORS = (1, 2, 3, 4, 5)
BAR_COLORS = (6, 7, 8, 9)
MIN_SIZE = 13
MAX_SIZE = 19
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue: deque[tuple[int, int]] = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _all_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]


def _detect_bars(grid: np.ndarray) -> tuple[dict[str, int], dict[str, int]]:
    hbars: list[dict[str, int]] = []
    vbars: list[dict[str, int]] = []

    for color in _all_colors(grid):
        coords = np.argwhere(grid == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1:
            ordered = np.sort(coords[:, 1])
            if len(ordered) >= 3 and int(ordered[-1]) - int(ordered[0]) + 1 == len(ordered):
                hbars.append(
                    {
                        "color": color,
                        "row": int(rows[0]),
                        "start": int(ordered[0]),
                        "end": int(ordered[-1]),
                    }
                )

        if len(cols) == 1:
            ordered = np.sort(coords[:, 0])
            if len(ordered) >= 3 and int(ordered[-1]) - int(ordered[0]) + 1 == len(ordered):
                vbars.append(
                    {
                        "color": color,
                        "col": int(cols[0]),
                        "start": int(ordered[0]),
                        "end": int(ordered[-1]),
                    }
                )

    assert len(hbars) == 1, "expected exactly one horizontal bar segment"
    assert len(vbars) == 1, "expected exactly one vertical bar segment"
    hbar = hbars[0]
    vbar = vbars[0]
    assert hbar["color"] != vbar["color"], "bar colors must differ"
    assert int(grid[hbar["row"], vbar["col"]]) == BACKGROUND, "bar crossing must be background"
    return hbar, vbar


def _winner_color(grid: np.ndarray, excluded: set[int]) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for color in _all_colors(grid):
        if color in excluded:
            continue
        comps = _components(grid == color)
        if len(comps) == 1:
            candidates.append((len(comps[0]), color))

    assert candidates, "expected at least one single-component color"
    max_area = max(area for area, _ in candidates)
    winners = [color for area, color in candidates if area == max_area]
    assert len(winners) == 1, "expected a unique largest single-component color"
    return winners[0], max_area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the lone horizontal and vertical bar, inject the largest valid body color at their crossing, and repaint that body's cells.

    Input structure:
    - A black background contains one horizontal single-color segment and one vertical single-color segment, each 1 cell thick.
    - Their row/column crossing cell is currently background rather than part of either segment.
    - Several other foreground colors form separated 4-connected monochrome objects; some colors are fragmented into multiple components.

    Rule:
    - Identify the unique horizontal segment color and the unique vertical segment color.
    - For every other foreground color, count its 4-connected monochrome components.
    - Keep only colors with exactly one component, then choose the unique largest such color by area; call it C.
    - Extend the horizontal segment across its full row in the horizontal segment's color.
    - Extend the vertical segment across its full column in the vertical segment's color.
    - Recolor the row/column crossing cell to C.
    - Recolor every original cell of color C to the horizontal segment's color, leaving every other non-bar object unchanged.

    Key insight:
    - The crossing color is not fixed: it is produced by a separate connected-component filter, and that same inferred color also tells you which object body must be repainted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hbar, vbar = _detect_bars(g)
    chosen, _ = _winner_color(g, excluded={hbar["color"], vbar["color"]})

    out = g.copy()
    out[hbar["row"], :] = hbar["color"]
    out[:, vbar["col"]] = vbar["color"]
    out[hbar["row"], vbar["col"]] = chosen
    out[g == chosen] = hbar["color"]
    return out


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = patch
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return out.copy()


def _grow_patch(area: int, require_2d: bool, max_side: int = 7) -> np.ndarray:
    if require_2d and random.random() < 0.35:
        rects = [
            (h, area // h)
            for h in range(2, min(max_side, area) + 1)
            if area % h == 0 and 2 <= area // h <= max_side
        ]
        if rects:
            return _random_transform(np.ones(random.choice(rects), dtype=bool))

    for _ in range(200):
        dims = [
            (h, w)
            for h in range(1 if not require_2d else 2, max_side + 1)
            for w in range(1 if not require_2d else 2, max_side + 1)
            if h * w >= area
        ]
        h, w = random.choice(dims)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            }
            cells.add(random.choice(tuple(frontier)))

        arr = np.array(tuple(cells), dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        if require_2d and (r1 == r0 or c1 == c0):
            continue

        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[arr[:, 0] - r0, arr[:, 1] - c0] = True
        if require_2d and (patch.shape[0] < 2 or patch.shape[1] < 2):
            continue
        return _random_transform(patch)

    raise RuntimeError("failed to grow a valid patch")


def _choose_single_areas(single_count: int) -> list[int]:
    winner_area = random.randint(7, 12)
    pool = list(range(max(4, winner_area - 5), winner_area))
    if len(pool) < single_count - 1:
        pool = list(range(4, winner_area))
    others = random.sample(pool, single_count - 1)
    return [winner_area, *others]


def _sizes_from_total(total: int, count: int, cap: int | None = None) -> list[int]:
    sizes = [1] * count
    sizes[0] = max(1, total - (count - 1))
    remaining = total - sum(sizes)

    while remaining > 0:
        choices = [i for i in range(count) if cap is None or sizes[i] < cap]
        if not choices:
            break
        idx = random.choice(choices)
        sizes[idx] += 1
        remaining -= 1

    random.shuffle(sizes)
    return sizes


def _fragment_sizes(branch: str, winner_area: int) -> list[int]:
    if branch == "largest_piece":
        big = winner_area + random.randint(1, 3)
        extras = [random.randint(1, 2) for _ in range(random.randint(1, 2))]
        return [big, *extras]

    if branch == "total_only":
        for _ in range(50):
            count = random.randint(2, 4)
            extra = random.randint(max(1, count - 1), 4)
            total = winner_area + extra
            largest_hi = min(winner_area - 1, total - (count - 1))
            largest_lo = max(2, (total + count - 1) // count)
            if largest_lo > largest_hi:
                continue
            largest = random.randint(largest_lo, largest_hi)
            sizes = [largest] + [1] * (count - 1)
            remaining = total - sum(sizes)
            while remaining > 0:
                choices = [i for i in range(1, count) if sizes[i] < largest]
                if not choices:
                    break
                idx = random.choice(choices)
                sizes[idx] += 1
                remaining -= 1
            if remaining == 0 and sum(sizes) > winner_area and max(sizes) < winner_area:
                random.shuffle(sizes)
                return sizes
        return _sizes_from_total(max(3, winner_area - 2), random.randint(2, 3), cap=max(2, winner_area - 1))

    count = random.randint(2, 3)
    total = random.randint(3, max(4, winner_area - 2))
    return _sizes_from_total(total, count, cap=max(2, winner_area - 1))


def _choose_segment_left_or_right(cross_col: int, width: int) -> tuple[int, int]:
    side = random.choice(("left", "right"))
    if side == "left":
        length = random.randint(3, min(6, cross_col))
        end = random.randint(length - 1, cross_col - 1)
        return end - length + 1, end
    length = random.randint(3, min(6, width - cross_col - 1))
    start = random.randint(cross_col + 1, width - length)
    return start, start + length - 1


def _choose_segment_up_or_down(cross_row: int, height: int) -> tuple[int, int]:
    side = random.choice(("up", "down"))
    if side == "up":
        length = random.randint(3, min(6, cross_row))
        end = random.randint(length - 1, cross_row - 1)
        return end - length + 1, end
    length = random.randint(3, min(6, height - cross_row - 1))
    start = random.randint(cross_row + 1, height - length)
    return start, start + length - 1


def _place_patch(
    grid: np.ndarray,
    color: int,
    patch: np.ndarray,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    pad: int = 1,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    cells = [(int(r), int(c)) for r, c in np.argwhere(patch)]
    positions: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for dr, dc in cells:
                rr = r0 + dr
                cc = c0 + dc
                if forbidden[rr, cc] or occupied[rr, cc]:
                    ok = False
                    break
                r_lo = max(0, rr - pad)
                r_hi = min(h, rr + pad + 1)
                c_lo = max(0, cc - pad)
                c_hi = min(w, cc + pad + 1)
                if occupied[r_lo:r_hi, c_lo:c_hi].any():
                    ok = False
                    break
            if ok:
                positions.append((r0, c0))

    if not positions:
        return False

    r0, c0 = random.choice(positions)
    for dr, dc in cells:
        rr = r0 + dr
        cc = c0 + dc
        grid[rr, cc] = color
        occupied[rr, cc] = True
    return True


def _manual_transform_from_meta(grid: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    out[meta["h_row"], :] = meta["h_color"]
    out[:, meta["v_col"]] = meta["v_color"]
    out[meta["h_row"], meta["v_col"]] = meta["winner_color"]
    out[grid == meta["winner_color"]] = meta["h_color"]
    return out


def _validate_generated(grid: np.ndarray, meta: dict[str, int]) -> None:
    hbar, vbar = _detect_bars(grid)
    assert hbar["color"] == meta["h_color"]
    assert hbar["row"] == meta["h_row"]
    assert vbar["color"] == meta["v_color"]
    assert vbar["col"] == meta["v_col"]
    assert 3 <= hbar["end"] - hbar["start"] + 1 <= 6
    assert 3 <= vbar["end"] - vbar["start"] + 1 <= 6

    other_colors = [color for color in _all_colors(grid) if color not in {hbar["color"], vbar["color"]}]
    assert 3 <= len(other_colors) <= 5

    singles: list[tuple[int, int]] = []
    fragmented: list[int] = []
    totals: dict[int, int] = {}
    largest_components: dict[int, int] = {}

    for color in other_colors:
        cells = np.argwhere(grid == color)
        assert not np.any(cells[:, 0] == hbar["row"])
        assert not np.any(cells[:, 1] == vbar["col"])
        comps = _components(grid == color)
        areas = [len(comp) for comp in comps]
        totals[color] = sum(areas)
        largest_components[color] = max(areas)
        if len(comps) == 1:
            singles.append((totals[color], color))
        else:
            fragmented.append(color)

    assert len(singles) >= 2
    assert fragmented
    winner_area = max(area for area, _ in singles)
    winners = [color for area, color in singles if area == winner_area]
    assert len(winners) == 1
    assert winners[0] == meta["winner_color"]
    assert any(totals[color] > winner_area or largest_components[color] > winner_area for color in fragmented)

    expected = _manual_transform_from_meta(grid, meta)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert int(actual[hbar["row"], vbar["col"]]) == meta["winner_color"]
    assert int(np.count_nonzero(actual == meta["winner_color"])) == 1


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    rows = random.randint(MIN_SIZE, MAX_SIZE)
    cols = random.randint(MIN_SIZE, MAX_SIZE)
    grid = np.zeros((rows, cols), dtype=int)

    h_color, v_color = random.sample(BAR_COLORS, 2)
    h_row = random.randint(3, rows - 4)
    v_col = random.randint(3, cols - 4)
    h_start, h_end = _choose_segment_left_or_right(v_col, cols)
    v_start, v_end = _choose_segment_up_or_down(h_row, rows)

    grid[h_row, h_start : h_end + 1] = h_color
    grid[v_start : v_end + 1, v_col] = v_color
    occupied = grid != BACKGROUND
    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[h_row, :] = True
    forbidden[:, v_col] = True

    object_count = random.randint(3, 5)
    colors = random.sample(OBJECT_COLORS, object_count)
    single_count = 2 if object_count == 3 else random.randint(2, object_count - 1)
    single_colors = random.sample(colors, single_count)
    multi_colors = [color for color in colors if color not in single_colors]

    single_areas = _choose_single_areas(single_count)
    winner_area = max(single_areas)
    random.shuffle(single_areas)
    winner_color = single_colors[single_areas.index(winner_area)]

    specs: list[tuple[int, list[np.ndarray]]] = []
    for color, area in zip(single_colors, single_areas):
        specs.append((color, [_grow_patch(area, require_2d=True)]))

    branches: list[str] = []
    if len(multi_colors) == 1:
        branches = [random.choice(("largest_piece", "total_only"))]
    elif len(multi_colors) >= 2:
        branches = ["largest_piece", "total_only"]
        while len(branches) < len(multi_colors):
            branches.append(random.choice(("largest_piece", "total_only", "small_noise")))
        random.shuffle(branches)

    for color, branch in zip(multi_colors, branches):
        patches = [_grow_patch(size, require_2d=size >= 4) for size in _fragment_sizes(branch, winner_area)]
        patches.sort(key=lambda patch: -int(patch.sum()))
        specs.append((color, patches))

    specs.sort(key=lambda item: -max(int(patch.sum()) for patch in item[1]))
    for color, patches in specs:
        for patch in patches:
            if not _place_patch(grid, color, patch, occupied, forbidden):
                return None

    meta = {
        "h_color": h_color,
        "v_color": v_color,
        "h_row": h_row,
        "v_col": v_col,
        "winner_color": winner_color,
    }
    try:
        _validate_generated(grid, meta)
    except AssertionError:
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 5")
