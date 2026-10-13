from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 26
REGION_COUNT_MIN = 2
REGION_COUNT_MAX = 4
RECT_H_RANGE = (5, 9)
RECT_W_RANGE = (5, 9)
REGION_MARGIN = 2
REGION_PLACE_TRIES = 200
MARKER_PLACE_TRIES = 200
GEN_RETRIES = 400


def _prefix(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int, copy=False)
    return np.pad(arr, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _border_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = _rect_sum(ps, r0, c0, r1, c1)
    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return total - inner


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _extract_entities(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    entities: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, GREY)):
        comps = _components(grid == color)
        seed: tuple[int, int] | None = None
        marker: list[tuple[int, int]] | None = None
        for comp in comps:
            if len(comp) == 1 and not _touches_edge(comp, (h, w)):
                assert seed is None, f"color {color} has multiple seed candidates"
                seed = comp[0]
                continue
            if len(comp) == 2 and _touches_edge(comp, (h, w)):
                rows = {r for r, _c in comp}
                cols = {c for _r, c in comp}
                if len(rows) == 1 or len(cols) == 1:
                    assert marker is None, f"color {color} has multiple marker candidates"
                    marker = comp
                    continue
            raise AssertionError(f"color {color} has an invalid component")
        assert seed is not None and marker is not None, f"color {color} must have one seed and one marker"
        rows = {r for r, _c in marker}
        axis = "horizontal" if len(rows) == 1 else "vertical"
        entities.append({"color": color, "seed": seed, "axis": axis})
    assert entities, "expected at least one seed/marker pair"
    return entities


def _find_gap_on_border(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> tuple[int, int]:
    gap: tuple[int, int] | None = None
    for c in range(left + 1, right):
        if grid[top, c] == BG:
            assert gap is None
            gap = (top, c)
        if grid[bottom, c] == BG:
            assert gap is None
            gap = (bottom, c)
    for r in range(top + 1, bottom):
        if grid[r, left] == BG:
            assert gap is None
            gap = (r, left)
        if grid[r, right] == BG:
            assert gap is None
            gap = (r, right)
    assert gap is not None, "rectangle border must contain one gap"
    return gap


def _find_box_for_seed(
    grid: np.ndarray,
    seed: tuple[int, int],
    color: int,
    grey_ps: np.ndarray,
    bg_ps: np.ndarray,
    other_ps: np.ndarray,
) -> tuple[int, int, int, int, tuple[int, int]]:
    h, w = grid.shape
    sr, sc = seed
    best: tuple[int, int, int, int, tuple[int, int]] | None = None
    best_area: int | None = None

    for top in range(sr - 1):
        for bottom in range(sr + 1, h):
            if grid[top, sc] not in (BG, GREY) or grid[bottom, sc] not in (BG, GREY):
                continue
            for left in range(sc - 1):
                if grid[top, left] != GREY or grid[bottom, left] != GREY:
                    continue
                for right in range(sc + 1, w):
                    if grid[top, right] != GREY or grid[bottom, right] != GREY:
                        continue
                    if top + 2 > bottom or left + 2 > right:
                        continue

                    border_other = _border_sum(other_ps, top, left, bottom, right)
                    if border_other != 0:
                        continue
                    border_bg = _border_sum(bg_ps, top, left, bottom, right)
                    if border_bg != 1:
                        continue

                    inner_other = _rect_sum(other_ps, top + 1, left + 1, bottom - 1, right - 1)
                    inner_grey = _rect_sum(grey_ps, top + 1, left + 1, bottom - 1, right - 1)
                    if inner_other != 1 or inner_grey != 0:
                        continue
                    if grid[sr, sc] != color:
                        continue

                    gap = _find_gap_on_border(grid, top, left, bottom, right)
                    area = (bottom - top + 1) * (right - left + 1)
                    if best is None or area < best_area:
                        best = (top, left, bottom, right, gap)
                        best_area = area
                    elif area == best_area:
                        raise AssertionError(f"seed {seed} has ambiguous rectangles")

    assert best is not None, f"no valid rectangle found for seed {seed}"
    return best


def _window_direction(
    seed: tuple[int, int],
    axis: str,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    sr, sc = seed
    if axis == "horizontal":
        left_dist = sc - left
        right_dist = right - sc
        assert left_dist != right_dist, "horizontal marker must imply a unique nearer side"
        if left_dist < right_dist:
            return (0, -1), (sr, left)
        return (0, 1), (sr, right)

    top_dist = sr - top
    bottom_dist = bottom - sr
    assert top_dist != bottom_dist, "vertical marker must imply a unique nearer side"
    if top_dist < bottom_dist:
        return (-1, 0), (top, sc)
    return (1, 0), (bottom, sc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_ps = _prefix(g == GREY)
    bg_ps = _prefix(g == BG)
    other_ps = _prefix((g != BG) & (g != GREY))

    out = g.copy()
    for entity in _extract_entities(g):
        color = int(entity["color"])
        seed = tuple(entity["seed"])
        top, left, bottom, right, gap = _find_box_for_seed(g, seed, color, grey_ps, bg_ps, other_ps)
        direction, window = _window_direction(seed, entity["axis"], top, left, bottom, right)

        out[gap] = GREY
        out[window] = BG

        dr, dc = direction
        rr, cc = seed
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            if out[rr, cc] != BG:
                break
            out[rr, cc] = color

    return out


def _direction_from_axis(axis: str, seed: tuple[int, int], top: int, left: int, bottom: int, right: int):
    return _window_direction(seed, axis, top, left, bottom, right)


def _path_cells(seed: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    rr, cc = seed
    cells: list[tuple[int, int]] = []
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def _expanded_clear(mask: np.ndarray, top: int, left: int, bottom: int, right: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(mask.shape[0] - 1, bottom + 1)
    c1 = min(mask.shape[1] - 1, right + 1)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _border_cells(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        cells.append((top, c))
        cells.append((bottom, c))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        cells.append((r, right))
    return cells


def _draw_region(
    grid: np.ndarray,
    color: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    seed: tuple[int, int],
    gap: tuple[int, int],
) -> None:
    for r, c in _border_cells(top, left, bottom, right):
        grid[r, c] = GREY
    grid[gap] = BG
    grid[seed] = color


def _choose_gap(
    rng: random.Random,
    top: int,
    left: int,
    bottom: int,
    right: int,
    window: tuple[int, int],
) -> tuple[int, int]:
    side_cells = {
        "top": [(top, c) for c in range(left + 1, right)],
        "bottom": [(bottom, c) for c in range(left + 1, right)],
        "left": [(r, left) for r in range(top + 1, bottom)],
        "right": [(r, right) for r in range(top + 1, bottom)],
    }
    side_names = list(side_cells)
    rng.shuffle(side_names)
    options: list[tuple[int, int]] = []
    for side in side_names:
        cells = [cell for cell in side_cells[side] if cell != window]
        if cells:
            if not options or len(options) == 1:
                options.extend(cells)
    assert options
    return rng.choice(options)


def _place_regions(rng: random.Random, grid: np.ndarray, count: int) -> list[dict] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    colors = rng.sample(SEED_COLORS, count)

    for color in colors:
        placed = False
        for _ in range(REGION_PLACE_TRIES):
            rect_h = rng.randint(*RECT_H_RANGE)
            rect_w = rng.randint(*RECT_W_RANGE)
            if rect_h >= h - 2 * REGION_MARGIN or rect_w >= w - 2 * REGION_MARGIN:
                continue

            top = rng.randint(REGION_MARGIN, h - rect_h - REGION_MARGIN)
            left = rng.randint(REGION_MARGIN, w - rect_w - REGION_MARGIN)
            bottom = top + rect_h - 1
            right = left + rect_w - 1
            if not _expanded_clear(occupied | blocked, top, left, bottom, right):
                continue

            axis = rng.choice(("horizontal", "vertical"))
            interior_rows = list(range(top + 1, bottom))
            interior_cols = list(range(left + 1, right))
            rng.shuffle(interior_rows)
            rng.shuffle(interior_cols)

            seed: tuple[int, int] | None = None
            for sr in interior_rows:
                for sc in interior_cols:
                    if axis == "horizontal" and sc - left == right - sc:
                        continue
                    if axis == "vertical" and sr - top == bottom - sr:
                        continue
                    seed = (sr, sc)
                    break
                if seed is not None:
                    break
            if seed is None:
                continue

            direction, window = _direction_from_axis(axis, seed, top, left, bottom, right)
            gap = _choose_gap(rng, top, left, bottom, right, window)
            path = _path_cells(seed, direction, grid.shape)
            if any(occupied[r, c] or blocked[r, c] for r, c in path):
                continue

            _draw_region(grid, color, top, left, bottom, right, seed, gap)
            for r, c in _border_cells(top, left, bottom, right):
                occupied[r, c] = True
            occupied[seed] = True
            for r, c in path:
                blocked[r, c] = True

            regions.append({"color": color, "axis": axis})
            placed = True
            break

        if not placed:
            return None

    for region in regions:
        color = int(region["color"])
        axis = str(region["axis"])
        ok = False
        for _ in range(MARKER_PLACE_TRIES):
            if axis == "horizontal":
                row = rng.choice([0, h - 1])
                c0 = rng.randint(0, w - 2)
                cells = [(row, c0), (row, c0 + 1)]
            else:
                col = rng.choice([0, w - 1])
                r0 = rng.randint(0, h - 2)
                cells = [(r0, col), (r0 + 1, col)]
            if any(occupied[r, c] or blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
                occupied[r, c] = True
            ok = True
            break
        if not ok:
            return None

    return regions


def _valid_instance(grid: np.ndarray, expected_count: int) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    colors = [int(v) for v in np.unique(grid) if v not in (BG, GREY)]
    if len(colors) != expected_count:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        count = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
        grid = np.full((h, w), BG, dtype=int)
        regions = _place_regions(rng, grid, count)
        if regions is None:
            continue
        if _valid_instance(grid, count):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
