import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0])


def _max_solid_rectangles(component: np.ndarray) -> list[tuple[int, int, int, int]]:
    if component.size == 0:
        return []
    r0, c0, r1, c1 = _bbox(component)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=int)
    for rr, cc in component:
        local[int(rr) - r0, int(cc) - c0] = 1
    prefix = np.pad(local.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    best_area = 0
    best = []
    for rr0 in range(h - 1):
        for cc0 in range(w - 1):
            if not local[rr0, cc0]:
                continue
            for rr1 in range(rr0 + 1, h):
                height = rr1 - rr0 + 1
                for cc1 in range(cc0 + 1, w):
                    area = height * (cc1 - cc0 + 1)
                    if area < best_area:
                        continue
                    if _rect_sum(prefix, rr0, cc0, rr1, cc1) != area:
                        continue
                    rect = (r0 + rr0, c0 + cc0, r0 + rr1, c0 + cc1)
                    if area > best_area:
                        best_area = area
                        best = [rect]
                    else:
                        best.append(rect)
    return best


def _minimal_period_rows(arr: np.ndarray) -> int:
    rows = arr.shape[0]
    for period in range(1, rows + 1):
        ok = True
        for r in range(rows):
            if not np.array_equal(arr[r], arr[r % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("row period not found")


def _minimal_period_cols(arr: np.ndarray) -> int:
    cols = arr.shape[1]
    for period in range(1, cols + 1):
        ok = True
        for c in range(cols):
            if not np.array_equal(arr[:, c], arr[:, c % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("col period not found")


def _visible_band_height(grid: np.ndarray) -> int:
    height = 0
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.any((row == BLACK) | (row == GREY)):
            break
        height += 1
    return height


def _infer_periodic_tile(grid: np.ndarray):
    band_h = _visible_band_height(grid)
    assert band_h >= 2, "expected a fully visible top band"
    band = np.asarray(grid[:band_h], dtype=int)
    pr = _minimal_period_rows(band)
    pc = _minimal_period_cols(band)
    tile = band[:pr, :pc].copy()
    tiled = np.tile(tile, (band_h // pr + 1, grid.shape[1] // pc + 1))[:band_h, : grid.shape[1]]
    assert np.array_equal(tiled, band), "top band is not a rectangular 2D period"
    return band_h, pr, pc, tile


def _pattern_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[int(r) % tile.shape[0], int(c) % tile.shape[1]])


def main(grid: np.ndarray) -> np.ndarray:
    """Open each black cavity at its largest solid rectangle(s) and reveal the periodic underlay by absolute coordinates.

    Input structure:
    - The top several full rows are an uncovered band of a colored 2D periodic pattern.
    - All lower non-pattern cells are grey cover, inside which several black 4-connected cavities appear.

    Rule:
    - Infer the whole-grid periodic underlay by extending the exposed top band with its minimal row and column periods.
    - For each black cavity, enumerate every solid all-black rectangle whose height and width are both at least 2, and select every maximum-area rectangle in that cavity.
    - Recolor only those selected rectangle cells with the underlay color at the same absolute grid coordinates; keep all remaining black cells and all grey cover unchanged.

    Key insight:
    - The window choice and the revealed colors come from two independent constraints: first find the largest rectangle(s) per cavity, then sample the hidden periodic pattern in absolute position rather than relative to the cavity.
    """
    g = np.asarray(grid, dtype=int)
    _band_h, _pr, _pc, tile = _infer_periodic_tile(g)
    out = g.copy()
    for component in _components(g == BLACK):
        for r0, c0, r1, c1 in _max_solid_rectangles(component):
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    out[r, c] = _pattern_color(tile, r, c)
    return out


def _make_tile(pr: int, pc: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.random.choice(colors, size=(pr, pc))
        if len(np.unique(tile)) < min(len(colors), 3):
            continue
        if _minimal_period_rows(tile) != pr or _minimal_period_cols(tile) != pc:
            continue
        row_signatures = {tuple(row.tolist()) for row in tile}
        col_signatures = {tuple(tile[:, c].tolist()) for c in range(pc)}
        if len(row_signatures) < 2 or len(col_signatures) < 2:
            continue
        return tile.astype(int)
    return None


def _trim_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_component(cells: set[tuple[int, int]]) -> np.ndarray:
    return np.array(sorted(_trim_cells(cells)), dtype=int)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _adds_bad_2x2(cells: set[tuple[int, int]], protected: set[tuple[int, int]], cand: tuple[int, int]) -> bool:
    trial = set(cells)
    trial.add(cand)
    r, c = cand
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            block = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
            if block <= trial and not block <= protected:
                return True
    return False


def _frontier_candidates(
    cells: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    patch_h: int,
    patch_w: int,
) -> list[tuple[int, int]]:
    out = []
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < patch_h and 0 <= nc < patch_w):
                continue
            cand = (nr, nc)
            if cand in cells:
                continue
            if sum((rr, cc) in cells for rr, cc in _neighbors4(nr, nc)) != 1:
                continue
            if _adds_bad_2x2(cells, protected, cand):
                continue
            out.append(cand)
    return out


def _component_branch(component: np.ndarray) -> str | None:
    rects = _max_solid_rectangles(component)
    if not rects:
        return None
    return "tie" if len(rects) > 1 else "single"


def _single_component(rect_h: int, rect_w: int) -> np.ndarray | None:
    area = rect_h * rect_w
    for _ in range(200):
        patch_h = rect_h + random.randint(2, 5)
        patch_w = rect_w + random.randint(2, 5)
        r0 = random.randint(0, patch_h - rect_h)
        c0 = random.randint(0, patch_w - rect_w)
        rect = {(r0 + dr, c0 + dc) for dr in range(rect_h) for dc in range(rect_w)}
        cells = set(rect)
        target_size = area + random.randint(2, max(4, rect_h + rect_w + 2))
        while len(cells) < target_size:
            frontier = _frontier_candidates(cells, rect, patch_h, patch_w)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) <= area:
            continue
        comp = _cells_to_component(cells)
        rects = _max_solid_rectangles(comp)
        if len(rects) != 1:
            continue
        rr0, cc0, rr1, cc1 = rects[0]
        if (rr1 - rr0 + 1, cc1 - cc0 + 1) != (rect_h, rect_w):
            continue
        if _component_branch(comp) != "single":
            continue
        return comp
    return None


def _tie_component(rect_h: int, rect_w: int) -> np.ndarray | None:
    for _ in range(200):
        cells = set()
        for dr in range(rect_h):
            for dc in range(rect_w):
                cells.add((dr, dc))
                cells.add((rect_h + dr, rect_w - 1 + dc))
        patch_h = 2 * rect_h + random.randint(0, 2)
        patch_w = 2 * rect_w - 1 + random.randint(0, 2)
        shifted = {(r, c) for r, c in cells}
        protected = set(shifted)
        extra_target = random.randint(0, max(2, rect_h + rect_w - 1))
        while len(shifted) < len(protected) + extra_target:
            frontier = _frontier_candidates(shifted, protected, patch_h, patch_w)
            if not frontier:
                break
            shifted.add(random.choice(frontier))
        comp = _cells_to_component(shifted)
        rects = _max_solid_rectangles(comp)
        if len(rects) < 2:
            continue
        areas = {(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects}
        if areas != {rect_h * rect_w}:
            continue
        if _component_branch(comp) != "tie":
            continue
        return comp
    return None


def _thin_component() -> np.ndarray | None:
    target = random.randint(3, 7)
    patch_h = random.randint(3, 7)
    patch_w = random.randint(3, 7)
    for _ in range(200):
        start = (random.randrange(patch_h), random.randrange(patch_w))
        cells = {start}
        while len(cells) < target:
            frontier = _frontier_candidates(cells, set(), patch_h, patch_w)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        comp = _cells_to_component(cells)
        if _max_solid_rectangles(comp):
            continue
        return comp
    return None


def _place_component(
    grid: np.ndarray,
    occupancy: np.ndarray,
    component: np.ndarray,
    min_r: int,
    max_r: int,
) -> tuple[int, int] | None:
    h, w = component[:, 0].max() + 1, component[:, 1].max() + 1
    grid_h, grid_w = grid.shape
    row_lo = min_r
    row_hi = max_r - h + 1
    if row_lo > row_hi or w + 2 > grid_w:
        return None
    rows = list(range(row_lo, row_hi + 1))
    cols = list(range(1, grid_w - w))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if occupancy[max(0, r0 - 1) : min(grid_h, r1 + 2), max(0, c0 - 1) : min(grid_w, c1 + 2)].any():
                continue
            for rr, cc in component:
                grid[r0 + int(rr), c0 + int(cc)] = BLACK
            occupancy[r0 : r1 + 1, c0 : c1 + 1] = True
            return r0, c0
    return None


def _render_pattern(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.tile(tile, (h // tile.shape[0] + 1, w // tile.shape[1] + 1))[:h, :w]


def _window_arrays(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    return [grid[r0 : r1 + 1, c0 : c1 + 1].copy() for r0, c0, r1, c1 in rects]


def _semantic_checks(
    inp: np.ndarray,
    tile: np.ndarray,
    band_h: int,
    shared_dims: tuple[int, int],
    shared_rect_positions: list[tuple[int, int, int, int]],
) -> bool:
    inferred_band_h, pr, pc, inferred_tile = _infer_periodic_tile(inp)
    if inferred_band_h != band_h:
        return False
    if pr != tile.shape[0] or pc != tile.shape[1]:
        return False
    if not np.array_equal(inferred_tile, tile):
        return False

    comps = _components(inp == BLACK)
    branches = {_component_branch(comp) for comp in comps}
    if "single" not in branches or "tie" not in branches:
        return False

    same_shape_rects = []
    for comp in comps:
        for rect in _max_solid_rectangles(comp):
            r0, c0, r1, c1 = rect
            if (r1 - r0 + 1, c1 - c0 + 1) == shared_dims:
                same_shape_rects.append(rect)
    if len(same_shape_rects) < 2:
        return False

    full_pattern = _render_pattern(tile, *inp.shape)
    windows = _window_arrays(full_pattern, shared_rect_positions)
    if len(windows) != 2 or np.array_equal(windows[0], windows[1]):
        return False
    if len(np.unique(windows[0])) < 2 and len(np.unique(windows[1])) < 2:
        return False

    out = main(inp)
    changed = (out != inp)
    if not changed.any():
        return False
    if np.any(inp[:band_h] != out[:band_h]):
        return False
    if np.any(inp[band_h:][(~changed[band_h:]) & (inp[band_h:] != GREY) & (inp[band_h:] != BLACK)]):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    pr = random.choice([2, 3, 4])
    pc = random.choice([3, 4, 5, 6])
    band_cycles = random.choice([2, 3])
    band_h = pr * band_cycles
    width = pc * random.choice([3, 4, 5])
    if width > 30:
        return None
    height = band_h + random.randint(9, 18)
    if height > 30:
        return None

    colors = random.sample(PATTERN_COLORS, random.randint(3, 5))
    tile = _make_tile(pr, pc, colors)
    if tile is None:
        return None
    pattern = _render_pattern(tile, height, width)

    inp = np.full((height, width), GREY, dtype=int)
    inp[:band_h] = pattern[:band_h]
    occupancy = np.zeros_like(inp, dtype=bool)
    occupancy[:band_h] = True

    shared_dims = random.choice([(2, 2), (2, 3), (3, 2), (2, 4)])
    comp_a = _single_component(*shared_dims)
    comp_b = _single_component(*shared_dims)
    tie_dims = random.choice([(2, 2), (2, 3), (3, 2), (2, 4)])
    comp_c = _tie_component(*tie_dims)
    if comp_a is None or comp_b is None or comp_c is None:
        return None

    placed = []
    for comp in [comp_a, comp_b, comp_c]:
        pos = _place_component(inp, occupancy, comp, band_h + 1, height - 1)
        if pos is None:
            return None
        placed.append((comp, pos))

    thin_count = random.randint(0, 2)
    for _ in range(thin_count):
        thin = _thin_component()
        if thin is None:
            continue
        _place_component(inp, occupancy, thin, band_h + 1, height - 1)

    shared_rect_positions = []
    for comp, (base_r, base_c) in placed[:2]:
        rects = _max_solid_rectangles(comp)
        if len(rects) != 1:
            return None
        r0, c0, r1, c1 = rects[0]
        shared_rect_positions.append((base_r + r0, base_c + c0, base_r + r1, base_c + c1))

    if len(shared_rect_positions) != 2:
        return None
    p0 = (shared_rect_positions[0][0] % pr, shared_rect_positions[0][1] % pc)
    p1 = (shared_rect_positions[1][0] % pr, shared_rect_positions[1][1] % pc)
    if p0 == p1:
        return None
    if np.array_equal(
        pattern[shared_rect_positions[0][0] : shared_rect_positions[0][2] + 1, shared_rect_positions[0][1] : shared_rect_positions[0][3] + 1],
        pattern[shared_rect_positions[1][0] : shared_rect_positions[1][2] + 1, shared_rect_positions[1][1] : shared_rect_positions[1][3] + 1],
    ):
        return None

    if not _semantic_checks(inp, tile, band_h, shared_dims, shared_rect_positions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
