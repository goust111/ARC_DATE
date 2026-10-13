from __future__ import annotations

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3)
PIVOT = 5
NEUTRAL = 7
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _find_pivot(grid: np.ndarray) -> tuple[int, int]:
    pivots = np.argwhere(grid == PIVOT)
    assert len(pivots) == 1, "expected exactly one grey pivot pixel"
    return int(pivots[0, 0]), int(pivots[0, 1])


def _reflect_cell(r: int, c: int, pivot_r: int, pivot_c: int) -> tuple[int, int]:
    return 2 * pivot_r - r, 2 * pivot_c - c


def _reflect_overlay(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    pivot_r, pivot_c = _find_pivot(out)
    for r_raw, c_raw in np.argwhere((out != BG) & (out != PIVOT)):
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect_cell(r, c, pivot_r, pivot_c)
        assert _in_bounds(out.shape, rr, cc), "reflected cell must stay in-bounds"
        out[rr, cc] = int(out[r, c])
    return out


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_touch_colors(grid: np.ndarray, comp: list[tuple[int, int]]) -> frozenset[int]:
    touched: set[int] = set()
    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in MARKER_COLORS:
                touched.add(int(grid[nr, nc]))
    return frozenset(touched)


def _touch_sets_after_reflection(grid: np.ndarray) -> list[frozenset[int]]:
    reflected = _reflect_overlay(grid)
    return [_component_touch_colors(reflected, comp) for comp in _color_components(reflected, NEUTRAL)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect all non-pivot foreground pixels through the grey pivot, then recolor orange components that gain a unique adjacent marker color.

    Input structure:
    - The background is black and there is exactly one grey pivot pixel.
    - Orange 4-connected components are neutral masks, while blue/red/green single pixels are isolated markers.
    - In the input itself, no marker is 4-adjacent to any orange component.

    Rule:
    - Keep the original grid and add a 180-degree reflection of every non-background, non-pivot pixel through the pivot.
    - On that reflected layout, inspect every orange 4-connected component and collect the distinct marker colors touching it by 4-neighborhood.
    - Recolor a component iff it touches exactly one marker color; otherwise leave it orange, then erase all marker pixels and keep the pivot grey.

    Key insight:
    - The markers do not label anything in the raw input; only the reflected copies create the adjacencies that determine which orange components inherit a color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG, "expected black background"
    pivot_r, pivot_c = _find_pivot(g)
    reflected = _reflect_overlay(g)
    marker_mask = np.isin(reflected, MARKER_COLORS)
    out = reflected.copy()

    for comp in _color_components(reflected, NEUTRAL):
        touched = _component_touch_colors(reflected, comp)
        if len(touched) == 1:
            color = next(iter(touched))
            for r, c in comp:
                out[r, c] = color

    out[marker_mask] = BG
    out[pivot_r, pivot_c] = PIVOT
    return out


def _marker_touch_indices(component: set[tuple[int, int]], markers: list[tuple[int, int]]) -> frozenset[int]:
    touched: set[int] = set()
    for idx, marker in enumerate(markers):
        mr, mc = marker
        if any((nr, nc) in component for nr, nc in _neighbors4(mr, mc)):
            touched.add(idx)
    return frozenset(touched)


def _component_blocked_cells(occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(occupied)
    for r, c in list(occupied):
        blocked.update(_neighbors4(r, c))
    return blocked


def _all_target_cells(h: int, w: int, pivot_r: int, pivot_c: int) -> set[tuple[int, int]]:
    row_hi = min(h - 2, 2 * pivot_r - 1)
    col_hi = min(w - 2, 2 * pivot_c - 1)
    return {
        (r, c)
        for r in range(pivot_r + 1, row_hi + 1)
        for c in range(pivot_c + 1, col_hi + 1)
    }


def _shortest_path_to_set(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start in goals:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        for nxt in _neighbors4(*cell):
            if nxt not in allowed or nxt in parent:
                continue
            parent[nxt] = cell
            if nxt in goals:
                path = [nxt]
                cur = cell
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append(nxt)
    return None


def _grow_component(
    seed_cells: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    min_size: int,
    max_size: int,
) -> set[tuple[int, int]] | None:
    component = set(seed_cells)
    if len(component) > max_size:
        return None
    target_size = random.randint(max(min_size, len(component)), max_size)
    while len(component) < target_size:
        frontier = [
            cell
            for cell in allowed
            if cell not in component and any(nei in component for nei in _neighbors4(*cell))
        ]
        if not frontier:
            break
        component.add(random.choice(frontier))
    return component if min_size <= len(component) <= max_size else None


def _build_touching_component(
    subset: frozenset[int],
    markers: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    blocked = _component_blocked_cells(occupied) | set(markers)
    forbidden = set(blocked)
    for idx, marker in enumerate(markers):
        if idx not in subset:
            forbidden.update(_neighbors4(*marker))
    allowed = target_cells - forbidden
    if not allowed:
        return None

    if not subset:
        for _ in range(50):
            seed = random.choice(tuple(allowed))
            component = _grow_component({seed}, allowed, 3, 8)
            if component is not None and _marker_touch_indices(component, markers) == subset:
                return component
        return None

    anchor_options: list[list[tuple[int, int]]] = []
    for idx in subset:
        options = [cell for cell in _neighbors4(*markers[idx]) if cell in allowed]
        if not options:
            return None
        random.shuffle(options)
        anchor_options.append(options)

    for _ in range(120):
        chosen = {random.choice(options) for options in anchor_options}
        connected = {next(iter(chosen))}
        failed = False
        for anchor in list(chosen)[1:]:
            path = _shortest_path_to_set(anchor, connected, allowed)
            if path is None:
                failed = True
                break
            connected.update(path)
            if len(connected) > 8:
                failed = True
                break
        if failed:
            continue
        component = _grow_component(connected, allowed, 3, 8)
        if component is not None and _marker_touch_indices(component, markers) == subset:
            return component
    return None


def _sample_marker_positions(h: int, w: int, pivot_r: int, pivot_c: int, count: int) -> list[tuple[int, int]] | None:
    row_min = pivot_r + 2
    col_min = pivot_c + 2
    row_max = min(h - 3, 2 * pivot_r - 2)
    col_max = min(w - 3, 2 * pivot_c - 2)
    if row_max - row_min + 1 < 5 or col_max - col_min + 1 < 5:
        return None

    for _ in range(120):
        base_r = random.randint(row_min, row_max - 4)
        base_c = random.randint(col_min, col_max - 4)
        cluster = [(base_r + dr, base_c + dc) for dr in range(5) for dc in range(5)]
        positions = random.sample(cluster, count)
        dists = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in combinations(positions, 2)]
        if any(dist < 2 or dist > 5 for dist in dists):
            continue
        return positions
    return None


def _choose_touch_subsets(marker_count: int, component_count: int) -> list[frozenset[int]] | None:
    marker_ids = tuple(range(marker_count))
    singles = [frozenset({idx}) for idx in marker_ids]
    conflicts = [
        frozenset(subset)
        for width in range(2, marker_count + 1)
        for subset in combinations(marker_ids, width)
    ]
    options = [frozenset()] + singles + conflicts

    def weight(subset: frozenset[int]) -> int:
        if not subset:
            return 2
        if len(subset) == 1:
            return 5
        return 4

    weights = [weight(option) for option in options]
    for _ in range(200):
        subsets = random.choices(options, weights=weights, k=component_count)
        if not any(len(subset) == 1 for subset in subsets):
            continue
        if not any(len(subset) != 1 for subset in subsets):
            continue
        used = set().union(*(subset for subset in subsets if subset))
        if used != set(marker_ids):
            continue
        outcome_colors = {
            NEUTRAL if len(subset) != 1 else MARKER_COLORS[next(iter(subset))]
            for subset in subsets
        }
        if len(outcome_colors) < 2:
            continue
        return subsets
    return None


def _reflect_component(
    component: set[tuple[int, int]],
    pivot_r: int,
    pivot_c: int,
) -> set[tuple[int, int]]:
    return {_reflect_cell(r, c, pivot_r, pivot_c) for r, c in component}


def _render_input(
    h: int,
    w: int,
    pivot_r: int,
    pivot_c: int,
    components_on_target: list[set[tuple[int, int]]],
    marker_positions: list[tuple[int, int]],
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[pivot_r, pivot_c] = PIVOT
    for component in components_on_target:
        for r, c in _reflect_component(component, pivot_r, pivot_c):
            grid[r, c] = NEUTRAL
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = color
    return grid


def _transform_variants(grid: np.ndarray) -> list[np.ndarray]:
    variants = [grid.copy(), np.flipud(grid).copy(), np.fliplr(grid).copy(), np.rot90(grid, 2).copy()]
    if grid.shape[0] == grid.shape[1]:
        variants.extend([np.rot90(grid, 1).copy(), np.rot90(grid, -1).copy()])
    return variants


def _validate_instance(
    grid: np.ndarray,
    marker_colors: list[int],
    intended_touch_sets: list[frozenset[int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False

    pivot_r, pivot_c = _find_pivot(grid)
    orange_components = _color_components(grid, NEUTRAL)
    if not (3 <= len(orange_components) <= 5):
        return False
    if any(not (3 <= len(comp) <= 8) for comp in orange_components):
        return False

    marker_coords = np.argwhere(np.isin(grid, marker_colors))
    if len(marker_coords) != len(marker_colors):
        return False
    for color in marker_colors:
        if int(np.sum(grid == color)) != 1:
            return False
    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        if any(
            _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == NEUTRAL
            for nr, nc in _neighbors4(r, c)
        ):
            return False
        if any(
            _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in MARKER_COLORS
            for nr, nc in _neighbors4(r, c)
        ):
            return False
        rr, cc = _reflect_cell(r, c, pivot_r, pivot_c)
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BG:
            return False

    reflected = _reflect_overlay(grid)
    reflected_touches = sorted(tuple(sorted(touches)) for touches in _touch_sets_after_reflection(grid))
    expected_touches = sorted(tuple(sorted(touches)) for touches in intended_touch_sets + intended_touch_sets)
    if reflected_touches != expected_touches:
        return False

    out = main(grid)
    marker_mask = np.isin(reflected, marker_colors)
    if np.any(out[marker_mask] != BG):
        return False
    if int(out[pivot_r, pivot_c]) != PIVOT:
        return False
    if np.array_equal(out, grid) or np.array_equal(out, reflected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    pivot_r = random.randint(8, h - 7)
    pivot_c = random.randint(8, w - 7)
    marker_count = random.choice((2, 3))
    component_count = random.randint(max(3, marker_count + 1), 5)
    marker_colors = random.sample(list(MARKER_COLORS), marker_count)

    marker_positions = _sample_marker_positions(h, w, pivot_r, pivot_c, marker_count)
    if marker_positions is None:
        return None
    subsets = _choose_touch_subsets(marker_count, component_count)
    if subsets is None:
        return None

    target_cells = _all_target_cells(h, w, pivot_r, pivot_c)
    placement_order = sorted(
        range(component_count),
        key=lambda idx: (len(subsets[idx]) == 0, -len(subsets[idx])),
    )
    components_on_target: list[set[tuple[int, int]] | None] = [None] * component_count
    occupied: set[tuple[int, int]] = set()

    for idx in placement_order:
        component = _build_touching_component(subsets[idx], marker_positions, occupied, target_cells)
        if component is None:
            return None
        components_on_target[idx] = component
        occupied.update(component)

    assert all(component is not None for component in components_on_target)
    rendered = _render_input(
        h,
        w,
        pivot_r,
        pivot_c,
        [component for component in components_on_target if component is not None],
        marker_positions,
        marker_colors,
    )

    transformed = random.choice(_transform_variants(rendered))
    intended_touch_sets = [
        frozenset(marker_colors[idx] for idx in subset)
        for subset in subsets
    ]
    return transformed if _validate_instance(transformed, marker_colors, intended_touch_sets) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
