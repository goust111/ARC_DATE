from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7)
AREAS = (1, 2, 3)
GRID_SIDE_RANGE = (13, 16)
RETRY_LIMIT = 600

# Exclude pure ascending/descending area order so distance ordering is necessary.
ORDER_PATTERNS = (
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
)

PATCHES = {
    1: (((0, 0),),),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    )


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            centroid_r = _round_half_up(coords[:, 0].mean())
            centroid_c = _round_half_up(coords[:, 1].mean())
            center_r, center_c = h // 2, w // 2
            distance = abs(int(centroid_r) - int(center_r)) + abs(int(centroid_c) - int(center_c))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "centroid": (int(centroid_r), int(centroid_c)),
                    "distance": int(distance),
                }
            )

    return components


def _build_output(components: list[dict]) -> np.ndarray:
    ordered = sorted(components, key=lambda comp: int(comp["distance"]))
    assert ordered, "expected at least one foreground component"

    inner_area = int(ordered[0]["area"])
    cur = np.full((inner_area, inner_area), int(ordered[0]["color"]), dtype=int)
    for comp in ordered[1:]:
        thickness = int(comp["area"])
        color = int(comp["color"])
        new_side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_side, new_side), color, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert scattered irregular components into concentric squares whose layer sizes are encoded by component areas.

    Input structure:
    - A black background contains several non-black monochromatic 4-connected components.
    - Each component has a distinct color and acts only as an area-bearing token.
    - The grid center is the integer midpoint `(H//2, W//2)`; component order is encoded by rounded-centroid distance to that center.

    Rule:
    - Find every non-black 4-connected component.
    - For each component, compute its area and the Manhattan distance from the grid center to the component centroid after rounding each centroid coordinate with ordinary half-up rounding.
    - Sort components by that distance from near to far.
    - The nearest component becomes a solid square of side length equal to its area.
    - Each later component wraps the current square with a square frame of thickness equal to its area, using that component's color.

    Key insight:
    - Component shape is irrelevant noise: area determines square thickness, while spatial position relative to the input center determines the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert components, "expected foreground components"
    distances = [int(comp["distance"]) for comp in components]
    assert len(distances) == len(set(distances)), "component center distances must be unique"
    return _build_output(components)


def _all_placements(height: int, width: int, area: int) -> list[dict]:
    placements: list[dict] = []
    for patch in PATCHES[int(area)]:
        max_r = max(r for r, _ in patch)
        max_c = max(c for _, c in patch)
        for r0 in range(height - max_r):
            for c0 in range(width - max_c):
                coords = np.asarray([(r0 + r, c0 + c) for r, c in patch], dtype=int)
                centroid_r = _round_half_up(coords[:, 0].mean())
                centroid_c = _round_half_up(coords[:, 1].mean())
                center_r, center_c = height // 2, width // 2
                distance = abs(int(centroid_r) - int(center_r)) + abs(int(centroid_c) - int(center_c))
                placements.append(
                    {
                        "area": int(area),
                        "coords": coords,
                        "bbox_left": int(coords[:, 1].min()),
                        "bbox_top": int(coords[:, 0].min()),
                        "centroid": (int(centroid_r), int(centroid_c)),
                        "distance": int(distance),
                    }
                )
    return placements


def _touches_or_overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    aset = {(int(r), int(c)) for r, c in a}
    bset = {(int(r), int(c)) for r, c in b}
    for r, c in aset:
        if (r, c) in bset:
            return True
        for nr, nc in _neighbors8(r, c):
            if (int(nr), int(nc)) in bset:
                return True
    return False


def _render_input(height: int, width: int, chosen: dict[int, dict], color_by_area: dict[int, int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for area in AREAS:
        color = int(color_by_area[area])
        coords = chosen[area]["coords"]
        grid[coords[:, 0], coords[:, 1]] = color
    return grid


def _independent_expected(color_by_area: dict[int, int], ordered_areas: tuple[int, ...]) -> np.ndarray:
    side = int(ordered_areas[0])
    cur = np.full((side, side), int(color_by_area[ordered_areas[0]]), dtype=int)
    for area in ordered_areas[1:]:
        thickness = int(area)
        nxt = np.full((cur.shape[0] + 2 * thickness, cur.shape[1] + 2 * thickness), int(color_by_area[area]), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _validate_instance(grid: np.ndarray, target_order: tuple[int, ...], color_by_area: dict[int, int]) -> bool:
    components = _find_components(grid)
    if len(components) != 3:
        return False
    if sorted(int(comp["area"]) for comp in components) != [1, 2, 3]:
        return False
    if len({int(comp["color"]) for comp in components}) != 3:
        return False

    distances = [int(comp["distance"]) for comp in components]
    if len(set(distances)) != 3 or min(np.diff(sorted(distances))) < 2:
        return False

    ordered = sorted(components, key=lambda comp: int(comp["distance"]))
    area_order = tuple(int(comp["area"]) for comp in ordered)
    if area_order != tuple(target_order):
        return False
    if area_order in ((1, 2, 3), (3, 2, 1)):
        return False

    left_order = tuple(
        int(comp["area"])
        for comp in sorted(components, key=lambda comp: (int(comp["bbox"][1]), int(comp["centroid"][0]), int(comp["area"])))
    )
    if left_order == area_order:
        return False

    row_order = tuple(
        int(comp["area"])
        for comp in sorted(components, key=lambda comp: (int(comp["centroid"][0]), int(comp["centroid"][1]), int(comp["area"])))
    )
    if row_order == area_order:
        return False

    centroid_rows = {int(comp["centroid"][0]) for comp in components}
    centroid_cols = {int(comp["centroid"][1]) for comp in components}
    if len(centroid_rows) == 1 or len(centroid_cols) == 1:
        return False

    expected = _independent_expected(color_by_area, area_order)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if max(actual.shape) > 30 or max(grid.shape) > 30:
        return False
    return True


def _choose_layout(height: int, width: int, target_order: tuple[int, ...]) -> dict[int, dict] | None:
    placements = {area: _all_placements(height, width, area) for area in AREAS}
    for _ in range(4000):
        chosen = {area: random.choice(placements[area]) for area in AREAS}
        if len({int(chosen[area]["distance"]) for area in AREAS}) != 3:
            continue
        if _touches_or_overlaps(chosen[1]["coords"], chosen[2]["coords"]):
            continue
        if _touches_or_overlaps(chosen[1]["coords"], chosen[3]["coords"]):
            continue
        if _touches_or_overlaps(chosen[2]["coords"], chosen[3]["coords"]):
            continue

        order = tuple(
            int(area)
            for area, _info in sorted(chosen.items(), key=lambda item: int(item[1]["distance"]))
        )
        if order != tuple(target_order):
            continue
        return chosen
    return None


def _choose_colors(target_order: tuple[int, ...]) -> dict[int, int] | None:
    for _ in range(200):
        colors = random.sample(PALETTE, len(AREAS))
        color_by_area = {area: int(color) for area, color in zip(AREAS, colors)}
        color_order = tuple(area for area in sorted(AREAS, key=lambda area: int(color_by_area[area])))
        if color_order == tuple(target_order) or color_order == tuple(reversed(target_order)):
            continue
        return color_by_area
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    target_order = random.choice(ORDER_PATTERNS)

    chosen = _choose_layout(height, width, target_order)
    if chosen is None:
        return None

    color_by_area = _choose_colors(target_order)
    if color_by_area is None:
        return None

    grid = _render_input(height, width, chosen, color_by_area)
    if not _validate_instance(grid, target_order, color_by_area):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
