from __future__ import annotations

from collections import deque
import random

import numpy as np


GEN_RETRIES = 600
BG_OPTIONS = (0, 7)
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 8)
GRID_SIZE_RANGE = (16, 28)


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(g: np.ndarray, bg: int) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: dict[int, list[list[tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.setdefault(color, []).append(comp)
    return comps


def _bbox_center(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)


def _is_isolated_singleton(g: np.ndarray, cell: tuple[int, int], bg: int) -> bool:
    h, w = g.shape
    r, c = cell
    for nr, nc in _neighbors4(int(r), int(c)):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
            return False
    return True


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _shift(cells: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r) + int(dr), int(c) + int(dc)) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored shape from its nearer portal to the matching far portal.

    Input structure:
    - The most common color is the background.
    - Each foreground color forms one 4-connected shape of at least three pixels.
    - The same color also appears as exactly two isolated one-pixel portal markers.
    - Portal markers have only background in their 4-neighborhood and are separated from all shapes.

    Rule:
    - For each color, identify the multi-pixel component as the shape and the two isolated singletons as its portal pair.
    - Compute the shape's integer bounding-box center, then choose the portal with smaller Manhattan distance to that center as the entrance.
    - Translate the whole shape so that the same bounding-box center lands on the other portal, clear the old shape, and remove both portal markers.

    Key insight:
    - The isolated pixels are not noise: color pairs each shape with its portals, proximity chooses the entrance, and the far portal gives the new anchor position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)

    moves: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []
    for color, comps in _components_by_color(g, bg).items():
        shapes = [comp for comp in comps if len(comp) >= 3]
        singletons = [comp[0] for comp in comps if len(comp) == 1 and _is_isolated_singleton(g, comp[0], bg)]
        assert len(shapes) == 1, "each color must have exactly one multi-pixel shape"
        assert len(singletons) == 2, "each color must have exactly two isolated portal pixels"

        shape = shapes[0]
        center = _bbox_center(shape)
        d0 = _manhattan(center, singletons[0])
        d1 = _manhattan(center, singletons[1])
        assert d0 != d1, "shape center must not be equidistant from its portals"
        entrance, exit_portal = (singletons[0], singletons[1]) if d0 < d1 else (singletons[1], singletons[0])

        dr = int(exit_portal[0] - center[0])
        dc = int(exit_portal[1] - center[1])
        shifted = _shift(shape, dr, dc)
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted shape must stay in bounds"
        moves.append((int(color), shape, shifted, entrance, exit_portal))

    assert moves, "expected at least one portal-guided shape"

    out = g.copy()
    for _color, shape, _shifted, entrance, exit_portal in moves:
        for r, c in shape:
            out[int(r), int(c)] = bg
        out[int(entrance[0]), int(entrance[1])] = bg
        out[int(exit_portal[0]), int(exit_portal[1])] = bg

    occupied: set[tuple[int, int]] = set()
    for color, _shape, shifted, _entrance, _exit_portal in moves:
        for r, c in shifted:
            loc = (int(r), int(c))
            assert loc not in occupied, "shifted shapes must not collide"
            occupied.add(loc)
            out[int(r), int(c)] = int(color)

    return out


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _shape_bbox(cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _random_shape(rng: random.Random) -> set[tuple[int, int]]:
    height = rng.choice((3, 3, 4, 5))
    width = rng.choice((3, 3, 4, 5))
    target = rng.randint(SHAPE_AREA_RANGE[0], min(SHAPE_AREA_RANGE[1], height * width))
    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    cells = _normalize(cells)
    if rng.random() < 0.5:
        max_c = max(c for _r, c in cells)
        cells = {(r, max_c - c) for r, c in cells}
    if rng.random() < 0.5:
        max_r = max(r for r, _c in cells)
        cells = {(max_r - r, c) for r, c in cells}
    if rng.random() < 0.35:
        cells = {(c, r) for r, c in cells}
    return _normalize(cells)


def _expanded(cells: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int, gap: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-int(gap), int(gap) + 1):
            for dc in range(-int(gap), int(gap) + 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _in_bounds(cells: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= int(r) < h and 0 <= int(c) < w for r, c in cells)


def _clear_against(
    cells: set[tuple[int, int]] | list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    gap: int = 1,
) -> bool:
    return _in_bounds(cells, h, w) and not (_expanded(cells, h, w, gap) & occupied)


def _stamp(grid: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _region_id(cell: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    return (0 if int(cell[0]) < h // 2 else 1, 0 if int(cell[1]) < w // 2 else 1)


def _candidate_entry_offsets() -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            dist = abs(dr) + abs(dc)
            if 3 <= dist <= 5:
                offsets.append((int(dr), int(dc)))
    return offsets


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["shape"]:  # type: ignore[index]
            out[int(r), int(c)] = int(bg)
        for r, c in spec["portals"]:  # type: ignore[index]
            out[int(r), int(c)] = int(bg)
    for spec in specs:
        _stamp(out, spec["shifted"], int(spec["color"]))  # type: ignore[arg-type,index]
    return out


def _place_one(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    input_occupied: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    offsets = _candidate_entry_offsets()
    for _ in range(500):
        shape_rel = _random_shape(rng)
        r0, c0, r1, c1 = _shape_bbox(shape_rel)
        shape_h = int(r1 - r0 + 1)
        shape_w = int(c1 - c0 + 1)
        if h - shape_h <= 6 or w - shape_w <= 6:
            continue
        top = rng.randint(2, h - shape_h - 3)
        left = rng.randint(2, w - shape_w - 3)
        shape = {(int(r + top), int(c + left)) for r, c in shape_rel}
        if not _clear_against(shape, input_occupied, h, w, gap=1):
            continue

        center = _bbox_center(shape)
        rng.shuffle(offsets)
        for dr_entry, dc_entry in offsets[:24]:
            entry = (int(center[0] + dr_entry), int(center[1] + dc_entry))
            occupied_with_shape = input_occupied | shape
            if not _clear_against([entry], occupied_with_shape, h, w, gap=1):
                continue
            d_entry = _manhattan(center, entry)

            for _exit_try in range(80):
                exit_portal = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
                if _region_id(entry, h, w) == _region_id(exit_portal, h, w):
                    continue
                if _manhattan(center, exit_portal) <= d_entry + 3:
                    continue
                if not _clear_against([exit_portal], occupied_with_shape | {entry}, h, w, gap=1):
                    continue

                delta_r = int(exit_portal[0] - center[0])
                delta_c = int(exit_portal[1] - center[1])
                shifted = set(_shift(shape, delta_r, delta_c))
                if not _in_bounds(shifted, h, w):
                    continue
                if shifted & (input_occupied | shape | {entry}):
                    continue
                if _expanded(shifted, h, w, gap=1) & output_blocked:
                    continue

                return {
                    "color": int(color),
                    "shape": shape,
                    "portals": (entry, exit_portal),
                    "center": center,
                    "shifted": shifted,
                }
    return None


def _validate_generated(grid: np.ndarray, specs: list[dict[str, object]], bg: int) -> bool:
    if _most_common_color(grid) != int(bg):
        return False
    comps = _components_by_color(grid, int(bg))
    if set(comps) != {int(spec["color"]) for spec in specs}:
        return False
    for spec in specs:
        color = int(spec["color"])
        parts = comps[color]
        if sorted(len(part) for part in parts)[:2] != [1, 1]:
            return False
        if sum(1 for part in parts if len(part) >= 3) != 1:
            return False
        for portal in spec["portals"]:  # type: ignore[index]
            if not _is_isolated_singleton(grid, portal, int(bg)):  # type: ignore[arg-type]
                return False

    expected = _expected_from_specs(grid, specs, int(bg))
    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected) and not np.array_equal(grid, expected))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    bg = rng.choice(BG_OPTIONS)
    colors = [c for c in range(10) if c != bg]
    count = rng.randint(*OBJECT_COUNT_RANGE)
    chosen = rng.sample(colors, count)

    grid = np.full((h, w), int(bg), dtype=int)
    specs: list[dict[str, object]] = []
    input_occupied: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    for color in chosen:
        spec = _place_one(rng, h, w, int(color), input_occupied, output_blocked)
        if spec is None:
            return None
        specs.append(spec)
        shape = set(spec["shape"])  # type: ignore[arg-type]
        portals = set(spec["portals"])  # type: ignore[arg-type]
        shifted = set(spec["shifted"])  # type: ignore[arg-type]
        input_occupied.update(shape)
        input_occupied.update(portals)
        output_blocked.update(_expanded(shifted, h, w, gap=1))

    all_portals: set[tuple[int, int]] = set()
    all_shapes: set[tuple[int, int]] = set()
    for spec in specs:
        all_portals.update(set(spec["portals"]))  # type: ignore[arg-type]
        all_shapes.update(set(spec["shape"]))  # type: ignore[arg-type]

    for spec in specs:
        allowed = {spec["portals"][1]}  # type: ignore[index]
        shifted = set(spec["shifted"])  # type: ignore[arg-type]
        if shifted & (all_portals - allowed):
            return None
        if shifted & all_shapes:
            return None

    for spec in specs:
        _stamp(grid, spec["shape"], int(spec["color"]))  # type: ignore[arg-type,index]
        _stamp(grid, spec["portals"], int(spec["color"]))  # type: ignore[arg-type,index]

    if not _validate_generated(grid, specs, int(bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
