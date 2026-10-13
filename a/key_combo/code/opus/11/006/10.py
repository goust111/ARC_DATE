from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
REGION_COLORS = (BLUE, RED, GREEN, YELLOW)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _bbox_area(shape: tuple[tuple[int, int], ...]) -> int:
    h, w = _shape_dims(shape)
    return h * w


def _shape_has_hole(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return bool(np.any(~mask & ~seen))


def _transform_shape(shape: tuple[tuple[int, int], ...], op: int) -> tuple[tuple[int, int], ...]:
    cells = list(shape)
    if op >= 4:
        cells = [(r, -c) for r, c in cells]
        op -= 4
    for _ in range(op):
        cells = [(c, -r) for r, c in cells]
    return _normalize(set(cells))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {_transform_shape(shape, op) for op in range(8)}
    return list(variants)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _normalize(set(coords))


def _min_distance_to_cell(coords: list[tuple[int, int]], cell: tuple[int, int]) -> int:
    tr, tc = cell
    return min(abs(r - tr) + abs(c - tc) for r, c in coords)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30
    assert int(g[h - 1, w - 1]) == MAGENTA
    assert int(np.count_nonzero(g == MAGENTA)) == 1

    pointer_color = int(g[h - 1, w - 2])
    assert pointer_color in REGION_COLORS

    payload = g.copy()
    payload[h - 1, w - 2] = BLACK
    payload[h - 1, w - 1] = BLACK

    allowed = {BLACK, MAGENTA, *REGION_COLORS}
    assert set(map(int, g.flat)) <= allowed

    comps: list[dict] = []
    for color in REGION_COLORS:
        color_comps = _connected_components(payload == color)
        assert len(color_comps) <= 1
        for comp in color_comps:
            shape = _component_shape(comp)
            area = len(comp)
            bbox_area = _bbox_area(shape)
            dims = _shape_dims(shape)
            assert 4 <= area <= 12
            assert dims[0] >= 2 and dims[1] >= 2
            assert bbox_area > area
            assert not _shape_has_hole(shape)
            comps.append(
                {
                    "color": color,
                    "coords": comp,
                    "area": area,
                    "shape": shape,
                    "bbox_area": bbox_area,
                }
            )

    assert 3 <= len(comps) <= 4
    colors = [comp["color"] for comp in comps]
    areas = [comp["area"] for comp in comps]
    assert len(set(colors)) == len(colors)
    assert len(set(areas)) == len(areas)

    occupied_by_color = {}
    for comp in comps:
        for cell in comp["coords"]:
            occupied_by_color[cell] = comp["color"]

    for (r, c), color in occupied_by_color.items():
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                other = occupied_by_color.get((rr, cc))
                assert other is None or other == color

    for cell in ((h - 1, w - 2), (h - 1, w - 1)):
        for rr, cc in (cell, *_neighbors8(*cell)):
            if 0 <= rr < h and 0 <= cc < w:
                assert (rr, cc) not in occupied_by_color

    target = [comp for comp in comps if comp["color"] == pointer_color]
    assert len(target) == 1
    target_comp = target[0]
    row_shift = target_comp["area"] % h
    col_shift = target_comp["area"] % w
    assert row_shift != 0 and col_shift != 0

    return {
        "grid": g,
        "payload": payload,
        "components": comps,
        "target": target_comp,
        "row_shift": row_shift,
        "col_shift": col_shift,
    }


def _transform_from_info(info: dict) -> np.ndarray:
    payload = info["payload"]
    return np.roll(payload, shift=(info["row_shift"], info["col_shift"]), axis=(0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """Cyclically shift the grid by the area of the region named by the bottom-right color pointer.

    Input structure:
    - A black background contains 3-4 separated irregular single-color 4-connected regions in colors 1-4.
    - The bottom-right cell is a magenta marker, and the cell immediately to its left is a single color pointer.
    - That pointer color matches exactly one foreground region; the two bottom-right marker cells are not part of any region.

    Rule:
    - Read the color immediately left of the magenta marker.
    - Ignore the two marker cells, find the unique 4-connected region of that color, and count its area A in pixels.
    - Clear the two marker cells to black, then cyclically roll the whole remaining grid right by A mod W and down by A mod H.

    Key insight:
    - The marker does not encode a shift directly; it only points to a region, and that region's area becomes the shared roll amount for both axes.
    """
    return _transform_from_info(_analyze_grid(grid))


def _random_polyomino(area: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    feasible_boxes = [
        (h, w)
        for h in range(2, min(6, area + 1))
        for w in range(2, min(6, area + 1))
        if area <= h * w - 1 and area >= max(h, w)
    ]
    if not feasible_boxes:
        return None

    for _ in range(200):
        box_h, box_w = rng.choice(feasible_boxes)
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            choice_pool = sorted(
                frontier,
                key=lambda cell: (
                    sum((nr, nc) in cells for nr, nc in _neighbors4(*cell)),
                    rng.random(),
                ),
                reverse=True,
            )
            cells.add(choice_pool[0])
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        if _bbox_area(shape) <= area:
            continue
        if _shape_has_hole(shape):
            continue
        return shape
    return None


def _placement_ok(cells: list[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    return all(cell not in blocked for cell in cells)


def _add_margin(cells: list[tuple[int, int]], blocked: set[tuple[int, int]], h: int, w: int) -> None:
    for r, c in cells:
        blocked.add((r, c))
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                blocked.add((rr, cc))


def _place_shapes(
    h: int,
    w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> list[dict] | None:
    blocked: set[tuple[int, int]] = set()
    reserved = [(h - 1, w - 2), (h - 1, w - 1)]
    _add_margin(reserved, blocked, h, w)

    placed: list[dict] = []
    items = sorted(enumerate(shapes), key=lambda item: _bbox_area(item[1]), reverse=True)
    for index, shape in items:
        variants = _shape_variants(shape)
        rng.shuffle(variants)
        found = None
        for variant in variants:
            sh, sw = _shape_dims(variant)
            positions = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
            rng.shuffle(positions)
            for r0, c0 in positions:
                cells = [(r0 + r, c0 + c) for r, c in variant]
                if _placement_ok(cells, blocked):
                    found = {
                        "index": index,
                        "shape": variant,
                        "coords": cells,
                    }
                    _add_margin(cells, blocked, h, w)
                    break
            if found is not None:
                break
        if found is None:
            return None
        placed.append(found)
    return sorted(placed, key=lambda item: item["index"])


def _choose_layout(rng: random.Random) -> tuple[int, int, int, bool]:
    wrap_mode = rng.random() < 0.5
    region_count = rng.choice([3, 4])
    if wrap_mode:
        h = rng.choice([10, 11])
        w = rng.choice([10, 11])
    else:
        h = rng.choice([11, 12, 13])
        w = rng.choice([11, 12, 13])
    return h, w, region_count, wrap_mode


def _choose_areas(h: int, w: int, region_count: int, wrap_mode: bool, rng: random.Random):
    if wrap_mode:
        target_area = 12
        other_areas = rng.sample([4, 5, 6, 7, 8, 9, 10], region_count - 1)
        areas = other_areas + [target_area]
        target_rank = len(areas) - 1
    else:
        all_areas = sorted(rng.sample(list(range(4, 11)), region_count))
        target_rank = 1
        target_area = all_areas[target_rank]
        areas = all_areas
    assert target_area % h != 0 and target_area % w != 0
    return areas, target_area, target_rank


def _build_input(
    h: int,
    w: int,
    placed: list[dict],
    target_area: int,
    target_rank: int,
    wrap_mode: bool,
    rng: random.Random,
) -> np.ndarray | None:
    if len(placed) == 0:
        return None

    sorted_by_area = sorted(placed, key=lambda item: len(item["coords"]))
    target_shape = sorted_by_area[target_rank]
    colors = rng.sample(list(REGION_COLORS), len(placed))
    rng.shuffle(colors)

    grid = np.zeros((h, w), dtype=int)
    components: list[dict] = []
    for color, item in zip(colors, placed):
        coords = item["coords"]
        for r, c in coords:
            grid[r, c] = color
        components.append(
            {
                "color": color,
                "coords": coords,
                "area": len(coords),
                "is_target": item is target_shape,
            }
        )

    target_comp = next(comp for comp in components if comp["is_target"])
    if target_comp["area"] != target_area:
        return None

    marker_anchor = (h - 1, w - 2)
    if not wrap_mode:
        target_dist = _min_distance_to_cell(target_comp["coords"], marker_anchor)
        if target_dist == min(_min_distance_to_cell(comp["coords"], marker_anchor) for comp in components):
            return None

    grid[h - 1, w - 2] = target_comp["color"]
    grid[h - 1, w - 1] = MAGENTA
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    out = _transform_from_info(info)
    if np.array_equal(out, info["payload"]):
        return False

    target_area = info["target"]["area"]
    bbox_area = info["target"]["bbox_area"]
    h, w = grid.shape
    payload = info["payload"]
    row_shift = info["row_shift"]
    col_shift = info["col_shift"]

    if np.array_equal(out, np.roll(payload, shift=(row_shift, 0), axis=(0, 1))):
        return False
    if np.array_equal(out, np.roll(payload, shift=(0, col_shift), axis=(0, 1))):
        return False

    counted_plus_pointer = (target_area + 1) % h, (target_area + 1) % w
    if np.array_equal(out, np.roll(payload, shift=counted_plus_pointer, axis=(0, 1))):
        return False

    if bbox_area % h == 0 or bbox_area % w == 0:
        return False
    if np.array_equal(out, np.roll(payload, shift=(bbox_area % h, bbox_area % w), axis=(0, 1))):
        return False

    for comp in info["components"]:
        if comp is info["target"]:
            continue
        alt_shift = (comp["area"] % h, comp["area"] % w)
        if np.array_equal(out, np.roll(payload, shift=alt_shift, axis=(0, 1))):
            return False

    return bool(np.array_equal(main(grid), out))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, region_count, wrap_mode = _choose_layout(rng)
    areas, target_area, target_rank = _choose_areas(h, w, region_count, wrap_mode, rng)

    shapes = []
    for area in areas:
        shape = _random_polyomino(area, rng)
        if shape is None:
            return None
        shapes.append(shape)

    placed = _place_shapes(h, w, shapes, rng)
    if placed is None:
        return None

    grid = _build_input(h, w, placed, target_area, target_rank, wrap_mode, rng)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
