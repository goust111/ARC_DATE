from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
RED = 2
GREY = 5
BLUE = 1
GREEN = 3

GRID_SHAPE = (20, 20)
LEGEND_ROWS = 2
LEGEND_COLS = 2
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = neighbors4 if connectivity == 4 else neighbors8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_legend(g: np.ndarray, bg: int):
    src_a = int(g[0, 0])
    dst_a = int(g[0, 1])
    src_b = int(g[1, 0])
    dst_b = int(g[1, 1])
    assert len({src_a, src_b}) == 2
    assert len({dst_a, dst_b}) == 2
    assert bg not in {src_a, src_b, dst_a, dst_b}
    mapping = {src_a: dst_a, src_b: dst_b}
    swap = {src_a: src_b, src_b: src_a}
    return mapping, swap, {src_a, src_b}


def _find_top_band(g: np.ndarray, source_colors: set[int]) -> tuple[int, int]:
    row = [int(v) for v in g[0]]
    runs: list[tuple[int, int]] = []
    c = LEGEND_COLS
    while c < len(row):
        if row[c] not in source_colors:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] in source_colors:
            c += 1
        end = c - 1
        seg = row[start : end + 1]
        if len(seg) >= 3 and len(seg) % 2 == 1 and all(seg[i] != seg[i - 1] for i in range(1, len(seg))):
            runs.append((start, end))
    assert len(runs) == 1
    return runs[0]


def _find_holes(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    coords: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    solid = np.zeros((local_h, local_w), dtype=bool)
    for r, c in coords:
        solid[r - r0, c - c0] = True
    holes: set[tuple[int, int]] = set()
    empty = ~solid
    for comp in components_mask(empty, connectivity=4):
        touches_border = any(
            rr == 0 or cc == 0 or rr == local_h - 1 or cc == local_w - 1 for rr, cc in comp
        )
        if touches_border:
            continue
        for rr, cc in comp:
            holes.add((r0 + int(rr), c0 + int(cc)))
    return holes


def _extract_objects(
    g: np.ndarray, bg: int, source_colors: set[int], band: tuple[int, int]
) -> list[dict]:
    left, right = band
    objects: list[dict] = []
    for color in sorted(source_colors):
        mask = g == int(color)
        mask[:LEGEND_ROWS, :LEGEND_COLS] = False
        mask[0, left : right + 1] = False
        for comp in components_mask(mask, connectivity=4):
            coords = {(int(r), int(c)) for r, c in comp}
            bbox = bbox_coords(comp)
            holes = _find_holes(g.shape, bbox, coords)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "holes": holes,
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3], obj["color"]))
    return objects


def _claimable_halo_cells(
    bbox: tuple[int, int, int, int],
    solid: set[tuple[int, int]],
    holes: set[tuple[int, int]],
    wedge_footprint: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    out: set[tuple[int, int]] = set()
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            cell = (r, c)
            if cell in solid or cell in holes or cell in wedge_footprint:
                continue
            out.add(cell)
    return out


def _solve_core(
    grid: np.ndarray,
    *,
    preserve_holes: bool = True,
    use_halo: bool = True,
    use_blocking: bool = True,
    reverse_order: bool = False,
    collect_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mapping, swap, source_colors = _parse_legend(g, bg)
    band = _find_top_band(g, source_colors)
    left, right = band
    full_wedge_footprint = _footprint(g.shape, band)
    objects = _extract_objects(g, bg, source_colors, band)
    order = list(reversed(objects)) if reverse_order else list(objects)

    solid_cells = set().union(*(obj["coords"] for obj in objects)) if objects else set()
    hole_cells = set().union(*(obj["holes"] for obj in objects)) if objects else set()
    blocked = solid_cells | (hole_cells if preserve_holes else set())

    out = np.full_like(g, int(bg))
    wedge_footprint: set[tuple[int, int]] = set()
    wedge_drawn: set[tuple[int, int]] = set()
    blocked_hits: set[tuple[int, int]] = set()
    halo_cells_by_color: dict[int, set[tuple[int, int]]] = {}

    for top_c in range(left, right + 1):
        base_color = int(g[0, top_c])
        r = 0
        c = top_c
        while r < h and c < w and c <= right - r:
            cell = (r, c)
            wedge_footprint.add(cell)
            src_color = base_color if r % 2 == 0 else swap[base_color]
            if r > 0 and use_blocking and cell in blocked:
                blocked_hits.add(cell)
                break
            out[r, c] = int(mapping[src_color])
            wedge_drawn.add(cell)
            r += 1
            c += 1

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = int(obj["color"])

    if preserve_holes:
        for r, c in hole_cells:
            out[r, c] = int(bg)

    if use_halo:
        for obj in order:
            halo_color = int(mapping[obj["color"]])
            r0, c0, r1, c1 = obj["bbox"]
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            for r in range(rr0, rr1 + 1):
                for c in range(cc0, cc1 + 1):
                    if int(g[r, c]) != int(bg):
                        continue
                    if preserve_holes and (r, c) in hole_cells:
                        continue
                    if int(out[r, c]) != int(bg):
                        continue
                    out[r, c] = halo_color
                    halo_cells_by_color.setdefault(halo_color, set()).add((r, c))

    if preserve_holes:
        for r, c in hole_cells:
            out[r, c] = int(bg)

    out[:LEGEND_ROWS, :LEGEND_COLS] = int(bg)

    if not collect_meta:
        return out
    return out, {
        "mapping": mapping,
        "band": band,
        "objects": objects,
        "full_wedge_footprint": full_wedge_footprint,
        "wedge_footprint": wedge_footprint,
        "wedge_drawn": wedge_drawn,
        "blocked_hits": blocked_hits,
        "hole_cells": hole_cells,
        "halo_cells_by_color": halo_cells_by_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a top-band wedge and same-colored object halos through one shared legend map.

    Input structure:
    - The background is orange, with a fixed 2x2 legend in the upper-left corner.
    - The legend's left column gives two source colors and its right column gives their mapped target colors.
    - A single alternating top-row band uses the legend source colors, and several lower 4-connected source-colored objects sit away from the border; at least one object contains enclosed background holes.

    Rule:
    - Treat each top-band pixel as the head of a diagonal ray. As the rays move one row down and one column right, their source color swaps between the two legend source colors on every step.
    - Recolor each ray pixel through the legend map, but stop a ray as soon as it hits an object pixel or an enclosed hole cell; those blockers keep their original object-or-background state.
    - Then, in ascending bbox top-left order, expand each object's bbox by one cell and fill still-background cells in that expanded box with the mapped halo color for that object's source color, without overwriting wedge pixels, other objects, or enclosed holes.
    - Finally erase the legend itself back to background.

    Key insight:
    - The same 2x2 legend controls two different effects at once: it recolors the diagonal wedge and also chooses each object's halo color, while object pixels and holes physically stop the diagonal rays.
    """
    return _solve_core(grid)


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


HOLE_MASKS = [
    _mask_from_rows(["111", "101", "111"]),
    _mask_from_rows(["1111", "1001", "1111"]),
    _mask_from_rows(["111", "101", "101", "111"]),
]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    rows = np.any(out, axis=1)
    cols = np.any(out, axis=0)
    return out[rows][:, cols]


def _mask_holes(mask: np.ndarray) -> set[tuple[int, int]]:
    solid = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    bbox = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    return _find_holes(mask.shape, bbox, solid)


def _random_solid_mask() -> np.ndarray:
    for _ in range(80):
        target = random.randint(4, 10)
        canvas = np.zeros((5, 5), dtype=bool)
        start = (2, 2)
        cells = {start}
        canvas[start] = True
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nxt = random.choice(tuple(frontier))
            cells.add(nxt)
            canvas[nxt] = True
        rows = np.any(canvas, axis=1)
        cols = np.any(canvas, axis=0)
        mask = canvas[rows][:, cols]
        if len(_mask_holes(mask)) == 0:
            return _transform_mask(mask)
    return _mask_from_rows(["11", "11"])


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _footprint(shape: tuple[int, int], band: tuple[int, int]) -> set[tuple[int, int]]:
    h, _w = shape
    left, right = band
    out: set[tuple[int, int]] = set()
    for top_c in range(left, right + 1):
        r = 0
        c = top_c
        while r < h and c <= right - r:
            out.add((r, c))
            r += 1
            c += 1
    return out


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(sorted(cells), dtype=int)
    return bbox_coords(arr)


def _claimable_cells_for_placement(
    cells: set[tuple[int, int]],
    holes: set[tuple[int, int]],
    shape: tuple[int, int],
    wedge_footprint: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    return _claimable_halo_cells(_bbox_from_cells(cells), cells, holes, wedge_footprint, shape)


def _placement_ok(
    cells: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if r < 2:
            return False
        if (r, c) in reserved:
            return False
    return True


def _reserve(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            out.add((nr, nc))
    return out


def _place_hole_object(
    shape: tuple[int, int],
    wedge_footprint: set[tuple[int, int]],
) -> tuple[np.ndarray, int, int] | None:
    candidates = [mask.copy() for mask in HOLE_MASKS]
    random.shuffle(candidates)
    for base in candidates:
        for _ in range(8):
            mask = _transform_mask(base)
            holes_local = _mask_holes(mask)
            h0, w0 = mask.shape
            positions = [
                (r0, c0)
                for r0 in range(2, shape[0] - h0 - 1)
                for c0 in range(1, shape[1] - w0 - 1)
            ]
            random.shuffle(positions)
            for r0, c0 in positions:
                solid = _mask_cells(mask, r0, c0)
                holes = {(r0 + r, c0 + c) for r, c in holes_local}
                bbox = _bbox_from_cells(solid)
                if not _placement_ok(solid, set(), shape):
                    continue
                if not (solid & wedge_footprint):
                    continue
                if not (holes & wedge_footprint):
                    continue
                shadow_ready = any(
                    (r, c) in wedge_footprint and (r + 1, c + 1) in wedge_footprint for r, c in solid
                )
                if not shadow_ready:
                    continue
                if bbox[0] <= 4:
                    return mask, r0, c0
    return None


def _place_relative_object(
    mask: np.ndarray,
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    first_cells: set[tuple[int, int]],
    first_holes: set[tuple[int, int]],
    wedge_footprint: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h0, w0 = mask.shape
    first_bbox = _bbox_from_cells(first_cells)
    first_claimable = _claimable_cells_for_placement(first_cells, first_holes, shape, wedge_footprint)
    positions = []
    for r0 in range(2, shape[0] - h0 - 1):
        for c0 in range(1, shape[1] - w0 - 1):
            positions.append((r0, c0))
    random.shuffle(positions)
    for r0, c0 in positions:
        solid = _mask_cells(mask, r0, c0)
        if not _placement_ok(solid, reserved, shape):
            continue
        bbox = _bbox_from_cells(solid)
        exp_a = _expanded_bbox(first_bbox, shape)
        exp_b = _expanded_bbox(bbox, shape)
        if exp_a[2] < exp_b[0] or exp_b[2] < exp_a[0] or exp_a[3] < exp_b[1] or exp_b[3] < exp_a[1]:
            continue
        second_claimable = _claimable_cells_for_placement(solid, set(), shape, wedge_footprint)
        if not (first_claimable & second_claimable):
            continue
        return r0, c0
    return None


def _place_random_object(
    mask: np.ndarray,
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h0, w0 = mask.shape
    positions = [
        (r0, c0)
        for r0 in range(2, shape[0] - h0 - 1)
        for c0 in range(1, shape[1] - w0 - 1)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        solid = _mask_cells(mask, r0, c0)
        if _placement_ok(solid, reserved, shape):
            return r0, c0
    return None


def _build_candidate() -> np.ndarray | None:
    h, w = GRID_SHAPE
    grid = np.full((h, w), BG, dtype=int)
    grid[0, 0] = RED
    grid[0, 1] = BLUE
    grid[1, 0] = GREY
    grid[1, 1] = GREEN

    band_width = random.choice((9, 11, 13))
    band_left = random.randint(3, w - band_width - 1)
    band_right = band_left + band_width - 1
    start_color = random.choice((RED, GREY))
    for i, c in enumerate(range(band_left, band_right + 1)):
        grid[0, c] = start_color if i % 2 == 0 else (GREY if start_color == RED else RED)

    wedge_footprint = _footprint(grid.shape, (band_left, band_right))
    hole_obj = _place_hole_object(grid.shape, wedge_footprint)
    if hole_obj is None:
        return None
    hole_mask, hole_r0, hole_c0 = hole_obj
    hole_color = random.choice((RED, GREY))
    hole_cells = _mask_cells(hole_mask, hole_r0, hole_c0)
    hole_holes = {(hole_r0 + r, hole_c0 + c) for r, c in _mask_holes(hole_mask)}
    reserved = _reserve(hole_cells)

    second_mask = _random_solid_mask()
    second_color = GREY if hole_color == RED else RED
    second_pos = _place_relative_object(second_mask, grid.shape, reserved, hole_cells, hole_holes, wedge_footprint)
    if second_pos is None:
        return None
    second_r0, second_c0 = second_pos
    second_cells = _mask_cells(second_mask, second_r0, second_c0)
    reserved |= _reserve(second_cells)

    objects = [
        {"color": hole_color, "cells": hole_cells},
        {"color": second_color, "cells": second_cells},
    ]

    num_objects = random.randint(2, 4)
    for _ in range(num_objects - 2):
        mask = _random_solid_mask()
        pos = _place_random_object(mask, grid.shape, reserved)
        if pos is None:
            return None
        r0, c0 = pos
        cells = _mask_cells(mask, r0, c0)
        reserved |= _reserve(cells)
        objects.append({"color": random.choice((RED, GREY)), "cells": cells})

    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out, meta = _solve_core(inp, collect_meta=True)
    except Exception:
        return False

    mapping = meta["mapping"]
    objects = meta["objects"]
    if len(objects) < 2:
        return False
    if not any(obj["holes"] for obj in objects):
        return False
    if {obj["color"] for obj in objects} != {RED, GREY}:
        return False
    if not any(obj["coords"] & meta["full_wedge_footprint"] for obj in objects):
        return False
    if not any(obj["holes"] & meta["full_wedge_footprint"] for obj in objects):
        return False
    if not meta["blocked_hits"]:
        return False
    if mapping[RED] not in meta["halo_cells_by_color"] or mapping[GREY] not in meta["halo_cells_by_color"]:
        return False
    if not all(int(out[r, c]) == BG for r, c in meta["hole_cells"]):
        return False

    no_halo = _solve_core(inp, use_halo=False)
    no_block = _solve_core(inp, use_blocking=False)
    no_holes = _solve_core(inp, preserve_holes=False)
    reverse_order = _solve_core(inp, reverse_order=True)
    if np.array_equal(out, no_halo):
        return False
    if np.array_equal(out, no_block):
        return False
    if np.array_equal(out, no_holes):
        return False
    if np.array_equal(out, reverse_order):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _build_candidate()
        if candidate is None:
            continue
        if _semantic_checks(candidate):
            return candidate
    raise RuntimeError("failed to generate valid input")
