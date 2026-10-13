from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]

ALL_COLORS = tuple(range(10))
MIN_SPRITE_AREA = 3
GEN_ATTEMPTS = 700


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or not bool(mask[r0, c0]):
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalized_shape(coords: np.ndarray) -> Shape:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _coord_set(coords: np.ndarray) -> set[Coord]:
    return {(int(r), int(c)) for r, c in coords}


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components(grid == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                    "shape": _normalized_shape(coords),
                }
            )
    return out


def _infer_shadow_rule(grid: np.ndarray) -> dict[str, object]:
    bg = _most_common_color(grid)
    components = _find_color_components(grid, bg)
    assert components, "expected foreground sprites"

    chips = [comp for comp in components if int(comp["area"]) < MIN_SPRITE_AREA]
    significant = [comp for comp in components if int(comp["area"]) >= MIN_SPRITE_AREA]
    assert len(significant) >= 4, "expected an exemplar pair and at least two unshadowed sprites"
    assert all(int(comp["area"]) == 1 for comp in chips), "small color witnesses must be singleton chips"

    chip_colors = {int(comp["color"]) for comp in chips}
    significant_by_color: dict[int, list[dict[str, object]]] = {}
    for comp in significant:
        significant_by_color.setdefault(int(comp["color"]), []).append(comp)

    assert all(len(comps) == 1 for comps in significant_by_color.values()), (
        "each non-background color should have one large component"
    )
    body_colors = {color for color in significant_by_color if color in chip_colors}
    shadow_only_colors = [color for color in significant_by_color if color not in chip_colors]
    assert len(body_colors) >= 3, "sprite body colors must be identified by color chips"
    assert len(shadow_only_colors) == 1, "exactly one large component color must lack a body chip"

    shadow_color = int(shadow_only_colors[0])
    shadow_comp = significant_by_color[shadow_color][0]

    pairings: list[tuple[dict[str, object], dict[str, object], int, int]] = []
    shadow_shape = shadow_comp["shape"]
    shadow_cells = _coord_set(shadow_comp["coords"])  # type: ignore[arg-type]
    sr0, sc0, _sr1, _sc1 = shadow_comp["bbox"]  # type: ignore[misc]

    for color in sorted(body_colors):
        body_comp = significant_by_color[color][0]
        if body_comp["shape"] != shadow_shape:
            continue
        br0, bc0, _br1, _bc1 = body_comp["bbox"]  # type: ignore[misc]
        dr, dc = int(sr0) - int(br0), int(sc0) - int(bc0)
        if dr == 0 and dc == 0:
            continue
        body_cells = _coord_set(body_comp["coords"])  # type: ignore[arg-type]
        if {(r + dr, c + dc) for r, c in body_cells} == shadow_cells:
            pairings.append((body_comp, shadow_comp, dr, dc))

    assert len(pairings) == 1, "the exemplar body and its shadow must form one unique translated shape pair"
    exemplar, shadow, dr, dc = pairings[0]
    exemplar_color = int(exemplar["color"])
    targets = [
        significant_by_color[color][0]
        for color in sorted(body_colors)
        if int(color) != exemplar_color
    ]
    assert targets, "expected unshadowed sprites to receive the learned shadow"

    return {
        "bg": int(bg),
        "shadow_color": int(shadow_color),
        "exemplar": exemplar,
        "shadow": shadow,
        "offset": (int(dr), int(dc)),
        "targets": targets,
        "significant": significant,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the exemplar shadow offset and color, then add matching shadows to every other sprite.

    Input structure:
    - A dominant background contains several separated monochromatic sprite bodies.
    - Each real sprite body color has a tiny isolated same-color chip; the shadow color lacks that chip
      and appears only as the exemplar shadow copy.
    - Exactly one body has a same-shape translated companion in the chipless shadow color.

    Rule:
    - Identify that translated exemplar pair, using the chipless member as the shadow.
    - The vector from exemplar body to shadow gives the shared shadow offset, and the chipless color gives
      the shadow color.
    - For every other real sprite body, paint the same translated shape in the shadow color; any shadow
      cells outside the grid are clipped, and existing foreground pixels stay on top of shadows.

    Key insight:
    - The transformation is not a fixed direction: a visible exemplar pair defines both the offset vector
      and the color to reuse on the remaining sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    info = _infer_shadow_rule(g)
    bg = int(info["bg"])
    shadow_color = int(info["shadow_color"])
    dr, dc = info["offset"]  # type: ignore[misc]

    out = g.copy()
    h, w = out.shape
    for comp in info["targets"]:  # type: ignore[assignment]
        coords = comp["coords"]  # type: ignore[index]
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                out[rr, cc] = shadow_color
    return out


# ------------------------------ generator ------------------------------


def _norm_shape_from_cells(cells: set[Coord]) -> Shape:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_shape(existing: set[Shape]) -> Shape | None:
    for _ in range(500):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(8, h * w)
        area = random.randint(4, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells: set[Coord] = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < h and 0 <= nc < w
        }
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue

        shape = _norm_shape_from_cells(cells)
        rows = {r for r, _c in shape}
        cols = {c for _r, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if shape in existing:
            continue
        existing.add(shape)
        return shape
    return None


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_cells(shape: Shape, top: int, left: int) -> set[Coord]:
    return {(int(top) + r, int(left) + c) for r, c in shape}


def _in_bounds(cells: Iterable[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _separated(a: Iterable[Coord], b: Iterable[Coord], padding: int = 1) -> bool:
    b_cells = set(b)
    for r, c in a:
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if (rr, cc) in b_cells:
                    return False
    return True


def _can_place(cells: set[Coord], occupied: np.ndarray, padding: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        r0, r1 = max(0, r - padding), min(h, r + padding + 2)
        c0, c1 = max(0, c - padding), min(w, c + padding + 2)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _paint(grid: np.ndarray, occupied: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        occupied[r, c] = True


def _valid_offsets_for(shape: Shape) -> list[Coord]:
    base = set(shape)
    offsets: list[Coord] = []
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 3:
                continue
            if abs(dr) + abs(dc) > 8:
                continue
            shifted = {(r + dr, c + dc) for r, c in base}
            if base.isdisjoint(shifted) and _separated(base, shifted, padding=1):
                offsets.append((dr, dc))
    random.shuffle(offsets)
    return offsets


def _place_exemplar_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: Shape,
    offset: Coord,
    body_color: int,
    shadow_color: int,
) -> tuple[set[Coord], set[Coord]] | None:
    h, w = grid.shape
    sh, sw = _shape_size(shape)
    dr, dc = offset
    for _ in range(1200):
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        body = _shape_cells(shape, top, left)
        shadow = {(r + dr, c + dc) for r, c in body}
        if not _in_bounds(shadow, grid.shape):
            continue
        if not body.isdisjoint(shadow) or not _separated(body, shadow, padding=1):
            continue
        if not _can_place(body, occupied, padding=1):
            continue
        temp = occupied.copy()
        for r, c in body:
            temp[r, c] = True
        if not _can_place(shadow, temp, padding=1):
            continue
        _paint(grid, occupied, body, body_color)
        _paint(grid, occupied, shadow, shadow_color)
        return body, shadow
    return None


def _candidate_sprite_positions(
    shape: Shape,
    occupied: np.ndarray,
    *,
    require_clipped_shadow: Coord | None = None,
) -> list[set[Coord]]:
    h, w = occupied.shape
    sh, sw = _shape_size(shape)
    out: list[set[Coord]] = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            cells = _shape_cells(shape, top, left)
            if not _can_place(cells, occupied, padding=1):
                continue
            if require_clipped_shadow is not None:
                dr, dc = require_clipped_shadow
                shifted = [(r + dr, c + dc) for r, c in cells]
                inside = [(r, c) for r, c in shifted if 0 <= r < h and 0 <= c < w]
                if len(inside) == 0 or len(inside) == len(shifted):
                    continue
                if not any(not occupied[r, c] for r, c in inside):
                    continue
            out.append(cells)
    random.shuffle(out)
    return out


def _place_single_sprite(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: Shape,
    color: int,
    *,
    clipped_offset: Coord | None = None,
) -> set[Coord] | None:
    candidates = _candidate_sprite_positions(shape, occupied, require_clipped_shadow=clipped_offset)
    if not candidates:
        return None
    cells = random.choice(candidates[: min(80, len(candidates))])
    _paint(grid, occupied, cells, color)
    return cells


def _place_shadow_overlap_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    source_shape: Shape,
    source_color: int,
    cover_shape: Shape,
    cover_color: int,
    offset: Coord,
) -> tuple[set[Coord], set[Coord]] | None:
    h, w = grid.shape
    source_positions = _candidate_sprite_positions(source_shape, occupied)
    random.shuffle(source_positions)
    dr, dc = offset

    for source_cells in source_positions[:160]:
        temp = occupied.copy()
        for r, c in source_cells:
            temp[r, c] = True
        shifted_inside = {
            (r + dr, c + dc)
            for r, c in source_cells
            if 0 <= r + dr < h and 0 <= c + dc < w
        }
        if not shifted_inside:
            continue

        cover_positions = _candidate_sprite_positions(cover_shape, temp)
        random.shuffle(cover_positions)
        for cover_cells in cover_positions[:180]:
            overlap = shifted_inside & cover_cells
            if not overlap:
                continue
            visible = any((cell not in cover_cells) and (not occupied[cell]) for cell in shifted_inside)
            if not visible:
                continue
            _paint(grid, occupied, source_cells, source_color)
            _paint(grid, occupied, cover_cells, cover_color)
            return source_cells, cover_cells
    return None


def _place_chip(grid: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for r, c in cells:
        if _can_place({(r, c)}, occupied, padding=1):
            grid[r, c] = int(color)
            occupied[r, c] = True
            return True
    return False


def _expected_from_records(
    grid: np.ndarray,
    sprite_records: list[dict[str, object]],
    exemplar_index: int,
    offset: Coord,
    shadow_color: int,
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    dr, dc = offset
    for i, record in enumerate(sprite_records):
        if i == exemplar_index:
            continue
        for r, c in record["cells"]:  # type: ignore[index]
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(shadow_color)
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(grid)
        info = _infer_shadow_rule(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = int(info["bg"])
    shadow_color = int(info["shadow_color"])
    dr, dc = info["offset"]  # type: ignore[misc]
    h, w = grid.shape
    clipped = 0
    covered_by_foreground = 0
    visible_by_sprite = 0

    for comp in info["targets"]:  # type: ignore[assignment]
        visible = 0
        for r, c in comp["coords"]:  # type: ignore[index]
            rr, cc = int(r) + int(dr), int(c) + int(dc)
            if not (0 <= rr < h and 0 <= cc < w):
                clipped += 1
            elif int(grid[rr, cc]) != bg:
                if int(grid[rr, cc]) != shadow_color:
                    covered_by_foreground += 1
            elif int(expected[rr, cc]) == shadow_color:
                visible += 1
        if visible > 0:
            visible_by_sprite += 1

    added = (grid == bg) & (expected == shadow_color)
    if int(np.count_nonzero(added)) < 4:
        return False
    if clipped == 0:
        return False
    if covered_by_foreground == 0:
        return False
    if visible_by_sprite < 2:
        return False
    if set(map(int, np.unique(expected))) == {bg}:
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    bg = random.choice(ALL_COLORS)
    available = [c for c in ALL_COLORS if c != bg]

    body_count = random.randint(4, min(5, len(available) - 1))
    picked = random.sample(available, body_count + 1)
    body_colors = picked[:body_count]
    shadow_color = picked[-1]

    shapes: list[Shape] = []
    used_shapes: set[Shape] = set()
    for _ in range(body_count):
        shape = _random_shape(used_shapes)
        if shape is None:
            return None
        shapes.append(shape)

    offsets = _valid_offsets_for(shapes[0])
    if not offsets:
        return None

    random.shuffle(offsets)
    for offset in offsets[:20]:
        grid = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        records: list[dict[str, object]] = []

        placed_pair = _place_exemplar_pair(
            grid, occupied, shapes[0], offset, body_colors[0], shadow_color
        )
        if placed_pair is None:
            continue
        exemplar_cells, _shadow_cells = placed_pair
        records.append({"color": body_colors[0], "shape": shapes[0], "cells": exemplar_cells})

        edge_cells = _place_single_sprite(
            grid, occupied, shapes[1], body_colors[1], clipped_offset=offset
        )
        if edge_cells is None:
            continue
        records.append({"color": body_colors[1], "shape": shapes[1], "cells": edge_cells})

        overlap_pair = _place_shadow_overlap_pair(
            grid,
            occupied,
            shapes[2],
            body_colors[2],
            shapes[3],
            body_colors[3],
            offset,
        )
        if overlap_pair is None:
            continue
        overlap_source, overlap_cover = overlap_pair
        records.append({"color": body_colors[2], "shape": shapes[2], "cells": overlap_source})
        records.append({"color": body_colors[3], "shape": shapes[3], "cells": overlap_cover})

        for idx in range(4, body_count):
            extra = _place_single_sprite(grid, occupied, shapes[idx], body_colors[idx])
            if extra is None:
                break
            records.append({"color": body_colors[idx], "shape": shapes[idx], "cells": extra})
        if len(records) != body_count:
            continue

        for color in body_colors:
            if not _place_chip(grid, occupied, color):
                break
        else:
            expected = _expected_from_records(
                grid, records, exemplar_index=0, offset=offset, shadow_color=shadow_color, bg=bg
            )
            if _semantic_checks(grid, expected):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid exemplar-shadow input")
