from __future__ import annotations

import hashlib
import itertools
import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _normalize_cells(cells: set[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_to_mask(cells: tuple[Coord, ...]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_rectangular(cells: tuple[Coord, ...]) -> bool:
    mask = _cells_to_mask(cells)
    return bool(mask.all())


def _rotated(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(mask, -int(turns_cw))


def _shape_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes() + str(mask.shape).encode()


def _enumerate_polyominoes(max_area: int = 8) -> dict[int, list[np.ndarray]]:
    by_area: dict[int, set[tuple[Coord, ...]]] = {1: {((0, 0),)}}
    for area in range(2, max_area + 1):
        grown: set[tuple[Coord, ...]] = set()
        for poly in by_area[area - 1]:
            poly_set = set(poly)
            frontier: set[Coord] = set()
            for r, c in poly:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in poly_set:
                        frontier.add((nr, nc))
            for cell in frontier:
                grown.add(_normalize_cells(poly_set | {cell}))
        by_area[area] = grown

    catalog: dict[int, list[np.ndarray]] = {}
    for area, polys in by_area.items():
        shapes = []
        for poly in polys:
            if area < 3 or _is_rectangular(poly):
                continue
            shapes.append(_cells_to_mask(poly))
        catalog[area] = sorted(shapes, key=lambda m: (m.shape[0], m.shape[1], _shape_key(m)))
    return catalog


_SHAPES_BY_AREA = _enumerate_polyominoes(8)
_ROTATION_TURNS_BY_RANK = (0, 1, 2, 3)
_INPUT_ROWS = tuple(range(9, 13))
_INPUT_COLS = tuple(range(9, 13))
_AREA_COMBINATIONS = tuple(itertools.combinations(range(3, 9), 4))
_FG_COLORS = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 600
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _shape_changes_under_rotation(mask: np.ndarray, turns_cw: int) -> bool:
    return not np.array_equal(mask, _rotated(mask, turns_cw))


def _choose_ranked_shapes(rng: random.Random) -> list[np.ndarray] | None:
    areas = list(rng.choice(_AREA_COMBINATIONS))
    shapes: list[np.ndarray] = []
    for rank, area in enumerate(areas):
        candidates = _SHAPES_BY_AREA[area]
        turns = _ROTATION_TURNS_BY_RANK[rank]
        if turns == 0:
            filtered = candidates
        else:
            filtered = [shape for shape in candidates if _shape_changes_under_rotation(shape, turns)]
        if not filtered:
            return None
        shapes.append(rng.choice(filtered).copy())
    h_max = max(int(shape.shape[0]) for shape in shapes)
    w_max = max(int(shape.shape[1]) for shape in shapes)
    for shape, turns in zip(shapes, _ROTATION_TURNS_BY_RANK):
        rotated = _rotated(shape, turns)
        if rotated.shape[0] > h_max or rotated.shape[1] > w_max:
            return None
    return shapes


def _component_records(grid: np.ndarray, connectivity: int = 4) -> list[dict]:
    fg_colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    comps = []
    for coords in _components_mask(grid == fg, connectivity=connectivity):
        bbox = _bbox_coords(coords)
        sprite = _crop_bbox(grid, bbox)
        comps.append(
            {
                "coords": coords,
                "bbox": bbox,
                "area": int(coords.shape[0]),
                "sprite": sprite,
                "color": fg,
            }
        )
    return comps


def _assemble_from_components(components: list[dict], *, use_area_sort: bool = True,
                              rotations: tuple[int, int, int, int] = _ROTATION_TURNS_BY_RANK,
                              max_from_rotated: bool = False,
                              strict_fit: bool = True) -> np.ndarray | None:
    comps = list(components)
    if use_area_sort:
        comps.sort(key=lambda comp: int(comp["area"]))
    else:
        comps.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["area"])))

    sprites = [comp["sprite"] for comp in comps]
    if max_from_rotated:
        dims_source = [_rotated(sprite, rot) for sprite, rot in zip(sprites, rotations)]
    else:
        dims_source = sprites

    h_max = max(int(sprite.shape[0]) for sprite in dims_source)
    w_max = max(int(sprite.shape[1]) for sprite in dims_source)
    out = np.zeros((2 * h_max + 1, 2 * w_max + 1), dtype=int)
    offsets = ((0, 0), (0, w_max + 1), (h_max + 1, 0), (h_max + 1, w_max + 1))

    for sprite, turns, (r0, c0) in zip(sprites, rotations, offsets):
        patch = _rotated(sprite, turns)
        h, w = patch.shape
        if h > h_max or w > w_max:
            if strict_fit:
                raise AssertionError("rotated component must fit inside its assigned slot")
            return None
        out[r0 : r0 + h, c0 : c0 + w] = np.where(patch != 0, patch, out[r0 : r0 + h, c0 : c0 + w])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort four components by area and stamp them into the output corners with rank-driven quarter turns.

    Input structure:
    - The grid has black background `0` and exactly one non-black foreground color.
    - The foreground consists of exactly four 4-connected components whose areas are all distinct.
    - Each component is a small irregular shape; components may touch only diagonally in 8-neighborhood terms.

    Rule:
    - Find the four 4-connected foreground components and measure each area.
    - Sort them from smallest area to largest area, then crop each component to its own bounding box.
    - Let `H_max` and `W_max` be the maximum original bounding-box height and width across the four components.
    - Build a black output of shape `(2*H_max+1, 2*W_max+1)` and place the ranked components in the four corner slots, top-left aligned inside each slot.
    - Apply rotations by rank: smallest uses `0°`, second uses `90°` clockwise, third uses `180°`, largest uses `270°` clockwise.

    Key insight:
    - Area ranking is the only signal that simultaneously determines both the destination corner and the rotation applied to each separated component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    components = _component_records(g, connectivity=4)
    assert len(components) == 4, "expected exactly four 4-connected components"

    areas = [int(comp["area"]) for comp in components]
    assert len(set(areas)) == 4, "expected distinct component areas"
    return _assemble_from_components(components)


def _paint_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> set[Coord]:
    cells = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                rr = top + r
                cc = left + c
                grid[rr, cc] = color
                cells.add((rr, cc))
    return cells


def _touches_4(existing: set[Coord], cells: set[Coord]) -> bool:
    for r, c in cells:
        if (r, c) in existing:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing:
                return True
    return False


def _has_diagonal_touch(groups: list[set[Coord]]) -> bool:
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for r, c in groups[i]:
                for nr, nc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
                    if (nr, nc) in groups[j]:
                        return True
    return False


def _compact_enough(groups: list[set[Coord]], shape: tuple[int, int]) -> bool:
    coords = np.array(sorted(set().union(*groups)), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1
    h, w = shape
    return bbox_h >= h - 3 and bbox_w >= w - 3


def _place_components(ranked_shapes: list[np.ndarray], color: int, shape: tuple[int, int],
                      require_diag_touch: bool, rng: random.Random) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)

    order = list(range(4))
    rng.shuffle(order)
    placed: dict[int, set[Coord]] = {}
    occupied: set[Coord] = set()

    for index in order:
        mask = ranked_shapes[index]
        mh, mw = mask.shape
        anchors = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
        rng.shuffle(anchors)

        placed_here = None
        for top, left in anchors:
            cells = {(top + r, left + c) for r in range(mh) for c in range(mw) if mask[r, c]}
            if _touches_4(occupied, cells):
                continue
            placed_here = cells
            break

        if placed_here is None:
            return None

        for rr, cc in placed_here:
            grid[rr, cc] = color
        placed[index] = placed_here
        occupied |= placed_here

    groups = [placed[i] for i in range(4)]
    if require_diag_touch and not _has_diagonal_touch(groups):
        return None
    if not _compact_enough(groups, shape):
        return None
    return grid


def _semantic_shortcuts_fail(inp: np.ndarray) -> bool:
    components = _component_records(inp, connectivity=4)
    if len(components) != 4:
        return False

    correct = _assemble_from_components(components)
    no_rotation = _assemble_from_components(components, rotations=(0, 0, 0, 0))
    input_order = _assemble_from_components(components, use_area_sort=False, strict_fit=False)
    rotated_bbox_ref = _assemble_from_components(components, max_from_rotated=True)

    if np.array_equal(correct, no_rotation):
        return False
    if input_order is not None and np.array_equal(correct, input_order):
        return False
    if np.array_equal(correct, rotated_bbox_ref):
        return False
    return True


def _sample_attempt(rng: random.Random) -> np.ndarray | None:
    ranked_shapes = _choose_ranked_shapes(rng)
    if ranked_shapes is None:
        return None

    # Force the original-slot reference frame to matter often.
    if all(mask.shape[0] == mask.shape[1] for mask in ranked_shapes[1:]):
        return None

    color = rng.choice(_FG_COLORS)
    canvas_shape = (rng.choice(_INPUT_ROWS), rng.choice(_INPUT_COLS))
    require_diag_touch = rng.random() < 0.55
    inp = _place_components(ranked_shapes, color, canvas_shape, require_diag_touch, rng)
    if inp is None:
        return None

    comps4 = _component_records(inp, connectivity=4)
    if len(comps4) != 4:
        return None
    if sorted(int(comp["area"]) for comp in comps4) != sorted(int(mask.sum()) for mask in ranked_shapes):
        return None
    if require_diag_touch and len(_component_records(inp, connectivity=8)) >= 4:
        return None
    if not _semantic_shortcuts_fail(inp):
        return None

    out = main(inp)
    if out.ndim != 2 or max(out.shape) > 30:
        return None

    input_sig = hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest()
    output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _sample_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
