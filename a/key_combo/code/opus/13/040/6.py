from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MAX_GENERATION_ATTEMPTS = 600
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 150
SHAPE_RADIUS = 3
GRID_SIDE_RANGE = (11, 16)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, fg_mask: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not fg_mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _sprite_from_component(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    box = _bbox_from_coords(coords)
    sprite = _crop_bbox(grid, box)
    anchor_cells = np.argwhere(sprite == GREY)
    assert anchor_cells.shape[0] == 1, "each sprite must contain exactly one gray anchor"
    anchor = tuple(map(int, anchor_cells[0]))
    payload = sorted(set(map(int, sprite.flatten())) - {BLACK, GREY})
    assert len(payload) == 1 and payload[0] in {BLUE, RED}, "each sprite must use one payload color"
    return sprite, anchor, int(payload[0])


def _sprite_pixels(sprite: np.ndarray, origin: tuple[int, int]) -> dict[tuple[int, int], int]:
    pixels: dict[tuple[int, int], int] = {}
    orow, ocol = origin
    for r, c in np.argwhere(sprite != BLACK):
        pixels[(int(r) - orow, int(c) - ocol)] = int(sprite[r, c])
    return pixels


def _overlay(
    sprite_a: np.ndarray,
    origin_a: tuple[int, int],
    sprite_b: np.ndarray,
    origin_b: tuple[int, int],
    *,
    overlap_mode: str = "checker",
    parity_reference: str = "canvas",
) -> np.ndarray:
    pixels_a = _sprite_pixels(sprite_a, origin_a)
    pixels_b = _sprite_pixels(sprite_b, origin_b)
    all_pos = sorted(set(pixels_a) | set(pixels_b))
    rows = [r for r, _ in all_pos]
    cols = [c for _, c in all_pos]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)

    for gr, gc in all_pos:
        rr = gr - min_r
        cc = gc - min_c
        a_col = pixels_a.get((gr, gc), BLACK)
        b_col = pixels_b.get((gr, gc), BLACK)
        if a_col != BLACK and b_col != BLACK:
            if overlap_mode == "a":
                color = a_col
            elif overlap_mode == "b":
                color = b_col
            else:
                if parity_reference == "canvas":
                    parity = (rr + cc) % 2
                elif parity_reference == "aligned":
                    parity = (gr + gc) % 2
                else:
                    raise ValueError("bad parity reference")
                color = a_col if parity == 0 else b_col
        else:
            color = a_col if a_col != BLACK else b_col
        out[rr, cc] = color
    return out


def _normalized_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, min_c = min(rows), min(cols)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_cells(cells: set[tuple[int, int]], kind: int) -> set[tuple[int, int]]:
    if kind == 0:
        fn = lambda r, c: (r, c)
    elif kind == 1:
        fn = lambda r, c: (-c, r)
    elif kind == 2:
        fn = lambda r, c: (-r, -c)
    elif kind == 3:
        fn = lambda r, c: (c, -r)
    elif kind == 4:
        fn = lambda r, c: (r, -c)
    elif kind == 5:
        fn = lambda r, c: (-r, c)
    elif kind == 6:
        fn = lambda r, c: (c, r)
    elif kind == 7:
        fn = lambda r, c: (-c, -r)
    else:
        raise ValueError("bad transform")
    return {fn(r, c) for r, c in cells}


def _is_asymmetric(cells: set[tuple[int, int]]) -> bool:
    base = _normalized_shape(cells)
    for kind in range(1, 8):
        if _normalized_shape(_transform_cells(cells, kind)) == base:
            return False
    return True


def _has_even_and_odd_non_anchor(cells: set[tuple[int, int]]) -> bool:
    parities = {(r + c) % 2 for r, c in cells if (r, c) != (0, 0)}
    return parities == {0, 1}


def _frontier(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                continue
            if abs(nr) <= SHAPE_RADIUS and abs(nc) <= SHAPE_RADIUS:
                out.add((nr, nc))
    return list(out)


def _grow_connected_shape(target_size: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < target_size:
        frontier = _frontier(cells)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _augment_shape(base: set[tuple[int, int]], target_size: int) -> set[tuple[int, int]] | None:
    cells = set(base)
    while len(cells) < target_size:
        frontier = _frontier(cells)
        frontier = [cell for cell in frontier if cell not in base]
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _shape_to_sprite(cells: set[tuple[int, int]], payload: int) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        sprite[r - r0, c - c0] = GREY if (r, c) == (0, 0) else payload
    anchor = (-r0, -c0)
    return sprite, anchor


def _shared_cells(shape_a: set[tuple[int, int]], shape_b: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(shape_a) & set(shape_b)


def _separated(abs_a: set[tuple[int, int]], abs_b: set[tuple[int, int]]) -> bool:
    for r, c in abs_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in abs_b:
                    return False
    return True


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    patch = grid[top : top + h, left : left + w]
    mask = sprite != BLACK
    patch[mask] = sprite[mask]


def _semantic_ok(shape_a: set[tuple[int, int]], shape_b: set[tuple[int, int]]) -> bool:
    shared = _shared_cells(shape_a, shape_b)
    if len(shared) < 4:
        return False
    if not _has_even_and_odd_non_anchor(shared):
        return False
    if not (_is_asymmetric(shape_a) and _is_asymmetric(shape_b)):
        return False
    if _normalized_shape(shape_a) == _normalized_shape(shape_b):
        return False
    if not (shape_a - shape_b and shape_b - shape_a):
        return False

    sprite_a, anchor_a = _shape_to_sprite(shape_a, RED)
    sprite_b, anchor_b = _shape_to_sprite(shape_b, BLUE)
    correct = _overlay(sprite_a, anchor_a, sprite_b, anchor_b)
    top_left = _overlay(sprite_a, (0, 0), sprite_b, (0, 0))
    aligned_parity = _overlay(
        sprite_a,
        anchor_a,
        sprite_b,
        anchor_b,
        parity_reference="aligned",
    )
    a_priority = _overlay(sprite_a, anchor_a, sprite_b, anchor_b, overlap_mode="a")
    b_priority = _overlay(sprite_a, anchor_a, sprite_b, anchor_b, overlap_mode="b")

    if np.array_equal(correct, top_left):
        return False
    if np.array_equal(correct, aligned_parity):
        return False
    if np.array_equal(correct, a_priority) or np.array_equal(correct, b_priority):
        return False
    return True


def _build_shapes() -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        base_size = random.choice((4, 5))
        base = _grow_connected_shape(base_size)
        if base is None or not _has_even_and_odd_non_anchor(base) or not _is_asymmetric(base):
            continue

        max_size = min(7, base_size + 2)
        size_a = random.randint(base_size + 1, max_size)
        size_b = random.randint(base_size + 1, max_size)
        shape_a = _augment_shape(base, size_a)
        shape_b = _augment_shape(base, size_b)
        if shape_a is None or shape_b is None:
            continue

        transform = random.randrange(8)
        shape_a = _transform_cells(shape_a, transform)
        shape_b = _transform_cells(shape_b, transform)
        if _semantic_ok(shape_a, shape_b):
            return shape_a, shape_b
    return None


def _render_input(shape_a: set[tuple[int, int]], shape_b: set[tuple[int, int]]) -> np.ndarray | None:
    sprite_a, _ = _shape_to_sprite(shape_a, RED)
    sprite_b, _ = _shape_to_sprite(shape_b, BLUE)
    ha, wa = sprite_a.shape
    hb, wb = sprite_b.shape

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(max(11, max(ha, hb) + 5), GRID_SIDE_RANGE[1])
        w = random.randint(max(11, max(wa, wb) + 5), GRID_SIDE_RANGE[1])
        pos_a = (random.randint(0, h - ha), random.randint(0, w - wa))
        pos_b = (random.randint(0, h - hb), random.randint(0, w - wb))

        abs_a = {
            (pos_a[0] + int(r), pos_a[1] + int(c))
            for r, c in np.argwhere(sprite_a != BLACK)
        }
        abs_b = {
            (pos_b[0] + int(r), pos_b[1] + int(c))
            for r, c in np.argwhere(sprite_b != BLACK)
        }
        if not _separated(abs_a, abs_b):
            continue

        grid = np.zeros((h, w), dtype=int)
        _place_sprite(grid, sprite_a, *pos_a)
        _place_sprite(grid, sprite_b, *pos_b)
        return grid
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    components = _find_components(grid, grid != BLACK)
    if len(components) != 2:
        return False

    sprites = []
    for coords in components:
        try:
            sprite, anchor, payload = _sprite_from_component(grid, coords)
        except AssertionError:
            return False
        if int(np.count_nonzero(sprite != BLACK)) < 4 or int(np.count_nonzero(sprite != BLACK)) > 7:
            return False
        sprites.append((payload, sprite, anchor))
    payloads = {payload for payload, _, _ in sprites}
    if payloads != {BLUE, RED}:
        return False

    sprite_a, anchor_a = next((sprite, anchor) for payload, sprite, anchor in sprites if payload == RED)
    sprite_b, anchor_b = next((sprite, anchor) for payload, sprite, anchor in sprites if payload == BLUE)
    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False

    checker = _overlay(sprite_a, anchor_a, sprite_b, anchor_b)
    if not np.array_equal(out, checker):
        return False
    if np.array_equal(out, _overlay(sprite_a, (0, 0), sprite_b, (0, 0))):
        return False
    if np.array_equal(out, _overlay(sprite_a, anchor_a, sprite_b, anchor_b, overlap_mode="a")):
        return False
    if np.array_equal(out, _overlay(sprite_a, anchor_a, sprite_b, anchor_b, overlap_mode="b")):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the two anchored sprites and merge overlap by checkerboard parity.

    Input structure:
    - The grid has a black background with exactly two separated 4-connected multicolor sprites.
    - One sprite uses red plus a single gray anchor pixel; the other uses blue plus a single gray anchor pixel.
    - Each sprite is considered inside its own tight bounding box before any alignment.

    Rule:
    - Crop each sprite to its bounding box and locate its unique gray anchor.
    - Align the two cropped sprites so the gray anchors coincide on one shared canvas.
    - For every canvas cell: keep the lone non-black pixel if only one sprite covers it; if both cover it, use the red-sprite pixel on even `(r+c)` canvas parity and the blue-sprite pixel on odd parity.
    - Return the tight crop of the merged non-black result.

    Key insight:
    - The gray pixels provide spatial registration, but the overlap is not ordinary overwrite: the aligned overlap must be sampled by checkerboard parity in the aligned canvas frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, g != BLACK)
    assert len(components) == 2, "expected exactly two foreground sprites"

    parsed: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for coords in components:
        sprite, anchor, payload = _sprite_from_component(g, coords)
        assert payload not in parsed, "each payload color should appear in exactly one sprite"
        parsed[payload] = (sprite, anchor)

    assert set(parsed) == {BLUE, RED}, "expected one red sprite and one blue sprite"
    sprite_a, anchor_a = parsed[RED]
    sprite_b, anchor_b = parsed[BLUE]
    return _overlay(sprite_a, anchor_a, sprite_b, anchor_b)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shapes = _build_shapes()
        if shapes is None:
            continue
        grid = _render_input(*shapes)
        if grid is None:
            continue
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
