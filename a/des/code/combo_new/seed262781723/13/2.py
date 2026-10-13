from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
COLOR_NAMES = {
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
}
ALPHABETICAL_COLOR_ORDER = tuple(sorted(COLOR_NAMES, key=lambda color: COLOR_NAMES[color]))
DRAW_RANK = {color: rank for rank, color in enumerate(ALPHABETICAL_COLOR_ORDER)}
GENERATOR_COLORS = (1, 3, 7, 2, 4)
MIN_AREA = 4
MAX_AREA = 8
GENERATION_ATTEMPTS = 400
MAX_OVERLAP_CELLS = 6


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_bbox(anchor: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    h, w = _shape_dims(shape)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _shape_cells(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = anchor
    return [(r0 + dr, c0 + dc) for dr, dc in shape]


def _paint_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    color: int,
) -> None:
    h, w = grid.shape
    for r, c in _shape_cells(shape, anchor):
        assert 0 <= r < h and 0 <= c < w, "swapped shape must stay in bounds"
        grid[r, c] = int(color)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _discover_pairs(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    assert colors, "expected at least one foreground pair"
    pairs: list[dict] = []
    for color in colors:
        components = _components_of_color(grid, color)
        assert len(components) == 2, "each non-background color must appear in exactly two objects"
        objects = []
        for coords in components:
            assert MIN_AREA <= len(coords) <= MAX_AREA, "object area must be 4-8 pixels"
            shape = _normalize_shape(coords)
            anchor = (min(r for r, _ in coords), min(c for _, c in coords))
            objects.append(
                {
                    "anchor": anchor,
                    "bbox": _shape_bbox(anchor, shape),
                    "shape": shape,
                }
            )
        objects.sort(key=lambda obj: obj["anchor"])
        assert objects[0]["shape"] != objects[1]["shape"], "same-color paired shapes must differ"
        assert not _boxes_intersect(objects[0]["bbox"], objects[1]["bbox"]), "paired bboxes must not overlap"
        pairs.append({"color": color, "objects": objects})
    return pairs


def _render_swapped(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for pair in sorted(pairs, key=lambda item: DRAW_RANK[item["color"]]):
        first, second = pair["objects"]
        _paint_shape(out, first["shape"], second["anchor"], pair["color"])
        _paint_shape(out, second["shape"], first["anchor"], pair["color"])
    return out


def _pair_output_cells(pair: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    first, second = pair["objects"]
    first_cells = set(_shape_cells(first["shape"], second["anchor"]))
    second_cells = set(_shape_cells(second["shape"], first["anchor"]))
    return first_cells | second_cells, first_cells & second_cells


def _shape_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    cells = set(coords)
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nbr in _neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return len(seen) == len(cells)


def _build_shape_library() -> tuple[tuple[tuple[int, int], ...], ...]:
    all_cells = [(r, c) for r in range(4) for c in range(4)]
    shapes = set()
    for area in range(MIN_AREA, MAX_AREA + 1):
        for combo in combinations(all_cells, area):
            if _shape_connected(combo):
                shapes.add(_normalize_shape(combo))
    return tuple(sorted(shapes, key=lambda shape: (len(shape), _shape_dims(shape), shape)))


SHAPE_LIBRARY = _build_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored objects and swap their shapes onto each other's anchors.

    Input structure:
    - The background is black.
    - The grid contains an even number of monochrome 4-connected objects, each using 4-8 cells.
    - Every non-background color appears exactly twice, creating one natural pair whose two tight bboxes do not overlap.

    Rule:
    - Pair the two objects of each color.
    - For each pair, take object A's shape as offsets from its bbox top-left and redraw that shape at object B's bbox top-left; also redraw B's shape at A's top-left.
    - Clear all original object pixels to black before drawing the swapped shapes.
    - Different color pairs are independent; if swapped pixels from different colors overlap, later drawing in alphabetical visible color-name order overwrites earlier drawing.

    Key insight:
    - Color identifies which two objects belong together, but the actual exchange is of normalized shape masks rather than of whole positions or bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    pairs = _discover_pairs(g)
    return _render_swapped(g.shape, pairs)


def _sample_pair_count(want_overlap: bool) -> int:
    if want_overlap:
        return random.choice((2, 3, 3))
    return random.choice((2, 2, 3, 4))


def _sample_grid_shape(want_overlap: bool, pair_count: int) -> tuple[int, int]:
    if want_overlap:
        low = 13 + pair_count
        high = 19 + pair_count
    else:
        low = 15 + pair_count
        high = 24 + pair_count
    return random.randint(low, min(high, 30)), random.randint(low, min(high, 30))


def _sample_pairs(pair_count: int) -> list[dict]:
    colors = random.sample(GENERATOR_COLORS, pair_count)
    pairs = []
    for color in colors:
        shape_a, shape_b = random.sample(SHAPE_LIBRARY, 2)
        pairs.append(
            {
                "color": int(color),
                "objects": [
                    {"shape": shape_a, "anchor": None},
                    {"shape": shape_b, "anchor": None},
                ],
            }
        )
    return pairs


def _fit_dims_for_role(pair: dict, role: int) -> tuple[int, int]:
    own_shape = pair["objects"][role]["shape"]
    incoming_shape = pair["objects"][1 - role]["shape"]
    own_h, own_w = _shape_dims(own_shape)
    inc_h, inc_w = _shape_dims(incoming_shape)
    return max(own_h, inc_h), max(own_w, inc_w)


def _can_place(
    anchor: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
    blocked: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    bbox = _shape_bbox(anchor, shape)
    if any(_boxes_intersect(bbox, other) for other in boxes):
        return False
    return all(not blocked[r, c] for r, c in _shape_cells(shape, anchor))


def _mark_blocked(blocked: np.ndarray, shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> None:
    h, w = blocked.shape
    for r, c in _shape_cells(shape, anchor):
        blocked[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_pairs(pairs: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    roles = []
    for pair_index, pair in enumerate(pairs):
        for role in (0, 1):
            fit_h, fit_w = _fit_dims_for_role(pair, role)
            own_shape = pair["objects"][role]["shape"]
            roles.append(
                {
                    "pair_index": pair_index,
                    "role": role,
                    "fit_h": fit_h,
                    "fit_w": fit_w,
                    "shape": own_shape,
                }
            )
    roles.sort(key=lambda item: (item["fit_h"] * item["fit_w"], len(item["shape"])), reverse=True)

    for role_info in roles:
        pair_index = role_info["pair_index"]
        role = role_info["role"]
        fit_h = role_info["fit_h"]
        fit_w = role_info["fit_w"]
        own_shape = role_info["shape"]
        candidates = [(r, c) for r in range(h - fit_h + 1) for c in range(w - fit_w + 1)]
        random.shuffle(candidates)
        anchor = None
        for candidate in candidates:
            if _can_place(candidate, own_shape, blocked, boxes):
                anchor = candidate
                break
        if anchor is None:
            return False
        pairs[pair_index]["objects"][role]["anchor"] = anchor
        boxes.append(_shape_bbox(anchor, own_shape))
        _mark_blocked(blocked, own_shape, anchor)
    return True


def _build_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for pair in pairs:
        for obj in pair["objects"]:
            _paint_shape(grid, obj["shape"], obj["anchor"], pair["color"])
    return grid


def _cross_pair_overlap_count(pairs: list[dict]) -> tuple[int, bool]:
    cell_sets = []
    for pair in pairs:
        pair_cells, self_overlap = _pair_output_cells(pair)
        if self_overlap:
            return 0, False
        cell_sets.append((pair["color"], pair_cells))
    overlap_cells: set[tuple[int, int]] = set()
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            overlap_cells |= cell_sets[i][1] & cell_sets[j][1]
    return len(overlap_cells), True


def _valid_input_structure(grid: np.ndarray, pairs: list[dict]) -> bool:
    discovered = _discover_pairs(grid)
    if len(discovered) != len(pairs):
        return False
    discovered_by_color = {pair["color"]: pair for pair in discovered}
    for pair in pairs:
        color = pair["color"]
        if color not in discovered_by_color:
            return False
        want_shapes = sorted(obj["shape"] for obj in pair["objects"])
        got_shapes = sorted(obj["shape"] for obj in discovered_by_color[color]["objects"])
        if want_shapes != got_shapes:
            return False
    return True


def _gen_attempt(want_overlap: bool) -> np.ndarray | None:
    pair_count = _sample_pair_count(want_overlap)
    pairs = _sample_pairs(pair_count)
    grid_shape = _sample_grid_shape(want_overlap, pair_count)
    if not _place_pairs(pairs, grid_shape):
        return None

    overlap_count, self_clear = _cross_pair_overlap_count(pairs)
    if not self_clear:
        return None
    if want_overlap:
        if overlap_count == 0 or overlap_count > MAX_OVERLAP_CELLS:
            return None
    elif overlap_count != 0:
        return None

    grid = _build_grid(grid_shape, pairs)
    if not _valid_input_structure(grid, pairs):
        return None

    expected = _render_swapped(grid_shape, pairs)
    if np.array_equal(expected, grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    want_overlap = bool(random.getrandbits(1))
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(want_overlap)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-swap ARC input")
