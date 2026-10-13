from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np


ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

COLOR_VALUES = tuple(range(10))
RECT_COUNT_RANGE = (2, 4)
PIECE_COUNT_RANGE = (2, 4)
MAX_GRID_SIDE = 30
MIN_GRID_SIDE = 18
MAX_GENERATION_ATTEMPTS = 300
RECT_PARTITION_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components_of_color(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(_normalize_coords(coords))

    return components


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _bbox_shape(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _piece_is_rectangle(coords: tuple[tuple[int, int], ...]) -> bool:
    h, w = _bbox_shape(coords)
    return len(coords) == h * w


def _ordered_factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(math.isqrt(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _packable_in_box(
    pieces: list[tuple[tuple[int, int], ...]],
    dims: tuple[int, int],
) -> bool:
    box_h, box_w = dims
    area = box_h * box_w
    if sum(len(piece) for piece in pieces) != area:
        return False

    placements_by_cell: list[list[tuple[int, int]]] = [[] for _ in range(area)]
    for piece_idx, piece in enumerate(pieces):
        arr = np.asarray(piece, dtype=int)
        piece_h = int(arr[:, 0].max()) + 1
        piece_w = int(arr[:, 1].max()) + 1
        if piece_h > box_h or piece_w > box_w:
            return False

        for r0 in range(box_h - piece_h + 1):
            for c0 in range(box_w - piece_w + 1):
                mask = 0
                for pr, pc in piece:
                    cell = (r0 + pr) * box_w + (c0 + pc)
                    mask |= 1 << cell
                for pr, pc in piece:
                    cell = (r0 + pr) * box_w + (c0 + pc)
                    placements_by_cell[cell].append((piece_idx, mask))

    full_mask = (1 << area) - 1
    all_used = (1 << len(pieces)) - 1

    @lru_cache(maxsize=None)
    def search(filled_mask: int, used_mask: int) -> bool:
        if filled_mask == full_mask:
            return used_mask == all_used

        remaining = full_mask & ~filled_mask
        first_open_bit = remaining & -remaining
        first_cell = first_open_bit.bit_length() - 1

        for piece_idx, placement_mask in placements_by_cell[first_cell]:
            if (used_mask >> piece_idx) & 1:
                continue
            if placement_mask & filled_mask:
                continue
            if search(filled_mask | placement_mask, used_mask | (1 << piece_idx)):
                return True
        return False

    return search(0, 0)


def _valid_rectangle_dims(pieces: list[tuple[tuple[int, int], ...]]) -> list[tuple[int, int]]:
    total_area = sum(len(piece) for piece in pieces)
    valid: list[tuple[int, int]] = []
    for dims in _ordered_factor_pairs(total_area):
        if _packable_in_box(pieces, dims):
            valid.append(dims)
    return valid


def _infer_rectangle_dims(pieces: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    valid = _valid_rectangle_dims(pieces)
    assert len(valid) == 1, "each color must reconstruct exactly one rectangle size"
    return valid[0]


def _render_nested_rectangles(
    rectangles: list[tuple[int, int, int]],
    bg: int,
) -> np.ndarray:
    max_h = max(h for h, _w, _color in rectangles)
    max_w = max(w for _h, w, _color in rectangles)
    assert all((max_h - h) % 2 == 0 for h, _w, _color in rectangles)
    assert all((max_w - w) % 2 == 0 for _h, w, _color in rectangles)

    out = np.full((max_h, max_w), bg, dtype=int)
    for h, w, color in sorted(rectangles, key=lambda item: item[0] * item[1], reverse=True):
        r0 = (max_h - h) // 2
        c0 = (max_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color fragments, reconstruct one rectangle per color, then stack
    the rectangles concentrically from largest to smallest.

    Input structure
    - One background color fills the empty space.
    - Each non-background color appears in 2 to 4 disconnected fragments.
    - For each color, its fragments can be translated and combined into exactly one solid rectangle.

    Rule
    - Infer each color's rectangle size from its total area and fragment shapes.
    - Sort the reconstructed rectangles by area.
    - Output the nested centered stack: the largest rectangle is the base layer and the
      smaller rectangles are painted on top, all sharing the same center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors

    rectangles: list[tuple[int, int, int]] = []
    for color in colors:
        pieces = _find_components_of_color(g, color)
        assert 2 <= len(pieces) <= 4
        dims = _infer_rectangle_dims(pieces)
        rectangles.append((dims[0], dims[1], color))

    areas = [h * w for h, w, _color in rectangles]
    assert len(set(areas)) == len(areas)
    return _render_nested_rectangles(rectangles, bg)


def _random_composition(total: int, parts: int, min_each: int) -> list[int] | None:
    if total < parts * min_each:
        return None
    remaining = total
    sizes: list[int] = []
    for i in range(parts):
        parts_left = parts - i - 1
        low = min_each
        high = remaining - parts_left * min_each
        pick = random.randint(low, high)
        sizes.append(pick)
        remaining -= pick
    random.shuffle(sizes)
    return sizes


def _sample_nested_dims(n_rects: int) -> list[tuple[int, int]]:
    for _ in range(200):
        height_pool = random.choice(([2, 4, 6, 8, 10], [3, 5, 7, 9]))
        width_pool = random.choice(([2, 4, 6, 8, 10], [3, 5, 7, 9]))
        heights = sorted(random.sample(height_pool, n_rects))
        widths = sorted(random.sample(width_pool, n_rects))
        dims = list(zip(heights, widths))
        if sum(h * w for h, w in dims) <= 170:
            return dims
    raise RuntimeError("failed to sample nested rectangle dimensions")


def _random_connected_partition(h: int, w: int, n_parts: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    min_piece = 2 if h * w >= 2 * n_parts else 1
    targets = _random_composition(h * w, n_parts, min_piece)
    if targets is None:
        return None

    labels = -np.ones((h, w), dtype=int)
    seeds = random.sample(cells, n_parts)
    frontiers: list[set[tuple[int, int]]] = [set() for _ in range(n_parts)]
    sizes = [1] * n_parts

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label

    for label, (r, c) in enumerate(seeds):
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[label].add((nr, nc))

    remaining = h * w - n_parts
    while remaining > 0:
        growable = [i for i in range(n_parts) if sizes[i] < targets[i] and frontiers[i]]
        if not growable:
            return None

        weights = [targets[i] - sizes[i] for i in growable]
        label = random.choices(growable, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[label]))
        frontiers[label].remove(cell)
        if labels[cell] != -1:
            continue

        labels[cell] = label
        sizes[label] += 1
        remaining -= 1
        cr, cc = cell
        for frontier in frontiers:
            frontier.discard(cell)
        for dr, dc in ORTHO_DIRS:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[label].add((nr, nc))

    pieces = []
    for label in range(n_parts):
        coords = np.argwhere(labels == label)
        if coords.size == 0:
            return None
        pieces.append(_normalize_coords([tuple(map(int, rc)) for rc in coords]))

    if all(_piece_is_rectangle(piece) for piece in pieces):
        return None
    return pieces


def _build_fragments_for_rectangle(
    h: int,
    w: int,
    n_parts: int,
) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(RECT_PARTITION_ATTEMPTS):
        pieces = _random_connected_partition(h, w, n_parts)
        if pieces is None:
            continue
        valid = _valid_rectangle_dims(pieces)
        if valid == [(h, w)]:
            return pieces
    return None


def _place_piece(
    grid: np.ndarray,
    blocked: np.ndarray,
    coords: tuple[tuple[int, int], ...],
    color: int,
) -> bool:
    piece_h, piece_w = _bbox_shape(coords)
    grid_h, grid_w = grid.shape
    if piece_h + 2 > grid_h or piece_w + 2 > grid_w:
        return False

    row_choices = list(range(1, grid_h - piece_h))
    col_choices = list(range(1, grid_w - piece_w))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    attempts = 0
    for r0 in row_choices:
        for c0 in col_choices:
            attempts += 1
            if attempts > PLACEMENT_ATTEMPTS:
                return False
            actual = [(r0 + r, c0 + c) for r, c in coords]
            if any(blocked[r, c] for r, c in actual):
                continue
            for r, c in actual:
                grid[r, c] = color
            for r, c in actual:
                blocked[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True
            return True
    return False


def _build_input_grid(
    fragments: list[tuple[tuple[tuple[int, int], ...], int]],
    bg: int,
) -> np.ndarray | None:
    total_area = sum(len(coords) for coords, _color in fragments)
    max_piece_h = max(_bbox_shape(coords)[0] for coords, _color in fragments)
    max_piece_w = max(_bbox_shape(coords)[1] for coords, _color in fragments)
    side_floor = max(
        MIN_GRID_SIDE,
        max_piece_h + 4,
        max_piece_w + 4,
        int(math.ceil(math.sqrt(total_area * 4.5))),
    )
    if side_floor > MAX_GRID_SIDE:
        return None

    grid_h = random.randint(side_floor, MAX_GRID_SIDE)
    grid_w = random.randint(side_floor, MAX_GRID_SIDE)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    ordered = fragments[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[0]), reverse=True)
    for coords, color in ordered:
        if not _place_piece(grid, blocked, coords, color):
            return None

    if int(np.sum(grid == bg)) <= int(np.sum(grid != bg)):
        return None
    return grid


def _expected_output(rectangles: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    return _render_nested_rectangles(rectangles, bg)


def _generate_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    dims_list = _sample_nested_dims(n_rects)
    bg = random.choice(COLOR_VALUES)
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_rects)

    rectangles: list[tuple[int, int, int]] = []
    fragments: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for (h, w), color in zip(dims_list, colors):
        area = h * w
        max_parts = min(PIECE_COUNT_RANGE[1], area - 1)
        if max_parts < PIECE_COUNT_RANGE[0]:
            return None
        n_parts = random.randint(PIECE_COUNT_RANGE[0], max_parts)
        pieces = _build_fragments_for_rectangle(h, w, n_parts)
        if pieces is None:
            return None
        rectangles.append((h, w, color))
        for piece in pieces:
            fragments.append((piece, color))

    grid = _build_input_grid(fragments, bg)
    if grid is None:
        return None

    expected = _expected_output(rectangles, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    for color in colors:
        comp_count = len(_find_components_of_color(grid, color))
        if not (2 <= comp_count <= 4):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
