import numpy as np

from collections import deque
from functools import lru_cache
from math import isqrt


BG = 0
ANCHOR_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()
GEN_ATTEMPTS = 400
PARTITION_ATTEMPTS = 250
PLACEMENT_ATTEMPTS = 250
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 25
OUTPUT_HEIGHT_MAX = 22
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


RECT_DIMS_BY_AREA: dict[int, list[tuple[int, int]]] = {}
for _h in range(2, 6):
    for _w in range(2, 6):
        _area = _h * _w
        if _area >= 6:
            RECT_DIMS_BY_AREA.setdefault(_area, []).append((_h, _w))
AVAILABLE_AREAS = tuple(sorted(RECT_DIMS_BY_AREA))


def _choice(seq):
    return seq[int(RNG.integers(len(seq)))]


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _is_solid_rectangle(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    return len(shape) == h * w


def _color_order(grid: np.ndarray, bg: int, anchor: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in grid.ravel(order="C"):
        color = int(value)
        if color in (bg, anchor) or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(cells)
    return parts


def _count_tilings(
    shapes: list[tuple[tuple[int, int], ...]],
    height: int,
    width: int,
    limit: int = 2,
) -> int:
    area = height * width
    ordered = sorted(shapes, key=lambda shape: (-len(shape), _shape_dims(shape), shape))
    cover_lists: list[list[list[int]]] = []
    for shape in ordered:
        sh, sw = _shape_dims(shape)
        if sh > height or sw > width:
            return 0
        by_cell = [[] for _ in range(area)]
        for top in range(height - sh + 1):
            for left in range(width - sw + 1):
                mask = 0
                indices: list[int] = []
                for r, c in shape:
                    idx = (top + r) * width + (left + c)
                    mask |= 1 << idx
                    indices.append(idx)
                for idx in indices:
                    by_cell[idx].append(mask)
        if not any(by_cell):
            return 0
        cover_lists.append(by_cell)

    full_mask = (1 << area) - 1
    piece_count = len(ordered)

    @lru_cache(maxsize=None)
    def search(occupied: int, used_mask: int) -> int:
        if occupied == full_mask:
            return 1
        remaining = full_mask ^ occupied
        target_bit = remaining & -remaining
        target = target_bit.bit_length() - 1
        total = 0
        for piece_index in range(piece_count):
            if used_mask & (1 << piece_index):
                continue
            for placement in cover_lists[piece_index][target]:
                if placement & occupied:
                    continue
                total += search(occupied | placement, used_mask | (1 << piece_index))
                if total >= limit:
                    return total
        return total

    return search(0, 0)


def _unique_rectangle_dims(
    shapes: list[tuple[tuple[int, int], ...]]
) -> tuple[int, int] | None:
    area = sum(len(shape) for shape in shapes)
    found: tuple[int, int] | None = None
    solutions = 0
    for h in range(1, isqrt(area) + 1):
        if area % h != 0:
            continue
        w = area // h
        dims = [(h, w)] if h == w else [(h, w), (w, h)]
        for height, width in dims:
            remaining = 2 - solutions
            count = _count_tilings(shapes, int(height), int(width), limit=max(1, remaining))
            if count == 0:
                continue
            if found is None:
                found = (int(height), int(width))
            solutions += count
            if solutions > 1:
                return None
    return found if solutions == 1 else None


def _stack_rectangles(rectangles: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: item["area"])
    out_h = sum(item["height"] for item in ordered)
    out_w = max(item["width"] for item in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    bottom = out_h
    for item in ordered:
        top = bottom - item["height"]
        out[top:bottom, out_w - item["width"] : out_w] = item["color"]
        bottom = top
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored fragments, reconstruct one solid rectangle per color,
    then stack the rectangles by ascending area against the grey anchor's
    bottom-right alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_bg = g[g != BG]
    colors, counts = np.unique(non_bg, return_counts=True)
    anchor_candidates = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
    assert len(anchor_candidates) == 1, "expected exactly one single-pixel anchor color"
    anchor = anchor_candidates[0]

    rectangles: list[dict[str, int]] = []
    for color in _color_order(g, BG, anchor):
        shapes = [_normalize(cells) for cells in _components(g == color)]
        assert 2 <= len(shapes) <= 4, "each color must have 2-4 fragments"
        dims = _unique_rectangle_dims(shapes)
        assert dims is not None, "fragments must tile a unique solid rectangle"
        height, width = dims
        rectangles.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert 3 <= len(rectangles) <= 5, "expected 3-5 colored groups"
    return _stack_rectangles(rectangles)


def _random_partition(height: int, width: int, piece_count: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    seed_indices = RNG.choice(len(cells), size=piece_count, replace=False)
    pieces = [{cells[int(index)]} for index in seed_indices]
    owner = {cells[int(index)]: i for i, index in enumerate(seed_indices)}
    unassigned = set(cells) - set(owner)

    while unassigned:
        frontier: list[tuple[tuple[int, int], list[int]]] = []
        for cell in unassigned:
            adjacent = []
            for dr, dc in NEIGHBORS4:
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in owner:
                    adjacent.append(owner[neighbor])
            if adjacent:
                frontier.append((cell, adjacent))
        if not frontier:
            return None
        cell, adjacent = _choice(frontier)
        min_size = min(len(pieces[index]) for index in adjacent)
        options = [index for index in adjacent if len(pieces[index]) == min_size]
        piece_index = int(_choice(options))
        pieces[piece_index].add(cell)
        owner[cell] = piece_index
        unassigned.remove(cell)

    if min(len(piece) for piece in pieces) < 2:
        return None

    shapes = [_normalize(list(piece)) for piece in pieces]
    if len(set(shapes)) != piece_count:
        return None
    if all(_is_solid_rectangle(shape) for shape in shapes):
        return None
    dims = _unique_rectangle_dims(shapes)
    if dims != (height, width):
        return None
    return shapes


def _generate_shapes(
    height: int, width: int, piece_count: int
) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(PARTITION_ATTEMPTS):
        shapes = _random_partition(height, width, piece_count)
        if shapes is not None:
            return shapes
    return None


def _sample_rect_specs(color_count: int) -> list[dict[str, int]] | None:
    for _ in range(PARTITION_ATTEMPTS):
        areas = [int(area) for area in RNG.choice(AVAILABLE_AREAS, size=color_count, replace=False)]
        specs: list[dict[str, int]] = []
        total_height = 0
        widths: list[int] = []
        for area in areas:
            height, width = _choice(RECT_DIMS_BY_AREA[area])
            specs.append({"area": area, "height": int(height), "width": int(width)})
            total_height += int(height)
            widths.append(int(width))
        if total_height > OUTPUT_HEIGHT_MAX:
            continue
        if len(set(widths)) == 1 and len(specs) > 1:
            continue
        return specs
    return None


def _scatter_fragments(
    grid_h: int,
    grid_w: int,
    colored_shapes: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    forbidden[max(0, grid_h - 2) : grid_h, max(0, grid_w - 2) : grid_w] = True

    ordered = sorted(
        colored_shapes,
        key=lambda item: (-len(item[1]), -_shape_dims(item[1])[0] * _shape_dims(item[1])[1]),
    )
    for color, shape in ordered:
        sh, sw = _shape_dims(shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(RNG.integers(0, grid_h - sh + 1))
            left = int(RNG.integers(0, grid_w - sw + 1))
            coords = [(top + r, left + c) for r, c in shape]
            if any(forbidden[r, c] for r, c in coords):
                continue
            for r, c in coords:
                grid[r, c] = int(color)
            for r, c in coords:
                forbidden[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True
            placed = True
            break
        if not placed:
            return None

    grid[grid_h - 1, grid_w - 1] = ANCHOR_COLOR
    return grid


def _build_expected_output(specs: list[dict[str, int]]) -> np.ndarray:
    return _stack_rectangles(specs)


def _gen_attempt() -> np.ndarray | None:
    color_count = int(RNG.integers(3, 6))
    colors = list(map(int, RNG.choice(COLOR_POOL, size=color_count, replace=False)))
    rect_specs = _sample_rect_specs(color_count)
    if rect_specs is None:
        return None
    colored_shapes: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    output_specs: list[dict[str, int]] = []

    for color, spec in zip(colors, rect_specs):
        area = int(spec["area"])
        valid_counts = [pieces for pieces in (2, 3, 4) if area >= 2 * pieces]
        piece_count = int(_choice(valid_counts))
        shapes = _generate_shapes(int(spec["height"]), int(spec["width"]), piece_count)
        if shapes is None:
            return None
        for shape in shapes:
            colored_shapes.append((color, shape))
        output_specs.append(
            {
                "color": int(color),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "area": area,
            }
        )

    grid_h = int(RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    grid_w = int(RNG.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    grid = _scatter_fragments(grid_h, grid_w, colored_shapes)
    if grid is None:
        return None

    expected = _build_expected_output(output_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
