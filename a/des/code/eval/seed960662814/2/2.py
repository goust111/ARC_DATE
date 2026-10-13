import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _mask_components(grid == color)


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _push_to_edge(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> np.ndarray:
    moved = coords.copy()
    if direction == "up":
        moved[:, 0] -= int(coords[:, 0].min())
    elif direction == "down":
        moved[:, 0] += shape[0] - 1 - int(coords[:, 0].max())
    elif direction == "left":
        moved[:, 1] -= int(coords[:, 1].min())
    elif direction == "right":
        moved[:, 1] += shape[1] - 1 - int(coords[:, 1].max())
    else:
        raise AssertionError(f"unknown direction: {direction}")
    return moved


def _wall_info(grid: np.ndarray, bg: int, coords: np.ndarray, color: int):
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    border = _border_cells((r0, c0, r1, c1))
    coord_set = {tuple(map(int, cell)) for cell in coords}
    if coord_set - border:
        return None

    missing = border - coord_set
    if len(missing) != 1:
        return None

    gap_r, gap_c = next(iter(missing))
    on_top = gap_r == r0
    on_bottom = gap_r == r1
    on_left = gap_c == c0
    on_right = gap_c == c1
    if sum([on_top, on_bottom, on_left, on_right]) != 1:
        return None

    if on_top:
        direction = "up"
    elif on_bottom:
        direction = "down"
    elif on_left:
        direction = "left"
    else:
        direction = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None
    if np.any(interior == color):
        return None

    sprite_mask = interior != bg
    sprite_components = _mask_components(sprite_mask)
    if len(sprite_components) != 1:
        return None

    sprite_rel = sprite_components[0]
    sprite_coords = sprite_rel + np.array([r0 + 1, c0 + 1], dtype=int)
    sprite_colors = set(map(int, interior[sprite_mask]))
    if len(sprite_colors) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "wall_color": int(color),
        "direction": direction,
        "gap": (int(gap_r), int(gap_c)),
        "wall_coords": coords.copy(),
        "sprite_coords": sprite_coords,
        "sprite_color": next(iter(sprite_colors)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each single-gap rectangular wall, use its gap side as a launch direction,
    slide the interior sprite to the corresponding grid edge, and recolor it with
    the wall color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape
    bg = _get_bg(g)

    wall_infos = []
    claimed_cells: set[tuple[int, int]] = set()
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _color_components(g, color):
            info = _wall_info(g, bg, comp, color)
            if info is None:
                continue
            wall_infos.append(info)
            claimed_cells.update(map(tuple, info["wall_coords"]))
            claimed_cells.update(map(tuple, info["sprite_coords"]))

    assert wall_infos, "expected at least one wall object"
    non_bg_cells = {tuple(map(int, cell)) for cell in np.argwhere(g != bg)}
    assert claimed_cells == non_bg_cells, "all foreground cells must belong to a wall or its interior sprite"

    out = g.copy()
    for info in wall_infos:
        sprite_coords = info["sprite_coords"]
        out[sprite_coords[:, 0], sprite_coords[:, 1]] = bg

    occupied = {tuple(map(int, cell)) for cell in np.argwhere(out != bg)}
    wall_infos.sort(key=lambda item: item["bbox"])
    for info in wall_infos:
        final_coords = _push_to_edge(info["sprite_coords"], info["direction"], (h, w))
        for r, c in final_coords:
            cell = (int(r), int(c))
            assert 0 <= r < h and 0 <= c < w
            assert cell not in occupied, "moved sprites must not overlap walls or one another"
        out[final_coords[:, 0], final_coords[:, 1]] = info["wall_color"]
        occupied.update(map(tuple, final_coords))

    return out


# --------------- generator ---------------

GEN_SPRITE_COLOR = 5
GEN_BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GEN_DIRECTIONS = ("up", "down", "left", "right")

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
OBJECT_COUNT_RANGE = (1, 4)

OUTER_H_RANGE = (6, 9)
OUTER_W_RANGE = (6, 9)
BOUNDARY_MARGIN = 5

SPRITE_AREA_RANGE = (3, 7)
SPRITE_BOX_MAX = 4

PLACEMENT_RETRIES = 250
GENERATION_RETRIES = 300


def _randint(low: int, high: int) -> int:
    return int(np.random.randint(low, high))


def _expanded_bbox_cells(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    cells = []
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.append((r, c))
    return cells


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return np.array(normalized, dtype=int)


def _shape_bbox(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _make_sprite_shape(inner_h: int, inner_w: int) -> np.ndarray:
    max_box_h = min(SPRITE_BOX_MAX, inner_h)
    max_box_w = min(SPRITE_BOX_MAX, inner_w)
    for _ in range(100):
        box_h = _randint(2, max_box_h + 1)
        box_w = _randint(2, max_box_w + 1)
        area = _randint(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], box_h * box_w) + 1)

        start = (_randint(0, box_h), _randint(0, box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in sorted(cells):
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[_randint(0, len(frontier))])
        if len(cells) != area:
            continue

        shape = _normalize_shape(cells)
        bbox_h, bbox_w = _shape_bbox(shape)
        if bbox_h > inner_h or bbox_w > inner_w:
            continue

        # Favor nontrivial polyominoes, but still allow occasional line shapes.
        if (bbox_h == 1 or bbox_w == 1) and np.random.random() < 0.8:
            continue
        if area == bbox_h * bbox_w and area >= 4 and np.random.random() < 0.7:
            continue
        return shape
    raise RuntimeError("failed to generate sprite shape")


def _gap_cell(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return (r0, _randint(c0 + 1, c1))
    if direction == "down":
        return (r1, _randint(c0 + 1, c1))
    if direction == "left":
        return (_randint(r0 + 1, r1), c0)
    if direction == "right":
        return (_randint(r0 + 1, r1), c1)
    raise AssertionError(f"unknown direction: {direction}")


def _sprite_absolute(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    return shape + np.array([top, left], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)
    bg = GEN_BG_CHOICES[_randint(0, len(GEN_BG_CHOICES))]
    palette = [c for c in range(10) if c not in {bg, GEN_SPRITE_COLOR}]
    object_count = _randint(*OBJECT_COUNT_RANGE)
    if object_count > len(palette):
        return None

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    blocked_output = np.zeros((h, w), dtype=bool)
    wall_colors = list(np.random.permutation(palette)[:object_count])

    for wall_color in wall_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            direction = GEN_DIRECTIONS[_randint(0, len(GEN_DIRECTIONS))]
            outer_h = _randint(*OUTER_H_RANGE)
            outer_w = _randint(*OUTER_W_RANGE)
            inner_h = outer_h - 2
            inner_w = outer_w - 2

            min_r0 = BOUNDARY_MARGIN
            max_r0 = h - BOUNDARY_MARGIN - outer_h
            min_c0 = BOUNDARY_MARGIN
            max_c0 = w - BOUNDARY_MARGIN - outer_w
            if min_r0 > max_r0 or min_c0 > max_c0:
                return None

            shape = _make_sprite_shape(inner_h, inner_w)
            shape_h, shape_w = _shape_bbox(shape)

            sprite_top = _randint(0, inner_h - shape_h + 1)
            sprite_left = _randint(0, inner_w - shape_w + 1)
            r0 = _randint(min_r0, max_r0 + 1)
            c0 = _randint(min_c0, max_c0 + 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            wall_bbox = (r0, c0, r1, c1)
            wall_pad = _expanded_bbox_cells(wall_bbox, (h, w), pad=1)
            if any(reserved[r, c] for r, c in wall_pad):
                continue

            sprite_abs = _sprite_absolute(shape, r0 + 1 + sprite_top, c0 + 1 + sprite_left)
            final_abs = _push_to_edge(sprite_abs, direction, (h, w))
            final_bbox = _bbox(final_abs)
            final_pad = _expanded_bbox_cells(final_bbox, (h, w), pad=1)
            if any(reserved[r, c] for r, c in final_pad):
                continue

            wall_cells = _border_cells(wall_bbox)
            gap = _gap_cell(wall_bbox, direction)
            wall_cells.remove(gap)
            final_set = {tuple(map(int, cell)) for cell in final_abs}
            if final_set & wall_cells:
                continue
            if any(blocked_output[r, c] for r, c in final_set):
                continue

            grid[[r for r, _ in wall_cells], [c for _, c in wall_cells]] = int(wall_color)
            grid[sprite_abs[:, 0], sprite_abs[:, 1]] = GEN_SPRITE_COLOR

            for r, c in wall_pad:
                reserved[r, c] = True
            for r, c in final_pad:
                reserved[r, c] = True
            for r, c in wall_cells:
                blocked_output[r, c] = True
            for r, c in final_set:
                blocked_output[r, c] = True

            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    # Every wall should produce at least one recolored boundary pixel.
    if np.count_nonzero(out == GEN_SPRITE_COLOR) != 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
