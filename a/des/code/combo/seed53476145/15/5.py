from __future__ import annotations

import numpy as np


BG_COLOR = 0
LINE_COLOR = 5
PALETTE = tuple(color for color in range(10) if color not in {BG_COLOR, LINE_COLOR})
ORTHO_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHO_STEPS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _extract_sprites(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    sprite_mask = (grid != BG_COLOR) & (grid != LINE_COLOR)
    components = _components_from_mask(sprite_mask)
    sprites: list[dict] = []
    coord_to_sprite: dict[tuple[int, int], int] = {}

    for idx, coords in enumerate(components):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each sprite must be monochrome"
        mask, bbox = _tight_mask(coords)
        sprite = {
            "coords": coords,
            "bbox": bbox,
            "origin": (bbox[0], bbox[1]),
            "mask": mask,
            "color": int(next(iter(colors))),
        }
        sprites.append(sprite)
        for coord in coords:
            coord_to_sprite[coord] = idx

    return sprites, coord_to_sprite


def _extract_pairs(grid: np.ndarray, sprites: list[dict], coord_to_sprite: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    line_components = _components_from_mask(grid == LINE_COLOR)
    pairs: list[tuple[int, int]] = []
    used_sprites: set[int] = set()
    height, width = grid.shape

    for coords in line_components:
        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        assert len(rows) == 1 or len(cols) == 1, "connector lines must be straight horizontal or vertical segments"

        adjacent_sprites: set[int] = set()
        for r, c in coords:
            for dr, dc in ORTHO_STEPS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if int(grid[nr, nc]) in {BG_COLOR, LINE_COLOR}:
                    continue
                sprite_id = coord_to_sprite.get((nr, nc))
                assert sprite_id is not None, "connector endpoints must touch sprites"
                adjacent_sprites.add(sprite_id)

        assert len(adjacent_sprites) == 2, "each connector must join exactly two sprites"
        a, b = sorted(adjacent_sprites)
        assert a not in used_sprites and b not in used_sprites, "each sprite may belong to at most one pair"
        assert int(sprites[a]["color"]) != int(sprites[b]["color"]), "paired sprites must have different colors"
        used_sprites.add(a)
        used_sprites.add(b)
        pairs.append((a, b))

    return pairs


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    height, width = out.shape
    mask_h, mask_w = mask.shape
    assert 0 <= top and 0 <= left and top + mask_h <= height and left + mask_w <= width
    rows, cols = np.nonzero(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist()):
        r = top + int(dr)
        c = left + int(dc)
        assert int(out[r, c]) in {BG_COLOR, color}, "swapped shapes must not overlap unrelated content"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the shapes of sprite pairs connected by grey lines while preserving each sprite's color and position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites, coord_to_sprite = _extract_sprites(g)
    pairs = _extract_pairs(g, sprites, coord_to_sprite)

    out = g.copy()
    out[out == LINE_COLOR] = BG_COLOR

    paired_ids = {idx for pair in pairs for idx in pair}
    for idx in paired_ids:
        for r, c in sprites[idx]["coords"]:
            out[r, c] = BG_COLOR

    for a, b in pairs:
        a_top, a_left = sprites[a]["origin"]
        b_top, b_left = sprites[b]["origin"]
        _paint_mask(out, sprites[b]["mask"], a_top, a_left, int(sprites[a]["color"]))
        _paint_mask(out, sprites[a]["mask"], b_top, b_left, int(sprites[b]["color"]))

    return out


# --------------- generator ---------------

GRID_MIN_SIZE = 20
GRID_MAX_SIZE = 30
PAIR_COUNT_CHOICES = (1, 2, 3)
UNPAIRED_COUNT_CHOICES = (0, 1, 2, 3)
SHAPE_SIDE_MAX = 4
LINE_LENGTH_MIN = 1
LINE_LENGTH_MAX = 6
PLACE_RETRIES = 180
GENERATION_ATTEMPTS = 400


def _normalize_mask_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    height = max(r for r, _c in norm) + 1
    width = max(c for _r, c in norm) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    box_h = int(rng.integers(1, SHAPE_SIDE_MAX + 1))
    box_w = int(rng.integers(1, SHAPE_SIDE_MAX + 1))
    max_area = box_h * box_w
    min_area = 1 if max_area <= 2 else 2
    target_area = int(rng.integers(min_area, max_area + 1))

    start = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
    cells = {start}
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ORTHO_STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier, "connected growth frontier should never be empty before the box is full"
        candidates = tuple(frontier)
        cells.add(candidates[int(rng.integers(0, len(candidates)))])

    mask = _normalize_mask_cells(cells)
    if bool(rng.integers(0, 2)):
        mask = np.flipud(mask)
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask)
    rotations = int(rng.integers(0, 4))
    return np.rot90(mask, rotations).copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    rows, cols = np.nonzero(mask)
    coords = tuple(sorted((int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())))
    return mask.shape, coords


def _side_port(mask: np.ndarray, side: str, rng: np.random.Generator) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    coords = [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]
    if side == "left":
        edge = min(c for _r, c in coords)
        options = [coord for coord in coords if coord[1] == edge]
    elif side == "right":
        edge = max(c for _r, c in coords)
        options = [coord for coord in coords if coord[1] == edge]
    elif side == "top":
        edge = min(r for r, _c in coords)
        options = [coord for coord in coords if coord[0] == edge]
    else:
        assert side == "bottom"
        edge = max(r for r, _c in coords)
        options = [coord for coord in coords if coord[0] == edge]
    return options[int(rng.integers(0, len(options)))]


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    return {(top + int(r), left + int(c)) for r, c in zip(rows.tolist(), cols.tolist())}


def _halo(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    halo = set(cells)
    for r, c in list(cells):
        for dr, dc in ORTHO_STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                halo.add((nr, nc))
    return halo


def _place_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        assert int(grid[r, c]) == BG_COLOR
        grid[r, c] = int(color)


def _try_place_pair(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    height, width = grid.shape

    for _ in range(PLACE_RETRIES):
        mask_a = _random_connected_mask(rng)
        mask_b = _random_connected_mask(rng)
        if _shape_signature(mask_a) == _shape_signature(mask_b):
            continue

        color_a, color_b = [int(x) for x in rng.choice(PALETTE, size=2, replace=False)]
        line_len = int(rng.integers(LINE_LENGTH_MIN, LINE_LENGTH_MAX + 1))
        horizontal = bool(rng.integers(0, 2))

        h_a, w_a = mask_a.shape
        h_b, w_b = mask_b.shape

        if horizontal:
            port_a = _side_port(mask_a, "right", rng)
            port_b = _side_port(mask_b, "left", rng)
            max_left = width - (w_a + line_len + w_b)
            row_low = max(0, port_b[0] - port_a[0])
            row_high = min(height - h_a, height - h_b + port_b[0] - port_a[0])
            if max_left < 0 or row_low > row_high:
                continue
            top_a = int(rng.integers(row_low, row_high + 1))
            left_a = int(rng.integers(0, max_left + 1))
            top_b = top_a + port_a[0] - port_b[0]
            left_b = left_a + w_a + line_len
            line_row = top_a + port_a[0]
            line_cells = {(line_row, col) for col in range(left_a + w_a, left_b)}
        else:
            port_a = _side_port(mask_a, "bottom", rng)
            port_b = _side_port(mask_b, "top", rng)
            max_top = height - (h_a + line_len + h_b)
            col_low = max(0, port_b[1] - port_a[1])
            col_high = min(width - w_a, width - w_b + port_b[1] - port_a[1])
            if max_top < 0 or col_low > col_high:
                continue
            top_a = int(rng.integers(0, max_top + 1))
            left_a = int(rng.integers(col_low, col_high + 1))
            top_b = top_a + h_a + line_len
            left_b = left_a + port_a[1] - port_b[1]
            line_col = left_a + port_a[1]
            line_cells = {(row, line_col) for row in range(top_a + h_a, top_b)}

        sprite_a = _mask_cells(mask_a, top_a, left_a)
        sprite_b = _mask_cells(mask_b, top_b, left_b)
        swapped_at_a = _mask_cells(mask_b, top_a, left_a)
        swapped_at_b = _mask_cells(mask_a, top_b, left_b)

        visible_cells = sprite_a | sprite_b | line_cells
        if len(visible_cells) != len(sprite_a) + len(sprite_b) + len(line_cells):
            continue
        if not swapped_at_a.isdisjoint(sprite_b):
            continue
        if not swapped_at_b.isdisjoint(sprite_a):
            continue
        if swapped_at_a & swapped_at_b:
            continue

        reserve_cells = _halo(visible_cells | swapped_at_a | swapped_at_b, height, width)
        if not reserve_cells.isdisjoint(reserved):
            continue

        _place_cells(grid, sprite_a, color_a)
        _place_cells(grid, sprite_b, color_b)
        _place_cells(grid, line_cells, LINE_COLOR)
        reserved.update(reserve_cells)
        return True

    return False


def _try_place_unpaired(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    height, width = grid.shape

    for _ in range(PLACE_RETRIES):
        mask = _random_connected_mask(rng)
        mask_h, mask_w = mask.shape
        if mask_h > height or mask_w > width:
            continue
        top = int(rng.integers(0, height - mask_h + 1))
        left = int(rng.integers(0, width - mask_w + 1))
        cells = _mask_cells(mask, top, left)
        reserve_cells = _halo(cells, height, width)
        if not reserve_cells.isdisjoint(reserved):
            continue
        color = int(rng.choice(PALETTE))
        _place_cells(grid, cells, color)
        reserved.update(reserve_cells)
        return True

    return False


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    unpaired_count = int(rng.choice(UNPAIRED_COUNT_CHOICES))
    min_side = max(GRID_MIN_SIZE, 14 + 3 * pair_count + 2 * unpaired_count)
    if min_side > GRID_MAX_SIZE:
        return None

    height = int(rng.integers(min_side, GRID_MAX_SIZE + 1))
    width = int(rng.integers(min_side, GRID_MAX_SIZE + 1))
    grid = np.full((height, width), BG_COLOR, dtype=int)
    reserved: set[tuple[int, int]] = set()

    for _ in range(pair_count):
        if not _try_place_pair(grid, reserved, rng):
            return None
    for _ in range(unpaired_count):
        if not _try_place_unpaired(grid, reserved, rng):
            return None

    if not np.any(grid == LINE_COLOR):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
