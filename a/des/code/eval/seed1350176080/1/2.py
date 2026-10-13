from __future__ import annotations

import numpy as np


BLACK = 0
BOARD_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_RNG = np.random.default_rng()

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (22, 31)
BOARD_H_RANGE = (9, 15)
BOARD_W_RANGE = (10, 18)
HOLE_COUNT_CHOICES = [2, 3, 4]
HOLE_AREA_RANGE = (4, 8)
SHAPE_MAX_SIDE = 5

SHAPE_RETRIES = 80
HOLE_PLACEMENT_RETRIES = 80
PIECE_PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0], dtype=int)
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rot = _tight_mask(np.rot90(mask, k))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = _tight_mask(np.rot90(mask, k))
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _rotation_to_match(src: np.ndarray, dst: np.ndarray) -> int | None:
    for k in range(4):
        if np.array_equal(_tight_mask(np.rot90(src, k)), dst):
            return k
    return None


def _largest_non_black_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid[grid != BLACK], return_counts=True)
    if len(colors) == 0:
        raise AssertionError("expected at least one non-black color")
    return int(colors[int(np.argmax(counts))])


def _extract_board(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    board_color = _largest_non_black_color(grid)
    coords = np.argwhere(grid == board_color)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return board_color, (int(r0), int(c0), int(r1), int(c1))


def _extract_holes(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK
    holes = []
    for coords in _component_coords(sub):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == sub.shape[1] - 1):
            continue
        mask = _coords_to_mask(coords)
        global_coords = coords + np.array([r0, c0], dtype=int)
        holes.append(
            {
                "coords": global_coords,
                "mask": mask,
                "area": int(mask.sum()),
                "key": _canonical_key(mask),
            }
        )
    holes.sort(key=lambda hole: (-hole["area"], tuple(hole["coords"].min(axis=0))))
    return holes


def _extract_pieces(grid: np.ndarray, board_color: int) -> list[dict]:
    piece_mask = (grid != BLACK) & (grid != board_color)
    pieces = []
    for coords in _component_coords(piece_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise AssertionError("each piece must be monochrome")
        color = next(iter(colors))
        mask = _coords_to_mask(coords)
        pieces.append(
            {
                "coords": coords,
                "mask": mask,
                "area": int(mask.sum()),
                "color": color,
                "key": _canonical_key(mask),
            }
        )
    pieces.sort(key=lambda piece: (-piece["area"], int(piece["color"])))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black hole in the grey board with its matching rotated piece.

    Input structure:
    - Black background.
    - One large grey rectangle contains several interior black holes.
    - Outside the rectangle, scattered monochrome pieces match the hole shapes up
      to rotation.

    Rule:
    - Match each piece to the unique hole with the same shape under 90-degree
      rotations.
    - Fill the hole with the piece color.
    - Remove the original outside pieces and keep the grey board unchanged.
    """
    g = np.asarray(grid, dtype=int)
    board_color, board_bbox = _extract_board(g)
    holes = _extract_holes(g, board_bbox)
    pieces = _extract_pieces(g, board_color)

    if len(holes) != len(pieces):
        raise AssertionError("number of holes and pieces must match")

    hole_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for hole in holes:
        if hole["key"] in hole_by_key:
            raise AssertionError("hole shapes must be unique up to rotation")
        hole_by_key[hole["key"]] = hole

    out = g.copy()
    out[(out != BLACK) & (out != board_color)] = BLACK

    for piece in pieces:
        hole = hole_by_key.get(piece["key"])
        if hole is None:
            raise AssertionError("every piece must match a hole")
        if piece["area"] != hole["area"]:
            raise AssertionError("matched piece and hole must have equal area")
        rot = _rotation_to_match(piece["mask"], hole["mask"])
        if rot is None:
            raise AssertionError("piece must match its hole under rotation")
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(piece["color"])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_shape(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    candidate = (r + dr, c + dc)
                    if candidate not in cells:
                        frontier.add(candidate)
            options = []
            for candidate in frontier:
                trial = _normalize_cells(cells | {candidate})
                h = max(r for r, _ in trial) + 1
                w = max(c for _, c in trial) + 1
                if h <= SHAPE_MAX_SIDE and w <= SHAPE_MAX_SIDE:
                    options.append(candidate)
            if not options:
                break
            cells.add(options[int(_RNG.integers(len(options)))])
        if len(cells) != area:
            continue
        mask = _cells_to_mask(cells)
        if min(mask.shape) < 2:
            continue
        if len(_unique_rotations(mask)) != 4:
            continue
        return mask
    return None


def _sample_distinct_shapes(count: int) -> list[np.ndarray] | None:
    shapes = []
    keys = set()
    for _ in range(count * SHAPE_RETRIES):
        area = int(_RNG.integers(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1]))
        shape = _random_shape(area)
        if shape is None:
            continue
        key = _canonical_key(shape)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _touches_or_near(existing: set[tuple[int, int]], coords: list[tuple[int, int]]) -> bool:
    if not existing:
        return False
    for r, c in coords:
        for er, ec in existing:
            if max(abs(r - er), abs(c - ec)) <= 1:
                return True
    return False


def _place_holes(board_h: int, board_w: int, shapes: list[np.ndarray]) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    holes = []
    for shape in shapes:
        h, w = map(int, shape.shape)
        if h >= board_h - 1 or w >= board_w - 1:
            return None
        coords_local = [(int(r), int(c)) for r, c in np.argwhere(shape)]
        placed = False
        for _ in range(HOLE_PLACEMENT_RETRIES):
            top = int(_RNG.integers(1, board_h - h))
            left = int(_RNG.integers(1, board_w - w))
            translated = [(top + r, left + c) for r, c in coords_local]
            if _touches_or_near(occupied, translated):
                continue
            placed = True
            occupied.update(translated)
            holes.append(
                {
                    "mask": shape,
                    "top": top,
                    "left": left,
                    "coords": translated,
                }
            )
            break
        if not placed:
            return None
    return holes


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_pieces(
    grid_h: int,
    grid_w: int,
    board_bbox: tuple[int, int, int, int],
    piece_specs: list[dict],
) -> list[dict] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    br0, bc0, br1, bc1 = board_bbox
    for r in range(max(0, br0 - 1), min(grid_h, br1 + 2)):
        for c in range(max(0, bc0 - 1), min(grid_w, bc1 + 2)):
            blocked[r, c] = True

    placements = []
    for spec in piece_specs:
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        coords_local = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        placed = False
        for _ in range(PIECE_PLACEMENT_RETRIES):
            top = int(_RNG.integers(0, grid_h - h + 1))
            left = int(_RNG.integers(0, grid_w - w + 1))
            translated = [(top + r, left + c) for r, c in coords_local]
            if any(blocked[r, c] for r, c in translated):
                continue
            placed = True
            _mark_blocked(blocked, translated)
            placements.append(
                {
                    "mask": mask,
                    "color": spec["color"],
                    "coords": translated,
                }
            )
            break
        if not placed:
            return None
    return placements


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    grid_w = int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    board_h = int(_RNG.integers(BOARD_H_RANGE[0], min(BOARD_H_RANGE[1], grid_h - 3)))
    board_w = int(_RNG.integers(BOARD_W_RANGE[0], min(BOARD_W_RANGE[1], grid_w - 3)))
    board_top = int(_RNG.integers(1, grid_h - board_h))
    board_left = int(_RNG.integers(1, grid_w - board_w))
    board_bbox = (board_top, board_left, board_top + board_h - 1, board_left + board_w - 1)

    hole_count = int(_RNG.choice(HOLE_COUNT_CHOICES))
    base_shapes = _sample_distinct_shapes(hole_count)
    if base_shapes is None:
        return None

    hole_rotations = [int(_RNG.integers(4)) for _ in range(hole_count)]
    piece_rotations = [int(_RNG.integers(4)) for _ in range(hole_count)]
    if all((hole_rotations[i] - piece_rotations[i]) % 4 == 0 for i in range(hole_count)):
        j = int(_RNG.integers(hole_count))
        piece_rotations[j] = int((hole_rotations[j] + _RNG.integers(1, 4)) % 4)

    colors = list(_RNG.permutation(PIECE_COLORS)[:hole_count])
    hole_masks = [np.rot90(base_shapes[i], hole_rotations[i]) for i in range(hole_count)]
    holes = _place_holes(board_h, board_w, hole_masks)
    if holes is None:
        return None

    piece_specs = [
        {"mask": np.rot90(base_shapes[i], piece_rotations[i]), "color": colors[i]}
        for i in range(hole_count)
    ]
    piece_placements = _place_pieces(grid_h, grid_w, board_bbox, piece_specs)
    if piece_placements is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    expected = np.full((grid_h, grid_w), BLACK, dtype=int)

    br0, bc0, br1, bc1 = board_bbox
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = BOARD_COLOR
    expected[br0 : br1 + 1, bc0 : bc1 + 1] = BOARD_COLOR

    for hole, color in zip(holes, colors):
        for r, c in hole["coords"]:
            grid[board_top + r, board_left + c] = BLACK
            expected[board_top + r, board_left + c] = color

    for placement in piece_placements:
        for r, c in placement["coords"]:
            grid[r, c] = placement["color"]

    if _largest_non_black_color(grid) != BOARD_COLOR:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        example = _build_example()
        if example is not None:
            return example[0]
    raise RuntimeError("failed to generate a valid input")
