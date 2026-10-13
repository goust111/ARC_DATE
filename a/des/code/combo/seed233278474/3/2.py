from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = N4 if connectivity == 4 else [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _rotate_once_cw(
    coords: tuple[tuple[int, int], ...], height: int, width: int
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rot = tuple(sorted((c, height - 1 - r) for r, c in coords))
    return rot, int(width), int(height)


def _rotate_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], turns_cw: int
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    cur, height, width = _normalize_coords(coords)
    for _ in range(turns_cw % 4):
        cur, height, width = _rotate_once_cw(cur, height, width)
    return cur, height, width


def _rotation_count(coords: tuple[tuple[int, int], ...]) -> int:
    return len({_rotate_coords(coords, k)[0] for k in range(4)})


def _template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    blue_components = _components(grid == BLUE, connectivity=4)
    assert blue_components, "expected a blue template"
    template = max(blue_components, key=len)
    r0, c0, r1, c1 = _bbox(template)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub[0, :] == BLUE)
    assert np.all(sub[-1, :] == BLUE)
    assert np.all(sub[:, 0] == BLUE)
    assert np.all(sub[:, -1] == BLUE)
    return int(r0), int(c0), int(r1), int(c1)


def _template_holes(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for pts in _components(sub == BLACK, connectivity=4):
        if any(rr in (0, sub.shape[0] - 1) or cc in (0, sub.shape[1] - 1) for rr, cc in pts):
            continue
        shape, _, _ = _normalize_coords(pts)
        abs_coords = tuple(sorted((r0 + rr, c0 + cc) for rr, cc in pts))
        holes.append({"shape": shape, "abs_coords": abs_coords})
    assert holes, "expected at least one hole in the template"
    return holes


def _marker_rotation(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return 0
    if mc == c1 + 1 and r0 <= mr <= r1:
        return 1
    if mr == r1 + 1 and c0 <= mc <= c1:
        return 2
    if mc == c0 - 1 and r0 <= mr <= r1:
        return 3
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered red fragment according to its green side marker and fit it into the
    matching black hole inside the central blue template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    template_bbox = _template_bbox(g)
    holes = _template_holes(g, template_bbox)
    hole_lookup: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for idx, hole in enumerate(holes):
        hole_lookup.setdefault(hole["shape"], []).append(idx)

    piece_components = _components(g == RED, connectivity=4)
    markers = [tuple(map(int, pos)) for pos in np.argwhere(g == GREEN)]
    assert piece_components, "expected scattered red pieces"
    assert len(piece_components) == len(holes) == len(markers), "pieces, holes, and markers must align"

    out = np.where(g == BLUE, BLUE, BLACK)
    used_markers: set[tuple[int, int]] = set()
    used_holes: set[int] = set()

    for piece in piece_components:
        piece_bbox = _bbox(piece)
        candidates = []
        for marker in markers:
            rot = _marker_rotation(marker, piece_bbox)
            if rot is not None:
                candidates.append((marker, rot))
        assert len(candidates) == 1, "each piece must have exactly one side marker"

        marker, turns_cw = candidates[0]
        assert marker not in used_markers, "markers must pair uniquely"
        used_markers.add(marker)

        rotated_shape, _, _ = _rotate_coords(piece, turns_cw)
        hole_candidates = [idx for idx in hole_lookup.get(rotated_shape, []) if idx not in used_holes]
        assert len(hole_candidates) == 1, "rotated piece must match exactly one remaining hole"

        hole_idx = hole_candidates[0]
        used_holes.add(hole_idx)
        for r, c in holes[hole_idx]["abs_coords"]:
            out[r, c] = RED

    assert len(used_markers) == len(markers)
    assert len(used_holes) == len(holes)
    return out


# --------------- generator ---------------

GEN_RETRIES = 500
GRID_MARGIN = 3
GRID_EXTRA_MIN = 8
GRID_EXTRA_MAX = 14
TEMPLATE_SIDE_MIN = 8
TEMPLATE_SIDE_MAX = 13
HOLE_COUNT_MIN = 2
HOLE_COUNT_MAX = 4
HOLE_AREA_MIN = 3
HOLE_AREA_MAX = 6
HOLE_BBOX_MAX = 4
PLACEMENT_PAD = 1
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 300


def _random_shape() -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_RETRIES):
        target_area = random.randint(HOLE_AREA_MIN, HOLE_AREA_MAX)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in N4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(random.choice(frontier))

        shape, height, width = _normalize_coords(tuple(cells))
        if height == 1 or width == 1:
            continue
        if height > HOLE_BBOX_MAX or width > HOLE_BBOX_MAX:
            continue
        if height * width == len(shape):
            continue
        if _rotation_count(shape) < 4:
            continue
        shape, _, _ = _rotate_coords(shape, random.randrange(4))
        return shape
    return None


def _can_place_cells(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _centered_choice(values: list[int], span: int) -> int:
    center = (span - 1) / 2.0
    best_dist = min(abs(v - center) for v in values)
    options = [v for v in values if abs(v - center) == best_dist]
    return int(random.choice(options))


def _piece_patch(
    piece_shape: tuple[tuple[int, int], ...], direction: int
) -> tuple[int, int, list[tuple[int, int, int]]]:
    shape, height, width = _normalize_coords(piece_shape)
    if direction == 0:
        marker_col = _centered_choice([c for r, c in shape if r == 0], width)
        cells = [(r + 1, c, RED) for r, c in shape] + [(0, marker_col, GREEN)]
        return height + 1, width, cells
    if direction == 1:
        marker_row = _centered_choice([r for r, c in shape if c == width - 1], height)
        cells = [(r, c, RED) for r, c in shape] + [(marker_row, width, GREEN)]
        return height, width + 1, cells
    if direction == 2:
        marker_col = _centered_choice([c for r, c in shape if r == height - 1], width)
        cells = [(r, c, RED) for r, c in shape] + [(height, marker_col, GREEN)]
        return height + 1, width, cells
    marker_row = _centered_choice([r for r, c in shape if c == 0], height)
    cells = [(r, c + 1, RED) for r, c in shape] + [(marker_row, 0, GREEN)]
    return height, width + 1, cells


def _place_holes(template_h: int, template_w: int, shapes: list[tuple[tuple[int, int], ...]]) -> list[list[tuple[int, int]]] | None:
    occupied = np.zeros((template_h, template_w), dtype=bool)
    placements: list[list[tuple[int, int]] | None] = [None] * len(shapes)

    for idx, shape in sorted(enumerate(shapes), key=lambda item: len(item[1]), reverse=True):
        _, height, width = _normalize_coords(shape)
        candidates = []
        for r0 in range(1, template_h - height):
            for c0 in range(1, template_w - width):
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if _can_place_cells(occupied, cells, pad=PLACEMENT_PAD):
                    candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        _mark_cells(occupied, cells)
        placements[idx] = cells
    return [cells for cells in placements if cells is not None]


def _place_piece_patch(
    grid: np.ndarray, occupied: np.ndarray, patch_h: int, patch_w: int, patch_cells: list[tuple[int, int, int]]
) -> bool:
    h, w = grid.shape
    candidates = []
    abs_coords = [(r, c) for r, c, _ in patch_cells]
    for r0 in range(h - patch_h + 1):
        for c0 in range(w - patch_w + 1):
            cells = [(r0 + dr, c0 + dc) for dr, dc in abs_coords]
            if _can_place_cells(occupied, cells, pad=PLACEMENT_PAD):
                candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    placed = []
    for dr, dc, color in patch_cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = color
        placed.append((rr, cc))
    _mark_cells(occupied, placed)
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.randint(HOLE_COUNT_MIN, HOLE_COUNT_MAX)

    hole_shapes = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(hole_count):
        shape = None
        for _ in range(SHAPE_RETRIES):
            candidate = _random_shape()
            if candidate is not None and candidate not in used_shapes:
                shape = candidate
                break
        if shape is None:
            return None
        used_shapes.add(shape)
        hole_shapes.append(shape)

    directions = [random.randrange(4) for _ in range(hole_count)]
    if hole_count > 1 and len(set(directions)) == 1:
        directions[random.randrange(hole_count)] = (directions[0] + random.randint(1, 3)) % 4
    if all(direction == 0 for direction in directions):
        directions[random.randrange(hole_count)] = random.randint(1, 3)

    template_h = random.randint(TEMPLATE_SIDE_MIN, TEMPLATE_SIDE_MAX)
    template_w = random.randint(TEMPLATE_SIDE_MIN, TEMPLATE_SIDE_MAX)
    hole_cells = _place_holes(template_h, template_w, hole_shapes)
    if hole_cells is None:
        return None

    grid_h = random.randint(template_h + GRID_EXTRA_MIN, template_h + GRID_EXTRA_MAX)
    grid_w = random.randint(template_w + GRID_EXTRA_MIN, template_w + GRID_EXTRA_MAX)
    center_r = (grid_h - template_h) // 2
    center_c = (grid_w - template_w) // 2
    r0 = min(max(GRID_MARGIN, center_r + random.randint(-1, 1)), grid_h - template_h - GRID_MARGIN)
    c0 = min(max(GRID_MARGIN, center_c + random.randint(-1, 1)), grid_w - template_w - GRID_MARGIN)

    output = np.full((grid_h, grid_w), BLACK, dtype=int)
    output[r0 : r0 + template_h, c0 : c0 + template_w] = BLUE

    input_grid = output.copy()
    piece_specs = []
    for shape, cells, direction in zip(hole_shapes, hole_cells, directions, strict=False):
        abs_hole = [(r0 + r, c0 + c) for r, c in cells]
        for rr, cc in abs_hole:
            output[rr, cc] = RED
            input_grid[rr, cc] = BLACK
        piece_shape, _, _ = _rotate_coords(shape, (-direction) % 4)
        piece_specs.append((piece_shape, direction))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[r0 : r0 + template_h, c0 : c0 + template_w] = True

    for piece_shape, direction in piece_specs:
        patch_h, patch_w, patch_cells = _piece_patch(piece_shape, direction)
        if not _place_piece_patch(input_grid, occupied, patch_h, patch_w, patch_cells):
            return None

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
