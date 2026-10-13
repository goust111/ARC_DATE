import random
from collections import deque

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 5
ARROW_COLOR = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TURN_TO_VECTOR = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
VECTOR_TO_TURN = {value: key for key, value in TURN_TO_VECTOR.items()}

MIN_PIECES = 2
MAX_PIECES = 4
POLY_AREA_RANGE = (4, 7)
POLY_BBOX_MAX = 4
GRID_SIZE_RANGE = (20, 30)
TEMPLATE_INTERIOR_RANGE = (8, 15)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    min_r = int(arr[:, 0].min())
    min_c = int(arr[:, 1].min())
    shifted = arr - np.array([min_r, min_c], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True
    return mask


def _rotate_mask(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(mask, -turns_cw)


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(np.argwhere(mask))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_shape_signature(_rotate_mask(mask, turns)) for turns in range(4))


def _piece_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    mask[rows - min_r, cols - min_c] = True
    return mask


def _boundary_cells(mask: np.ndarray, turn: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if turn == 0:
        edge = int(coords[:, 0].min())
        return [(int(r), int(c)) for r, c in coords if int(r) == edge]
    if turn == 1:
        edge = int(coords[:, 1].max())
        return [(int(r), int(c)) for r, c in coords if int(c) == edge]
    if turn == 2:
        edge = int(coords[:, 0].max())
        return [(int(r), int(c)) for r, c in coords if int(r) == edge]
    edge = int(coords[:, 1].min())
    return [(int(r), int(c)) for r, c in coords if int(c) == edge]


def _compose_piece_patch(mask: np.ndarray, color: int, turn: int, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    dr, dc = TURN_TO_VECTOR[turn]
    anchor_r, anchor_c = rng.choice(_boundary_cells(mask, turn))

    if turn == 0:  # up
        patch = np.zeros((h + 3, w), dtype=int)
        piece_r0, piece_c0 = 3, 0
    elif turn == 1:  # right
        patch = np.zeros((h, w + 3), dtype=int)
        piece_r0, piece_c0 = 0, 0
    elif turn == 2:  # down
        patch = np.zeros((h + 3, w), dtype=int)
        piece_r0, piece_c0 = 0, 0
    else:  # left
        patch = np.zeros((h, w + 3), dtype=int)
        piece_r0, piece_c0 = 0, 3

    patch[piece_r0 : piece_r0 + h, piece_c0 : piece_c0 + w][mask] = color
    base_r = piece_r0 + anchor_r
    base_c = piece_c0 + anchor_c
    for step in range(1, 4):
        patch[base_r + step * dr, base_c + step * dc] = ARROW_COLOR
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0 - padding
        or br1 + padding < ar0 - padding
        or ac1 + padding < bc0 - padding
        or bc1 + padding < ac0 - padding
    )


def _random_polyomino(rng: random.Random) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = rng.randint(*POLY_AREA_RANGE)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            candidates = []
            for nxt in frontier:
                test = cells | {nxt}
                rows = [r for r, _ in test]
                cols = [c for _, c in test]
                height = max(rows) - min(rows) + 1
                width = max(cols) - min(cols) + 1
                if height <= POLY_BBOX_MAX and width <= POLY_BBOX_MAX:
                    candidates.append(nxt)
            if not candidates:
                break
            cells.add(rng.choice(candidates))

        if len(cells) != target_area:
            continue
        mask = _mask_from_cells(cells)
        if min(mask.shape) < 2:
            continue
        signatures = {_shape_signature(_rotate_mask(mask, turns)) for turns in range(4)}
        if len(signatures) != 4:
            continue
        return mask
    return None


def _generate_shapes(rng: random.Random, count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(count):
        for _ in range(SHAPE_ATTEMPTS):
            shape = _random_polyomino(rng)
            if shape is None:
                continue
            sig = _canonical_signature(shape)
            if sig in signatures:
                continue
            signatures.add(sig)
            shapes.append(shape)
            break
        else:
            return None
    return shapes


def _place_holes(
    rng: random.Random,
    interior_h: int,
    interior_w: int,
    shapes: list[np.ndarray],
) -> list[tuple[int, int, np.ndarray]] | None:
    occupied_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int, np.ndarray]] = []
    for shape in sorted(shapes, key=lambda m: m.shape[0] * m.shape[1], reverse=True):
        h, w = shape.shape
        candidates: list[tuple[int, int]] = []
        for r in range(interior_h - h + 1):
            for c in range(interior_w - w + 1):
                box = (r, c, r + h - 1, c + w - 1)
                if any(_boxes_overlap(box, other, padding=1) for other in occupied_boxes):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placements.append((top, left, shape))
        occupied_boxes.append((top, left, top + h - 1, left + w - 1))
    return placements


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    patch: np.ndarray,
    reserved_boxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            box = (r, c, r + ph - 1, c + pw - 1)
            if any(_boxes_overlap(box, other, padding=1) for other in reserved_boxes):
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    top, left = rng.choice(candidates)
    grid[top : top + ph, left : left + pw][patch != BACKGROUND] = patch[patch != BACKGROUND]
    reserved_boxes.append((top, left, top + ph - 1, left + pw - 1))
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered colored fragments into the matching holes inside the grey template.

    Input structure:
    - The background is black.
    - A large grey template contains one or more black internal holes.
    - Elsewhere, each colored fragment is a single monochrome 4-connected shape.
    - Every fragment has one adjacent maroon 3-pixel straight arrow whose base touches the fragment and whose tip points up, right, down, or left.

    Rule:
    - Find every black hole fully enclosed by grey pixels.
    - For each colored fragment, read the adjacent maroon arrow direction: up means 0 degrees, right means 90 degrees clockwise, down means 180 degrees, and left means 90 degrees counterclockwise.
    - Rotate the fragment by that amount and place it into the unique hole with the same shape.
    - Remove the fragment and the arrow from their original positions.
    - Keep the grey template and the remaining black background unchanged.

    Key insight:
    - The maroon marker does not identify a destination directly; it tells how to rotate a fragment before exact shape matching against the grey template's holes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    hole_by_shape: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for comp in _components(g == BACKGROUND):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        comp_set = {tuple(map(int, coord)) for coord in comp}
        boundary_colors = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if (nr, nc) not in comp_set:
                    boundary_colors.add(int(g[nr, nc]))
        if boundary_colors == {TEMPLATE_COLOR}:
            sig = _normalize_coords(comp)
            assert sig not in hole_by_shape, "hole shapes must be unique"
            hole_by_shape[sig] = comp

    assert hole_by_shape, "expected at least one enclosed template hole"

    pieces = []
    cell_to_piece: dict[tuple[int, int], int] = {}
    for color in np.unique(g):
        color = int(color)
        if color in {BACKGROUND, TEMPLATE_COLOR, ARROW_COLOR}:
            continue
        for comp in _components(g == color):
            idx = len(pieces)
            coords = np.asarray(comp, dtype=int)
            pieces.append({"color": color, "coords": coords})
            for r, c in coords:
                cell_to_piece[(int(r), int(c))] = idx
            out[coords[:, 0], coords[:, 1]] = BACKGROUND

    assert len(pieces) == len(hole_by_shape), "piece count must match hole count"

    arrow_turn_for_piece: dict[int, int] = {}
    for arrow in _components(g == ARROW_COLOR):
        coords = [tuple(map(int, coord)) for coord in arrow]
        rows = sorted(r for r, _ in coords)
        cols = sorted(c for _, c in coords)
        same_row = rows[0] == rows[-1] and cols == list(range(cols[0], cols[0] + 3))
        same_col = cols[0] == cols[-1] and rows == list(range(rows[0], rows[0] + 3))
        assert same_row or same_col, "arrow must be a 3-pixel straight line"

        touching_pieces = set()
        base_cells = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                idx = cell_to_piece.get((r + dr, c + dc))
                if idx is not None:
                    touching_pieces.add(idx)
                    base_cells.add((r, c))
        assert len(touching_pieces) == 1, "each arrow must touch exactly one piece"
        assert len(base_cells) == 1, "each arrow must have a unique base next to its piece"
        piece_idx = next(iter(touching_pieces))
        base_r, base_c = next(iter(base_cells))

        arrow_set = set(coords)
        next_cells = [
            (r, c)
            for r, c in arrow_set
            if (r, c) != (base_r, base_c) and abs(r - base_r) + abs(c - base_c) == 1
        ]
        assert len(next_cells) == 1
        dr = next_cells[0][0] - base_r
        dc = next_cells[0][1] - base_c
        assert (base_r + 2 * dr, base_c + 2 * dc) in arrow_set, "arrow cells must extend away from the piece"
        turn = VECTOR_TO_TURN[(dr, dc)]
        assert piece_idx not in arrow_turn_for_piece, "piece cannot have multiple arrows"
        arrow_turn_for_piece[piece_idx] = turn

        arrow_arr = np.asarray(coords, dtype=int)
        out[arrow_arr[:, 0], arrow_arr[:, 1]] = BACKGROUND

    assert len(arrow_turn_for_piece) == len(pieces), "every piece must have one arrow"

    used_shapes = set()
    for idx, piece in enumerate(pieces):
        mask = _piece_mask_from_coords(piece["coords"])
        rotated = _rotate_mask(mask, arrow_turn_for_piece[idx])
        sig = _shape_signature(rotated)
        assert sig in hole_by_shape, "rotated piece must match a hole"
        assert sig not in used_shapes, "hole matching must be unambiguous"
        used_shapes.add(sig)
        hole = hole_by_shape[sig]
        out[hole[:, 0], hole[:, 1]] = piece["color"]

    assert len(used_shapes) == len(hole_by_shape)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(MIN_PIECES, MAX_PIECES)
    shapes = _generate_shapes(rng, piece_count)
    if shapes is None:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        interior_h = rng.randint(*TEMPLATE_INTERIOR_RANGE)
        interior_w = rng.randint(*TEMPLATE_INTERIOR_RANGE)
        hole_placements = _place_holes(rng, interior_h, interior_w, shapes)
        if hole_placements is None:
            continue

        template_h = interior_h + 2
        template_w = interior_w + 2
        min_h = max(GRID_SIZE_RANGE[0], template_h + 6)
        min_w = max(GRID_SIZE_RANGE[0], template_w + 6)
        if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
            continue
        grid_h = rng.randint(min_h, GRID_SIZE_RANGE[1])
        grid_w = rng.randint(min_w, GRID_SIZE_RANGE[1])
        template_r0 = rng.randint(1, grid_h - template_h - 1)
        template_c0 = rng.randint(1, grid_w - template_w - 1)

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = TEMPLATE_COLOR
        for top, left, shape in hole_placements:
            rows, cols = np.where(shape)
            grid[
                template_r0 + 1 + top + rows,
                template_c0 + 1 + left + cols,
            ] = BACKGROUND

        reserved_boxes = [(template_r0, template_c0, template_r0 + template_h - 1, template_c0 + template_w - 1)]
        colors = rng.sample(PIECE_COLORS, piece_count)
        order = list(range(piece_count))
        rng.shuffle(order)
        success = True
        for idx in order:
            turn = rng.randrange(4)
            piece_mask = _rotate_mask(shapes[idx], -turn)
            patch = _compose_piece_patch(piece_mask, colors[idx], turn, rng)
            if not _place_patch(rng, grid, patch, reserved_boxes):
                success = False
                break
        if not success:
            continue

        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
