import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
VISIBLE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
HOLE_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _red_object_and_pieces(grid: np.ndarray):
    comps = _component_coords(grid != BLACK)
    assert comps, "expected at least one foreground component"
    red_candidates = []
    pieces = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        rec = {"coords": coords, "bbox": _bbox(coords)}
        if colors == {RED}:
            red_candidates.append(rec)
        else:
            pieces.append(rec)
    assert len(red_candidates) == 1, "expected exactly one pure-red object"
    return red_candidates[0], pieces


def _extract_holes(crop: np.ndarray):
    holes = []
    for coords in _component_coords(crop == BLACK):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == crop.shape[1] - 1):
            continue
        box = _bbox(coords)
        hole_mask = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=bool)
        rel = coords - np.array([[box[0], box[1]]], dtype=int)
        hole_mask[rel[:, 0], rel[:, 1]] = True
        holes.append({"coords": coords, "bbox": box, "mask": hole_mask})
    assert holes, "expected enclosed black holes"
    return holes


def _sort_holes(holes, crop_shape):
    h, w = crop_shape
    if h >= w:
        return sorted(holes, key=lambda hole: (hole["bbox"][0], hole["bbox"][1]))
    return sorted(holes, key=lambda hole: (hole["bbox"][1], hole["bbox"][0]))


def _piece_sprite(grid: np.ndarray, piece):
    return _crop(grid, piece["bbox"])


def _match_piece_to_hole(piece_sprite: np.ndarray, hole_mask: np.ndarray):
    for k in range(4):
        rotated = np.rot90(piece_sprite, k=k)
        red_mask = rotated == RED
        if red_mask.shape == hole_mask.shape and np.array_equal(red_mask, hole_mask):
            return rotated
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the large red object, scan its holes by the object's aspect ratio, and embed the matching rotated pieces.

    Input structure:
    - The background is black.
    - One large pure-red connected object contains several enclosed black hole components.
    - Elsewhere on the background are small multicolor puzzle pieces whose red cells form hole-matching stencils.

    Rule:
    - Crop the pure-red object to its bounding box to obtain the output canvas.
    - If the cropped object is at least as tall as it is wide, order holes by the top-left corner of each hole bbox in row-major order; otherwise use column-major order.
    - For each hole in that order, find the unique unused piece whose red cells, after a possible 0/90/180/270 rotation, exactly match the hole shape.
    - Restore the matched hole cells to red and stamp only the non-red cells of the matched rotated piece onto the cropped canvas.

    Key insight:
    - The object's global aspect ratio chooses the local hole scan order, while the red pixels on each loose piece are only a stencil for rotation-based shape matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_object, pieces = _red_object_and_pieces(g)
    crop = _crop(g, red_object["bbox"])
    holes = _sort_holes(_extract_holes(crop), crop.shape)
    assert len(holes) == len(pieces), "hole/piece count mismatch"

    unused = [_piece_sprite(g, piece) for piece in pieces]
    out = crop.copy()
    for hole in holes:
        matches = []
        for idx, sprite in enumerate(unused):
            if sprite is None:
                continue
            rotated = _match_piece_to_hole(sprite, hole["mask"])
            if rotated is not None:
                matches.append((idx, rotated))
        assert len(matches) == 1, "each hole must have a unique unused matching piece"
        idx, rotated = matches[0]
        unused[idx] = None

        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
        r0, c0, _r1, _c1 = hole["bbox"]
        for rr in range(rotated.shape[0]):
            for cc in range(rotated.shape[1]):
                color = int(rotated[rr, cc])
                if color not in (BLACK, RED):
                    out[r0 + rr, c0 + cc] = color

    return out


def _choose_dimensions():
    if random.random() < 0.5:
        h = random.randint(11, 15)
        w = random.randint(8, min(13, h))
        if w > h:
            w = h
    else:
        w = random.randint(11, 15)
        h = random.randint(8, min(13, w - 1))
        if h >= w:
            h = w - 1
    return h, w


def _pad_bbox(box, pad, shape):
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(shape[0] - 1, r1 + pad),
        min(shape[1] - 1, c1 + pad),
    )


def _boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_holes(obj_h: int, obj_w: int, shapes):
    boxes = []
    placed = []
    for base in shapes:
        ok = False
        for _ in range(120):
            mask = np.rot90(base, k=random.randint(0, 3))
            hh, ww = mask.shape
            if hh > obj_h - 2 or ww > obj_w - 2:
                continue
            r0 = random.randint(1, obj_h - hh - 1)
            c0 = random.randint(1, obj_w - ww - 1)
            box = (r0, c0, r0 + hh - 1, c0 + ww - 1)
            padded = _pad_bbox(box, 1, (obj_h, obj_w))
            if any(_boxes_overlap(padded, other) for other in boxes):
                continue
            boxes.append(padded)
            placed.append({"mask": mask, "bbox": box})
            ok = True
            break
        if not ok:
            return None
    return placed


def _choose_piece_colors(n_cells: int):
    palette = random.sample(VISIBLE_COLORS, k=random.randint(1, min(3, n_cells)))
    return [palette[i % len(palette)] for i in range(n_cells)]


def _make_piece_from_hole(mask: np.ndarray):
    h, w = mask.shape
    complement = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    random.shuffle(complement)
    visible = []
    for r, c in complement:
        if any(mask[nr, nc] for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w):
            visible.append((r, c))
    if not visible:
        return None
    visible_count = random.randint(1, min(2, len(visible)))
    chosen = visible[:visible_count]
    piece = np.zeros((h, w), dtype=int)
    piece[mask] = RED
    colors = _choose_piece_colors(len(chosen))
    for color, (r, c) in zip(colors, chosen):
        piece[r, c] = int(color)
    return piece


def _find_free_position(grid: np.ndarray, ph: int, pw: int):
    h, w = grid.shape
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if np.all(grid[rr0:rr1, cc0:cc1] == BLACK):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(red_canvas: np.ndarray, hole_specs, piece_specs):
    out = red_canvas.copy()
    for hole, piece in zip(hole_specs, piece_specs):
        matched = piece.get("aligned")
        if "external" in piece:
            matched = _match_piece_to_hole(piece["external"], hole["mask"])
            assert matched is not None
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
        r0, c0, _r1, _c1 = hole["bbox"]
        for rr in range(matched.shape[0]):
            for cc in range(matched.shape[1]):
                color = int(matched[rr, cc])
                if color not in (BLACK, RED):
                    out[r0 + rr, c0 + cc] = color
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, required_rotation: bool, piece_specs):
    out = main(inp)
    assert np.array_equal(out, expected), "solver output disagrees with constructed target"

    red_object, pieces = _red_object_and_pieces(inp)
    crop = _crop(inp, red_object["bbox"])
    holes = _sort_holes(_extract_holes(crop), crop.shape)
    sprites = [_piece_sprite(inp, piece) for piece in pieces]
    assert len(holes) == len(sprites) >= 3

    row_order = [hole["bbox"] for hole in sorted(holes, key=lambda h: (h["bbox"][0], h["bbox"][1]))]
    col_order = [hole["bbox"] for hole in sorted(holes, key=lambda h: (h["bbox"][1], h["bbox"][0]))]
    assert row_order != col_order, "hole anchors should distinguish scan orders"

    matched_piece_indices = set()
    for hole in holes:
        piece_matches = []
        for piece_idx, sprite in enumerate(sprites):
            matched = False
            for k in range(4):
                rotated = np.rot90(sprite, k=k)
                if rotated.shape == hole["mask"].shape and np.array_equal(rotated == RED, hole["mask"]):
                    matched = True
                    break
            if matched:
                piece_matches.append(piece_idx)
        assert len(piece_matches) == 1, "each hole must have exactly one matching piece"
        matched_piece_indices.add(piece_matches[0])
    assert len(matched_piece_indices) == len(holes), "piece-hole matching must be bijective"
    if required_rotation:
        assert any(spec["rotation_to_match"] != 0 for spec in piece_specs), "expected at least one non-trivial rotation case"


def _gen_attempt():
    obj_h, obj_w = _choose_dimensions()
    n_holes = random.randint(3, 5)
    chosen_shapes = random.sample(HOLE_SHAPES, k=n_holes)
    placed = _place_holes(obj_h, obj_w, chosen_shapes)
    if placed is None:
        return None

    red_canvas = np.full((obj_h, obj_w), RED, dtype=int)
    hole_specs = []
    piece_specs = []
    for entry in placed:
        mask = entry["mask"]
        r0, c0, r1, c1 = entry["bbox"]
        coords = np.argwhere(mask) + np.array([[r0, c0]], dtype=int)
        red_canvas[coords[:, 0], coords[:, 1]] = BLACK
        hole_specs.append({"bbox": entry["bbox"], "coords": coords, "mask": mask})

        aligned = _make_piece_from_hole(mask)
        if aligned is None:
            return None
        piece_specs.append({"aligned": aligned})

    hole_specs = _sort_holes(hole_specs, red_canvas.shape)
    piece_specs = [piece_specs[[entry["bbox"] for entry in placed].index(hole["bbox"])] for hole in hole_specs]

    row_order = [hole["bbox"] for hole in sorted(hole_specs, key=lambda h: (h["bbox"][0], h["bbox"][1]))]
    col_order = [hole["bbox"] for hole in sorted(hole_specs, key=lambda h: (h["bbox"][1], h["bbox"][0]))]
    if row_order == col_order:
        return None

    grid_h = random.randint(max(obj_h + 4, 18), 26)
    grid_w = random.randint(max(obj_w + 4, 18), 26)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    obj_r = random.randint(1, grid_h - obj_h - 1)
    obj_c = random.randint(1, grid_w - obj_w - 1)
    grid[obj_r : obj_r + obj_h, obj_c : obj_c + obj_w] = red_canvas

    force_rotation = random.random() < 0.7
    rotating_index = random.randrange(n_holes) if force_rotation else None
    for idx, spec in enumerate(piece_specs):
        piece = spec["aligned"]
        k = random.randint(1, 3) if idx == rotating_index else random.randint(0, 3)
        placed_piece = np.rot90(piece, k=k)
        pos = _find_free_position(grid, placed_piece.shape[0], placed_piece.shape[1])
        if pos is None:
            return None
        pr, pc = pos
        patch = placed_piece != BLACK
        grid[pr : pr + placed_piece.shape[0], pc : pc + placed_piece.shape[1]][patch] = placed_piece[patch]
        spec["external"] = placed_piece
        spec["rotation_to_match"] = (-k) % 4

    expected = _expected_output(red_canvas, hole_specs, piece_specs)
    _semantic_checks(grid, expected, force_rotation, piece_specs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
