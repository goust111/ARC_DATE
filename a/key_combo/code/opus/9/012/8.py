from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_task_8", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()
Color = _ARC.Color
bbox = _ARC.bbox
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
paste = _ARC.paste


PIECE_COLORS = [c for c in Color.NOT_BLACK if c != Color.RED]
AREA_CHOICES = (3, 4, 5, 6, 7, 8)
GENERATION_RETRIES = 400
SHAPE_RETRIES = 400
PLACEMENT_RETRIES = 200


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    cells = [tuple(map(int, cell)) for cell in coords]
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    return _normalize_coords(coords)


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    b = bbox(patch != Color.BLACK)
    assert b is not None
    return crop_bbox(patch, b)


def _rotated_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return _crop_nonzero(np.rot90(mask.astype(int), k)).astype(bool)


def _stack_patches(patches: list[np.ndarray]) -> np.ndarray:
    width = max(patch.shape[1] for patch in patches)
    height = sum(patch.shape[0] for patch in patches)
    out = np.full((height, width), Color.BLACK, dtype=int)
    row = 0
    for patch in patches:
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h
    return out


def _shape_has_four_rotations(shape: tuple[tuple[int, int], ...]) -> bool:
    mask = _coords_to_mask(shape)
    rotations = {_shape_signature(np.rot90(mask, k)) for k in range(4)}
    return len(rotations) == 4


def _random_shape(area: int) -> tuple[tuple[int, int], ...]:
    assert area in AREA_CHOICES
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
            frontier -= cells
            nr, nc = random.choice(sorted(frontier))
            cells.add((nr, nc))
            sig = _normalize_coords(cells)
            if max(r for r, _ in sig) >= 4 or max(c for _, c in sig) >= 4:
                break
        sig = _normalize_coords(cells)
        if len(sig) != area:
            continue
        if max(r for r, _ in sig) >= 4 or max(c for _, c in sig) >= 4:
            continue
        if not _shape_has_four_rotations(sig):
            continue
        return sig
    raise AssertionError(f"failed to generate asymmetric shape of area {area}")


def _colorize_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = _coords_to_mask(shape)
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    colors = random.sample(PIECE_COLORS, k=1 if random.random() < 0.35 else 2)
    patch = np.full(mask.shape, Color.BLACK, dtype=int)
    if len(colors) == 1:
        for r, c in coords:
            patch[r, c] = colors[0]
        return patch

    second_count = random.randint(1, len(coords) - 1)
    second_indices = set(random.sample(range(len(coords)), second_count))
    for idx, (r, c) in enumerate(coords):
        patch[r, c] = colors[1] if idx in second_indices else colors[0]
    return patch


def _red_component_and_pieces(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    objects = find_components(
        grid, bg=Color.BLACK, connectivity=8, monochromatic=False, return_format="objects"
    )
    red_components = []
    pieces = []
    for obj in objects:
        colors = set(int(v) for v in np.unique(obj) if int(v) != Color.BLACK)
        if colors == {Color.RED}:
            red_components.append(obj)
        else:
            pieces.append(obj)
    assert len(red_components) == 1, "expected one all-red carrier object"
    assert 2 <= len(pieces) <= 4, "expected 2-4 outside sprites"
    return red_components[0], pieces


def _extract_holes(red_object: np.ndarray) -> list[dict]:
    red_box = bbox(red_object == Color.RED)
    assert red_box is not None
    cropped = crop_bbox(red_object, red_box)
    h, w = cropped.shape
    holes = []
    for comp in components_mask(cropped == Color.BLACK, connectivity=4):
        cells = [tuple(map(int, cell)) for cell in comp]
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        sig = _normalize_coords(cells)
        holes.append(
            {
                "area": len(sig),
                "shape": sig,
                "mask": _coords_to_mask(sig),
            }
        )
    assert 2 <= len(holes) <= 4, "expected 2-4 interior holes"
    assert len({hole["area"] for hole in holes}) == len(holes), "hole areas must be unique"
    assert len({hole["shape"] for hole in holes}) == len(holes), "hole shapes must be unique"
    return holes


def _solve_scene(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_object, piece_objects = _red_component_and_pieces(g)
    holes = _extract_holes(red_object)
    holes_by_shape = {hole["shape"]: idx for idx, hole in enumerate(holes)}

    matches = []
    used_holes = set()
    for piece_object in piece_objects:
        patch = _crop_nonzero(piece_object)
        candidates = []
        for k in range(4):
            rotated = _crop_nonzero(np.rot90(patch, k))
            sig = _shape_signature(rotated != Color.BLACK)
            if sig in holes_by_shape:
                candidates.append((holes_by_shape[sig], rotated, k))
        assert len(candidates) == 1, "each piece must match exactly one hole under one rotation"
        hole_idx, rotated, k = candidates[0]
        assert hole_idx not in used_holes, "hole matching must be one-to-one"
        used_holes.add(hole_idx)
        matches.append(
            {
                "hole_idx": hole_idx,
                "area": holes[hole_idx]["area"],
                "rotated_patch": rotated,
                "rotation": k,
            }
        )

    assert len(matches) == len(holes), "all holes must be matched"
    ordered = sorted(matches, key=lambda item: item["area"])
    out = _stack_patches([item["rotated_patch"] for item in ordered])
    return out, {"holes": holes, "matches": ordered}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside sprite into its matching hole shape, then output the rotated sprites sorted by hole area.

    Input structure:
    - A black background contains one large red connected object with 2-4 enclosed black holes.
    - Separate multicolor sprites lie away from the red object and from each other on the background.
    - Each sprite's occupied shape matches exactly one hole after a 0/90/180/270-degree rotation.

    Rule:
    - Find every interior hole of the red object and record each hole's shape and area.
    - For each outside sprite, test the four right-angle rotations and identify the unique hole whose shape it matches.
    - Rotate the sprite into that matching orientation.
    - Sort the rotated sprites by their matched hole areas from smallest to largest, then stack their cropped bounding boxes top-to-bottom, left-aligned, on a black output canvas whose width is the maximum cropped sprite width.

    Key insight:
    - The final row order is not read from sprite positions or colors; it is determined only after shape matching reveals which hole area each sprite belongs to.
    """
    return _solve_scene(grid)[0]


def _mark_forbidden(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w:
                mask[rr, cc] = True


def _place_holes(
    obj_h: int, obj_w: int, hole_masks: list[np.ndarray]
) -> list[list[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        forbidden = np.zeros((obj_h, obj_w), dtype=bool)
        placements: list[list[tuple[int, int]]] = [None] * len(hole_masks)  # type: ignore[assignment]
        order = sorted(
            range(len(hole_masks)),
            key=lambda idx: hole_masks[idx].shape[0] * hole_masks[idx].shape[1],
            reverse=True,
        )
        ok = True
        for idx in order:
            mask = hole_masks[idx]
            h, w = mask.shape
            candidates = []
            for r0 in range(1, obj_h - h):
                for c0 in range(1, obj_w - w):
                    cells = [
                        (r0 + r, c0 + c)
                        for r, c in np.argwhere(mask)
                    ]
                    if any(forbidden[r, c] for r, c in cells):
                        continue
                    candidates.append(cells)
            if not candidates:
                ok = False
                break
            cells = random.choice(candidates)
            placements[idx] = cells
            _mark_forbidden(forbidden, cells)
        if ok:
            return placements
    return None


def _build_red_object(
    areas: list[int], hole_shapes: list[tuple[tuple[int, int], ...]], hole_rots: list[int]
) -> tuple[np.ndarray, list[np.ndarray]]:
    total_hole_area = sum(areas)
    dims = [
        (h, w)
        for h in range(7, 12)
        for w in range(7, 12)
        if 40 <= h * w - total_hole_area <= 60
    ]
    random.shuffle(dims)
    for obj_h, obj_w in dims:
        rotated_holes = [_rotated_mask(_coords_to_mask(shape), rot) for shape, rot in zip(hole_shapes, hole_rots)]
        placements = _place_holes(obj_h, obj_w, rotated_holes)
        if placements is None:
            continue
        obj = np.full((obj_h, obj_w), Color.RED, dtype=int)
        for cells in placements:
            for r, c in cells:
                obj[r, c] = Color.BLACK
        return obj, rotated_holes
    raise AssertionError("failed to place holes inside red object")


def _piece_variants(
    hole_shapes: list[tuple[tuple[int, int], ...]], hole_rots: list[int], piece_rots: list[int]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    input_pieces = []
    output_pieces = []
    for shape, hole_rot, piece_rot in zip(hole_shapes, hole_rots, piece_rots):
        base_patch = _colorize_shape(shape)
        input_pieces.append(_crop_nonzero(np.rot90(base_patch, piece_rot)))
        output_pieces.append(_crop_nonzero(np.rot90(base_patch, hole_rot)))
    return input_pieces, output_pieces


def _layout_right(red_obj: np.ndarray, pieces: list[np.ndarray], area_order: list[int]) -> np.ndarray | None:
    n = len(pieces)
    perm = random.sample(range(n), k=n)
    if perm == area_order:
        return None
    gap_between_pieces = random.randint(1, 2)
    gap_to_object = random.randint(2, 4)
    strip_h = sum(pieces[idx].shape[0] for idx in perm) + gap_between_pieces * (n - 1)
    max_pw = max(piece.shape[1] for piece in pieces)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    H_min = max(red_obj.shape[0], strip_h) + top_pad + bottom_pad
    W_min = left_pad + red_obj.shape[1] + gap_to_object + max_pw + right_pad
    if H_min > 30 or W_min > 30:
        return None
    H = random.randint(H_min, min(30, H_min + 4))
    W = random.randint(W_min, min(30, W_min + 4))
    grid = np.full((H, W), Color.BLACK, dtype=int)

    obj_r = random.randint(0, H - red_obj.shape[0])
    obj_c = left_pad
    paste(grid, red_obj, obj_r, obj_c, transparent=Color.BLACK)

    start_r = random.randint(0, H - strip_h)
    base_c = obj_c + red_obj.shape[1] + gap_to_object
    row = start_r
    for idx in perm:
        piece = pieces[idx]
        slack = W - right_pad - base_c - piece.shape[1]
        col = base_c if slack <= 0 else random.randint(base_c, base_c + slack)
        paste(grid, piece, row, col, transparent=Color.BLACK)
        row += piece.shape[0] + gap_between_pieces
    return grid


def _layout_bottom(red_obj: np.ndarray, pieces: list[np.ndarray], area_order: list[int]) -> np.ndarray | None:
    n = len(pieces)
    perm = random.sample(range(n), k=n)
    if perm == area_order:
        return None
    gap_between_pieces = random.randint(1, 2)
    gap_to_object = random.randint(2, 4)
    strip_w = sum(pieces[idx].shape[1] for idx in perm) + gap_between_pieces * (n - 1)
    max_ph = max(piece.shape[0] for piece in pieces)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    H_min = top_pad + red_obj.shape[0] + gap_to_object + max_ph + bottom_pad
    W_min = max(left_pad + red_obj.shape[1] + right_pad, left_pad + strip_w + right_pad)
    if H_min > 30 or W_min > 30:
        return None
    H = random.randint(H_min, min(30, H_min + 4))
    W = random.randint(W_min, min(30, W_min + 4))
    grid = np.full((H, W), Color.BLACK, dtype=int)

    obj_r = top_pad
    center_lo = left_pad
    center_hi = W - right_pad - red_obj.shape[1]
    obj_c = random.randint(center_lo, center_hi)
    paste(grid, red_obj, obj_r, obj_c, transparent=Color.BLACK)

    base_r = obj_r + red_obj.shape[0] + gap_to_object
    start_c = random.randint(left_pad, W - right_pad - strip_w)
    col = start_c
    for idx in perm:
        piece = pieces[idx]
        slack = H - bottom_pad - base_r - piece.shape[0]
        row = base_r if slack <= 0 else random.randint(base_r, base_r + slack)
        paste(grid, piece, row, col, transparent=Color.BLACK)
        col += piece.shape[1] + gap_between_pieces
    return grid


def _validate_case(
    grid: np.ndarray,
    expected_output: np.ndarray,
    expected_patches: list[np.ndarray],
    piece_rots: list[int],
    hole_rots: list[int],
) -> bool:
    try:
        solved, meta = _solve_scene(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected_output):
        return False
    matched_patches = [item["rotated_patch"] for item in meta["matches"]]
    if len(matched_patches) != len(expected_patches):
        return False
    if any(not np.array_equal(a, b) for a, b in zip(matched_patches, expected_patches)):
        return False
    if not any((pr - hr) % 4 != 0 for pr, hr in zip(piece_rots, hole_rots)):
        return False
    if max(grid.shape) > 30 or max(expected_output.shape) > 30:
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n = random.randint(2, 4)
    areas = random.sample(AREA_CHOICES, k=n)
    hole_shapes = [_random_shape(area) for area in areas]
    hole_rots = [random.randint(0, 3) for _ in range(n)]
    piece_rots = [random.randint(0, 3) for _ in range(n)]
    if all((pr - hr) % 4 == 0 for pr, hr in zip(piece_rots, hole_rots)):
        idx = random.randrange(n)
        piece_rots[idx] = (hole_rots[idx] + random.randint(1, 3)) % 4

    red_obj, _rotated_holes = _build_red_object(areas, hole_shapes, hole_rots)
    input_pieces, output_pieces = _piece_variants(hole_shapes, hole_rots, piece_rots)
    area_order = [idx for idx, _ in sorted(enumerate(areas), key=lambda item: item[1])]
    expected_patches = [output_pieces[idx] for idx in area_order]
    expected_output = _stack_patches(expected_patches)

    layouts = [_layout_right, _layout_bottom]
    random.shuffle(layouts)
    for layout in layouts:
        grid = layout(red_obj, input_pieces, area_order)
        if grid is None:
            continue
        if _validate_case(grid, expected_output, expected_patches, piece_rots, hole_rots):
            return grid, expected_output
    raise AssertionError("failed to build valid layout")


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            grid, _ = _generate_case()
            return grid
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid input grid")
