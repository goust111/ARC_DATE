from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
PIECE_COLORS = [1, 3, 4, 6, 7, 8, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _tight_crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != 0)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = np.asarray(mask, dtype=bool)
    return tuple(mask.shape), mask.tobytes()


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_tight_crop_nonzero(np.rot90(mask.astype(int), k)).astype(bool)) for k in range(4))


def _rotation_variant_count(mask: np.ndarray) -> int:
    return len({_mask_key(_tight_crop_nonzero(np.rot90(mask.astype(int), k)).astype(bool)) for k in range(4)})


def _rotate_mask_and_red(mask: np.ndarray, red_pos: tuple[int, int], k: int) -> tuple[np.ndarray, tuple[int, int]]:
    marked = np.zeros(mask.shape, dtype=int)
    marked[np.asarray(mask, dtype=bool)] = 1
    marked[int(red_pos[0]), int(red_pos[1])] = 2
    rotated = _tight_crop_nonzero(np.rot90(marked, int(k)))
    red_coords = np.argwhere(rotated == 2)
    assert red_coords.shape == (1, 2)
    red = tuple(int(v) for v in red_coords[0])
    return rotated > 0, red


def _can_place_with_padding(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        canvas[int(top + dr), int(left + dc)] = int(color)


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = canvas[top : top + h, left : left + w]
    nz = patch != BG
    assert target.shape == patch.shape
    assert not bool(np.any((target != BG) & nz))
    target[nz] = patch[nz]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each template hole using the shape-matching outside piece whose red marker
    determines the required rotation.
    """
    g = np.asarray(grid, dtype=int)
    fg_comps = _components(g != BG)
    assert fg_comps, "expected foreground objects"

    template_coords = max(fg_comps, key=lambda comp: int(len(comp)))
    template_mask = np.zeros(g.shape, dtype=bool)
    template_mask[template_coords[:, 0], template_coords[:, 1]] = True
    out = np.full_like(g, BG)
    out[template_mask] = g[template_mask]

    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    sub = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    holes = []
    for comp in _components(sub == BG):
        if any(
            int(r) == 0 or int(c) == 0 or int(r) == sub.shape[0] - 1 or int(c) == sub.shape[1] - 1
            for r, c in comp
        ):
            continue
        hole_mask, (hr0, hc0) = _tight_mask_from_coords(comp)
        red_neighbors: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and int(sub[nr, nc]) == RED:
                    red_neighbors.add((int(tr0 + nr), int(tc0 + nc)))
        assert len(red_neighbors) == 1, "each hole must have exactly one template red marker"
        holes.append(
            {
                "mask": hole_mask,
                "top": (int(tr0 + hr0), int(tc0 + hc0)),
                "red": next(iter(red_neighbors)),
            }
        )

    piece_mask = (g != BG) & (~template_mask)
    pieces = []
    for comp in _components(piece_mask):
        pmask, _ = _tight_mask_from_coords(comp)
        pr0, pc0, pr1, pc1 = _bbox(comp)
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        red_coords = np.argwhere((patch == RED) & pmask)
        assert red_coords.shape == (1, 2), "each piece must have exactly one red marker pixel"
        piece_colors = {int(v) for v in patch[pmask] if int(v) != RED}
        assert len(piece_colors) == 1, "each piece should be monochrome apart from its red marker"
        pieces.append(
            {
                "mask": pmask,
                "red": tuple(int(v) for v in red_coords[0]),
                "color": next(iter(piece_colors)),
            }
        )

    assert len(pieces) == len(holes), "piece count must equal hole count"
    used_holes: set[int] = set()

    for piece in pieces:
        candidates: list[int] = []
        for hole_idx, hole in enumerate(holes):
            for k in range(4):
                rot_mask, rot_red = _rotate_mask_and_red(piece["mask"], piece["red"], k)
                if _mask_key(rot_mask) != _mask_key(hole["mask"]):
                    continue
                red_abs = (int(hole["top"][0] + rot_red[0]), int(hole["top"][1] + rot_red[1]))
                if _manhattan(red_abs, hole["red"]) != 1:
                    continue
                candidates.append(int(hole_idx))
        assert len(candidates) == 1, "each piece must match exactly one hole"
        hole_idx = candidates[0]
        assert hole_idx not in used_holes, "different pieces cannot fill the same hole"
        used_holes.add(hole_idx)
        hole = holes[hole_idx]
        _paste_mask(out, hole["mask"], hole["top"][0], hole["top"][1], int(piece["color"]))

    return out


def _build_shape_library() -> tuple[list[np.ndarray], list[np.ndarray]]:
    raw = [
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [0, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    symmetric: list[np.ndarray] = []
    asymmetric: list[np.ndarray] = []
    for rows in raw:
        mask = np.array(rows, dtype=bool)
        key = _canonical_rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        if _rotation_variant_count(mask) < 4:
            symmetric.append(mask)
        else:
            asymmetric.append(mask)
    return symmetric, asymmetric


_SYM_SHAPES, _ASYM_SHAPES = _build_shape_library()
_ALL_SHAPES = _SYM_SHAPES + _ASYM_SHAPES

GEN_PIECE_COUNT = (2, 4)
GEN_TEMPLATE_H = (11, 18)
GEN_TEMPLATE_W = (11, 18)
GEN_TEMPLATE_BORDER = 1
GEN_HOLE_PAD = 2
GEN_GRID_EXTRA_H = (2, 7)
GEN_RIGHT_GAP = (4, 7)
GEN_RIGHT_EXTRA_W = (4, 10)
GEN_TEMPLATE_LEFT = (1, 3)
GEN_EDGE_MARGIN = (1, 3)
GEN_PIECE_PAD = 1
GEN_CASE_ATTEMPTS = 400
GEN_PLACE_ATTEMPTS = 250
GEN_FORCE_SYM_PROB = 0.5


def _choose_shapes(count: int) -> list[np.ndarray]:
    assert count <= len(_ALL_SHAPES)
    if bool(_SYM_SHAPES) and random.random() < GEN_FORCE_SYM_PROB:
        first = random.choice(_SYM_SHAPES)
        remaining = [shape for shape in _ALL_SHAPES if _canonical_rotation_key(shape) != _canonical_rotation_key(first)]
        extra = random.sample(remaining, count - 1)
        chosen = [first] + extra
    else:
        chosen = random.sample(_ALL_SHAPES, count)
    random.shuffle(chosen)
    return [shape.copy() for shape in chosen]


def _place_holes(template_h: int, template_w: int, masks: list[np.ndarray]) -> list[dict] | None:
    occupied = np.zeros((template_h, template_w), dtype=bool)
    holes: list[dict] = []
    indexed = list(enumerate(masks))
    indexed.sort(key=lambda item: int(item[1].sum()), reverse=True)

    for original_idx, mask in indexed:
        mh, mw = mask.shape
        positions = [
            (r, c)
            for r in range(GEN_TEMPLATE_BORDER, template_h - mh - GEN_TEMPLATE_BORDER + 1)
            for c in range(GEN_TEMPLATE_BORDER, template_w - mw - GEN_TEMPLATE_BORDER + 1)
        ]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            if not _can_place_with_padding(occupied, mask, int(top), int(left), pad=GEN_HOLE_PAD):
                continue
            for dr, dc in np.argwhere(mask):
                occupied[int(top + dr), int(left + dc)] = True
            holes.append({"index": int(original_idx), "mask": mask, "top": (int(top), int(left))})
            placed = True
            break
        if not placed:
            return None

    holes.sort(key=lambda hole: int(hole["index"]))
    return holes


def _local_hole_cells(mask: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    return {(int(top[0] + r), int(top[1] + c)) for r, c in np.argwhere(mask)}


def _marker_candidates(mask: np.ndarray, top: tuple[int, int], all_hole_cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    hole_cells = _local_hole_cells(mask, top)
    for rr, cc in np.argwhere(mask):
        cell = (int(rr), int(cc))
        abs_cell = (int(top[0] + rr), int(top[1] + cc))
        for dr, dc in DIRS4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                continue
            boundary = (int(abs_cell[0] + dr), int(abs_cell[1] + dc))
            adj_hole_count = 0
            for edr, edc in DIRS4:
                neighbor = (int(boundary[0] + edr), int(boundary[1] + edc))
                if neighbor in all_hole_cells:
                    adj_hole_count += 1
            if adj_hole_count != 1:
                continue
            if boundary in hole_cells or boundary in all_hole_cells:
                continue
            candidates.append((cell, boundary))
    random.shuffle(candidates)
    return candidates


def _valid_rotations_for_hole(
    input_mask: np.ndarray,
    input_red: tuple[int, int],
    hole_mask: np.ndarray,
    hole_top: tuple[int, int],
    hole_red: tuple[int, int],
) -> list[int]:
    valid: list[int] = []
    for k in range(4):
        rot_mask, rot_red = _rotate_mask_and_red(input_mask, input_red, k)
        if _mask_key(rot_mask) != _mask_key(hole_mask):
            continue
        red_abs = (int(hole_top[0] + rot_red[0]), int(hole_top[1] + rot_red[1]))
        if _manhattan(red_abs, hole_red) == 1:
            valid.append(int(k))
    return valid


def _build_piece_patch(mask: np.ndarray, red_pos: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    patch[int(red_pos[0]), int(red_pos[1])] = RED
    return patch


def _construct_template_and_pieces(holes: list[dict], colors: list[int], template_h: int, template_w: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    template_in = np.full((template_h, template_w), GREY, dtype=int)
    template_out = template_in.copy()
    all_hole_cells: set[tuple[int, int]] = set()
    for hole in holes:
        all_hole_cells |= _local_hole_cells(hole["mask"], hole["top"])

    piece_patches: list[np.ndarray] = []
    used_boundary: set[tuple[int, int]] = set()

    for hole, color in zip(holes, colors, strict=False):
        mask = hole["mask"]
        top = hole["top"]
        for dr, dc in np.argwhere(mask):
            template_in[int(top[0] + dr), int(top[1] + dc)] = BG
            template_out[int(top[0] + dr), int(top[1] + dc)] = int(color)

        candidates = _marker_candidates(mask, top, all_hole_cells)
        found = None
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)
        for input_rot in rotations:
            for red_cell, boundary in candidates:
                if boundary in used_boundary:
                    continue
                input_mask, input_red = _rotate_mask_and_red(mask, red_cell, input_rot)
                valid = _valid_rotations_for_hole(input_mask, input_red, mask, top, boundary)
                if len(valid) != 1:
                    continue
                found = (input_mask, input_red, boundary)
                break
            if found is not None:
                break
        if found is None:
            return None

        input_mask, input_red, boundary = found
        used_boundary.add(boundary)
        template_in[int(boundary[0]), int(boundary[1])] = RED
        template_out[int(boundary[0]), int(boundary[1])] = RED
        piece_patches.append(_build_piece_patch(input_mask, input_red, int(color)))

    return template_in, template_out, piece_patches


def _place_pieces_on_grid(template_in: np.ndarray, template_out: np.ndarray, piece_patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    template_h, template_w = template_in.shape
    max_piece_h = max(int(p.shape[0]) for p in piece_patches)
    total_piece_w = sum(int(p.shape[1]) for p in piece_patches) + (len(piece_patches) - 1) * 2

    template_left = random.randint(*GEN_TEMPLATE_LEFT)
    middle_gap = random.randint(*GEN_RIGHT_GAP)
    right_extra = random.randint(*GEN_RIGHT_EXTRA_W)
    top_margin = random.randint(*GEN_EDGE_MARGIN)
    bottom_margin = random.randint(*GEN_EDGE_MARGIN)
    grid_h = max(
        int(template_h + top_margin + bottom_margin),
        int(max_piece_h + random.randint(*GEN_GRID_EXTRA_H) + 2),
    )
    grid_w = int(template_left + template_w + middle_gap + total_piece_w + right_extra + random.randint(*GEN_EDGE_MARGIN))
    template_top = random.randint(1, max(1, grid_h - template_h - 1))
    piece_min_col = int(template_left + template_w + middle_gap)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    _paste_patch(inp, template_in, int(template_top), int(template_left))
    _paste_patch(out, template_out, int(template_top), int(template_left))

    occupied = inp != BG
    order = list(range(len(piece_patches)))
    random.shuffle(order)
    for idx in order:
        patch = piece_patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            top = random.randint(0, grid_h - ph)
            left = random.randint(piece_min_col, grid_w - pw)
            if not _can_place_with_padding(occupied, patch != BG, int(top), int(left), pad=GEN_PIECE_PAD):
                continue
            _paste_patch(inp, patch, int(top), int(left))
            occupied = inp != BG
            placed = True
            break
        if not placed:
            return None

    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_CASE_ATTEMPTS):
        count = random.randint(*GEN_PIECE_COUNT)
        colors = random.sample(PIECE_COLORS, count)
        base_shapes = _choose_shapes(count)
        masks = [_tight_crop_nonzero(np.rot90(shape.astype(int), random.randrange(4))).astype(bool) for shape in base_shapes]

        template_h = random.randint(*GEN_TEMPLATE_H)
        template_w = random.randint(*GEN_TEMPLATE_W)
        holes = _place_holes(template_h, template_w, masks)
        if holes is None:
            continue

        built = _construct_template_and_pieces(holes, colors, template_h, template_w)
        if built is None:
            continue
        template_in, template_out, piece_patches = built

        placed = _place_pieces_on_grid(template_in, template_out, piece_patches)
        if placed is None:
            continue
        inp, expected = placed

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp, expected

    raise AssertionError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
