import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_DELTA = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
ROT_FOR_SIDE = {"top": 0, "left": 1, "bottom": 2, "right": 3}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _crop_bool(mask)
    return min(_mask_key(_crop_bool(np.rot90(base, k))) for k in range(4))


def _anchor_sides(mask: np.ndarray, anchor_pos: tuple[int, int]) -> set[str]:
    rows, cols = np.where(mask)
    ar, ac = anchor_pos
    sides = set()
    if ar == int(rows.min()):
        sides.add("top")
    if ar == int(rows.max()):
        sides.add("bottom")
    if ac == int(cols.min()):
        sides.add("left")
    if ac == int(cols.max()):
        sides.add("right")
    return sides


def _find_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_comps = _components(grid == GREY)
    assert grey_comps, "expected at least one grey component"
    largest = max(grey_comps, key=len)
    return _bbox(largest)


def _extract_slots(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict]:
    tr0, tc0, tr1, tc1 = template_bbox
    sub = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]
    slots = []
    for comp in _components(sub == BLACK):
        sr0, sc0, sr1, sc1 = _bbox(comp)
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        mask[comp[:, 0] - sr0, comp[:, 1] - sc0] = True

        hint_positions = set()
        side_candidates = set()
        for rr, cc in comp:
            for nr, nc in _neighbors4(int(rr), int(cc), sub.shape[0], sub.shape[1]):
                val = int(sub[nr, nc])
                if val in (BLACK, GREY):
                    continue
                hint_positions.add((int(nr), int(nc)))
                dr = int(nr - rr)
                dc = int(nc - cc)
                side_candidates.add({(-1, 0): "top", (1, 0): "bottom", (0, -1): "left", (0, 1): "right"}[(dr, dc)])

        assert len(hint_positions) == 1, "each slot must have exactly one hint pixel"
        assert len(side_candidates) == 1, "hint pixel must indicate exactly one side"
        hint_r, hint_c = next(iter(hint_positions))
        slots.append(
            {
                "mask": mask,
                "signature": _rotation_signature(mask),
                "coords_abs": comp + np.array([tr0, tc0], dtype=int),
                "hint_pos_abs": (tr0 + hint_r, tc0 + hint_c),
                "hint_color": int(sub[hint_r, hint_c]),
                "hint_side": next(iter(side_candidates)),
            }
        )
    return slots


def _extract_pieces(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict]:
    tr0, tc0, tr1, tc1 = template_bbox
    outer = (grid != BLACK).copy()
    outer[tr0 : tr1 + 1, tc0 : tc1 + 1] = False
    pieces = []
    for comp in _components(outer):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        vals = patch[patch != BLACK]
        colors, counts = np.unique(vals, return_counts=True)
        assert len(colors) == 2, "each piece must have one body color and one anchor color"
        order = np.argsort(-counts)
        body_color = int(colors[order[0]])
        anchor_candidates = [int(col) for col in colors if int(col) != body_color]
        assert len(anchor_candidates) == 1, "piece anchor color must be unique"
        anchor_color = anchor_candidates[0]
        assert int(np.count_nonzero(patch == anchor_color)) == 1, "anchor color must appear exactly once"
        pieces.append(
            {
                "patch": patch,
                "signature": _rotation_signature(patch != BLACK),
                "body_color": body_color,
                "anchor_color": anchor_color,
                "coords_abs": comp,
            }
        )
    return pieces


def _choose_rotation(piece: dict, slot: dict) -> np.ndarray:
    anchor_color = piece["anchor_color"]
    slot_mask = slot["mask"]
    target_side = slot["hint_side"]
    for k in range(4):
        rotated = np.rot90(piece["patch"], k)
        rotated = _crop_patch(rotated)
        mask = rotated != BLACK
        if not np.array_equal(mask, slot_mask):
            continue
        anchor_pos = tuple(np.argwhere(rotated == anchor_color)[0])
        if target_side in _anchor_sides(mask, anchor_pos):
            return rotated
    raise AssertionError("no rotation aligns the piece anchor with the slot hint")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a grey template's matching slots, using edge hints to fix rotation.

    Input structure:
    - Background is black. One large grey filled rectangle sits near the center as a template.
    - Inside the template are one or more black slot shapes. Each slot has exactly one non-grey hint pixel on an adjacent template edge cell.
    - Outside the template are scattered connected fragments. Each fragment has one body color plus a single differently colored anchor pixel.

    Rule:
    - Find the template, extract each black slot shape, and match every outside fragment to the unique slot with the same shape up to 90-degree rotation.
    - For the matched slot, use the fragment's anchor color and the slot's hint color to confirm the pairing, then rotate the fragment so the anchor lies on the same side as the hint pixel.
    - Clear the scattered fragments, restore every hint pixel to grey, and fill each slot with the fragment's body color, including the former anchor cell.

    Key insight:
    - Shape alone decides which fragment belongs to which slot, but the colored anchor and colored edge hint are needed to resolve the fragment's final orientation.
    """
    g = np.asarray(grid, dtype=int)
    template_bbox = _find_template_bbox(g)
    slots = _extract_slots(g, template_bbox)
    pieces = _extract_pieces(g, template_bbox)

    assert len(slots) == len(pieces) and slots, "piece count must equal slot count"
    slot_by_signature = {}
    for slot in slots:
        sig = slot["signature"]
        assert sig not in slot_by_signature, "slot shapes must be unique up to rotation"
        slot_by_signature[sig] = slot

    out = g.copy()
    for piece in pieces:
        coords = piece["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for slot in slots:
        hr, hc = slot["hint_pos_abs"]
        out[hr, hc] = GREY

    for piece in pieces:
        sig = piece["signature"]
        assert sig in slot_by_signature, "every piece must match a slot shape"
        slot = slot_by_signature[sig]
        assert piece["anchor_color"] == slot["hint_color"], "anchor color must match the slot hint color"
        _choose_rotation(piece, slot)
        coords = slot["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = piece["body_color"]

    return out


def _shape_square(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=bool)


def _shape_plus(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mid = n // 2
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _shape_diamond(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    mask = np.zeros((size, size), dtype=bool)
    mid = radius
    for r in range(size):
        for c in range(size):
            if abs(r - mid) + abs(c - mid) <= radius:
                mask[r, c] = True
    return mask


def _shape_octagon5() -> np.ndarray:
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    mask[0, 4] = False
    mask[4, 0] = False
    mask[4, 4] = False
    return mask


def _anchor_for_side(mask: np.ndarray, anchor_top: tuple[int, int], side: str) -> tuple[int, int]:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = 1
    patch[anchor_top] = 2
    rotated = np.rot90(patch, ROT_FOR_SIDE[side])
    anchor_pos = tuple(np.argwhere(rotated == 2)[0])
    assert np.array_equal(rotated != 0, mask), "shape library must stay invariant under quarter turns"
    assert _anchor_sides(mask, anchor_pos) == {side}, "anchor must uniquely identify one side"
    return anchor_pos


def _make_shape_library() -> list[dict]:
    shapes = [
        {"name": "square3", "mask": _shape_square(3), "anchor": (0, 1)},
        {"name": "square4", "mask": _shape_square(4), "anchor": (0, 1)},
        {"name": "square5", "mask": _shape_square(5), "anchor": (0, 2)},
        {"name": "plus3", "mask": _shape_plus(3), "anchor": (0, 1)},
        {"name": "plus5", "mask": _shape_plus(5), "anchor": (0, 2)},
        {"name": "diamond5", "mask": _shape_diamond(2), "anchor": (0, 2)},
        {"name": "octagon5", "mask": _shape_octagon5(), "anchor": (0, 2)},
    ]
    for shape in shapes:
        mask = shape["mask"]
        for k in range(4):
            assert np.array_equal(mask, np.rot90(mask, k)), f"{shape['name']} must be 4-way symmetric"
        shape["anchors"] = {side: _anchor_for_side(mask, shape["anchor"], side) for side in SIDES}
        shape["signature"] = _rotation_signature(mask)
    sigs = [shape["signature"] for shape in shapes]
    assert len(set(sigs)) == len(sigs), "shape library entries must be unique up to rotation"
    return shapes


SHAPE_LIBRARY = _make_shape_library()


def _build_piece_patch(shape: dict, body_color: int, anchor_color: int, rotation_k: int) -> np.ndarray:
    patch = np.zeros(shape["mask"].shape, dtype=int)
    patch[shape["mask"]] = body_color
    ar, ac = shape["anchor"]
    patch[ar, ac] = anchor_color
    return _crop_patch(np.rot90(patch, rotation_k))


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int):
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    assert not np.any(target[mask] != BLACK), "objects must not overlap"
    target[mask] = patch[mask]


def _zone_bounds(
    side: str,
    patch_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    ph, pw = patch_shape
    tr0, tc0, tr1, tc1 = template_bbox
    gh, gw = grid_shape
    if side == "top":
        return 0, tr0 - ph - 1, tc0, tc1 - pw + 1
    if side == "bottom":
        return tr1 + 2, gh - ph, tc0, tc1 - pw + 1
    if side == "left":
        return tr0, tr1 - ph + 1, 0, tc0 - pw - 1
    return tr0, tr1 - ph + 1, tc1 + 2, gw - pw


def _build_expected_output(grid_shape: tuple[int, int], template_bbox: tuple[int, int, int, int], slots: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = GREY
    for slot in slots:
        coords = slot["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = slot["body_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_slots = rng.randint(2, 4)
    chosen_shapes = rng.sample(SHAPE_LIBRARY, n_slots)

    gap_y = rng.randint(1, 2)
    gap_x = rng.randint(1, 2)
    template_h = 2 * 1 + 2 * 7 + gap_y
    template_w = 2 * 1 + 2 * 7 + gap_x

    base_margin = 6
    extra_h = rng.randint(0, 30 - (template_h + 2 * base_margin))
    extra_w = rng.randint(0, 30 - (template_w + 2 * base_margin))
    top = base_margin + rng.randint(0, extra_h)
    bottom = base_margin + (extra_h - (top - base_margin))
    left = base_margin + rng.randint(0, extra_w)
    right = base_margin + (extra_w - (left - base_margin))

    grid_h = template_h + top + bottom
    grid_w = template_w + left + right
    grid = np.zeros((grid_h, grid_w), dtype=int)
    template = np.full((template_h, template_w), GREY, dtype=int)
    template_bbox = (top, left, top + template_h - 1, left + template_w - 1)

    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(colors)
    hint_colors = colors[:n_slots]
    body_colors = colors[n_slots : 2 * n_slots]

    cell_rows = [1, 1 + 7 + gap_y]
    cell_cols = [1, 1 + 7 + gap_x]
    cells = [(r, c) for r in cell_rows for c in cell_cols]
    rng.shuffle(cells)

    slot_records = []
    for shape, hint_color, body_color, (cell_r, cell_c) in zip(chosen_shapes, hint_colors, body_colors, cells):
        mask = shape["mask"]
        sh, sw = mask.shape
        offset_r = rng.randint(1, 7 - sh - 1)
        offset_c = rng.randint(1, 7 - sw - 1)
        origin_r = cell_r + offset_r
        origin_c = cell_c + offset_c

        patch = template[origin_r : origin_r + sh, origin_c : origin_c + sw]
        assert np.all(patch[mask] == GREY), "slot cells must start grey"
        patch[mask] = BLACK

        hint_side = rng.choice(SIDES)
        ar, ac = shape["anchors"][hint_side]
        dr, dc = SIDE_TO_DELTA[hint_side]
        hint_r = origin_r + ar + dr
        hint_c = origin_c + ac + dc
        assert template[hint_r, hint_c] == GREY, "hint must sit on untouched template border"
        template[hint_r, hint_c] = hint_color

        rel_coords = np.argwhere(mask)
        abs_coords = rel_coords + np.array([top + origin_r, left + origin_c], dtype=int)
        slot_records.append(
            {
                "shape": shape,
                "hint_color": hint_color,
                "body_color": body_color,
                "hint_side": hint_side,
                "coords_abs": abs_coords,
            }
        )

    grid[top : top + template_h, left : left + template_w] = template

    piece_sides = list(SIDES)
    rng.shuffle(piece_sides)
    for slot, piece_side in zip(slot_records, piece_sides):
        patch = _build_piece_patch(slot["shape"], slot["body_color"], slot["hint_color"], rng.randrange(4))
        r0, r1, c0, c1 = _zone_bounds(piece_side, patch.shape, template_bbox, grid.shape)
        if r0 > r1 or c0 > c1:
            return None
        pr = rng.randint(r0, r1)
        pc = rng.randint(c0, c1)
        _paste(grid, patch, pr, pc)

    expected = _build_expected_output(grid.shape, template_bbox, slot_records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
