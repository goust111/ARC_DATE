from __future__ import annotations

import random

import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=bool)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[int, int, bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k).astype(bool)
        key = _mask_key(rot)
        if key not in seen:
            seen.add(key)
            out.append(rot.copy())
    return out


def _rotation_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_key(rot) for rot in _unique_rotations(mask))


def _match_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    for rot in _unique_rotations(src):
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            return rot
    return None


def _extract_holes(grid: np.ndarray, bg: int, template_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = template_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict[str, object]] = []
    for comp in _mask_components(sub == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1):
            continue
        hr0, hc0, hr1, hc1 = _bbox(comp)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        rel = comp - np.array([hr0, hc0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "mask": mask,
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "sig": _rotation_signature(mask),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each scattered piece into the uniquely matching hole of the grey template and leave a mirrored grey ghost behind.

    Input structure:
    - The background is the dominant color.
    - One large grey connected component forms a template, and it contains several background-colored interior holes.
    - Outside the template sit several small monochrome colored pieces whose shapes match the holes up to 90-degree rotation.

    Rule:
    - Find the unique largest foreground component as the template and extract every interior background hole inside its bounding box.
    - For each outside piece, match its shape to exactly one hole under rotation and fill that hole with the piece color.
    - Replace the original piece by a grey horizontal mirror of the piece inside the same tight bounding box.
    - Keep the rest of the template and the background unchanged.

    Key insight:
    - The puzzle combines two linked shape operations: rotation-based jigsaw fitting into the template holes, then a left-right mirrored grey afterimage at each source location.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _color_components(g, bg)
    assert comps, "expected a template and at least one piece"

    max_area = max(int(comp["area"]) for comp in comps)
    template_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(template_candidates) == 1, "expected a unique largest template"
    template = template_candidates[0]
    template_color = int(template["color"])
    holes = _extract_holes(g, bg, template["bbox"])
    pieces = [comp for comp in comps if comp is not template]

    assert holes, "expected at least one hole in the template"
    assert len(holes) == len(pieces), "pieces and holes must match one-to-one"

    hole_by_sig: dict[tuple[int, int, bytes], dict[str, object]] = {}
    for hole in holes:
        sig = hole["sig"]
        assert sig not in hole_by_sig, "hole shapes must be unique under rotation"
        hole_by_sig[sig] = hole

    out = g.copy()
    for piece in pieces:
        piece_mask, piece_bbox = _tight_mask(piece["coords"])
        sig = _rotation_signature(piece_mask)
        assert sig in hole_by_sig, "every piece must match a template hole"
        hole = hole_by_sig.pop(sig)
        hole_mask = hole["mask"]
        assert _match_rotation(piece_mask, hole_mask) is not None

        hr0, hc0, hr1, hc1 = hole["bbox"]
        out[hr0 : hr1 + 1, hc0 : hc1 + 1][hole_mask] = int(piece["color"])

        pr0, pc0, pr1, pc1 = piece_bbox
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        ghost_mask = np.fliplr(piece_mask)
        out[pr0 : pr1 + 1, pc0 : pc1 + 1][ghost_mask] = template_color

    assert not hole_by_sig, "all holes should be filled"
    return out


_GEN_BG_CHOICES = [0, 1, 8]
_GEN_TEMPLATE_COLOR = 5
_GEN_PIECE_COLORS = [2, 3, 4, 6, 7, 9]
_GEN_HOLE_COUNTS = [2, 3, 4]
_GEN_GRID_RANGE = (21, 30)
_GEN_TEMPLATE_RANGE = (10, 17)
_GEN_ATTEMPTS = 400


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[1, 1], [1, 1], [1, 0]],
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 0, 0], [0, 1, 1, 1]],
        [[1, 1, 1, 1], [0, 1, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
    ]
    seen: set[tuple[int, int, bytes]] = set()
    out: list[np.ndarray] = []
    for rows in raw:
        mask = np.array(rows, dtype=bool)
        sig = _rotation_signature(mask)
        if sig in seen:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        seen.add(sig)
        out.append(mask)
    return out


_SHAPES = _build_shape_library()


def _orientations_for_piece(mask: np.ndarray) -> list[np.ndarray]:
    choices = [rot for rot in _unique_rotations(mask) if not np.array_equal(rot, np.fliplr(rot))]
    return choices if choices else _unique_rotations(mask)


def _mark_cells(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, value: bool = True) -> None:
    for dr, dc in np.argwhere(mask):
        canvas[int(r0 + dr), int(c0 + dc)] = value


def _can_place_mask(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int, border: int) -> bool:
    h, w = occ.shape
    mh, mw = mask.shape
    if r0 < border or c0 < border or r0 + mh > h - border or c0 + mw > w - border:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(h, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(w, cc + padding + 1)
        if np.any(occ[r1:r2, c1:c2]):
            return False
    return True


def _place_many(area_shape: tuple[int, int], masks: list[np.ndarray], occ: np.ndarray, border: int, padding: int) -> list[tuple[int, int]] | None:
    hh, ww = area_shape
    positions: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for idx in order:
        mask = masks[idx]
        candidates: list[tuple[int, int]] = []
        mh, mw = mask.shape
        for r0 in range(border, hh - mh - border + 1):
            for c0 in range(border, ww - mw - border + 1):
                if _can_place_mask(occ, mask, r0, c0, padding=padding, border=border):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        _mark_cells(occ, mask, r0, c0)
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(color)


def _expected_from_layout(
    inp: np.ndarray,
    bg: int,
    hole_masks: list[np.ndarray],
    hole_positions: list[tuple[int, int]],
    piece_masks: list[np.ndarray],
    piece_positions: list[tuple[int, int]],
    piece_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for hole_mask, (hr0, hc0), color in zip(hole_masks, hole_positions, piece_colors, strict=False):
        out[hr0 : hr0 + hole_mask.shape[0], hc0 : hc0 + hole_mask.shape[1]][hole_mask] = int(color)
    for piece_mask, (pr0, pc0) in zip(piece_masks, piece_positions, strict=False):
        rows, cols = np.where(piece_mask)
        out[pr0 + rows, pc0 + cols] = int(bg)
        out[pr0 : pr0 + piece_mask.shape[0], pc0 : pc0 + piece_mask.shape[1]][np.fliplr(piece_mask)] = _GEN_TEMPLATE_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_GEN_BG_CHOICES)
    piece_colors = [c for c in _GEN_PIECE_COLORS if c != bg and c != _GEN_TEMPLATE_COLOR]
    piece_count = random.choice(_GEN_HOLE_COUNTS)
    base_shapes = random.sample(_SHAPES, piece_count)

    hole_masks = [random.choice(_unique_rotations(mask)) for mask in base_shapes]
    piece_masks = [random.choice(_orientations_for_piece(mask)) for mask in base_shapes]

    grid_h = random.randint(*_GEN_GRID_RANGE)
    grid_w = random.randint(*_GEN_GRID_RANGE)
    template_h = random.randint(_GEN_TEMPLATE_RANGE[0], min(_GEN_TEMPLATE_RANGE[1], grid_h - 5))
    template_w = random.randint(_GEN_TEMPLATE_RANGE[0], min(_GEN_TEMPLATE_RANGE[1], grid_w - 5))
    top = random.randint(2, grid_h - template_h - 2)
    left = random.randint(2, grid_w - template_w - 2)

    template = np.full((template_h, template_w), _GEN_TEMPLATE_COLOR, dtype=int)
    hole_occ = np.zeros((template_h, template_w), dtype=bool)
    hole_positions = _place_many((template_h, template_w), hole_masks, hole_occ, border=1, padding=1)
    if hole_positions is None:
        return None
    for mask, (r0, c0) in zip(hole_masks, hole_positions, strict=False):
        template[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = bg

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[top : top + template_h, left : left + template_w] = template

    occ = np.zeros((grid_h, grid_w), dtype=bool)
    occ[top : top + template_h, left : left + template_w] = True
    piece_positions = _place_many((grid_h, grid_w), piece_masks, occ, border=1, padding=1)
    if piece_positions is None:
        return None

    chosen_colors = random.sample(piece_colors, piece_count)
    for mask, (r0, c0), color in zip(piece_masks, piece_positions, chosen_colors, strict=False):
        _paint_mask(grid, mask, r0, c0, color)

    abs_hole_positions = [(top + r0, left + c0) for r0, c0 in hole_positions]
    expected = _expected_from_layout(
        grid,
        bg,
        hole_masks,
        abs_hole_positions,
        piece_masks,
        piece_positions,
        chosen_colors,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
