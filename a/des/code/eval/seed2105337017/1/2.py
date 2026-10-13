import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PIECE_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


SHAPE_LIBRARY = (
    {
        "name": "square2",
        "base": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "red": (-1, 0),
        "symmetric": True,
    },
    {
        "name": "bar3",
        "base": ((0, 0), (0, 1), (0, 2)),
        "red": (-1, 1),
        "symmetric": True,
    },
    {
        "name": "l3",
        "base": ((0, 0), (1, 0), (1, 1)),
        "red": (-1, 0),
        "symmetric": False,
    },
    {
        "name": "t4",
        "base": ((0, 0), (0, 1), (0, 2), (1, 1)),
        "red": (2, 1),
        "symmetric": False,
    },
    {
        "name": "s4",
        "base": ((0, 1), (0, 2), (1, 0), (1, 1)),
        "red": (-1, 1),
        "symmetric": False,
    },
    {
        "name": "p5",
        "base": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
        "red": (2, 1),
        "symmetric": False,
    },
)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _tighten(mask: np.ndarray, extra: np.ndarray | None = None):
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    if extra is None:
        return cropped
    extra = np.asarray(extra, dtype=bool)
    return cropped, extra[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in np.asarray(mask, dtype=bool))


def _rotation_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tighten(mask)
    return min(_mask_key(np.rot90(tight, k)) for k in range(4))


def _marker_side(occ: np.ndarray, red: np.ndarray) -> str:
    occ = np.asarray(occ, dtype=bool)
    red = np.asarray(red, dtype=bool)
    assert int(red.sum()) == 1
    base = occ & ~red
    assert int(base.sum()) >= 1
    base_rows, base_cols = np.where(base)
    rr, cc = map(int, np.argwhere(red)[0])
    r0 = int(base_rows.min())
    r1 = int(base_rows.max())
    c0 = int(base_cols.min())
    c1 = int(base_cols.max())

    if rr < r0 and c0 <= cc <= c1:
        return "top"
    if rr > r1 and c0 <= cc <= c1:
        return "bottom"
    if cc < c0 and r0 <= rr <= r1:
        return "left"
    if cc > c1 and r0 <= rr <= r1:
        return "right"
    raise AssertionError("red marker must protrude from exactly one side of the base shape")


def _component_descriptor(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)

    occ = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    red = np.zeros_like(occ)
    for r, c in coords:
        occ[r - r0, c - c0] = True
        if int(grid[r, c]) == RED:
            red[r - r0, c - c0] = True
    occ, red = _tighten(occ, red)
    base = _tighten(occ & ~red)
    return {
        "coords": sorted((int(r), int(c)) for r, c in coords),
        "occ": occ,
        "red": red,
        "base": base,
        "side": _marker_side(occ, red),
    }


def _rotate_descriptor(desc: dict, k: int) -> dict:
    occ = np.rot90(desc["occ"], k)
    red = np.rot90(desc["red"], k)
    occ, red = _tighten(occ, red)
    return {
        "occ": occ,
        "red": red,
        "base": _tighten(occ & ~red),
        "side": _marker_side(occ, red),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-template hole with the matching colored piece after rotating the
    piece so that its red orientation marker lands on the same side as the hole's
    red marker. The red markers disappear into the final fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_components = _find_components(g == GREY)
    assert grey_components
    template = max(grey_components, key=len)
    tr0 = min(r for r, _ in template)
    tc0 = min(c for _, c in template)
    tr1 = max(r for r, _ in template)
    tc1 = max(c for _, c in template)

    sub = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    slots: list[dict] = []
    for comp in _find_components(sub != GREY):
        if any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in comp):
            continue
        abs_coords = [(tr0 + r, tc0 + c) for r, c in comp]
        values = [int(g[r, c]) for r, c in abs_coords]
        assert values.count(RED) == 1
        assert set(values) <= {BLACK, RED}
        slots.append(_component_descriptor(g, abs_coords))

    piece_mask = (g != BLACK) & (g != GREY)
    pieces: list[dict] = []
    for comp in _find_components(piece_mask):
        values = [int(g[r, c]) for r, c in comp]
        non_red = sorted({value for value in values if value != RED})
        if not non_red:
            continue
        assert RED in values
        assert len(non_red) == 1
        desc = _component_descriptor(g, comp)
        desc["color"] = int(non_red[0])
        pieces.append(desc)

    assert slots
    assert len(slots) == len(pieces)

    slot_by_signature: dict[tuple[str, ...], dict] = {}
    for slot in slots:
        sig = _rotation_key(slot["base"])
        assert sig not in slot_by_signature
        slot_by_signature[sig] = slot

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BLACK

    used: set[tuple[str, ...]] = set()
    for piece in pieces:
        sig = _rotation_key(piece["base"])
        assert sig in slot_by_signature
        assert sig not in used
        slot = slot_by_signature[sig]

        match_count = 0
        for k in range(4):
            rotated = _rotate_descriptor(piece, k)
            if np.array_equal(rotated["base"], slot["base"]) and rotated["side"] == slot["side"]:
                match_count += 1
        assert match_count == 1

        for r, c in slot["coords"]:
            out[r, c] = int(piece["color"])
        used.add(sig)

    return out


def _coords_to_patch(base_coords: tuple[tuple[int, int], ...], red_coord: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    full_coords = list(base_coords) + [red_coord]
    rows = [r for r, _ in full_coords]
    cols = [c for _, c in full_coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    occ = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    red = np.zeros_like(occ)
    for r, c in base_coords:
        occ[r - r0, c - c0] = True
    rr, cc = red_coord
    occ[rr - r0, cc - c0] = True
    red[rr - r0, cc - c0] = True
    occ, red = _tighten(occ, red)
    return occ, red


def _shape_patch(shape: dict, rot_k: int) -> tuple[np.ndarray, np.ndarray]:
    occ, red = _coords_to_patch(shape["base"], shape["red"])
    occ = np.rot90(occ, rot_k)
    red = np.rot90(red, rot_k)
    occ, red = _tighten(occ, red)
    return occ, red


def _colored_patch(occ: np.ndarray, red: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(occ.shape, BLACK, dtype=int)
    patch[occ] = int(color)
    patch[red] = RED
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_patches(region_h: int, region_w: int, sizes: list[tuple[int, int]], margin: int = 1, gap: int = 1):
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        ph, pw = sizes[idx]
        max_r = region_h - ph - margin
        max_c = region_w - pw - margin
        if max_r < margin or max_c < margin:
            return None
        candidates = [(r, c) for r in range(margin, max_r + 1) for c in range(margin, max_c + 1)]
        random.shuffle(candidates)
        placed = None
        for r, c in candidates:
            box = (r - gap, c - gap, r + ph - 1 + gap, c + pw - 1 + gap)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            placed = (r, c)
            boxes.append(box)
            break
        if placed is None:
            return None
        positions[idx] = placed
    return positions


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, occ: np.ndarray, r0: int, c0: int) -> None:
    region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    region[occ] = patch[occ]


def _absolute_coords(occ: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(occ)]


def _select_shapes(count: int) -> list[dict]:
    symmetric = [shape for shape in SHAPE_LIBRARY if shape["symmetric"]]
    chosen = [random.choice(symmetric)]
    remaining = [shape for shape in SHAPE_LIBRARY if shape not in chosen]
    chosen.extend(random.sample(remaining, count - 1))
    random.shuffle(chosen)
    return chosen


def _build_expected(inp: np.ndarray, piece_cells: list[list[tuple[int, int]]], slot_cells: list[list[tuple[int, int]]], colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for coords in piece_cells:
        for r, c in coords:
            out[r, c] = BLACK
    for coords, color in zip(slot_cells, colors):
        for r, c in coords:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.randint(2, 4)
    shapes = _select_shapes(piece_count)
    colors = random.sample(PIECE_COLORS, piece_count)

    slot_rots = [random.randrange(4) for _ in range(piece_count)]
    piece_rots = [random.randrange(4) for _ in range(piece_count)]
    if all((piece_rots[i] - slot_rots[i]) % 4 == 0 for i in range(piece_count)):
        j = random.randrange(piece_count)
        piece_rots[j] = (piece_rots[j] + random.randint(1, 3)) % 4

    slot_parts = []
    piece_parts = []
    for idx, shape in enumerate(shapes):
        slot_occ, slot_red = _shape_patch(shape, slot_rots[idx])
        piece_occ, piece_red = _shape_patch(shape, piece_rots[idx])
        slot_parts.append((slot_occ, slot_red))
        piece_parts.append((piece_occ, piece_red))

    max_slot_h = max(part[0].shape[0] for part in slot_parts)
    max_slot_w = max(part[0].shape[1] for part in slot_parts)
    max_piece_h = max(part[0].shape[0] for part in piece_parts)
    max_piece_w = max(part[0].shape[1] for part in piece_parts)

    min_h = max(18, 10 + piece_count + max(max_slot_h, max_piece_h))
    min_w = max(28, 22 + 2 * piece_count + max_slot_w + max_piece_w)
    h = random.randint(min_h, max(min_h, 26))
    w = random.randint(min_w, max(min_w, 38))
    split = w // 2

    panel_w_lo = max(max_slot_w + 4, 8 + piece_count)
    panel_w_hi = split - 2
    if panel_w_lo > panel_w_hi:
        return None
    panel_h_lo = max(max_slot_h + 4, 8 + piece_count)
    panel_h_hi = h - 2
    if panel_h_lo > panel_h_hi:
        return None

    panel_w = random.randint(panel_w_lo, panel_w_hi)
    panel_h = random.randint(panel_h_lo, panel_h_hi)
    panel_c_hi = max(1, split - panel_w - 1)
    panel_c = random.randint(1, panel_c_hi)
    panel_r = random.randint(1, h - panel_h - 1)

    slot_positions = _place_patches(panel_h, panel_w, [part[0].shape for part in slot_parts], margin=1, gap=1)
    if slot_positions is None:
        return None

    piece_zone_c0 = split + 1
    piece_zone_w = w - piece_zone_c0
    piece_positions = _place_patches(h, piece_zone_w, [part[0].shape for part in piece_parts], margin=1, gap=1)
    if piece_positions is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    panel = np.full((panel_h, panel_w), GREY, dtype=int)
    slot_cells: list[list[tuple[int, int]]] = []
    for (occ, red), pos in zip(slot_parts, slot_positions):
        assert pos is not None
        rr, cc = pos
        patch = _colored_patch(occ, red, BLACK)
        _paint_patch(panel, patch, occ, rr, cc)
        slot_cells.append(_absolute_coords(occ, panel_r + rr, panel_c + cc))
    grid[panel_r : panel_r + panel_h, panel_c : panel_c + panel_w] = panel

    piece_cells: list[list[tuple[int, int]]] = []
    for color, (occ, red), pos in zip(colors, piece_parts, piece_positions):
        assert pos is not None
        rr, cc = pos
        cc += piece_zone_c0
        patch = _colored_patch(occ, red, color)
        _paint_patch(grid, patch, occ, rr, cc)
        piece_cells.append(_absolute_coords(occ, rr, cc))

    expected = _build_expected(grid, piece_cells, slot_cells, colors)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


_shape_signatures = [_rotation_key(_coords_to_patch(shape["base"], shape["red"])[0] & ~_coords_to_patch(shape["base"], shape["red"])[1]) for shape in SHAPE_LIBRARY]
assert len(set(_shape_signatures)) == len(_shape_signatures)
