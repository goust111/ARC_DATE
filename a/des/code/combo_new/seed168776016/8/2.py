import numpy as np


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
RNG = np.random.default_rng()


SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = _trim_mask(np.rot90(mask, k))
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for rot in _rotation_variants(mask):
        keys.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
    return min(keys)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    mask = (grid != BLACK) & (grid != GREY)
    for coords in _components(mask):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "piece must be monochrome"
        color = int(colors[0])
        shape = _coords_to_mask(coords)
        pieces.append(
            {
                "coords": coords,
                "color": color,
                "signature": _shape_signature(shape),
                "area": int(shape.sum()),
            }
        )
    return pieces


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == GREY)
            and np.all(sub[-1, :] == GREY)
            and np.all(sub[:, 0] == GREY)
            and np.all(sub[:, -1] == GREY)
        ):
            continue

        enclosed = []
        for black_coords in _components(sub == BLACK):
            if (
                np.any(black_coords[:, 0] == 0)
                or np.any(black_coords[:, 0] == sub.shape[0] - 1)
                or np.any(black_coords[:, 1] == 0)
                or np.any(black_coords[:, 1] == sub.shape[1] - 1)
            ):
                continue
            enclosed.append(black_coords)

        small = [comp for comp in enclosed if 2 <= len(comp) <= 4]
        large = [comp for comp in enclosed if len(comp) >= 6]
        if len(small) != 1 or not large:
            continue

        slot = small[0]
        slot_mask = _coords_to_mask(slot)
        slot_abs = slot.copy()
        slot_abs[:, 0] += r0
        slot_abs[:, 1] += c0
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "slot_coords": slot_abs,
                "signature": _shape_signature(slot_mask),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the matching mould notch inside the grey frame.

    Input structure:
    - The background is black.
    - The upper half contains several grey rectangular frame components with a single
      small enclosed black notch attached to one interior side.
    - The lower half contains several monochrome colored fragments, each a 4-connected
      polyomino of 2-4 cells, with one unique fragment per frame notch up to rotation.

    Rule:
    - Detect every grey frame and extract its small enclosed black notch shape.
    - Detect every colored fragment outside the frames and compare shapes modulo
      90-degree rotations.
    - For each frame, erase the matching fragment from its original location and fill
      the frame's notch cells with that fragment's color.
    - Keep all grey frame pixels and all unmatched black background cells unchanged.

    Key insight:
    - The grey frame encodes a target pocket as a tiny enclosed black region; the
      colored fragment must be matched by shape under rotation rather than by color,
      position, or size alone.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    pieces = _extract_pieces(g)
    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "each frame must have exactly one piece"

    piece_by_sig: dict[tuple[int, int, bytes], dict] = {}
    for piece in pieces:
        sig = piece["signature"]
        assert sig not in piece_by_sig, "piece shapes must be unique up to rotation"
        piece_by_sig[sig] = piece

    out = g.copy()
    for frame in frames:
        sig = frame["signature"]
        assert sig in piece_by_sig, "missing piece for frame notch"
        piece = piece_by_sig.pop(sig)
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = BLACK
        out[frame["slot_coords"][:, 0], frame["slot_coords"][:, 1]] = piece["color"]

    assert not piece_by_sig, "all pieces must be used"
    return out


def _random_rotation(mask: np.ndarray) -> np.ndarray:
    return _trim_mask(np.rot90(mask, int(RNG.integers(0, 4))))


def _build_frame_patch(slot_mask: np.ndarray, side: int) -> np.ndarray:
    slot_h, slot_w = map(int, slot_mask.shape)
    if side in (LEFT, RIGHT):
        min_h = max(7, slot_h + 4)
        min_w = max(8, slot_w + 6)
    else:
        min_h = max(8, slot_h + 6)
        min_w = max(7, slot_w + 4)

    h = int(RNG.integers(min_h, min(min_h + 4, 13) + 1))
    w = int(RNG.integers(min_w, min(min_w + 4, 13) + 1))

    patch = np.full((h, w), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    if side == LEFT:
        slot_r = int(RNG.integers(1, h - slot_h))
        slot_c = 1
    elif side == RIGHT:
        slot_r = int(RNG.integers(1, h - slot_h))
        slot_c = w - 1 - slot_w
    elif side == TOP:
        slot_r = 1
        slot_c = int(RNG.integers(1, w - slot_w))
    else:
        slot_r = h - 1 - slot_h
        slot_c = int(RNG.integers(1, w - slot_w))

    slot_cells = {tuple(cell) for cell in np.argwhere(slot_mask)}
    slot_rows = {cell[0] for cell in slot_cells}
    slot_cols = {cell[1] for cell in slot_cells}
    min_row, max_row = min(slot_rows), max(slot_rows)
    min_col, max_col = min(slot_cols), max(slot_cols)
    face_extreme = {
        TOP: lambda rr, cc: rr == min_row,
        RIGHT: lambda rr, cc: cc == max_col,
        BOTTOM: lambda rr, cc: rr == max_row,
        LEFT: lambda rr, cc: cc == min_col,
    }

    for rr, cc in slot_cells:
        pr = slot_r + rr
        pc = slot_c + cc
        for dir_id, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            nbr = (rr + dr, cc + dc)
            if nbr in slot_cells:
                continue
            if dir_id == side and face_extreme[side](rr, cc):
                continue
            ar, ac = pr + dr, pc + dc
            if 0 <= ar < h and 0 <= ac < w:
                patch[ar, ac] = GREY

    return patch


def _bbox_separated(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_box(
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    obj_h: int,
    obj_w: int,
    gap: int,
) -> tuple[int, int] | None:
    if obj_h > row_hi - row_lo + 1 or obj_w > w:
        return None
    row_min = row_lo
    row_max = row_hi - obj_h + 1
    col_min = 0
    col_max = w - obj_w
    if row_min > row_max or col_min > col_max:
        return None
    for _ in range(120):
        top = int(RNG.integers(row_min, row_max + 1))
        left = int(RNG.integers(col_min, col_max + 1))
        box = (top, left, top + obj_h - 1, left + obj_w - 1)
        if all(_bbox_separated(box, prev, gap) for prev in boxes):
            boxes.append(box)
            return top, left
    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color


def _shape_library_unique() -> list[np.ndarray]:
    seen: set[tuple[int, int, bytes]] = set()
    uniq = []
    for shape in SHAPE_LIBRARY:
        sig = _shape_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(_trim_mask(shape))
    return uniq


UNIQUE_SHAPES = _shape_library_unique()


def _gen_attempt() -> np.ndarray | None:
    piece_count = int(RNG.integers(2, 5))
    if piece_count > len(UNIQUE_SHAPES):
        return None

    shape_indices = RNG.choice(len(UNIQUE_SHAPES), size=piece_count, replace=False)
    colors = RNG.choice(COLOR_POOL, size=piece_count, replace=False)

    frame_patches = []
    piece_masks = []
    piece_sigs = set()
    for idx in shape_indices:
        base = UNIQUE_SHAPES[int(idx)]
        slot_mask = _random_rotation(base)
        side = int(RNG.integers(0, 4))
        frame_patch = _build_frame_patch(slot_mask, side)
        piece_mask = _random_rotation(base)
        sig = _shape_signature(piece_mask)
        if sig in piece_sigs:
            return None
        piece_sigs.add(sig)
        frame_patches.append(frame_patch)
        piece_masks.append(piece_mask)

    max_frame_h = max(int(p.shape[0]) for p in frame_patches)
    max_piece_h = max(int(p.shape[0]) for p in piece_masks)
    min_h = max(18, 2 * max(max_frame_h, max_piece_h) + 4)
    min_w = max(
        18,
        max(max(int(p.shape[1]) for p in frame_patches), max(int(p.shape[1]) for p in piece_masks)) + 6,
    )
    if min_h > 30 or min_w > 30:
        return None

    h = int(RNG.integers(min_h, 31))
    w = int(RNG.integers(min_w, 31))
    split = h // 2
    top_hi = split - 1
    bottom_lo = split + 1
    if top_hi < max_frame_h - 1 or h - bottom_lo < max_piece_h:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    order = list(range(piece_count))
    RNG.shuffle(order)
    for i in order:
        patch = frame_patches[i]
        ph, pw = map(int, patch.shape)
        placed = _place_box(boxes, h, w, 0, top_hi, ph, pw, gap=1)
        if placed is None:
            return None
        top, left = placed
        sub = grid[top : top + ph, left : left + pw]
        overwrite = patch != BLACK
        if np.any(sub[overwrite] != BLACK):
            return None
        sub[overwrite] = patch[overwrite]

    for i in order:
        mask = piece_masks[i]
        mh, mw = map(int, mask.shape)
        placed = _place_box(boxes, h, w, bottom_lo, h - 1, mh, mw, gap=1)
        if placed is None:
            return None
        top, left = placed
        _draw_mask(grid, mask, top, left, int(colors[i]))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_extract_frames(grid)) != piece_count:
        return None
    if len(_extract_pieces(grid)) != piece_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
