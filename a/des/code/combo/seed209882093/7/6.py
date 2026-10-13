import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
ORANGE = 7
YELLOW = 4
MAGENTA = 6

LABEL_COLORS = (MAROON, ORANGE, YELLOW, MAGENTA)
BODY_COLORS = (1, 2, 3, 8)
QUADRANTS = ("tl", "tr", "bl", "br")
CORNER_LABELS = {
    "tl": MAROON,
    "tr": ORANGE,
    "bl": YELLOW,
    "br": MAGENTA,
}
OUTER_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _bbox_of_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    bbox = _bbox_of_mask(arr != BLACK)
    if bbox is None:
        return arr[:0, :0].copy()
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _component_array(grid: np.ndarray, coords):
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    for r, c in coords:
        out[r - int(r0), c - int(c0)] = int(grid[r, c])
    return out, (int(r0), int(c0), int(r1), int(c1))


def _flat_sides(mask: np.ndarray):
    return {
        "top": bool(mask[0, :].all()),
        "bottom": bool(mask[-1, :].all()),
        "left": bool(mask[:, 0].all()),
        "right": bool(mask[:, -1].all()),
    }


def _orient_piece(arr: np.ndarray, quadrant: str) -> np.ndarray:
    desired = OUTER_SIDES[quadrant]
    others = tuple(side for side in ("top", "bottom", "left", "right") if side not in desired)
    for strict in (True, False):
        for k in range(4):
            rot = np.rot90(arr, k)
            flat = _flat_sides(rot != BLACK)
            if not all(flat[side] for side in desired):
                continue
            if strict and any(flat[side] for side in others):
                continue
            return rot
    raise AssertionError(f"could not orient piece for quadrant {quadrant}")


def _frame_candidate(grid: np.ndarray, bbox):
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] != BLACK):
        return False
    if not np.all(sub[-1, :] != BLACK):
        return False
    if not np.all(sub[:, 0] != BLACK):
        return False
    if not np.all(sub[:, -1] != BLACK):
        return False
    if not np.all(sub[1:-1, 1:-1] == BLACK):
        return False
    corners = (int(sub[0, 0]), int(sub[0, -1]), int(sub[-1, 0]), int(sub[-1, -1]))
    return len(set(corners)) == 4 and all(color in LABEL_COLORS for color in corners)


def _detect_frame(grid: np.ndarray):
    comps = _find_components(grid != BLACK)
    candidate = None
    for coords in comps:
        pts = np.asarray(coords, dtype=int)
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        if _frame_candidate(grid, bbox):
            candidate = bbox
            break
    if candidate is None:
        raise AssertionError("frame not found")
    return candidate


def _frame_label_map(grid: np.ndarray, bbox):
    r0, c0, r1, c1 = bbox
    return {
        int(grid[r0, c0]): "tl",
        int(grid[r0, c1]): "tr",
        int(grid[r1, c0]): "bl",
        int(grid[r1, c1]): "br",
    }


def _piece_body_color(arr: np.ndarray, label_color: int) -> int:
    colors = set(map(int, arr[arr != BLACK].flatten()))
    colors.discard(GREY)
    colors.discard(int(label_color))
    assert len(colors) == 1
    return next(iter(colors))


def _extract_pieces(grid: np.ndarray, frame_bbox):
    pieces = {}
    label_map = _frame_label_map(grid, frame_bbox)
    for coords in _find_components(grid != BLACK):
        arr, bbox = _component_array(grid, coords)
        if bbox == frame_bbox:
            continue
        labels = [color for color in LABEL_COLORS if np.any(arr == color)]
        if len(labels) != 1:
            continue
        label_color = int(labels[0])
        quadrant = label_map[label_color]
        body = _piece_body_color(arr, label_color)
        cleaned = arr.copy()
        cleaned[cleaned == label_color] = body
        pieces[quadrant] = cleaned
    if set(pieces) != set(QUADRANTS):
        raise AssertionError("expected exactly four labeled pieces")
    return pieces


def _assemble_from_pieces(pieces: dict[str, np.ndarray], shape):
    out = np.zeros(shape, dtype=int)
    h, w = shape
    for quadrant in QUADRANTS:
        piece = _orient_piece(pieces[quadrant], quadrant)
        ph, pw = piece.shape
        r0 = 0 if quadrant in ("tl", "tr") else h - ph
        c0 = 0 if quadrant in ("tl", "bl") else w - pw
        mask = piece != BLACK
        window = out[r0 : r0 + ph, c0 : c0 + pw]
        overlap = mask & (window != BLACK)
        if np.any(overlap & (window != piece)):
            raise AssertionError("piece overlap while assembling")
        window[mask] = piece[mask]
    if np.any(out == BLACK):
        raise AssertionError("assembled frame has holes")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four labeled jagged fragments into the rectangular frame crop.

    The frame corner labels map colors to the four output quadrants. Each piece
    has one matching label color and exactly two flat sides, which identify the
    rotation needed to face the frame boundary; the remaining jagged sides then
    interlock when the rotated piece is aligned to its target corner.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _detect_frame(g)
    r0, c0, r1, c1 = frame_bbox
    pieces = _extract_pieces(g, frame_bbox)
    return _assemble_from_pieces(pieces, (int(r1 - r0 + 1), int(c1 - c0 + 1)))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                boundary[r, c] = True
                break
    return boundary


def _random_profile(rng: np.random.Generator, length: int, base: int) -> np.ndarray:
    while True:
        offsets = rng.integers(-1, 2, size=length)
        profile = base + offsets
        if len(set(map(int, profile.tolist()))) >= 2:
            return profile.astype(int)


def _owner_grid(height: int, width: int, hsplit: np.ndarray, vsplit: np.ndarray) -> np.ndarray:
    owner = np.empty((height, width), dtype="<U2")
    for r in range(height):
        for c in range(width):
            top = r < int(hsplit[c])
            left = c < int(vsplit[r])
            if top and left:
                owner[r, c] = "tl"
            elif top:
                owner[r, c] = "tr"
            elif left:
                owner[r, c] = "bl"
            else:
                owner[r, c] = "br"
    return owner


def _random_edge_pair(
    rng: np.random.Generator,
    length: int,
    both_true_positions=(),
):
    forced = set(int(i) for i in both_true_positions)
    while True:
        a = np.ones(length, dtype=bool)
        b = np.ones(length, dtype=bool)
        for i in range(length):
            if i in forced:
                continue
            state = int(rng.integers(0, 3))
            if state == 0:
                a[i], b[i] = True, False
            elif state == 1:
                a[i], b[i] = False, True
            else:
                a[i], b[i] = True, True
        if np.all(a) or np.all(b):
            continue
        return a, b


def _piece_mask_tl(ht: int, wl: int, right: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    mask = np.ones((ht + 1, wl + 1), dtype=bool)
    mask[:, -1] = right
    mask[-1, :] = bottom
    mask[-1, -1] = bool(right[-1] and bottom[-1])
    return mask


def _piece_mask_tr(ht: int, wr: int, left: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    mask = np.ones((ht + 1, wr + 1), dtype=bool)
    mask[:, 0] = left
    mask[-1, :] = bottom
    mask[-1, 0] = bool(left[-1] and bottom[0])
    return mask


def _piece_mask_bl(hb: int, wl: int, right: np.ndarray, top: np.ndarray) -> np.ndarray:
    mask = np.ones((hb + 1, wl + 1), dtype=bool)
    mask[:, -1] = right
    mask[0, :] = top
    mask[0, -1] = bool(right[0] and top[-1])
    return mask


def _piece_mask_br(hb: int, wr: int, left: np.ndarray, top: np.ndarray) -> np.ndarray:
    mask = np.ones((hb + 1, wr + 1), dtype=bool)
    mask[:, 0] = left
    mask[0, :] = top
    mask[0, 0] = bool(left[0] and top[0])
    return mask


def _color_piece(mask: np.ndarray, body_color: int, label_color: int, rng: np.random.Generator) -> np.ndarray:
    boundary = _boundary_mask(mask)
    interior = np.argwhere(mask & ~boundary)
    if interior.shape[0] < 2:
        raise AssertionError("piece needs interior body cells")
    arr = np.zeros(mask.shape, dtype=int)
    arr[mask] = int(body_color)
    arr[boundary] = GREY
    lr, lc = map(int, interior[rng.integers(0, interior.shape[0])])
    arr[lr, lc] = int(label_color)
    return arr


def _build_solution_and_pieces(rng: np.random.Generator):
    ht = int(rng.integers(4, 7))
    hb = int(rng.integers(4, 7))
    wl = int(rng.integers(4, 7))
    wr = int(rng.integers(4, 7))

    tl_right, tr_left = _random_edge_pair(rng, ht + 1, both_true_positions=(0, ht))
    bl_right, br_left = _random_edge_pair(rng, hb + 1, both_true_positions=(0, hb))
    tl_bottom, bl_top = _random_edge_pair(rng, wl + 1, both_true_positions=(0, wl))
    tr_bottom, br_top = _random_edge_pair(rng, wr + 1, both_true_positions=(0, wr))

    masks = {
        "tl": _piece_mask_tl(ht, wl, tl_right, tl_bottom),
        "tr": _piece_mask_tr(ht, wr, tr_left, tr_bottom),
        "bl": _piece_mask_bl(hb, wl, bl_right, bl_top),
        "br": _piece_mask_br(hb, wr, br_left, br_top),
    }

    color_order = list(rng.permutation(np.array(BODY_COLORS, dtype=int)))
    body_by_quadrant = {quad: int(color_order[i]) for i, quad in enumerate(QUADRANTS)}
    pieces = {}
    cleaned = {}
    for quadrant in QUADRANTS:
        try:
            piece = _color_piece(masks[quadrant], body_by_quadrant[quadrant], CORNER_LABELS[quadrant], rng)
        except AssertionError:
            return None
        flat = _flat_sides(piece != BLACK)
        desired = OUTER_SIDES[quadrant]
        others = tuple(side for side in ("top", "bottom", "left", "right") if side not in desired)
        if not all(flat[side] for side in desired):
            return None
        if any(flat[side] for side in others):
            return None
        pieces[quadrant] = piece
        cleaned_piece = piece.copy()
        cleaned_piece[cleaned_piece == CORNER_LABELS[quadrant]] = body_by_quadrant[quadrant]
        cleaned[quadrant] = cleaned_piece

    try:
        solution = _assemble_from_pieces(cleaned, (ht + hb + 1, wl + wr + 1))
    except AssertionError:
        return None
    return solution, pieces


def _frame_array(shape):
    h, w = shape
    frame = np.zeros((h, w), dtype=int)
    frame[0, :] = GREY
    frame[-1, :] = GREY
    frame[:, 0] = GREY
    frame[:, -1] = GREY
    frame[0, 0] = MAROON
    frame[0, -1] = ORANGE
    frame[-1, 0] = YELLOW
    frame[-1, -1] = MAGENTA
    return frame


def _placement_bounds(canvas_shape, top: int, left: int, obj_shape, gap: int):
    h, w = obj_shape
    H, W = canvas_shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(H, top + h + gap)
    c1 = min(W, left + w + gap)
    return r0, c0, r1, c1


def _can_place(
    canvas: np.ndarray,
    obj: np.ndarray,
    top: int,
    left: int,
    gap: int = 1,
    blocked: np.ndarray | None = None,
) -> bool:
    h, w = obj.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0, c0, r1, c1 = _placement_bounds(canvas.shape, top, left, obj.shape, gap)
    if blocked is not None and np.any(blocked[r0:r1, c0:c1]):
        return False
    return bool(np.all(canvas[r0:r1, c0:c1] == BLACK))


def _place(
    canvas: np.ndarray,
    obj: np.ndarray,
    rng: np.random.Generator,
    gap: int = 1,
    blocked: np.ndarray | None = None,
) -> bool:
    h, w = obj.shape
    H, W = canvas.shape
    for _ in range(400):
        top = int(rng.integers(1, H - h))
        left = int(rng.integers(1, W - w))
        if not _can_place(canvas, obj, top, left, gap=gap, blocked=blocked):
            continue
        mask = obj != BLACK
        canvas[top : top + h, left : left + w][mask] = obj[mask]
        if blocked is not None:
            r0, c0, r1, c1 = _placement_bounds(canvas.shape, top, left, obj.shape, gap)
            blocked[r0:r1, c0:c1] = True
        return True
    return False


def _gen_attempt(rng: np.random.Generator):
    built = _build_solution_and_pieces(rng)
    if built is None:
        return None
    solution, pieces = built

    piece_arrays = []
    for quadrant in QUADRANTS:
        piece = pieces[quadrant]
        k = int(rng.integers(0, 4))
        piece_arrays.append(np.rot90(piece, k))

    frame = _frame_array(solution.shape)
    max_h = max([frame.shape[0]] + [arr.shape[0] for arr in piece_arrays])
    max_w = max([frame.shape[1]] + [arr.shape[1] for arr in piece_arrays])
    canvas_h = int(max(24, solution.shape[0] + max_h + 10 + rng.integers(0, 5)))
    canvas_w = int(max(24, solution.shape[1] + 2 * max_w + 10 + rng.integers(0, 5)))
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)

    objects = [frame] + piece_arrays
    order = list(rng.permutation(len(objects)))
    placed = [None] * len(objects)
    for idx in order:
        if not _place(canvas, objects[idx], rng, gap=1, blocked=blocked):
            return None
        placed[idx] = True

    try:
        solved = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(solved, solution):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
