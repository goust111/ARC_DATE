import numpy as np


BLACK = 0
GREY = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(keys)


def _rotation_match(piece_mask: np.ndarray, hole_mask: np.ndarray) -> int | None:
    for k in range(4):
        rot = np.rot90(piece_mask, k)
        if rot.shape == hole_mask.shape and np.array_equal(rot, hole_mask):
            return int(k)
    return None


def _solve_assignment(holes: list[dict], pieces: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for hi, hole in enumerate(holes):
        options: list[tuple[int, int]] = []
        for pi, piece in enumerate(pieces):
            rot = _rotation_match(piece["mask"], hole["mask"])
            if rot is not None:
                options.append((pi, rot))
        candidates[hi] = options

    order = sorted(range(len(holes)), key=lambda hi: len(candidates[hi]))
    assigned: dict[int, tuple[int, int]] = {}
    used_pieces: set[int] = set()

    def dfs(i: int) -> bool:
        if i == len(order):
            return True
        hi = order[i]
        for pi, rot in candidates[hi]:
            if pi in used_pieces:
                continue
            used_pieces.add(pi)
            assigned[hi] = (pi, rot)
            if dfs(i + 1):
                return True
            used_pieces.remove(pi)
            del assigned[hi]
        return False

    ok = dfs(0)
    assert ok, "holes and pieces must admit a one-to-one rotation match"
    return assigned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered colored pieces back into the matching holes of a central grey template.

    Input structure:
    - Background is black.
    - One large grey 4-connected template sits near the center of the grid.
    - Inside the template are one or more black 4-connected holes fully enclosed by grey.
    - Outside the template are monochrome colored pieces, one unique non-black/non-grey color per piece.

    Rule:
    - Keep the grey template in place.
    - Extract every enclosed black hole inside the template and every outside colored piece.
    - Match each piece to the unique hole with the same shape under a 0°, 90°, 180°, or 270° rotation.
    - Fill the matched hole with that piece's color and erase the original outside piece back to black.

    Key insight:
    Shape identity is rotation-dependent: the piece-hole pairing is determined by trying the four right-angle orientations, not by original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_components = _components(g == GREY)
    assert grey_components, "input must contain a grey template"
    template_coords = max(grey_components, key=lambda comp: int(len(comp)))
    template_mask, (tr0, tc0, _tr1, _tc1) = _coords_to_mask(template_coords)

    holes: list[dict] = []
    inv = ~template_mask
    for comp in _components(inv):
        if _touches_edge(comp, inv.shape):
            continue
        global_coords = comp + np.array([tr0, tc0], dtype=int)
        vals = g[global_coords[:, 0], global_coords[:, 1]]
        if not np.all(vals == BLACK):
            continue
        hole_mask, (hr0, hc0, _hr1, _hc1) = _coords_to_mask(comp)
        holes.append(
            {
                "mask": hole_mask,
                "top_left": (int(tr0 + hr0), int(tc0 + hc0)),
                "coords": global_coords,
            }
        )
    assert holes, "template must contain enclosed black holes"

    pieces: list[dict] = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (BLACK, GREY)):
        for comp in _components(g == color):
            piece_mask, _ = _coords_to_mask(comp)
            pieces.append({"color": int(color), "mask": piece_mask})
    assert len(pieces) == len(holes), "piece count must equal hole count"

    assignment = _solve_assignment(holes, pieces)

    out = np.zeros_like(g)
    out[template_coords[:, 0], template_coords[:, 1]] = GREY
    for hi, hole in enumerate(holes):
        pi, _rot = assignment[hi]
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(pieces[pi]["color"])
    return out


def _random_polyomino() -> np.ndarray:
    for _ in range(300):
        area = int(RNG.integers(4, 7))
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells and nxt not in frontier:
                        frontier.append(nxt)
            pick = frontier[int(RNG.integers(len(frontier)))]
            cells.add(pick)

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        min_r, min_c = min(rs), min(cs)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        max_r = max(r for r, _ in norm)
        max_c = max(c for _, c in norm)
        mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        h, w = mask.shape
        rot_keys = {(tuple(map(int, np.rot90(mask, k).shape)), np.rot90(mask, k).tobytes()) for k in range(4)}
        if min(h, w) < 2:
            continue
        if max(h, w) > 4:
            continue
        if int(mask.sum()) == h * w:
            continue
        if len(rot_keys) < 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric polyomino")


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(600):
        shape = _random_polyomino()
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _place_masks_inside_template(shapes: list[np.ndarray], template_h: int, template_w: int) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    reserved = np.zeros((template_h, template_w), dtype=bool)
    order = sorted(range(len(shapes)), key=lambda i: (-int(shapes[i].sum()), -int(max(shapes[i].shape))))

    for idx in order:
        mask = shapes[idx]
        cells = np.argwhere(mask)
        h, w = mask.shape
        positions = [(r, c) for r in range(1, template_h - h) for c in range(1, template_w - w)]
        RNG.shuffle(positions)
        for top, left in positions:
            if any(reserved[top + rr, left + cc] for rr, cc in cells):
                continue
            placements[idx] = (int(top), int(left))
            for rr, cc in cells:
                r = int(top + rr)
                c = int(left + cc)
                reserved[max(0, r - 1) : min(template_h, r + 2), max(0, c - 1) : min(template_w, c + 2)] = True
            break
        if placements[idx] is None:
            return None
    return [(int(r), int(c)) for r, c in placements]


def _place_pieces(
    piece_masks: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    template_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(piece_masks)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    tr0, tc0, tr1, tc1 = template_bbox
    reserved[max(0, tr0 - 1) : min(grid_h, tr1 + 2), max(0, tc0 - 1) : min(grid_w, tc1 + 2)] = True
    order = sorted(range(len(piece_masks)), key=lambda i: (-int(piece_masks[i].sum()), -int(max(piece_masks[i].shape))))

    for idx in order:
        mask = piece_masks[idx]
        cells = np.argwhere(mask)
        h, w = mask.shape
        positions = [(r, c) for r in range(1, grid_h - h) for c in range(1, grid_w - w)]
        RNG.shuffle(positions)
        for top, left in positions:
            if any(reserved[top + rr, left + cc] for rr, cc in cells):
                continue
            placements[idx] = (int(top), int(left))
            for rr, cc in cells:
                r = int(top + rr)
                c = int(left + cc)
                reserved[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True
            break
        if placements[idx] is None:
            return None
    return [(int(r), int(c)) for r, c in placements]


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    cells = np.argwhere(mask)
    for rr, cc in cells:
        grid[top + rr, left + cc] = int(color)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = int(RNG.integers(2, 6))
    shapes = _sample_shapes(piece_count)
    if shapes is None:
        return None

    padded_area = sum((shape.shape[0] + 2) * (shape.shape[1] + 2) for shape in shapes)
    base_side = int(np.ceil(np.sqrt(padded_area * 1.15)))
    max_h = max(int(shape.shape[0]) for shape in shapes)
    max_w = max(int(shape.shape[1]) for shape in shapes)

    hole_positions = None
    template_h = template_w = 0
    for _ in range(80):
        h_min = max(max_h + 4, base_side - 1)
        w_min = max(max_w + 4, base_side - 1)
        h_max = min(18, h_min + 5)
        w_max = min(18, w_min + 5)
        if h_min > h_max or w_min > w_max:
            return None
        template_h = int(RNG.integers(h_min, h_max + 1))
        template_w = int(RNG.integers(w_min, w_max + 1))
        hole_positions = _place_masks_inside_template(shapes, template_h, template_w)
        if hole_positions is not None:
            break
    if hole_positions is None:
        return None

    grid_h_min = max(20, template_h + 8)
    grid_w_min = max(20, template_w + 8)
    if grid_h_min > 30 or grid_w_min > 30:
        return None
    grid_h = int(RNG.integers(grid_h_min, 31))
    grid_w = int(RNG.integers(grid_w_min, 31))

    center_r = (grid_h - template_h) // 2
    center_c = (grid_w - template_w) // 2
    r_low = max(1, center_r - 2)
    r_high = min(grid_h - template_h - 1, center_r + 2)
    c_low = max(1, center_c - 2)
    c_high = min(grid_w - template_w - 1, center_c + 2)
    template_r0 = int(RNG.integers(r_low, r_high + 1))
    template_c0 = int(RNG.integers(c_low, c_high + 1))
    template_bbox = (template_r0, template_c0, template_r0 + template_h - 1, template_c0 + template_w - 1)

    rotations = RNG.integers(0, 4, size=piece_count)
    if np.all(rotations == 0):
        pick = int(RNG.integers(piece_count))
        rotations[pick] = int(RNG.integers(1, 4))
    piece_masks = [np.rot90(shape, int(rotations[i])) for i, shape in enumerate(shapes)]
    piece_positions = _place_pieces(piece_masks, grid_h, grid_w, template_bbox)
    if piece_positions is None:
        return None

    colors = [int(x) for x in RNG.choice(PIECE_COLORS, size=piece_count, replace=False)]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = GREY

    out = np.zeros((grid_h, grid_w), dtype=int)
    out[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = GREY

    for shape, (hole_r, hole_c), color in zip(shapes, hole_positions, colors):
        _draw_mask(inp, shape, template_r0 + hole_r, template_c0 + hole_c, BLACK)
        _draw_mask(out, shape, template_r0 + hole_r, template_c0 + hole_c, color)

    for piece_mask, (piece_r, piece_c), color in zip(piece_masks, piece_positions, colors):
        _draw_mask(inp, piece_mask, piece_r, piece_c, color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, out = attempt
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
