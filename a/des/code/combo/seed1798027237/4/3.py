import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PIECE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_cross_bands(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    orange_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == ORANGE)]
    orange_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == ORANGE)]
    assert orange_rows and orange_cols, "orange cross bands must exist"
    row_band = (orange_rows[0], orange_rows[-1])
    col_band = (orange_cols[0], orange_cols[-1])
    assert list(range(row_band[0], row_band[1] + 1)) == orange_rows
    assert list(range(col_band[0], col_band[1] + 1)) == orange_cols
    return row_band, col_band


def _template_from_ul(ul: np.ndarray) -> np.ndarray:
    grey_coords = np.argwhere(ul == GREY)
    assert grey_coords.size > 0, "top-left quadrant must contain a grey template"
    r0, c0, r1, c1 = _bbox(grey_coords)
    template = ul[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all((template == GREY) | (template == BLACK)), "template must contain only grey and black"
    return template


def _extract_holes(template: np.ndarray) -> list[dict]:
    holes = []
    seen = set()
    for coords in _components(template == BLACK):
        if _touches_edge(coords, template.shape):
            continue
        mask, r0, c0 = _coords_to_mask(coords)
        sig = _shape_signature(mask)
        assert sig not in seen, "hole shapes must be unique"
        seen.add(sig)
        holes.append({"coords": coords, "mask": mask, "r0": r0, "c0": c0, "sig": sig})
    assert holes, "template must contain at least one interior hole"
    return holes


def _piece_records(region: np.ndarray, turns_ccw: int) -> list[dict]:
    records = []
    piece_mask = (region != BLACK) & (region != GREY) & (region != ORANGE)
    for coords in _components(piece_mask):
        colors = {int(region[r, c]) for r, c in coords}
        assert len(colors) == 1, "each piece must be monochrome"
        mask, _, _ = _coords_to_mask(coords)
        standard = np.rot90(mask, turns_ccw).copy()
        records.append(
            {
                "color": int(next(iter(colors))),
                "mask": standard,
                "sig": _shape_signature(standard),
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a grey template's interior holes by rotating scattered pieces according to quadrant.

    Input structure:
    - Black background with a full orange cross that splits the grid into four quadrants.
    - The top-left quadrant contains a grey template rectangle with black holes.
    - The other quadrants contain monochrome pieces that match the holes after
      quadrant-specific rotation.

    Rule:
    - Right-top pieces rotate clockwise 90 degrees.
    - Left-bottom pieces rotate clockwise 270 degrees.
    - Right-bottom pieces rotate 180 degrees.
    - Match each rotated piece to one unique hole and fill that hole with the piece color.

    Output:
    - The tight crop of the template rectangle with all holes filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (rr0, rr1), (cc0, cc1) = _find_cross_bands(g)
    ul = g[:rr0, :cc0]
    ur = g[:rr0, cc1 + 1 :]
    ll = g[rr1 + 1 :, :cc0]
    lr = g[rr1 + 1 :, cc1 + 1 :]

    template = _template_from_ul(ul)
    holes = _extract_holes(template)
    hole_by_sig = {hole["sig"]: hole for hole in holes}

    pieces = []
    pieces.extend(_piece_records(ur, 3))
    pieces.extend(_piece_records(ll, 1))
    pieces.extend(_piece_records(lr, 2))
    assert len(pieces) == len(holes), "piece count must equal hole count"

    out = template.copy()
    used = set()
    for piece in pieces:
        assert piece["sig"] in hole_by_sig, "every piece must match a hole after rotation"
        hole = hole_by_sig[piece["sig"]]
        key = (hole["r0"], hole["c0"])
        assert key not in used, "each hole must be filled exactly once"
        used.add(key)
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = piece["color"]

    assert len(used) == len(holes)
    return out


def _random_shape() -> np.ndarray | None:
    target = random.randint(3, 7)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rmin, cmin = min(rows), min(cols)
    norm = sorted((r - rmin, c - cmin) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if h < 2 or w < 2 or h > 4 or w > 5:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    if mask.all():
        return None

    for k in (1, 2, 3):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return None
    return mask


def _sample_holes(count: int) -> list[np.ndarray] | None:
    holes: list[np.ndarray] = []
    seen = set()
    attempts = 0
    while len(holes) < count and attempts < 500:
        attempts += 1
        mask = _random_shape()
        if mask is None:
            continue
        sig = _shape_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        holes.append(mask)
    return holes if len(holes) == count else None


def _place_masks(height: int, width: int, masks: list[np.ndarray], border: int = 1) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = list(range(len(masks)))
    order.sort(key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    result: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        r_lo = border
        c_lo = border
        r_hi = height - mh - border
        c_hi = width - mw - border
        if r_hi < r_lo or c_hi < c_lo:
            return None
        candidates = []
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(c_lo, c_hi + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + mh + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + mw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[max(0, r0 - 1) : min(height, r0 + mh + 1), max(0, c0 - 1) : min(width, c0 + mw + 1)] = True
        result[idx] = (r0, c0)

    for pos in result:
        assert pos is not None
        placements.append(pos)
    return placements


def _quadrant_assignment(count: int) -> list[str]:
    quads = ["ur", "ll", "lr"]
    assignment = quads.copy()
    for _ in range(count - 3):
        assignment.append(random.choice(quads))
    random.shuffle(assignment)
    return assignment


def _inverse_rotated(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "ur":
        return np.rot90(mask, 1).copy()
    if quadrant == "ll":
        return np.rot90(mask, 3).copy()
    if quadrant == "lr":
        return np.rot90(mask, 2).copy()
    raise ValueError(quadrant)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(3, 6)
    hole_masks = _sample_holes(hole_count)
    if hole_masks is None:
        return None

    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    board_h = random.randint(max(8, max_h + 4), 12)
    board_w = random.randint(max(8, max_w + 4), 12)

    hole_positions = _place_masks(board_h, board_w, hole_masks, border=1)
    if hole_positions is None:
        return None

    colors = random.sample(PIECE_COLORS, hole_count)
    assignment = _quadrant_assignment(hole_count)

    board_in = np.full((board_h, board_w), GREY, dtype=int)
    board_out = board_in.copy()
    pieces_by_quad = {"ur": [], "ll": [], "lr": []}

    for idx, mask in enumerate(hole_masks):
        r0, c0 = hole_positions[idx]
        _paint_mask(board_in, mask, r0, c0, BLACK)
        _paint_mask(board_out, mask, r0, c0, colors[idx])
        quadrant = assignment[idx]
        pieces_by_quad[quadrant].append({"mask": _inverse_rotated(mask, quadrant), "color": colors[idx]})

    top_h = random.randint(board_h + 2, min(14, board_h + 4))
    left_w = random.randint(board_w + 2, min(14, board_w + 4))
    bottom_h = random.randint(9, 13)
    right_w = random.randint(9, 13)
    cross_t = random.choice([1, 2])

    if top_h <= board_h + 1 or left_w <= board_w + 1:
        return None

    ur_masks = [item["mask"] for item in pieces_by_quad["ur"]]
    ll_masks = [item["mask"] for item in pieces_by_quad["ll"]]
    lr_masks = [item["mask"] for item in pieces_by_quad["lr"]]

    ur_pos = _place_masks(top_h, right_w, ur_masks, border=1)
    ll_pos = _place_masks(bottom_h, left_w, ll_masks, border=1)
    lr_pos = _place_masks(bottom_h, right_w, lr_masks, border=1)
    if ur_pos is None or ll_pos is None or lr_pos is None:
        return None

    grid = np.full((top_h + cross_t + bottom_h, left_w + cross_t + right_w), BLACK, dtype=int)
    grid[top_h : top_h + cross_t, :] = ORANGE
    grid[:, left_w : left_w + cross_t] = ORANGE

    board_r0 = random.randint(1, top_h - board_h - 1)
    board_c0 = random.randint(1, left_w - board_w - 1)
    grid[board_r0 : board_r0 + board_h, board_c0 : board_c0 + board_w] = board_in

    quad_specs = [
        ("ur", top_h, right_w, 0, left_w + cross_t, ur_pos),
        ("ll", bottom_h, left_w, top_h + cross_t, 0, ll_pos),
        ("lr", bottom_h, right_w, top_h + cross_t, left_w + cross_t, lr_pos),
    ]
    for name, qh, qw, base_r, base_c, positions in quad_specs:
        canvas = np.full((qh, qw), BLACK, dtype=int)
        for item, (r0, c0) in zip(pieces_by_quad[name], positions):
            _paint_mask(canvas, item["mask"], r0, c0, item["color"])
        grid[base_r : base_r + qh, base_c : base_c + qw] = canvas

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, board_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
