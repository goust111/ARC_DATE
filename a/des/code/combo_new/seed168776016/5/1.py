import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FILL_BY_QUADRANT = {
    "tl": RED,
    "tr": BLUE,
    "bl": GREEN,
    "br": YELLOW,
}
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400


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
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _rotation_key(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _piece_quadrant(piece_coords: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_r = fr0 + fr1
    frame_c = fc0 + fc1

    pr0, pc0, pr1, pc1 = _bbox(piece_coords)
    piece_r = pr0 + pr1
    piece_c = pc0 + pc1
    if piece_r == frame_r:
        piece_r = float(2 * piece_coords[:, 0].mean())
    if piece_c == frame_c:
        piece_c = float(2 * piece_coords[:, 1].mean())

    if piece_r < frame_r and piece_c < frame_c:
        return "tl"
    if piece_r < frame_r and piece_c > frame_c:
        return "tr"
    if piece_r > frame_r and piece_c < frame_c:
        return "bl"
    if piece_r > frame_r and piece_c > frame_c:
        return "br"
    raise AssertionError("piece center must be strictly inside one quadrant")


def _match_holes_to_pieces(
    holes: list[dict[str, object]], pieces: list[dict[str, object]]
) -> dict[int, int]:
    candidates: list[list[int]] = []
    for hole in holes:
        hole_mask = hole["mask"]
        hole_area = int(hole["coords"].shape[0])
        matches: list[int] = []
        for idx, piece in enumerate(pieces):
            if int(piece["coords"].shape[0]) != hole_area:
                continue
            piece_mask = piece["mask"]
            for rot in range(4):
                if np.array_equal(np.rot90(piece_mask, rot), hole_mask):
                    matches.append(idx)
                    break
        assert matches, "every hole must match at least one piece"
        candidates.append(matches)

    order = sorted(range(len(holes)), key=lambda i: (len(candidates[i]), -int(holes[i]["coords"].shape[0])))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        hole_idx = order[pos]
        for piece_idx in candidates[hole_idx]:
            if piece_idx in used:
                continue
            used.add(piece_idx)
            assignment[hole_idx] = piece_idx
            if backtrack(pos + 1):
                return True
            used.remove(piece_idx)
            del assignment[hole_idx]
        return False

    assert backtrack(0), "hole-piece matching must be unique and complete"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each external fragment to a frame hole, rotate it into place, and recolor the fill by quadrant.

    Input structure:
    - Background is Black.
    - A large Grey solid rectangle sits near the grid center, with several Black irregular holes fully enclosed inside it.
    - Outside the rectangle are several disconnected monochrome fragment sprites; each fragment matches exactly one hole up to 90-degree rotation.

    Rule:
    - Find every enclosed Black hole inside the Grey rectangle.
    - For each hole, find the unique outside fragment whose shape is congruent under rotation.
    - Determine whether that fragment's center lies in the upper-left, upper-right, lower-left, or lower-right quadrant relative to the rectangle center.
    - Remove all outside fragments, keep the Grey rectangle unchanged, and fill each hole with the quadrant color: upper-left Red, upper-right Blue, lower-left Green, lower-right Yellow.

    Key insight:
    - Shape matching identifies which fragment belongs to which hole, but the fill color comes only from the fragment's spatial quadrant, not from the fragment's own color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_comps = _components(g == GREY)
    assert grey_comps, "expected a grey frame"
    frame_coords = max(grey_comps, key=lambda coords: coords.shape[0])
    frame_bbox = _bbox(frame_coords)
    fr0, fc0, fr1, fc1 = frame_bbox

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    hole_coords_abs: list[np.ndarray] = []
    for coords in _components(sub == BLACK):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == sub.shape[1] - 1):
            continue
        shifted = coords + np.array([fr0, fc0], dtype=int)
        hole_coords_abs.append(shifted)
    assert hole_coords_abs, "expected enclosed holes inside the frame"

    holes: list[dict[str, object]] = []
    for coords in hole_coords_abs:
        holes.append({"coords": coords, "mask": _mask_from_coords(coords)})

    pieces: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for coords in _components(g == color):
            pieces.append(
                {
                    "coords": coords,
                    "mask": _mask_from_coords(coords),
                    "quadrant": _piece_quadrant(coords, frame_bbox),
                }
            )
    assert len(pieces) == len(holes), "piece count must equal hole count"

    assignment = _match_holes_to_pieces(holes, pieces)
    out = g.copy()

    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for hole_idx, piece_idx in assignment.items():
        coords = holes[hole_idx]["coords"]
        color = FILL_BY_QUADRANT[str(pieces[piece_idx]["quadrant"])]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return _mask_from_coords(coords)


def _random_shape(area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r, min_c = min(rows), min(cols)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if area == h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    return None


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(count):
        for _ in range(160):
            area = random.randint(3, 6)
            shape = _random_shape(area)
            if shape is None:
                continue
            key = _rotation_key(shape)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(shape)
            break
        else:
            return None
    return chosen


def _dilate_coords(coords: np.ndarray, h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _place_holes_in_frame(frame_h: int, frame_w: int, shapes: list[np.ndarray]) -> list[dict[str, object]] | None:
    blocked = np.zeros((frame_h, frame_w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    holes: list[dict[str, object]] = []
    ordered = sorted(shapes, key=lambda mask: int(mask.sum()), reverse=True)
    for shape in ordered:
        placed = False
        for _ in range(200):
            oriented = np.rot90(shape, random.randrange(4))
            h, w = oriented.shape
            if frame_h - h - 1 < 1 or frame_w - w - 1 < 1:
                return None
            top = random.randint(1, frame_h - h - 1)
            left = random.randint(1, frame_w - w - 1)
            local = np.argwhere(oriented) + np.array([top, left], dtype=int)
            forbidden = _dilate_coords(local, frame_h, frame_w)
            if any(blocked[r, c] for r, c in forbidden):
                continue
            for r, c in forbidden:
                blocked[r, c] = True
            holes.append({"coords": local, "base": shape, "mask": oriented})
            placed = True
            break
        if not placed:
            return None
    return holes


def _corner_ranges(
    grid_shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], mask_shape: tuple[int, int], quadrant: str
) -> tuple[range, range] | None:
    h, w = grid_shape
    fr0, fc0, fr1, fc1 = frame_bbox
    mh, mw = mask_shape
    gap = 1

    if quadrant == "tl":
        r0, r1 = 0, fr0 - gap - mh
        c0, c1 = 0, fc0 - gap - mw
    elif quadrant == "tr":
        r0, r1 = 0, fr0 - gap - mh
        c0, c1 = fc1 + gap + 1, w - mw
    elif quadrant == "bl":
        r0, r1 = fr1 + gap + 1, h - mh
        c0, c1 = 0, fc0 - gap - mw
    else:
        r0, r1 = fr1 + gap + 1, h - mh
        c0, c1 = fc1 + gap + 1, w - mw

    if r0 > r1 or c0 > c1:
        return None
    return range(r0, r1 + 1), range(c0, c1 + 1)


def _place_piece(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], mask: np.ndarray, quadrant: str, color: int
) -> np.ndarray | None:
    ranges = _corner_ranges(grid.shape, frame_bbox, mask.shape, quadrant)
    if ranges is None:
        return None
    row_range, col_range = ranges
    candidates = [(r, c) for r in row_range for c in col_range]
    random.shuffle(candidates)
    for top, left in candidates:
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        if np.any(grid[coords[:, 0], coords[:, 1]] != BLACK):
            continue
        grid[coords[:, 0], coords[:, 1]] = color
        return coords
    return None


def _expected_output(
    grid: np.ndarray, hole_infos: list[dict[str, object]], piece_infos: list[dict[str, object]]
) -> np.ndarray:
    out = grid.copy()
    for piece in piece_infos:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK
    for hole, piece in zip(hole_infos, piece_infos):
        coords = hole["abs_coords"]
        out[coords[:, 0], coords[:, 1]] = FILL_BY_QUADRANT[str(piece["quadrant"])]
    return out


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.choice([2, 3, 4])
    shapes = _sample_shapes(hole_count)
    if shapes is None:
        return None

    frame_h = random.randint(10, 16)
    frame_w = random.randint(10, 16)
    top_margin = random.randint(5, 7)
    bottom_margin = max(5, min(7, top_margin + random.randint(-1, 1)))
    left_margin = random.randint(5, 7)
    right_margin = max(5, min(7, left_margin + random.randint(-1, 1)))

    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    holes_local = _place_holes_in_frame(frame_h, frame_w, shapes)
    if holes_local is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    fr0, fc0 = top_margin, left_margin
    fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
    frame_bbox = (fr0, fc0, fr1, fc1)
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = GREY

    hole_infos: list[dict[str, object]] = []
    for hole in holes_local:
        abs_coords = hole["coords"] + np.array([fr0, fc0], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = BLACK
        hole_infos.append({"abs_coords": abs_coords, "base": hole["base"]})

    quadrants = random.sample(["tl", "tr", "bl", "br"], hole_count)
    random.shuffle(hole_infos)

    piece_infos: list[dict[str, object]] = []
    used_colors: set[int] = set()
    for hole, quadrant in zip(hole_infos, quadrants):
        piece_mask = np.rot90(hole["base"], random.randrange(4))
        fill_color = FILL_BY_QUADRANT[quadrant]
        available = [c for c in PIECE_COLORS if c not in used_colors and c != fill_color]
        if not available:
            return None
        color = random.choice(available)
        coords = _place_piece(grid, frame_bbox, piece_mask, quadrant, color)
        if coords is None:
            return None
        used_colors.add(color)
        piece_infos.append({"coords": coords, "quadrant": quadrant})

    expected = _expected_output(grid, hole_infos, piece_infos)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
