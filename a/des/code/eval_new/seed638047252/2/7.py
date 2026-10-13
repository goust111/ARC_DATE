import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
COMPASS_COLOR = 7
PIECE_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_RETRIES = 300


SHAPE_LIBRARY = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1, 1]], dtype=int),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_color_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if val == bg:
                continue
            rr, cc = top + r, left + c
            if 0 <= rr < dst.shape[0] and 0 <= cc < dst.shape[1]:
                dst[rr, cc] = val


def _quadrant_bounds(
    frame_bbox: tuple[int, int, int, int]
) -> dict[str, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return {
        "tl": (r0 + 1, mid_r, c0 + 1, mid_c),
        "tr": (r0 + 1, mid_r, mid_c + 1, c1),
        "bl": (mid_r + 1, r1, c0 + 1, mid_c),
        "br": (mid_r + 1, r1, mid_c + 1, c1),
    }


def _find_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], dict[int, tuple[int, int]]]:
    h, w = grid.shape
    best = None
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            if side_h != side_w or side_h < 7 or side_h % 2 == 0:
                continue
            mask = np.zeros((side_h, side_w), dtype=bool)
            local = coords - np.array([r0, c0], dtype=int)
            mask[local[:, 0], local[:, 1]] = True

            mid = side_h // 2
            expected = np.zeros_like(mask)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            expected[mid, :] = True
            expected[:, mid] = True
            if not np.array_equal(mask, expected):
                continue

            markers: dict[int, tuple[int, int]] = {}
            ok = True
            for qr0, qr1, qc0, qc1 in _quadrant_bounds((r0, c0, r1, c1)).values():
                region = grid[qr0:qr1, qc0:qc1]
                pts = np.argwhere((region != bg) & (region != color))
                if pts.shape[0] != 1:
                    ok = False
                    break
                mr = int(qr0 + pts[0, 0])
                mc = int(qc0 + pts[0, 1])
                mcolor = int(grid[mr, mc])
                if mcolor in markers:
                    ok = False
                    break
                markers[mcolor] = (mr, mc)
            if not ok or len(markers) != 4:
                continue

            score = (
                side_h,
                -(abs((r0 + r1) - (h - 1)) + abs((c0 + c1) - (w - 1))),
            )
            if best is None or score > best[0]:
                best = (score, int(color), (r0, c0, r1, c1), markers)

    if best is None:
        raise AssertionError("frame not found")
    _, color, bbox, markers = best
    return color, bbox, markers


def _find_compass(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    candidates = []
    edge_positions = []
    for c in range(w - 2):
        edge_positions.append((0, c))
        edge_positions.append((h - 3, c))
    for r in range(1, h - 3):
        edge_positions.append((r, 0))
        edge_positions.append((r, w - 3))

    for sr, sc in edge_positions:
        if not (sr + 2 < h and sc + 2 < w):
            continue
        if not (sr + 2 < r0 or sr > r1 or sc + 2 < c0 or sc > c1):
            continue
        sub = grid[sr : sr + 3, sc : sc + 3]
        if int(sub[1, 1]) != frame_color:
            continue
        if any(int(sub[rr, cc]) != bg for rr, cc in ((0, 0), (0, 2), (2, 0), (2, 2))):
            continue
        cardinals = [int(sub[0, 1]), int(sub[1, 2]), int(sub[2, 1]), int(sub[1, 0])]
        active = [idx for idx, val in enumerate(cardinals) if val != bg and val != frame_color]
        if len(active) != 1:
            continue
        if any(cardinals[idx] != bg for idx in range(4) if idx != active[0]):
            continue
        dist = abs((sr + 1) * 2 - (h - 1)) + abs((sc + 1) * 2 - (w - 1))
        candidates.append((dist, sr, sc, active[0]))

    if not candidates:
        raise AssertionError("compass not found")

    _, sr, sc, direction = min(candidates)
    k_map = {0: 0, 1: 3, 2: 2, 3: 1}
    return sr, sc, k_map[direction]


def _extract_pieces(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    compass_pos: tuple[int, int],
    marker_colors: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    piece_zone = np.ones_like(grid, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    piece_zone[r0 : r1 + 1, c0 : c1 + 1] = False
    cr, cc = compass_pos
    piece_zone[cr : cr + 3, cc : cc + 3] = False

    pieces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for color in marker_colors:
        coords_list = _components((grid == color) & piece_zone)
        if not coords_list:
            raise AssertionError(f"piece for color {color} not found")
        coords = max(coords_list, key=lambda arr: arr.shape[0])
        pieces[int(color)] = (_crop_color_component(grid, coords, bg), coords)
    return pieces


def _assemble_output(
    grid: np.ndarray,
    bg: int,
    markers: dict[int, tuple[int, int]],
    pieces: dict[int, tuple[np.ndarray, np.ndarray]],
    rotation_k: int,
    compass_pos: tuple[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    cr, cc = compass_pos
    out[cr : cr + 3, cc : cc + 3] = bg
    for mr, mc in markers.values():
        out[mr, mc] = bg
    for _color, (_piece, coords) in pieces.items():
        out[coords[:, 0], coords[:, 1]] = bg
    for color, (piece, _coords) in pieces.items():
        rotated = np.rot90(piece, rotation_k)
        mr, mc = markers[color]
        top = int(mr - rotated.shape[0] // 2)
        left = int(mc - rotated.shape[1] // 2)
        _paste_non_bg(out, rotated, top, left, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the scattered colored fragments by the compass instruction and place them into the matching central puzzle quadrants.

    Input structure:
    - The background is black, with one large central square built from a grey border plus a grey cross that splits the interior into four equal quadrants.
    - Each quadrant contains exactly one single-pixel colored marker at its center, and the same four colors also appear on four small monochrome fragments outside the central square.
    - A 3x3 compass sits on one grid edge: its center is grey and exactly one cardinal neighbor cell is orange, encoding one global rotation.

    Rule:
    - Detect the central framed square and read the four marker colors and positions inside its quadrants.
    - Decode the compass direction as a shared rotation for all fragments: up = 0 degrees, right = 90 degrees clockwise, down = 180 degrees, left = 90 degrees counterclockwise.
    - Match each outside fragment to the quadrant marker of the same color, rotate the fragment, and translate it so the center of its bounding box lands on that marker; clear the original fragments and erase the compass.

    Key insight:
    - The transformation combines three cues: the grey square identifies the four destination quadrants, color equality links each fragment to its target, and the compass provides one rotation that must be applied uniformly before centering each fragment on its marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frame_bbox, markers = _find_frame(g, bg)
    compass_r, compass_c, rotation_k = _find_compass(g, bg, frame_color, frame_bbox)
    pieces = _extract_pieces(
        g,
        bg,
        frame_bbox,
        (compass_r, compass_c),
        list(markers.keys()),
    )
    return _assemble_output(
        g,
        bg,
        markers,
        pieces,
        rotation_k,
        (compass_r, compass_c),
    )


def _marker_positions(frame_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    side = r1 - r0 + 1
    q = (side - 3) // 2
    top_r = r0 + 1 + q // 2
    bot_r = r0 + q + 2 + q // 2
    left_c = c0 + 1 + q // 2
    right_c = c0 + q + 2 + q // 2
    return {
        "tl": (top_r, left_c),
        "tr": (top_r, right_c),
        "bl": (bot_r, left_c),
        "br": (bot_r, right_c),
    }


def _compass_top_left(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, w // 2 - 1
    if side == "bottom":
        return h - 3, w // 2 - 1
    if side == "left":
        return h // 2 - 1, 0
    return h // 2 - 1, w - 3


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[mid_r, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, mid_c] = FRAME_COLOR


def _colorize_shape(shape: np.ndarray, color: int) -> np.ndarray:
    return np.where(shape != 0, color, 0).astype(int)


def _choose_canvas(side: int, rng: random.Random) -> tuple[int, int]:
    candidates_h = [v for v in range(side + 10, 30, 2)]
    candidates_w = [v for v in range(side + 10, 30, 2)]
    if not candidates_h or not candidates_w:
        raise AssertionError("no valid canvas sizes")
    return rng.choice(candidates_h), rng.choice(candidates_w)


def _place_piece_in_region(
    grid: np.ndarray,
    piece: np.ndarray,
    region: tuple[int, int, int, int],
    rng: random.Random,
) -> np.ndarray | None:
    rr0, rr1, cc0, cc1 = region
    h, w = piece.shape
    max_top = rr1 - h + 1
    max_left = cc1 - w + 1
    if max_top < rr0 or max_left < cc0:
        return None
    tops = list(range(rr0, max_top + 1))
    lefts = list(range(cc0, max_left + 1))
    rng.shuffle(tops)
    rng.shuffle(lefts)
    for top in tops:
        for left in lefts:
            block = grid[top : top + h, left : left + w]
            if np.any((piece != 0) & (block != BG)):
                continue
            placed = grid.copy()
            _paste_non_bg(placed, piece, top, left, BG)
            return placed
    return None


def _build_expected(
    grid: np.ndarray,
    markers: dict[int, tuple[int, int]],
    pieces: dict[int, tuple[np.ndarray, np.ndarray]],
    rotation_k: int,
    compass_pos: tuple[int, int],
) -> np.ndarray:
    return _assemble_output(grid, BG, markers, pieces, rotation_k, compass_pos)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    q = rng.choice([5, 7])
    side = 2 * q + 3
    h, w = _choose_canvas(side, rng)
    grid = np.full((h, w), BG, dtype=int)

    frame_top = (h - side) // 2
    frame_left = (w - side) // 2
    frame_bbox = (frame_top, frame_left, frame_top + side - 1, frame_left + side - 1)
    _draw_frame(grid, frame_bbox)

    marker_cells = _marker_positions(frame_bbox)
    color_order = rng.sample(PIECE_COLORS, 4)
    labels = ["tl", "tr", "bl", "br"]
    markers: dict[int, tuple[int, int]] = {}
    for label, color in zip(labels, color_order):
        r, c = marker_cells[label]
        grid[r, c] = color
        markers[int(color)] = (r, c)

    compass_side = rng.choice(["top", "right", "bottom", "left"])
    compass_r, compass_c = _compass_top_left(compass_side, h, w)
    grid[compass_r + 1, compass_c + 1] = FRAME_COLOR
    direction = rng.randrange(4)
    drdc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
    grid[compass_r + 1 + drdc[0], compass_c + 1 + drdc[1]] = COMPASS_COLOR
    rotation_k = {0: 0, 1: 3, 2: 2, 3: 1}[direction]

    corners = {
        "tl": (0, frame_top - 1, 0, frame_left - 1),
        "tr": (0, frame_top - 1, frame_left + side, w - 1),
        "bl": (frame_top + side, h - 1, 0, frame_left - 1),
        "br": (frame_top + side, h - 1, frame_left + side, w - 1),
    }
    piece_colors = rng.sample(color_order, len(color_order))
    corner_order = rng.sample(labels, len(labels))
    chosen_shapes = rng.sample(SHAPE_LIBRARY, 4)
    pieces: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for color, corner, base_shape in zip(piece_colors, corner_order, chosen_shapes):
        piece = _colorize_shape(np.rot90(base_shape, rng.randrange(4)), color)
        if piece.shape[0] > q or piece.shape[1] > q:
            return None
        updated = _place_piece_in_region(grid, piece, corners[corner], rng)
        if updated is None:
            return None
        grid = updated
        coords = np.argwhere(grid == color)
        outside = []
        for r, c in coords:
            if not (frame_top <= int(r) <= frame_top + side - 1 and frame_left <= int(c) <= frame_left + side - 1):
                outside.append((int(r), int(c)))
        if not outside:
            return None
        piece_coords = np.array(outside, dtype=int)
        pieces[int(color)] = (_crop_color_component(grid, piece_coords, BG), piece_coords)

    expected = _build_expected(grid, markers, pieces, rotation_k, (compass_r, compass_c))

    quadrant_regions = _quadrant_bounds(frame_bbox)
    for color, (piece, _coords) in pieces.items():
        rotated = np.rot90(piece, rotation_k)
        mr, mc = markers[color]
        top = mr - rotated.shape[0] // 2
        left = mc - rotated.shape[1] // 2
        qr0, qr1, qc0, qc1 = next(
            bounds for bounds in quadrant_regions.values() if bounds[0] <= mr < bounds[1] and bounds[2] <= mc < bounds[3]
        )
        painted = np.argwhere(rotated != BG)
        if not len(painted):
            return None
        if int(rotated[rotated.shape[0] // 2, rotated.shape[1] // 2]) == BG:
            return None
        for rr, cc in painted:
            ar, ac = top + int(rr), left + int(cc)
            if not (qr0 <= ar < qr1 and qc0 <= ac < qc1):
                return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
