from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ORIENTATION_ORDER = ("identity", "hflip", "vflip", "both")
COLOR_BY_ORIENTATION = {
    "identity": BLUE,
    "hflip": GREEN,
    "vflip": YELLOW,
    "both": ORANGE,
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _apply_orientation(mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "identity":
        return mask.copy()
    if orientation == "hflip":
        return np.fliplr(mask)
    if orientation == "vflip":
        return np.flipud(mask)
    if orientation == "both":
        return np.flipud(np.fliplr(mask))
    raise ValueError(f"unknown orientation: {orientation}")


def _signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return arr.shape, arr.tobytes()


def _red_component(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one red object"
    return comps[0]


def _find_holes(grid: np.ndarray) -> list[dict[str, object]]:
    red_mask = grid == RED
    red_coords = _red_component(red_mask)
    r0, c0, r1, c1 = _bbox(red_coords)
    crop = red_mask[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict[str, object]] = []
    for coords in _components(~crop):
        rr = coords[:, 0]
        cc = coords[:, 1]
        if np.any(rr == 0) or np.any(rr == crop.shape[0] - 1) or np.any(cc == 0) or np.any(cc == crop.shape[1] - 1):
            continue
        hole_mask = _crop_mask(~crop, coords)
        hr0, hc0, hr1, hc1 = _bbox(coords)
        holes.append(
            {
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "mask": hole_mask,
            }
        )
    assert holes, "expected at least one interior hole"
    return holes


def _find_pieces(grid: np.ndarray) -> list[np.ndarray]:
    pieces = []
    for coords in _components(grid == GREY):
        pieces.append(_crop_mask(grid == GREY, coords))
    assert pieces, "expected at least one grey piece"
    return pieces


def _all_matches(pieces: list[np.ndarray], holes: list[dict[str, object]]) -> list[list[tuple[int, str]]]:
    matches: list[list[tuple[int, str]]] = []
    for piece in pieces:
        local: list[tuple[int, str]] = []
        for hole_idx, hole in enumerate(holes):
            hole_mask = hole["mask"]
            assert isinstance(hole_mask, np.ndarray)
            for orientation in ORIENTATION_ORDER:
                placed = _apply_orientation(piece, orientation)
                if placed.shape == hole_mask.shape and np.array_equal(placed, hole_mask):
                    local.append((hole_idx, orientation))
        matches.append(local)
    return matches


def _solve_matching(matches: list[list[tuple[int, str]]], hole_count: int) -> list[tuple[int, str]] | None:
    piece_count = len(matches)
    order = sorted(range(piece_count), key=lambda idx: len(matches[idx]))
    solution: list[tuple[int, str] | None] = [None] * piece_count
    used_holes = [False] * hole_count
    solutions_found = 0
    winner: list[tuple[int, str]] | None = None

    def dfs(pos: int) -> None:
        nonlocal solutions_found, winner
        if solutions_found > 1:
            return
        if pos == piece_count:
            solutions_found += 1
            winner = [(hole_idx, orientation) for hole_idx, orientation in solution if hole_idx is not None]
            return

        piece_idx = order[pos]
        for hole_idx, orientation in matches[piece_idx]:
            if used_holes[hole_idx]:
                continue
            used_holes[hole_idx] = True
            solution[piece_idx] = (hole_idx, orientation)
            dfs(pos + 1)
            solution[piece_idx] = None
            used_holes[hole_idx] = False

    dfs(0)
    if solutions_found != 1 or winner is None:
        return None
    return winner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each grey piece into the unique red-hole position and recolor it by the flip needed to make it fit.

    Input structure:
    - A black background contains one central 4-connected red object with an irregular outer boundary.
    - Inside the red object's bounding box there are several disjoint black hole shapes.
    - Separate grey 4-connected pieces lie away from the red object and match the holes only after specific flips.

    Rule:
    - Crop every grey piece and every interior hole to its tight bounding box.
    - For each piece, test the four allowed orientations: unchanged, left-right flip, up-down flip, and both flips.
    - Find the unique one-to-one assignment where a transformed piece exactly equals one hole shape, then fill that hole while erasing the original grey piece positions.
    - Color the filled hole blue for unchanged, green for left-right flip, yellow for up-down flip, and orange for both flips.

    Key insight:
    - The output color is not an intrinsic property of a piece; it encodes which flip search succeeds when matching that piece to a hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    holes = _find_holes(g)
    pieces = _find_pieces(g)
    assert len(holes) == len(pieces), "expected one piece per hole"

    matches = _all_matches(pieces, holes)
    assert all(matches_for_piece for matches_for_piece in matches), "every piece must match at least one hole"
    assignment = _solve_matching(matches, len(holes))
    assert assignment is not None, "expected a unique piece-hole matching"

    out = g.copy()
    out[out == GREY] = BLACK
    for hole_idx, orientation in assignment:
        r0, c0, _r1, _c1 = holes[hole_idx]["bbox"]
        hole_mask = holes[hole_idx]["mask"]
        assert isinstance(hole_mask, np.ndarray)
        color = COLOR_BY_ORIENTATION[orientation]
        patch = np.where(hole_mask, color, BLACK)
        view = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[hole_mask] = patch[hole_mask]

    return out


def _shape_orbit(shape: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_signature(_apply_orientation(shape, orientation)) for orientation in ORIENTATION_ORDER}


def _random_shape() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(3, 5)
        target = random.randint(3, min(8, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in DIR4
            if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
        }
        while int(mask.sum()) < target and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            mask[rr, cc] = True
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if any(np.array_equal(mask, _apply_orientation(mask, orientation)) for orientation in ORIENTATION_ORDER[1:]):
            continue
        return mask
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_holes(red_h: int, red_w: int, shapes: list[np.ndarray]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    margin = 4
    for shape in sorted(shapes, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        h, w = shape.shape
        positions = [
            (r, c)
            for r in range(margin, red_h - h - margin + 1)
            for c in range(margin, red_w - w - margin + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r, c in positions:
            box = (r, c, r + h - 1, c + w - 1)
            if any(_boxes_overlap(box, other, gap=1) for other in boxes):
                continue
            boxes.append(box)
            placements.append((r, c))
            placed = True
            break
        if not placed:
            return None
    by_shape = {id(shape): pos for shape, pos in zip(sorted(shapes, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True), placements)}
    return [by_shape[id(shape)] for shape in shapes]


def _carve_notches(mask: np.ndarray) -> None:
    h, w = mask.shape
    notch_count = random.randint(2, 4)
    for _ in range(notch_count):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            depth = random.randint(1, min(2, max(1, h // 5)))
            span = random.randint(2, max(2, w - 4))
            c0 = random.randint(1, w - span - 1)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            mask[rows, c0 : c0 + span] = False
        else:
            depth = random.randint(1, min(2, max(1, w // 5)))
            span = random.randint(2, max(2, h - 4))
            r0 = random.randint(1, h - span - 1)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            mask[r0 : r0 + span, cols] = False


def _place_pieces(
    grid: np.ndarray,
    red_bbox: tuple[int, int, int, int],
    piece_shapes: list[np.ndarray],
) -> bool:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    side_order = ["top", "bottom", "left", "right"]
    random.shuffle(side_order)

    for piece in piece_shapes:
        ph, pw = piece.shape
        patch = np.where(piece, GREY, BLACK)
        candidates: list[tuple[int, int]] = []
        for side in side_order:
            if side == "top":
                rows = range(0, min(4, h - ph + 1))
                cols = range(0, w - pw + 1)
            elif side == "bottom":
                start = max(0, h - ph - 3)
                rows = range(start, h - ph + 1)
                cols = range(0, w - pw + 1)
            elif side == "left":
                rows = range(0, h - ph + 1)
                cols = range(0, min(4, w - pw + 1))
            else:
                rows = range(0, h - ph + 1)
                start = max(0, w - pw - 3)
                cols = range(start, w - pw + 1)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0))
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _boxes_overlap(box, red_bbox, gap=1):
                continue
            if any(_boxes_overlap(box, other, gap=1) for other in placed_boxes):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(view[piece] != BLACK):
                continue
            view[piece] = GREY
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(
    grid: np.ndarray,
    hole_shapes: list[np.ndarray],
    hole_positions: list[tuple[int, int]],
    orientations: list[str],
    red_origin: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[out == GREY] = BLACK
    rr0, cc0 = red_origin
    for shape, (r, c), orientation in zip(hole_shapes, hole_positions, orientations):
        color = COLOR_BY_ORIENTATION[orientation]
        view = out[rr0 + r : rr0 + r + shape.shape[0], cc0 + c : cc0 + c + shape.shape[1]]
        view[shape] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    piece_count = random.choice([3, 4])
    orientations = list(random.sample(ORIENTATION_ORDER, piece_count)) if piece_count == 3 else list(ORIENTATION_ORDER)
    random.shuffle(orientations)

    hole_shapes: list[np.ndarray] = []
    orbit_signatures: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(piece_count):
        shape = None
        for _ in range(80):
            candidate = _random_shape()
            if candidate is None:
                continue
            orbit = _shape_orbit(candidate)
            if orbit & orbit_signatures:
                continue
            shape = candidate
            orbit_signatures |= orbit
            break
        if shape is None:
            return None
        hole_shapes.append(shape)

    red_h = random.randint(15, 19)
    red_w = random.randint(15, 19)
    hole_positions = _place_holes(red_h, red_w, hole_shapes)
    if hole_positions is None:
        return None

    H = random.randint(red_h + 7, min(30, red_h + 11))
    W = random.randint(red_w + 7, min(30, red_w + 11))
    red_r0 = random.randint(max(2, (H - red_h) // 2 - 1), min(H - red_h - 2, (H - red_h) // 2 + 1))
    red_c0 = random.randint(max(2, (W - red_w) // 2 - 1), min(W - red_w - 2, (W - red_w) // 2 + 1))

    red_mask = np.ones((red_h, red_w), dtype=bool)
    _carve_notches(red_mask)
    for shape, (r, c) in zip(hole_shapes, hole_positions):
        red_mask[r : r + shape.shape[0], c : c + shape.shape[1]][shape] = False
    if len(_components(red_mask)) != 1:
        return None

    grid = np.zeros((H, W), dtype=int)
    red_view = grid[red_r0 : red_r0 + red_h, red_c0 : red_c0 + red_w]
    red_view[red_mask] = RED

    piece_shapes = [_apply_orientation(shape, orientation) for shape, orientation in zip(hole_shapes, orientations)]
    if not _place_pieces(grid, (red_r0, red_c0, red_r0 + red_h - 1, red_c0 + red_w - 1), piece_shapes):
        return None

    expected = _build_expected_output(grid, hole_shapes, hole_positions, orientations, (red_r0, red_c0))
    red_ratio = float(np.count_nonzero(grid == RED)) / float(H * W)
    if not (0.24 <= red_ratio <= 0.42):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
