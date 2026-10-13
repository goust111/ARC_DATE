import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
INTERFACE_COLORS = (1, 2, 3, 4)
BODY_COLORS = (6, 7, 8, 9)
SLOTS = ("top", "right", "bottom", "left")
SLOT_BOUNDARIES = {
    "top": ("main_upper", "anti_upper"),
    "right": ("anti_upper", "main_lower"),
    "bottom": ("main_lower", "anti_lower"),
    "left": ("main_upper", "anti_lower"),
}
SLOT_NEIGHBORS = {
    "top": (("left", "main_upper"), ("right", "anti_upper")),
    "right": (("top", "anti_upper"), ("bottom", "main_lower")),
    "bottom": (("right", "main_lower"), ("left", "anti_lower")),
    "left": (("top", "main_upper"), ("bottom", "anti_lower")),
}
MAX_GENERATION_ATTEMPTS = 300


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    labels = np.zeros(mask.shape, dtype=int)
    h, w = mask.shape
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            labels[r, c] = cid
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        queue.append((nr, nc))
    return labels, cid


def _slot_geometry(side: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], dict[str, np.ndarray | tuple[int, int, int, int]]]]:
    rr, cc = np.indices((side, side))
    half = side // 2

    bodies = {
        "top": (rr < cc) & (rr < side - 1 - cc),
        "right": (cc > rr) & (cc > side - 1 - rr),
        "bottom": (rr > cc) & (rr > side - 1 - cc),
        "left": (cc < rr) & (cc < side - 1 - rr),
    }

    boundaries = {name: np.zeros((side, side), dtype=bool) for name in {b for pair in SLOT_BOUNDARIES.values() for b in pair}}
    for i in range(half):
        boundaries["main_upper"][i, i] = True
        boundaries["anti_upper"][i, side - 1 - i] = True
    for i in range(half, side):
        boundaries["main_lower"][i, i] = True
        boundaries["anti_lower"][i, side - 1 - i] = True

    targets = {}
    for slot in SLOTS:
        for boundary_name in SLOT_BOUNDARIES[slot]:
            union = bodies[slot] | boundaries[boundary_name]
            r0, c0, r1, c1 = _bbox_from_mask(union)
            targets[(slot, boundary_name)] = {
                "bbox": (r0, c0, r1, c1),
                "mask": union[r0 : r1 + 1, c0 : c1 + 1],
                "body": bodies[slot][r0 : r1 + 1, c0 : c1 + 1],
                "interface": boundaries[boundary_name][r0 : r1 + 1, c0 : c1 + 1],
            }

    return bodies, boundaries, targets


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_mask = grid == FRAME
    assert frame_mask.any(), "frame not found"
    r0, c0, r1, c1 = _bbox_from_mask(frame_mask)
    assert r1 - r0 == c1 - c0, "frame must be square"
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame[0, :] == FRAME)
    assert np.all(frame[-1, :] == FRAME)
    assert np.all(frame[:, 0] == FRAME)
    assert np.all(frame[:, -1] == FRAME)
    return r0, c0, r1, c1


def _extract_pieces(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[int, dict[str, np.ndarray | tuple[int, int, int, int] | list[tuple[int, int]]]]:
    r0, c0, r1, c1 = frame_bbox
    mask = grid != BG
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    labels, n = _label_components(mask, connectivity=8)
    pieces = {}
    for cid in range(1, n + 1):
        comp = labels == cid
        br0, bc0, br1, bc1 = _bbox_from_mask(comp)
        patch = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        local_mask = comp[br0 : br1 + 1, bc0 : bc1 + 1]
        patch[~local_mask] = BG
        colors = {int(v) for v in patch[local_mask]}
        interface = list(colors & set(INTERFACE_COLORS))
        assert len(interface) == 1, "each piece must expose exactly one interface color"
        interface_color = interface[0]
        pieces[interface_color] = {
            "patch": patch,
            "bbox": (br0, bc0, br1, bc1),
            "coords": [(int(rr), int(cc)) for rr, cc in np.argwhere(comp)],
        }
    return pieces


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _target_patch(slot: str, boundary_name: str, side: int, body_color: int, interface_color: int) -> tuple[np.ndarray, dict[str, np.ndarray | tuple[int, int, int, int]]]:
    _bodies, _boundaries, targets = _slot_geometry(side)
    spec = targets[(slot, boundary_name)]
    patch = np.zeros(spec["mask"].shape, dtype=int)
    patch[spec["body"]] = int(body_color)
    patch[spec["interface"]] = int(interface_color)
    return patch, spec


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble loose triangle fragments into the matching missing slots of a central framed X-split puzzle.

    Input structure:
    - The background is black, and one grey square frame sits near the center.
    - Inside the frame, the even-sized square interior is split by two diagonals into four triangular wedges, with two wedges already body-colored and two wedges left black.
    - Each empty wedge exposes exactly one colored half-diagonal interface line, and two loose two-color triangular fragments with matching interface lines are scattered outside the frame.

    Rule:
    - Detect the framed square and rebuild its four wedge masks and four half-diagonal interface options from the frame size.
    - For each empty wedge, read which half-diagonal interface line is colored and use that interface color to find the matching loose fragment.
    - Rotate the fragment until its occupied mask exactly matches the target wedge-plus-interface shape, then paste it into the empty wedge and erase the fragment from its original location.
    - Keep the grey frame, the already filled wedges, and the visible interface pixels unchanged.

    Key insight:
    - The fragment's destination is determined by the exposed interface color, while the required rotation is determined by which half-diagonal edge inside the frame carries that color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    fr0, fc0, fr1, fc1 = _find_frame(g)
    side = fr1 - fr0 - 1
    bodies, boundaries, targets = _slot_geometry(side)
    interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]

    missing = {}
    for slot in SLOTS:
        slot_vals = interior[bodies[slot]]
        if np.all(slot_vals == BG):
            clue = None
            for boundary_name in SLOT_BOUNDARIES[slot]:
                boundary_vals = {int(v) for v in interior[boundaries[boundary_name]] if int(v) != BG}
                if boundary_vals:
                    assert len(boundary_vals) == 1, "interface line must use one color"
                    clue = (boundary_name, int(next(iter(boundary_vals))))
                    break
            assert clue is not None, "missing wedge must expose a clue line"
            missing[slot] = clue

    pieces = _extract_pieces(g, (fr0, fc0, fr1, fc1))
    for piece in pieces.values():
        for rr, cc in piece["coords"]:
            out[rr, cc] = BG

    for slot, (boundary_name, interface_color) in missing.items():
        assert interface_color in pieces, "missing interface piece"
        piece_patch = pieces[interface_color]["patch"]
        spec = targets[(slot, boundary_name)]
        placed = False
        for k in range(4):
            rotated = np.rot90(piece_patch, k)
            if rotated.shape != spec["mask"].shape:
                continue
            if not np.array_equal(rotated != BG, spec["mask"]):
                continue
            if not np.array_equal(rotated == interface_color, spec["interface"]):
                continue
            tr0, tc0, tr1, tc1 = spec["bbox"]
            region = out[fr0 + 1 + tr0 : fr0 + 1 + tr1 + 1, fc0 + 1 + tc0 : fc0 + 1 + tc1 + 1]
            region[spec["mask"]] = rotated[spec["mask"]]
            placed = True
            break
        assert placed, "piece rotation not found"

    return out


def _centered_start(total: int, size: int, rng: random.Random) -> int:
    base = (total - size) // 2
    lo = max(1, base - 2)
    hi = min(total - size - 1, base + 2)
    if hi < lo:
        lo = hi = max(0, min(total - size, base))
    return rng.randint(lo, hi)


def _compose_expected(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: int,
    slot_colors: dict[str, int],
    missing_specs: dict[str, tuple[str, int]],
    piece_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1 in piece_bboxes:
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch != BG] = BG

    fr0, fc0, _fr1, _fc1 = frame_bbox
    for slot, (boundary_name, interface_color) in missing_specs.items():
        patch, spec = _target_patch(slot, boundary_name, side, slot_colors[slot], interface_color)
        tr0, tc0, tr1, tc1 = spec["bbox"]
        region = out[fr0 + 1 + tr0 : fr0 + 1 + tr1 + 1, fc0 + 1 + tc0 : fc0 + 1 + tc1 + 1]
        region[spec["mask"]] = patch[spec["mask"]]
    return out


def _place_piece(
    grid: np.ndarray,
    patch: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    placed_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    frame_pad = (frame_bbox[0] - 1, frame_bbox[1] - 1, frame_bbox[2] + 1, frame_bbox[3] + 1)
    for _ in range(400):
        r0 = rng.randint(0, h - ph)
        c0 = rng.randint(0, w - pw)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _boxes_separated(box, frame_pad, pad=0):
            continue
        if any(not _boxes_separated(box, other, pad=1) for other in placed_boxes):
            continue
        mask = patch != BG
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(region[mask] != BG):
            continue
        region[mask] = patch[mask]
        return box
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice((8, 10, 12, 14))
    frame_size = side + 2
    min_dim = max(frame_size + 10, 22)
    if min_dim > 30:
        return None

    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)
    grid = np.zeros((h, w), dtype=int)

    fr0 = _centered_start(h, frame_size, rng)
    fc0 = _centered_start(w, frame_size, rng)
    fr1 = fr0 + frame_size - 1
    fc1 = fc0 + frame_size - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME

    bodies, boundaries, _targets = _slot_geometry(side)
    interior = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]

    slot_colors = dict(zip(SLOTS, rng.sample(BODY_COLORS, len(SLOTS))))
    filled = set(rng.sample(SLOTS, 2))
    missing = [slot for slot in SLOTS if slot not in filled]

    for slot in filled:
        interior[bodies[slot]] = slot_colors[slot]

    interface_values = rng.sample(INTERFACE_COLORS, len(missing))
    missing_specs: dict[str, tuple[str, int]] = {}
    for slot, interface_color in zip(missing, interface_values):
        options = [(neighbor, boundary_name) for neighbor, boundary_name in SLOT_NEIGHBORS[slot] if neighbor in filled]
        if not options:
            return None
        _neighbor, boundary_name = rng.choice(options)
        interior[boundaries[boundary_name]] = interface_color
        missing_specs[slot] = (boundary_name, interface_color)

    piece_boxes = []
    piece_order = missing[:]
    rng.shuffle(piece_order)
    for slot in piece_order:
        boundary_name, interface_color = missing_specs[slot]
        patch, _spec = _target_patch(slot, boundary_name, side, slot_colors[slot], interface_color)
        patch = np.rot90(patch, rng.randrange(4))
        box = _place_piece(grid, patch, frame_bbox, piece_boxes, rng)
        if box is None:
            return None
        piece_boxes.append(box)

    expected = _compose_expected(grid, frame_bbox, side, slot_colors, missing_specs, piece_boxes)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
