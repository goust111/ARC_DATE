import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_CHOICES = ((2, 2), (2, 3), (3, 2))
BOX_CHOICES = ((5, 7), (7, 5), (7, 7))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    assert bool(mask.any())
    return _crop_mask(mask.astype(bool))


def _rot_norm(mask: np.ndarray, k: int) -> np.ndarray:
    return _normalize_mask(np.rot90(mask, k))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = _normalize_mask(mask)
    return (tuple(int(x) for x in m.shape), m.astype(np.uint8).tobytes())


def _shape_from_profile(profile: tuple[int, int, int, int], box_h: int, box_w: int) -> np.ndarray:
    top, right, bottom, left = profile
    mask = np.zeros((box_h, box_w), dtype=bool)
    mask[1 : box_h - 1, 1 : box_w - 1] = True
    mid_r = box_h // 2
    mid_c = box_w // 2

    if top > 0:
        mask[0, mid_c] = True
    else:
        mask[1, mid_c] = False

    if right > 0:
        mask[mid_r, box_w - 1] = True
    else:
        mask[mid_r, box_w - 2] = False

    if bottom > 0:
        mask[box_h - 1, mid_c] = True
    else:
        mask[box_h - 2, mid_c] = False

    if left > 0:
        mask[mid_r, 0] = True
    else:
        mask[mid_r, 1] = False

    return mask


def _profile_layout(
    rows: int, cols: int, rng: random.Random
) -> list[list[tuple[int, int, int, int]]]:
    prof = [[[None, None, None, None] for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols - 1):
            s = rng.choice((-1, 1))
            prof[r][c][1] = s
            prof[r][c + 1][3] = -s

    for r in range(rows - 1):
        for c in range(cols):
            s = rng.choice((-1, 1))
            prof[r][c][2] = s
            prof[r + 1][c][0] = -s

    for c in range(cols):
        prof[0][c][0] = rng.choice((-1, 1))
        prof[rows - 1][c][2] = rng.choice((-1, 1))

    for r in range(rows):
        prof[r][0][3] = rng.choice((-1, 1))
        prof[r][cols - 1][1] = rng.choice((-1, 1))

    return [[tuple(int(v) for v in prof[r][c]) for c in range(cols)] for r in range(rows)]


def _valid_shape_set(shapes: list[np.ndarray]) -> bool:
    if not shapes:
        return False

    keys: list[set[tuple[tuple[int, int], bytes]]] = []
    for shape in shapes:
        comps = _components(shape)
        if len(comps) != 1:
            return False
        base = _mask_key(shape)
        rots = {_mask_key(np.rot90(shape, k)) for k in range(4)}
        if len(rots) != 4:
            return False
        if base not in rots:
            return False
        keys.append(rots)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] & keys[j]:
                return False
    return True


def _board_from_layout(
    layout: list[list[tuple[int, int, int, int]]], box_h: int, box_w: int
) -> tuple[np.ndarray, list[dict]]:
    rows = len(layout)
    cols = len(layout[0])
    board_h = rows * (box_h - 1) + 3
    board_w = cols * (box_w - 1) + 3
    board = np.full((board_h, board_w), FRAME, dtype=int)
    slots: list[dict] = []

    for r in range(rows):
        for c in range(cols):
            mask = _shape_from_profile(layout[r][c], box_h, box_w)
            r0 = 1 + r * (box_h - 1)
            c0 = 1 + c * (box_w - 1)
            section = board[r0 : r0 + box_h, c0 : c0 + box_w]
            section[mask] = BG
            slots.append(
                {
                    "row": r,
                    "col": c,
                    "origin": (r0, c0),
                    "mask": mask,
                    "norm": _normalize_mask(mask),
                }
            )
    return board, slots


def _extract_slots(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    frame_mask = grid == frame_color
    frame_comps = _components(frame_mask)
    assert frame_comps
    board_coords = max(frame_comps, key=lambda pts: int(pts.shape[0]))
    br0, bc0, br1, bc1 = _bbox(board_coords)

    slots: list[dict] = []
    for coords in _components(grid == bg):
        r0, c0, r1, c1 = _bbox(coords)
        if not (br0 < r0 and r1 < br1 and bc0 < c0 and c1 < bc1):
            continue
        touches_frame = False
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and bool(frame_mask[nr, nc]):
                    touches_frame = True
                    break
            if touches_frame:
                break
        if not touches_frame:
            continue
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords.copy(),
                "mask": _normalize_mask(local),
            }
        )

    slots.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    return slots


def _extract_pieces(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, frame_color}]
    pieces: list[dict] = []
    for color in colors:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            pieces.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords.copy(),
                    "mask": _normalize_mask(local),
                }
            )
    pieces.sort(key=lambda item: int(item["color"]))
    return pieces


def _match_pieces_to_slots(pieces: list[dict], slots: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for pi, piece in enumerate(pieces):
        opts: list[tuple[int, int]] = []
        for si, slot in enumerate(slots):
            for k in range(4):
                if np.array_equal(_rot_norm(piece["mask"], k), slot["mask"]):
                    opts.append((si, k))
        candidates[pi] = opts

    order = sorted(range(len(pieces)), key=lambda i: (len(candidates[i]), int(pieces[i]["color"])))
    used_slots: set[int] = set()
    assign: dict[int, tuple[int, int]] = {}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        pi = order[idx]
        for si, k in candidates[pi]:
            if si in used_slots:
                continue
            used_slots.add(si)
            assign[pi] = (si, k)
            if dfs(idx + 1):
                return True
            used_slots.remove(si)
            del assign[pi]
        return False

    ok = dfs(0)
    assert ok, "failed to match every piece to a unique slot"
    return assign


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    assert non_bg
    frame_color = max(non_bg, key=lambda c: int((g == c).sum()))

    slots = _extract_slots(g, bg, frame_color)
    pieces = _extract_pieces(g, bg, frame_color)
    assert slots and pieces and len(slots) == len(pieces)

    assignment = _match_pieces_to_slots(pieces, slots)

    out = np.full_like(g, bg)
    out[g == frame_color] = frame_color
    for pi, (si, _rot) in assignment.items():
        color = int(pieces[pi]["color"])
        coords = slots[si]["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered monochrome jigsaw pieces into the matching slots of the central frame.

    Input structure:
    - The background is black.
    - A large grey rectangular frame sits near the center, with grey separator lines dividing it into equal slot boxes.
    - Inside the frame, each slot is an empty black region whose outline has a one-pixel tab or notch at the midpoint of each side.
    - Outside the frame, several colored monochrome pieces appear as disconnected components; each piece matches one slot after some rotation.

    Rule:
    - Identify the grey frame and extract every enclosed black slot shape.
    - Extract each colored piece outside the frame and compare it against the slot shapes under the four right-angle rotations.
    - Match every piece to the unique slot with the same rotated outline, then place that piece's color into the slot.
    - Clear the original piece locations back to black while keeping the grey frame structure unchanged.

    Key insight:
    - The critical step is to treat each slot and piece as a shape-matching problem under rotation: the visible tabs and notches encode a unique contour, and only one rotation of one piece fits each empty slot.
    """
    return _solve(grid)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[rr, cc] = True


def _piece_candidates(mask: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    mh, mw = mask.shape
    coords = np.argwhere(mask)
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            placed = coords + np.array([r0, c0], dtype=int)
            if bool(blocked[placed[:, 0], placed[:, 1]].any()):
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _expected_output(
    grid_shape: tuple[int, int],
    board: np.ndarray,
    board_pos: tuple[int, int],
    slots: list[dict],
    slot_colors: list[int],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    br, bc = board_pos
    out[br : br + board.shape[0], bc : bc + board.shape[1]] = board
    for slot, color in zip(slots, slot_colors):
        r0, c0 = slot["origin"]
        region = out[br + r0 : br + r0 + slot["mask"].shape[0], bc + c0 : bc + c0 + slot["mask"].shape[1]]
        region[slot["mask"]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows, cols = rng.choice(GRID_CHOICES)
    box_h, box_w = rng.choice(BOX_CHOICES)
    slot_count = rows * cols

    layout = None
    board = None
    slots = None
    for _ in range(120):
        trial = _profile_layout(rows, cols, rng)
        trial_board, trial_slots = _board_from_layout(trial, box_h, box_w)
        if _valid_shape_set([slot["norm"] for slot in trial_slots]):
            layout = trial
            board = trial_board
            slots = trial_slots
            break
    if layout is None or board is None or slots is None:
        return None

    h_low = max(board.shape[0] + 8, 24 if slot_count <= 4 else 27)
    w_low = max(board.shape[1] + 8, 24 if slot_count <= 4 else 27)
    if h_low > 30 or w_low > 30:
        return None

    H = rng.randint(h_low, 30)
    W = rng.randint(w_low, 30)
    grid = np.full((H, W), BG, dtype=int)

    row_slack = H - board.shape[0] - 2
    col_slack = W - board.shape[1] - 2
    board_r = max(1, min(H - board.shape[0] - 1, (H - board.shape[0]) // 2 + rng.randint(-1, 1)))
    board_c = max(1, min(W - board.shape[1] - 1, (W - board.shape[1]) // 2 + rng.randint(-1, 1)))
    if row_slack < 0 or col_slack < 0:
        return None
    grid[board_r : board_r + board.shape[0], board_c : board_c + board.shape[1]] = board

    slot_colors = list(rng.sample(PIECE_COLORS, slot_count))
    piece_specs = []
    for slot, color in zip(slots, slot_colors):
        k = rng.randrange(4)
        piece_specs.append(
            {
                "color": int(color),
                "mask": _rot_norm(slot["mask"], k),
            }
        )
    rng.shuffle(piece_specs)

    blocked = np.zeros((H, W), dtype=bool)
    blocked[
        max(0, board_r - 1) : min(H, board_r + board.shape[0] + 1),
        max(0, board_c - 1) : min(W, board_c + board.shape[1] + 1),
    ] = True

    piece_positions: dict[int, tuple[int, int]] = {}
    for spec in piece_specs:
        cands = _piece_candidates(spec["mask"], blocked)
        rng.shuffle(cands)
        placed = False
        for r0, c0 in cands:
            coords = np.argwhere(spec["mask"]) + np.array([r0, c0], dtype=int)
            if bool(blocked[coords[:, 0], coords[:, 1]].any()):
                continue
            grid[coords[:, 0], coords[:, 1]] = int(spec["color"])
            _mark_blocked(blocked, coords)
            piece_positions[int(spec["color"])] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != BG:
        return None
    non_bg = [int(c) for c in np.unique(grid) if int(c) != BG]
    if not non_bg or max(non_bg, key=lambda c: int((grid == c).sum())) != FRAME:
        return None

    side_tags = set()
    for spec in piece_specs:
        r0, c0 = piece_positions[int(spec["color"])]
        center_r = r0 + spec["mask"].shape[0] / 2.0
        center_c = c0 + spec["mask"].shape[1] / 2.0
        board_center_r = board_r + board.shape[0] / 2.0
        board_center_c = board_c + board.shape[1] / 2.0
        if center_r < board_center_r:
            side_tags.add("top")
        else:
            side_tags.add("bottom")
        if center_c < board_center_c:
            side_tags.add("left")
        else:
            side_tags.add("right")
    if len(side_tags) < 3:
        return None

    expected = _expected_output((H, W), board, (board_r, board_c), slots, slot_colors)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jigsaw-frame input")
