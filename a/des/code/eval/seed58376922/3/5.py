import random
from collections import Counter

import numpy as np


BG = 8
FRAME = 5
WALL = 3
PIECE_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]

LAYOUT_OPTIONS = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2)]
SLOT_H_RANGE = (4, 6)
SLOT_W_RANGE = (4, 6)
EXTRA_MARGIN_RANGE = (8, 14)
FRAME_MARGIN_RANGE = (2, 6)
MAX_GEN_ATTEMPTS = 500
MAX_SLOT_ATTEMPTS = 200

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return (tuple(int(x) for x in trimmed.shape), trimmed.tobytes())


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _has_nontrivial_rotation_symmetry(mask: np.ndarray) -> bool:
    base = _mask_signature(mask)
    for k in range(1, 4):
        if _mask_signature(np.rot90(mask, k)) == base:
            return True
    return False


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    best_color = None
    best_coords = None
    best_area = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            area = int(len(coords))
            if area > best_area:
                best_area = area
                best_color = color
                best_coords = coords
    assert best_color is not None and best_coords is not None
    return int(best_color), best_coords


def _extract_slots(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    slots = []
    for coords in _components(grid == bg):
        if coords.size == 0:
            continue
        if np.any(coords[:, 0] <= fr0) or np.any(coords[:, 0] >= fr1):
            continue
        if np.any(coords[:, 1] <= fc0) or np.any(coords[:, 1] >= fc1):
            continue
        if np.any(coords[:, 0] < fr0) or np.any(coords[:, 0] > fr1):
            continue
        if np.any(coords[:, 1] < fc0) or np.any(coords[:, 1] > fc1):
            continue
        slots.append({"coords": coords, "mask": _coords_to_mask(coords)})
    slots.sort(key=lambda item: (int(item["coords"][:, 0].min()), int(item["coords"][:, 1].min())))
    return slots


def _extract_pieces(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    pieces = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        for coords in _components(grid == color):
            pieces.append({"color": color, "coords": coords, "mask": _coords_to_mask(coords)})
    pieces.sort(key=lambda item: (item["color"], int(item["coords"][:, 0].min()), int(item["coords"][:, 1].min())))
    return pieces


def _rotation_matches(piece_mask: np.ndarray, slot_mask: np.ndarray) -> list[int]:
    matches = []
    slot_sig = _mask_signature(slot_mask)
    for k in range(4):
        if _mask_signature(np.rot90(piece_mask, k)) == slot_sig:
            matches.append(int(k))
    return matches


def _unique_matching(edges: list[list[int]]) -> dict[int, int]:
    order = sorted(range(len(edges)), key=lambda idx: len(edges[idx]))
    used_slots: set[int] = set()
    assign: dict[int, int] = {}
    solutions: list[dict[int, int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(assign.copy())
            return
        piece_idx = order[pos]
        for slot_idx in edges[piece_idx]:
            if slot_idx in used_slots:
                continue
            used_slots.add(slot_idx)
            assign[piece_idx] = slot_idx
            dfs(pos + 1)
            del assign[piece_idx]
            used_slots.remove(slot_idx)

    dfs(0)
    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-frame cavity with the uniquely matching external piece color.

    Cavities are background components fully enclosed by the largest frame.
    External pieces are monochrome components whose masks match a cavity mask
    under one of the four right-angle rotations.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_color, frame_coords = _find_frame(g, bg)
    frame_bbox = _bbox(frame_coords)
    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True

    slots = _extract_slots(g, bg, frame_bbox)
    pieces = _extract_pieces(g, bg, frame_color)
    assert slots and len(slots) == len(pieces)

    edges: list[list[int]] = []
    for piece in pieces:
        candidates = []
        for slot_idx, slot in enumerate(slots):
            if _rotation_matches(piece["mask"], slot["mask"]):
                candidates.append(int(slot_idx))
        assert candidates
        edges.append(candidates)

    piece_to_slot = _unique_matching(edges)
    slot_to_piece = {slot_idx: piece_idx for piece_idx, slot_idx in piece_to_slot.items()}

    out = np.full_like(g, bg)
    out[frame_mask] = frame_color
    for slot_idx, slot in enumerate(slots):
        piece = pieces[slot_to_piece[slot_idx]]
        coords = slot["coords"]
        out[coords[:, 0], coords[:, 1]] = int(piece["color"])
    return out


def _sample_slot_shape(slot_h: int, slot_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    canvas = np.zeros((slot_h + 2, slot_w + 2), dtype=bool)
    canvas[1 : 1 + slot_h, 1 : 1 + slot_w] = True

    top_positions = list(range(1, slot_w - 1))
    side_positions = list(range(1, slot_h - 1))
    if not top_positions or not side_positions:
        return None

    features: list[tuple[str, int, int]] = []
    for side, choices in (("top", top_positions), ("bottom", top_positions), ("left", side_positions), ("right", side_positions)):
        pos = random.choice(choices)
        kind = random.choice([-1, 1])
        features.append((side, pos, kind))

    extra_sides = random.sample(["top", "bottom", "left", "right"], random.randint(0, 2))
    for side in extra_sides:
        choices = top_positions if side in ("top", "bottom") else side_positions
        base_positions = [pos for curr_side, pos, _kind in features if curr_side == side]
        avail = [pos for pos in choices if pos not in base_positions]
        if not avail:
            continue
        features.append((side, random.choice(avail), random.choice([-1, 1])))

    signs = {kind for _side, _pos, kind in features}
    if len(signs) < 2:
        return None

    for side, pos, kind in features:
        if side == "top":
            col = 1 + pos
            if kind == 1:
                canvas[0, col] = True
            else:
                canvas[1, col] = False
        elif side == "bottom":
            col = 1 + pos
            if kind == 1:
                canvas[slot_h + 1, col] = True
            else:
                canvas[slot_h, col] = False
        elif side == "left":
            row = 1 + pos
            if kind == 1:
                canvas[row, 0] = True
            else:
                canvas[row, 1] = False
        else:
            row = 1 + pos
            if kind == 1:
                canvas[row, slot_w + 1] = True
            else:
                canvas[row, slot_w] = False

    if len(_components(canvas)) != 1:
        return None

    shape = _trim_mask(canvas)
    if shape.size == 0 or _has_nontrivial_rotation_symmetry(shape):
        return None
    return canvas, shape


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _place_piece(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
    random.shuffle(positions)
    fg = patch != BG
    for r, c in positions:
        if blocked[r : r + h, c : c + w].any():
            continue
        target = grid[r : r + h, c : c + w]
        if np.any(target[fg] != BG):
            continue
        target[fg] = patch[fg]
        _reserve(blocked, r, c, h, w)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    slot_rows, slot_cols = random.choice(LAYOUT_OPTIONS)
    slot_h = random.randint(*SLOT_H_RANGE)
    slot_w = random.randint(*SLOT_W_RANGE)
    slot_count = slot_rows * slot_cols

    frame_h = 2 * WALL + slot_rows * slot_h + (slot_rows - 1) * WALL
    frame_w = 2 * WALL + slot_cols * slot_w + (slot_cols - 1) * WALL
    grid_h = frame_h + random.randint(*EXTRA_MARGIN_RANGE)
    grid_w = frame_w + random.randint(*EXTRA_MARGIN_RANGE)
    if grid_h <= frame_h + 3 or grid_w <= frame_w + 3:
        return None

    frame_r = random.randint(*FRAME_MARGIN_RANGE)
    frame_c = random.randint(*FRAME_MARGIN_RANGE)
    if frame_r + frame_h >= grid_h or frame_c + frame_w >= grid_w:
        frame_r = random.randint(1, grid_h - frame_h - 1)
        frame_c = random.randint(1, grid_w - frame_w - 1)

    slot_specs = []
    seen_keys = set()
    for _idx in range(slot_count):
        spec = None
        for _ in range(MAX_SLOT_ATTEMPTS):
            spec = _sample_slot_shape(slot_h, slot_w)
            if spec is None:
                continue
            carve_mask, shape_mask = spec
            key = _canonical_rotation_key(shape_mask)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            spec = {"carve": carve_mask, "shape": shape_mask}
            break
        if spec is None or not isinstance(spec, dict):
            return None
        slot_specs.append(spec)

    colors = random.sample(PIECE_COLORS, slot_count)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    grid[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = FRAME
    expected[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = FRAME

    piece_records = []
    for idx in range(slot_count):
        sr = idx // slot_cols
        sc = idx % slot_cols
        slot_r = frame_r + WALL + sr * (slot_h + WALL)
        slot_c = frame_c + WALL + sc * (slot_w + WALL)

        carve = slot_specs[idx]["carve"]
        start_r = slot_r - 1
        start_c = slot_c - 1
        area = grid[start_r : start_r + carve.shape[0], start_c : start_c + carve.shape[1]]
        area[carve] = BG

        expected_area = expected[start_r : start_r + carve.shape[0], start_c : start_c + carve.shape[1]]
        expected_area[carve] = colors[idx]

        rotation = random.randint(0, 3)
        piece_mask = _trim_mask(np.rot90(slot_specs[idx]["shape"], rotation))
        patch = np.full(piece_mask.shape, BG, dtype=int)
        patch[piece_mask] = colors[idx]
        piece_records.append({"patch": patch, "area": int(piece_mask.sum())})

    blocked = np.zeros_like(grid, dtype=bool)
    _reserve(blocked, frame_r, frame_c, frame_h, frame_w)
    piece_records.sort(key=lambda item: (-item["area"], item["patch"].shape[0], item["patch"].shape[1]))
    for record in piece_records:
        if not _place_piece(grid, blocked, record["patch"]):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
