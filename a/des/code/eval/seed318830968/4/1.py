import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 300


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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return _tight_mask_from_coords(coords)


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns))


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _normalize_mask(mask)
    return tuple(norm.shape), norm.tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_key(np.rot90(mask, k)) for k in range(4))


def _has_internal_hole(mask: np.ndarray) -> bool:
    m = _normalize_mask(mask)
    if m.size == 0:
        return False
    h, w = m.shape
    empty = ~m
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        cr, cc = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(empty & ~seen))


def _sorted_coords(coords: np.ndarray, rotation_cw: int) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    rot = int(rotation_cw) % 4
    if rot == 0:
        key = lambda rc: (rc[0], rc[1])
    elif rot == 1:
        key = lambda rc: (-rc[1], rc[0])
    elif rot == 2:
        key = lambda rc: (-rc[0], -rc[1])
    else:
        key = lambda rc: (rc[1], -rc[0])
    return sorted(pts, key=key)


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray):
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if sub.shape[0] < 3 or sub.shape[1] < 3:
        return None
    if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
        return None
    if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
        return None
    if np.any((sub != BG) & (sub != color)):
        return None
    holes = []
    for comp in _components(sub == BG):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1):
            continue
        holes.append(comp)
    if len(holes) != 1:
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "sub": sub.copy(),
        "hole_local": holes[0],
        "hole_mask": _tight_mask_from_coords(holes[0]),
    }


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    frames = []
    pieces = []
    for color in colors:
        for coords in _components(g == color):
            frame = _frame_info(g, color, coords)
            if frame is not None:
                frames.append(frame)
            else:
                pieces.append(
                    {
                        "color": int(color),
                        "bbox": _bbox(coords),
                        "mask": _tight_mask_from_coords(coords),
                    }
                )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    pieces.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames, pieces


def _assign_matches(frames, pieces) -> list[tuple[int, int]]:
    assert len(frames) == len(pieces) and frames, "expected equal non-zero counts of frames and pieces"
    candidates: list[list[tuple[int, int]]] = []
    for frame in frames:
        opts = []
        for pi, piece in enumerate(pieces):
            for rot in range(4):
                if np.array_equal(_rotate_cw(piece["mask"], rot), frame["hole_mask"]):
                    opts.append((pi, rot))
        assert opts, "every frame must have a matching piece"
        candidates.append(opts)
    order = sorted(range(len(frames)), key=lambda idx: len(candidates[idx]))
    used: set[int] = set()
    assignment: list[tuple[int, int] | None] = [None] * len(frames)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for pi, rot in candidates[fi]:
            if pi in used:
                continue
            used.add(pi)
            assignment[fi] = (pi, rot)
            if dfs(pos + 1):
                return True
            used.remove(pi)
            assignment[fi] = None
        return False

    assert dfs(0), "shape matching must admit a unique full assignment"
    return [item for item in assignment if item is not None]


def _build_output(frames, pieces, assignment) -> np.ndarray:
    min_r = min(frame["bbox"][0] for frame in frames)
    min_c = min(frame["bbox"][1] for frame in frames)
    max_r = max(frame["bbox"][2] for frame in frames)
    max_c = max(frame["bbox"][3] for frame in frames)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for fi, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        rr0 = r0 - min_r
        cc0 = c0 - min_c
        out[rr0 : rr0 + (r1 - r0 + 1), cc0 : cc0 + (c1 - c0 + 1)] = frame["sub"]
        pi, rot = assignment[fi]
        fill_color = pieces[pi]["color"]
        ordered = _sorted_coords(frame["hole_local"], rot)
        for r, c in ordered[: len(ordered) // 2]:
            out[rr0 + r, cc0 + c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each external colored fragment to a frame hole by rotation, then use
    the required clockwise rotation as the cue for how to half-fill that hole.
    """
    frames, pieces = _analyze_grid(np.asarray(grid, dtype=int))
    assignment = _assign_matches(frames, pieces)
    return _build_output(frames, pieces, assignment)


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        base_h = rng.randint(3, 6)
        base_w = rng.randint(3, 6)
        max_area = min(base_h * base_w - 1, 13)
        min_area = max(5, (base_h * base_w) // 3)
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)
        start = (rng.randrange(base_h), rng.randrange(base_w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[rng.randrange(len(frontier))])
        if len(cells) != target:
            continue
        mask = np.zeros((base_h, base_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        h, w = mask.shape
        if h < 2 or w < 2 or h > 7 or w > 7:
            continue
        if int(mask.sum()) < 5 or mask.all():
            continue
        if _has_internal_hole(mask):
            continue
        if len({_rotation_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a valid hole shape")


def _make_frame_patch(hole_mask: np.ndarray, rng: random.Random):
    hole_h, hole_w = hole_mask.shape
    frame_h = rng.randint(max(hole_h + 2, 6), min(hole_h + 4, 10))
    frame_w = rng.randint(max(hole_w + 2, 6), min(hole_w + 4, 10))
    top_pad = rng.randint(0, frame_h - 2 - hole_h)
    left_pad = rng.randint(0, frame_w - 2 - hole_w)
    patch = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    rr = 1 + top_pad
    cc = 1 + left_pad
    window = patch[rr : rr + hole_h, cc : cc + hole_w]
    window[hole_mask] = BG
    hole_coords = np.argwhere(patch == BG)
    return patch, hole_coords


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _expected_output(frame_specs) -> np.ndarray:
    min_r = min(spec["top"] for spec in frame_specs)
    min_c = min(spec["left"] for spec in frame_specs)
    max_r = max(spec["top"] + spec["frame"].shape[0] - 1 for spec in frame_specs)
    max_c = max(spec["left"] + spec["frame"].shape[1] - 1 for spec in frame_specs)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for spec in frame_specs:
        frame = spec["frame"]
        top = spec["top"] - min_r
        left = spec["left"] - min_c
        h, w = frame.shape
        out[top : top + h, left : left + w] = frame
        ordered = _sorted_coords(spec["hole_coords"], spec["rotation_cw"])
        for r, c in ordered[: len(ordered) // 2]:
            out[top + r, left + c] = spec["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    holes: list[np.ndarray] = []
    while len(holes) < n_frames:
        shape = _random_shape(rng)
        key = _canonical_shape_key(shape)
        if key in shape_keys:
            continue
        shape_keys.add(key)
        holes.append(shape)

    colors = list(rng.sample(PIECE_COLORS, n_frames))
    rotations = rng.sample([0, 1, 2, 3], n_frames)

    frame_specs = []
    piece_specs = []
    for hole_mask, color, rotation_cw in zip(holes, colors, rotations):
        frame_patch, hole_coords = _make_frame_patch(hole_mask, rng)
        piece_mask = np.rot90(hole_mask, rotation_cw)
        piece_patch = np.full(piece_mask.shape, color, dtype=int)
        frame_specs.append(
            {
                "frame": frame_patch,
                "hole_coords": hole_coords,
                "color": int(color),
                "rotation_cw": int(rotation_cw),
            }
        )
        piece_specs.append({"patch": piece_patch, "mask": piece_mask, "color": int(color)})

    total_bbox_area = sum(spec["frame"].shape[0] * spec["frame"].shape[1] for spec in frame_specs)
    total_bbox_area += sum(spec["patch"].shape[0] * spec["patch"].shape[1] for spec in piece_specs)
    min_dim = max(24, int(np.ceil(np.sqrt(total_bbox_area * 2.8))))
    max_dim = max(min_dim + 4, min(40, min_dim + 8))
    H = rng.randint(min_dim, max_dim)
    W = rng.randint(min_dim, max_dim)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    frame_order = list(range(n_frames))
    rng.shuffle(frame_order)
    for idx in sorted(frame_order, key=lambda i: frame_specs[i]["frame"].size, reverse=True):
        patch = frame_specs[idx]["frame"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, H - h)
            left = rng.randint(0, W - w)
            if not _can_place(occupied, top, left, h, w, pad=2):
                continue
            grid[top : top + h, left : left + w] = patch
            _place_bbox(occupied, top, left, h, w)
            frame_specs[idx]["top"] = int(top)
            frame_specs[idx]["left"] = int(left)
            placed = True
            break
        if not placed:
            return None

    piece_order = list(range(n_frames))
    rng.shuffle(piece_order)
    for idx in sorted(piece_order, key=lambda i: piece_specs[i]["patch"].size, reverse=True):
        patch = piece_specs[idx]["patch"]
        mask = piece_specs[idx]["mask"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, H - h)
            left = rng.randint(0, W - w)
            if not _can_place(occupied, top, left, h, w, pad=1):
                continue
            window = grid[top : top + h, left : left + w]
            window[mask] = patch[mask]
            _place_bbox(occupied, top, left, h, w)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(frame_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate several grey frames with distinct asymmetric hole shapes and
    matching external colored fragments under unique rotation assignments.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
