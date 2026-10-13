import numpy as np
from collections import deque


BG = 0
WALL = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")
RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 400
ROOM_COUNT_RANGE = (2, 5)
GRID_SIDE_RANGE = (24, 31)
ROOM_SIZE_RANGE = (8, 13)
THICKNESS_CHOICES = (2, 3)
PLACEMENT_PAD = 1
PLACE_ATTEMPTS = 200


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
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if key not in seen:
            seen.add(key)
            rots.append(rot)
    return rots


def _canonical_shape(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()) for rot in _rotations(mask))


def _ring_mask(h: int, w: int, t: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    dist = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
    return dist < t


def _infer_wall_thickness(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    h, w = mask.shape
    rr, cc = coords[:, 0], coords[:, 1]
    dist = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
    return int(dist.max()) + 1


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components(grid == WALL):
        r0, c0, r1, c1 = _bbox_coords(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        grey = patch == WALL
        h, w = patch.shape
        if h < 5 or w < 5:
            continue
        t = _infer_wall_thickness(grey)
        if t <= 0 or 2 * t >= h or 2 * t >= w:
            continue
        ring = _ring_mask(h, w, t)
        if np.any(grey & ~ring):
            continue
        missing = ring & ~grey
        if not missing.any():
            continue
        comps = _components(missing)
        if len(comps) != 1:
            continue
        rows, cols = np.nonzero(missing)
        on_top = bool(np.all(rows < t))
        on_bottom = bool(np.all(rows >= h - t))
        on_left = bool(np.all(cols < t))
        on_right = bool(np.all(cols >= w - t))
        if sum((on_top, on_bottom, on_left, on_right)) != 1:
            continue
        if on_top and not (int(rows.min()) == 0 and int(rows.max()) == t - 1):
            continue
        if on_bottom and not (int(rows.min()) == h - t and int(rows.max()) == h - 1):
            continue
        if on_left and not (int(cols.min()) == 0 and int(cols.max()) == t - 1):
            continue
        if on_right and not (int(cols.min()) == w - t and int(cols.max()) == w - 1):
            continue
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "t": t,
                "missing": missing,
                "hole_shape": _tight_mask(missing),
            }
        )
    return rooms


def _extract_keys(grid: np.ndarray) -> list[dict]:
    keys = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, WALL)):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox_coords(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            keys.append(
                {
                    "color": color,
                    "coords": comp,
                    "shape": mask,
                }
            )
    return keys


def _matching(rooms: list[dict], keys: list[dict]) -> dict[int, int]:
    adjacency = []
    for room in rooms:
        want = room["hole_shape"]
        matches = []
        for ki, key in enumerate(keys):
            if any(np.array_equal(rot, want) for rot in _rotations(key["shape"])):
                matches.append(ki)
        adjacency.append(matches)

    key_to_room = [-1] * len(keys)

    def dfs(room_idx: int, seen: set[int]) -> bool:
        for key_idx in adjacency[room_idx]:
            if key_idx in seen:
                continue
            seen.add(key_idx)
            prev = key_to_room[key_idx]
            if prev == -1 or dfs(prev, seen):
                key_to_room[key_idx] = room_idx
                return True
        return False

    for room_idx in range(len(rooms)):
        dfs(room_idx, set())

    room_to_key = {}
    for key_idx, room_idx in enumerate(key_to_room):
        if room_idx != -1:
            room_to_key[room_idx] = key_idx
    return room_to_key


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key shape to the unique room wall gap it fits under rotation,
    then paint that room's whole interior and gap with the key color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = _extract_rooms(g)
    keys = _extract_keys(g)
    room_to_key = _matching(rooms, keys)

    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = BG

    for room_idx, key_idx in room_to_key.items():
        room = rooms[room_idx]
        color = int(keys[key_idx]["color"])
        r0, c0, r1, c1 = room["bbox"]
        t = int(room["t"])
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[room["missing"]] = color
        sub[t : sub.shape[0] - t, t : sub.shape[1] - t] = color

    return out


def _orient_base_hole(base: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return base
    if side == "bottom":
        return np.flipud(base)
    if side == "left":
        return np.rot90(base, 3)
    if side == "right":
        return np.rot90(base, 1)
    raise ValueError(side)


def _build_room(h: int, w: int, t: int, side: str, offset: int, base_hole: np.ndarray) -> dict:
    patch = np.zeros((h, w), dtype=int)
    patch[_ring_mask(h, w, t)] = WALL

    oriented = _orient_base_hole(base_hole, side)
    hole_mask = np.zeros((h, w), dtype=bool)
    oh, ow = oriented.shape
    if side == "top":
        r, c = 0, offset
    elif side == "bottom":
        r, c = h - t, offset
    elif side == "left":
        r, c = offset, 0
    else:
        r, c = offset, w - t
    hole_mask[r : r + oh, c : c + ow] = oriented
    patch[hole_mask] = BG

    return {
        "patch": patch,
        "hole_mask": hole_mask,
        "hole_shape": _tight_mask(hole_mask),
        "t": t,
        "side": side,
    }


def _make_base_hole(t: int, length: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((t, length), dtype=bool)
        r = 0
        c = int(RNG.integers(length))
        mask[r, c] = True

        steps = 0
        while r < t - 1 and steps < 4 * length:
            steps += 1
            moves = []
            if r < t - 1:
                moves.append((r + 1, c))
            if c > 0:
                moves.append((r, c - 1))
            if c + 1 < length:
                moves.append((r, c + 1))
            probs = np.array([0.55 if nr == r + 1 else 0.225 for nr, _ in moves], dtype=float)
            probs /= probs.sum()
            idx = int(RNG.choice(len(moves), p=probs))
            r, c = moves[idx]
            mask[r, c] = True
        if r != t - 1:
            continue

        target = int(RNG.integers(t + 2, min(t * length, t + length + 1)))
        for _ in range(3 * length):
            if int(mask.sum()) >= target:
                break
            frontier = set()
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < t and 0 <= nc < length and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(RNG.integers(len(frontier)))]
            mask[nr, nc] = True

        if not mask[0].any() or not mask[t - 1].any():
            continue
        if int(mask.sum()) < t + 2 or int(mask.sum()) >= t * length:
            continue
        if len(_components(mask)) != 1:
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _place_patch(blocked: np.ndarray, h: int, w: int, margin: int = 1) -> tuple[int, int] | None:
    H, W = blocked.shape
    if h + 1 >= H or w + 1 >= W:
        return None
    for _ in range(PLACE_ATTEMPTS):
        r = int(RNG.integers(1, H - h))
        c = int(RNG.integers(1, W - w))
        r0 = max(0, r - margin)
        c0 = max(0, c - margin)
        r1 = min(H, r + h + margin)
        c1 = min(W, c + w + margin)
        if not blocked[r0:r1, c0:c1].any():
            blocked[r:r + h, c:c + w] = True
            return r, c
    return None


def _sample_room(existing_shapes: set[tuple[int, int, bytes]]) -> dict | None:
    for _ in range(150):
        t = int(RNG.choice(THICKNESS_CHOICES))
        side = SIDES[int(RNG.integers(len(SIDES)))]
        h = int(RNG.integers(max(2 * t + 3, ROOM_SIZE_RANGE[0]), ROOM_SIZE_RANGE[1]))
        w = int(RNG.integers(max(2 * t + 3, ROOM_SIZE_RANGE[0]), ROOM_SIZE_RANGE[1]))
        run = w if side in ("top", "bottom") else h
        max_len = min(run - 2, 7)
        if max_len < 4:
            continue
        length = int(RNG.integers(4, max_len + 1))
        base = _make_base_hole(t, length)
        if base is None:
            continue
        shape = _canonical_shape(base)
        if shape in existing_shapes:
            continue
        offset = int(RNG.integers(1, run - length))
        room = _build_room(h, w, t, side, offset, base)
        if len(_extract_rooms(room["patch"])) != 1:
            continue
        room["canonical_shape"] = shape
        return room
    return None


def _gen_attempt() -> np.ndarray | None:
    room_count = int(RNG.integers(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1]))
    leave_unmatched = bool(room_count > 1 and RNG.integers(2) == 0)
    matched_count = room_count - (1 if leave_unmatched else 0)

    rooms = []
    seen_shapes: set[tuple[int, int, bytes]] = set()
    for _ in range(room_count):
        room = _sample_room(seen_shapes)
        if room is None:
            return None
        rooms.append(room)
        seen_shapes.add(room["canonical_shape"])

    matched_indices = set(int(i) for i in RNG.choice(room_count, size=matched_count, replace=False))
    key_colors = list(RNG.choice(KEY_COLORS, size=matched_count, replace=False))
    color_iter = iter(key_colors)
    key_specs = []
    for idx, room in enumerate(rooms):
        room["matched"] = idx in matched_indices
        room["fill_color"] = None
        if not room["matched"]:
            continue
        color = int(next(color_iter))
        room["fill_color"] = color
        rotation = int(RNG.integers(4))
        key_mask = np.rot90(room["hole_shape"], rotation)
        key_patch = np.zeros(key_mask.shape, dtype=int)
        key_patch[key_mask] = color
        key_specs.append({"patch": key_patch, "color": color})

    H = int(RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]))
    W = int(RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]))
    total_area = sum(room["patch"].shape[0] * room["patch"].shape[1] for room in rooms)
    total_area += sum(key["patch"].shape[0] * key["patch"].shape[1] for key in key_specs)
    if total_area > (H * W) // 2:
        return None

    grid = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    room_order = sorted(range(len(rooms)), key=lambda i: rooms[i]["patch"].shape[0] * rooms[i]["patch"].shape[1], reverse=True)
    for idx in room_order:
        patch = rooms[idx]["patch"]
        pos = _place_patch(blocked, patch.shape[0], patch.shape[1], margin=PLACEMENT_PAD)
        if pos is None:
            return None
        r, c = pos
        rooms[idx]["pos"] = (r, c)
        grid[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
        expected[r : r + patch.shape[0], c : c + patch.shape[1]] = patch

    for key in key_specs:
        patch = key["patch"]
        pos = _place_patch(blocked, patch.shape[0], patch.shape[1], margin=PLACEMENT_PAD)
        if pos is None:
            return None
        r, c = pos
        key["pos"] = (r, c)
        mask = patch != BG
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]

    for key in key_specs:
        r, c = key["pos"]
        patch = key["patch"]
        expected[r : r + patch.shape[0], c : c + patch.shape[1]][patch != BG] = BG

    for room in rooms:
        if not room["matched"]:
            continue
        color = int(room["fill_color"])
        r, c = room["pos"]
        patch = room["patch"]
        h, w = patch.shape
        t = int(room["t"])
        sub = expected[r : r + h, c : c + w]
        sub[room["hole_mask"]] = color
        sub[t : h - t, t : w - t] = color

    if len(_extract_rooms(grid)) != room_count:
        return None
    if len(_extract_keys(grid)) != matched_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid key-lock room puzzle")
