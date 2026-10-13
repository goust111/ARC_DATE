from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
ROTATION_STEPS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MAX_GENERATION_RETRIES = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _pattern_library() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1 << (h * w)):
                if bits.bit_count() < 3 or bits.bit_count() > 5:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not _is_connected(mask):
                    continue
                variants = []
                for k in range(4):
                    rot = np.rot90(mask, -k)
                    variants.append((rot.shape, rot.tobytes()))
                if len(set(variants)) != 4:
                    continue
                canonical = min(variants)
                if canonical in seen:
                    continue
                seen.add(canonical)
                patterns.append(mask)
    assert patterns, "expected at least one asymmetric pattern"
    return patterns


PATTERN_LIBRARY = _pattern_library()


def _detect_side(room: np.ndarray) -> str:
    h, w = room.shape
    assert h >= 5 and w >= 5, "room must have an interior"
    assert not np.any(room[1:-1, 1:-1] == BLUE), "blue border must be one cell thick"

    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    gap_coords = np.argwhere(perimeter & (room == BLACK))
    assert len(gap_coords) == 1, "room border must have exactly one gap"
    gap_r, gap_c = map(int, gap_coords[0])

    ok = room.copy()
    ok[gap_r, gap_c] = BLUE
    assert np.all(ok[perimeter] == BLUE), "all remaining perimeter cells must be blue"

    if gap_r == 0 and 0 < gap_c < w - 1:
        return "top"
    if gap_r == h - 1 and 0 < gap_c < w - 1:
        return "bottom"
    if gap_c == 0 and 0 < gap_r < h - 1:
        return "left"
    if gap_c == w - 1 and 0 < gap_r < h - 1:
        return "right"
    raise AssertionError("gap must lie on a non-corner border cell")


def _extract_pattern(interior: np.ndarray) -> np.ndarray:
    mask = (interior != BLACK) & (interior != BLUE)
    coords = np.argwhere(mask)
    assert coords.size > 0, "room interior must contain a pattern"
    r0, c0, r1, c1 = _bbox(coords)
    return interior[r0 : r1 + 1, c0 : c1 + 1].copy()


def _anchor_position(inner_shape: tuple[int, int], patch_shape: tuple[int, int], side: str) -> tuple[int, int]:
    ih, iw = inner_shape
    ph, pw = patch_shape
    assert ph <= ih and pw <= iw, "rotated pattern must fit the room interior"
    if side == "top":
        return 0, 0
    if side == "right":
        return 0, iw - pw
    if side == "bottom":
        return ih - ph, iw - pw
    if side == "left":
        return ih - ph, 0
    raise AssertionError("unknown room side")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room's corner pattern according to the side of the border gap.

    Input structure:
    - The background is black.
    - Several disconnected blue one-pixel-thick rectangular borders form rooms.
    - Each blue border has exactly one one-cell black gap on its top, right, bottom, or left side.
    - Inside each room is a small non-black, non-blue asymmetric pattern, initially occupying the upper-left part of the interior; all rooms share the same shape but may use different colors.

    Rule:
    - Keep every blue border and its black gap unchanged.
    - For each room, read the gap side: top, right, bottom, left correspond to 0, 90, 180, and 270 degrees clockwise.
    - Extract the room's interior pattern, rotate that pattern by the indicated amount, and clear the rest of the interior to black.
    - Place the rotated pattern into the matching corner of the room interior: upper-left, upper-right, lower-right, or lower-left.

    Key insight:
    - The border gap is the orientation code: it determines both how far the interior motif rotates and which corner the rotated motif must occupy.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = sorted(_components(g == BLUE), key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    for coords in rooms:
        r0, c0, r1, c1 = _bbox(coords)
        room = out[r0 : r1 + 1, c0 : c1 + 1]
        side = _detect_side(room)
        interior = room[1:-1, 1:-1]
        patch = _extract_pattern(interior)
        rotated = np.rot90(patch, -ROTATION_STEPS[side])
        interior[:] = BLACK
        pr, pc = _anchor_position(interior.shape, rotated.shape, side)
        interior[pr : pr + rotated.shape[0], pc : pc + rotated.shape[1]] = rotated
    return out


def _build_room(inner_h: int, inner_w: int, side: str, color: int, pattern_mask: np.ndarray) -> np.ndarray:
    room = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    room[0, :] = BLUE
    room[-1, :] = BLUE
    room[:, 0] = BLUE
    room[:, -1] = BLUE

    gap_r = room.shape[0] // 2
    gap_c = room.shape[1] // 2
    if side == "top":
        room[0, gap_c] = BLACK
    elif side == "right":
        room[gap_r, -1] = BLACK
    elif side == "bottom":
        room[-1, gap_c] = BLACK
    elif side == "left":
        room[gap_r, 0] = BLACK
    else:
        raise AssertionError("unknown room side")

    ph, pw = pattern_mask.shape
    patch = room[1 : 1 + ph, 1 : 1 + pw]
    patch[pattern_mask] = int(color)
    return room


def _candidate_positions(
    occupied: np.ndarray, room_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    rh, rw = room_shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - rh):
        for c in range(1, w - rw):
            rr0 = max(0, r - 1)
            rr1 = min(h, r + rh + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + rw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    return candidates


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    room_count = int(rng.integers(4, 7))
    min_side = 18 + 2 * max(0, room_count - 4)
    grid_h = int(rng.integers(min_side, 31))
    grid_w = int(rng.integers(min_side, 31))

    pattern_mask = PATTERN_LIBRARY[int(rng.integers(len(PATTERN_LIBRARY)))]
    min_inner = max(int(max(pattern_mask.shape)), 3)
    inner_choices = [d for d in (3, 5, 7) if d >= min_inner]
    if not inner_choices:
        return None

    colors = rng.choice(np.array(COLOR_POOL, dtype=int), size=room_count, replace=False)
    sides = list(SIDES)
    if room_count > 4:
        extra = [SIDES[int(i)] for i in rng.integers(0, 4, size=room_count - 4)]
        sides.extend(extra)
    rng.shuffle(sides)

    specs = []
    for idx in range(room_count):
        inner_h = int(rng.choice(inner_choices))
        inner_w = int(rng.choice(inner_choices))
        room = _build_room(inner_h, inner_w, sides[idx], int(colors[idx]), pattern_mask)
        specs.append({"room": room, "side": sides[idx]})

    order = sorted(
        range(room_count),
        key=lambda idx: int(specs[idx]["room"].shape[0] * specs[idx]["room"].shape[1]),
        reverse=True,
    )
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * room_count

    for idx in order:
        room = specs[idx]["room"]
        candidates = _candidate_positions(occupied, room.shape)
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        placements[idx] = (r, c)
        rh, rw = room.shape
        occupied[max(0, r - 1) : min(grid_h, r + rh + 1), max(0, c - 1) : min(grid_w, c + rw + 1)] = True

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, spec in enumerate(specs):
        pos = placements[idx]
        assert pos is not None
        r, c = pos
        room = spec["room"]
        rh, rw = room.shape
        grid[r : r + rh, c : c + rw] = room

    solved = main(grid)
    assert solved.shape == grid.shape
    assert not np.array_equal(solved, grid)
    assert max(grid.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-rotation puzzle")
