import random

import numpy as np


BG = 0
ROOM_COLOR = 8
PATTERN_COLORS = tuple(color for color in range(1, 10) if color != ROOM_COLOR)
DIRECTIONS = ("up", "down", "left", "right")

GRID_HEIGHT_RANGE = (24, 36)
GRID_WIDTH_RANGE = (24, 36)
ROOM_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (5, 8)
INNER_WIDTH_RANGE = (5, 8)
ROOM_SPACING_PAD = 2
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pts = np.argwhere(mask)
    assert pts.size > 0, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool), (int(r0), int(c0), int(r1), int(c1))


def _is_hv_symmetric(mask: np.ndarray) -> bool:
    tight, _ = _tight_crop(mask)
    return bool(np.array_equal(tight, np.flipud(tight)) and np.array_equal(tight, np.fliplr(tight)))


def _room_info(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    points = {(int(r), int(c)) for r, c in coords}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if not points.issubset(border):
        return None
    if len(points) != len(border) - 1:
        return None

    missing = list(border - points)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        gap_side = "up"
    elif gap_r == r1:
        gap_side = "down"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (int(gap_r), int(gap_c)),
        "gap_side": gap_side,
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    rooms: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            info = _room_info(coords)
            if info is None:
                continue
            info["color"] = int(color)
            rooms.append(info)
    rooms.sort(key=lambda item: item["bbox"])
    assert rooms, "expected at least one room"
    return rooms


def _find_defect_direction(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    assert mask.any(), "pattern mask must be non-empty"
    pts = np.argwhere(mask)
    candidates: list[str] = []
    for r, c in pts:
        reduced = mask.copy()
        reduced[int(r), int(c)] = False
        if not reduced.any():
            continue
        if not _is_hv_symmetric(reduced):
            continue
        _, (rr0, cc0, rr1, cc1) = _tight_crop(reduced)
        if int(r) < rr0:
            candidates.append("up")
        elif int(r) > rr1:
            candidates.append("down")
        elif int(c) < cc0:
            candidates.append("left")
        elif int(c) > cc1:
            candidates.append("right")
    assert len(candidates) == 1, f"expected one symmetry defect, got {candidates}"
    return candidates[0]


def _translation_for_room(room_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if direction == "up":
        return -(inner_h + 1), 0
    if direction == "down":
        return inner_h + 1, 0
    if direction == "left":
        return 0, -(inner_w + 1)
    assert direction == "right"
    return 0, inner_w + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each gap-framed room, detect the direction of the internal motif's unique
    symmetry-breaking protrusion and copy the room interior outward in that direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    rooms = _find_rooms(g, bg)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        room_color = int(room["color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_mask = (inner != bg) & (inner != room_color)
        assert pattern_mask.any(), "each room must contain a non-background pattern"
        pattern_colors = {int(v) for v in inner[pattern_mask]}
        assert len(pattern_colors) == 1, "each room pattern must use exactly one color"

        direction = _find_defect_direction(pattern_mask)
        dr, dc = _translation_for_room(room["bbox"], direction)

        pts = np.argwhere(pattern_mask)
        for rr, cc in pts:
            src_r = r0 + 1 + int(rr)
            src_c = c0 + 1 + int(cc)
            dst_r = src_r + dr
            dst_c = src_c + dc
            assert 0 <= dst_r < h and 0 <= dst_c < w, "copied pattern must stay in bounds"
            assert int(g[dst_r, dst_c]) == bg, "copied pattern must land on blank space"
            out[dst_r, dst_c] = int(g[src_r, src_c])
    return out


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight, _ = _tight_crop(mask)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _square(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def _plus(core: int, arm: int) -> np.ndarray:
    size = core + 2 * arm
    mask = np.zeros((size, size), dtype=bool)
    start = arm
    mask[:, start : start + core] = True
    mask[start : start + core, :] = True
    return mask


def _diamond(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                mask[r, c] = True
    return mask


def _ring(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _octagon(size: int, cut: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    for rr in range(cut):
        for cc in range(cut - rr):
            cells = [
                (rr, cc),
                (rr, size - 1 - cc),
                (size - 1 - rr, cc),
                (size - 1 - rr, size - 1 - cc),
            ]
            for r, c in cells:
                mask[r, c] = False
    return mask


def _build_base_shapes() -> list[np.ndarray]:
    shapes = [
        _square(3),
        _square(5),
        _plus(1, 1),
        _plus(1, 2),
        _plus(1, 3),
        _plus(3, 1),
        _plus(3, 2),
        _diamond(1),
        _diamond(2),
        _diamond(3),
        _ring(3),
        _ring(5),
        _ring(7),
        _octagon(5, 1),
        _octagon(7, 1),
        _octagon(7, 2),
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for shape in shapes:
        if not _is_hv_symmetric(shape):
            continue
        center_r = shape.shape[0] // 2
        center_c = shape.shape[1] // 2
        if not (
            shape[0, center_c]
            and shape[-1, center_c]
            and shape[center_r, 0]
            and shape[center_r, -1]
        ):
            continue
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        out.append(shape.copy())
    return out


BASE_SHAPES = _build_base_shapes()


def _add_protrusion(base: np.ndarray, direction: str) -> np.ndarray:
    h, w = base.shape
    center_r = h // 2
    center_c = w // 2
    if direction == "up":
        out = np.zeros((h + 1, w), dtype=bool)
        out[1:, :] = base
        out[0, center_c] = True
        return out
    if direction == "down":
        out = np.zeros((h + 1, w), dtype=bool)
        out[:-1, :] = base
        out[h, center_c] = True
        return out
    if direction == "left":
        out = np.zeros((h, w + 1), dtype=bool)
        out[:, 1:] = base
        out[center_r, 0] = True
        return out
    out = np.zeros((h, w + 1), dtype=bool)
    out[:, :-1] = base
    out[center_r, w] = True
    return out


def _sample_pattern(inner_h: int, inner_w: int, direction: str, rng: random.Random) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for base in BASE_SHAPES:
        mask = _add_protrusion(base, direction)
        if mask.shape[0] <= inner_h and mask.shape[1] <= inner_w:
            candidates.append(mask)
    if not candidates:
        return None
    return rng.choice(candidates).copy()


def _room_total_bbox(top: int, left: int, inner_h: int, inner_w: int, direction: str) -> tuple[int, int, int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if direction == "up":
        return top - inner_h, left, top + outer_h - 1, left + outer_w - 1
    if direction == "down":
        return top, left, top + outer_h - 1 + inner_h, left + outer_w - 1
    if direction == "left":
        return top, left - inner_w, top + outer_h - 1, left + outer_w - 1
    return top, left, top + outer_h - 1, left + outer_w - 1 + inner_w


def _bbox_is_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _candidate_positions(
    grid_h: int,
    grid_w: int,
    inner_h: int,
    inner_w: int,
    direction: str,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    positions: list[tuple[int, int]] = []
    for top in range(grid_h - outer_h + 1):
        for left in range(grid_w - outer_w + 1):
            total_bbox = _room_total_bbox(top, left, inner_h, inner_w, direction)
            if _bbox_is_free(occupied, total_bbox, ROOM_SPACING_PAD):
                positions.append((int(top), int(left)))
    return positions


def _place_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    direction: str,
    pattern_mask: np.ndarray,
    pattern_color: int,
    rng: random.Random,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    grid[top, left : left + outer_w] = ROOM_COLOR
    grid[top + outer_h - 1, left : left + outer_w] = ROOM_COLOR
    grid[top : top + outer_h, left] = ROOM_COLOR
    grid[top : top + outer_h, left + outer_w - 1] = ROOM_COLOR

    if direction == "up":
        gap_c = rng.randint(left + 1, left + outer_w - 2)
        grid[top, gap_c] = BG
    elif direction == "down":
        gap_c = rng.randint(left + 1, left + outer_w - 2)
        grid[top + outer_h - 1, gap_c] = BG
    elif direction == "left":
        gap_r = rng.randint(top + 1, top + outer_h - 2)
        grid[gap_r, left] = BG
    else:
        gap_r = rng.randint(top + 1, top + outer_h - 2)
        grid[gap_r, left + outer_w - 1] = BG

    ph, pw = pattern_mask.shape
    max_row_offset = inner_h - ph
    max_col_offset = inner_w - pw
    row_offset = rng.randint(0, max_row_offset)
    col_offset = rng.randint(0, max_col_offset)
    inner_top = top + 1 + row_offset
    inner_left = left + 1 + col_offset
    grid[inner_top : inner_top + ph, inner_left : inner_left + pw][pattern_mask] = int(pattern_color)


def _choose_directions(count: int, rng: random.Random) -> list[str]:
    dirs = list(DIRECTIONS)
    rng.shuffle(dirs)
    return dirs[:count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_HEIGHT_RANGE)
    grid_w = rng.randint(*GRID_WIDTH_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    room_count = rng.randint(*ROOM_COUNT_RANGE)
    directions = _choose_directions(room_count, rng)
    rng.shuffle(directions)

    for direction in directions:
        placed = False
        for _ in range(60):
            inner_h = rng.randint(*INNER_HEIGHT_RANGE)
            inner_w = rng.randint(*INNER_WIDTH_RANGE)
            pattern_mask = _sample_pattern(inner_h, inner_w, direction, rng)
            if pattern_mask is None:
                continue
            positions = _candidate_positions(grid_h, grid_w, inner_h, inner_w, direction, occupied)
            if not positions:
                continue
            top, left = rng.choice(positions)
            total_bbox = _room_total_bbox(top, left, inner_h, inner_w, direction)
            pattern_color = rng.choice(PATTERN_COLORS)
            _place_room(grid, top, left, inner_h, inner_w, direction, pattern_mask, pattern_color, rng)
            _mark_bbox(occupied, total_bbox)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid room-copy input")
