import random

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
ROOM_COUNT_RANGE = (1, 4)
ROOM_INNER_H_RANGE = (1, 6)
ROOM_INNER_W_RANGE = (1, 6)
KEY_SIZE_RANGE = (3, 5)
EXTRA_KEY_PRESENT_PROB = 0.5
EXTRA_KEY_COUNT_RANGE = (1, 2)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 500

_RNG = random.Random()


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
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
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


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        if not np.all(sub[border] != BG):
            continue
        if not np.all(sub[~border] == BG):
            continue
        gap_rel = np.argwhere(border & (sub != GREY))
        if gap_rel.shape[0] != 1:
            continue
        gr, gc = map(int, gap_rel[0])
        color = int(sub[gr, gc])
        if color in (BG, GREY):
            continue
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": (r0 + gr, c0 + gc),
                "color": color,
            }
        )
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-framed room using the color of its single border marker,
    restore the marker to grey, and remove the matching outside key object.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = _detect_rooms(g)
    assert rooms, "expected at least one marked room"

    gap_coords_by_color: dict[int, set[tuple[int, int]]] = {}
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        color = int(room["color"])
        gap = tuple(room["gap"])
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        out[gap] = GREY
        gap_coords_by_color.setdefault(color, set()).add(gap)

    for color, gap_coords in gap_coords_by_color.items():
        for comp in _components(g == color):
            cells = {tuple(map(int, cell)) for cell in comp}
            if cells & gap_coords:
                continue
            out[comp[:, 0], comp[:, 1]] = BG

    return out


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _random_polyomino(rng: random.Random, size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(NEIGHBORS4)
        cells.add((r + dr, c + dc))
    coords = _normalize(cells)
    if rng.random() < 0.5:
        coords[:, 1] *= -1
    turns = rng.randrange(4)
    for _ in range(turns):
        coords = np.stack([coords[:, 1], -coords[:, 0]], axis=1)
    coords = _normalize({(int(r), int(c)) for r, c in coords})
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    H, W = reserved.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    H, W = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    reserved[r0:r1, c0:c1] = True


def _place_box(rng: random.Random, reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = reserved.shape
    if h > H - 2 or w > W - 2:
        return None
    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(1, H - h - 1)
        left = rng.randint(1, W - w - 1)
        if _can_place(reserved, top, left, h, w):
            return top, left
    return None


def _random_gap(rng: random.Random, h: int, w: int) -> tuple[int, int]:
    side = rng.randrange(4)
    if side == 0:
        return 0, rng.randint(1, w - 2)
    if side == 1:
        return h - 1, rng.randint(1, w - 2)
    if side == 2:
        return rng.randint(1, h - 2), 0
    return rng.randint(1, h - 2), w - 1


def _draw_room(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int, rng: random.Random) -> dict:
    bottom = top + h - 1
    right = left + w - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    gr, gc = _random_gap(rng, h, w)
    gap = (top + gr, left + gc)
    grid[gap] = color
    return {"bbox": (top, left, bottom, right), "gap": gap, "color": int(color)}


def _place_shape(grid: np.ndarray, reserved: np.ndarray, mask: np.ndarray, color: int, rng: random.Random) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    pos = _place_box(rng, reserved, h, w)
    if pos is None:
        return None
    top, left = pos
    coords = np.argwhere(mask)
    coords = coords + np.array([top, left], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    _reserve(reserved, top, left, h, w)
    return coords


def _expected_output(grid: np.ndarray, rooms: list[dict], keys: list[dict]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(room["color"])
        out[tuple(room["gap"])] = GREY
    for key in keys:
        if key["matched"]:
            coords = key["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    if rng.random() < EXTRA_KEY_PRESENT_PROB:
        extra_count = rng.randint(*EXTRA_KEY_COUNT_RANGE)
    else:
        extra_count = 0
    if room_count + extra_count > len(COLORS):
        return None

    grid = np.zeros((H, W), dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    colors = rng.sample(COLORS, room_count + extra_count)
    room_colors = colors[:room_count]
    extra_colors = colors[room_count:]

    room_specs = []
    for color in room_colors:
        inner_h = rng.randint(*ROOM_INNER_H_RANGE)
        inner_w = rng.randint(*ROOM_INNER_W_RANGE)
        room_specs.append((color, inner_h + 2, inner_w + 2))
    room_specs.sort(key=lambda item: -(item[1] * item[2]))

    rooms = []
    for color, h, w in room_specs:
        pos = _place_box(rng, reserved, h, w)
        if pos is None:
            return None
        top, left = pos
        rooms.append(_draw_room(grid, top, left, h, w, color, rng))
        _reserve(reserved, top, left, h, w)

    key_specs = [{"color": color, "matched": True} for color in room_colors]
    key_specs.extend({"color": color, "matched": False} for color in extra_colors)
    rng.shuffle(key_specs)

    keys = []
    for spec in key_specs:
        size = rng.randint(*KEY_SIZE_RANGE)
        mask = _random_polyomino(rng, size)
        coords = _place_shape(grid, reserved, mask, spec["color"], rng)
        if coords is None:
            return None
        keys.append({"color": int(spec["color"]), "matched": bool(spec["matched"]), "coords": coords})

    expected = _expected_output(grid, rooms, keys)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid room-key puzzle input")
