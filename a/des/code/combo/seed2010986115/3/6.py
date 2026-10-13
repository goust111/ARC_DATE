import numpy as np


BG = 0
FRAME = 5
PAYLOAD_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 32
ROOM_COUNT_MIN = 1
ROOM_COUNT_MAX = 4
ROOM_OUTER_MIN = 5
ROOM_OUTER_MAX = 9
MAX_PAYLOAD = 8
MAX_RETRIES = 400
PLACE_TRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _door_side(door: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = door
    r0, c0, r1, c1 = bbox
    matches = []
    if r == r0:
        matches.append("top")
    if r == r1:
        matches.append("bottom")
    if c == c0:
        matches.append("left")
    if c == c1:
        matches.append("right")
    assert len(matches) == 1, "door must lie on exactly one side"
    return matches[0]


def _sorted_payload(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    payload: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color not in (BG, FRAME):
                payload.append((r, c, color))

    if side == "top":
        payload.sort(key=lambda item: (item[0], item[1]))
    elif side == "bottom":
        payload.sort(key=lambda item: (r1 - item[0], item[1]))
    elif side == "left":
        payload.sort(key=lambda item: (item[1], item[0]))
    else:
        payload.sort(key=lambda item: (c1 - item[1], item[0]))
    return payload


def _parse_room(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "room must have a non-empty interior"

    perimeter = _perimeter_cells(r0, c0, r1, c1)
    assert len(comp) == len(perimeter) - 1, "room border must be a rectangle with one missing cell"

    comp_set = set(comp)
    assert all((r == r0 or r == r1 or c == c0 or c == c1) for r, c in comp_set), \
        "frame pixels must lie on the rectangle perimeter"

    missing = [(r, c) for r, c in perimeter if (r, c) not in comp_set]
    assert len(missing) == 1, "expected exactly one door"
    door = missing[0]
    assert int(grid[door]) == BG, "door cell must be background"
    side = _door_side(door, (r0, c0, r1, c1))

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert not np.any(interior == FRAME), "room interior must not contain border color"
    payload = _sorted_payload(grid, (r0, c0, r1, c1), side)

    return {
        "bbox": (r0, c0, r1, c1),
        "door": door,
        "side": side,
        "payload": payload,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each room's interior pixels outward through its door, ordered by distance from the door side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = [_parse_room(g, comp) for comp in _components(g == FRAME)]
    assert rooms, "expected at least one grey room"

    out = g.copy()
    placements: dict[tuple[int, int], int] = {}

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    h, w = g.shape
    for room in rooms:
        dr, dc = DIRS[room["side"]]
        door_r, door_c = room["door"]
        for idx, (_, _, color) in enumerate(room["payload"], start=1):
            nr = door_r + dr * idx
            nc = door_c + dc * idx
            assert 0 <= nr < h and 0 <= nc < w, "queue must stay inside the grid"
            assert int(g[nr, nc]) == BG, "queue path must start on background"
            key = (nr, nc)
            assert key not in placements, "queues from different rooms must not overlap"
            placements[key] = color

    for (r, c), color in placements.items():
        out[r, c] = color
    return out


def _queue_cells(door: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[side]
    return [(door[0] + dr * step, door[1] + dc * step) for step in range(1, length + 1)]


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _available_payload_cells(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    parity = int(rng.integers(0, 2))
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r + c) % 2 == parity
    ]
    rng.shuffle(cells)
    return cells


def _sample_room(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    side: str,
) -> tuple[dict, set[tuple[int, int]]] | None:
    h, w = grid_shape
    for _ in range(PLACE_TRIES):
        outer_h = int(rng.integers(ROOM_OUTER_MIN, ROOM_OUTER_MAX + 1))
        outer_w = int(rng.integers(ROOM_OUTER_MIN, ROOM_OUTER_MAX + 1))
        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        r0 = int(rng.integers(1, h - outer_h))
        c0 = int(rng.integers(1, w - outer_w))
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        exterior = {
            "top": r0,
            "bottom": h - 1 - r1,
            "left": c0,
            "right": w - 1 - c1,
        }[side]
        if exterior < 1:
            continue

        payload_cells = _available_payload_cells(rng, (r0, c0, r1, c1))
        max_payload = min(len(payload_cells), exterior, MAX_PAYLOAD)
        if max_payload < 1:
            continue

        if max_payload >= 2 and rng.random() < 0.85:
            count = int(rng.integers(2, max_payload + 1))
        else:
            count = int(rng.integers(1, max_payload + 1))
        payload_cells = payload_cells[:count]

        if side in ("top", "bottom"):
            door = (
                r0 if side == "top" else r1,
                int(rng.integers(c0 + 1, c1)),
            )
        else:
            door = (
                int(rng.integers(r0 + 1, r1)),
                c0 if side == "left" else c1,
            )

        queue = _queue_cells(door, side, count)
        if not all(0 <= r < h and 0 <= c < w for r, c in queue):
            continue

        room_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        queue_cells = set(queue)
        exact = room_cells | queue_cells
        expanded = _expand(exact, grid_shape, pad=1)
        if any(cell in blocked for cell in expanded):
            continue

        if not queue_cells.isdisjoint(room_cells):
            continue

        if count <= len(PAYLOAD_COLORS):
            colors = rng.choice(PAYLOAD_COLORS, size=count, replace=False)
        else:
            colors = rng.choice(PAYLOAD_COLORS, size=count, replace=True)

        return (
            {
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "door": door,
                "payload": list(zip(payload_cells, colors.tolist())),
            },
            expanded,
        )
    return None


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    door_r, door_c = room["door"]
    grid[door_r, door_c] = BG
    for (r, c), color in room["payload"]:
        grid[r, c] = int(color)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    room_count = int(rng.integers(ROOM_COUNT_MIN, ROOM_COUNT_MAX + 1))
    size_min = max(GRID_MIN, 14 + room_count * 3)
    h = int(rng.integers(size_min, GRID_MAX + 1))
    w = int(rng.integers(size_min, GRID_MAX + 1))

    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    rooms: list[dict] = []

    side_order = [str(side) for side in rng.permutation(SIDES)]
    while len(side_order) < room_count:
        side_order.append(str(rng.choice(SIDES)))
    rng.shuffle(side_order)

    for side in side_order[:room_count]:
        placed = _sample_room(rng, grid.shape, blocked, side)
        if placed is None:
            return None
        room, expanded = placed
        rooms.append(room)
        blocked |= expanded

    for room in rooms:
        _draw_room(grid, room)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
