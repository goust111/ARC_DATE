import numpy as np


BG_COLOR = 0
WALL_COLOR = 1
STRIPE_COLORS = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
ORIENTATIONS = ("up", "down", "left", "right")

GRID_MIN = 14
GRID_MAX = 25
ROOMS_MIN = 1
ROOMS_MAX = 4
ROOM_SIDE_MIN = 4
ROOM_SIDE_MAX = 8
OUTER_MARGIN = 2
ROOM_PAD = 1
PLACEMENT_TRIES = 100
GENERATION_RETRIES = 400

_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rooms(grid: np.ndarray) -> list[dict[str, int | str]]:
    h, w = grid.shape
    rooms: list[dict[str, int | str]] = []
    for comp in _components(grid == WALL_COLOR):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h < ROOM_SIDE_MIN or box_w < ROOM_SIDE_MIN:
            continue

        local = np.zeros((box_h, box_w), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True

        border = np.zeros_like(local)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if np.any(local & ~border):
            continue

        missing = np.argwhere(border & ~local)
        if missing.shape[0] != 1:
            continue

        mr, mc = map(int, missing[0])
        if mr == 0 and 0 < mc < box_w - 1:
            side = "up"
        elif mr == box_h - 1 and 0 < mc < box_w - 1:
            side = "down"
        elif mc == 0 and 0 < mr < box_h - 1:
            side = "left"
        elif mc == box_w - 1 and 0 < mr < box_h - 1:
            side = "right"
        else:
            continue

        rooms.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "side": side})

    rooms.sort(key=lambda room: (int(room["r0"]), int(room["c0"])))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue-walled room with stripes projected from the edge band
    indicated by its unique wall gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    for room in _find_rooms(g):
        r0 = int(room["r0"])
        c0 = int(room["c0"])
        r1 = int(room["r1"])
        c1 = int(room["c1"])
        side = str(room["side"])

        if side == "up":
            for c in range(c0 + 1, c1):
                out[r0 + 1 : r1, c] = g[0, c]
        elif side == "down":
            for c in range(c0 + 1, c1):
                out[r0 + 1 : r1, c] = g[h - 1, c]
        elif side == "left":
            for r in range(r0 + 1, r1):
                out[r, c0 + 1 : c1] = g[r, 0]
        elif side == "right":
            for r in range(r0 + 1, r1):
                out[r, c0 + 1 : c1] = g[r, w - 1]

    return out


def _make_stripe(length: int, rng: np.random.Generator) -> np.ndarray:
    n_runs = int(rng.integers(3, min(6, length) + 1))
    if n_runs == 1:
        return np.full(length, int(rng.choice(STRIPE_COLORS)), dtype=int)

    cuts = sorted(rng.choice(np.arange(1, length), size=n_runs - 1, replace=False).tolist())
    bounds = [0] + cuts + [length]
    colors = rng.choice(STRIPE_COLORS, size=n_runs, replace=False)

    stripe = np.empty(length, dtype=int)
    for idx in range(n_runs):
        stripe[bounds[idx] : bounds[idx + 1]] = int(colors[idx])
    return stripe


def _span_has_variation(values: np.ndarray, start: int, stop: int) -> bool:
    return len(set(map(int, values[start:stop].tolist()))) >= 2


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    room_h: int,
    room_w: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1
    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR

    if side == "up":
        gap_c = int(rng.integers(c0 + 1, c1))
        grid[r0, gap_c] = BG_COLOR
    elif side == "down":
        gap_c = int(rng.integers(c0 + 1, c1))
        grid[r1, gap_c] = BG_COLOR
    elif side == "left":
        gap_r = int(rng.integers(r0 + 1, r1))
        grid[gap_r, c0] = BG_COLOR
    else:
        gap_r = int(rng.integers(r0 + 1, r1))
        grid[gap_r, c1] = BG_COLOR


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_rooms = int(rng.integers(ROOMS_MIN, ROOMS_MAX + 1))
    min_side = max(GRID_MIN, 9 + 3 * num_rooms)
    h = int(rng.integers(min_side, GRID_MAX + 1))
    w = int(rng.integers(min_side, GRID_MAX + 1))

    top = _make_stripe(w, rng)
    bottom = _make_stripe(w, rng)
    left = _make_stripe(h, rng)
    right = _make_stripe(h, rng)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[0, :] = top
    grid[h - 1, :] = bottom
    grid[:, 0] = left
    grid[:, w - 1] = right

    orientations = list(rng.permutation(ORIENTATIONS))[:num_rooms]
    used_rects: list[tuple[int, int, int, int]] = []

    for side in orientations:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            max_room_h = min(ROOM_SIDE_MAX, h - 2 * OUTER_MARGIN)
            max_room_w = min(ROOM_SIDE_MAX, w - 2 * OUTER_MARGIN)
            room_h = int(rng.integers(ROOM_SIDE_MIN, max_room_h + 1))
            room_w = int(rng.integers(ROOM_SIDE_MIN, max_room_w + 1))

            max_r0 = h - OUTER_MARGIN - room_h
            max_c0 = w - OUTER_MARGIN - room_w
            if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
                return None

            r0 = int(rng.integers(OUTER_MARGIN, max_r0 + 1))
            c0 = int(rng.integers(OUTER_MARGIN, max_c0 + 1))
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1

            padded = (r0 - ROOM_PAD, c0 - ROOM_PAD, r1 + ROOM_PAD, c1 + ROOM_PAD)
            if any(_rects_overlap(padded, other) for other in used_rects):
                continue

            if side in ("up", "down"):
                source = top if side == "up" else bottom
                if not _span_has_variation(source, c0 + 1, c1):
                    continue
            else:
                source = left if side == "left" else right
                if not _span_has_variation(source, r0 + 1, r1):
                    continue

            _draw_room(grid, r0, c0, room_h, room_w, side, rng)
            used_rects.append(padded)
            placed = True
            break

        if not placed:
            return None

    rooms = _find_rooms(grid)
    found_sides = sorted(str(room["side"]) for room in rooms)
    if len(rooms) != num_rooms or found_sides != sorted(orientations):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-stripe puzzle")
