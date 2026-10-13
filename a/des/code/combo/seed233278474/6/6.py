import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VECTOR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_VECTOR)


# Generator configuration.
BG = 0
WALL = 5
ROOM_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 27
ROOM_COUNT_CHOICES = np.array([1, 2, 2, 3, 3, 4], dtype=int)
ROOM_OUTER_MIN = 6
ROOM_OUTER_MAX = 10
ROOM_PIXELS_MIN = 1
ROOM_PIXELS_MAX = 6
SPEC_TRIES = 100
PLACEMENT_TRIES = 160
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _room_info_from_component(coords: np.ndarray) -> dict[str, object] | None:
    top, left, bottom, right = _bbox(coords)
    height = bottom - top + 1
    width = right - left + 1
    if height < 5 or width < 5:
        return None

    local = {(int(r - top), int(c - left)) for r, c in coords.tolist()}
    perimeter = set()
    for c in range(width):
        perimeter.add((0, c))
        perimeter.add((height - 1, c))
    for r in range(height):
        perimeter.add((r, 0))
        perimeter.add((r, width - 1))

    if not local <= perimeter:
        return None
    if len(local) != len(perimeter) - 1:
        return None

    missing = perimeter - local
    if len(missing) != 1:
        return None
    gap_r, gap_c = next(iter(missing))
    if gap_r == 0 and 0 < gap_c < width - 1:
        side = "top"
    elif gap_r == height - 1 and 0 < gap_c < width - 1:
        side = "bottom"
    elif gap_c == 0 and 0 < gap_r < height - 1:
        side = "left"
    elif gap_c == width - 1 and 0 < gap_r < height - 1:
        side = "right"
    else:
        return None

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": height,
        "width": width,
        "side": side,
        "gap_abs": (top + gap_r, left + gap_c),
        "gap_rel": (gap_r, gap_c),
    }


def _find_rooms(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _color_components(grid, color)
        rooms: list[dict[str, object]] = []
        ok = True
        for comp in comps:
            room = _room_info_from_component(comp)
            if room is None:
                ok = False
                break
            rooms.append(room)
        if ok and rooms:
            candidates.append((color, rooms))
    assert len(candidates) == 1, "expected a unique wall color"
    wall_color, rooms = candidates[0]
    rooms.sort(key=lambda room: (int(room["top"]), int(room["left"])))
    return wall_color, rooms


def _room_pattern(grid: np.ndarray, room: dict[str, object], wall_color: int, bg: int) -> tuple[int, np.ndarray, tuple[int, int]]:
    top = int(room["top"])
    left = int(room["left"])
    bottom = int(room["bottom"])
    right = int(room["right"])

    pixels: list[tuple[int, int]] = []
    colors = set()
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            value = int(grid[r, c])
            if value in {bg, wall_color}:
                continue
            assert top + 1 < r < bottom - 1
            assert left + 1 < c < right - 1
            pixels.append((r, c))
            colors.add(value)
    assert pixels, "each room must contain a foreground pattern"
    assert len(colors) == 1, "each room must contain one foreground color"

    gap_r, gap_c = room["gap_abs"]
    distances = [abs(r - gap_r) + abs(c - gap_c) for r, c in pixels]
    best = min(distances)
    anchors = [pixel for pixel, dist in zip(pixels, distances) if dist == best]
    assert len(anchors) == 1, "anchor must be unique"
    anchor = anchors[0]

    side = str(room["side"])
    if side in {"top", "bottom"}:
        assert anchor[1] == gap_c
    else:
        assert anchor[0] == gap_r

    return colors.pop(), np.array(pixels, dtype=int), anchor


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each room's interior monochrome pattern outward through the wall gap."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    wall_color, rooms = _find_rooms(g, bg)

    moves: list[tuple[int, np.ndarray]] = []
    source_cells: list[tuple[int, int]] = []
    for room in rooms:
        color, pixels, anchor = _room_pattern(g, room, wall_color, bg)
        dr, dc = SIDE_TO_VECTOR[str(room["side"])]
        gap_r, gap_c = room["gap_abs"]
        outside = (gap_r + dr, gap_c + dc)
        shift = np.array([outside[0] - anchor[0], outside[1] - anchor[1]], dtype=int)
        dest = pixels + shift
        assert np.all(dest[:, 0] >= 0) and np.all(dest[:, 0] < g.shape[0])
        assert np.all(dest[:, 1] >= 0) and np.all(dest[:, 1] < g.shape[1])
        for rr, cc in dest.tolist():
            assert int(g[rr, cc]) == bg, "destinations must be empty background"
        source_cells.extend((int(r), int(c)) for r, c in pixels.tolist())
        moves.append((color, dest))

    out = g.copy()
    for r, c in source_cells:
        out[r, c] = bg
    for color, dest in moves:
        out[dest[:, 0], dest[:, 1]] = color
    return out


def _dilate(cells: set[tuple[int, int]], pad: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                out.add((r + dr, c + dc))
    return out


def _make_room_spec(rng: np.random.Generator, color: int) -> dict[str, object]:
    side = str(rng.choice(SIDES))
    height = int(rng.integers(ROOM_OUTER_MIN, ROOM_OUTER_MAX + 1))
    width = int(rng.integers(ROOM_OUTER_MIN, ROOM_OUTER_MAX + 1))

    if side in {"top", "bottom"}:
        gap_rel = (
            0 if side == "top" else height - 1,
            int(rng.integers(2, width - 2)),
        )
        anchor = (
            int(rng.integers(2, height - 2)),
            int(gap_rel[1]),
        )
    else:
        gap_rel = (
            int(rng.integers(2, height - 2)),
            0 if side == "left" else width - 1,
        )
        anchor = (
            int(gap_rel[0]),
            int(rng.integers(2, width - 2)),
        )

    safe_cells = [(r, c) for r in range(2, height - 2) for c in range(2, width - 2)]
    anchor_dist = abs(anchor[0] - gap_rel[0]) + abs(anchor[1] - gap_rel[1])
    candidates: list[tuple[int, int]] = []
    for r, c in safe_cells:
        if (r, c) == anchor:
            continue
        if abs(r - gap_rel[0]) + abs(c - gap_rel[1]) <= anchor_dist:
            continue
        if side == "top" and r > anchor[0]:
            continue
        if side == "bottom" and r < anchor[0]:
            continue
        if side == "left" and c > anchor[1]:
            continue
        if side == "right" and c < anchor[1]:
            continue
        candidates.append((r, c))

    max_pixels = min(ROOM_PIXELS_MAX, 1 + len(candidates))
    pixel_count = int(rng.integers(ROOM_PIXELS_MIN, max_pixels + 1))
    pixels = [anchor]
    if pixel_count > 1:
        picked = rng.choice(len(candidates), size=pixel_count - 1, replace=False)
        for idx in np.atleast_1d(picked):
            pixels.append(candidates[int(idx)])

    if side == "top":
        outside = (-1, gap_rel[1])
    elif side == "bottom":
        outside = (height, gap_rel[1])
    elif side == "left":
        outside = (gap_rel[0], -1)
    else:
        outside = (gap_rel[0], width)

    shift = (outside[0] - anchor[0], outside[1] - anchor[1])
    dest = {(r + shift[0], c + shift[1]) for r, c in pixels}
    room_cells = {(r, c) for r in range(height) for c in range(width)}
    active = room_cells | dest
    reserve = _dilate(active, pad=1)
    return {
        "color": color,
        "side": side,
        "height": height,
        "width": width,
        "gap_rel": gap_rel,
        "pixels": tuple(pixels),
        "active": active,
        "reserve": reserve,
    }


def _place_room(
    spec: dict[str, object],
    reserved: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    height = int(spec["height"])
    width = int(spec["width"])
    active = spec["active"]
    min_r = min(r for r, _ in active)
    max_r = max(r for r, _ in active)
    min_c = min(c for _, c in active)
    max_c = max(c for _, c in active)

    grid_h, grid_w = grid_shape
    top_low = max(1, -min_r)
    top_high = min(grid_h - height - 1, grid_h - 1 - max_r)
    left_low = max(1, -min_c)
    left_high = min(grid_w - width - 1, grid_w - 1 - max_c)
    if top_low > top_high or left_low > left_high:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = int(rng.integers(top_low, top_high + 1))
        left = int(rng.integers(left_low, left_high + 1))
        if all((top + r, left + c) not in reserved for r, c in active):
            return top, left
    return None


def _draw_room(grid: np.ndarray, spec: dict[str, object], top: int, left: int) -> None:
    height = int(spec["height"])
    width = int(spec["width"])
    gap_rel = spec["gap_rel"]

    for r in range(height):
        for c in range(width):
            if r in {0, height - 1} or c in {0, width - 1}:
                if (r, c) != gap_rel:
                    grid[top + r, left + c] = WALL

    color = int(spec["color"])
    for r, c in spec["pixels"]:
        grid[top + r, left + c] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    grid_w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    room_count = int(rng.choice(ROOM_COUNT_CHOICES))
    colors = [int(c) for c in rng.choice(ROOM_COLORS, size=room_count, replace=False)]

    reserved: set[tuple[int, int]] = set()
    placements: list[tuple[dict[str, object], int, int]] = []
    for color in colors:
        placed = False
        for _ in range(SPEC_TRIES):
            spec = _make_room_spec(rng, color)
            pos = _place_room(spec, reserved, (grid_h, grid_w), rng)
            if pos is None:
                continue
            top, left = pos
            for rr, cc in spec["reserve"]:
                ar = top + rr
                ac = left + cc
                if 0 <= ar < grid_h and 0 <= ac < grid_w:
                    reserved.add((ar, ac))
            placements.append((spec, top, left))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, top, left in placements:
        _draw_room(grid, spec, top, left)

    bg_count = int(np.count_nonzero(grid == BG))
    non_bg_counts = [int(np.count_nonzero(grid == color)) for color in np.unique(grid) if int(color) != BG]
    if not non_bg_counts or bg_count <= max(non_bg_counts):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
