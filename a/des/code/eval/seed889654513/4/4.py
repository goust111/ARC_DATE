from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
AZURE = 8
NON_SPECIAL_COLORS = [1, 2, 4, 5, 6, 7, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

MIN_ROOM_SIDE = 3
ROOM_COUNT_RANGE = (3, 6)
TOP_HEIGHT_RANGE = (12, 20)
TOP_WIDTH_RANGE = (15, 24)
MAX_GENERATION_ATTEMPTS = 300
PROTRUSION_PROBABILITY = 0.65

PATTERN_LIBRARY = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int),
)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREEN, axis=1))
    assert len(rows) == 1, "expected exactly one full-width green separator"
    return int(rows[0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _legend_mapping(grid: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    lower = grid[sep + 1 :]
    assert lower.shape[0] >= 3, "legend area must contain a 3x3 pattern band"

    active_rows = [
        r for r in range(lower.shape[0]) if np.any((lower[r] != BLACK) & (lower[r] != AZURE))
    ]
    assert active_rows, "legend rows not found"
    r0 = min(active_rows)
    r1 = max(active_rows)
    assert r1 - r0 == 2, "legend must occupy exactly three rows"
    band = lower[r0 : r1 + 1]

    active_cols = [
        c for c in range(band.shape[1]) if np.any((band[:, c] != BLACK) & (band[:, c] != AZURE))
    ]
    assert active_cols, "legend columns not found"

    spans: list[tuple[int, int]] = []
    start = prev = active_cols[0]
    for c in active_cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        spans.append((start, prev))
        start = prev = c
    spans.append((start, prev))

    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in spans:
        assert c1 - c0 == 2, "each legend tile must be 3 columns wide"
        tile = band[:, c0 : c1 + 1].copy()
        colors = [int(x) for x in np.unique(tile) if int(x) not in {BLACK, AZURE}]
        assert len(colors) == 1, "each legend tile must use azure plus one foreground color"
        color = colors[0]
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = tile
    return mapping


def _room_components(grid: np.ndarray, sep: int) -> list[np.ndarray]:
    top = grid[:sep]
    return _connected_components(top != BLACK)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black-walled room according to the legend color and seed-count parity.

    Input structure:
    - The area above the green separator contains black walls around azure rooms.
    - Below the separator sits a legend of 3x3 two-color texture tiles separated by black columns.
    - Every seeded room contains one foreground color that matches one legend tile.

    Rule:
    - The seed color selects the 3x3 tile with the same non-azure color.
    - If a room contains an odd number of seeds, tile that pattern across the full room,
      aligned to the global grid origin.
    - If the seed count is even, leave the room unchanged.
    - Seeds are overwritten when filling happens.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    mapping = _legend_mapping(g, sep)
    rooms = _room_components(g, sep)

    out = g.copy()
    top = g[:sep]
    for comp in rooms:
        rr = comp[:, 0]
        cc = comp[:, 1]
        values = top[rr, cc]
        seed_colors = [int(x) for x in np.unique(values) if int(x) != AZURE]
        if not seed_colors:
            continue
        assert len(seed_colors) == 1, "each room must contain exactly one seed color"
        seed_color = seed_colors[0]
        assert seed_color in mapping, "room seed color must appear in the legend"
        seed_count = int(np.count_nonzero(values == seed_color))
        if seed_count % 2 == 0:
            continue
        tile = mapping[seed_color]
        out[rr, cc] = tile[rr % 3, cc % 3]
    return out


def _split_options(room: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = room
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    options: list[str] = []
    if w >= 2 * MIN_ROOM_SIDE + 1:
        options.append("v")
    if h >= 2 * MIN_ROOM_SIDE + 1:
        options.append("h")
    return options


def _partition_top(top_h: int, width: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    top = np.full((top_h, width), AZURE, dtype=int)
    top[0, :] = BLACK
    top[-1, :] = BLACK
    top[:, 0] = BLACK
    top[:, -1] = BLACK

    rooms = [(1, 1, top_h - 2, width - 2)]
    target_rooms = int(RNG.integers(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1] + 1))

    for _ in range(80):
        if len(rooms) >= target_rooms:
            break
        indexed = [(idx, room, _split_options(room)) for idx, room in enumerate(rooms)]
        indexed = [item for item in indexed if item[2]]
        if not indexed:
            break
        weights = np.array(
            [(room[2] - room[0] + 1) * (room[3] - room[1] + 1) for _, room, _ in indexed],
            dtype=float,
        )
        pick = int(RNG.choice(len(indexed), p=weights / weights.sum()))
        room_idx, room, options = indexed[pick]
        r0, c0, r1, c1 = room
        orientation = str(RNG.choice(options))
        rooms.pop(room_idx)
        if orientation == "v":
            wall_c = int(RNG.integers(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE + 1))
            top[r0 : r1 + 1, wall_c] = BLACK
            rooms.append((r0, c0, r1, wall_c - 1))
            rooms.append((r0, wall_c + 1, r1, c1))
        else:
            wall_r = int(RNG.integers(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE + 1))
            top[wall_r, c0 : c1 + 1] = BLACK
            rooms.append((r0, c0, wall_r - 1, c1))
            rooms.append((wall_r + 1, c0, r1, c1))

    if len(rooms) < ROOM_COUNT_RANGE[0]:
        return None
    return top, rooms


def _add_protrusion(top: np.ndarray, room: tuple[int, int, int, int]) -> None:
    if RNG.random() >= PROTRUSION_PROBABILITY:
        return

    r0, c0, r1, c1 = room
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    options: list[str] = []
    if w >= 5 and h >= 3:
        options.extend(["left", "right"])
    if h >= 5 and w >= 3:
        options.extend(["top", "bottom"])
    if not options:
        return

    side = str(RNG.choice(options))
    if side == "left":
        rr = int(RNG.integers(r0 + 1, r1))
        length = int(RNG.integers(1, w - 1))
        top[rr, c0 : c0 + length] = BLACK
    elif side == "right":
        rr = int(RNG.integers(r0 + 1, r1))
        length = int(RNG.integers(1, w - 1))
        top[rr, c1 - length + 1 : c1 + 1] = BLACK
    elif side == "top":
        cc = int(RNG.integers(c0 + 1, c1))
        length = int(RNG.integers(1, h - 1))
        top[r0 : r0 + length, cc] = BLACK
    else:
        cc = int(RNG.integers(c0 + 1, c1))
        length = int(RNG.integers(1, h - 1))
        top[r1 - length + 1 : r1 + 1, cc] = BLACK


def _build_top_layout(top_h: int, width: int) -> np.ndarray | None:
    result = _partition_top(top_h, width)
    if result is None:
        return None
    top, rooms = result
    for room in rooms:
        _add_protrusion(top, room)

    comps = _connected_components(top != BLACK)
    if len(comps) < ROOM_COUNT_RANGE[0]:
        return None
    return top


def _sample_seed_positions(top: np.ndarray, comp: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    all_cells = [tuple(cell) for cell in comp.tolist()]
    roomy_cells = [
        (r, c)
        for r, c in all_cells
        if sum(
            1
            for dr, dc in DIR4
            if 0 <= r + dr < top.shape[0] and 0 <= c + dc < top.shape[1] and top[r + dr, c + dc] == BLACK
        )
        == 0
    ]
    candidates = roomy_cells if len(roomy_cells) >= count else all_cells

    for min_dist in (4, 3, 2, 1, 0):
        for _ in range(40):
            chosen: list[tuple[int, int]] = []
            for idx in RNG.permutation(len(candidates)):
                r, c = candidates[int(idx)]
                if all(abs(r - rr) + abs(c - cc) >= min_dist for rr, cc in chosen):
                    chosen.append((r, c))
                    if len(chosen) == count:
                        return chosen
    return None


def _build_parities(n_rooms: int) -> list[int]:
    if n_rooms == 1:
        return [int(RNG.integers(0, 2))]
    odd_count = n_rooms // 2 if RNG.random() < 0.5 else (n_rooms + 1) // 2
    parities = [1] * odd_count + [0] * (n_rooms - odd_count)
    RNG.shuffle(parities)
    return parities


def _tile_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full((3, 3), AZURE, dtype=int)
    tile[mask.astype(bool)] = int(color)
    return tile


def _validate_generated(grid: np.ndarray) -> bool:
    sep = _find_separator_row(grid)
    mapping = _legend_mapping(grid, sep)
    rooms = _room_components(grid, sep)
    if len(rooms) < ROOM_COUNT_RANGE[0]:
        return False

    odd_rooms = 0
    even_rooms = 0
    out = main(grid)
    for comp in rooms:
        rr = comp[:, 0]
        cc = comp[:, 1]
        values = grid[rr, cc]
        seed_colors = [int(x) for x in np.unique(values) if int(x) != AZURE]
        if len(seed_colors) != 1:
            return False
        seed_color = seed_colors[0]
        if seed_color not in mapping:
            return False
        seed_count = int(np.count_nonzero(values == seed_color))
        changed = bool(np.any(out[rr, cc] != grid[rr, cc]))
        if seed_count % 2 == 1:
            odd_rooms += 1
            if not changed:
                return False
        else:
            even_rooms += 1
            if changed:
                return False

    if len(rooms) >= 2 and (odd_rooms == 0 or even_rooms == 0):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = int(RNG.integers(TOP_HEIGHT_RANGE[0], TOP_HEIGHT_RANGE[1] + 1))
    width = int(RNG.integers(TOP_WIDTH_RANGE[0], TOP_WIDTH_RANGE[1] + 1))

    max_patterns = min(4, len(NON_SPECIAL_COLORS), (width + 1) // 4)
    if max_patterns < 2:
        return None

    top = _build_top_layout(top_h, width)
    if top is None:
        return None

    comps = _connected_components(top != BLACK)
    n_rooms = len(comps)
    if n_rooms < ROOM_COUNT_RANGE[0]:
        return None

    legend_count = int(RNG.integers(2, min(max_patterns, n_rooms) + 1))
    legend_colors = list(RNG.choice(NON_SPECIAL_COLORS, size=legend_count, replace=False))
    pattern_ids = list(RNG.choice(len(PATTERN_LIBRARY), size=legend_count, replace=False))
    legend_tiles = {
        int(color): _tile_from_mask(PATTERN_LIBRARY[int(pid)], int(color))
        for color, pid in zip(legend_colors, pattern_ids)
    }

    room_colors = legend_colors.copy()
    while len(room_colors) < n_rooms:
        room_colors.append(int(RNG.choice(legend_colors)))
    RNG.shuffle(room_colors)
    parities = _build_parities(n_rooms)

    seeded_top = top.copy()
    for comp, seed_color, parity in zip(comps, room_colors, parities):
        max_seed = min(6, max(2, len(comp) // 4))
        options = [n for n in range(1, max_seed + 1) if n % 2 == parity]
        if not options:
            return None
        seed_count = int(RNG.choice(options))
        positions = _sample_seed_positions(seeded_top, comp, seed_count)
        if positions is None:
            return None
        for r, c in positions:
            seeded_top[r, c] = int(seed_color)

    grid = np.full((top_h + 4, width), BLACK, dtype=int)
    grid[:top_h] = seeded_top
    grid[top_h, :] = GREEN

    span = 4 * legend_count - 1
    start_c = int(RNG.integers(0, width - span + 1))
    for i, color in enumerate(legend_colors):
        c0 = start_c + 4 * i
        grid[top_h + 1 : top_h + 4, c0 : c0 + 3] = legend_tiles[int(color)]

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
