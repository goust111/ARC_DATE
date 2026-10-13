from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DEGREE_TO_COLOR = {
    1: BLUE,
    2: RED,
    3: YELLOW,
    4: GREEN,
}

GEN_RETRIES = 400
HUB_MODE_PROB = 0.45
CELL_COUNT_MIN = 3
CELL_COUNT_MAX_EXCLUSIVE = 6
CELL_SIZE_MIN = 2
CELL_SIZE_MAX_EXCLUSIVE = 5
MAX_EXTRA_HUB_ROOMS = 2
MAX_GROWTH_ROOMS = 7


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _room_graph(grid: np.ndarray) -> tuple[list[np.ndarray], list[set[int]], np.ndarray]:
    rooms = _components(grid == BLACK)
    assert rooms, "expected at least one black room"

    room_of = np.full(grid.shape, -1, dtype=int)
    for rid, coords in enumerate(rooms):
        room_of[coords[:, 0], coords[:, 1]] = rid

    neighbors = [set() for _ in rooms]
    h, w = grid.shape
    for r, c in np.argwhere(grid == GREY):
        touching = set()
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                rid = int(room_of[nr, nc])
                if rid >= 0:
                    touching.add(rid)
        if len(touching) >= 2:
            for rid in touching:
                neighbors[rid].update(touching - {rid})

    return rooms, neighbors, room_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each black room by the number of other rooms it touches across one-cell grey walls.

    Input structure:
    - Grey cells form one-cell-thick walls that partition the grid into black 4-connected rooms.
    - Room shapes and sizes may vary, and the same room can span multiple irregular lobes connected through open passages.
    - The outer border is not a room; only black connected components count as rooms.

    Rule:
    - Treat each black 4-connected component as one room.
    - Two rooms are neighbors when some grey wall cell is 4-neighbor-adjacent to both rooms.
    - Count each room's distinct neighboring rooms k, then recolor the whole room:
      k=1 -> blue, k=2 -> red, k=3 -> yellow, k=4 -> green.
    - Keep all grey wall cells unchanged.

    Key insight:
    - The output color depends on the room-adjacency graph induced by the separating grey walls, not on room area, shape, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREY}, "expected only black rooms and grey walls"

    rooms, neighbors, _room_of = _room_graph(g)
    out = g.copy()
    for rid, coords in enumerate(rooms):
        degree = len(neighbors[rid])
        assert degree in DEGREE_TO_COLOR, f"unsupported room degree {degree}"
        out[coords[:, 0], coords[:, 1]] = DEGREE_TO_COLOR[degree]
    return out


def _macro_neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    h, w = shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _expand_from_seeds(shape: tuple[int, int], seeds: list[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    room_ids = np.full((h, w), -1, dtype=int)
    sizes = np.zeros(len(seeds), dtype=int)
    for rid, (r, c) in enumerate(seeds):
        room_ids[r, c] = rid
        sizes[rid] = 1

    remaining = h * w - len(seeds)
    while remaining > 0:
        frontier = []
        for r in range(h):
            for c in range(w):
                if room_ids[r, c] != -1:
                    continue
                touching = sorted(
                    {
                        int(room_ids[nr, nc])
                        for nr, nc in _macro_neighbors(shape, r, c)
                        if room_ids[nr, nc] != -1
                    }
                )
                if touching:
                    frontier.append((r, c, touching))
        if not frontier:
            raise RuntimeError("partition growth got stuck")

        cell_r, cell_c, touching = frontier[int(rng.integers(0, len(frontier)))]
        weights = np.array([1.0 / sizes[rid] for rid in touching], dtype=float)
        weights /= weights.sum()
        owner = int(rng.choice(np.array(touching, dtype=int), p=weights))
        room_ids[cell_r, cell_c] = owner
        sizes[owner] += 1
        remaining -= 1

    return room_ids


def _growth_partition(mh: int, mw: int, rng: np.random.Generator) -> np.ndarray:
    cells = [(r, c) for r in range(mh) for c in range(mw)]
    max_rooms = min(MAX_GROWTH_ROOMS, len(cells) - 1)
    room_count = int(rng.integers(4, max_rooms + 1))
    seed_ids = rng.choice(len(cells), size=room_count, replace=False)
    seeds = [cells[int(i)] for i in np.atleast_1d(seed_ids)]
    return _expand_from_seeds((mh, mw), seeds, rng)


def _hub_partition(mh: int, mw: int, rng: np.random.Generator) -> np.ndarray:
    hub_r = int(rng.integers(1, mh - 1))
    hub_c = int(rng.integers(1, mw - 1))
    seeds = [
        (hub_r, hub_c),
        (hub_r - 1, hub_c),
        (hub_r + 1, hub_c),
        (hub_r, hub_c - 1),
        (hub_r, hub_c + 1),
    ]
    used = set(seeds)
    extras = [(r, c) for r in range(mh) for c in range(mw) if (r, c) not in used]
    extra_count = int(rng.integers(0, min(MAX_EXTRA_HUB_ROOMS, len(extras)) + 1))
    if extra_count:
        extra_ids = rng.choice(len(extras), size=extra_count, replace=False)
        seeds.extend(extras[int(i)] for i in np.atleast_1d(extra_ids))
    return _expand_from_seeds((mh, mw), seeds, rng)


def _macro_graph(room_ids: np.ndarray) -> tuple[list[set[int]], np.ndarray, np.ndarray]:
    room_count = int(room_ids.max()) + 1
    neighbors = [set() for _ in range(room_count)]
    mh, mw = room_ids.shape

    for r in range(mh):
        for c in range(mw - 1):
            a = int(room_ids[r, c])
            b = int(room_ids[r, c + 1])
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)

    for r in range(mh - 1):
        for c in range(mw):
            a = int(room_ids[r, c])
            b = int(room_ids[r + 1, c])
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)

    degrees = np.array([len(items) for items in neighbors], dtype=int)
    cell_counts = np.bincount(room_ids.ravel(), minlength=room_count)
    return neighbors, degrees, cell_counts


def _axis_starts(sizes: list[int]) -> list[int]:
    starts = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _render_room_labels(room_ids: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    mh, mw = room_ids.shape
    h = 2 + sum(row_heights) + (mh - 1)
    w = 2 + sum(col_widths) + (mw - 1)
    labels = np.full((h, w), -1, dtype=int)
    row_starts = _axis_starts(row_heights)
    col_starts = _axis_starts(col_widths)

    for r in range(mh):
        for c in range(mw):
            rid = int(room_ids[r, c])
            r0 = row_starts[r]
            c0 = col_starts[c]
            labels[r0 : r0 + row_heights[r], c0 : c0 + col_widths[c]] = rid

    for r in range(mh - 1):
        sep_r = row_starts[r] + row_heights[r]
        for c in range(mw):
            if room_ids[r, c] == room_ids[r + 1, c]:
                rid = int(room_ids[r, c])
                c0 = col_starts[c]
                labels[sep_r, c0 : c0 + col_widths[c]] = rid

    for r in range(mh):
        r0 = row_starts[r]
        for c in range(mw - 1):
            if room_ids[r, c] == room_ids[r, c + 1]:
                rid = int(room_ids[r, c])
                sep_c = col_starts[c] + col_widths[c]
                labels[r0 : r0 + row_heights[r], sep_c] = rid

    for r in range(mh - 1):
        sep_r = row_starts[r] + row_heights[r]
        for c in range(mw - 1):
            rid = int(room_ids[r, c])
            if (
                rid == int(room_ids[r, c + 1])
                and rid == int(room_ids[r + 1, c])
                and rid == int(room_ids[r + 1, c + 1])
            ):
                sep_c = col_starts[c] + col_widths[c]
                labels[sep_r, sep_c] = rid

    return labels


def _labels_to_input(labels: np.ndarray) -> np.ndarray:
    return np.where(labels >= 0, BLACK, GREY).astype(int)


def _labels_to_output(labels: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    palette = np.array([DEGREE_TO_COLOR[int(k)] for k in degrees], dtype=int)
    out = np.full(labels.shape, GREY, dtype=int)
    mask = labels >= 0
    out[mask] = palette[labels[mask]]
    return out


def _has_area_degree_inversion(areas: np.ndarray, degrees: np.ndarray) -> bool:
    n = len(areas)
    for i in range(n):
        for j in range(i + 1, n):
            if (int(areas[i]) - int(areas[j])) * (int(degrees[i]) - int(degrees[j])) < 0:
                return True
    return False


def _degree_profile_ok(mode: str, degrees: np.ndarray) -> bool:
    values = {int(v) for v in degrees.tolist()}
    if not values <= {1, 2, 3, 4}:
        return False
    if len(values) < 3 or 1 not in values or 2 not in values:
        return False
    if mode == "hub":
        return 4 in values
    return max(values) >= 3


def _sample_partition(rng: np.random.Generator) -> tuple[str, np.ndarray]:
    mode = "hub" if rng.random() < HUB_MODE_PROB else "growth"
    mh = int(rng.integers(CELL_COUNT_MIN, CELL_COUNT_MAX_EXCLUSIVE))
    mw = int(rng.integers(CELL_COUNT_MIN, CELL_COUNT_MAX_EXCLUSIVE))
    if mode == "hub":
        return mode, _hub_partition(mh, mw, rng)
    return mode, _growth_partition(mh, mw, rng)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode, room_ids = _sample_partition(rng)
    neighbors, degrees, cell_counts = _macro_graph(room_ids)
    if not _degree_profile_ok(mode, degrees):
        return None
    if np.max(cell_counts) < 2 or len(set(map(int, cell_counts.tolist()))) < 2:
        return None

    row_heights = [int(rng.integers(CELL_SIZE_MIN, CELL_SIZE_MAX_EXCLUSIVE)) for _ in range(room_ids.shape[0])]
    col_widths = [int(rng.integers(CELL_SIZE_MIN, CELL_SIZE_MAX_EXCLUSIVE)) for _ in range(room_ids.shape[1])]
    labels = _render_room_labels(room_ids, row_heights, col_widths)
    inp = _labels_to_input(labels)

    if int(np.sum(inp == BLACK)) <= int(np.sum(inp == GREY)):
        return None
    if len(_components(inp == BLACK)) != len(neighbors):
        return None

    areas = np.bincount(labels[labels >= 0], minlength=len(neighbors))
    if not _has_area_degree_inversion(areas, degrees):
        return None

    expected = _labels_to_output(labels, degrees)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
