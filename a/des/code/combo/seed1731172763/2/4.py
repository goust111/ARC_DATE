import numpy as np


BG = 8
PATH = 5
OBJECT_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in {BG, PATH}:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(pts, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return objects


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not visited[nr, nc] and bool(mask[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))

    return comps


def _find(parents: list[int], x: int) -> int:
    while parents[x] != x:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x


def _union(parents: list[int], ranks: list[int], a: int, b: int) -> None:
    ra = _find(parents, a)
    rb = _find(parents, b)
    if ra == rb:
        return
    if ranks[ra] < ranks[rb]:
        ra, rb = rb, ra
    parents[rb] = ra
    if ranks[ra] == ranks[rb]:
        ranks[ra] += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by grey connection paths, then stamp the smallest shape onto the largest object.

    Input structure:
    - Background is azure.
    - Grey pixels form one-cell-wide straight or L-shaped paths.
    - Other colors form 4-connected monochrome objects.

    Rule:
    - Grey paths determine grouping but are erased in the output.
    - Within each multi-object group, sort objects by area.
    - Normalize the smallest object to its bbox, then stamp that mask in the
      smallest object's color at the largest object's center.
    - Erase the smallest object's original location.
    - Other non-grey objects remain in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    objects = _extract_objects(g)
    obj_id = np.full((h, w), -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        obj_id[coords[:, 0], coords[:, 1]] = idx

    parents = list(range(len(objects)))
    ranks = [0] * len(objects)

    for comp in _mask_components(g == PATH):
        neighbors: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                oid = int(obj_id[nr, nc])
                if oid != -1:
                    neighbors.add(oid)
        ids = sorted(neighbors)
        for i in range(1, len(ids)):
            _union(parents, ranks, ids[0], ids[i])

    groups: dict[int, list[dict[str, object]]] = {}
    for idx, obj in enumerate(objects):
        groups.setdefault(_find(parents, idx), []).append(obj)

    out = g.copy()
    out[out == PATH] = BG

    for members in groups.values():
        if len(members) < 2:
            continue

        ordered = sorted(
            members,
            key=lambda obj: (
                int(obj["area"]),
                int(obj["bbox"][0]),
                int(obj["bbox"][1]),
                int(obj["color"]),
            ),
        )
        smallest = ordered[0]
        largest = ordered[-1]

        sr0, sc0, sr1, sc1 = map(int, smallest["bbox"])
        lr0, lc0, lr1, lc1 = map(int, largest["bbox"])
        stamp_h = sr1 - sr0 + 1
        stamp_w = sc1 - sc0 + 1
        anchor_r = (lr0 + lr1) // 2
        anchor_c = (lc0 + lc1) // 2
        top = anchor_r - stamp_h // 2
        left = anchor_c - stamp_w // 2

        rel = smallest["coords"] - np.array([sr0, sc0], dtype=int)
        for dr, dc in rel:
            rr = int(top + dr)
            cc = int(left + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(smallest["color"])

        src = smallest["coords"]
        out[src[:, 0], src[:, 1]] = BG

    return out


SMALL_MASKS = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
]

MID_MASKS = [
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
]

LARGE_MASKS = [
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

ORTHOGONAL_SLOTS = ["left", "right", "top", "bottom"]
DIAGONAL_SLOTS = ["top-left", "top-right", "bottom-left", "bottom-right"]
PAIR_SLOTS = [
    ("left", "right"),
    ("top", "bottom"),
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
    ("top-left", "bottom-right"),
    ("top-right", "bottom-left"),
    ("top-left", "right"),
    ("top-right", "left"),
    ("bottom-left", "right"),
    ("bottom-right", "left"),
    ("top-left", "bottom"),
    ("top-right", "bottom"),
    ("bottom-left", "top"),
    ("bottom-right", "top"),
]


def _rot_flip(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _choose_masks(group_size: int) -> list[np.ndarray]:
    small = _rot_flip(SMALL_MASKS[int(_RNG.integers(0, len(SMALL_MASKS)))])
    large = _rot_flip(LARGE_MASKS[int(_RNG.integers(0, len(LARGE_MASKS)))])
    if group_size == 2:
        return [small, large]
    mid = _rot_flip(MID_MASKS[int(_RNG.integers(0, len(MID_MASKS)))])
    return [small, mid, large]


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    placed = coords + np.array([top, left], dtype=int)
    grid[placed[:, 0], placed[:, 1]] = color
    return placed


def _side_contact(mask: np.ndarray, top: int, left: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(mask)
    if side == "north":
        edge = coords[coords[:, 0] == coords[:, 0].min()]
        edge = edge[np.argsort(np.abs(edge[:, 1] - mask.shape[1] / 2.0))]
        r, c = map(int, edge[0])
        return (top + r, left + c), (top + r - 1, left + c)
    if side == "south":
        edge = coords[coords[:, 0] == coords[:, 0].max()]
        edge = edge[np.argsort(np.abs(edge[:, 1] - mask.shape[1] / 2.0))]
        r, c = map(int, edge[0])
        return (top + r, left + c), (top + r + 1, left + c)
    if side == "west":
        edge = coords[coords[:, 1] == coords[:, 1].min()]
        edge = edge[np.argsort(np.abs(edge[:, 0] - mask.shape[0] / 2.0))]
        r, c = map(int, edge[0])
        return (top + r, left + c), (top + r, left + c - 1)
    edge = coords[coords[:, 1] == coords[:, 1].max()]
    edge = edge[np.argsort(np.abs(edge[:, 0] - mask.shape[0] / 2.0))]
    r, c = map(int, edge[0])
    return (top + r, left + c), (top + r, left + c + 1)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _manhattan_path(
    a: tuple[int, int], b: tuple[int, int], first_axis: str
) -> list[tuple[int, int]]:
    if a[0] == b[0] or a[1] == b[1]:
        return _segment(a, b)
    corner = (a[0], b[1]) if first_axis == "h" else (b[0], a[1])
    first = _segment(a, corner)
    second = _segment(corner, b)
    return first + second[1:]


def _slot_position(
    slot: str,
    obj_shape: tuple[int, int],
    large_top: int,
    large_left: int,
    large_shape: tuple[int, int],
    gap: int,
) -> tuple[int, int]:
    oh, ow = obj_shape
    lh, lw = large_shape
    center_r = large_top + lh // 2
    center_c = large_left + lw // 2

    if slot == "left":
        return center_r - oh // 2, large_left - gap - ow
    if slot == "right":
        return center_r - oh // 2, large_left + lw + gap
    if slot == "top":
        return large_top - gap - oh, center_c - ow // 2
    if slot == "bottom":
        return large_top + lh + gap, center_c - ow // 2
    if slot == "top-left":
        return large_top - gap - oh, large_left - gap - ow
    if slot == "top-right":
        return large_top - gap - oh, large_left + lw + gap
    if slot == "bottom-left":
        return large_top + lh + gap, large_left - gap - ow
    return large_top + lh + gap, large_left + lw + gap


def _contact_options(slot: str) -> list[tuple[str, str]]:
    if slot == "left":
        return [("east", "west")]
    if slot == "right":
        return [("west", "east")]
    if slot == "top":
        return [("south", "north")]
    if slot == "bottom":
        return [("north", "south")]
    if slot == "top-left":
        return [("east", "north"), ("south", "west")]
    if slot == "top-right":
        return [("west", "north"), ("south", "east")]
    if slot == "bottom-left":
        return [("east", "south"), ("north", "west")]
    return [("west", "south"), ("north", "east")]


def _fits(bounds: tuple[int, int, int, int], top: int, left: int, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bounds
    h, w = shape
    return r0 <= top and c0 <= left and top + h - 1 <= r1 and left + w - 1 <= c1


def _path_clear(
    path: list[tuple[int, int]],
    occupied: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bounds
    for r, c in path:
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
        if occupied[r, c]:
            return False
    return True


def _stamp_fits(
    large_mask: np.ndarray,
    large_top: int,
    large_left: int,
    small_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> bool:
    lr0, lc0 = large_top, large_left
    lh, lw = large_mask.shape
    sh, sw = small_mask.shape
    center_r = lr0 + lh // 2
    center_c = lc0 + lw // 2
    top = center_r - sh // 2
    left = center_c - sw // 2
    r0, c0, r1, c1 = bounds
    return r0 <= top and c0 <= left and top + sh - 1 <= r1 and left + sw - 1 <= c1


def _slot_choices(group_size: int) -> list[str]:
    if group_size == 2:
        if bool(_RNG.integers(0, 2)):
            return [ORTHOGONAL_SLOTS[int(_RNG.integers(0, len(ORTHOGONAL_SLOTS)))]]
        return [DIAGONAL_SLOTS[int(_RNG.integers(0, len(DIAGONAL_SLOTS)))]]
    pair = PAIR_SLOTS[int(_RNG.integers(0, len(PAIR_SLOTS)))]
    return [pair[0], pair[1]]


def _group_attempt(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bounds
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1
    group_size = int(_RNG.integers(2, 4))
    masks = _choose_masks(group_size)
    colors = _RNG.choice(np.array(OBJECT_COLORS, dtype=int), size=group_size, replace=False).tolist()
    slots = _slot_choices(group_size)
    gap = int(_RNG.integers(1, 3))

    large_mask = masks[-1]
    lh, lw = large_mask.shape
    large_top = r0 + (cell_h - lh) // 2
    large_left = c0 + (cell_w - lw) // 2
    if not _fits(bounds, large_top, large_left, large_mask.shape):
        return False
    if not _stamp_fits(large_mask, large_top, large_left, masks[0], bounds):
        return False

    placements: list[tuple[np.ndarray, int, int, int, str]] = []
    occupied = np.zeros_like(grid, dtype=bool)

    large_coords = np.argwhere(large_mask) + np.array([large_top, large_left], dtype=int)
    occupied[large_coords[:, 0], large_coords[:, 1]] = True
    placements.append((large_mask, large_top, large_left, int(colors[-1]), "large"))

    for mask, color, slot in zip(masks[:-1], colors[:-1], slots):
        top, left = _slot_position(slot, mask.shape, large_top, large_left, large_mask.shape, gap)
        if not _fits(bounds, top, left, mask.shape):
            return False
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        if np.any(occupied[coords[:, 0], coords[:, 1]]):
            return False
        placements.append((mask, top, left, int(color), slot))
        occupied[coords[:, 0], coords[:, 1]] = True

    all_paths: list[list[tuple[int, int]]] = []
    for mask, top, left, _color, slot in placements[1:]:
        options = _contact_options(slot)
        order = _RNG.permutation(len(options))
        found = None
        for idx in order:
            obj_side, large_side = options[int(idx)]
            _obj_cell, start = _side_contact(mask, top, left, obj_side)
            _large_cell, end = _side_contact(large_mask, large_top, large_left, large_side)
            axes = ["h", "v"] if bool(_RNG.integers(0, 2)) else ["v", "h"]
            for axis in axes:
                path = _manhattan_path(start, end, axis)
                if _path_clear(path, occupied, bounds):
                    found = path
                    break
            if found is not None:
                break
        if found is None:
            return False
        all_paths.append(found)
        for pr, pc in found:
            occupied[pr, pc] = True

    for mask, top, left, color, _slot in placements:
        _place_mask(grid, mask, top, left, color)
    for path in all_paths:
        for pr, pc in path:
            grid[pr, pc] = PATH

    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(24, 31))
    w = int(_RNG.integers(24, 31))
    grid = np.full((h, w), BG, dtype=int)

    margin = 1
    gap = 2
    cell_h = (h - 2 * margin - gap) // 2
    cell_w = (w - 2 * margin - gap) // 2
    if cell_h < 10 or cell_w < 10:
        return None

    cells = [
        (margin, margin, margin + cell_h - 1, margin + cell_w - 1),
        (margin, margin + cell_w + gap, margin + cell_h - 1, margin + 2 * cell_w + gap - 1),
        (margin + cell_h + gap, margin, margin + 2 * cell_h + gap - 1, margin + cell_w - 1),
        (
            margin + cell_h + gap,
            margin + cell_w + gap,
            margin + 2 * cell_h + gap - 1,
            margin + 2 * cell_w + gap - 1,
        ),
    ]
    valid_cells = [cell for cell in cells if cell[2] < h and cell[3] < w]
    if len(valid_cells) < 1:
        return None

    group_count = int(_RNG.integers(1, min(3, len(valid_cells)) + 1))
    chosen = _RNG.choice(len(valid_cells), size=group_count, replace=False)
    for idx in chosen:
        if not _group_attempt(grid, valid_cells[int(idx)]):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.sum(grid == PATH) < group_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
