import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "right", "bottom", "left")
STEP_TO_SIDE = {
    (-1, 0): "top",
    (0, 1): "right",
    (1, 0): "bottom",
    (0, -1): "left",
}

GEN_RETRIES = 400
MIN_TILE = 3
MAX_TILE = 5
MIN_GAP = 2

OPEN_PATTERNS = [
    [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
]

CLOSED_PATTERNS = [
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1), (3, 1)],
]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _side_values(patch: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return patch[0, :]
    if side == "right":
        return patch[:, -1]
    if side == "bottom":
        return patch[-1, :]
    return patch[:, 0]


def _side_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "right":
        mask[:, -1] = True
    elif side == "bottom":
        mask[-1, :] = True
    else:
        mask[:, 0] = True
    return mask


def _side_length(shape: tuple[int, int], side: str) -> int:
    return int(shape[0] if side in {"left", "right"} else shape[1])


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    comps = _find_components(grid != BG)
    pieces: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
        assert np.all(patch != BG), "pieces must be dense rectangles"
        fill = int(Counter(patch.reshape(-1).tolist()).most_common(1)[0][0])

        sides: dict[str, int] = {}
        for side in SIDES:
            line = _side_values(patch, side).tolist()
            counts = Counter(int(v) for v in line)
            for color, count in counts.items():
                if int(color) == fill:
                    continue
                if int(count) >= len(line) - 1:
                    sides[side] = int(color)
                    break

        assert 1 <= len(sides) <= 2, "each piece must expose one or two marked sides"
        marker_mask = np.zeros_like(patch, dtype=bool)
        for side in sides:
            marker_mask |= _side_mask(patch.shape, side)

        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "sides": sides,
                "marker_mask": marker_mask,
            }
        )

    pieces.sort(key=lambda piece: (piece["bbox"][0], piece["bbox"][1]))
    return pieces


def _pair_offset(piece_a: dict, side_a: str, piece_b: dict, side_b: str) -> tuple[int, int]:
    shape_a = piece_a["patch"].shape
    shape_b = piece_b["patch"].shape
    assert _side_length(shape_a, side_a) == _side_length(shape_b, side_b), "paired edge lengths must match"

    if side_a == "right" and side_b == "left":
        return 0, int(shape_a[1])
    if side_a == "left" and side_b == "right":
        return 0, -int(shape_b[1])
    if side_a == "bottom" and side_b == "top":
        return int(shape_a[0]), 0
    if side_a == "top" and side_b == "bottom":
        return -int(shape_b[0]), 0
    raise AssertionError("paired marker sides must face each other")


def _solve_positions(pieces: list[dict]) -> dict[int, tuple[int, int]]:
    by_color: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        for side, color in piece["sides"].items():
            by_color[int(color)].append((int(idx), side))

    relations: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for color, occ in by_color.items():
        assert len(occ) == 2, f"marker color {color} must appear on exactly two pieces"
        (a, side_a), (b, side_b) = occ
        dr, dc = _pair_offset(pieces[a], side_a, pieces[b], side_b)
        relations[a].append((b, int(dr), int(dc)))
        relations[b].append((a, -int(dr), -int(dc)))

    start = 0
    positions = {start: (0, 0)}
    q = deque([start])
    while q:
        cur = q.popleft()
        base_r, base_c = positions[cur]
        for nxt, dr, dc in relations.get(cur, []):
            want = (int(base_r + dr), int(base_c + dc))
            if nxt in positions:
                assert positions[nxt] == want, "piece constraints must be consistent"
                continue
            positions[nxt] = want
            q.append(nxt)

    assert len(positions) == len(pieces), "all pieces must belong to one assembly"
    return positions


def _render_and_recolor(pieces: list[dict], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in positions.values())
    min_c = min(c for _, c in positions.values())
    norm = {idx: (r - min_r, c - min_c) for idx, (r, c) in positions.items()}
    max_r = max(norm[idx][0] + pieces[idx]["patch"].shape[0] for idx in norm)
    max_c = max(norm[idx][1] + pieces[idx]["patch"].shape[1] for idx in norm)

    canvas = np.full((int(max_r), int(max_c)), BG, dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    marker_mask = np.zeros_like(canvas, dtype=bool)

    for idx, piece in enumerate(pieces):
        r0, c0 = norm[idx]
        patch = piece["patch"]
        h, w = patch.shape
        view_occ = occupied[r0 : r0 + h, c0 : c0 + w]
        assert not bool(view_occ.any()), "assembled pieces must not overlap"
        canvas[r0 : r0 + h, c0 : c0 + w] = patch
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        marker_mask[r0 : r0 + h, c0 : c0 + w] |= piece["marker_mask"]

    out = canvas.copy()
    out[marker_mask] = GREY

    h, w = out.shape
    traversable = (out != BG) & (out != GREY)
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if traversable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if traversable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if traversable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    out[seen] = AZURE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble rectangle pieces by matching edge colors, then recolor cells by whether the new grey seams enclose them.

    Input structure:
    - The background is black.
    - Several dense rectangular pieces are scattered apart, each with a dominant fill color and one or two colored edge strips.
    - Every marker color appears on exactly two pieces, marking the two edges that must meet.

    Rule:
    - Detect each rectangular piece and identify which side or sides carry marker colors.
    - Pair equal marker colors and translate the two pieces so the matching sides face each other, using the top-left piece as the anchor for the full assembly.
    - After all paired adjacencies are satisfied, recolor every marked edge pixel to grey.
    - Treat grey pixels as walls: fill pixels connected to the cropped output boundary become Azure, while fill pixels cut off from the boundary keep their original colors.

    Key insight:
    - The colored edge strips encode the spatial reconstruction, and only after rebuilding the composite can the solver decide which fill regions are outside-connected versus enclosed by the grey seam network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pieces = _extract_pieces(g)
    positions = _solve_positions(pieces)
    return _render_and_recolor(pieces, positions)


def _transform_path(path: list[tuple[int, int]], rot: int, flip: bool) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in path]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(int(rot) % 4):
        pts = [(c, -r) for r, c in pts]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(int(r - min_r), int(c - min_c)) for r, c in pts]


def _adjacency_from_path(path: list[tuple[int, int]]) -> list[set[int]]:
    cell_to_idx = {cell: idx for idx, cell in enumerate(path)}
    adj = [set() for _ in path]
    for idx, (r, c) in enumerate(path):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            j = cell_to_idx.get((nr, nc))
            if j is not None:
                adj[idx].add(int(j))
    return adj


def _assign_fill_colors(adj: list[set[int]], forbidden: list[set[int]]) -> list[int] | None:
    palette = list(PALETTE)
    random.shuffle(palette)
    order = sorted(range(len(adj)), key=lambda idx: (-len(adj[idx]), -len(forbidden[idx])))
    fills = [-1] * len(adj)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        blocked = set(int(v) for v in forbidden[idx])
        for nbr in adj[idx]:
            if fills[nbr] >= 0:
                blocked.add(int(fills[nbr]))
        choices = [color for color in palette if color not in blocked]
        random.shuffle(choices)
        for color in choices:
            fills[idx] = int(color)
            if dfs(pos + 1):
                return True
        fills[idx] = -1
        return False

    return [int(v) for v in fills] if dfs(0) else None


def _make_patch(height: int, width: int, fill: int, sides: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((int(height), int(width)), int(fill), dtype=int)
    marker_mask = np.zeros((int(height), int(width)), dtype=bool)
    for side in SIDES:
        color = sides.get(side)
        if color is None:
            continue
        marker_mask |= _side_mask(patch.shape, side)
        if side == "top":
            patch[0, :] = int(color)
        elif side == "right":
            patch[:, -1] = int(color)
        elif side == "bottom":
            patch[-1, :] = int(color)
        else:
            patch[:, 0] = int(color)
    return patch, marker_mask


def _random_partition(total: int, parts: int) -> list[int]:
    vals = [0] * int(parts)
    for _ in range(int(total)):
        vals[random.randrange(int(parts))] += 1
    return vals


def _slot_starts(count: int, piece: int, total: int) -> list[int]:
    base = int(count) * int(piece) + max(0, int(count) - 1) * MIN_GAP
    slack = int(total) - base
    extras = _random_partition(slack, int(count) + 1)
    starts: list[int] = []
    cur = int(extras[0])
    for idx in range(int(count)):
        starts.append(cur)
        cur += int(piece)
        if idx + 1 < int(count):
            cur += MIN_GAP + int(extras[idx + 1])
    return starts


def _scatter_input(pieces: list[dict]) -> np.ndarray | None:
    n = len(pieces)
    piece_h, piece_w = pieces[0]["patch"].shape
    candidates: list[tuple[int, int, int, int]] = []
    for row_slots in range(1, 4):
        col_slots = (n + row_slots - 1) // row_slots
        min_h = row_slots * piece_h + max(0, row_slots - 1) * MIN_GAP
        min_w = col_slots * piece_w + max(0, col_slots - 1) * MIN_GAP
        if min_h <= 30 and min_w <= 30:
            candidates.append((int(row_slots), int(col_slots), int(min_h), int(min_w)))
    if not candidates:
        return None

    row_slots, col_slots, min_h, min_w = random.choice(candidates)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    row_starts = _slot_starts(row_slots, piece_h, grid_h)
    col_starts = _slot_starts(col_slots, piece_w, grid_w)
    slots = [(r, c) for r in row_starts for c in col_starts]
    random.shuffle(slots)
    chosen_slots = slots[:n]

    order = list(range(n))
    random.shuffle(order)
    grid = np.full((int(grid_h), int(grid_w)), BG, dtype=int)
    for piece_idx, (r0, c0) in zip(order, chosen_slots):
        patch = pieces[piece_idx]["patch"]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
    return grid


def _build_layout(path: list[tuple[int, int]], tile_h: int, tile_w: int) -> tuple[list[dict], dict[int, tuple[int, int]]] | None:
    edge_colors = random.sample(PALETTE, len(path) - 1)
    sides: list[dict[str, int]] = [dict() for _ in path]
    for idx, color in enumerate(edge_colors):
        a = path[idx]
        b = path[idx + 1]
        step = (int(b[0] - a[0]), int(b[1] - a[1]))
        side_a = STEP_TO_SIDE[step]
        side_b = STEP_TO_SIDE[(-step[0], -step[1])]
        sides[idx][side_a] = int(color)
        sides[idx + 1][side_b] = int(color)

    adj = _adjacency_from_path(path)
    forbidden = [set(side_map.values()) for side_map in sides]
    fills = _assign_fill_colors(adj, forbidden)
    if fills is None:
        return None

    pieces: list[dict] = []
    positions: dict[int, tuple[int, int]] = {}
    for idx, side_map in enumerate(sides):
        patch, marker_mask = _make_patch(tile_h, tile_w, fills[idx], side_map)
        pieces.append(
            {
                "bbox": (0, 0, tile_h - 1, tile_w - 1),
                "patch": patch,
                "sides": dict(side_map),
                "marker_mask": marker_mask,
            }
        )
        positions[idx] = (int(path[idx][0] * tile_h), int(path[idx][1] * tile_w))
    return pieces, positions


def _gen_attempt() -> np.ndarray | None:
    want_closed = bool(random.getrandbits(1))
    pattern = random.choice(CLOSED_PATTERNS if want_closed else OPEN_PATTERNS)
    path = _transform_path(pattern, rot=random.randrange(4), flip=bool(random.getrandbits(1)))
    bbox_h = max(r for r, _ in path) + 1
    bbox_w = max(c for _, c in path) + 1

    tile_h_max = min(MAX_TILE, 30 // bbox_h)
    tile_w_max = min(MAX_TILE, 30 // bbox_w)
    if tile_h_max < MIN_TILE or tile_w_max < MIN_TILE:
        return None
    tile_h = random.randint(MIN_TILE, tile_h_max)
    tile_w = random.randint(MIN_TILE, tile_w_max)

    built = _build_layout(path, tile_h, tile_w)
    if built is None:
        return None
    pieces, positions = built
    expected = _render_and_recolor(pieces, positions)
    kept = (expected != BG) & (expected != GREY) & (expected != AZURE)
    if bool(kept.any()) != want_closed:
        return None

    grid = _scatter_input(pieces)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
