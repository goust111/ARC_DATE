import random
from collections import deque

import numpy as np


BG = 0
RED = 2
WALL_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GEN_RETRIES = 300
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SPAN_CHOICES = (4, 5, 6)
OPEN_EXTRA_PROB = 0.2
PREOPEN_PROB = 0.45
REACH_ALL_PROB = 0.5


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _max_run(arr: np.ndarray) -> int:
    best = cur = 0
    for v in arr:
        if v != BG:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _detect_separator_rows(grid: np.ndarray, sep_cols: list[int] | None = None) -> list[int]:
    rows = []
    h, w = grid.shape
    if sep_cols is None:
        col_mask = np.zeros(w, dtype=bool)
        count_threshold = max(5, int(np.ceil(0.6 * w)))
        score_threshold = 4
    else:
        col_mask = np.zeros(w, dtype=bool)
        col_mask[sep_cols] = True
        interior_cols = [c for c in range(w) if not col_mask[c]]
        count_threshold = max(3, int(np.ceil(0.5 * max(1, len(interior_cols)))))
        score_threshold = 3
    for r in range(h):
        count = 0
        score = 0
        for c in range(w):
            if col_mask[c]:
                continue
            if grid[r, c] == BG:
                continue
            count += 1
            if (c > 0 and grid[r, c - 1] != BG) or (c + 1 < w and grid[r, c + 1] != BG):
                score += 1
        if count >= count_threshold and score >= score_threshold:
            rows.append(r)
    assert len(rows) >= 2
    return rows


def _detect_separator_cols(grid: np.ndarray, sep_rows: list[int]) -> list[int]:
    cols = []
    h, w = grid.shape
    row_mask = np.zeros(h, dtype=bool)
    row_mask[sep_rows] = True
    interior_rows = [r for r in range(h) if not row_mask[r]]
    count_threshold = max(3, int(np.ceil(0.5 * max(1, len(interior_rows)))))
    for c in range(w):
        count = 0
        score = 0
        for r in interior_rows:
            if grid[r, c] == BG:
                continue
            count += 1
            up_ok = r > 0 and (not row_mask[r - 1]) and grid[r - 1, c] != BG
            down_ok = r + 1 < h and (not row_mask[r + 1]) and grid[r + 1, c] != BG
            if up_ok or down_ok:
                score += 1
        if count >= count_threshold and score >= 3:
            cols.append(c)
    assert len(cols) >= 2
    return cols


def _find_room_index(pos: int, seps: list[int]) -> int:
    for i in range(len(seps) - 1):
        if seps[i] < pos < seps[i + 1]:
            return i
    raise AssertionError("position is not inside a room")


def _wall_color(grid: np.ndarray, room: dict, side: str) -> int:
    r0, r1, c0, c1 = room["r0"], room["r1"], room["c0"], room["c1"]
    if side == "top":
        vals = [int(v) for v in grid[r0, c0 + 1 : c1] if int(v) != BG]
    elif side == "bottom":
        vals = [int(v) for v in grid[r1, c0 + 1 : c1] if int(v) != BG]
    elif side == "left":
        vals = [int(v) for v in grid[r0 + 1 : r1, c0] if int(v) != BG]
    elif side == "right":
        vals = [int(v) for v in grid[r0 + 1 : r1, c1] if int(v) != BG]
    else:
        raise ValueError(side)
    assert vals
    colors, counts = np.unique(np.array(vals, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _door_position(room: dict, side: str, key: tuple[int, int] | None) -> tuple[int, int]:
    r0, r1, c0, c1 = room["r0"], room["r1"], room["c0"], room["c1"]
    if side in ("top", "bottom"):
        length = c1 - c0 - 1
        row = r0 if side == "top" else r1
        if length % 2 == 1:
            col = c0 + (length + 1) // 2
        else:
            left_mid = c0 + length // 2
            right_mid = left_mid + 1
            assert key is not None
            col = left_mid if key[1] <= left_mid else right_mid
        return (row, col)

    length = r1 - r0 - 1
    col = c0 if side == "left" else c1
    if length % 2 == 1:
        row = r0 + (length + 1) // 2
    else:
        upper_mid = r0 + length // 2
        lower_mid = upper_mid + 1
        assert key is not None
        row = upper_mid if key[0] <= upper_mid else lower_mid
    return (row, col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rough_rows = _detect_separator_rows(g)
    sep_cols = _detect_separator_cols(g, rough_rows)
    sep_rows = _detect_separator_rows(g, sep_cols)
    sep_cols = _detect_separator_cols(g, sep_rows)

    rooms: dict[tuple[int, int], dict] = {}
    for ri in range(len(sep_rows) - 1):
        for ci in range(len(sep_cols) - 1):
            r0, r1 = sep_rows[ri], sep_rows[ri + 1]
            c0, c1 = sep_cols[ci], sep_cols[ci + 1]
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            rooms[(ri, ci)] = {"r0": r0, "r1": r1, "c0": c0, "c1": c1}

    row_sep_mask = np.zeros(h, dtype=bool)
    col_sep_mask = np.zeros(w, dtype=bool)
    row_sep_mask[sep_rows] = True
    col_sep_mask[sep_cols] = True

    room_keys: dict[tuple[int, int], list[tuple[int, int]]] = {room_id: [] for room_id in rooms}
    seed_cells: list[tuple[int, int]] | None = None
    seed_room: tuple[int, int] | None = None

    seed_mask = (g == RED) & (~row_sep_mask[:, None]) & (~col_sep_mask[None, :])
    seed_comps = _components(seed_mask)
    for comp in seed_comps:
        if len(comp) != 2:
            continue
        (r0, c0), (r1, c1) = comp
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            continue
        assert seed_cells is None
        seed_cells = comp
        seed_room = (_find_room_index(r0, sep_rows), _find_room_index(c0, sep_cols))

    marker_mask = (g != BG) & (~row_sep_mask[:, None]) & (~col_sep_mask[None, :]) & (~seed_mask)
    comps = _components(marker_mask)
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        room_id = (_find_room_index(rs[0], sep_rows), _find_room_index(cs[0], sep_cols))
        assert all((_find_room_index(r, sep_rows), _find_room_index(c, sep_cols)) == room_id for r, c in comp)
        assert len(comp) == 1
        room_keys[room_id].append(comp[0])

    assert seed_cells is not None and seed_room is not None

    out = g.copy()
    for room_id, keys in room_keys.items():
        room = rooms[room_id]
        wall_by_color: dict[int, str] = {}
        for side in ("top", "bottom", "left", "right"):
            color = _wall_color(g, room, side)
            assert color not in wall_by_color
            wall_by_color[color] = side
        for key in keys:
            key_color = int(g[key])
            assert key_color in wall_by_color
            side = wall_by_color[key_color]
            dr, dc = _door_position(room, side, key)
            out[dr, dc] = BG
            out[key] = BG

    passable = (out == BG)
    for r, c in seed_cells:
        passable[r, c] = True

    seen = np.zeros_like(passable, dtype=bool)
    q = deque(seed_cells)
    for r, c in seed_cells:
        seen[r, c] = True

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    fill_mask = seen & (out == BG)
    out[fill_mask] = RED
    return out


def _room_partitions(rng: random.Random, count_range: tuple[int, int]) -> list[int]:
    count = rng.randint(*count_range)
    return [rng.choice(ROOM_SPAN_CHOICES) for _ in range(count)]


def _separator_positions(spans: list[int]) -> list[int]:
    out = [0]
    cur = 0
    for span in spans:
        cur += span + 1
        out.append(cur)
    return out


def _edge_colors(m: int, n: int, rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
    palette = list(WALL_COLORS)
    rng.shuffle(palette)
    hcols = palette[:4]
    vcols = palette[4:]
    horiz = [[hcols[(i + 2 * j) % 4] for j in range(n)] for i in range(m + 1)]
    vert = [[vcols[(2 * i + j) % 4] for j in range(n + 1)] for i in range(m)]
    return horiz, vert


def _all_edges(m: int, n: int) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    edges: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for r in range(m):
        for c in range(n):
            if r + 1 < m:
                edges.append(((r, c), (r + 1, c), "h"))
            if c + 1 < n:
                edges.append(((r, c), (r, c + 1), "v"))
    return edges


def _choose_reachable_set(rng: random.Random, m: int, n: int, source: tuple[int, int]) -> set[tuple[int, int]]:
    total = m * n
    if total == 1:
        return {source}
    if rng.random() < REACH_ALL_PROB:
        target = total
    else:
        target = rng.randint(2, total - 1)

    reachable = {source}
    frontier = [source]
    while len(reachable) < target and frontier:
        cell = rng.choice(frontier)
        nbrs = []
        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in reachable:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(cell)
            continue
        nxt = rng.choice(nbrs)
        reachable.add(nxt)
        frontier.append(nxt)
    return reachable


def _choose_open_edges(
    rng: random.Random, m: int, n: int, source: tuple[int, int]
) -> tuple[set[frozenset], set[frozenset], dict[frozenset, tuple[int, int]], set[tuple[int, int]]]:
    reachable = _choose_reachable_set(rng, m, n, source)
    open_edges: set[frozenset] = set()
    tree_edges: list[frozenset] = []
    reached = {source}
    while reached != reachable:
        pending = []
        for dst in (reachable - reached):
            r, c = dst
            dst_neighbors = []
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in reached:
                    dst_neighbors.append((nr, nc))
            if dst_neighbors:
                pending.append((dst, dst_neighbors))
        assert pending
        dst, dst_neighbors = rng.choice(pending)
        src = rng.choice(dst_neighbors)
        edge = frozenset((src, dst))
        open_edges.add(edge)
        tree_edges.append(edge)
        reached.add(dst)

    all_edges = _all_edges(m, n)
    for a, b, _ in all_edges:
        edge = frozenset((a, b))
        if edge in open_edges:
            continue
        same_side = (a in reachable) == (b in reachable)
        if same_side and rng.random() < OPEN_EXTRA_PROB:
            open_edges.add(edge)

    key_edges: set[frozenset] = set()
    preopen_edges: set[frozenset] = set()
    key_owner: dict[frozenset, tuple[int, int]] = {}
    must_have_key = [rng.choice(tree_edges)] if tree_edges else []
    for edge in open_edges:
        if edge in must_have_key or rng.random() >= PREOPEN_PROB:
            key_edges.add(edge)
            a, b = tuple(edge)
            key_owner[edge] = rng.choice((a, b))
        else:
            preopen_edges.add(edge)

    return open_edges, key_edges, key_owner, reachable


def _edge_side(a: tuple[int, int], b: tuple[int, int], owner: tuple[int, int]) -> str:
    if a[0] == b[0]:
        left = a if a[1] < b[1] else b
        right = b if a[1] < b[1] else a
        return "right" if owner == left else "left"
    top = a if a[0] < b[0] else b
    bottom = b if a[0] < b[0] else a
    return "bottom" if owner == top else "top"


def _paint_walls(
    grid: np.ndarray,
    row_seps: list[int],
    col_seps: list[int],
    horiz: list[list[int]],
    vert: list[list[int]],
) -> None:
    m = len(row_seps) - 1
    n = len(col_seps) - 1
    for i in range(m + 1):
        r = row_seps[i]
        for j in range(n):
            c0, c1 = col_seps[j], col_seps[j + 1]
            grid[r, c0 : c1 + 1] = horiz[i][j]
    for i in range(m):
        r0, r1 = row_seps[i], row_seps[i + 1]
        for j in range(n + 1):
            c = col_seps[j]
            grid[r0 : r1 + 1, c] = vert[i][j]


def _wall_segment_color(
    room: tuple[int, int], side: str, horiz: list[list[int]], vert: list[list[int]]
) -> int:
    r, c = room
    if side == "top":
        return horiz[r][c]
    if side == "bottom":
        return horiz[r + 1][c]
    if side == "left":
        return vert[r][c]
    if side == "right":
        return vert[r][c + 1]
    raise ValueError(side)


def _door_for_edge(
    a: tuple[int, int],
    b: tuple[int, int],
    row_seps: list[int],
    col_seps: list[int],
    rng: random.Random,
) -> tuple[int, int]:
    if a[0] == b[0]:
        left = a if a[1] < b[1] else b
        r0, r1 = row_seps[left[0]], row_seps[left[0] + 1]
        c = col_seps[left[1] + 1]
        interior = list(range(r0 + 1, r1))
        assert interior
        return (rng.choice(interior), c)
    top = a if a[0] < b[0] else b
    r = row_seps[top[0] + 1]
    c0, c1 = col_seps[top[1]], col_seps[top[1] + 1]
    interior = list(range(c0 + 1, c1))
    assert interior
    return (r, rng.choice(interior))


def _room_cells(room: tuple[int, int], row_seps: list[int], col_seps: list[int]) -> list[tuple[int, int]]:
    r, c = room
    return [
        (rr, cc)
        for rr in range(row_seps[r] + 1, row_seps[r + 1])
        for cc in range(col_seps[c] + 1, col_seps[c + 1])
    ]


def _place_key_positions(
    rng: random.Random,
    room: tuple[int, int],
    sides: list[str],
    row_seps: list[int],
    col_seps: list[int],
) -> dict[str, tuple[int, int]] | None:
    r, c = room
    r0, r1 = row_seps[r], row_seps[r + 1]
    c0, c1 = col_seps[c], col_seps[c + 1]
    rows = list(range(r0 + 2, r1 - 1))
    cols = list(range(c0 + 2, c1 - 1))
    if not rows or not cols:
        return None
    placed: dict[str, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()

    for side in sides:
        candidates: list[tuple[int, int]] = []
        if side in ("top", "bottom"):
            left_half = [x for x in cols if x < (c0 + c1) / 2.0]
            right_half = [x for x in cols if x > (c0 + c1) / 2.0]
            pick_right = rng.random() < 0.5
            pref = right_half if pick_right and right_half else left_half
            if not pref:
                pref = right_half if right_half else cols
            row_order = rows[:]
            rng.shuffle(row_order)
            col_order = pref[:]
            rng.shuffle(col_order)
            for rr in row_order:
                for cc in col_order:
                    candidates.append((rr, cc))
        else:
            upper_half = [x for x in rows if x < (r0 + r1) / 2.0]
            lower_half = [x for x in rows if x > (r0 + r1) / 2.0]
            pick_lower = rng.random() < 0.5
            pref = lower_half if pick_lower and lower_half else upper_half
            if not pref:
                pref = lower_half if lower_half else rows
            row_order = pref[:]
            rng.shuffle(row_order)
            col_order = cols[:]
            rng.shuffle(col_order)
            for rr in row_order:
                for cc in col_order:
                    candidates.append((rr, cc))

        chosen = None
        for cell in candidates:
            if cell in used:
                continue
            if any(abs(cell[0] - ur) + abs(cell[1] - uc) <= 1 for ur, uc in used):
                continue
            chosen = cell
            break
        if chosen is None:
            return None
        placed[side] = chosen
        used.add(chosen)

    return placed


def _place_seed(
    rng: random.Random,
    room: tuple[int, int],
    row_seps: list[int],
    col_seps: list[int],
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r, c = room
    r0, r1 = row_seps[r], row_seps[r + 1]
    c0, c1 = col_seps[c], col_seps[c + 1]
    orient = rng.choice(("h", "v"))
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orient == "h":
        for rr in range(r0 + 2, r1 - 1):
            for cc in range(c0 + 2, c1 - 2):
                pairs.append(((rr, cc), (rr, cc + 1)))
    else:
        for rr in range(r0 + 2, r1 - 2):
            for cc in range(c0 + 2, c1 - 1):
                pairs.append(((rr, cc), (rr + 1, cc)))
    rng.shuffle(pairs)
    for a, b in pairs:
        if a in blocked or b in blocked:
            continue
        if any(abs(a[0] - br) + abs(a[1] - bc) <= 1 for br, bc in blocked):
            continue
        if any(abs(b[0] - br) + abs(b[1] - bc) <= 1 for br, bc in blocked):
            continue
        return a, b
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_spans = _room_partitions(rng, ROOM_ROWS_RANGE)
    col_spans = _room_partitions(rng, ROOM_COLS_RANGE)
    m, n = len(row_spans), len(col_spans)
    row_seps = _separator_positions(row_spans)
    col_seps = _separator_positions(col_spans)
    h, w = row_seps[-1] + 1, col_seps[-1] + 1

    grid = np.zeros((h, w), dtype=int)
    horiz, vert = _edge_colors(m, n, rng)
    _paint_walls(grid, row_seps, col_seps, horiz, vert)

    rooms = [(r, c) for r in range(m) for c in range(n)]
    source = rng.choice(rooms)
    open_edges, key_edges, key_owner, reachable = _choose_open_edges(rng, m, n, source)

    for edge in open_edges - key_edges:
        a, b = tuple(edge)
        dr, dc = _door_for_edge(a, b, row_seps, col_seps, rng)
        grid[dr, dc] = BG

    room_side_keys: dict[tuple[int, int], list[str]] = {room: [] for room in rooms}
    for edge in key_edges:
        a, b = tuple(edge)
        owner = key_owner[edge]
        side = _edge_side(a, b, owner)
        room_side_keys[owner].append(side)

    key_positions: dict[tuple[int, int], dict[str, tuple[int, int]]] = {}
    blocked: set[tuple[int, int]] = set()
    for room, sides in room_side_keys.items():
        if not sides:
            continue
        sides = list(dict.fromkeys(sides))
        placed = _place_key_positions(rng, room, sides, row_seps, col_seps)
        if placed is None:
            return None
        key_positions[room] = placed
        blocked.update(placed.values())

    seed = _place_seed(rng, source, row_seps, col_seps, blocked)
    if seed is None:
        return None

    for room, side_map in key_positions.items():
        for side, cell in side_map.items():
            color = _wall_segment_color(room, side, horiz, vert)
            grid[cell] = color

    grid[seed[0]] = RED
    grid[seed[1]] = RED

    solved = main(grid)
    changed = int(np.sum(solved != grid))
    if changed < 4:
        return None
    if not np.any((solved == RED) & (grid == BG)):
        return None

    seed_room_cells = set(_room_cells(source, row_seps, col_seps))
    reached_non_source = any((r, c) not in seed_room_cells for r, c in zip(*np.where((solved == RED) & (grid == BG))))
    if len(reachable) > 1 and not reached_non_source:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
