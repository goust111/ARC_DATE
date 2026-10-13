from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _label_black_chambers(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = grid.shape
    chamber_ids = np.full((h, w), -1, dtype=int)
    chambers: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG or chamber_ids[r, c] != -1:
                continue
            cid = len(chambers)
            q = deque([(r, c)])
            chamber_ids[r, c] = cid
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != BG or chamber_ids[nr, nc] != -1:
                        continue
                    chamber_ids[nr, nc] = cid
                    q.append((nr, nc))
            chambers.append(np.array(coords, dtype=int))

    return chambers, chamber_ids


def _iter_edge_positions(shape: tuple[int, int]):
    h, w = shape
    for c in range(1, w - 1):
        yield (0, c, 1, 0, "top")
        yield (h - 1, c, -1, 0, "bottom")
    for r in range(1, h - 1):
        yield (r, 0, 0, 1, "left")
        yield (r, w - 1, 0, -1, "right")


def _trace_edge_vote(
    grid: np.ndarray,
    chamber_ids: np.ndarray,
    edge_pos: tuple[int, int, int, int, str],
) -> dict:
    r0, c0, dr, dc, side = edge_pos
    r = r0 + dr
    c = c0 + dc
    target = None

    while _in_bounds(grid.shape, r, c):
        val = int(grid[r, c])
        if val == WALL:
            if target is None:
                return {
                    "pos": (r0, c0),
                    "side": side,
                    "target": None,
                    "behind_wall_bg": False,
                }

            rr, cc = r, c
            while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == WALL:
                rr += dr
                cc += dc
            behind_wall_bg = False
            while _in_bounds(grid.shape, rr, cc):
                if int(grid[rr, cc]) == BG:
                    behind_wall_bg = True
                    break
                rr += dr
                cc += dc
            return {
                "pos": (r0, c0),
                "side": side,
                "target": target,
                "behind_wall_bg": behind_wall_bg,
            }

        if val == BG and target is None:
            cid = int(chamber_ids[r, c])
            assert cid >= 0
            target = cid

        r += dr
        c += dc

    return {
        "pos": (r0, c0),
        "side": side,
        "target": target,
        "behind_wall_bg": False,
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    chambers, chamber_ids = _label_black_chambers(grid)
    markers = []
    votes: dict[int, list[int]] = defaultdict(list)
    chamber_sides: dict[int, set[str]] = defaultdict(set)

    for edge_pos in _iter_edge_positions(grid.shape):
        r, c, _dr, _dc, _side = edge_pos
        color = int(grid[r, c])
        if color in (BG, WALL):
            continue
        info = _trace_edge_vote(grid, chamber_ids, edge_pos)
        info["color"] = color
        markers.append(info)
        if info["target"] is not None:
            cid = int(info["target"])
            votes[cid].append(color)
            chamber_sides[cid].add(info["side"])

    return {
        "chambers": chambers,
        "chamber_ids": chamber_ids,
        "markers": markers,
        "votes": votes,
        "chamber_sides": chamber_sides,
    }


def _majority_color(colors: list[int]) -> int:
    counts = Counter(colors)
    top = counts.most_common()
    assert top and (len(top) == 1 or top[0][1] > top[1][1]), "expected a unique majority vote"
    return int(top[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """Color each reachable chamber by the majority color of the edge markers that first see it.

    Input structure
    - The background is black and grey wall pixels divide the black area into 4-connected chambers.
    - The only other colors are single-pixel markers placed on the outer border of the grid.

    Rule
    - Each border marker looks straight inward, perpendicular to its edge, and votes for the first black chamber reached before the first grey wall; a marker whose very first interior hit is a wall casts no vote.
    - Every chamber with at least one vote is filled entirely with its unique majority marker color.
    - Grey walls, unused black chambers, and the original border markers stay unchanged.

    Key insight
    - The colored edge pixels are not line segments to extend: they are remote voters, and the same grey walls both split chambers and truncate sight lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    analysis = _analyze_grid(g)
    for cid, colors in analysis["votes"].items():
        fill = _majority_color(colors)
        coords = analysis["chambers"][cid]
        out[coords[:, 0], coords[:, 1]] = fill
    return out


# --------------- generator (o5x) ---------------


def _build_axis_lengths(count: int, thin_first: bool, thin_last: bool) -> list[int]:
    lengths = []
    for idx in range(count):
        is_outer = idx == 0 or idx == count - 1
        if idx == 0 and thin_first:
            lengths.append(1)
        elif idx == count - 1 and thin_last:
            lengths.append(1)
        elif is_outer:
            lengths.append(random.randint(2, 4))
        else:
            lengths.append(random.randint(2, 5))
    return lengths


def _axis_layout(lengths: list[int]) -> tuple[list[int], list[int], list[int], int]:
    starts: list[int] = []
    ends: list[int] = []
    walls: list[int] = []
    cursor = 0
    for idx, size in enumerate(lengths):
        starts.append(cursor)
        ends.append(cursor + size - 1)
        cursor += size
        if idx < len(lengths) - 1:
            walls.append(cursor)
            cursor += 1
    return starts, ends, walls, cursor


def _cell_boxes(
    row_starts: list[int],
    row_ends: list[int],
    col_starts: list[int],
    col_ends: list[int],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes = {}
    for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
        for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
            boxes[(i, j)] = (r0, r1, c0, c1)
    return boxes


def _cell_neighbors(cell: tuple[int, int], n_rows: int, n_cols: int):
    i, j = cell
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < n_rows and 0 <= nj < n_cols:
            yield (ni, nj)


def _boundary_cells(n_rows: int, n_cols: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, j) for j in range(n_cols)],
        "bottom": [(n_rows - 1, j) for j in range(n_cols)],
        "left": [(i, 0) for i in range(n_rows)],
        "right": [(i, n_cols - 1) for i in range(n_rows)],
    }


def _cell_path(
    start: tuple[int, int],
    end: tuple[int, int],
    n_rows: int,
    n_cols: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cell = q.popleft()
        if cell == end:
            break
        nbrs = list(_cell_neighbors(cell, n_rows, n_cols))
        random.shuffle(nbrs)
        for nxt in nbrs:
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cell
            q.append(nxt)
    if end not in prev:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _carve_opening(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> None:
    ia, ja = a
    ib, jb = b
    r0a, r1a, c0a, c1a = boxes[a]
    r0b, r1b, c0b, c1b = boxes[b]

    if ia != ib:
        top, bottom = (a, b) if ia < ib else (b, a)
        wall_row = boxes[top][1] + 1
        c0 = max(boxes[top][2], boxes[bottom][2])
        c1 = min(boxes[top][3], boxes[bottom][3])
        grid[wall_row, random.randint(c0, c1)] = BG
        return

    left, right = (a, b) if ja < jb else (b, a)
    wall_col = boxes[left][3] + 1
    r0 = max(boxes[left][0], boxes[right][0])
    r1 = min(boxes[left][1], boxes[right][1])
    grid[random.randint(r0, r1), wall_col] = BG


def _build_base_grid() -> np.ndarray | None:
    n_rows = random.choice((3, 4))
    n_cols = random.choice((3, 4))

    thin_sides = set(random.sample(("top", "bottom", "left", "right"), k=random.choice((1, 2))))
    row_lengths = _build_axis_lengths(n_rows, "top" in thin_sides, "bottom" in thin_sides)
    col_lengths = _build_axis_lengths(n_cols, "left" in thin_sides, "right" in thin_sides)

    row_starts, row_ends, wall_rows, h = _axis_layout(row_lengths)
    col_starts, col_ends, wall_cols, w = _axis_layout(col_lengths)
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    boxes = _cell_boxes(row_starts, row_ends, col_starts, col_ends)
    interior_cells = [(i, j) for i in range(1, n_rows - 1) for j in range(1, n_cols - 1)]
    if not interior_cells:
        return None
    sealed = random.choice(interior_cells)

    by_side = _boundary_cells(n_rows, n_cols)
    path = None
    for _ in range(40):
        start_side = random.choice(("top", "left", "bottom", "right"))
        end_side = random.choice([s for s in ("top", "left", "bottom", "right") if s != start_side])
        start = random.choice([c for c in by_side[start_side] if c != sealed])
        end = random.choice([c for c in by_side[end_side] if c != sealed])
        path = _cell_path(start, end, n_rows, n_cols, {sealed})
        if path is not None and len(path) >= 3:
            break
    if path is None:
        return None

    open_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a, b in zip(path, path[1:]):
        open_edges.add(_edge_key(a, b))

    path_set = set(path)
    for cell in path:
        nbrs = [n for n in _cell_neighbors(cell, n_rows, n_cols) if n != sealed and n not in path_set]
        random.shuffle(nbrs)
        for nxt in nbrs[:1]:
            if random.random() < 0.45:
                open_edges.add(_edge_key(cell, nxt))

    for i in range(n_rows):
        for j in range(n_cols):
            cell = (i, j)
            for nxt in _cell_neighbors(cell, n_rows, n_cols):
                if cell > nxt or cell == sealed or nxt == sealed:
                    continue
                if _edge_key(cell, nxt) in open_edges:
                    continue
                if random.random() < 0.2:
                    open_edges.add(_edge_key(cell, nxt))

    for a, b in open_edges:
        _carve_opening(grid, a, b, boxes)

    return grid


def _edge_candidates(grid: np.ndarray) -> tuple[list[dict], list[dict], list[np.ndarray]]:
    chambers, chamber_ids = _label_black_chambers(grid)
    valid: list[dict] = []
    invalid: list[dict] = []
    for edge_pos in _iter_edge_positions(grid.shape):
        r, c, _dr, _dc, _side = edge_pos
        if int(grid[r, c]) != BG:
            continue
        info = _trace_edge_vote(grid, chamber_ids, edge_pos)
        if info["target"] is None:
            invalid.append(info)
        else:
            valid.append(info)
    return valid, invalid, chambers


def _can_add_marker(pos: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    if pos in used:
        return False
    for rr, cc in used:
        if abs(rr - pos[0]) + abs(cc - pos[1]) == 1:
            return False
    return True


def _pick_primary_positions(cands: list[dict], total: int, used: set[tuple[int, int]]) -> list[dict] | None:
    available = [c for c in cands if _can_add_marker(c["pos"], used)]
    if len(available) < total:
        return None

    selected: list[dict] = []
    local_used = set(used)
    behind = [c for c in available if c["behind_wall_bg"]]
    first_pool = behind or available
    first = random.choice(first_pool)
    selected.append(first)
    local_used.add(first["pos"])

    sides = {first["side"]}
    other_side = [c for c in available if c["side"] not in sides and _can_add_marker(c["pos"], local_used)]
    if other_side:
        second = random.choice(other_side)
        selected.append(second)
        local_used.add(second["pos"])
        sides.add(second["side"])

    while len(selected) < total:
        pool = [c for c in available if c["pos"] not in {s["pos"] for s in selected} and _can_add_marker(c["pos"], local_used)]
        if not pool:
            break
        choice = random.choice(pool)
        selected.append(choice)
        local_used.add(choice["pos"])
        sides.add(choice["side"])

    if len(selected) != total or len(sides) < 2 or not any(c["behind_wall_bg"] for c in selected):
        return None
    return selected


def _pick_positions(cands: list[dict], total: int, used: set[tuple[int, int]]) -> list[dict] | None:
    available = [c for c in cands if _can_add_marker(c["pos"], used)]
    random.shuffle(available)
    selected: list[dict] = []
    local_used = set(used)
    for cand in available:
        if _can_add_marker(cand["pos"], local_used):
            selected.append(cand)
            local_used.add(cand["pos"])
            if len(selected) == total:
                return selected
    return None


def _vote_colors(total: int, winner: int, palette: list[int], mixed: bool) -> list[int]:
    others = [c for c in palette if c != winner]
    if not mixed:
        return [winner] * total

    if total == 1:
        return [winner]
    if total == 2:
        return [winner, winner]
    if total == 3:
        colors = [winner, winner, others[0]]
    elif total == 4:
        if random.random() < 0.5:
            colors = [winner, winner, others[0], others[1]]
        else:
            colors = [winner, winner, winner, others[0]]
    else:
        if random.random() < 0.5:
            colors = [winner, winner, winner, others[0], others[1]]
        else:
            colors = [winner, winner, winner, winner, others[0]]
    random.shuffle(colors)
    return colors


def _plan_markers(base_grid: np.ndarray) -> list[tuple[tuple[int, int], int]] | None:
    valid, invalid, chambers = _edge_candidates(base_grid)
    if len(chambers) < 3 or not invalid:
        return None

    grouped: dict[int, list[dict]] = defaultdict(list)
    for cand in valid:
        grouped[int(cand["target"])].append(cand)

    if len(grouped) < 2:
        return None

    unreachable = [idx for idx in range(len(chambers)) if idx not in grouped]
    if not unreachable:
        return None

    primary_choices = [
        cid
        for cid, cands in grouped.items()
        if len(cands) >= 3
        and len({c["side"] for c in cands}) >= 2
        and any(c["behind_wall_bg"] for c in cands)
    ]
    if not primary_choices:
        return None
    primary = random.choice(primary_choices)

    secondary_choices = [cid for cid, cands in grouped.items() if cid != primary and len(cands) >= 2]
    if not secondary_choices:
        return None
    secondary = random.choice(secondary_choices)

    tertiary = None
    tertiary_choices = [cid for cid, cands in grouped.items() if cid not in {primary, secondary} and len(cands) >= 1]
    if tertiary_choices and random.random() < 0.6:
        tertiary = random.choice(tertiary_choices)

    chamber_order = [primary, secondary] + ([tertiary] if tertiary is not None else [])
    winners = random.sample(MARKER_COLORS, k=len(chamber_order))
    palette = list(MARKER_COLORS)
    placements: list[tuple[tuple[int, int], int]] = []
    used: set[tuple[int, int]] = set()

    for idx, cid in enumerate(chamber_order):
        cands = grouped[cid]
        if idx == 0:
            total = min(len(cands), random.choice((3, 4)))
            picked = _pick_primary_positions(cands, total, used)
            mixed = True
        elif idx == 1:
            total = min(len(cands), random.choice((2, 3)))
            total = max(total, 2)
            picked = _pick_positions(cands, total, used)
            mixed = total >= 3 and random.random() < 0.5
        else:
            total = 1 if len(cands) == 1 else random.choice((1, 2))
            picked = _pick_positions(cands, total, used)
            mixed = False

        if picked is None:
            return None

        colors = _vote_colors(len(picked), winners[idx], palette, mixed=mixed)
        for cand, color in zip(picked, colors):
            placements.append((cand["pos"], int(color)))
            used.add(cand["pos"])

    invalid_count = min(len(invalid), random.choice((1, 2, 3)))
    invalid_picks = _pick_positions(invalid, invalid_count, used)
    if invalid_picks is None or len(invalid_picks) < 1:
        return None

    invalid_colors = [c for c in MARKER_COLORS if c not in winners]
    if not invalid_colors:
        invalid_colors = list(MARKER_COLORS)
    for cand in invalid_picks:
        placements.append((cand["pos"], random.choice(invalid_colors)))
        used.add(cand["pos"])

    if len({pos for pos, _color in placements}) != len(placements):
        return None
    return placements


def _apply_markers(base_grid: np.ndarray, placements: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    grid = base_grid.copy()
    for (r, c), color in placements:
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    non_special = (grid != BG) & (grid != WALL)
    marker_coords = np.argwhere(non_special)
    if marker_coords.size == 0:
        return False
    if any(0 < r < grid.shape[0] - 1 and 0 < c < grid.shape[1] - 1 for r, c in marker_coords):
        return False

    marker_set = {tuple(map(int, coord)) for coord in marker_coords}
    for r, c in marker_set:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in marker_set:
                return False

    analysis = _analyze_grid(grid)
    chambers = analysis["chambers"]
    votes = analysis["votes"]
    markers = analysis["markers"]

    if len(chambers) < 3 or len(markers) < 5:
        return False

    valid_markers = [m for m in markers if m["target"] is not None]
    invalid_markers = [m for m in markers if m["target"] is None]
    if len(valid_markers) < 4 or len(invalid_markers) < 1:
        return False
    if len(votes) < 2:
        return False
    if len(chambers) - len(votes) < 1:
        return False
    if len({m["side"] for m in markers}) < 2:
        return False
    if not any(m["behind_wall_bg"] for m in valid_markers):
        return False
    if not any(len(set(vote_colors)) >= 2 for vote_colors in votes.values()):
        return False
    if not any(len({m["side"] for m in valid_markers if m["target"] == cid}) >= 2 for cid in votes):
        return False

    for vote_colors in votes.values():
        counts = Counter(vote_colors)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] <= top[1][1]:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[grid == WALL] == WALL):
        return False
    if not np.array_equal(out[non_special], grid[non_special]):
        return False

    changed = False
    for cid, coords in enumerate(chambers):
        vals = out[coords[:, 0], coords[:, 1]]
        if cid in votes:
            expected = _majority_color(votes[cid])
            if not np.all(vals == expected):
                return False
            if expected != BG:
                changed = True
        else:
            if not np.all(vals == BG):
                return False

    return changed


def _gen_attempt() -> np.ndarray | None:
    base_grid = _build_base_grid()
    if base_grid is None:
        return None

    placements = _plan_markers(base_grid)
    if placements is None:
        return None

    grid = _apply_markers(base_grid, placements)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-voting chamber puzzle")
