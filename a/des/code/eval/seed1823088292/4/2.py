import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
VOTE_COLORS = (1, 2, 3, 4, 7)

GEN_ATTEMPTS = 300
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_H_RANGE = (3, 5)
ROOM_W_RANGE = (3, 5)
PALETTE_SIZE_RANGE = (3, 5)
TIE_PROB = 0.55
MAX_TIE_ROOMS = 2
CLEAR_MULTICOLOR_PROB = 0.7


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_wall_color(grid):
    colors = [int(v) for v in grid.reshape(-1) if int(v) != BG]
    if not colors:
        raise ValueError("grid must contain walls")
    return Counter(colors).most_common(1)[0][0]


def _find_rooms(grid, wall_color):
    h, w = grid.shape
    rooms = []
    for pts in _components(grid != wall_color):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == wall_color):
            continue
        rooms.append((r0, c0, r1, c1))
    rooms.sort()
    if not rooms:
        raise ValueError("no enclosed rooms found")
    return rooms


def _first_seen_map(grid, wall_color):
    first = {}
    for idx, value in enumerate(grid.reshape(-1)):
        color = int(value)
        if color in (BG, wall_color) or color in first:
            continue
        first[color] = idx
    return first


def _resolve_room_color(local_counts, global_counts, first_seen):
    best = max(local_counts.values())
    candidates = [color for color, count in local_counts.items() if count == best]
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda color: (global_counts[color], first_seen.get(color, 10**9)))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each wall-delimited room by its voted majority color, then remove
    walls between adjacent rooms that end up with the same color.
    """
    g = np.asarray(grid, dtype=int)
    wall_color = _infer_wall_color(g)
    rooms = _find_rooms(g, wall_color)

    vote_mask = (g != BG) & (g != wall_color)
    global_counts = Counter(map(int, g[vote_mask].reshape(-1)))
    first_seen = _first_seen_map(g, wall_color)

    out = g.copy()
    room_fills = {}
    for room in rooms:
        r0, c0, r1, c1 = room
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        votes = window[(window != BG) & (window != wall_color)]
        if votes.size == 0:
            raise ValueError("each room must contain at least one vote pixel")
        local_counts = Counter(map(int, votes.reshape(-1)))
        fill = _resolve_room_color(local_counts, global_counts, first_seen)
        room_fills[room] = fill
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    for i, room_a in enumerate(rooms):
        r0, c0, r1, c1 = room_a
        fill_a = room_fills[room_a]
        for room_b in rooms[i + 1 :]:
            rr0, cc0, rr1, cc1 = room_b
            if fill_a != room_fills[room_b]:
                continue
            if r0 == rr0 and r1 == rr1 and c1 + 2 == cc0:
                out[r0 : r1 + 1, c1 + 1] = fill_a
            elif c0 == cc0 and c1 == cc1 and r1 + 2 == rr0:
                out[r1 + 1, c0 : c1 + 1] = fill_a

    h, w = out.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if out[r, c] != wall_color:
                continue
            neighbors = [
                int(out[r - 1, c]),
                int(out[r + 1, c]),
                int(out[r, c - 1]),
                int(out[r, c + 1]),
            ]
            if neighbors[0] != wall_color and len(set(neighbors)) == 1:
                out[r, c] = neighbors[0]

    return out


def _adjacent_pairs(n_rows, n_cols):
    pairs = []
    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols:
                pairs.append(((r, c), (r, c + 1)))
            if r + 1 < n_rows:
                pairs.append(((r, c), (r + 1, c)))
    return pairs


def _sample_majorities(rng, n_rows, n_cols, palette):
    commonness = list(palette)
    rng.shuffle(commonness)
    weights = [i + 1 for i in range(len(commonness))]
    pairs = _adjacent_pairs(n_rows, n_cols)
    common_color = commonness[-1]
    for _ in range(100):
        picks = rng.choices(commonness, weights=weights, k=n_rows * n_cols)
        majors = np.array(picks, dtype=int).reshape(n_rows, n_cols)
        used = {int(v) for v in majors.reshape(-1)}
        if len(used) < 2 or not np.any(majors == common_color):
            continue
        same = 0
        diff = 0
        for (ar, ac), (br, bc) in pairs:
            if int(majors[ar, ac]) == int(majors[br, bc]):
                same += 1
            else:
                diff += 1
        if same > 0 and diff > 0:
            return majors, commonness
    return None, None


def _choose_tie_rooms(rng, majors, common_color):
    if rng.random() >= TIE_PROB:
        return []
    positions = [
        (r, c)
        for r in range(majors.shape[0])
        for c in range(majors.shape[1])
        if int(majors[r, c]) != common_color
    ]
    if not positions:
        return None
    rng.shuffle(positions)
    distinct = []
    seen_colors = set()
    for pos in positions:
        color = int(majors[pos])
        if color in seen_colors:
            continue
        distinct.append(pos)
        seen_colors.add(color)
    if not distinct:
        return None
    count = rng.randint(1, min(MAX_TIE_ROOMS, len(distinct)))
    return distinct[:count]


def _make_clear_votes(rng, majority, palette, weight_map, require_multicolor=False):
    if require_multicolor:
        total = rng.randint(3, 5)
        multicolor = True
    else:
        total = rng.randint(1, 5)
        multicolor = total >= 3 and rng.random() < CLEAR_MULTICOLOR_PROB
    if not multicolor:
        return [majority] * total

    n_other = len(palette) - 1
    valid_majorities = [
        count
        for count in range(total // 2 + 1, total)
        if total - count <= n_other * (count - 1)
    ]
    if not valid_majorities:
        return [majority] * total
    majority_count = rng.choice(valid_majorities)
    remaining = total - majority_count
    others = [color for color in palette if color != majority]
    other_counts = {color: 0 for color in others}
    while remaining > 0:
        candidates = [color for color in others if other_counts[color] < majority_count - 1]
        color = rng.choices(candidates, weights=[weight_map[col] for col in candidates], k=1)[0]
        other_counts[color] += 1
        remaining -= 1

    votes = [majority] * majority_count
    for color, count in other_counts.items():
        votes.extend([color] * count)
    rng.shuffle(votes)
    return votes


def _room_origin(row, col, room_h, room_w):
    return 1 + row * (room_h + 1), 1 + col * (room_w + 1)


def _place_votes(grid, row, col, room_h, room_w, votes, rng):
    r0, c0 = _room_origin(row, col, room_h, room_w)
    cells = [(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)]
    for (r, c), color in zip(rng.sample(cells, len(votes)), votes):
        grid[r, c] = color


def _build_grid(n_rows, n_cols, room_h, room_w):
    height = n_rows * room_h + n_rows + 1
    width = n_cols * room_w + n_cols + 1
    grid = np.full((height, width), WALL, dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            r0, c0 = _room_origin(r, c, room_h, room_w)
            grid[r0 : r0 + room_h, c0 : c0 + room_w] = BG
    return grid


def _gen_attempt(rng):
    n_rows = rng.randint(*ROOM_ROWS_RANGE)
    n_cols = rng.randint(*ROOM_COLS_RANGE)
    room_h = rng.randint(*ROOM_H_RANGE)
    room_w = rng.randint(*ROOM_W_RANGE)

    palette_size = rng.randint(*PALETTE_SIZE_RANGE)
    palette = list(rng.sample(VOTE_COLORS, palette_size))

    majors, commonness = _sample_majorities(rng, n_rows, n_cols, palette)
    if majors is None:
        return None
    common_color = commonness[-1]
    weight_map = {color: rank + 1 for rank, color in enumerate(commonness)}

    tie_rooms = _choose_tie_rooms(rng, majors, common_color)
    if tie_rooms is None:
        return None
    tie_set = set(tie_rooms)

    clear_rooms = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if (r, c) not in tie_set
    ]
    if not clear_rooms:
        return None

    force_multicolor = None
    if not tie_rooms or rng.random() < 0.5:
        force_multicolor = rng.choice(clear_rooms)

    room_votes = {}
    global_counts = Counter()
    for pos in clear_rooms:
        majority = int(majors[pos])
        require_multicolor = pos == force_multicolor
        votes = _make_clear_votes(rng, majority, palette, weight_map, require_multicolor)
        room_votes[pos] = votes
        global_counts.update(votes)

    for pos in tie_rooms:
        majority = int(majors[pos])
        candidates = [color for color in palette if color != majority and global_counts[color] > global_counts[majority]]
        if not candidates:
            return None
        competitor = max(candidates, key=lambda color: (global_counts[color], weight_map[color]))
        tie_count = rng.choice((1, 2))
        votes = [majority] * tie_count + [competitor] * tie_count
        room_votes[pos] = votes
        global_counts.update(votes)

    if not any(len(set(votes)) > 1 for votes in room_votes.values()):
        return None

    grid = _build_grid(n_rows, n_cols, room_h, room_w)
    for (r, c), votes in room_votes.items():
        _place_votes(grid, r, c, room_h, room_w, votes, rng)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if len(_find_rooms(grid, WALL)) != n_rows * n_cols:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
