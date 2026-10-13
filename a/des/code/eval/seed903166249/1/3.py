from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)
DIR_ORDER = (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)

ARROW_TEMPLATES: dict[tuple[int, int], frozenset[tuple[int, int]]] = {
    DIR_UP: frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)}),
    DIR_DOWN: frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)}),
    DIR_LEFT: frozenset({(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3)}),
    DIR_RIGHT: frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)}),
}
TEMPLATE_TO_DIR = {shape: direction for direction, shape in ARROW_TEMPLATES.items()}


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev + 1))
        start = prev = idx
    groups.append((start, prev + 1))
    return groups


def _complement_spans(length: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = 0
    for start, end in bands:
        if prev < start:
            spans.append((prev, start))
        prev = end
    if prev < length:
        spans.append((prev, length))
    return spans


def _detect_wall_and_rooms(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        full_rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        full_cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if len(full_rows) < 2 or len(full_cols) < 2:
            continue
        row_bands = _group_consecutive(full_rows)
        col_bands = _group_consecutive(full_cols)
        row_spans = _complement_spans(h, row_bands)
        col_spans = _complement_spans(w, col_bands)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue
        if len({r1 - r0 for r0, r1 in row_spans}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_spans}) != 1:
            continue
        score = len(full_rows) + len(full_cols)
        candidates.append((score, int((grid == color).sum()), color, row_spans, col_spans))
    assert candidates, "failed to detect grid walls"
    candidates.sort()
    _score, _count, wall_color, row_spans, col_spans = candidates[-1]
    return int(wall_color), row_spans, col_spans


def _detect_bg_color(grid: np.ndarray, wall_color: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    counts.pop(int(wall_color), None)
    assert counts, "expected at least one non-wall color"
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    rows = coords[:, 0] - int(coords[:, 0].min())
    cols = coords[:, 1] - int(coords[:, 1].min())
    return frozenset((int(r), int(c)) for r, c in zip(rows, cols))


def _infer_arrow_direction(coords: np.ndarray) -> tuple[int, int]:
    shape = _normalize_coords(coords)
    if shape in TEMPLATE_TO_DIR:
        return TEMPLATE_TO_DIR[shape]

    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    row_counts = [sum(1 for r, _ in shape if r == row) for row in range(h)]
    col_counts = [sum(1 for _, c in shape if c == col) for col in range(w)]

    if h >= w:
        if h >= 4 and row_counts[0] == 1 and max(row_counts[1:]) >= 3:
            return DIR_UP
        if h >= 4 and row_counts[-1] == 1 and max(row_counts[:-1]) >= 3:
            return DIR_DOWN
    if w >= h:
        if w >= 4 and col_counts[0] == 1 and max(col_counts[1:]) >= 3:
            return DIR_LEFT
        if w >= 4 and col_counts[-1] == 1 and max(col_counts[:-1]) >= 3:
            return DIR_RIGHT
    raise AssertionError(f"unrecognized arrow shape: {sorted(shape)}")


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    wall_color, row_spans, col_spans = _detect_wall_and_rooms(g)
    bg_color = _detect_bg_color(g, wall_color)

    arrows: dict[tuple[int, int], dict] = {}
    for room_r, (r0, r1) in enumerate(row_spans):
        for room_c, (c0, c1) in enumerate(col_spans):
            room = g[r0:r1, c0:c1]
            fg_mask = (room != bg_color) & (room != wall_color)
            if not bool(fg_mask.any()):
                continue
            colors = np.unique(room[fg_mask])
            assert colors.size == 1, "each arrow room must contain exactly one non-background color"
            color = int(colors[0])
            coords = np.argwhere(room == color)
            direction = _infer_arrow_direction(coords)
            global_coords = coords + np.array([[r0, c0]])
            source_key = tuple(map(int, global_coords.min(axis=0)))
            arrows[(room_r, room_c)] = {
                "room": (room_r, room_c),
                "direction": direction,
                "color": color,
                "source_key": source_key,
            }

    return {
        "grid": g,
        "wall_color": wall_color,
        "bg_color": bg_color,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "arrow_map": arrows,
        "room_rows": len(row_spans),
        "room_cols": len(col_spans),
    }


def _simulate_room_rays(room_rows: int, room_cols: int, arrow_map: dict[tuple[int, int], dict]) -> tuple[dict, dict]:
    claimed: dict[tuple[int, int], dict] = {}
    stats = defaultdict(int)
    active: list[dict] = []
    for room, arrow in sorted(arrow_map.items(), key=lambda item: item[1]["source_key"]):
        active.append(
            {
                "source_room": room,
                "current_room": room,
                "direction": tuple(arrow["direction"]),
                "color": int(arrow["color"]),
                "source_key": tuple(arrow["source_key"]),
            }
        )

    while active:
        proposals: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for ray in active:
            dr, dc = ray["direction"]
            r0, c0 = ray["current_room"]
            target = (int(r0 + dr), int(c0 + dc))
            tr, tc = target
            if not (0 <= tr < room_rows and 0 <= tc < room_cols):
                stats["boundary_stop"] += 1
                continue
            if target in arrow_map:
                stats["arrow_stop"] += 1
                continue
            if target in claimed:
                stats["filled_stop"] += 1
                continue
            proposals[target].append(ray)

        if not proposals:
            break

        next_active: list[dict] = []
        for target in sorted(proposals):
            contenders = sorted(proposals[target], key=lambda ray: ray["source_key"])
            winner = contenders[0]
            if len(contenders) > 1:
                stats["tie_room"] += 1
                stats["tie_loser"] += len(contenders) - 1
            claimed[target] = {
                "color": int(winner["color"]),
                "source_key": tuple(winner["source_key"]),
                "source_room": tuple(winner["source_room"]),
            }
            next_active.append(
                {
                    "source_room": tuple(winner["source_room"]),
                    "current_room": target,
                    "direction": tuple(winner["direction"]),
                    "color": int(winner["color"]),
                    "source_key": tuple(winner["source_key"]),
                }
            )

        active = sorted(next_active, key=lambda ray: ray["source_key"])

    stats["filled_rooms"] = len(claimed)
    return claimed, dict(stats)


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grid of equally-sized rooms separated by full wall rows/columns, arrows
    shoot their color into neighboring rooms along the arrow direction.

    Each arrow keeps advancing through empty rooms until it would leave the room
    grid, hit another arrow room, or hit a room already colored by an earlier
    propagation step. Propagation is simultaneous, and same-step collisions are
    resolved by the arrow whose source appears farther toward the top-left.
    """
    parsed = _parse_grid(grid)
    out = np.array(parsed["grid"], copy=True)
    claimed, _stats = _simulate_room_rays(parsed["room_rows"], parsed["room_cols"], parsed["arrow_map"])

    row_spans = parsed["row_spans"]
    col_spans = parsed["col_spans"]
    for (room_r, room_c), info in claimed.items():
        r0, r1 = row_spans[room_r]
        c0, c1 = col_spans[room_c]
        out[r0:r1, c0:c1] = int(info["color"])

    return out


def _room_bounds(room_r: int, room_c: int, room_h: int, room_w: int) -> tuple[int, int, int, int]:
    r0 = 1 + room_r * (room_h + 1)
    c0 = 1 + room_c * (room_w + 1)
    return r0, r0 + room_h, c0, c0 + room_w


def _ray_path(start: tuple[int, int], direction: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < room_rows and 0 <= c < room_cols):
            return path
        path.append((int(r), int(c)))


def _try_place_tie(
    rng: random.Random, room_rows: int, room_cols: int, occupied: set[tuple[int, int]]
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]] | None:
    options: list[tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]]] = []
    for row in range(room_rows):
        for c1 in range(room_cols):
            for c2 in range(c1 + 2, room_cols, 2):
                a = (row, c1)
                b = (row, c2)
                if a in occupied or b in occupied:
                    continue
                protected = {(row, c) for c in range(c1, c2 + 1)}
                options.append(([(a, DIR_RIGHT), (b, DIR_LEFT)], protected))
    for col in range(room_cols):
        for r1 in range(room_rows):
            for r2 in range(r1 + 2, room_rows, 2):
                a = (r1, col)
                b = (r2, col)
                if a in occupied or b in occupied:
                    continue
                protected = {(r, col) for r in range(r1, r2 + 1)}
                options.append(([(a, DIR_DOWN), (b, DIR_UP)], protected))
    if not options:
        return None
    return rng.choice(options)


def _try_place_arrow_block(
    rng: random.Random, room_rows: int, room_cols: int, occupied: set[tuple[int, int]]
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]] | None:
    options: list[tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]]] = []
    for row in range(room_rows):
        for c1 in range(room_cols - 2):
            for c2 in range(c1 + 2, room_cols):
                a = (row, c1)
                b = (row, c2)
                if a in occupied or b in occupied:
                    continue
                for b_dir in (DIR_UP, DIR_DOWN, DIR_RIGHT):
                    arrows = [(a, DIR_RIGHT), (b, b_dir)]
                    protected = {(row, c) for c in range(c1, c2 + 1)}
                    options.append((arrows, protected))
    for col in range(room_cols):
        for r1 in range(room_rows - 2):
            for r2 in range(r1 + 2, room_rows):
                a = (r1, col)
                b = (r2, col)
                if a in occupied or b in occupied:
                    continue
                for b_dir in (DIR_LEFT, DIR_RIGHT, DIR_DOWN):
                    arrows = [(a, DIR_DOWN), (b, b_dir)]
                    protected = {(r, col) for r in range(r1, r2 + 1)}
                    options.append((arrows, protected))
    if not options:
        return None
    return rng.choice(options)


def _try_place_filled_block(
    rng: random.Random, room_rows: int, room_cols: int, occupied: set[tuple[int, int]]
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]] | None:
    options: list[tuple[list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]]] = []
    for cross_r in range(room_rows):
        for cross_c in range(room_cols):
            for dist_a in (1, 2):
                for dist_b in (dist_a + 1, dist_a + 2):
                    src_a = (cross_r - dist_a, cross_c)
                    src_b = (cross_r, cross_c - dist_b)
                    if not (0 <= src_a[0] < room_rows and 0 <= src_b[1] < room_cols):
                        continue
                    if src_a in occupied or src_b in occupied or src_a == src_b:
                        continue
                    arrows = [(src_a, DIR_DOWN), (src_b, DIR_RIGHT)]
                    protected = set(_ray_path(src_a, DIR_DOWN, room_rows, room_cols)[:dist_a])
                    protected.update(_ray_path(src_b, DIR_RIGHT, room_rows, room_cols)[:dist_b])
                    options.append((arrows, protected))

                    src_a = (cross_r + dist_a, cross_c)
                    src_b = (cross_r, cross_c + dist_b)
                    if not (0 <= src_a[0] < room_rows and 0 <= src_b[1] < room_cols):
                        continue
                    if src_a in occupied or src_b in occupied or src_a == src_b:
                        continue
                    arrows = [(src_a, DIR_UP), (src_b, DIR_LEFT)]
                    protected = set(_ray_path(src_a, DIR_UP, room_rows, room_cols)[:dist_a])
                    protected.update(_ray_path(src_b, DIR_LEFT, room_rows, room_cols)[:dist_b])
                    options.append((arrows, protected))
    if not options:
        return None
    return rng.choice(options)


def _direction_for_room(
    rng: random.Random, room: tuple[int, int], room_rows: int, room_cols: int
) -> tuple[int, int]:
    r, c = room
    dirs = list(DIR_ORDER)
    inward: list[tuple[int, int]] = []
    if r > 0:
        inward.append(DIR_UP)
    if r < room_rows - 1:
        inward.append(DIR_DOWN)
    if c > 0:
        inward.append(DIR_LEFT)
    if c < room_cols - 1:
        inward.append(DIR_RIGHT)
    if inward and rng.random() < 0.7:
        return rng.choice(inward)
    return rng.choice(dirs)


def _path_hits_protected(
    room: tuple[int, int],
    direction: tuple[int, int],
    protected: set[tuple[int, int]],
    room_rows: int,
    room_cols: int,
) -> bool:
    if room in protected:
        return True
    return any(pos in protected for pos in _ray_path(room, direction, room_rows, room_cols))


def _draw_layout(
    rng: random.Random,
    room_rows: int,
    room_cols: int,
    room_h: int,
    room_w: int,
    placements: list[tuple[tuple[int, int], tuple[int, int]]],
    colors: list[int],
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    h = room_rows * room_h + room_rows + 1
    w = room_cols * room_w + room_cols + 1
    grid = np.full((h, w), BG, dtype=int)
    for r in range(0, h, room_h + 1):
        grid[r, :] = WALL
    for c in range(0, w, room_w + 1):
        grid[:, c] = WALL

    arrow_map: dict[tuple[int, int], dict] = {}
    for (room, direction), color in zip(placements, colors):
        template = ARROW_TEMPLATES[direction]
        rows = [r for r, _ in template]
        cols = [c for _, c in template]
        th = max(rows) + 1
        tw = max(cols) + 1
        max_off_r = room_h - th
        max_off_c = room_w - tw
        if max_off_r >= 2:
            off_r = rng.randint(1, max_off_r - 1)
        else:
            off_r = rng.randint(0, max_off_r)
        if max_off_c >= 2:
            off_c = rng.randint(1, max_off_c - 1)
        else:
            off_c = rng.randint(0, max_off_c)
        room_r, room_c = room
        r0, r1, c0, c1 = _room_bounds(room_r, room_c, room_h, room_w)
        points = []
        for dr, dc in template:
            rr = r0 + off_r + dr
            cc = c0 + off_c + dc
            grid[rr, cc] = int(color)
            points.append((rr, cc))
        arrow_map[room] = {
            "room": room,
            "direction": direction,
            "color": int(color),
            "source_key": min(points),
        }
        assert bool((grid[r0:r1, c0:c1] == WALL).sum() == 0)
    return grid, arrow_map


def _render_claims(
    grid: np.ndarray,
    room_h: int,
    room_w: int,
    claims: dict[tuple[int, int], dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for (room_r, room_c), info in claims.items():
        r0, r1, c0, c1 = _room_bounds(room_r, room_c, room_h, room_w)
        out[r0:r1, c0:c1] = int(info["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario = rng.choice(("tie", "arrow_stop", "filled_stop"))
    room_rows = rng.randint(3, 6)
    room_cols = rng.randint(3, 6)
    room_h = rng.randint(5, 7)
    room_w = rng.randint(5, 7)

    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []
    protected: set[tuple[int, int]] = set()

    if scenario == "tie":
        placed = _try_place_tie(rng, room_rows, room_cols, occupied)
    elif scenario == "arrow_stop":
        placed = _try_place_arrow_block(rng, room_rows, room_cols, occupied)
    else:
        placed = _try_place_filled_block(rng, room_rows, room_cols, occupied)
    if placed is None:
        return None

    base_arrows, base_protected = placed
    for room, direction in base_arrows:
        placements.append((room, direction))
        occupied.add(room)
    protected.update(base_protected)

    max_arrows = min(len(COLOR_POOL), max(2, (room_rows * room_cols) // 2))
    target_arrows = rng.randint(2, max_arrows)
    attempts = 0
    while len(placements) < target_arrows and attempts < 120:
        attempts += 1
        room = (rng.randrange(room_rows), rng.randrange(room_cols))
        if room in occupied:
            continue
        direction = _direction_for_room(rng, room, room_rows, room_cols)
        if _path_hits_protected(room, direction, protected, room_rows, room_cols):
            continue
        placements.append((room, direction))
        occupied.add(room)

    if len(placements) < 2:
        return None
    if len(set(direction for _room, direction in placements)) < 2:
        return None

    colors = rng.sample(COLOR_POOL, len(placements))
    grid, arrow_map = _draw_layout(rng, room_rows, room_cols, room_h, room_w, placements, colors)
    claims, stats = _simulate_room_rays(room_rows, room_cols, arrow_map)

    if stats.get("boundary_stop", 0) == 0:
        return None
    if stats.get("filled_rooms", 0) < 2:
        return None
    if scenario == "tie" and stats.get("tie_room", 0) == 0:
        return None
    if scenario == "arrow_stop" and stats.get("arrow_stop", 0) == 0:
        return None
    if scenario == "filled_stop" and stats.get("filled_stop", 0) == 0:
        return None

    expected = _render_claims(grid, room_h, room_w, claims)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return None

    for room_r in range(room_rows):
        for room_c in range(room_cols):
            r0, r1, c0, c1 = _room_bounds(room_r, room_c, room_h, room_w)
            room = expected[r0:r1, c0:c1]
            if (room_r, room_c) not in arrow_map:
                vals = set(map(int, np.unique(room)))
                if vals != {BG} and len(vals) != 1:
                    return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-arrow propagation puzzle")
