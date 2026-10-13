from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BG = 0
_MAX_ATTEMPTS = 1000


def _unique_mode(counts: Counter[int], role: str) -> int:
    assert counts, f"expected at least one {role} candidate"
    top = max(counts.values())
    winners = [int(color) for color, n in counts.items() if n == top]
    assert len(winners) == 1, f"expected a unique {role}"
    return winners[0]


def _runs_excluding(n: int, separators: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(int(n)):
        if i in separators:
            if start is not None:
                runs.append((int(start), int(i - 1)))
                start = None
        elif start is None:
            start = int(i)
    if start is not None:
        runs.append((int(start), int(n - 1)))
    return runs


def _parse_lattice(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_colors: dict[int, list[int]] = {}
    for r in range(h):
        color = int(g[r, 0])
        if np.all(g[r, :] == color):
            row_colors.setdefault(color, []).append(int(r))

    col_colors: dict[int, list[int]] = {}
    for c in range(w):
        color = int(g[0, c])
        if np.all(g[:, c] == color):
            col_colors.setdefault(color, []).append(int(c))

    wall_candidates = sorted(set(row_colors) & set(col_colors))
    assert len(wall_candidates) == 1, "expected one color forming full separator rows and columns"
    wall = int(wall_candidates[0])

    sep_rows = [r for r in row_colors[wall] if np.all(g[r, :] == wall)]
    sep_cols = [c for c in col_colors[wall] if np.all(g[:, c] == wall)]
    assert sep_rows and sep_cols, "expected separator rows and columns"

    row_bands = _runs_excluding(h, set(sep_rows))
    col_bands = _runs_excluding(w, set(sep_cols))
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a room array"

    non_wall = g[g != wall]
    bg = _unique_mode(Counter(map(int, non_wall.ravel())), "background")
    return int(bg), int(wall), row_bands, col_bands


def _extract_initial_rooms(
    grid: np.ndarray,
    bg: int,
    wall: int,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> list[list[int | None]]:
    rooms: list[list[int | None]] = []
    for r0, r1 in row_bands:
        row: list[int | None] = []
        for c0, c1 in col_bands:
            patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            seed_values = patch[(patch != int(bg)) & (patch != int(wall))]
            if seed_values.size == 0:
                row.append(None)
                continue
            assert seed_values.size == 1, "each seeded room must contain one seed pixel"
            row.append(int(seed_values[0]))
        rooms.append(row)
    return rooms


def _simulate_rooms(
    initial: list[list[int | None]],
) -> tuple[list[list[int]], list[list[int | None]], list[tuple[int, int, int, tuple[int, ...]]]]:
    room_rows = len(initial)
    room_cols = len(initial[0])
    colors: list[list[int | None]] = [[initial[r][c] for c in range(room_cols)] for r in range(room_rows)]
    fill_round: list[list[int | None]] = [
        [0 if colors[r][c] is not None else None for c in range(room_cols)] for r in range(room_rows)
    ]
    tie_skips: list[tuple[int, int, int, tuple[int, ...]]] = []

    round_index = 0
    while any(colors[r][c] is None for r in range(room_rows) for c in range(room_cols)):
        changes: list[tuple[int, int, int]] = []
        for r in range(room_rows):
            for c in range(room_cols):
                if colors[r][c] is not None:
                    continue
                votes: list[int] = []
                for dr, dc in _DIRS4:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < room_rows and 0 <= cc < room_cols and colors[rr][cc] is not None:
                        votes.append(int(colors[rr][cc]))
                if not votes:
                    continue
                counts = Counter(votes)
                top = max(counts.values())
                winners = [int(color) for color, n in counts.items() if n == top]
                if len(winners) == 1:
                    changes.append((int(r), int(c), int(winners[0])))
                else:
                    tie_skips.append((int(round_index + 1), int(r), int(c), tuple(sorted(winners))))

        assert changes, "empty rooms must eventually receive a unique majority"
        round_index += 1
        for r, c, color in changes:
            colors[r][c] = int(color)
            fill_round[r][c] = int(round_index)

    return [[int(colors[r][c]) for c in range(room_cols)] for r in range(room_rows)], fill_round, tie_skips


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a rectangular room lattice by synchronous neighbor-majority cascades from seed rooms.

    Input structure:
    - Full separator rows and columns of one wall color divide the grid into equal rectangular rooms.
    - The background fills empty rooms, and a minority of rooms contain exactly one colored seed pixel.
    - Seed colors are ordinary visible colors distinct from the background and separator wall.

    Rule:
    - First recolor each seeded room completely with its seed color.
    - Then repeat synchronous rounds: an empty room whose colored up/down/left/right neighbors have a
      unique majority color becomes that color at the end of the round; tied rooms stay empty.
    - Continue until every room is colored, while keeping separator wall pixels unchanged.

    Key insight:
    - Later-filled rooms become new votes, so the result must be computed as a round-by-round cascade
      rather than by one-step adjacency or nearest-seed distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, wall, row_bands, col_bands = _parse_lattice(g)
    initial = _extract_initial_rooms(g, bg, wall, row_bands, col_bands)
    final, _rounds, _ties = _simulate_rooms(initial)

    out = g.copy()
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(final[rr][cc])
    return out


def _room_bounds(room_r: int, room_c: int, room_h: int, room_w: int) -> tuple[int, int, int, int]:
    r0 = 1 + int(room_r) * (int(room_h) + 1)
    c0 = 1 + int(room_c) * (int(room_w) + 1)
    return int(r0), int(c0), int(r0 + room_h - 1), int(c0 + room_w - 1)


def _render_walls(room_rows: int, room_cols: int, room_h: int, room_w: int, wall: int) -> np.ndarray:
    h = int(room_rows) * int(room_h) + int(room_rows) + 1
    w = int(room_cols) * int(room_w) + int(room_cols) + 1
    grid = np.full((h, w), _BG, dtype=int)
    for r in range(0, h, int(room_h) + 1):
        grid[r, :] = int(wall)
    for c in range(0, w, int(room_w) + 1):
        grid[:, c] = int(wall)
    return grid


def _render_expected(
    final: list[list[int]],
    room_h: int,
    room_w: int,
    wall: int,
) -> np.ndarray:
    room_rows = len(final)
    room_cols = len(final[0])
    out = _render_walls(room_rows, room_cols, room_h, room_w, wall)
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0, c0, r1, c1 = _room_bounds(rr, cc, room_h, room_w)
            out[r0 : r1 + 1, c0 : c1 + 1] = int(final[rr][cc])
    return out


def _choose_room_size(room_rows: int, room_cols: int) -> tuple[int, int]:
    max_h = min(4, (30 - int(room_rows) - 1) // int(room_rows))
    max_w = min(4, (30 - int(room_cols) - 1) // int(room_cols))
    assert max_h >= 2 and max_w >= 2
    return random.randint(2, max_h), random.randint(2, max_w)


def _add_seed(seeds: dict[tuple[int, int], int], pos: tuple[int, int], color: int) -> bool:
    old = seeds.get(pos)
    if old is not None and int(old) != int(color):
        return False
    seeds[pos] = int(color)
    return True


def _seed_pixel(
    room: tuple[int, int],
    room_h: int,
    room_w: int,
    role: str,
    target_room: tuple[int, int] | None = None,
) -> tuple[int, int]:
    rr, cc = room
    r0, c0, r1, c1 = _room_bounds(rr, cc, room_h, room_w)
    if role == "near_left":
        return (int((r0 + r1) // 2), int(c0))
    if role == "far_left":
        return (int(r0), int(c0))
    if role == "far_top":
        if target_room is None:
            return (int(r0), int(c0))
        _tr, tc = target_room
        _xr0, xc0, _xr1, xc1 = _room_bounds(rr + 1, tc, room_h, room_w)
        target_c = (xc0 + xc1) // 2
        far_c = c0 if abs(c0 - target_c) >= abs(c1 - target_c) else c1
        return (int(r0), int(far_c))
    return (random.randint(r0, r1), random.randint(c0, c1))


def _render_input(
    seeds: dict[tuple[int, int], int],
    room_rows: int,
    room_cols: int,
    room_h: int,
    room_w: int,
    wall: int,
    roles: dict[tuple[int, int], str],
    target_room: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    grid = _render_walls(room_rows, room_cols, room_h, room_w, wall)
    pixels: dict[tuple[int, int], tuple[int, int]] = {}
    for room, color in seeds.items():
        pixel = _seed_pixel(room, room_h, room_w, roles.get(room, "random"), target_room)
        pixels[room] = pixel
        grid[pixel] = int(color)
    return grid, pixels


def _physical_nearest_seed_color(
    target_room: tuple[int, int],
    room_h: int,
    room_w: int,
    seeds: dict[tuple[int, int], int],
    pixels: dict[tuple[int, int], tuple[int, int]],
) -> set[int]:
    tr, tc = target_room
    r0, c0, r1, c1 = _room_bounds(tr, tc, room_h, room_w)
    target = ((r0 + r1) // 2, (c0 + c1) // 2)
    distances: list[tuple[int, int]] = []
    for room, color in seeds.items():
        sr, sc = pixels[room]
        distances.append((abs(int(sr) - int(target[0])) + abs(int(sc) - int(target[1])), int(color)))
    best = min(dist for dist, _color in distances)
    return {color for dist, color in distances if dist == best}


def _initial_matrix(
    room_rows: int,
    room_cols: int,
    seeds: dict[tuple[int, int], int],
) -> list[list[int | None]]:
    return [[seeds.get((r, c)) for c in range(room_cols)] for r in range(room_rows)]


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(5, 6)
    room_cols = random.randint(5, 7)
    room_h, room_w = _choose_room_size(room_rows, room_cols)

    wall = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    palette_pool = [color for color in range(1, 10) if color != wall]
    palette = random.sample(palette_pool, random.randint(2, min(4, len(palette_pool))))
    a_color, b_color = int(palette[0]), int(palette[1])

    tie_col = random.randint(1, room_cols - 2)
    major_r = random.randint(3, room_rows - 2)
    major_c = random.randint(1, room_cols - 2)

    tie_room = (0, tie_col)
    tie_helper = (1, tie_col)
    majority_room = (major_r, major_c)
    seeds: dict[tuple[int, int], int] = {}

    fixed = [
        ((0, tie_col - 1), a_color),
        ((0, tie_col + 1), b_color),
        ((1, tie_col + 1), b_color),
        ((major_r - 1, major_c), b_color),
        ((major_r, major_c - 1), b_color),
        ((major_r, major_c + 1), a_color),
    ]
    for pos, color in fixed:
        if not _add_seed(seeds, pos, color):
            return None

    required_empty = {
        tie_room,
        tie_helper,
        majority_room,
        (major_r + 1, major_c),
        (1, tie_col - 1),
        (2, tie_col),
    }
    if any(pos in seeds for pos in required_empty):
        return None

    available = [
        (r, c)
        for r in range(room_rows)
        for c in range(room_cols)
        if (r, c) not in seeds and (r, c) not in required_empty
    ]
    random.shuffle(available)
    extra_count = random.randint(0, max(1, (room_rows * room_cols) // 8))
    for pos in available[:extra_count]:
        seeds[pos] = int(random.choice(palette))

    initial = _initial_matrix(room_rows, room_cols, seeds)
    try:
        final, rounds, ties = _simulate_rooms(initial)
    except AssertionError:
        return None

    if final[majority_room[0]][majority_room[1]] != b_color or rounds[majority_room[0]][majority_room[1]] != 1:
        return None
    if final[tie_room[0]][tie_room[1]] != b_color or rounds[tie_room[0]][tie_room[1]] is None:
        return None
    if int(rounds[tie_room[0]][tie_room[1]]) < 2:
        return None
    if not any(round_no == 1 and r == tie_room[0] and c == tie_room[1] for round_no, r, c, _winners in ties):
        return None
    max_round = max(int(rounds[r][c]) for r in range(room_rows) for c in range(room_cols))
    if max_round < 3:
        return None

    roles = {
        (major_r, major_c + 1): "near_left",
        (major_r, major_c - 1): "far_left",
        (major_r - 1, major_c): "far_top",
    }
    grid, seed_pixels = _render_input(seeds, room_rows, room_cols, room_h, room_w, wall, roles, majority_room)
    nearest_colors = _physical_nearest_seed_color(majority_room, room_h, room_w, seeds, seed_pixels)
    if nearest_colors != {a_color}:
        return None

    expected = _render_expected(final, room_h, room_w, wall)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-majority room lattice")
