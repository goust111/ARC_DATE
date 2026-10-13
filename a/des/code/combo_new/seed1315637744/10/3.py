import random

import numpy as np


BG = 0
MAX_GAP_WIDTH = 3
SEED_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

# --------------- solver ---------------


def _wall_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    choices = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != BG]
    assert choices, "expected at least one non-background color"
    return max(choices, key=lambda item: item[1])[0]


def _line_indices(grid: np.ndarray, wall: int, axis: int) -> list[int]:
    line_length = grid.shape[1 - axis]
    counts = np.count_nonzero(grid == wall, axis=1 - axis)
    lines = [int(i) for i, count in enumerate(counts) if line_length - MAX_GAP_WIDTH <= int(count) < line_length]
    assert len(lines) >= 2, "expected at least two wall lines per axis"
    assert lines[0] == 0 and lines[-1] == grid.shape[axis] - 1, "outer border must be walls"
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:])), "rooms must have non-empty interiors"
    return lines


def _gap_width(segment: np.ndarray, wall: int) -> int:
    assert segment.ndim == 1
    assert bool(np.all((segment == BG) | (segment == wall))), "wall segment may only contain wall or gap cells"
    mask = segment == BG
    if not bool(mask.any()):
        return 0
    gap_idx = np.flatnonzero(mask)
    assert np.all(gap_idx == np.arange(gap_idx[0], gap_idx[0] + len(gap_idx))), "gap must be contiguous"
    assert 1 <= len(gap_idx) <= MAX_GAP_WIDTH
    return int(len(gap_idx))


def _centered_span(center: int, width: int, lo: int, hi: int) -> tuple[int, int]:
    # Even widths use a deterministic left/top bias because a discrete grid has no exact half-cell center.
    start = int(center) - (int(width) - 1) // 2
    end = start + int(width) - 1
    assert lo <= start <= end <= hi, "strip width must fit inside the room"
    return start, end


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a room-filling strip directed away from the unique wall gap on that room.

    Input structure:
    - The background is black.
    - One-cell-thick maroon horizontal and vertical walls, including the outer border, partition the grid into rectangular rooms.
    - Each wall line has exactly one contiguous black gap of width one to three, and some rooms contain one colored seed pixel away from the walls.

    Rule:
    - For each room that contains a seed, inspect its four wall segments and find the unique side that contains a gap.
    - Use that gap width as the strip width, then grow the seed in the opposite direction until the strip reaches the opposite wall without painting over the walls.
    - Keep the seed, all maroon walls, all wall gaps, and every unseeded room unchanged.

    Key insight:
    - A wall gap is not an exit; it simultaneously encodes the strip width and points to the opposite growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall = _wall_color(g)
    wall_rows = _line_indices(g, wall, axis=0)
    wall_cols = _line_indices(g, wall, axis=1)
    out = g.copy()

    for room_r in range(len(wall_rows) - 1):
        r0, r1 = wall_rows[room_r], wall_rows[room_r + 1]
        for room_c in range(len(wall_cols) - 1):
            c0, c1 = wall_cols[room_c], wall_cols[room_c + 1]
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_mask = (interior != BG) & (interior != wall)
            if not bool(seed_mask.any()):
                continue

            seeds = np.argwhere(seed_mask)
            assert len(seeds) == 1, "each seeded room must contain exactly one seed pixel"
            seed_r = int(seeds[0, 0]) + r0 + 1
            seed_c = int(seeds[0, 1]) + c0 + 1
            color = int(g[seed_r, seed_c])

            top_gap = _gap_width(g[r0, c0 + 1 : c1], wall)
            bottom_gap = _gap_width(g[r1, c0 + 1 : c1], wall)
            left_gap = _gap_width(g[r0 + 1 : r1, c0], wall)
            right_gap = _gap_width(g[r0 + 1 : r1, c1], wall)
            gaps = [(side, width) for side, width in (("top", top_gap), ("bottom", bottom_gap), ("left", left_gap), ("right", right_gap)) if width > 0]
            assert len(gaps) == 1, "each seeded room must see exactly one wall gap"
            side, width = gaps[0]

            if side == "top":
                s0, s1 = _centered_span(seed_c, width, c0 + 1, c1 - 1)
                out[seed_r:r1, s0 : s1 + 1] = color
            elif side == "bottom":
                s0, s1 = _centered_span(seed_c, width, c0 + 1, c1 - 1)
                out[r0 + 1 : seed_r + 1, s0 : s1 + 1] = color
            elif side == "left":
                s0, s1 = _centered_span(seed_r, width, r0 + 1, r1 - 1)
                out[s0 : s1 + 1, seed_c:c1] = color
            else:
                assert side == "right"
                s0, s1 = _centered_span(seed_r, width, r0 + 1, r1 - 1)
                out[s0 : s1 + 1, c0 + 1 : seed_c + 1] = color

    return out


# --------------- generator ---------------

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (4, 6)
GENERATION_ATTEMPTS = 400


def _sample_room_sizes(count: int) -> list[int]:
    return [random.randint(*ROOM_SIZE_RANGE) for _ in range(int(count))]


def _line_positions(sizes: list[int]) -> list[int]:
    lines = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        lines.append(pos)
    return lines


def _paint_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int], wall: int) -> None:
    for r in wall_rows:
        grid[int(r), :] = wall
    for c in wall_cols:
        grid[:, int(c)] = wall


def _sample_horizontal_gaps(wall_rows: list[int], wall_cols: list[int]) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    room_widths = [wall_cols[i + 1] - wall_cols[i] - 1 for i in range(len(wall_cols) - 1)]
    for r in wall_rows:
        seg = random.randrange(len(room_widths))
        width = random.randint(1, min(MAX_GAP_WIDTH, room_widths[seg]))
        start = random.randint(wall_cols[seg] + 1, wall_cols[seg + 1] - width)
        specs.append((seg, start, width))
    return specs


def _sample_vertical_gaps(wall_rows: list[int], wall_cols: list[int]) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    room_heights = [wall_rows[i + 1] - wall_rows[i] - 1 for i in range(len(wall_rows) - 1)]
    for c in wall_cols:
        seg = random.randrange(len(room_heights))
        width = random.randint(1, min(MAX_GAP_WIDTH, room_heights[seg]))
        start = random.randint(wall_rows[seg] + 1, wall_rows[seg + 1] - width)
        specs.append((seg, start, width))
    return specs


def _carve_gaps(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int], h_gaps: list[tuple[int, int, int]], v_gaps: list[tuple[int, int, int]]) -> None:
    for r, (_seg, start, width) in zip(wall_rows, h_gaps):
        grid[int(r), int(start) : int(start) + int(width)] = BG
    for c, (_seg, start, width) in zip(wall_cols, v_gaps):
        grid[int(start) : int(start) + int(width), int(c)] = BG


def _seed_positions(r0: int, r1: int, c0: int, c1: int, side: str, width: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        col_lo = c0 + 1 + (width - 1) // 2
        col_hi = c1 - 1 - width // 2
        if side == "top":
            row_lo, row_hi = r0 + 1, r1 - 2
        else:
            row_lo, row_hi = r0 + 2, r1 - 1
        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                positions.append((r, c))
    else:
        row_lo = r0 + 1 + (width - 1) // 2
        row_hi = r1 - 1 - width // 2
        if side == "left":
            col_lo, col_hi = c0 + 1, c1 - 2
        else:
            col_lo, col_hi = c0 + 2, c1 - 1
        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                positions.append((r, c))
    return positions


def _eligible_rooms(
    wall_rows: list[int],
    wall_cols: list[int],
    h_gaps: list[tuple[int, int, int]],
    v_gaps: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for room_r in range(len(wall_rows) - 1):
        r0, r1 = wall_rows[room_r], wall_rows[room_r + 1]
        for room_c in range(len(wall_cols) - 1):
            c0, c1 = wall_cols[room_c], wall_cols[room_c + 1]
            gaps = []
            if h_gaps[room_r][0] == room_c:
                gaps.append(("top", int(h_gaps[room_r][2])))
            if h_gaps[room_r + 1][0] == room_c:
                gaps.append(("bottom", int(h_gaps[room_r + 1][2])))
            if v_gaps[room_c][0] == room_r:
                gaps.append(("left", int(v_gaps[room_c][2])))
            if v_gaps[room_c + 1][0] == room_r:
                gaps.append(("right", int(v_gaps[room_c + 1][2])))
            if len(gaps) != 1:
                continue

            side, width = gaps[0]
            positions = _seed_positions(r0, r1, c0, c1, side, width)
            if not positions:
                continue
            rooms.append(
                {
                    "room": (room_r, room_c),
                    "bounds": (r0, r1, c0, c1),
                    "side": side,
                    "width": width,
                    "positions": positions,
                }
            )
    return rooms


def _pick_seed_rooms(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    by_side: dict[str, list[dict[str, object]]] = {}
    for item in candidates:
        by_side.setdefault(str(item["side"]), []).append(item)

    sides = list(by_side)
    assert sides
    if len(sides) == 1:
        chosen_side_count = 1
    else:
        chosen_side_count = random.randint(2, min(4, len(sides)))

    selected: list[dict[str, object]] = []
    used_rooms: set[tuple[int, int]] = set()
    for side in random.sample(sides, chosen_side_count):
        item = random.choice(by_side[side])
        selected.append(item)
        used_rooms.add(tuple(item["room"]))

    extra_pool = [item for item in candidates if tuple(item["room"]) not in used_rooms]
    extra_max = min(2, len(extra_pool), len(SEED_COLORS) - len(selected))
    if extra_max > 0:
        extra_count = random.randint(0, extra_max)
        for item in random.sample(extra_pool, extra_count):
            selected.append(item)
    return selected


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    room_heights = _sample_room_sizes(room_rows)
    room_widths = _sample_room_sizes(room_cols)
    wall_rows = _line_positions(room_heights)
    wall_cols = _line_positions(room_widths)
    grid = np.full((wall_rows[-1] + 1, wall_cols[-1] + 1), BG, dtype=int)
    wall = 9

    _paint_walls(grid, wall_rows, wall_cols, wall)
    h_gaps = _sample_horizontal_gaps(wall_rows, wall_cols)
    v_gaps = _sample_vertical_gaps(wall_rows, wall_cols)
    _carve_gaps(grid, wall_rows, wall_cols, h_gaps, v_gaps)

    candidates = _eligible_rooms(wall_rows, wall_cols, h_gaps, v_gaps)
    if not candidates:
        return None
    if len({str(item["side"]) for item in candidates}) < 2 and room_rows * room_cols > 2:
        return None

    selected = _pick_seed_rooms(candidates)
    colors = random.sample(SEED_COLORS, len(selected))
    for color, item in zip(colors, selected):
        pos = random.choice(item["positions"])
        grid[int(pos[0]), int(pos[1])] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-gap growth input")
