import random
from collections import Counter

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
SEPARATE_PLACEMENT_TRIES = 160
ATTACHED_PLACEMENT_TRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    candidates = [(count, color) for color, count in counts.items() if int(color) != int(bg)]
    assert candidates, "expected at least one non-background color"
    return int(max(candidates)[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _extract_rooms(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = _most_common_color(g)
    wall = _wall_color(g, bg)

    rooms: list[dict[str, int]] = []
    for cells in _components(g != wall):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in cells):
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        inner_h = int(r1 - r0 + 1)
        inner_w = int(c1 - c0 + 1)
        assert len(cells) == inner_h * inner_w, "room interior must be a full rectangle"
        assert r0 > 0 and c0 > 0 and r1 < height - 1 and c1 < width - 1, "room must be enclosed"

        outer = g[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
        assert bool(np.all(outer[0, :] == wall))
        assert bool(np.all(outer[-1, :] == wall))
        assert bool(np.all(outer[:, 0] == wall))
        assert bool(np.all(outer[:, -1] == wall))

        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_positions = np.argwhere(interior != bg)
        assert seed_positions.shape == (1, 2), "each room must contain exactly one seed pixel"
        local_r, local_c = map(int, seed_positions[0])

        rooms.append(
            {
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_r": int(r0 + local_r),
                "seed_c": int(c0 + local_c),
                "seed_color": int(interior[local_r, local_c]),
            }
        )

    assert rooms, "expected at least one enclosed room"
    return bg, wall, rooms


def _fill_interval(center: int, size: int, lower: int, upper: int) -> tuple[int, int]:
    start = int(center - (size - 1) // 2)
    stop = int(center + size // 2)
    return max(lower, start), min(upper, stop)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room seed into a clipped inner rectangle while keeping the grey walls unchanged.

    Input structure:
    - Black background surrounds one or more axis-aligned grey rectangular wall outlines.
    - Every enclosed room interior is a solid rectangular non-wall region containing
      black cells plus exactly one non-black seed pixel.
    - Rooms may be isolated or may share wall segments, but the grey walls keep the
      room interiors 4-disconnected from one another and from the outside.

    Rule:
    - Keep the grey wall pixels unchanged.
    - For each enclosed room, measure the interior width W and height H.
    - Use the seed color to draw a solid (W-2) x (H-2) rectangle centered on the
      seed; if either dimension is smaller than 1, leave only the seed.
    - When an even fill size needs a tie-break, place the extra row or column below
      or to the right of the seed, then clip the fill to the room interior so walls
      are never overwritten.

    Key insight:
    - The fill shape is not fixed: each room's hidden interior size sets the
      rectangle dimensions, and the seed position determines where that reduced
      rectangle is centered and therefore how it gets clipped by the room walls.
    """
    g = np.asarray(grid, dtype=int)
    _, wall, rooms = _extract_rooms(g)

    out = g.copy()
    for room in rooms:
        fill_h = int(room["inner_h"] - 2)
        fill_w = int(room["inner_w"] - 2)
        if fill_h < 1 or fill_w < 1:
            continue

        row_start, row_stop = _fill_interval(room["seed_r"], fill_h, room["r0"], room["r1"])
        col_start, col_stop = _fill_interval(room["seed_c"], fill_w, room["c0"], room["c1"])
        out[row_start : row_stop + 1, col_start : col_stop + 1] = int(room["seed_color"])

    out[g == wall] = wall
    return out


def _safe_seed_positions(length: int) -> list[int]:
    fill = length - 2
    low = int((fill - 1) // 2)
    high = int(length - 1 - fill // 2)
    return list(range(low, high + 1))


def _clipped_seed_positions(length: int) -> list[int]:
    fill = length - 2
    return [
        pos
        for pos in range(length)
        if pos - (fill - 1) // 2 < 0 or pos + fill // 2 >= length
    ]


def _make_room_spec(rng: random.Random, style: str, active_colors: list[int]) -> dict[str, int | str]:
    if style == "tiny":
        inner_h = 3
        inner_w = 3
        seed_r = rng.randrange(inner_h)
        seed_c = rng.randrange(inner_w)
    else:
        inner_h = rng.randint(4, 8)
        inner_w = rng.randint(4, 8)
        if style == "center":
            seed_r = rng.choice(_safe_seed_positions(inner_h))
            seed_c = rng.choice(_safe_seed_positions(inner_w))
        else:
            clipped_rows = _clipped_seed_positions(inner_h)
            clipped_cols = _clipped_seed_positions(inner_w)
            assert clipped_rows or clipped_cols, "offset room must have a clipped placement"

            clip_row = bool(clipped_rows) and (not clipped_cols or rng.random() < 0.5)
            if clip_row:
                seed_r = rng.choice(clipped_rows)
                seed_c = rng.randrange(inner_w)
                if clipped_cols and rng.random() < 0.5:
                    seed_c = rng.choice(clipped_cols)
            else:
                seed_r = rng.randrange(inner_h)
                seed_c = rng.choice(clipped_cols)
                if clipped_rows and rng.random() < 0.5:
                    seed_r = rng.choice(clipped_rows)

    return {
        "style": style,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "seed_r": int(seed_r),
        "seed_c": int(seed_c),
        "seed_color": int(rng.choice(active_colors)),
    }


def _draw_room(grid: np.ndarray, r0: int, c0: int, spec: dict[str, int | str]) -> dict[str, int]:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    seed_r = int(spec["seed_r"])
    seed_c = int(spec["seed_c"])
    seed_color = int(spec["seed_color"])

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)

    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR
    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 + 1 : r1, c0 + 1 : c1] = BG_COLOR
    grid[r0 + 1 + seed_r, c0 + 1 + seed_c] = seed_color

    return {
        "r0": int(r0),
        "r1": int(r1),
        "c0": int(c0),
        "c1": int(c1),
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _fits_separate(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> bool:
    height, width = grid.shape
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)
    if r0 < 1 or c0 < 1 or r1 >= height - 1 or c1 >= width - 1:
        return False

    pad_r0 = max(0, r0 - 1)
    pad_c0 = max(0, c0 - 1)
    pad_r1 = min(height, r1 + 2)
    pad_c1 = min(width, c1 + 2)
    return bool(np.all(grid[pad_r0:pad_r1, pad_c0:pad_c1] == BG_COLOR))


def _attach_offsets(target_low: int, target_high: int, new_size: int) -> list[int]:
    low = int(target_low - new_size + 3)
    high = int(target_high - 2)
    if low > high:
        return []
    return list(range(low, high + 1))


def _fits_attached(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    side: str,
) -> bool:
    height, width = grid.shape
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)
    if r0 < 1 or c0 < 1 or r1 >= height - 1 or c1 >= width - 1:
        return False

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            interior = r0 < r < r1 and c0 < c < c1
            current = int(grid[r, c])
            if interior:
                if current != BG_COLOR:
                    return False
                continue

            on_shared_side = (
                (side == "left" and c == c1)
                or (side == "right" and c == c0)
                or (side == "top" and r == r1)
                or (side == "bottom" and r == r0)
            )
            if current == BG_COLOR:
                continue
            if current == WALL_COLOR and on_shared_side:
                continue
            return False

    return True


def _place_room_separate(
    grid: np.ndarray,
    spec: dict[str, int | str],
    rooms: list[dict[str, int]],
    rng: random.Random,
) -> bool:
    outer_h = int(spec["inner_h"]) + 2
    outer_w = int(spec["inner_w"]) + 2
    height, width = grid.shape

    if outer_h > height - 2 or outer_w > width - 2:
        return False

    for _ in range(SEPARATE_PLACEMENT_TRIES):
        r0 = rng.randint(1, height - outer_h - 1)
        c0 = rng.randint(1, width - outer_w - 1)
        if _fits_separate(grid, r0, c0, outer_h, outer_w):
            rooms.append(_draw_room(grid, r0, c0, spec))
            return True
    return False


def _place_room_attached(
    grid: np.ndarray,
    spec: dict[str, int | str],
    rooms: list[dict[str, int]],
    rng: random.Random,
) -> bool:
    outer_h = int(spec["inner_h"]) + 2
    outer_w = int(spec["inner_w"]) + 2

    candidates = list(range(len(rooms)))
    rng.shuffle(candidates)
    sides = ["left", "right", "top", "bottom"]

    for room_index in candidates:
        target = rooms[room_index]
        side_order = sides[:]
        rng.shuffle(side_order)
        for side in side_order:
            if side == "left":
                c0 = int(target["c0"] - outer_w + 1)
                offsets = _attach_offsets(target["r0"], target["r1"], outer_h)
                rng.shuffle(offsets)
                for r0 in offsets[:ATTACHED_PLACEMENT_TRIES]:
                    if _fits_attached(grid, r0, c0, outer_h, outer_w, side):
                        rooms.append(_draw_room(grid, r0, c0, spec))
                        return True
            elif side == "right":
                c0 = int(target["c1"])
                offsets = _attach_offsets(target["r0"], target["r1"], outer_h)
                rng.shuffle(offsets)
                for r0 in offsets[:ATTACHED_PLACEMENT_TRIES]:
                    if _fits_attached(grid, r0, c0, outer_h, outer_w, side):
                        rooms.append(_draw_room(grid, r0, c0, spec))
                        return True
            elif side == "top":
                r0 = int(target["r0"] - outer_h + 1)
                offsets = _attach_offsets(target["c0"], target["c1"], outer_w)
                rng.shuffle(offsets)
                for c0 in offsets[:ATTACHED_PLACEMENT_TRIES]:
                    if _fits_attached(grid, r0, c0, outer_h, outer_w, side):
                        rooms.append(_draw_room(grid, r0, c0, spec))
                        return True
            else:
                r0 = int(target["r1"])
                offsets = _attach_offsets(target["c0"], target["c1"], outer_w)
                rng.shuffle(offsets)
                for c0 in offsets[:ATTACHED_PLACEMENT_TRIES]:
                    if _fits_attached(grid, r0, c0, outer_h, outer_w, side):
                        rooms.append(_draw_room(grid, r0, c0, spec))
                        return True

    return False


def _sample_styles(rng: random.Random, room_count: int) -> list[str]:
    need_tiny = room_count >= 3 and rng.random() < 0.6
    need_center = rng.random() < 0.7
    styles = ["offset", "center" if need_center else "offset"]
    if need_tiny:
        styles.append("tiny")
    while len(styles) < room_count:
        styles.append(rng.choice(["center", "offset", "tiny"]))
    rng.shuffle(styles)
    return styles[:room_count]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    shared_layout = rng.random() < 0.5
    room_count = rng.randint(3, 6) if shared_layout else rng.randint(2, 4)
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    active_color_count = rng.randint(2, min(5, len(SEED_COLORS)))
    active_colors = list(rng.sample(SEED_COLORS, active_color_count))
    styles = _sample_styles(rng, room_count)
    rooms: list[dict[str, int]] = []

    for index, style in enumerate(styles):
        spec = _make_room_spec(rng, style, active_colors)
        if index == 0:
            if not _place_room_separate(grid, spec, rooms, rng):
                return None
            continue

        if shared_layout and index == 1:
            if not _place_room_attached(grid, spec, rooms, rng):
                return None
            continue

        if shared_layout and rng.random() < 0.45:
            if _place_room_attached(grid, spec, rooms, rng):
                continue
            if _place_room_separate(grid, spec, rooms, rng):
                continue
            return None

        if _place_room_separate(grid, spec, rooms, rng):
            continue
        if shared_layout and _place_room_attached(grid, spec, rooms, rng):
            continue
        return None

    try:
        _, _, found_rooms = _extract_rooms(grid)
    except AssertionError:
        return None

    if len(found_rooms) != len(rooms):
        return None

    if not any(room["inner_h"] > 3 or room["inner_w"] > 3 for room in rooms):
        return None

    if not np.any(main(grid) != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth input")
