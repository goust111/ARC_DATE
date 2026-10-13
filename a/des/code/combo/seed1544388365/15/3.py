import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for comp in components_mask(grid == WALL):
        r0, c0, r1, c1 = _bbox(comp)
        comp_set = set(comp)
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0 + 1, r1):
            expected.add((r, c0))
            expected.add((r, c1))

        missing = list(expected - comp_set)
        extras = comp_set - expected
        assert not extras, "grey pixels must lie on rectangular room borders"
        assert len(missing) == 1, "each room must have exactly one wall gap"

        gap_r, gap_c = missing[0]
        hits = {
            "up": gap_r == r0,
            "down": gap_r == r1,
            "left": gap_c == c0,
            "right": gap_c == c1,
        }
        assert sum(hits.values()) == 1, "gap must lie on exactly one wall"
        gap_side = next(side for side, ok in hits.items() if ok)

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": (gap_r, gap_c),
                "gap_side": gap_side,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    assert rooms, "expected at least one grey room"
    return rooms


def _room_payload(
    grid: np.ndarray, room: dict
) -> tuple[tuple[int, int], str, list[tuple[int, int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inside_mask = (interior != BG) & (interior != WALL)
    inside_components = components_mask(inside_mask)
    assert inside_components, "room interior must contain a marker and a pattern"

    singleton_components = [comp for comp in inside_components if len(comp) == 1]
    assert len(singleton_components) == 1, "room must contain exactly one singleton marker"

    marker_rel_r, marker_rel_c = singleton_components[0][0]
    marker = (r0 + 1 + marker_rel_r, c0 + 1 + marker_rel_c)

    pattern_cells: list[tuple[int, int, int]] = []
    pattern_coords: list[tuple[int, int]] = []
    for rel_r, rel_c in np.argwhere(inside_mask):
        abs_r = r0 + 1 + int(rel_r)
        abs_c = c0 + 1 + int(rel_c)
        if (abs_r, abs_c) == marker:
            continue
        pattern_coords.append((abs_r, abs_c))
        pattern_cells.append((abs_r, abs_c, int(grid[abs_r, abs_c])))

    assert pattern_cells, "room must contain a non-empty pattern"
    pr0, pc0, pr1, pc1 = _bbox(pattern_coords)

    mr, mc = marker
    side_tests = {
        "up": mr < pr0 and pc0 <= mc <= pc1,
        "down": mr > pr1 and pc0 <= mc <= pc1,
        "left": mc < pc0 and pr0 <= mr <= pr1,
        "right": mc > pc1 and pr0 <= mr <= pr1,
    }
    assert sum(side_tests.values()) == 1, "marker must lie on exactly one side of the pattern"
    direction = next(side for side, ok in side_tests.items() if ok)
    return marker, direction, pattern_cells, (pr0, pc0, pr1, pc1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each room's launch direction from the singleton marker, and only eject
    the pattern when that direction matches the wall gap side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    rooms = _find_rooms(g)
    for room in rooms:
        marker, direction, pattern_cells, (pr0, pc0, pr1, pc1) = _room_payload(g, room)
        r0, c0, r1, c1 = room["bbox"]
        mr, mc = marker
        out[mr, mc] = BG

        if direction != room["gap_side"]:
            continue

        if direction == "up":
            dr, dc = (r0 - 1) - pr1, 0
        elif direction == "down":
            dr, dc = (r1 + 1) - pr0, 0
        elif direction == "left":
            dr, dc = 0, (c0 - 1) - pc1
        else:
            dr, dc = 0, (c1 + 1) - pc0

        for r, c, _ in pattern_cells:
            out[r, c] = BG

        targets: list[tuple[int, int, int]] = []
        for r, c, color in pattern_cells:
            nr, nc = r + dr, c + dc
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "ejected pattern must stay on-grid"
            assert int(out[nr, nc]) == BG, "ejected pattern must land on background"
            targets.append((nr, nc, color))

        for nr, nc, color in targets:
            out[nr, nc] = color

    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (18, 26)
ROOM_COUNT_RANGE = (1, 3)
ROOM_PADDING = 1
MAX_ROOM_PLACE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 300


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        raw_h = rng.randint(2, 4)
        raw_w = rng.randint(2, 4)
        target_area = rng.randint(3, min(raw_h * raw_w, 7))

        cells = {(rng.randrange(raw_h), rng.randrange(raw_w))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target_area:
            continue

        raw = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            raw[r, c] = True

        coords = np.argwhere(raw)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        shape = raw[r0 : r1 + 1, c0 : c1 + 1]
        if shape.shape[0] < 2 or shape.shape[1] < 2:
            continue
        return shape
    raise RuntimeError("failed to sample a valid pattern shape")


def _marker_margins(side: str) -> tuple[int, int, int, int]:
    return (
        2 if side == "up" else 1,
        2 if side == "down" else 1,
        2 if side == "left" else 1,
        2 if side == "right" else 1,
    )


def _reserve_bounds(
    r0: int, c0: int, room_h: int, room_w: int, gap_side: str, depth: int
) -> tuple[int, int, int, int]:
    rr0 = r0 - ROOM_PADDING - (depth if gap_side == "up" else 0)
    rr1 = r0 + room_h - 1 + ROOM_PADDING + (depth if gap_side == "down" else 0)
    cc0 = c0 - ROOM_PADDING - (depth if gap_side == "left" else 0)
    cc1 = c0 + room_w - 1 + ROOM_PADDING + (depth if gap_side == "right" else 0)
    return rr0, rr1, cc0, cc1


def _place_room(
    rng: random.Random, occupied: np.ndarray, spec: dict, grid_h: int, grid_w: int
) -> bool:
    room_h = spec["interior_h"] + 2
    room_w = spec["interior_w"] + 2
    depth = spec["shape"].shape[0] if spec["gap_side"] in ("up", "down") else spec["shape"].shape[1]

    min_r0 = ROOM_PADDING + (depth if spec["gap_side"] == "up" else 0)
    max_r0 = grid_h - room_h - ROOM_PADDING - (depth if spec["gap_side"] == "down" else 0)
    min_c0 = ROOM_PADDING + (depth if spec["gap_side"] == "left" else 0)
    max_c0 = grid_w - room_w - ROOM_PADDING - (depth if spec["gap_side"] == "right" else 0)
    if min_r0 > max_r0 or min_c0 > max_c0:
        return False

    for _ in range(MAX_ROOM_PLACE_ATTEMPTS):
        r0 = rng.randint(min_r0, max_r0)
        c0 = rng.randint(min_c0, max_c0)
        rr0, rr1, cc0, cc1 = _reserve_bounds(r0, c0, room_h, room_w, spec["gap_side"], depth)
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        spec["bbox"] = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        return True
    return False


def _make_room_spec(
    rng: random.Random, pattern_color: int, marker_color: int, match: bool
) -> dict:
    shape = _random_shape(rng)
    shape_h, shape_w = shape.shape
    gap_side = rng.choice(DIRECTIONS)
    marker_side = gap_side if match else rng.choice([d for d in DIRECTIONS if d != gap_side])

    top_margin, bottom_margin, left_margin, right_margin = _marker_margins(marker_side)
    min_interior_h = shape_h + top_margin + bottom_margin
    min_interior_w = shape_w + left_margin + right_margin
    interior_h = min_interior_h + rng.randint(0, 3)
    interior_w = min_interior_w + rng.randint(0, 3)

    pat_top = rng.randint(top_margin, interior_h - shape_h - bottom_margin)
    pat_left = rng.randint(left_margin, interior_w - shape_w - right_margin)

    return {
        "shape": shape,
        "gap_side": gap_side,
        "marker_side": marker_side,
        "pattern_color": pattern_color,
        "marker_color": marker_color,
        "interior_h": interior_h,
        "interior_w": interior_w,
        "pat_top": pat_top,
        "pat_left": pat_left,
    }


def _render_room(grid: np.ndarray, rng: random.Random, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    shape = spec["shape"]
    shape_h, shape_w = shape.shape
    interior_top = r0 + 1
    interior_left = c0 + 1
    pat_top = interior_top + spec["pat_top"]
    pat_left = interior_left + spec["pat_left"]
    pat_bottom = pat_top + shape_h - 1
    pat_right = pat_left + shape_w - 1

    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL

    if spec["gap_side"] == "up":
        gap = (r0, rng.randint(pat_left, pat_right))
    elif spec["gap_side"] == "down":
        gap = (r1, rng.randint(pat_left, pat_right))
    elif spec["gap_side"] == "left":
        gap = (rng.randint(pat_top, pat_bottom), c0)
    else:
        gap = (rng.randint(pat_top, pat_bottom), c1)
    grid[gap] = BG

    for sr, sc in np.argwhere(shape):
        grid[pat_top + int(sr), pat_left + int(sc)] = spec["pattern_color"]

    if spec["marker_side"] == "up":
        marker_r = interior_top + rng.randint(0, spec["pat_top"] - 2)
        marker_c = rng.randint(pat_left, pat_right)
    elif spec["marker_side"] == "down":
        marker_r = interior_top + rng.randint(spec["pat_top"] + shape_h + 1, spec["interior_h"] - 1)
        marker_c = rng.randint(pat_left, pat_right)
    elif spec["marker_side"] == "left":
        marker_r = rng.randint(pat_top, pat_bottom)
        marker_c = interior_left + rng.randint(0, spec["pat_left"] - 2)
    else:
        marker_r = rng.randint(pat_top, pat_bottom)
        marker_c = interior_left + rng.randint(spec["pat_left"] + shape_w + 1, spec["interior_w"] - 1)

    grid[marker_r, marker_c] = spec["marker_color"]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_SIDE_RANGE)
    grid_w = rng.randint(*GRID_SIDE_RANGE)
    room_count = rng.randint(*ROOM_COUNT_RANGE)

    if 2 * room_count > len(PALETTE):
        return None

    if room_count == 1:
        match_flags = [rng.choice([False, True])]
    else:
        match_count = rng.randint(1, room_count - 1)
        match_flags = [True] * match_count + [False] * (room_count - match_count)
        rng.shuffle(match_flags)

    colors = rng.sample(PALETTE, 2 * room_count)
    specs = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in range(room_count):
        spec = _make_room_spec(rng, colors[2 * idx], colors[2 * idx + 1], match_flags[idx])
        if not _place_room(rng, occupied, spec, grid_h, grid_w):
            return None
        specs.append(spec)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        _render_room(grid, rng, spec)

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
