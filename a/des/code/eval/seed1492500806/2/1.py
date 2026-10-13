import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
AZURE = 8

PRIMARY_COLORS = (RED, GREEN, BLUE)
PRIMARY_TO_BIT = {
    RED: 1,
    GREEN: 2,
    BLUE: 4,
}
BIT_TO_COLOR = {
    1: RED,
    2: GREEN,
    4: BLUE,
    3: YELLOW,   # red + green
    5: MAGENTA,  # red + blue
    6: AZURE,    # green + blue
}
DUAL_COLOR_PAIRS = (
    (RED, BLUE),
    (RED, GREEN),
    (BLUE, GREEN),
)
DUAL_ALIGNMENTS = ("cross", "same_row", "same_col")

GRID_H_RANGE = (12, 21)
GRID_W_RANGE = (12, 21)
INTERIOR_H_RANGE = (2, 8)
INTERIOR_W_RANGE = (2, 8)
ROOM_PADDING = 1
OUTER_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400
ROOM_PLACEMENT_ATTEMPTS = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(coords)
    return comps


def _rectangle_perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0 + 1, r1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            perimeter = _rectangle_perimeter(r0, c0, r1, c1)
            if len(comp) != len(perimeter):
                continue
            if set(comp) != perimeter:
                continue
            rooms.append({"bbox": (r0, c0, r1, c1), "wall_color": color})
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _room_seeds(grid: np.ndarray, room: dict[str, object], bg: int) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    wall_color = int(room["wall_color"])
    seeds: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == bg:
                continue
            assert value != wall_color, "room walls must be one cell thick"
            assert value in PRIMARY_TO_BIT, "seeds must be red, green, or blue"
            seeds.append((r, c, value))
    assert len(seeds) <= 2, "each room may contain at most two seeds"
    return seeds


def _paint_room(out: np.ndarray, room: dict[str, object], seeds: list[tuple[int, int, int]]) -> None:
    r0, c0, r1, c1 = room["bbox"]
    bits = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=np.uint8)
    seed_lookup = {(r, c): color for r, c, color in seeds}

    for sr, sc, color in seeds:
        bit = PRIMARY_TO_BIT[color]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = sr + dr, sc + dc
            while r0 < nr < r1 and c0 < nc < c1:
                bits[nr - (r0 + 1), nc - (c0 + 1)] |= bit
                nr += dr
                nc += dc

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in seed_lookup:
                out[r, c] = seed_lookup[(r, c)]
                continue
            mask = int(bits[r - (r0 + 1), c - (c0 + 1)])
            if mask:
                assert mask in BIT_TO_COLOR, "unexpected color overlap"
                out[r, c] = BIT_TO_COLOR[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    rooms = _find_rooms(g, bg)
    assert rooms, "expected at least one closed rectangular room"

    out = g.copy()
    covered = np.zeros_like(g, dtype=bool)

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        for r, c in _rectangle_perimeter(r0, c0, r1, c1):
            covered[r, c] = True
        seeds = _room_seeds(g, room, bg)
        for r, c, _ in seeds:
            covered[r, c] = True
        if seeds:
            _paint_room(out, room, seeds)

    assert np.all((g == bg) | covered), "all non-background pixels must be room walls or seeds"
    return out


def _balanced_choices(options, count: int, rng: np.random.Generator):
    options = list(options)
    chosen = []
    while len(chosen) < count:
        order = list(rng.permutation(len(options)))
        chosen.extend(options[i] for i in order)
    return chosen[:count]


def _sample_room_specs(room_count: int, scenario: str, rng: np.random.Generator) -> list[dict[str, object]]:
    if scenario == "single_only":
        modes = ["single"] * room_count
        if room_count > 1 and rng.random() < 0.35:
            modes[int(rng.integers(room_count))] = "empty"
    elif scenario == "mixed":
        modes = ["single", "dual"]
        while len(modes) < room_count:
            pick = rng.choice(["single", "dual", "empty"], p=[0.4, 0.4, 0.2])
            modes.append(str(pick))
    else:
        modes = ["dual"] * room_count

    rng.shuffle(modes)

    single_colors = _balanced_choices(PRIMARY_COLORS, modes.count("single"), rng)
    dual_pairs = _balanced_choices(DUAL_COLOR_PAIRS, modes.count("dual"), rng)
    dual_alignments = _balanced_choices(DUAL_ALIGNMENTS, modes.count("dual"), rng)
    single_idx = 0
    dual_idx = 0
    specs: list[dict[str, object]] = []
    for mode in modes:
        spec: dict[str, object] = {"mode": mode}
        if mode == "single":
            spec["color"] = int(single_colors[single_idx])
            single_idx += 1
        elif mode == "dual":
            spec["pair"] = tuple(int(v) for v in dual_pairs[dual_idx])
            spec["alignment"] = str(dual_alignments[dual_idx])
            dual_idx += 1

        min_h = INTERIOR_H_RANGE[0]
        min_w = INTERIOR_W_RANGE[0]
        if mode == "dual" and spec["alignment"] == "same_col":
            min_h = max(min_h, 3)
        if mode == "dual" and spec["alignment"] == "same_row":
            min_w = max(min_w, 3)
        spec["interior_h"] = int(rng.integers(min_h, INTERIOR_H_RANGE[1]))
        spec["interior_w"] = int(rng.integers(min_w, INTERIOR_W_RANGE[1]))
        specs.append(spec)

    return specs


def _place_rooms(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    occupancy = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["interior_h"]) * int(specs[i]["interior_w"]),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        room_h = int(spec["interior_h"]) + 2
        room_w = int(spec["interior_w"]) + 2
        max_r0 = h - OUTER_MARGIN - room_h
        max_c0 = w - OUTER_MARGIN - room_w
        if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
            return None

        placed_room = None
        for _ in range(ROOM_PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(OUTER_MARGIN, max_r0 + 1))
            c0 = int(rng.integers(OUTER_MARGIN, max_c0 + 1))
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            pr0 = max(0, r0 - ROOM_PADDING)
            pc0 = max(0, c0 - ROOM_PADDING)
            pr1 = min(h - 1, r1 + ROOM_PADDING)
            pc1 = min(w - 1, c1 + ROOM_PADDING)
            if np.any(occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1]):
                continue
            occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            placed_room = dict(spec)
            placed_room["bbox"] = (r0, c0, r1, c1)
            break

        if placed_room is None:
            return None
        placed[idx] = placed_room

    return [room for room in placed if room is not None]


def _single_seed(room: dict[str, object], rng: np.random.Generator) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    r = int(rng.integers(r0 + 1, r1))
    c = int(rng.integers(c0 + 1, c1))
    return [(r, c, int(room["color"]))]


def _dual_seeds(room: dict[str, object], rng: np.random.Generator) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    rows = np.arange(r0 + 1, r1)
    cols = np.arange(c0 + 1, c1)
    color_a, color_b = room["pair"]
    if rng.random() < 0.5:
        color_a, color_b = color_b, color_a

    alignment = room["alignment"]
    if alignment == "cross":
        picked_rows = sorted(int(v) for v in rng.choice(rows, size=2, replace=False))
        picked_cols = sorted(int(v) for v in rng.choice(cols, size=2, replace=False))
        if rng.random() < 0.5:
            pos_a = (picked_rows[0], picked_cols[0])
            pos_b = (picked_rows[1], picked_cols[1])
        else:
            pos_a = (picked_rows[0], picked_cols[1])
            pos_b = (picked_rows[1], picked_cols[0])
    elif alignment == "same_row":
        row = int(rng.choice(rows))
        picked_cols = sorted(int(v) for v in rng.choice(cols, size=2, replace=False))
        pos_a = (row, picked_cols[0])
        pos_b = (row, picked_cols[1])
    else:
        col = int(rng.choice(cols))
        picked_rows = sorted(int(v) for v in rng.choice(rows, size=2, replace=False))
        pos_a = (picked_rows[0], col)
        pos_b = (picked_rows[1], col)

    return [(pos_a[0], pos_a[1], int(color_a)), (pos_b[0], pos_b[1], int(color_b))]


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int], wall_color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    scenario = str(rng.choice(["single_only", "mixed", "dual_only"]))

    area = h * w
    if area < 180:
        max_rooms = 2
    elif area < 280:
        max_rooms = 3
    else:
        max_rooms = 4
    min_rooms = 2 if scenario != "dual_only" else 1
    if max_rooms < min_rooms:
        return None

    room_count = int(rng.integers(min_rooms, max_rooms + 1))
    specs = _sample_room_specs(room_count, scenario, rng)
    rooms = _place_rooms(h, w, specs, rng)
    if rooms is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for room in rooms:
        _draw_room(grid, room["bbox"], GRAY)

    for room in rooms:
        mode = room["mode"]
        if mode == "single":
            seeds = _single_seed(room, rng)
        elif mode == "dual":
            seeds = _dual_seeds(room, rng)
        else:
            seeds = []
        for r, c, color in seeds:
            grid[r, c] = color

    found_rooms = _find_rooms(grid, BLACK)
    if len(found_rooms) != len(rooms):
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None

    if any(room["mode"] == "dual" for room in rooms):
        if not np.any(np.isin(out, [YELLOW, MAGENTA, AZURE])):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-ray puzzle input")
