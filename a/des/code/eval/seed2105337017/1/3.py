from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _second_common_color(grid: np.ndarray, exclude: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    candidates = [(int(count), int(value)) for value, count in zip(values, counts) if int(value) != int(exclude)]
    assert candidates, "expected a non-background wall color"
    return max(candidates)[1]


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _detect_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    wall_mask = g == int(wall)
    prefix = _prefix_sum(wall_mask)
    h, w = map(int, g.shape)
    rooms: list[dict] = []

    for r0 in range(h - 3):
        for r1 in range(r0 + 3, h):
            height = r1 - r0 + 1
            for c0 in range(w - 3):
                for c1 in range(c0 + 3, w):
                    width = c1 - c0 + 1
                    interior_wall = _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if interior_wall != 0:
                        continue

                    border_len = 2 * (height + width) - 4
                    border_wall = _rect_sum(prefix, r0, c0, r1, c1) - interior_wall
                    if border_wall != border_len - 1:
                        continue

                    gap: tuple[int, int] | None = None
                    valid = True
                    for rr, cc in _border_coords(r0, c0, r1, c1):
                        value = int(g[rr, cc])
                        if value == int(wall):
                            continue
                        if value != int(bg) or gap is not None:
                            valid = False
                            break
                        gap = (int(rr), int(cc))

                    if not valid or gap is None:
                        continue

                    gr, gc = gap
                    if (gr, gc) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                        continue

                    if gc == c0:
                        side = "left"
                    elif gc == c1:
                        side = "right"
                    elif gr == r0:
                        side = "top"
                    else:
                        side = "bottom"

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    furniture = np.argwhere((interior != int(bg)) & (interior != int(wall)))
                    rooms.append(
                        {
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "door": (int(gr), int(gc)),
                            "side": side,
                            "furniture": [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in furniture],
                        }
                    )
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find pairs of rectangular rooms connected by a one-pixel door and mirror
    each furnished room into its empty neighbor across the shared wall.

    The wall direction picks the mirror axis:
    - rooms side-by-side -> horizontal flip across the shared vertical wall
    - rooms stacked vertically -> vertical flip across the shared horizontal wall
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    wall = _second_common_color(g, bg)
    rooms = _detect_rooms(g, bg, wall)

    by_door: dict[tuple[int, int], list[dict]] = {}
    for room in rooms:
        by_door.setdefault(room["door"], []).append(room)

    out = g.copy()
    opposite = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }

    for door, pair in by_door.items():
        if len(pair) != 2:
            continue
        first, second = pair
        if opposite[first["side"]] != second["side"]:
            continue

        sources = [room for room in pair if room["furniture"]]
        targets = [room for room in pair if not room["furniture"]]
        if len(sources) != 1 or len(targets) != 1:
            continue

        source = sources[0]
        target = targets[0]
        axis_r, axis_c = map(int, door)
        tr0, tc0, tr1, tc1 = target["bbox"]

        for sr, sc in source["furniture"]:
            color = int(g[sr, sc])
            if source["side"] in {"left", "right"}:
                rr, cc = int(sr), 2 * axis_c - int(sc)
            else:
                rr, cc = 2 * axis_r - int(sr), int(sc)
            if tr0 < rr < tr1 and tc0 < cc < tc1:
                out[rr, cc] = color

    return out


# --------------- generator ---------------

BG = 0
WALL = 1
FURNITURE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
GRID_MIN = 20
GRID_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
ROOM_HEIGHT_RANGE = (5, 8)
ROOM_WIDTH_RANGE = (5, 8)
PLACEMENT_MARGIN = 2
FURNITURE_MIN_CELLS = 2
FURNITURE_MAX_CELLS = 8
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _grow_connected_shape(h: int, w: int, size: int) -> np.ndarray:
    seed = (random.randrange(h), random.randrange(w))
    selected = {seed}
    frontier: set[tuple[int, int]] = set(_neighbors4(seed[0], seed[1], h, w))

    while len(selected) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], h, w):
            if nxt not in selected:
                frontier.add(nxt)

    coords = np.array(sorted(selected), dtype=int)
    return coords


def _draw_room(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = WALL
    grid[top + height - 1, left : left + width] = WALL
    grid[top : top + height, left] = WALL
    grid[top : top + height, left + width - 1] = WALL


def _room_bbox_for_pair(pair: dict, which: str) -> tuple[int, int, int, int]:
    top = int(pair["top"])
    left = int(pair["left"])
    height = int(pair["room_h"])
    width = int(pair["room_w"])

    if pair["orient"] == "lr":
        if which == "left":
            return top, left, top + height - 1, left + width - 1
        return top, left + width - 1, top + height - 1, left + 2 * width - 2

    if which == "top":
        return top, left, top + height - 1, left + width - 1
    return top + height - 1, left, top + 2 * height - 2, left + width - 1


def _pair_dims(orient: str, room_h: int, room_w: int) -> tuple[int, int]:
    if orient == "lr":
        return room_h, 2 * room_w - 1
    return 2 * room_h - 1, room_w


def _place_pairs(grid_h: int, grid_w: int, n_pairs: int) -> list[dict] | None:
    occupancy = np.zeros((grid_h, grid_w), dtype=bool)
    orientations = ["lr", "ud"]
    if n_pairs > 2:
        orientations.extend(random.choice(["lr", "ud"]) for _ in range(n_pairs - 2))
    random.shuffle(orientations)

    pairs: list[dict] = []
    for idx in range(n_pairs):
        orient = orientations[idx]
        placed = False
        for _ in range(200):
            room_h = random.randint(*ROOM_HEIGHT_RANGE)
            room_w = random.randint(*ROOM_WIDTH_RANGE)
            pair_h, pair_w = _pair_dims(orient, room_h, room_w)
            if pair_h + 2 > grid_h or pair_w + 2 > grid_w:
                continue

            top = random.randint(1, grid_h - pair_h - 1)
            left = random.randint(1, grid_w - pair_w - 1)
            r0 = max(0, top - PLACEMENT_MARGIN)
            c0 = max(0, left - PLACEMENT_MARGIN)
            r1 = min(grid_h, top + pair_h + PLACEMENT_MARGIN)
            c1 = min(grid_w, left + pair_w + PLACEMENT_MARGIN)
            if occupancy[r0:r1, c0:c1].any():
                continue

            occupancy[top : top + pair_h, left : left + pair_w] = True
            if orient == "lr":
                side = random.choice(["left", "right"])
                door = (top + random.randint(1, room_h - 2), left + room_w - 1)
            else:
                side = random.choice(["top", "bottom"])
                door = (top + room_h - 1, left + random.randint(1, room_w - 2))
            pairs.append(
                {
                    "orient": orient,
                    "top": top,
                    "left": left,
                    "room_h": room_h,
                    "room_w": room_w,
                    "source_side": side,
                    "door": door,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return pairs


def _paint_furniture(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    color: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    max_cells = min(FURNITURE_MAX_CELLS, inner_h * inner_w - 1)
    size = random.randint(FURNITURE_MIN_CELLS, max_cells)
    shape = _grow_connected_shape(inner_h, inner_w, size)
    coords = [(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in shape]
    for rr, cc in coords:
        grid[rr, cc] = int(color)
    return coords


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        door_r, door_c = spec["door"]
        sr0, sc0, sr1, sc1 = spec["source_bbox"]
        tr0, tc0, tr1, tc1 = spec["target_bbox"]
        _ = (sr0, sc0, sr1, sc1)
        for rr, cc in spec["furniture"]:
            color = int(inp[rr, cc])
            if spec["orient"] == "lr":
                mr, mc = int(rr), 2 * int(door_c) - int(cc)
            else:
                mr, mc = 2 * int(door_r) - int(rr), int(cc)
            assert tr0 < mr < tr1 and tc0 < mc < tc1
            out[mr, mc] = color
    return out


def _bg_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == BG


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    pairs = _place_pairs(grid_h, grid_w, n_pairs)
    if pairs is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = random.sample(FURNITURE_COLORS, n_pairs)
    specs: list[dict] = []

    for pair, color in zip(pairs, colors):
        if pair["orient"] == "lr":
            left_bbox = _room_bbox_for_pair(pair, "left")
            right_bbox = _room_bbox_for_pair(pair, "right")
            _draw_room(grid, left_bbox[0], left_bbox[1], pair["room_h"], pair["room_w"])
            _draw_room(grid, right_bbox[0], right_bbox[1], pair["room_h"], pair["room_w"])
            source_bbox = left_bbox if pair["source_side"] == "left" else right_bbox
            target_bbox = right_bbox if pair["source_side"] == "left" else left_bbox
        else:
            top_bbox = _room_bbox_for_pair(pair, "top")
            bottom_bbox = _room_bbox_for_pair(pair, "bottom")
            _draw_room(grid, top_bbox[0], top_bbox[1], pair["room_h"], pair["room_w"])
            _draw_room(grid, bottom_bbox[0], bottom_bbox[1], pair["room_h"], pair["room_w"])
            source_bbox = top_bbox if pair["source_side"] == "top" else bottom_bbox
            target_bbox = bottom_bbox if pair["source_side"] == "top" else top_bbox

        door_r, door_c = pair["door"]
        grid[door_r, door_c] = BG
        furniture = _paint_furniture(grid, source_bbox, int(color))
        specs.append(
            {
                "orient": pair["orient"],
                "door": pair["door"],
                "source_bbox": source_bbox,
                "target_bbox": target_bbox,
                "furniture": furniture,
            }
        )

    if not _bg_is_dominant(grid):
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-mirroring input")
