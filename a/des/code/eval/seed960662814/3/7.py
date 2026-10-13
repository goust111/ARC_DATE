from __future__ import annotations

import numpy as np


BG = 0
WALL = 5
ARROW = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PALETTE = [1, 2, 3, 4, 6, 7, 8]
ROOM_COUNT_CHOICES = [1, 2, 3]
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
ROOM_OUTER_H_RANGE = (7, 11)
ROOM_OUTER_W_RANGE = (7, 11)
MAX_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int, *, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        dirs = DIR4
    else:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(grid, WALL, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        perimeter = 2 * (height + width) - 4
        if len(comp) != perimeter:
            continue
        border_ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != (grid[r, c] == WALL):
                    border_ok = False
                    break
            if not border_ok:
                break
        if border_ok:
            rooms.append((r0, c0, r1, c1))
    rooms.sort()
    return rooms


def _arrow_cells_for(room: tuple[int, int, int, int], direction: str, anchor: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = room
    if direction == "right":
        return {(anchor - 1, c1 + 1), (anchor + 1, c1 + 1), (anchor, c1 + 2)}
    if direction == "left":
        return {(anchor - 1, c0 - 1), (anchor + 1, c0 - 1), (anchor, c0 - 2)}
    if direction == "down":
        return {(r1 + 1, anchor - 1), (r1 + 1, anchor + 1), (r1 + 2, anchor)}
    if direction == "up":
        return {(r0 - 1, anchor - 1), (r0 - 1, anchor + 1), (r0 - 2, anchor)}
    raise AssertionError("unknown direction")


def _find_arrow_direction(grid: np.ndarray, room: tuple[int, int, int, int]) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = room
    nearby = {
        (r, c)
        for r in range(max(0, r0 - 2), min(h, r1 + 3))
        for c in range(max(0, c0 - 2), min(w, c1 + 3))
        if not (r0 <= r <= r1 and c0 <= c <= c1) and grid[r, c] == ARROW
    }

    candidates: list[tuple[str, int]] = []
    for anchor in range(r0 + 1, r1):
        if _arrow_cells_for(room, "right", anchor) == nearby:
            candidates.append(("right", anchor))
        if _arrow_cells_for(room, "left", anchor) == nearby:
            candidates.append(("left", anchor))
    for anchor in range(c0 + 1, c1):
        if _arrow_cells_for(room, "down", anchor) == nearby:
            candidates.append(("down", anchor))
        if _arrow_cells_for(room, "up", anchor) == nearby:
            candidates.append(("up", anchor))

    assert len(candidates) == 1, "expected one unique arrow per room"
    direction = candidates[0][0]
    if direction == "right":
        return 0, 1
    if direction == "left":
        return 0, -1
    if direction == "down":
        return 1, 0
    return -1, 0


def _room_contents(
    grid: np.ndarray, room: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int]], int, str, int, int]:
    r0, c0, r1, c1 = room
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape

    stripe_rows = [(rr, int(interior[rr, 0])) for rr in range(ih) if interior[rr, 0] != BG and np.all(interior[rr] == interior[rr, 0])]
    stripe_cols = [(cc, int(interior[0, cc])) for cc in range(iw) if interior[0, cc] != BG and np.all(interior[:, cc] == interior[0, cc])]
    assert len(stripe_rows) + len(stripe_cols) == 1, "expected one stripe in room interior"

    if stripe_rows:
        stripe_orient = "h"
        stripe_idx, stripe_color = stripe_rows[0]
    else:
        stripe_orient = "v"
        stripe_idx, stripe_color = stripe_cols[0]

    sprite_coords: list[tuple[int, int]] = []
    sprite_colors = set()
    for rr in range(ih):
        for cc in range(iw):
            value = int(interior[rr, cc])
            if value in (BG, stripe_color):
                continue
            sprite_coords.append((r0 + 1 + rr, c0 + 1 + cc))
            sprite_colors.add(value)

    assert sprite_coords and len(sprite_colors) == 1, "expected one monochrome sprite in each room"
    sprite_color = sprite_colors.pop()
    return sprite_coords, sprite_color, stripe_orient, int(stripe_idx), int(stripe_color)


def _projection_shift(
    sprite_coords: list[tuple[int, int]], room: tuple[int, int, int, int], direction: tuple[int, int]
) -> int:
    r0, c0, r1, c1 = room
    dr, dc = direction
    rows = [r for r, _ in sprite_coords]
    cols = [c for _, c in sprite_coords]
    if (dr, dc) == (0, 1):
        return (c1 - 1) - max(cols)
    if (dr, dc) == (0, -1):
        return min(cols) - (c0 + 1)
    if (dr, dc) == (1, 0):
        return (r1 - 1) - max(rows)
    return min(rows) - (r0 + 1)


def _crosses_stripe(
    sprite_coords: list[tuple[int, int]],
    shift: int,
    direction: tuple[int, int],
    stripe_orient: str,
    stripe_idx: int,
    room: tuple[int, int, int, int],
) -> bool:
    if shift <= 0:
        return False

    r0, c0, _r1, _c1 = room
    dr, dc = direction
    if (dr, dc) in ((0, 1), (0, -1)):
        if stripe_orient != "v":
            return False
        stripe_col = c0 + 1 + stripe_idx
        if dc == 1:
            return any(c < stripe_col <= c + shift for _r, c in sprite_coords)
        return any(c - shift <= stripe_col < c for _r, c in sprite_coords)

    if stripe_orient != "h":
        return False
    stripe_row = r0 + 1 + stripe_idx
    if dr == 1:
        return any(r < stripe_row <= r + shift for r, _c in sprite_coords)
    return any(r - shift <= stripe_row < r for r, _c in sprite_coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's sprite toward the opposite wall.

    The direction comes from the maroon arrow outside the room. If the moving sprite crosses the
    interior stripe, the projected copy uses the stripe color; otherwise it keeps the sprite color.
    The original sprite remains in place.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        direction = _find_arrow_direction(g, room)
        sprite_coords, sprite_color, stripe_orient, stripe_idx, stripe_color = _room_contents(g, room)
        shift = _projection_shift(sprite_coords, room, direction)
        use_stripe_color = _crosses_stripe(sprite_coords, shift, direction, stripe_orient, stripe_idx, room)
        proj_color = stripe_color if use_stripe_color else sprite_color
        dr, dc = direction
        for r, c in sprite_coords:
            out[r + dr * shift, c + dc * shift] = proj_color

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _shape_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    area = int(rng.integers(2, 5))
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        frontier_list = list(frontier)
        cells.add(frontier_list[int(rng.integers(0, len(frontier_list)))])
    return _shape_mask(_normalize(cells))


def _projection_gap_ok(mask: np.ndarray, shift: int, direction: str) -> bool:
    if direction in ("left", "right"):
        return shift >= mask.shape[1]
    return shift >= mask.shape[0]


def _sample_parallel_row(
    interior_h: int, shape_h: int, stripe_row: int, rng: np.random.Generator
) -> int | None:
    candidates = [sr for sr in range(interior_h - shape_h + 1) if not (sr <= stripe_row < sr + shape_h)]
    if not candidates:
        return None
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _sample_parallel_col(
    interior_w: int, shape_w: int, stripe_col: int, rng: np.random.Generator
) -> int | None:
    candidates = [sc for sc in range(interior_w - shape_w + 1) if not (sc <= stripe_col < sc + shape_w)]
    if not candidates:
        return None
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _choose_room_pattern(
    interior_h: int, interior_w: int, direction: str, cross_flag: bool, rng: np.random.Generator
) -> tuple[np.ndarray, int, int, str, int]:
    for _ in range(300):
        mask = _random_shape(rng)
        shape_h, shape_w = mask.shape

        if direction in ("left", "right"):
            if interior_w < 2 * shape_w or interior_h < shape_h + 1:
                continue
            if cross_flag:
                stripe_orient = "v"
            else:
                stripe_orient = "h" if int(rng.integers(0, 2)) == 0 else "v"

            if stripe_orient == "h":
                stripe_idx = int(rng.integers(0, interior_h))
                sr = _sample_parallel_row(interior_h, shape_h, stripe_idx, rng)
                if sr is None:
                    continue
                valid_sc = []
                for sc in range(interior_w - shape_w + 1):
                    shift = (interior_w - (sc + shape_w)) if direction == "right" else sc
                    if _projection_gap_ok(mask, shift, direction):
                        valid_sc.append(sc)
                if not valid_sc:
                    continue
                sc = int(valid_sc[int(rng.integers(0, len(valid_sc)))])
            else:
                sr = int(rng.integers(0, interior_h - shape_h + 1))
                valid_pairs: list[tuple[int, int]] = []
                for sc in range(interior_w - shape_w + 1):
                    shift = (interior_w - (sc + shape_w)) if direction == "right" else sc
                    if not _projection_gap_ok(mask, shift, direction):
                        continue
                    if direction == "right":
                        stripe_candidates = range(sc + shape_w, interior_w) if cross_flag else range(0, sc)
                    else:
                        stripe_candidates = range(0, sc) if cross_flag else range(sc + shape_w, interior_w)
                    for stripe_idx in stripe_candidates:
                        valid_pairs.append((sc, stripe_idx))
                if not valid_pairs:
                    continue
                sc, stripe_idx = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
            return mask, int(sr), int(sc), stripe_orient, int(stripe_idx)

        if interior_h < 2 * shape_h or interior_w < shape_w + 1:
            continue
        if cross_flag:
            stripe_orient = "h"
        else:
            stripe_orient = "v" if int(rng.integers(0, 2)) == 0 else "h"

        if stripe_orient == "v":
            stripe_idx = int(rng.integers(0, interior_w))
            sc = _sample_parallel_col(interior_w, shape_w, stripe_idx, rng)
            if sc is None:
                continue
            valid_sr = []
            for sr in range(interior_h - shape_h + 1):
                shift = (interior_h - (sr + shape_h)) if direction == "down" else sr
                if _projection_gap_ok(mask, shift, direction):
                    valid_sr.append(sr)
            if not valid_sr:
                continue
            sr = int(valid_sr[int(rng.integers(0, len(valid_sr)))])
        else:
            sc = int(rng.integers(0, interior_w - shape_w + 1))
            valid_pairs = []
            for sr in range(interior_h - shape_h + 1):
                shift = (interior_h - (sr + shape_h)) if direction == "down" else sr
                if not _projection_gap_ok(mask, shift, direction):
                    continue
                if direction == "down":
                    stripe_candidates = range(sr + shape_h, interior_h) if cross_flag else range(0, sr)
                else:
                    stripe_candidates = range(0, sr) if cross_flag else range(sr + shape_h, interior_h)
                for stripe_idx in stripe_candidates:
                    valid_pairs.append((sr, stripe_idx))
            if not valid_pairs:
                continue
            sr, stripe_idx = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
        return mask, int(sr), int(sc), stripe_orient, int(stripe_idx)

    raise RuntimeError("failed to choose a valid room pattern")


def _manual_crosses(
    sprite_cells: list[tuple[int, int]],
    room: tuple[int, int, int, int],
    direction: str,
    stripe_orient: str,
    stripe_idx: int,
) -> bool:
    vec = {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}[direction]
    shift = _projection_shift(sprite_cells, room, vec)
    return _crosses_stripe(sprite_cells, shift, vec, stripe_orient, stripe_idx, room)


def _render_pair(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((spec["height"], spec["width"]), BG, dtype=int)
    expected = grid.copy()

    for room in spec["rooms"]:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        expected[r0 : r1 + 1, c0] = WALL
        expected[r0 : r1 + 1, c1] = WALL
        expected[r0, c0 : c1 + 1] = WALL
        expected[r1, c0 : c1 + 1] = WALL

        if room["stripe_orient"] == "h":
            rr = r0 + 1 + room["stripe_idx"]
            grid[rr, c0 + 1 : c1] = room["stripe_color"]
            expected[rr, c0 + 1 : c1] = room["stripe_color"]
        else:
            cc = c0 + 1 + room["stripe_idx"]
            grid[r0 + 1 : r1, cc] = room["stripe_color"]
            expected[r0 + 1 : r1, cc] = room["stripe_color"]

        for rr, cc in room["sprite_cells"]:
            grid[rr, cc] = room["sprite_color"]
            expected[rr, cc] = room["sprite_color"]
        for rr, cc in room["arrow_cells"]:
            grid[rr, cc] = ARROW
            expected[rr, cc] = ARROW

    for room in spec["rooms"]:
        vec = {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}[room["direction"]]
        shift = _projection_shift(room["sprite_cells"], room["bbox"], vec)
        proj_color = room["stripe_color"] if _manual_crosses(
            room["sprite_cells"], room["bbox"], room["direction"], room["stripe_orient"], room["stripe_idx"]
        ) else room["sprite_color"]
        dr, dc = vec
        for rr, cc in room["sprite_cells"]:
            expected[rr + dr * shift, cc + dc * shift] = proj_color

    return grid, expected


def _sample_room_specs(height: int, width: int, room_count: int, rng: np.random.Generator) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    rooms: list[dict] = []
    directions = ["right", "left", "down", "up"]
    cross_flags = [bool(rng.integers(0, 2)) for _ in range(room_count)]

    if room_count >= 2 and len(set(cross_flags)) == 1:
        cross_flags[0] = not cross_flags[0]

    color_pool = list(rng.permutation(PALETTE))
    while len(color_pool) < 2 * room_count:
        color_pool.extend(list(rng.permutation(PALETTE)))

    for i in range(room_count):
        placed = False
        for _ in range(300):
            outer_h = int(rng.integers(*ROOM_OUTER_H_RANGE))
            outer_w = int(rng.integers(*ROOM_OUTER_W_RANGE))
            direction = directions[int(rng.integers(0, len(directions)))]
            if height - outer_h - 4 <= 0 or width - outer_w - 4 <= 0:
                continue
            r0 = int(rng.integers(2, height - outer_h - 1))
            c0 = int(rng.integers(2, width - outer_w - 1))
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            reserve = occupied[r0 - 2 : r1 + 3, c0 - 2 : c1 + 3]
            if reserve.any():
                continue

            try:
                mask, sr, sc, stripe_orient, stripe_idx = _choose_room_pattern(
                    outer_h - 2, outer_w - 2, direction, cross_flags[i], rng
                )
            except RuntimeError:
                continue

            sprite_cells = [
                (r0 + 1 + sr + rr, c0 + 1 + sc + cc)
                for rr in range(mask.shape[0])
                for cc in range(mask.shape[1])
                if mask[rr, cc]
            ]
            sprite_color = int(color_pool[2 * i])
            stripe_color = int(color_pool[2 * i + 1])
            if stripe_color == sprite_color:
                continue

            if direction in ("right", "left"):
                anchor = int(rng.integers(r0 + 1, r1))
            else:
                anchor = int(rng.integers(c0 + 1, c1))
            arrow_cells = _arrow_cells_for((r0, c0, r1, c1), direction, anchor)

            cells_ok = all(0 <= rr < height and 0 <= cc < width for rr, cc in arrow_cells)
            if not cells_ok:
                continue

            room = {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "stripe_orient": stripe_orient,
                "stripe_idx": stripe_idx,
                "stripe_color": stripe_color,
                "sprite_color": sprite_color,
                "sprite_cells": sprite_cells,
                "arrow_cells": sorted(arrow_cells),
            }

            test_grid, _ = _render_pair({"height": height, "width": width, "rooms": [room]})
            try:
                detected = _find_arrow_direction(test_grid, room["bbox"])
                parsed = _room_contents(test_grid, room["bbox"])
            except Exception:
                continue
            if detected != {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}[direction]:
                continue
            if parsed[1] != sprite_color or parsed[2] != stripe_orient or parsed[3] != stripe_idx or parsed[4] != stripe_color:
                continue
            if _manual_crosses(sprite_cells, room["bbox"], direction, stripe_orient, stripe_idx) != cross_flags[i]:
                continue

            occupied[r0 - 2 : r1 + 3, c0 - 2 : c1 + 3] = True
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None
    return rooms


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    room_count = int(ROOM_COUNT_CHOICES[int(rng.integers(0, len(ROOM_COUNT_CHOICES)))])

    rooms = _sample_room_specs(height, width, room_count, rng)
    if rooms is None:
        return None

    inp, expected = _render_pair({"height": height, "width": width, "rooms": rooms})
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _generate_case(np.random.default_rng())
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input")
