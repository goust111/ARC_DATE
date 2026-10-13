import numpy as np
from collections import Counter, deque


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAIR_RETRIES = 120
ROOM_SHAPE_RETRIES = 80
GRID_RETRIES = 400
CANVAS_SIZE = 30


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r1, c) for c in range(c0, c1 + 1)]
    cells += [(r, c0) for r in range(r0 + 1, r1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    return cells


def _make_room(h: int, w: int, side: str, anchor: int) -> dict:
    bbox = (0, 0, h + 1, w + 1)
    if side == "top":
        portal = (0, anchor)
    elif side == "bottom":
        portal = (h + 1, anchor)
    elif side == "left":
        portal = (anchor, 0)
    else:
        portal = (anchor, w + 1)
    return {
        "bbox": bbox,
        "side": side,
        "portal_pos": portal,
        "inner_h": h,
        "inner_w": w,
    }


def _to_local(room: dict, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    r0, c0, r1, c1 = room["bbox"]
    pr, pc = room["portal_pos"]
    side = room["side"]
    if side == "top":
        return r - r0, c - pc
    if side == "bottom":
        return r1 - r, c - pc
    if side == "left":
        return c - c0, r - pr
    return c1 - c, r - pr


def _from_local(room: dict, depth: int, along: int) -> tuple[int, int]:
    r0, c0, r1, c1 = room["bbox"]
    pr, pc = room["portal_pos"]
    side = room["side"]
    if side == "top":
        return r0 + depth, pc + along
    if side == "bottom":
        return r1 - depth, pc + along
    if side == "left":
        return pr + along, c0 + depth
    return pr + along, c1 - depth


def _in_interior(room: dict, cell: tuple[int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = room["bbox"]
    return r0 < r < r1 and c0 < c < c1


def _project_cells(src_room: dict, dst_room: dict, cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    projected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        depth, along = _to_local(src_room, cell)
        if depth <= 0:
            return None
        mapped = _from_local(dst_room, depth, along)
        if not _in_interior(dst_room, mapped) or mapped in seen:
            return None
        seen.add(mapped)
        projected.append(mapped)
    return projected


def _random_connected_shape(
    rng: np.random.Generator,
    h: int,
    w: int,
    size: int,
) -> list[tuple[int, int]] | None:
    start = (int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1)))
    cells = {start}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 1 <= nr <= h and 1 <= nc <= w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)
    return sorted(cells)


def _build_pair(
    rng: np.random.Generator,
    portal_color: int,
    object_colors: tuple[int, int],
) -> list[dict] | None:
    h = int(rng.integers(4, 7))
    w = int(rng.integers(4, 7))
    axis = int(rng.integers(2))
    if axis == 0:
        side_a, side_b = ("top", "bottom") if rng.random() < 0.5 else ("bottom", "top")
        anchor_a = int(rng.integers(1, w + 1))
        anchor_b = int(rng.integers(1, w + 1))
    else:
        side_a, side_b = ("left", "right") if rng.random() < 0.5 else ("right", "left")
        anchor_a = int(rng.integers(1, h + 1))
        anchor_b = int(rng.integers(1, h + 1))

    room_a = _make_room(h, w, side_a, anchor_a)
    room_b = _make_room(h, w, side_b, anchor_b)

    for _ in range(PAIR_RETRIES):
        size_a = int(rng.integers(2, 5))
        cells_a = _random_connected_shape(rng, h, w, size_a)
        if cells_a is None:
            continue
        proj_a = _project_cells(room_a, room_b, cells_a)
        if proj_a is None:
            continue

        for _ in range(ROOM_SHAPE_RETRIES):
            size_b = int(rng.integers(2, 5))
            cells_b = _random_connected_shape(rng, h, w, size_b)
            if cells_b is None:
                continue
            proj_b = _project_cells(room_b, room_a, cells_b)
            if proj_b is None:
                continue
            if set(cells_a) & set(proj_b):
                continue
            if set(cells_b) & set(proj_a):
                continue
            room_a_spec = {
                **room_a,
                "outer_h": h + 2,
                "outer_w": w + 2,
                "portal_color": portal_color,
                "object_color": object_colors[0],
                "object_cells": cells_a,
            }
            room_b_spec = {
                **room_b,
                "outer_h": h + 2,
                "outer_w": w + 2,
                "portal_color": portal_color,
                "object_color": object_colors[1],
                "object_cells": cells_b,
            }
            return [room_a_spec, room_b_spec]
    return None


def _place_rooms(rng: np.random.Generator, rooms: list[dict]) -> list[dict] | None:
    blocked = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    placed: list[dict | None] = [None] * len(rooms)
    order = rng.permutation(len(rooms))
    for idx in order:
        room = rooms[int(idx)]
        oh, ow = room["outer_h"], room["outer_w"]
        placed_here = False
        for _ in range(200):
            r0 = int(rng.integers(0, CANVAS_SIZE - oh + 1))
            c0 = int(rng.integers(0, CANVAS_SIZE - ow + 1))
            if blocked[r0 : r0 + oh, c0 : c0 + ow].any():
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(CANVAS_SIZE, r0 + oh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(CANVAS_SIZE, c0 + ow + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            shifted = room.copy()
            shifted["bbox"] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            pr, pc = room["portal_pos"]
            shifted["portal_pos"] = (r0 + pr, c0 + pc)
            shifted["object_cells"] = [(r0 + r, c0 + c) for r, c in room["object_cells"]]
            placed[int(idx)] = shifted
            placed_here = True
            break
        if not placed_here:
            return None
    return [room for room in placed if room is not None]


def _draw_rooms(rooms: list[dict]) -> np.ndarray:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        pr, pc = room["portal_pos"]
        grid[pr, pc] = room["portal_color"]
        for rr, cc in room["object_cells"]:
            grid[rr, cc] = room["object_color"]
    return grid


def _crop_with_margin(rng: np.random.Generator, grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = int(rng.integers(0, 3))
    bottom = int(rng.integers(0, 3))
    left = int(rng.integers(0, 3))
    right = int(rng.integers(0, 3))
    rr0 = max(0, int(r0) - top)
    rr1 = min(grid.shape[0] - 1, int(r1) + bottom)
    cc0 = max(0, int(c0) - left)
    cc1 = min(grid.shape[1] - 1, int(c1) + right)
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _parse_rooms(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _mode(grid.reshape(-1))
    non_bg = grid[grid != bg]
    assert non_bg.size > 0, "expected at least one room"
    wall = _mode(non_bg)

    rooms: list[dict] = []
    for comp in _components(grid == wall):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        portal_candidates = []
        for r, c in _perimeter_cells(bbox):
            val = int(grid[r, c])
            if val not in (bg, wall):
                portal_candidates.append((r, c, val))
        assert len(portal_candidates) == 1, "each room must have exactly one portal"
        pr, pc, portal_color = portal_candidates[0]
        assert (pr in (r0, r1)) ^ (pc in (c0, c1)), "portal must be on a single wall"
        if pr == r0:
            side = "top"
        elif pr == r1:
            side = "bottom"
        elif pc == c0:
            side = "left"
        else:
            side = "right"

        inner_mask = np.zeros_like(grid, dtype=bool)
        inner_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        obj_mask = inner_mask & (grid != bg) & (grid != wall)
        comps = _components(obj_mask)
        assert len(comps) == 1, "each room must contain exactly one connected object"
        obj_coords = comps[0]
        obj_colors = {int(grid[r, c]) for r, c in obj_coords}
        assert len(obj_colors) == 1, "room object must be monochrome"
        rooms.append(
            {
                "bbox": bbox,
                "side": side,
                "portal_pos": (pr, pc),
                "portal_color": portal_color,
                "object_color": obj_colors.pop(),
                "object_cells": obj_coords,
            }
        )
    assert rooms, "failed to identify any rooms"
    return bg, wall, rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match same-colored wall portals across rooms and project each room's object into its paired room.

    Input structure:
    - The background is a dominant color, and each room is enclosed by a one-pixel-thick grey rectangular wall.
    - Every room contains exactly one small connected monochrome object fully inside the walls.
    - Each room also has exactly one non-grey, non-background portal pixel placed on one wall cell, replacing that wall color.
    - Portal colors occur in pairs, so exactly two different rooms share each portal color.

    Rule:
    - Identify each rectangular room, its unique wall portal, and its unique interior object.
    - Pair rooms whose portal pixels have the same color.
    - For each ordered pair of paired rooms, convert the source object's cells into wall-relative local coordinates measured from the source portal, then place the same local pattern just inside the target portal's wall.
    - Keep every original object and every portal pixel unchanged; only add the projected copy in the paired room using the source object's color.

    Key insight:
    - The portal color determines the long-range room pairing, while the placement itself is solved in local room coordinates: depth from the portal wall and offset along that wall are preserved when the object is projected into the partner room.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall, rooms = _parse_rooms(g)

    by_portal: dict[int, list[dict]] = {}
    for room in rooms:
        by_portal.setdefault(int(room["portal_color"]), []).append(room)

    out = g.copy()
    for portal_color, pair in by_portal.items():
        assert len(pair) == 2, f"portal color {portal_color} must appear in exactly two rooms"
        a, b = pair
        for src, dst in ((a, b), (b, a)):
            projected = _project_cells(src, dst, src["object_cells"])
            assert projected is not None, "projected object must fit inside the paired room"
            for rr, cc in projected:
                assert g[rr, cc] == bg, "projected object should land on clear background"
                out[rr, cc] = int(src["object_color"])
    _ = wall
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 4))
    portal_colors = rng.choice(COLOR_POOL, size=pair_count, replace=False).tolist()

    rooms: list[dict] = []
    for portal_color in portal_colors:
        candidates = [c for c in COLOR_POOL if c != portal_color]
        object_colors = tuple(int(x) for x in rng.choice(candidates, size=2, replace=False))
        pair_rooms = _build_pair(rng, int(portal_color), object_colors)
        if pair_rooms is None:
            return None
        rooms.extend(pair_rooms)

    placed = _place_rooms(rng, rooms)
    if placed is None:
        return None

    grid = _draw_rooms(placed)
    grid = _crop_with_margin(rng, grid)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-room grid")
