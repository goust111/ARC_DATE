from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _label_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    edge_touch: list[bool] = []
    label = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1 or int(grid[r, c]) != BLACK:
                continue
            q = deque([(r, c)])
            labels[r, c] = label
            touches = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if labels[nr, nc] != -1 or int(grid[nr, nc]) != BLACK:
                        continue
                    labels[nr, nc] = label
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches = True
                    q.append((nr, nc))
            edge_touch.append(touches)
            label += 1
    return labels, edge_touch


def _classify_red_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    black_labels, black_edge = _label_black_regions(grid)
    comps = []
    for coords in _components_of_color(grid, RED):
        adjacent_black = set()
        touches_edge = False
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr in (0, h - 1) or cc in (0, w - 1):
                touches_edge = True
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK:
                    adjacent_black.add(int(black_labels[nr, nc]))
        closed = bool(adjacent_black) and (not touches_edge) and all(not black_edge[label] for label in adjacent_black)
        comps.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "closed": closed,
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor red components inside a grey maze by combining enclosure and area rank.

    Input structure:
    - The grid uses black background, grey wall pixels, and red 4-connected foreground components.
    - Grey walls carve the black space into chambers, some fully sealed and some connected to the outer border.
    - There are multiple red components with varying areas placed in those chambers or in the open black space.

    Rule:
    - Identify every red 4-connected component and test whether its adjacent black region can reach any grid-edge black pixel through 4-connected black paths.
    - Components whose adjacent black region cannot reach the edge are the closed class; all others stay red.
    - Compute the lower median of the closed-class component areas, recolor closed components with area at least that median blue, and recolor smaller closed components green.

    Key insight:
    - Area alone is insufficient: only red components trapped inside grey-enclosed black chambers participate in the median comparison, while equally large open components remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREY}), "unexpected input colors"

    comps = _classify_red_components(g)
    closed = [comp for comp in comps if comp["closed"]]
    assert closed, "expected at least one closed red component"

    areas = sorted(int(comp["area"]) for comp in closed)
    median = areas[(len(areas) - 1) // 2]

    out = g.copy()
    for comp in closed:
        fill = BLUE if int(comp["area"]) >= median else GREEN
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray:
    assert area >= 1 and area <= max_h * max_w
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rs = [p[0] for p in trial]
                    cs = [p[1] for p in trial]
                    if max(rs) - min(rs) + 1 <= max_h and max(cs) - min(cs) + 1 <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        h = max(r for r, _ in cells) - min_r + 1
        w = max(c for _, c in cells) - min_c + 1
        patch = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            patch[r - min_r, c - min_c] = True

        variants = []
        for k in range(4):
            rot = np.rot90(patch, k)
            for variant in (rot, np.fliplr(rot)):
                if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
                    variants.append(variant.copy())
        if variants:
            return random.choice(variants)
    raise RuntimeError("failed to generate polyomino")


def _choose_closed_areas(count: int) -> list[int]:
    while True:
        areas = [random.randint(2, 10) for _ in range(count)]
        ordered = sorted(areas)
        median = ordered[(count - 1) // 2]
        if len(set(areas)) == 1:
            continue
        if not any(a < median for a in areas):
            continue
        if not any(a > median for a in areas):
            continue
        return areas


def _make_room_structure(closed: bool) -> dict:
    inner_h = random.randint(3, 5)
    inner_w = random.randint(3, 5)
    walls = set()

    for r in range(inner_h + 2):
        walls.add((r, 0))
        walls.add((r, inner_w + 1))
    for c in range(inner_w + 2):
        walls.add((0, c))
        walls.add((inner_h + 1, c))

    gap_cells: set[tuple[int, int]] = set()
    if not closed:
        side = random.choice("NSEW")
        if side in "NS":
            gap_len = 1 if inner_w <= 3 else random.choice([1, 1, 2])
            start = random.randint(1, inner_w - gap_len + 1)
            rr = 0 if side == "N" else inner_h + 1
            for c in range(start, start + gap_len):
                gap_cells.add((rr, c))
        else:
            gap_len = 1 if inner_h <= 3 else random.choice([1, 1, 2])
            start = random.randint(1, inner_h - gap_len + 1)
            cc = 0 if side == "W" else inner_w + 1
            for r in range(start, start + gap_len):
                gap_cells.add((r, cc))
        walls -= gap_cells

    branch_count = random.randint(1, 3)
    for _ in range(branch_count):
        side = random.choice("NSEW")
        if side in "NS":
            anchor_c = random.randint(1, inner_w)
            anchor = (0, anchor_c) if side == "N" else (inner_h + 1, anchor_c)
            step = (-1, 0) if side == "N" else (1, 0)
        else:
            anchor_r = random.randint(1, inner_h)
            anchor = (anchor_r, 0) if side == "W" else (anchor_r, inner_w + 1)
            step = (0, -1) if side == "W" else (0, 1)
        if anchor in gap_cells:
            continue
        cur = anchor
        length = random.randint(1, 2)
        for _ in range(length):
            cur = (cur[0] + step[0], cur[1] + step[1])
            walls.add(cur)
        if random.random() < 0.5:
            turn = random.choice([(step[1], step[0]), (-step[1], -step[0])])
            cur = (cur[0] + turn[0], cur[1] + turn[1])
            walls.add(cur)

    min_r = min(r for r, _ in walls)
    min_c = min(c for _, c in walls)
    walls = {(r - min_r, c - min_c) for r, c in walls}
    inner_r0 = 1 - min_r
    inner_c0 = 1 - min_c
    return {
        "walls": walls,
        "bbox_h": max(r for r, _ in walls) + 1,
        "bbox_w": max(c for _, c in walls) + 1,
        "inner_r0": inner_r0,
        "inner_c0": inner_c0,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "closed": closed,
    }


def _room_padding_cells(room: dict, top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    padded = set()
    for r, c in room["walls"]:
        rr = top + r
        cc = left + c
        padded.add((rr, cc))
        for nr, nc in _neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                padded.add((nr, nc))
    return padded


def _place_room(room: dict, blocked: set[tuple[int, int]], h: int, w: int) -> dict | None:
    max_r = h - room["bbox_h"] - 1
    max_c = w - room["bbox_w"] - 1
    if max_r < 1 or max_c < 1:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(1, max_r)
        left = random.randint(1, max_c)
        global_walls = {(top + r, left + c) for r, c in room["walls"]}
        if any((r, c) in blocked for r, c in global_walls):
            continue
        padded = _room_padding_cells(room, top, left, h, w)
        if any(cell in blocked for cell in padded):
            continue
        room_copy = dict(room)
        room_copy["top"] = top
        room_copy["left"] = left
        room_copy["global_walls"] = global_walls
        room_copy["blocked_cells"] = padded
        return room_copy
    return None


def _interior_mask_for_room(room: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0 = room["top"] + room["inner_r0"]
    c0 = room["left"] + room["inner_c0"]
    mask[r0 : r0 + room["inner_h"], c0 : c0 + room["inner_w"]] = True
    return mask


def _edge_reachable_black_mask(grid: np.ndarray) -> np.ndarray:
    labels, edge_touch = _label_black_regions(grid)
    mask = np.zeros(grid.shape, dtype=bool)
    for label, touches in enumerate(edge_touch):
        if touches:
            mask |= labels == label
    return mask


def _red_forbidden_mask(grid: np.ndarray) -> np.ndarray:
    forbidden = grid == RED
    extra = forbidden.copy()
    h, w = grid.shape
    for r, c in np.argwhere(grid == RED):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                extra[nr, nc] = True
    return extra


def _adjacent_black_exists(region_mask: np.ndarray, placed_cells: list[tuple[int, int]]) -> bool:
    placed_set = set(placed_cells)
    for r, c in placed_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < region_mask.shape[0] and 0 <= nc < region_mask.shape[1]:
                if region_mask[nr, nc] and (nr, nc) not in placed_set:
                    return True
    return False


def _place_patch_in_region(grid: np.ndarray, region_mask: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    forbidden = _red_forbidden_mask(grid)
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            placed = []
            ok = True
            for pr, pc in np.argwhere(patch):
                rr = r0 + int(pr)
                cc = c0 + int(pc)
                if not region_mask[rr, cc] or forbidden[rr, cc]:
                    ok = False
                    break
                placed.append((rr, cc))
            if ok and _adjacent_black_exists(region_mask, placed):
                candidates.append(placed)
    if not candidates:
        return False
    for r, c in random.choice(candidates):
        grid[r, c] = RED
    return True


def _component_summary(grid: np.ndarray) -> dict:
    comps = _classify_red_components(grid)
    closed = [comp for comp in comps if comp["closed"]]
    open_comps = [comp for comp in comps if not comp["closed"]]
    areas = sorted(int(comp["area"]) for comp in closed)
    median = areas[(len(areas) - 1) // 2] if areas else None
    return {
        "components": comps,
        "closed": closed,
        "open": open_comps,
        "median": median,
    }


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    grid = np.zeros((h, w), dtype=int)

    total_count = random.randint(5, 7)
    closed_count = random.choice([3, 3, 4])
    if closed_count >= total_count:
        closed_count = total_count - 1
    open_count = total_count - closed_count

    closed_areas = _choose_closed_areas(closed_count)
    open_areas = [random.randint(2, 10) for _ in range(open_count)]

    blocked: set[tuple[int, int]] = set()
    placed_rooms = []
    for _ in range(closed_count):
        room = _make_room_structure(closed=True)
        placed = _place_room(room, blocked, h, w)
        if placed is None:
            return None
        placed_rooms.append(placed)
        blocked |= placed["blocked_cells"]

    open_room = None
    if open_count > 0 and random.random() < 0.75:
        room = _make_room_structure(closed=False)
        placed = _place_room(room, blocked, h, w)
        if placed is not None:
            open_room = placed
            placed_rooms.append(placed)
            blocked |= placed["blocked_cells"]

    for room in placed_rooms:
        for r, c in room["global_walls"]:
            grid[r, c] = GREY

    for room, area in zip([r for r in placed_rooms if r["closed"]], closed_areas):
        inner_area = room["inner_h"] * room["inner_w"]
        if inner_area <= area:
            return None
        max_h = room["inner_h"]
        max_w = room["inner_w"]
        patch = _random_polyomino(area, max_h=max_h, max_w=max_w)
        if not _place_patch_in_region(grid, _interior_mask_for_room(room, grid.shape), patch):
            return None

    remaining_open_areas = list(open_areas)
    if open_room is not None and remaining_open_areas:
        area = remaining_open_areas.pop(0)
        inner_area = open_room["inner_h"] * open_room["inner_w"]
        if inner_area <= area:
            return None
        patch = _random_polyomino(
            area,
            max_h=open_room["inner_h"],
            max_w=open_room["inner_w"],
        )
        if not _place_patch_in_region(grid, _interior_mask_for_room(open_room, grid.shape), patch):
            return None

    edge_mask = _edge_reachable_black_mask(grid)
    for area in remaining_open_areas:
        patch = _random_polyomino(area, max_h=min(4, area), max_w=min(5, area))
        if not _place_patch_in_region(grid, edge_mask, patch):
            return None

    summary = _component_summary(grid)
    if len(summary["components"]) != total_count:
        return None
    if len(summary["closed"]) < 3 or len(summary["open"]) < 1:
        return None
    if summary["median"] is None:
        return None
    if not any(int(comp["area"]) < int(summary["median"]) for comp in summary["closed"]):
        return None
    if not any(int(comp["area"]) > int(summary["median"]) for comp in summary["closed"]):
        return None

    out = main(grid)
    out_colors = set(map(int, np.unique(out)))
    if not {BLUE, GREEN, RED}.issubset(out_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
