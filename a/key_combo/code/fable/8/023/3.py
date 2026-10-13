from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, find_components, get_bg, in_bounds, neighbors4


SIDE_CHOICES = (3, 5, 7, 9)
COUNT_CHOICES = (3, 4)
GRID_SIDE_RANGE = (28, 30)
TRUE_VOTE_CHOICES = tuple(range(1, 8))
DECOY_AREA_CHOICES = (1, 2, 3, 4)
GEN_MAX_ATTEMPTS = 1000
ANCHOR_CANDIDATE_LIMIT = 120
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _square_bounds(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = map(int, center)
    d = int(side) // 2
    return r - d, c - d, r + d, c + d


def _paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    r0, c0, r1, c1 = _square_bounds(center, side)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _line_degree(mask: np.ndarray, r: int, c: int) -> int:
    deg = 0
    for nr, nc in neighbors4(r, c):
        if in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            deg += 1
    return deg


def _touches_mask(mask: np.ndarray, r: int, c: int) -> bool:
    for nr, nc in neighbors4(r, c):
        if in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            return True
    return False


def _find_square_and_line(component: dict, shape: tuple[int, int]) -> dict:
    coords = component["coords"]
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([[r0, c0]])
    crop[local[:, 0], local[:, 1]] = True

    candidates = []
    max_side = min(crop.shape)
    for side in range(max_side if max_side % 2 == 1 else max_side - 1, 0, -2):
        for sr in range(0, crop.shape[0] - side + 1):
            for sc in range(0, crop.shape[1] - side + 1):
                if not crop[sr : sr + side, sc : sc + side].all():
                    continue

                square_local = np.zeros_like(crop, dtype=bool)
                square_local[sr : sr + side, sc : sc + side] = True
                line_local = crop & ~square_local
                if not line_local.any():
                    continue

                line_parts = components_mask(line_local, connectivity=4)
                if len(line_parts) != 1:
                    continue

                line_coords = [tuple(map(int, p)) for p in line_parts[0]]
                deg_by_cell = {}
                touch_cells = []
                endpoint_cells = []
                isolated = 0
                branched = False
                for rr, cc in line_coords:
                    deg = _line_degree(line_local, rr, cc)
                    deg_by_cell[(rr, cc)] = deg
                    if deg > 2:
                        branched = True
                        break
                    if deg == 0:
                        isolated += 1
                    if deg == 1:
                        endpoint_cells.append((rr, cc))
                    if _touches_mask(square_local, rr, cc):
                        touch_cells.append((rr, cc))
                if branched:
                    continue
                if len(line_coords) == 1:
                    if isolated != 1:
                        continue
                elif len(endpoint_cells) != 2 or isolated != 0:
                    continue
                if len(touch_cells) != 1:
                    continue
                if deg_by_cell[touch_cells[0]] > 1:
                    continue

                touch_local = touch_cells[0]
                if len(line_coords) == 1:
                    far_local = touch_local
                else:
                    far_local = endpoint_cells[0] if endpoint_cells[1] == touch_local else endpoint_cells[1]

                square_global = np.zeros(shape, dtype=bool)
                square_global[r0 + sr : r0 + sr + side, c0 + sc : c0 + sc + side] = True
                line_global = np.zeros(shape, dtype=bool)
                for rr, cc in line_coords:
                    line_global[r0 + rr, c0 + cc] = True

                candidates.append(
                    {
                        "color": color,
                        "side": int(side),
                        "bbox": (r0 + sr, c0 + sc, r0 + sr + side - 1, c0 + sc + side - 1),
                        "center": (r0 + sr + side // 2, c0 + sc + side // 2),
                        "square_mask": square_global,
                        "line_mask": line_global,
                        "line_touch": (r0 + touch_local[0], c0 + touch_local[1]),
                        "line_end": (r0 + far_local[0], c0 + far_local[1]),
                    }
                )
        if candidates:
            break

    assert len(candidates) == 1, "expected exactly one odd solid square attached to a simple same-color line"
    return candidates[0]


def _touches_line(line_mask: np.ndarray, coords: np.ndarray) -> bool:
    for r, c in coords:
        rr, cc = int(r), int(c)
        for nr, nc in neighbors4(rr, cc):
            if in_bounds(line_mask.shape, nr, nc) and line_mask[nr, nc]:
                return True
    return False


def _parse_grid(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    grey_components = find_components(
        g,
        bg=bg,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    color_components = find_components(
        g,
        bg=bg,
        colors=[c for c in np.unique(g) if int(c) not in {bg, Color.GREY}],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    objects = []
    used_grey_ids = []
    for comp in color_components:
        spec = _find_square_and_line(comp, g.shape)
        attached = [i for i, grey in enumerate(grey_components) if _touches_line(spec["line_mask"], grey["coords"])]
        assert len(attached) == 1, "each line must identify exactly one grey vote cluster"
        grey_id = attached[0]
        grey = grey_components[grey_id]
        spec["votes"] = int(grey["area"])
        spec["grey_id"] = grey_id
        objects.append(spec)
        used_grey_ids.append(grey_id)

    assert objects, "expected at least one square-line object"
    vote_counts = [obj["votes"] for obj in objects]
    assert len(set(vote_counts)) == len(vote_counts), "grey vote counts must be unique"

    decoys = [grey for i, grey in enumerate(grey_components) if i not in set(used_grey_ids)]
    return bg, grey_components, objects, decoys


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reassign odd square sizes by ranking the grey vote clusters reached through each square's same-color guide line.

    Input structure:
    - A background grid contains several separated monochrome solid odd-sided squares in distinct colors.
    - Each square is part of a same-color 1-cell-wide orthogonal polyline leading to one grey marker cluster.
    - Extra grey clusters may appear elsewhere, but they are not connected to any guide line.

    Rule:
    - For each colored object, isolate the solid square and trace its attached line to the unique grey cluster that touches that line.
    - Count the pixels in those line-reached grey clusters, then rank the squares from largest vote count to smallest vote count.
    - Keep every square's original center and color, assign the largest original side length to the highest vote total, the next largest side to the next vote total, and erase all lines and grey clusters.

    Key insight:
    - The grey pixel counts do not compute new sizes directly; they only provide the ranking key used to reshuffle the existing multiset of odd square side lengths among the original square centers.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, _grey_components, objects, _decoys = _parse_grid(g)
    ranked = sorted(objects, key=lambda obj: -obj["votes"])
    sizes = sorted((int(obj["side"]) for obj in objects), reverse=True)

    out = np.full(g.shape, bg, dtype=int)
    for obj, side in zip(ranked, sizes):
        _paint_square(out, obj["center"], int(side), int(obj["color"]))
    return out


def _slot_centers(h: int, w: int) -> list[tuple[int, int]]:
    top = 5
    bottom = h - 6
    left = 5
    right = w - 6
    return [(top, left), (top, right), (bottom, left), (bottom, right)]


def _sample_centers(h: int, w: int, n: int) -> list[tuple[int, int]]:
    slots = random.sample(_slot_centers(h, w), n)
    centers = []
    for r, c in slots:
        rr = int(np.clip(r + random.randint(-1, 1), 4, h - 5))
        cc = int(np.clip(c + random.randint(-1, 1), 4, w - 5))
        centers.append((rr, cc))
    return centers


def _square_mask_for_generation(shape: tuple[int, int], center: tuple[int, int], side: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = _square_bounds(center, side)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _expanded_square_halo(shape: tuple[int, int], center: tuple[int, int], side: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = _square_bounds(center, side)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(shape[0] - 1, r1 + 1)
    c1 = min(shape[1] - 1, c1 + 1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _start_cells(center: tuple[int, int], side: int) -> list[tuple[tuple[int, int], str]]:
    r0, c0, r1, c1 = _square_bounds(center, side)
    cr, cc = center
    return [
        ((r0 - 1, cc), "up"),
        ((r1 + 1, cc), "down"),
        ((cr, c0 - 1), "left"),
        ((cr, c1 + 1), "right"),
    ]


def _reachable_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    if not in_bounds(shape, *start) or not in_bounds(shape, *goal):
        return None
    if blocked[start] or blocked[goal]:
        return None

    q = deque([start])
    prev = {start: None}
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        for nr, nc in neighbors4(*cell):
            nxt = (int(nr), int(nc))
            if not in_bounds(shape, *nxt) or blocked[nxt] or nxt in prev:
                continue
            prev[nxt] = cell
            q.append(nxt)
    if goal not in prev:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _grow_blob(
    allowed: np.ndarray,
    seed: tuple[int, int],
    area: int,
) -> list[tuple[int, int]] | None:
    if not allowed[seed]:
        return None
    blob = {seed}
    frontier = {seed}
    while len(blob) < int(area):
        candidates = []
        for r, c in frontier:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if in_bounds(allowed.shape, nr, nc) and allowed[nr, nc] and (nr, nc) not in blob:
                    candidates.append((nr, nc))
        if not candidates:
            return None
        nxt = random.choice(candidates)
        blob.add(nxt)
        frontier.add(nxt)
    return list(blob)


def _free_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    random.shuffle(coords)
    return [tuple(map(int, rc)) for rc in coords]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _place_true_cluster_and_path(
    shape: tuple[int, int],
    square_masks: list[np.ndarray],
    occupied: np.ndarray,
    center: tuple[int, int],
    side: int,
    vote_area: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    shape_mask = np.zeros(shape, dtype=bool)
    for sq in square_masks:
        shape_mask |= sq

    starts = _start_cells(center, side)
    random.shuffle(starts)
    free = ~occupied

    for start, _name in starts:
        if not in_bounds(shape, *start) or not free[start]:
            continue

        blocked_from_existing = _dilate4(occupied)
        blocked_from_existing[start] = False

        anchor_candidates = _free_cells(free.copy())[:ANCHOR_CANDIDATE_LIMIT]
        for anchor in anchor_candidates:
            if anchor == start:
                continue
            if abs(anchor[0] - center[0]) + abs(anchor[1] - center[1]) < 8:
                continue
            if sum(abs(anchor[k] - start[k]) for k in (0, 1)) < 7:
                continue
            if _touches_mask(shape_mask, *anchor):
                continue

            local_allowed = (~_dilate4(occupied)).copy()
            local_allowed[start] = False
            local_allowed[anchor] = True
            blob = _grow_blob(local_allowed, anchor, vote_area)
            if blob is None:
                continue
            blob_mask = np.zeros(shape, dtype=bool)
            for cell in blob:
                blob_mask[cell] = True

            contact_options = []
            for br, bc in blob:
                for nr, nc in neighbors4(br, bc):
                    cell = (int(nr), int(nc))
                    if not in_bounds(shape, *cell) or occupied[cell] or blob_mask[cell] or cell == start:
                        continue
                    if _touches_mask(shape_mask, *cell):
                        continue
                    contact_options.append(cell)
            random.shuffle(contact_options)
            for contact in contact_options[:24]:
                blocked = blocked_from_existing | _dilate4(blob_mask)
                blocked[start] = False
                blocked[contact] = False
                path = _reachable_path(shape, start, contact, blocked)
                if path is None or len(path) < 3:
                    continue
                return blob, path
    return None


def _place_decoy_blob(
    shape: tuple[int, int],
    occupied: np.ndarray,
    center: tuple[int, int],
    side: int,
    area: int,
) -> list[tuple[int, int]] | None:
    cr, cc = center
    r0, c0, r1, c1 = _square_bounds(center, side)
    ring = []
    for r in range(max(1, r0 - 4), min(shape[0] - 1, r1 + 5)):
        for c in range(max(1, c0 - 4), min(shape[1] - 1, c1 + 5)):
            if occupied[r, c]:
                continue
            dist = min(abs(r - r0), abs(r - r1), abs(c - c0), abs(c - c1))
            manhattan = abs(r - cr) + abs(c - cc)
            if 2 <= dist <= 4 and manhattan <= 7:
                ring.append((r, c))
    random.shuffle(ring)
    for seed in ring:
        allowed = (~_dilate4(occupied)).copy()
        blob = _grow_blob(allowed, seed, area)
        if blob is not None:
            return blob
    return None


def _grey_distance_summary(objects: list[dict], grey_components: list[dict]) -> list[tuple[int, int, int]]:
    pairs = []
    for obj in objects:
        cr, cc = map(int, obj["center"])
        true_dist = -1
        best_dist = 10**9
        best_id = -1
        for grey_id, grey in enumerate(grey_components):
            rr = grey["coords"][:, 0]
            cc2 = grey["coords"][:, 1]
            dist = int(np.min(np.abs(rr - cr) + np.abs(cc2 - cc)))
            if dist < best_dist:
                best_dist = dist
                best_id = int(grey_id)
            if int(grey_id) == int(obj["grey_id"]):
                true_dist = dist
        pairs.append((best_id, best_dist, true_dist))
    return pairs


def _named_slot_centers(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (7, 7),
        "tr": (7, w - 8),
        "bl": (h - 8, 7),
        "br": (h - 8, w - 8),
    }


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("polyline segments must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        cells.extend(seg if i == 0 else seg[1:])
    return cells


def _grow_blob_in_zone(
    zone_cells: list[tuple[int, int]],
    anchor: tuple[int, int],
    area: int,
) -> list[tuple[int, int]] | None:
    allowed = set(zone_cells)
    if anchor not in allowed or len(allowed) < int(area):
        return None
    blob = {anchor}
    while len(blob) < int(area):
        frontier = []
        for r, c in blob:
            for nr, nc in neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if nxt in allowed and nxt not in blob:
                    frontier.append(nxt)
        if not frontier:
            return None
        blob.add(random.choice(frontier))
    return list(blob)


def _route_plan(slot: str, center: tuple[int, int], side: int, h: int, w: int) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = _square_bounds(center, side)
    if slot == "tl":
        route_col = w // 2 - 2
        end_row = h - 6
        start = (r1 + 1, center[1])
        turn_row = start[0] + 1
        points = [start, (turn_row, start[1]), (turn_row, route_col), (end_row, route_col)]
        anchor = (end_row + 1, route_col)
        zone_rows = range(anchor[0], min(h - 1, anchor[0] + 3))
    elif slot == "tr":
        route_col = w // 2 + 2
        end_row = h - 5
        start = (r1 + 1, center[1])
        turn_row = start[0] + 1
        points = [start, (turn_row, start[1]), (turn_row, route_col), (end_row, route_col)]
        anchor = (end_row + 1, route_col)
        zone_rows = range(anchor[0], min(h - 1, anchor[0] + 3))
    elif slot == "bl":
        route_col = 1
        end_row = 8
        start = (center[0], c0 - 1)
        points = [start, (start[0], route_col), (end_row, route_col)]
        anchor = (end_row - 1, route_col)
        zone_rows = range(1, anchor[0] + 1)
        zone_cols = range(route_col, route_col + 1)
    else:
        route_col = w - 2
        end_row = 8
        start = (center[0], c1 + 1)
        points = [start, (start[0], route_col), (end_row, route_col)]
        anchor = (end_row - 1, route_col)
        zone_rows = range(1, anchor[0] + 1)
        zone_cols = range(route_col, route_col + 1)

    path = _polyline_cells(points)
    if slot in {"tl", "tr"}:
        zone_cols = range(max(1, route_col - 1), min(w - 1, route_col + 2))
    zone_cells = [(r, c) for r in zone_rows for c in zone_cols]
    return path, anchor, zone_cells


def _decoy_zone(slot: str, center: tuple[int, int], side: int, h: int, w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _square_bounds(center, side)
    if slot == "tl":
        rows = range(max(1, r0 - 3), r0)
        cols = range(max(1, c0 - 3), c0)
    elif slot == "tr":
        rows = range(max(1, r0 - 3), r0)
        cols = range(c1 + 1, min(w - 1, c1 + 4))
    elif slot == "bl":
        rows = range(r1 + 1, min(h - 1, r1 + 4))
        cols = range(max(1, c0 - 3), c0)
    else:
        rows = range(r1 + 1, min(h - 1, r1 + 4))
        cols = range(c1 + 1, min(w - 1, c1 + 4))
    return [(r, c) for r in rows for c in cols]


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30

    bg, grey_components, objects, decoys = _parse_grid(inp)
    assert len(objects) in COUNT_CHOICES
    assert decoys, "generator must include detached grey decoys"
    assert len({obj["color"] for obj in objects}) == len(objects)
    assert len({obj["side"] for obj in objects}) >= 2
    assert any(obj["votes"] != obj["side"] for obj in objects)

    ranked = sorted(objects, key=lambda obj: -obj["votes"])
    size_order = sorted((obj["side"] for obj in objects), reverse=True)
    assigned = {
        (obj["center"], obj["color"]): side for obj, side in zip(ranked, size_order)
    }
    assert any(assigned[(obj["center"], obj["color"])] != obj["side"] for obj in objects), "size ranking must cause a real reshuffle"

    nearest_pairs = _grey_distance_summary(objects, grey_components)
    assert any(best_id != int(obj["grey_id"]) and best_dist < true_dist for obj, (best_id, best_dist, true_dist) in zip(objects, nearest_pairs)), "at least one object should have a closer grey decoy than its true cluster"

    out_bg = int(get_bg(out))
    out_components = find_components(
        out,
        bg=out_bg,
        colors=[c for c in np.unique(out) if int(c) != out_bg],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(out_components) == len(objects)
    seen = set()
    for comp in out_components:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1) and side % 2 == 1
        assert int(comp["area"]) == side * side
        center = (r0 + side // 2, c0 + side // 2)
        key = (center, color)
        assert key in assigned
        assert assigned[key] == side
        seen.add(key)
    assert len(seen) == len(objects)
    assert np.array_equal(out, main(inp))
    assert int(np.count_nonzero(out == Color.GREY)) == 0
    assert int(np.count_nonzero(out == bg)) < out.size


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n = random.choice(COUNT_CHOICES)
    bg = Color.BLACK
    grid = np.full((h, w), bg, dtype=int)

    sizes = sorted(random.sample(SIDE_CHOICES, n), reverse=True)
    votes = random.sample(TRUE_VOTE_CHOICES, n)
    colors = random.sample([c for c in range(1, 10) if c != Color.GREY], n)
    slot_map = _named_slot_centers(h, w)
    chosen_slots = random.sample(list(slot_map), n)
    specs = []
    for slot, side, vote, color in zip(chosen_slots, sizes, votes, colors):
        center = slot_map[slot]
        r0, c0, r1, c1 = _square_bounds(center, side)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            return None
        specs.append({"slot": slot, "center": center, "side": side, "votes": vote, "color": color})

    occupied = np.zeros((h, w), dtype=bool)
    true_clusters = []
    paths = []
    decoy_blobs = []

    for spec in specs:
        square_mask = _square_mask_for_generation((h, w), spec["center"], spec["side"])
        if np.any(square_mask & occupied):
            return None
        occupied |= square_mask

    for spec in specs:
        path, anchor, zone_cells = _route_plan(spec["slot"], spec["center"], spec["side"], h, w)
        blob = _grow_blob_in_zone([cell for cell in zone_cells if not occupied[cell]], anchor, spec["votes"])
        if blob is None:
            return None
        blob_mask = np.zeros((h, w), dtype=bool)
        for cell in blob:
            blob_mask[cell] = True
        path_mask = np.zeros((h, w), dtype=bool)
        for cell in path:
            if not in_bounds((h, w), *cell) or occupied[cell]:
                return None
            path_mask[cell] = True
        if np.any(path_mask & blob_mask):
            return None
        if np.any(_dilate4(blob_mask) & (occupied & ~_square_mask_for_generation((h, w), spec["center"], spec["side"]))):
            return None
        if np.any(path_mask & _dilate4(occupied & ~_square_mask_for_generation((h, w), spec["center"], spec["side"]))):
            return None
        occupied |= blob_mask | path_mask
        true_clusters.append(blob)
        paths.append(path)

    for spec in specs:
        zone = [cell for cell in _decoy_zone(spec["slot"], spec["center"], spec["side"], h, w) if not occupied[cell]]
        if not zone:
            continue
        area = min(random.choice(DECOY_AREA_CHOICES), len(zone))
        decoy = _grow_blob_in_zone(zone, random.choice(zone), area)
        if decoy is None:
            continue
        decoy_mask = np.zeros((h, w), dtype=bool)
        for cell in decoy:
            decoy_mask[cell] = True
        if np.any(decoy_mask & occupied) or np.any(_dilate4(decoy_mask) & occupied):
            continue
        occupied |= decoy_mask
        decoy_blobs.append(decoy)

    if not decoy_blobs:
        return None

    for spec in specs:
        _paint_square(grid, spec["center"], spec["side"], spec["color"])
    for path, spec in zip(paths, specs):
        for r, c in path:
            grid[r, c] = spec["color"]
    for blob in true_clusters + decoy_blobs:
        for r, c in blob:
            grid[r, c] = Color.GREY

    out = main(grid)
    _validate_instance(grid, out)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            grid = _generate_once()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
