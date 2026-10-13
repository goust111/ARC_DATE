from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
EDGE_ORDER = ("left", "top", "right", "bottom")
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _detect_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    by_edge: dict[str, tuple[int, int, int]] = {}
    for c in range(1, w - 1):
        if int(grid[0, c]) != BG:
            by_edge["top"] = (0, c, int(grid[0, c]))
        if int(grid[h - 1, c]) != BG:
            by_edge["bottom"] = (h - 1, c, int(grid[h - 1, c]))
    for r in range(1, h - 1):
        if int(grid[r, 0]) != BG:
            by_edge["left"] = (r, 0, int(grid[r, 0]))
        if int(grid[r, w - 1]) != BG:
            by_edge["right"] = (r, w - 1, int(grid[r, w - 1]))
    markers = []
    for edge in EDGE_ORDER:
        if edge in by_edge:
            r, c, color = by_edge[edge]
            markers.append({"edge": edge, "r": r, "c": c, "color": color})
    colors = [m["color"] for m in markers]
    assert 2 <= len(markers) <= 4
    assert len(colors) == len(set(colors))
    return markers


def _distance_to_edge(edge: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if edge == "left":
        return c0
    if edge == "top":
        return r0
    if edge == "right":
        return w - 1 - c1
    return h - 1 - r1


def _component_sort_key(edge: str, item: dict) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = item["bbox"]
    if edge in ("left", "right"):
        return item["distance"], r0, c0
    return item["distance"], c0, r0


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[rr + int(r0), cc + int(c0)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each grey sprite to the edge selected by its hole count under the border-marker dictionary, erasing unmatched sprites.

    Input structure:
    - A black background grid contains 2-4 distinct single-pixel colored markers on non-corner border cells.
    - Several disjoint grey (5) 4-connected sprites lie strictly inside the grid; each sprite may contain 0 or more enclosed black holes.
    - The present marker edges define an ordered direction dictionary in clockwise order starting from the left edge: left, top, right, bottom, skipping missing edges.

    Rule:
    - Count each grey sprite's enclosed black regions using 4-connectivity inside the sprite's tight bounding box.
    - If a sprite has h holes and 1 <= h <= number_of_markers, map h to the h-th present marker edge and translate the whole sprite to that edge.
    - Keep the coordinate parallel to the target edge unchanged; along the perpendicular axis, pack sprites assigned to the same edge from nearest to farthest so the first touches the edge and later ones touch the previous packed sprite. Sprites with h = 0 or h above the dictionary size are erased. Border markers stay unchanged.

    Key insight:
    The topology of each grey sprite, not its geometry or position, selects a direction through the ordered border-marker legend; only after this hole-count lookup can the final edge compaction be determined.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers = _detect_markers(g)
    hole_to_edge = {idx + 1: marker["edge"] for idx, marker in enumerate(markers)}

    groups: dict[str, list[dict]] = {edge: [] for edge in EDGE_ORDER}
    for coords in _components_of_color(g, GREY):
        mask, bbox = _tight_mask_from_coords(coords)
        holes = _count_holes(mask)
        if 1 <= holes <= len(markers):
            edge = hole_to_edge[holes]
            groups[edge].append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "distance": _distance_to_edge(edge, bbox, (h, w)),
                }
            )

    out = np.full_like(g, BG)
    for edge in EDGE_ORDER:
        offset = 0
        for item in sorted(groups[edge], key=lambda x, e=edge: _component_sort_key(e, x)):
            mask = item["mask"]
            r0, c0, r1, c1 = item["bbox"]
            ph, pw = mask.shape
            if edge == "left":
                dest_r0, dest_c0 = r0, offset
                offset += pw
            elif edge == "right":
                dest_r0, dest_c0 = r0, w - offset - pw
                offset += pw
            elif edge == "top":
                dest_r0, dest_c0 = offset, c0
                offset += ph
            else:
                dest_r0, dest_c0 = h - offset - ph, c0
                offset += ph
            _paste_mask(out, mask, dest_r0, dest_c0, GREY)

    for marker in markers:
        out[marker["r"], marker["c"]] = marker["color"]
    return out


def _shape_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


HOLE_TEMPLATES = {
    1: [
        _shape_from_rows(["111", "101", "111"]),
        _shape_from_rows(["1111", "1001", "1111"]),
        _shape_from_rows(["1111", "1011", "1001", "1111"]),
        _shape_from_rows(["1111", "1001", "1011", "1111"]),
    ],
    2: [
        _shape_from_rows(["11111", "10101", "11111"]),
        _shape_from_rows(["111", "101", "111", "101", "111"]),
        _shape_from_rows(["11111", "10101", "11111", "00100"]),
        _shape_from_rows(["111110", "101010", "111110"]),
    ],
}


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    return mask[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1].copy()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _normalize_mask(out)


def _try_add_exterior_pixel(mask: np.ndarray, holes: int) -> np.ndarray:
    if int(mask.sum()) >= 15:
        return mask
    padded = np.pad(mask, 1, constant_values=False)
    candidates = []
    h, w = padded.shape
    for r in range(h):
        for c in range(w):
            if padded[r, c]:
                continue
            if not any(0 <= nr < h and 0 <= nc < w and padded[nr, nc] for nr, nc in _neighbors4(r, c)):
                continue
            trial = padded.copy()
            trial[r, c] = True
            trial = _normalize_mask(trial)
            if _count_holes(trial) == holes and 5 <= int(trial.sum()) <= 15:
                candidates.append(trial)
    return random.choice(candidates) if candidates and random.random() < 0.7 else mask


def _random_polyomino() -> np.ndarray:
    for _ in range(200):
        target = random.randint(5, 9)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if abs(nr) > 4 or abs(nc) > 4 or (nr, nc) in cells:
                        continue
                    frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, c0 = min(rows), min(cols)
        h = max(rows) - r0 + 1
        w = max(cols) - c0 + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        mask = _normalize_mask(mask)
        if _count_holes(mask) != 0:
            continue
        if mask.shape[0] * mask.shape[1] == int(mask.sum()) and min(mask.shape) >= 2:
            continue
        return mask
    raise RuntimeError("failed to build a 0-hole sprite")


def _random_mask_for_holes(holes: int) -> np.ndarray:
    if holes == 0:
        return _random_polyomino()
    mask = _random_transform(random.choice(HOLE_TEMPLATES[holes]))
    for _ in range(random.randint(0, 2)):
        mask = _try_add_exterior_pixel(mask, holes)
    assert _count_holes(mask) == holes
    assert 5 <= int(mask.sum()) <= 15
    return mask


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    rr, cc = np.nonzero(mask)
    for dr, dc in zip(rr.tolist(), cc.tolist()):
        r = r0 + dr
        c = c0 + dc
        r1 = max(0, r - pad)
        r2 = min(h, r + pad + 1)
        c1 = max(0, c - pad)
        c2 = min(w, c + pad + 1)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _stamp_occupied(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    rr, cc = np.nonzero(mask)
    occupied[rr + int(r0), cc + int(c0)] = True


def _candidate_parallel_positions(length: int, size: int) -> list[int]:
    vals = list(range(1, length - size))
    random.shuffle(vals)
    return vals


def _place_output_objects(h: int, w: int, objects: list[dict]) -> np.ndarray | None:
    valid = [obj for obj in objects if obj["edge"] is not None]
    groups: dict[str, list[dict]] = {edge: [] for edge in EDGE_ORDER}
    for obj in valid:
        groups[obj["edge"]].append(obj)
    for edge in EDGE_ORDER:
        random.shuffle(groups[edge])
        offset = 0
        for obj in groups[edge]:
            obj["order_edge"] = edge
            obj["order_offset"] = offset
            if edge in ("left", "right"):
                offset += obj["mask"].shape[1]
            else:
                offset += obj["mask"].shape[0]
        limit = w - 2 if edge in ("left", "right") else h - 2
        if offset > limit:
            return None

    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(valid, key=lambda obj: int(obj["mask"].sum()), reverse=True)
    for obj in order:
        mask = obj["mask"]
        mh, mw = mask.shape
        edge = obj["edge"]
        offset = obj["order_offset"]
        if edge == "left":
            c0 = offset
            choices = _candidate_parallel_positions(h, mh)
            attr = "out_r0"
        elif edge == "right":
            c0 = w - offset - mw
            choices = _candidate_parallel_positions(h, mh)
            attr = "out_r0"
        elif edge == "top":
            r0 = offset
            choices = _candidate_parallel_positions(w, mw)
            attr = "out_c0"
        else:
            r0 = h - offset - mh
            choices = _candidate_parallel_positions(w, mw)
            attr = "out_c0"

        placed = False
        for pos in choices:
            if edge in ("left", "right"):
                r0 = pos
            else:
                c0 = pos
            if not _can_place_mask(occupied, mask, r0, c0, pad=1):
                continue
            obj["out_r0"] = r0
            obj["out_c0"] = c0
            _stamp_occupied(occupied, mask, r0, c0)
            placed = True
            break
        if not placed:
            return None

    expected = np.full((h, w), BG, dtype=int)
    for obj in valid:
        _paste_mask(expected, obj["mask"], obj["out_r0"], obj["out_c0"], GREY)
    return expected


def _choose_marker_positions(expected: np.ndarray, edges: list[str]) -> list[dict] | None:
    h, w = expected.shape
    occupied = expected != BG
    colors = random.sample(MARKER_COLORS, len(edges))
    markers: list[dict] = []
    for edge, color in zip(edges, colors):
        candidates: list[tuple[int, int]] = []
        if edge in ("top", "bottom"):
            row = 0 if edge == "top" else h - 1
            for c in range(1, w - 1):
                if occupied[row, c]:
                    continue
                r1 = max(0, row - 1)
                r2 = min(h, row + 2)
                c1 = max(0, c - 1)
                c2 = min(w, c + 2)
                if occupied[r1:r2, c1:c2].any():
                    continue
                candidates.append((row, c))
        else:
            col = 0 if edge == "left" else w - 1
            for r in range(1, h - 1):
                if occupied[r, col]:
                    continue
                r1 = max(0, r - 1)
                r2 = min(h, r + 2)
                c1 = max(0, col - 1)
                c2 = min(w, col + 2)
                if occupied[r1:r2, c1:c2].any():
                    continue
                candidates.append((r, col))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        markers.append({"edge": edge, "r": r, "c": c, "color": color})
        occupied[r, c] = True
    markers.sort(key=lambda m: EDGE_ORDER.index(m["edge"]))
    return markers


def _place_valid_inputs(h: int, w: int, objects: list[dict], occupied: np.ndarray) -> bool:
    for edge in EDGE_ORDER:
        group = [obj for obj in objects if obj["edge"] == edge]
        group.sort(key=lambda obj: obj["order_offset"])
        prev = None
        for obj in group:
            mh, mw = obj["mask"].shape
            if edge == "left":
                lo = obj["out_c0"] + 1
                hi = w - mw - 1
                if prev is not None:
                    lo = max(lo, prev + 1)
                candidates = list(range(lo, hi + 1))
                random.shuffle(candidates)
                placed = False
                for c0 in candidates:
                    if _can_place_mask(occupied, obj["mask"], obj["out_r0"], c0, pad=1):
                        obj["in_r0"], obj["in_c0"] = obj["out_r0"], c0
                        _stamp_occupied(occupied, obj["mask"], obj["in_r0"], obj["in_c0"])
                        prev = c0
                        placed = True
                        break
            elif edge == "right":
                lo = 1
                hi = obj["out_c0"] - 1
                if prev is not None:
                    hi = min(hi, prev - 1)
                candidates = list(range(lo, hi + 1))
                random.shuffle(candidates)
                placed = False
                for c0 in candidates:
                    if _can_place_mask(occupied, obj["mask"], obj["out_r0"], c0, pad=1):
                        obj["in_r0"], obj["in_c0"] = obj["out_r0"], c0
                        _stamp_occupied(occupied, obj["mask"], obj["in_r0"], obj["in_c0"])
                        prev = c0
                        placed = True
                        break
            elif edge == "top":
                lo = obj["out_r0"] + 1
                hi = h - mh - 1
                if prev is not None:
                    lo = max(lo, prev + 1)
                candidates = list(range(lo, hi + 1))
                random.shuffle(candidates)
                placed = False
                for r0 in candidates:
                    if _can_place_mask(occupied, obj["mask"], r0, obj["out_c0"], pad=1):
                        obj["in_r0"], obj["in_c0"] = r0, obj["out_c0"]
                        _stamp_occupied(occupied, obj["mask"], obj["in_r0"], obj["in_c0"])
                        prev = r0
                        placed = True
                        break
            else:
                lo = 1
                hi = obj["out_r0"] - 1
                if prev is not None:
                    hi = min(hi, prev - 1)
                candidates = list(range(lo, hi + 1))
                random.shuffle(candidates)
                placed = False
                for r0 in candidates:
                    if _can_place_mask(occupied, obj["mask"], r0, obj["out_c0"], pad=1):
                        obj["in_r0"], obj["in_c0"] = r0, obj["out_c0"]
                        _stamp_occupied(occupied, obj["mask"], obj["in_r0"], obj["in_c0"])
                        prev = r0
                        placed = True
                        break
            if not placed:
                return False
    return True


def _place_invalid_inputs(h: int, w: int, objects: list[dict], occupied: np.ndarray) -> bool:
    for obj in [o for o in objects if o["edge"] is None]:
        mh, mw = obj["mask"].shape
        candidates = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _can_place_mask(occupied, obj["mask"], r0, c0, pad=1):
                continue
            obj["in_r0"], obj["in_c0"] = r0, c0
            _stamp_occupied(occupied, obj["mask"], r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_input_grid(h: int, w: int, markers: list[dict], objects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        _paste_mask(grid, obj["mask"], obj["in_r0"], obj["in_c0"], GREY)
    for marker in markers:
        grid[marker["r"], marker["c"]] = marker["color"]
    return grid


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, objects: list[dict], markers: list[dict]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    comps = _components_of_color(inp, GREY)
    if len(comps) != len(objects):
        return False
    holes = sorted(_count_holes(_tight_mask_from_coords(comp)[0]) for comp in comps)
    wanted = sorted(obj["holes"] for obj in objects)
    if holes != wanted:
        return False
    if sum(1 for obj in objects if 1 <= obj["holes"] <= len(markers)) < 2:
        return False
    if sum(1 for obj in objects if obj["holes"] == 0 or obj["holes"] > len(markers)) < 1:
        return False
    if any(not (5 <= int(obj["mask"].sum()) <= 15) for obj in objects):
        return False
    marker_edges = [m["edge"] for m in markers]
    if marker_edges != [edge for edge in EDGE_ORDER if edge in marker_edges]:
        return False
    return True


def _generate_case() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    edge_count = random.randint(2, 4)
    selected_edges = random.sample(list(EDGE_ORDER), edge_count)
    ordered_edges = [edge for edge in EDGE_ORDER if edge in selected_edges]
    hole_to_edge = {idx + 1: edge for idx, edge in enumerate(ordered_edges)}

    object_count = random.randint(3, 5)
    valid_count = random.randint(2, object_count - 1)
    hole_plan = [1, 2]
    while len(hole_plan) < valid_count:
        hole_plan.append(random.choice([1, 2]))
    while len(hole_plan) < object_count:
        hole_plan.append(0)
    random.shuffle(hole_plan)

    objects = []
    for holes in hole_plan:
        mask = _random_mask_for_holes(holes)
        objects.append(
            {
                "holes": holes,
                "mask": mask,
                "edge": hole_to_edge.get(holes),
            }
        )

    expected = _place_output_objects(h, w, objects)
    if expected is None:
        return None
    markers = _choose_marker_positions(expected, ordered_edges)
    if markers is None:
        return None
    for marker in markers:
        expected[marker["r"], marker["c"]] = marker["color"]

    occupied = np.zeros((h, w), dtype=bool)
    for marker in markers:
        occupied[marker["r"], marker["c"]] = True
    if not _place_valid_inputs(h, w, objects, occupied):
        return None
    if not _place_invalid_inputs(h, w, objects, occupied):
        return None

    inp = _build_input_grid(h, w, markers, objects)
    return inp if _validate_generated_case(inp, expected, objects, markers) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
