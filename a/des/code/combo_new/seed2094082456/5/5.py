import numpy as np


BG = 0
ARROW = 2
CONTAINER = 5
OBJECT_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400

DIR_VECS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

ARROW_PATTERNS = {
    "right": {
        "mask": np.array(
            [
                [0, 0, 0, 1],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "tail": (1, 0),
        "front": (1, 2),
    },
    "left": {
        "mask": np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
            ],
            dtype=int,
        ),
        "tail": (1, 3),
        "front": (1, 1),
    },
    "down": {
        "mask": np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=int,
        ),
        "tail": (0, 1),
        "front": (2, 1),
    },
    "up": {
        "mask": np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        "tail": (3, 1),
        "front": (1, 1),
    },
}

BASE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (-1, 1), (0, 1), (1, 1)],
]


def _neighbors(cell: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = cell
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                comp.append(cell)
                for nr, nc in _neighbors(cell, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return mask, r0, c0


def _rotate_point(point: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = point
    if direction == "right":
        return r, c
    if direction == "left":
        return r, -c
    if direction == "down":
        return c, r
    return -c, r


def _shape_for_direction(base_shape: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    return sorted(_rotate_point(pt, direction) for pt in base_shape)


def _draw_arrow(grid: np.ndarray, tail: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    tr, tc = tail
    if direction == "right":
        cells = [(tr, tc), (tr, tc + 1), (tr, tc + 2), (tr - 1, tc + 3), (tr + 1, tc + 3)]
    elif direction == "left":
        cells = [(tr, tc), (tr, tc - 1), (tr, tc - 2), (tr - 1, tc - 3), (tr + 1, tc - 3)]
    elif direction == "down":
        cells = [(tr, tc), (tr + 1, tc), (tr + 2, tc), (tr + 3, tc - 1), (tr + 3, tc + 1)]
    else:
        cells = [(tr, tc), (tr - 1, tc), (tr - 2, tc), (tr - 3, tc - 1), (tr - 3, tc + 1)]
    for r, c in cells:
        grid[r, c] = ARROW
    return cells


def _paint_container(
    grid: np.ndarray,
    edge_direction: str,
    start: int,
    cross_size: int,
    depth: int,
) -> dict:
    h, w = grid.shape
    if edge_direction == "right":
        r0 = start
        c0 = 0
        r1 = r0 + cross_size + 1
        c1 = depth
        grid[r0, c0 : c1 + 1] = CONTAINER
        grid[r1, c0 : c1 + 1] = CONTAINER
        grid[r0 : r1 + 1, c0] = CONTAINER
        opening = [(r, c1) for r in range(r0 + 1, r1)]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1 + 1)]
    elif edge_direction == "left":
        r0 = start
        c1 = w - 1
        c0 = c1 - depth
        r1 = r0 + cross_size + 1
        grid[r0, c0 : c1 + 1] = CONTAINER
        grid[r1, c0 : c1 + 1] = CONTAINER
        grid[r0 : r1 + 1, c1] = CONTAINER
        opening = [(r, c0) for r in range(r0 + 1, r1)]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0, c1)]
    elif edge_direction == "down":
        r0 = 0
        c0 = start
        r1 = depth
        c1 = c0 + cross_size + 1
        grid[r0 : r1 + 1, c0] = CONTAINER
        grid[r0 : r1 + 1, c1] = CONTAINER
        grid[r0, c0 : c1 + 1] = CONTAINER
        opening = [(r1, c) for c in range(c0 + 1, c1)]
        interior = [(r, c) for r in range(r0 + 1, r1 + 1) for c in range(c0 + 1, c1)]
    else:
        r1 = h - 1
        c0 = start
        r0 = r1 - depth
        c1 = c0 + cross_size + 1
        grid[r0 : r1 + 1, c0] = CONTAINER
        grid[r0 : r1 + 1, c1] = CONTAINER
        grid[r1, c0 : c1 + 1] = CONTAINER
        opening = [(r0, c) for c in range(c0 + 1, c1)]
        interior = [(r, c) for r in range(r0, r1) for c in range(c0 + 1, c1)]

    return {
        "direction": edge_direction,
        "opening": opening,
        "opening_set": set(opening),
        "interior_set": set(interior),
        "bbox": (r0, c0, r1, c1),
    }


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _components(grid == ARROW, connectivity=8):
        if len(comp) != 5:
            continue
        mask, r0, c0 = _normalize(comp)
        for direction, spec in ARROW_PATTERNS.items():
            if mask.shape == spec["mask"].shape and np.array_equal(mask, spec["mask"]):
                tail = (r0 + spec["tail"][0], c0 + spec["tail"][1])
                front = (r0 + spec["front"][0], c0 + spec["front"][1])
                arrows.append(
                    {
                        "direction": direction,
                        "tail": tail,
                        "front": front,
                        "cells": comp,
                    }
                )
                break
    arrows.sort(key=lambda a: (a["tail"][0], a["tail"][1], a["direction"]))
    return arrows


def _find_containers(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = grid.shape
    containers = []
    opening_lookup: dict[tuple[int, int], int] = {}
    for comp in _components(grid == CONTAINER, connectivity=4):
        mask, r0, c0 = _normalize(comp)
        r1 = r0 + mask.shape[0] - 1
        c1 = c0 + mask.shape[1] - 1
        rec = None
        if c0 == 0 and mask.shape[0] >= 3 and mask.shape[1] >= 2:
            expected = np.zeros_like(mask)
            expected[0, :] = 1
            expected[-1, :] = 1
            expected[:, 0] = 1
            if np.array_equal(mask, expected):
                opening = [(r, c1) for r in range(r0 + 1, r1)]
                interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1 + 1)]
                rec = {
                    "direction": "right",
                    "opening": opening,
                    "opening_set": set(opening),
                    "interior_set": set(interior),
                    "bbox": (r0, c0, r1, c1),
                }
        elif c1 == w - 1 and mask.shape[0] >= 3 and mask.shape[1] >= 2:
            expected = np.zeros_like(mask)
            expected[0, :] = 1
            expected[-1, :] = 1
            expected[:, -1] = 1
            if np.array_equal(mask, expected):
                opening = [(r, c0) for r in range(r0 + 1, r1)]
                interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0, c1)]
                rec = {
                    "direction": "left",
                    "opening": opening,
                    "opening_set": set(opening),
                    "interior_set": set(interior),
                    "bbox": (r0, c0, r1, c1),
                }
        elif r0 == 0 and mask.shape[0] >= 2 and mask.shape[1] >= 3:
            expected = np.zeros_like(mask)
            expected[:, 0] = 1
            expected[:, -1] = 1
            expected[0, :] = 1
            if np.array_equal(mask, expected):
                opening = [(r1, c) for c in range(c0 + 1, c1)]
                interior = [(r, c) for r in range(r0 + 1, r1 + 1) for c in range(c0 + 1, c1)]
                rec = {
                    "direction": "down",
                    "opening": opening,
                    "opening_set": set(opening),
                    "interior_set": set(interior),
                    "bbox": (r0, c0, r1, c1),
                }
        elif r1 == h - 1 and mask.shape[0] >= 2 and mask.shape[1] >= 3:
            expected = np.zeros_like(mask)
            expected[:, 0] = 1
            expected[:, -1] = 1
            expected[-1, :] = 1
            if np.array_equal(mask, expected):
                opening = [(r0, c) for c in range(c0 + 1, c1)]
                interior = [(r, c) for r in range(r0, r1) for c in range(c0 + 1, c1)]
                rec = {
                    "direction": "up",
                    "opening": opening,
                    "opening_set": set(opening),
                    "interior_set": set(interior),
                    "bbox": (r0, c0, r1, c1),
                }
        if rec is None:
            continue
        cid = len(containers)
        containers.append(rec)
        for cell in rec["opening"]:
            opening_lookup[cell] = cid
    return containers, opening_lookup


def _find_objects(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    objects = []
    lookup: dict[tuple[int, int], int] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, ARROW, CONTAINER)):
        for comp in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            oid = len(objects)
            rec = {
                "color": color,
                "cells": comp,
                "bbox": (r0, c0, r1, c1),
            }
            objects.append(rec)
            for cell in comp:
                lookup[cell] = oid
    return objects, lookup


def _scan_target(
    grid: np.ndarray,
    arrow: dict,
    object_lookup: dict[tuple[int, int], int],
) -> tuple[int | None, tuple[int, int] | None, int | None]:
    h, w = grid.shape
    dr, dc = DIR_VECS[arrow["direction"]]
    r, c = arrow["front"]
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None, None, None
        steps += 1
        v = int(grid[r, c])
        if v in (BG, ARROW, CONTAINER):
            continue
        oid = object_lookup.get((r, c))
        if oid is None:
            return None, None, None
        return oid, (r, c), steps


def _scan_container(
    grid: np.ndarray,
    arrow: dict,
    opening_lookup: dict[tuple[int, int], int],
) -> tuple[int | None, tuple[int, int] | None]:
    h, w = grid.shape
    dr, dc = DIR_VECS[arrow["direction"]]
    r, c = arrow["tail"]
    while True:
        r -= dr
        c -= dc
        if not (0 <= r < h and 0 <= c < w):
            return None, None
        cid = opening_lookup.get((r, c))
        if cid is not None:
            return cid, (r, c)


def _collect_transfers(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    arrows = _find_arrows(grid)
    containers, opening_lookup = _find_containers(grid)
    objects, object_lookup = _find_objects(grid)
    transfers = []
    for arrow in arrows:
        oid, hit_cell, distance = _scan_target(grid, arrow, object_lookup)
        cid, opening_cell = _scan_container(grid, arrow, opening_lookup)
        if oid is None or cid is None or hit_cell is None or opening_cell is None or distance is None:
            continue
        transfers.append(
            {
                "arrow": arrow,
                "object_id": oid,
                "container_id": cid,
                "hit_cell": hit_cell,
                "opening_cell": opening_cell,
                "distance": distance,
            }
        )
    return arrows, containers, objects, transfers


def _apply_transfers(grid: np.ndarray, arrows: list[dict], containers: list[dict], objects: list[dict], transfers: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG
    moved_object_ids = {transfer["object_id"] for transfer in transfers}
    for oid in moved_object_ids:
        for r, c in objects[oid]["cells"]:
            out[r, c] = BG

    by_container: dict[int, list[dict]] = {}
    for transfer in transfers:
        by_container.setdefault(transfer["container_id"], []).append(transfer)

    for cid, items in by_container.items():
        container = containers[cid]
        items.sort(
            key=lambda item: (
                item["distance"],
                item["opening_cell"][0],
                item["opening_cell"][1],
                item["object_id"],
            )
        )
        depth = 0
        for item in items:
            obj = objects[item["object_id"]]
            hit_r, hit_c = item["hit_cell"]
            open_r, open_c = item["opening_cell"]
            r0, c0, r1, c1 = obj["bbox"]
            direction = container["direction"]
            if direction == "right":
                dr = open_r - hit_r
                dc = open_c - depth - c1
                extent = c1 - c0 + 1
            elif direction == "left":
                dr = open_r - hit_r
                dc = open_c + depth - c0
                extent = c1 - c0 + 1
            elif direction == "down":
                dr = open_r - depth - r1
                dc = open_c - hit_c
                extent = r1 - r0 + 1
            else:
                dr = open_r + depth - r0
                dc = open_c - hit_c
                extent = r1 - r0 + 1

            translated = [(r + dr, c + dc) for r, c in obj["cells"]]
            if not all(cell in container["interior_set"] for cell in translated):
                continue
            for (src_r, src_c), (dst_r, dst_c) in zip(obj["cells"], translated):
                out[dst_r, dst_c] = int(grid[src_r, src_c])
            depth += extent + 1

    return out


def _expected_from_specs(grid: np.ndarray, container: dict, items: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for item in items:
        for r, c in item["arrow_cells"]:
            out[r, c] = BG
        for r, c in item["object_cells"]:
            out[r, c] = BG

    depth = 0
    for item in sorted(items, key=lambda x: (x["distance"], x["opening_cell"][0], x["opening_cell"][1], x["color"])):
        rows = [r for r, _ in item["object_cells"]]
        cols = [c for _, c in item["object_cells"]]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        hit_r, hit_c = item["contact_cell"]
        open_r, open_c = item["opening_cell"]
        if container["direction"] == "right":
            dr = open_r - hit_r
            dc = open_c - depth - c1
            extent = c1 - c0 + 1
        elif container["direction"] == "left":
            dr = open_r - hit_r
            dc = open_c + depth - c0
            extent = c1 - c0 + 1
        elif container["direction"] == "down":
            dr = open_r - depth - r1
            dc = open_c - hit_c
            extent = r1 - r0 + 1
        else:
            dr = open_r + depth - r0
            dc = open_c - hit_c
            extent = r1 - r0 + 1

        translated = [(r + dr, c + dc) for r, c in item["object_cells"]]
        assert all(cell in container["interior_set"] for cell in translated)
        for r, c in translated:
            out[r, c] = item["color"]
        depth += extent + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the nearest object seen by each arrow into the container behind it, ordering copies by distance.

    Input structure:
    - The background is black.
    - Small colored objects are 4-connected components of 2 to 4 pixels.
    - Red arrows are 8-connected 5-pixel markers: a 3-pixel shaft plus two diagonal head pixels, pointing up/down/left/right.
    - Grey edge containers are open rectangular frames touching one border, with the open side facing the grid interior.

    Rule:
    - For each arrow, scan forward along its pointing row or column and take the first non-black, non-red, non-grey object cell encountered.
    - Scan backward from the arrow tail to find the first container opening on the same row or column.
    - Remove the arrow and the targeted object from their original locations.
    - Copy each targeted object into its matched container. Objects assigned to the same container keep the row/column where the arrow met them aligned with the opening, and they are packed inward with a one-cell gap in increasing arrow-to-object distance.

    Key insight:
    - Every arrow encodes both the source object and the destination container: the head chooses what to grab, while the tail direction chooses where the copy must be packed.
    """
    g = np.asarray(grid, dtype=int)
    arrows, containers, objects, transfers = _collect_transfers(g)
    return _apply_transfers(g, arrows, containers, objects, transfers)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    direction = rng.choice(["right", "left", "down", "up"])
    item_count = int(rng.integers(1, 4))
    base_shapes = [BASE_SHAPES[int(rng.integers(len(BASE_SHAPES)))] for _ in range(item_count)]
    shapes = [_shape_for_direction(shape, direction) for shape in base_shapes]
    distances = list(rng.choice(np.arange(2, 9), size=item_count, replace=False))
    back_gaps = [int(rng.integers(2, 5)) for _ in range(item_count)]
    colors = list(rng.choice(OBJECT_COLORS, size=item_count, replace=False))

    if direction in ("right", "left"):
        extents = [max(c for _, c in shape) - min(c for _, c in shape) + 1 for shape in shapes]
        cross_spans = [(min(r for r, _ in shape), max(r for r, _ in shape)) for shape in shapes]
        depth = sum(extents) + max(0, item_count - 1) + int(rng.integers(0, 3))
        extra_cross = int(rng.integers(0, 3))
        cross_size = 4 * item_count + 1 + extra_cross
        max_forward = max(
            gap + 2 + dist + max(c for _, c in shape)
            for gap, dist, shape in zip(back_gaps, distances, shapes)
        )
        min_h = cross_size + 4
        min_w = depth + max_forward + 2
        if min_h > 30 or min_w > 30:
            return None
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
        start = int(rng.integers(1, h - (cross_size + 2)))
        grid = np.zeros((h, w), dtype=int)
        container = _paint_container(grid, direction, start, cross_size, depth)
        slack = cross_size - (4 * item_count - 1)
        shift = int(rng.integers(0, slack + 1))
        lane_rels = [2 + shift + 4 * i for i in range(item_count)]
        rng.shuffle(lane_rels)
        occupied: set[tuple[int, int]] = set()
        items = []
        opening_col = container["bbox"][3] if direction == "right" else container["bbox"][1]

        for shape, color, dist, gap, lane_rel, (min_cross, max_cross) in zip(
            shapes, colors, distances, back_gaps, lane_rels, cross_spans
        ):
            open_r = container["bbox"][0] + lane_rel
            open_c = opening_col
            if open_r + min_cross < container["bbox"][0] + 1 or open_r + max_cross > container["bbox"][2] - 1:
                return None
            if direction == "right":
                tail = (open_r, open_c + gap)
                contact = (open_r, tail[1] + 2 + dist)
            else:
                tail = (open_r, open_c - gap)
                contact = (open_r, tail[1] - 2 - dist)
            object_cells = [(contact[0] + dr, contact[1] + dc) for dr, dc in shape]
            arrow_cells = []
            h_ok = True
            for r, c in object_cells:
                if not (0 <= r < h and 0 <= c < w):
                    h_ok = False
                    break
            if not h_ok:
                return None
            arrow_cells = _draw_arrow(grid, tail, direction)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
                return None
            for cell in arrow_cells:
                if cell in occupied or cell in container["interior_set"]:
                    return None
                occupied.add(cell)
            for cell in object_cells:
                if cell in occupied or int(grid[cell]) != BG:
                    return None
                occupied.add(cell)
            for r, c in object_cells:
                grid[r, c] = color
            items.append(
                {
                    "color": color,
                    "distance": dist,
                    "opening_cell": (open_r, open_c),
                    "contact_cell": contact,
                    "object_cells": sorted(object_cells),
                    "arrow_cells": sorted(arrow_cells),
                }
            )
    else:
        extents = [max(r for r, _ in shape) - min(r for r, _ in shape) + 1 for shape in shapes]
        cross_spans = [(min(c for _, c in shape), max(c for _, c in shape)) for shape in shapes]
        depth = sum(extents) + max(0, item_count - 1) + int(rng.integers(0, 3))
        extra_cross = int(rng.integers(0, 3))
        cross_size = 4 * item_count + 1 + extra_cross
        max_forward = max(
            gap + 2 + dist + max(r for r, _ in shape)
            for gap, dist, shape in zip(back_gaps, distances, shapes)
        )
        min_h = depth + max_forward + 2
        min_w = cross_size + 4
        if min_h > 30 or min_w > 30:
            return None
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
        start = int(rng.integers(1, w - (cross_size + 2)))
        grid = np.zeros((h, w), dtype=int)
        container = _paint_container(grid, direction, start, cross_size, depth)
        slack = cross_size - (4 * item_count - 1)
        shift = int(rng.integers(0, slack + 1))
        lane_rels = [2 + shift + 4 * i for i in range(item_count)]
        rng.shuffle(lane_rels)
        occupied: set[tuple[int, int]] = set()
        items = []
        opening_row = container["bbox"][2] if direction == "down" else container["bbox"][0]

        for shape, color, dist, gap, lane_rel, (min_cross, max_cross) in zip(
            shapes, colors, distances, back_gaps, lane_rels, cross_spans
        ):
            open_r = opening_row
            open_c = container["bbox"][1] + lane_rel
            if open_c + min_cross < container["bbox"][1] + 1 or open_c + max_cross > container["bbox"][3] - 1:
                return None
            if direction == "down":
                tail = (open_r + gap, open_c)
                contact = (tail[0] + 2 + dist, open_c)
            else:
                tail = (open_r - gap, open_c)
                contact = (tail[0] - 2 - dist, open_c)
            object_cells = [(contact[0] + dr, contact[1] + dc) for dr, dc in shape]
            v_ok = True
            for r, c in object_cells:
                if not (0 <= r < h and 0 <= c < w):
                    v_ok = False
                    break
            if not v_ok:
                return None
            arrow_cells = _draw_arrow(grid, tail, direction)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
                return None
            for cell in arrow_cells:
                if cell in occupied or cell in container["interior_set"]:
                    return None
                occupied.add(cell)
            for cell in object_cells:
                if cell in occupied or int(grid[cell]) != BG:
                    return None
                occupied.add(cell)
            for r, c in object_cells:
                grid[r, c] = color
            items.append(
                {
                    "color": color,
                    "distance": dist,
                    "opening_cell": (open_r, open_c),
                    "contact_cell": contact,
                    "object_cells": sorted(object_cells),
                    "arrow_cells": sorted(arrow_cells),
                }
            )

    expected = _expected_from_specs(grid, container, items)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
