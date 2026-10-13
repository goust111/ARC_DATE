from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

FOREGROUND_COLORS = (BLUE, RED, GREEN, YELLOW)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _touches_or_overlaps(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            return True
        if occupied[rr, cc]:
            return True
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _simulate_fall(grid: np.ndarray, *, return_records: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 4
    assert np.all(g[h - 1] == ORANGE), "expected a full-width orange bottom stripe"

    bg = _get_bg(g)
    components = _find_components(g, bg=bg)
    moving = []
    for comp in components:
        coords = comp["coords"]
        is_stripe = int(comp["color"]) == ORANGE and int(comp["area"]) == w and np.all(coords[:, 0] == h - 1)
        if not is_stripe:
            moving.append(comp)

    moving.sort(key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["color"])))

    out = g.copy()
    for comp in moving:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    stripe_mask = np.zeros((h, w), dtype=bool)
    stripe_mask[h - 1, :] = True
    settled_mask = stripe_mask.copy()
    settled_objects_mask = np.zeros((h, w), dtype=bool)

    records = []
    for comp in moving:
        coords = comp["coords"]
        area = int(comp["area"])
        actual_shift = area
        block_source = None

        for step in range(1, area + 1):
            candidate = coords.copy()
            candidate[:, 0] += step
            if _touches_or_overlaps(candidate, settled_mask):
                actual_shift = step - 1
                if _touches_or_overlaps(candidate, settled_objects_mask):
                    block_source = "object"
                else:
                    block_source = "stripe"
                break

        final_coords = coords.copy()
        final_coords[:, 0] += actual_shift
        final_color = int(comp["color"]) if actual_shift == area else GREY
        out[final_coords[:, 0], final_coords[:, 1]] = final_color
        settled_mask[final_coords[:, 0], final_coords[:, 1]] = True
        settled_objects_mask[final_coords[:, 0], final_coords[:, 1]] = True

        records.append(
            {
                "input_color": int(comp["color"]),
                "final_color": final_color,
                "area": area,
                "actual_shift": int(actual_shift),
                "blocked": bool(actual_shift < area),
                "block_source": block_source,
                "bbox": comp["bbox"],
                "final_bbox": _bbox(final_coords),
            }
        )

    if return_records:
        return out, records
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each foreground object downward by its area, but stop early before it would touch settled obstacles.

    Input structure:
    - The background is black, and the bottom row is a full-width orange stripe.
    - Above the stripe are 3-4 monochrome 4-connected foreground objects with distinct colors.
    - Each object's area is unique, so the processing order is determined by size.

    Rule:
    - Compute every foreground object's area and process the objects from smallest area to largest area.
    - Each object tries to move straight downward by exactly its own area in rows.
    - While testing that descent one row at a time, stop the object at the last safe row before it would become 4-neighbor-adjacent to an already settled object or to the orange stripe.
    - Objects that complete the full descent keep their color; objects that are stopped early are recolored grey. The stripe stays fixed.

    Key insight:
    - Area controls both the motion distance and the hidden execution order, and that order changes later collisions and therefore also changes the final colors.
    """
    return _simulate_fall(grid, return_records=False)


def _random_connected_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        height = int(rng.integers(1, min(3, area) + 1))
        min_width = max(1, (area + height - 1) // height)
        max_width = min(4, area)
        if min_width > max_width:
            continue
        width = int(rng.integers(min_width, max_width + 1))
        start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(0, len(frontier)))]
            cells.add(pick)
        if len(cells) == area:
            coords = np.array(sorted(cells), dtype=int)
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()
            return coords
    raise RuntimeError(f"failed to generate connected shape of area {area}")


def _shape_size(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _place(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    return shape + np.array([int(top), int(left)], dtype=int)


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            return False
        if occupied[rr, cc]:
            return False
        for dr, dc in NEIGHBORS8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _span_overlaps(a: tuple[int, int], b: tuple[int, int], *, gap: int = 0) -> bool:
    return not (a[1] + gap < b[0] or b[1] + gap < a[0])


def _has_shared_column(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    cols_a = set(map(int, coords_a[:, 1]))
    cols_b = set(map(int, coords_b[:, 1]))
    return bool(cols_a & cols_b)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 17))
    w = int(rng.integers(12, 16))
    count = int(rng.integers(3, 5))

    areas = sorted(map(int, rng.choice(np.arange(2, 7), size=count, replace=False)))
    colors = list(map(int, rng.choice(np.array(FOREGROUND_COLORS), size=count, replace=False)))
    shapes = [_random_connected_shape(area, rng) for area in areas]

    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = ORANGE
    occupied = np.zeros((h, w), dtype=bool)
    objects = []

    shape0 = shapes[0]
    height0, width0 = _shape_size(shape0)
    max_top0 = h - 3 - areas[0] - (height0 - 1)
    if max_top0 < 1:
        return None
    top0 = int(rng.integers(1, max_top0 + 1))
    left0 = int(rng.integers(0, w - width0 + 1))
    coords0 = _place(shape0, top0, left0)
    if not _can_place(coords0, occupied):
        return None
    _paint(grid, coords0, colors[0])
    occupied[coords0[:, 0], coords0[:, 1]] = True
    objects.append({"coords": coords0, "span": (left0, left0 + width0 - 1)})

    shape1 = shapes[1]
    height1, width1 = _shape_size(shape1)
    left1_min = max(0, left0 - width1 + 1)
    left1_max = min(w - width1, left0 + width0 - 1)
    if left1_min > left1_max:
        return None

    placed_second = False
    for _ in range(120):
        left1 = int(rng.integers(left1_min, left1_max + 1))
        top1_max = top0 - height1 - 1
        if top1_max < 0:
            break
        bias_low = max(0, top0 + areas[0] - areas[1] - (height1 - 1) - 1)
        top1_low = min(bias_low, top1_max)
        top1 = int(rng.integers(top1_low, top1_max + 1))
        coords1 = _place(shape1, top1, left1)
        if not _can_place(coords1, occupied):
            continue

        probe = np.full((h, w), BG, dtype=int)
        probe[h - 1, :] = ORANGE
        _paint(probe, coords0, colors[0])
        _paint(probe, coords1, colors[1])
        _, records = _simulate_fall(probe, return_records=True)
        by_color = {rec["input_color"]: rec for rec in records}
        rec0 = by_color[colors[0]]
        rec1 = by_color[colors[1]]
        if rec0["actual_shift"] != areas[0]:
            continue
        if rec1["block_source"] != "object":
            continue
        if rec1["actual_shift"] >= areas[1]:
            continue
        if not _has_shared_column(coords0, coords1):
            continue

        _paint(grid, coords1, colors[1])
        occupied[coords1[:, 0], coords1[:, 1]] = True
        objects.append({"coords": coords1, "span": (left1, left1 + width1 - 1)})
        placed_second = True
        break
    if not placed_second:
        return None

    for idx in range(2, count):
        shape = shapes[idx]
        height, width = _shape_size(shape)
        max_top = h - 3 - areas[idx] - (height - 1)
        if max_top < 0:
            return None

        candidates = []
        for left in range(0, w - width + 1):
            span = (left, left + width - 1)
            if any(_span_overlaps(span, placed["span"], gap=1) for placed in objects):
                continue
            candidates.append(left)
        if not candidates:
            return None

        placed = False
        for _ in range(120):
            left = int(candidates[int(rng.integers(0, len(candidates)))])
            top = int(rng.integers(0, max_top + 1))
            coords = _place(shape, top, left)
            if not _can_place(coords, occupied):
                continue
            _paint(grid, coords, colors[idx])
            occupied[coords[:, 0], coords[:, 1]] = True
            objects.append({"coords": coords, "span": (left, left + width - 1)})
            placed = True
            break
        if not placed:
            return None

    out, records = _simulate_fall(grid, return_records=True)
    records.sort(key=lambda rec: rec["area"])

    if sum(rec["blocked"] for rec in records) != 1:
        return None
    if records[0]["input_color"] != colors[0] or records[0]["actual_shift"] != areas[0]:
        return None
    if records[1]["input_color"] != colors[1] or records[1]["block_source"] != "object":
        return None
    if records[1]["actual_shift"] >= areas[1]:
        return None
    if records[1]["final_color"] != GREY:
        return None
    if not np.any(out == GREY):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
