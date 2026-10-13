from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8
CARRIER_COLORS = {BLUE, TEAL}

SHAPES = ((2, 2), (1, 3))
GRID_H_RANGE = (18, 22)
GRID_W_RANGE = (16, 20)
CARRIER_COUNT_CHOICES = (3, 4)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _carrier_color(grid: np.ndarray, coords: np.ndarray) -> int:
    adjacent = set()
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                value = int(grid[nr, nc])
                if value != RED:
                    adjacent.add(value)
    carrier = adjacent & CARRIER_COLORS
    assert len(carrier) == 1, "each red object must have exactly one carrier color"
    return next(iter(carrier))


def _containing_region(base: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    marker = np.zeros(base.shape, dtype=bool)
    marker[coords[:, 0], coords[:, 1]] = True
    hits = []
    for region in _components(base == int(color)):
        if np.any(marker[region[:, 0], region[:, 1]]):
            hits.append(region)
    assert len(hits) == 1, "each red object must belong to exactly one carrier region"
    return hits[0]


def _slide_objects(grid: np.ndarray) -> list[dict]:
    red_objects = _components(grid == RED)
    assert red_objects, "expected at least one red object"

    base = np.array(grid, copy=True)
    objects: list[dict] = []
    for coords in red_objects:
        carrier = _carrier_color(grid, coords)
        base[coords[:, 0], coords[:, 1]] = carrier
        objects.append({"coords": coords.copy(), "carrier": carrier})

    occupied = np.zeros(grid.shape, dtype=bool)
    for obj in objects:
        coords = obj["coords"]
        carrier = int(obj["carrier"])
        r0, c0, r1, c1 = _bbox(coords)
        region = _containing_region(base, coords, carrier)
        reg_r0, reg_c0, reg_r1, reg_c1 = _bbox(region)
        del reg_r0, reg_r1
        width = int(c1 - c0 + 1)
        dst_c0 = int(reg_c0 if carrier == TEAL else reg_c1 - width + 1)

        moved = coords.copy()
        moved[:, 1] += dst_c0 - c0
        assert np.all(base[moved[:, 0], moved[:, 1]] == carrier), "slid object must stay inside its carrier"
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "slid objects must not overlap"
        occupied[moved[:, 0], moved[:, 1]] = True

        obj["moved"] = moved
        obj["row_key"] = int(r0)
    return objects


def _static_blocked(obstacle_mask: np.ndarray) -> np.ndarray:
    blocked = np.zeros_like(obstacle_mask, dtype=bool)
    blocked[-1, :] = True
    blocked |= obstacle_mask
    blocked[:-1, :] |= obstacle_mask[1:, :]
    return blocked


def _drop_slid_objects(shape: tuple[int, int], objects: list[dict], obstacle_mask: np.ndarray) -> list[np.ndarray]:
    blocked = _static_blocked(obstacle_mask)
    settled_mask = np.zeros(shape, dtype=bool)
    settled: list[np.ndarray] = []

    order = sorted(
        objects,
        key=lambda obj: (
            int(obj["moved"][:, 0].max()),
            int(obj["moved"][:, 0].min()),
            int(obj["moved"][:, 1].min()),
        ),
        reverse=True,
    )

    for obj in order:
        coords = obj["moved"].copy()
        while True:
            shifted = coords + np.array([1, 0], dtype=int)
            if int(shifted[:, 0].max()) >= shape[0]:
                break
            if np.any(blocked[shifted[:, 0], shifted[:, 1]]):
                break
            if np.any(settled_mask[shifted[:, 0], shifted[:, 1]]):
                break
            coords = shifted
        settled_mask[coords[:, 0], coords[:, 1]] = True
        settled.append(coords)
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each red object to a carrier edge, erase the carriers, then let the objects fall.

    Input structure:
    - The grid has a black background, several horizontal blue or teal carrier bands, red rectangular objects inside those carriers, and grey obstacle pixels near the bottom.
    - Every red object is fully contained in exactly one carrier band and has visible space on both its left and right inside the carrier.
    - Carrier rows lie above the grey obstacles and different carriers do not touch.

    Rule:
    - Identify the blue or teal carrier containing each red 4-connected object.
    - In a blue carrier, slide the red object right until its bounding box is flush with the carrier's right edge; in a teal carrier, slide it left until flush with the carrier's left edge.
    - Erase all blue and teal carrier pixels to black while keeping the red objects at their slid positions.
    - Drop the red objects straight downward from those slid positions. Grey obstacle pixels, the one-row cushion above each grey pixel, the bottom row, and previously settled red pixels block further falling.

    Key insight:
    - The carrier color determines the horizontal column where an object will lose support, and that new column determines where gravity places the final red object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    objects = _slide_objects(g)
    obstacle_mask = g == GREY
    settled = _drop_slid_objects(g.shape, objects, obstacle_mask)

    out = np.zeros_like(g)
    out[obstacle_mask] = GREY
    for coords in settled:
        out[coords[:, 0], coords[:, 1]] = RED
    return out


def _make_obstacles(h: int, w: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        segments: list[tuple[int, int, int]] = []
        c = random.randint(0, 1)
        while c < w:
            gap = random.randint(1, 3) if segments else random.randint(0, 2)
            c += gap
            if c >= w - 1:
                break
            seg_len = random.randint(2, min(5, w - c))
            seg_h = 2 if random.random() < 0.45 else 1
            segments.append((c, c + seg_len - 1, seg_h))
            c += seg_len
        if not (2 <= len(segments) <= 4):
            continue
        heights = {seg_h for _, _, seg_h in segments}
        if not heights.issuperset({1}) or 2 not in heights:
            continue
        covered = sum(c1 - c0 + 1 for c0, c1, _ in segments)
        if not (w // 3 <= covered <= (2 * w) // 3):
            continue
        for c0, c1, seg_h in segments:
            mask[h - seg_h :, c0 : c1 + 1] = True
        if np.any(mask[h - 3, :]) and np.any(~mask[h - 2, :]):
            return mask
        if np.any(~mask[h - 1, :]):
            return mask
    return None


def _static_top_blocked_rows(obstacle_mask: np.ndarray) -> np.ndarray:
    blocked = _static_blocked(obstacle_mask)
    tops = np.full(obstacle_mask.shape[1], obstacle_mask.shape[0] - 1, dtype=int)
    for c in range(obstacle_mask.shape[1]):
        rows = np.flatnonzero(blocked[:, c])
        tops[c] = int(rows[0])
    return tops


def _sample_carrier_rows(h: int, heights: list[int]) -> list[int] | None:
    max_bottom = h - 5
    total_needed = sum(heights) + (len(heights) - 1)
    if total_needed > max_bottom:
        return None
    extra = max_bottom - total_needed
    slots = [0] * (len(heights) + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1
    rows = []
    r = 1 + slots[0]
    for i, height in enumerate(heights):
        rows.append(r)
        r += height
        if i < len(heights) - 1:
            r += 1 + slots[i + 1]
    return rows


def _interval_overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _candidate_lefts(
    w: int,
    carrier_w: int,
    obj_w: int,
    color: int,
) -> list[int]:
    if color == TEAL:
        lo, hi = 0, w - carrier_w
    else:
        lo, hi = carrier_w - obj_w, w - obj_w
    return list(range(lo, hi + 1))


def _choose_destination_left(
    top_blocked: np.ndarray,
    w: int,
    carrier_w: int,
    obj_w: int,
    color: int,
    assigned: list[dict],
    want_obstacle: bool,
    allow_stack: bool,
) -> tuple[int, str] | None:
    feasible = _candidate_lefts(w, carrier_w, obj_w, color)
    random.shuffle(feasible)

    stack_targets = [
        item["dest_left"]
        for item in assigned
        if item["obj_w"] == obj_w and item["dest_left"] in feasible
    ]
    if allow_stack and stack_targets:
        return random.choice(stack_targets), "stack"

    good = []
    other = []
    for left in feasible:
        right = left + obj_w - 1
        if any(
            _interval_overlaps(left, right, item["dest_left"], item["dest_left"] + item["obj_w"] - 1)
            for item in assigned
        ):
            continue
        hits_obstacle = int(np.min(top_blocked[left : right + 1])) < int(top_blocked.shape[0] + 10) and int(
            np.min(top_blocked[left : right + 1])
        ) < int(top_blocked.max())
        all_bottom = bool(np.all(top_blocked[left : right + 1] == top_blocked.max()))
        if want_obstacle and hits_obstacle:
            good.append(left)
        elif (not want_obstacle) and all_bottom:
            good.append(left)
        else:
            other.append(left)
    if good:
        return random.choice(good), "obstacle" if want_obstacle else "bottom"
    if other:
        return random.choice(other), "obstacle-mixed" if want_obstacle else "bottom-mixed"
    return None


def _build_input() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    obstacle_mask = _make_obstacles(h, w)
    if obstacle_mask is None:
        return None
    top_blocked = _static_top_blocked_rows(obstacle_mask)
    bottom_only = [c for c in range(w) if int(top_blocked[c]) == h - 1]
    obstacle_cols = [c for c in range(w) if int(top_blocked[c]) < h - 1]
    if not bottom_only or not obstacle_cols:
        return None

    count = random.choice(CARRIER_COUNT_CHOICES)
    colors = [BLUE, TEAL]
    while len(colors) < count:
        colors.append(random.choice((BLUE, TEAL)))
    random.shuffle(colors)

    shapes = [random.choice(SHAPES) for _ in range(count)]
    if count >= 3 and random.random() < 0.6:
        same_width = random.choice((2, 3))
        idxs = random.sample(range(count), 2)
        for idx in idxs:
            shapes[idx] = (2, 2) if same_width == 2 else (1, 3)

    carrier_heights = [random.randint(2, 3) for _ in range(count)]
    carrier_widths = [random.randint(8, min(12, w)) for _ in range(count)]
    if any(carrier_widths[i] < shapes[i][1] + 2 for i in range(count)):
        return None

    top_rows = _sample_carrier_rows(h, carrier_heights)
    if top_rows is None:
        return None

    specs = []
    for i in range(count):
        specs.append(
            {
                "carrier_h": carrier_heights[i],
                "carrier_w": carrier_widths[i],
                "color": colors[i],
                "shape": shapes[i],
                "carrier_r0": top_rows[i],
            }
        )

    assigned: list[dict] = []
    seen_obstacle = False
    seen_bottom = False
    descending = sorted(range(count), key=lambda i: specs[i]["carrier_r0"], reverse=True)
    stack_budget = 1 if random.random() < 0.45 else 0

    for idx in descending:
        spec = specs[idx]
        obj_h, obj_w = spec["shape"]
        allow_stack = stack_budget > 0

        prefer_obstacle = (not seen_obstacle) or (seen_bottom and random.random() < 0.6)
        choice = _choose_destination_left(
            top_blocked,
            w,
            spec["carrier_w"],
            obj_w,
            spec["color"],
            assigned,
            want_obstacle=prefer_obstacle,
            allow_stack=allow_stack,
        )
        if choice is None:
            choice = _choose_destination_left(
                top_blocked,
                w,
                spec["carrier_w"],
                obj_w,
                spec["color"],
                assigned,
                want_obstacle=not prefer_obstacle,
                allow_stack=False,
            )
        if choice is None:
            return None

        dest_left, mode = choice
        if mode == "stack":
            stack_budget -= 1
        elif "bottom" in mode:
            seen_bottom = True
        else:
            seen_obstacle = True

        if spec["color"] == TEAL:
            carrier_c0 = dest_left
        else:
            carrier_c0 = dest_left - (spec["carrier_w"] - obj_w)
        if not (0 <= carrier_c0 <= w - spec["carrier_w"]):
            return None

        row_offset = random.randint(0, spec["carrier_h"] - obj_h)
        max_gap = spec["carrier_w"] - obj_w - 1
        gap = random.randint(1, max_gap)
        input_left = dest_left + gap if spec["color"] == TEAL else dest_left - gap
        if not (carrier_c0 + 1 <= input_left <= carrier_c0 + spec["carrier_w"] - obj_w - 1):
            return None

        spec["carrier_c0"] = carrier_c0
        spec["obj_r0"] = spec["carrier_r0"] + row_offset
        spec["dest_left"] = dest_left
        spec["input_left"] = input_left
        spec["obj_w"] = obj_w
        assigned.append({"dest_left": dest_left, "obj_w": obj_w})

    if not (seen_obstacle and seen_bottom):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[obstacle_mask] = GREY
    for spec in specs:
        r0 = spec["carrier_r0"]
        c0 = spec["carrier_c0"]
        h0 = spec["carrier_h"]
        w0 = spec["carrier_w"]
        color = spec["color"]
        if np.any(grid[r0 : r0 + h0, c0 : c0 + w0] != BG):
            return None
        grid[r0 : r0 + h0, c0 : c0 + w0] = color

    for spec in specs:
        obj_h, obj_w = spec["shape"]
        r0 = spec["obj_r0"]
        c0 = spec["input_left"]
        if np.any(grid[r0 : r0 + obj_h, c0 : c0 + obj_w] == GREY):
            return None
        grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = RED

    out = main(grid)
    if out.shape != grid.shape:
        return None

    total_red = sum(spec["shape"][0] * spec["shape"][1] for spec in specs)
    if int(np.count_nonzero(grid == RED)) != total_red:
        return None
    if int(np.count_nonzero(out == RED)) != total_red:
        return None
    if np.array_equal(out, grid):
        return None

    carriers = [spec["color"] for spec in specs]
    if BLUE not in carriers or TEAL not in carriers:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid carrier-slide gravity puzzle")
