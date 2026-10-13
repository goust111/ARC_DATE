from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4)
LONG_SIDES = (5, 7, 9, 11)
GEN_MARGIN = 1
GEN_CLEARANCE = 2
GEN_GRID_MIN = 15
GEN_GRID_MAX = 21
GEN_MAX_ATTEMPTS = 500
GEN_CORE_ATTEMPTS = 200
GEN_EXTRA_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _sorted_rectangles(rectangles: list[dict]) -> list[dict]:
    return sorted(rectangles, key=lambda rect: tuple(rect["bbox"]))


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rectangles: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            r0 = int(coord_array[:, 0].min())
            c0 = int(coord_array[:, 1].min())
            r1 = int(coord_array[:, 0].max())
            c1 = int(coord_array[:, 1].max())
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            assert len(coords) == box_h * box_w
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)

            short_side, long_side = sorted((box_h, box_w))
            assert short_side == 3
            assert long_side >= 5 and long_side % 2 == 1

            rectangles.append({"bbox": (r0, c0, r1, c1), "color": color})

    assert rectangles
    return _sorted_rectangles(rectangles)


def _hole_cells(rect: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_w > box_h:
        center_r = r0 + box_h // 2
        return [(center_r, c) for c in range(c0 + 1, c1, 2)]
    center_c = c0 + box_w // 2
    return [(r, center_c) for r in range(r0 + 1, r1, 2)]


def _ray_specs(rectangles: list[dict]) -> list[dict]:
    rays: list[dict] = []
    for rect_id, rect in enumerate(_sorted_rectangles(rectangles)):
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_w > box_h:
            for hole_r, hole_c in _hole_cells(rect):
                rays.append(
                    {
                        "rect_id": rect_id,
                        "color": color,
                        "start": (r0 - 1, hole_c),
                        "direction": (-1, 0),
                    }
                )
                rays.append(
                    {
                        "rect_id": rect_id,
                        "color": color,
                        "start": (r1 + 1, hole_c),
                        "direction": (1, 0),
                    }
                )
        else:
            for hole_r, hole_c in _hole_cells(rect):
                rays.append(
                    {
                        "rect_id": rect_id,
                        "color": color,
                        "start": (hole_r, c0 - 1),
                        "direction": (0, -1),
                    }
                )
                rays.append(
                    {
                        "rect_id": rect_id,
                        "color": color,
                        "start": (hole_r, c1 + 1),
                        "direction": (0, 1),
                    }
                )
    return rays


def _draw_rectangles(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _emit_ray(
    out: np.ndarray,
    owner: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> tuple[str, int]:
    r, c = start
    dr, dc = direction
    length = 0

    while _in_bounds(out.shape, r, c) and int(out[r, c]) == BLACK:
        out[r, c] = int(color)
        length += 1
        nr = r + dr
        nc = c + dc
        if not _in_bounds(out.shape, nr, nc):
            return "boundary", length
        if int(out[nr, nc]) != BLACK:
            return ("rectangle" if int(owner[nr, nc]) >= 0 else "ray"), length
        r, c = nr, nc

    if not _in_bounds(out.shape, r, c):
        return "boundary", 0
    return ("rectangle" if int(owner[r, c]) >= 0 else "ray"), 0


def _simulate_from_rectangles(
    shape: tuple[int, int], rectangles: list[dict]
) -> tuple[np.ndarray, dict[str, int | list[int]]]:
    rects = _sorted_rectangles(rectangles)
    out = _draw_rectangles(shape, rects)
    owner = np.full(shape, -1, dtype=int)
    for rect_id, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        owner[r0 : r1 + 1, c0 : c1 + 1] = rect_id

    for rect in rects:
        for hole_r, hole_c in _hole_cells(rect):
            out[hole_r, hole_c] = BLACK
            owner[hole_r, hole_c] = -1

    stats: dict[str, int | list[int]] = {
        "boundary_blocks": 0,
        "rectangle_blocks": 0,
        "ray_blocks": 0,
        "ray_lengths": [],
    }
    for ray in _ray_specs(rects):
        stop_kind, length = _emit_ray(
            out=out,
            owner=owner,
            start=ray["start"],
            direction=ray["direction"],
            color=int(ray["color"]),
        )
        stats["ray_lengths"].append(length)
        key = f"{stop_kind}_blocks"
        stats[key] = int(stats[key]) + 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Punch alternating center-line holes through each 3-wide rectangle and emit perpendicular rays until they hit an obstacle.

    Input structure
    - The grid has a black background and several separated monochromatic rectangles.
    - Every rectangle has short side length 3 and odd long side length at least 5.
    - Rectangles may be horizontal or vertical.

    Rule
    - For each rectangle, erase every other cell on its center line, starting one cell in from an end of the long axis.
    - Each erased hole becomes a launch point for two rays perpendicular to the long axis, starting from the first exterior cells outside the rectangle.
    - Extend each ray step by step until the next step would leave the grid or enter a non-black cell in the current output.
    - Keep the punched rectangles in place and draw the rays in the rectangle's own color.

    Key insight
    - The alternating hole pattern fixes the launch rows or columns, while the surrounding rectangle layout determines how far each perpendicular ray can travel before it is blocked.
    """
    g = np.asarray(grid, dtype=int)
    rectangles = _find_rectangles(g)
    assert np.array_equal(_draw_rectangles(g.shape, rectangles), g)
    out, _stats = _simulate_from_rectangles(g.shape, rectangles)
    return out


def _box_inside(shape: tuple[int, int], bbox: tuple[int, int, int, int], margin: int = GEN_MARGIN) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return margin <= r0 and margin <= c0 and r1 <= h - 1 - margin and c1 <= w - 1 - margin


def _boxes_clear(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    clearance: int = GEN_CLEARANCE,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + clearance < br0 or br1 + clearance < ar0 or ac1 + clearance < bc0 or bc1 + clearance < ac0


def _length_choices(limit: int) -> list[int]:
    return [length for length in LONG_SIDES if length <= limit]


def _sample_core_pair(shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    for _ in range(GEN_CORE_ATTEMPTS):
        gap = random.randint(2, 4)
        driver_horizontal = bool(random.getrandbits(1))

        if driver_horizontal:
            driver_lengths = _length_choices(w - 2 * GEN_MARGIN)
            blocker_lengths = _length_choices(h - 2 * GEN_MARGIN - 3 - gap)
            if not driver_lengths or not blocker_lengths:
                continue
            driver_len = random.choice(driver_lengths)
            blocker_len = random.choice(blocker_lengths)
            c0 = random.randint(GEN_MARGIN, w - GEN_MARGIN - driver_len)
            hole_offset = random.choice(list(range(1, driver_len - 1, 2)))
            hole_c = c0 + hole_offset
            blocker_c0 = hole_c - 1
            if blocker_c0 < GEN_MARGIN or blocker_c0 + 2 > w - 1 - GEN_MARGIN:
                continue

            if random.choice((True, False)):
                min_r0 = GEN_MARGIN + blocker_len + gap
                max_r0 = h - GEN_MARGIN - 3
                if min_r0 > max_r0:
                    continue
                driver_r0 = random.randint(min_r0, max_r0)
                blocker_r0 = driver_r0 - gap - blocker_len
            else:
                min_r0 = GEN_MARGIN
                max_r0 = h - GEN_MARGIN - 3 - gap - blocker_len
                if min_r0 > max_r0:
                    continue
                driver_r0 = random.randint(min_r0, max_r0)
                blocker_r0 = driver_r0 + 3 + gap

            driver_box = (driver_r0, c0, driver_r0 + 2, c0 + driver_len - 1)
            blocker_box = (blocker_r0, blocker_c0, blocker_r0 + blocker_len - 1, blocker_c0 + 2)
        else:
            driver_lengths = _length_choices(h - 2 * GEN_MARGIN)
            blocker_lengths = _length_choices(w - 2 * GEN_MARGIN - 3 - gap)
            if not driver_lengths or not blocker_lengths:
                continue
            driver_len = random.choice(driver_lengths)
            blocker_len = random.choice(blocker_lengths)
            r0 = random.randint(GEN_MARGIN, h - GEN_MARGIN - driver_len)
            hole_offset = random.choice(list(range(1, driver_len - 1, 2)))
            hole_r = r0 + hole_offset
            blocker_r0 = hole_r - 1
            if blocker_r0 < GEN_MARGIN or blocker_r0 + 2 > h - 1 - GEN_MARGIN:
                continue

            if random.choice((True, False)):
                min_c0 = GEN_MARGIN + blocker_len + gap
                max_c0 = w - GEN_MARGIN - 3
                if min_c0 > max_c0:
                    continue
                driver_c0 = random.randint(min_c0, max_c0)
                blocker_c0 = driver_c0 - gap - blocker_len
            else:
                min_c0 = GEN_MARGIN
                max_c0 = w - GEN_MARGIN - 3 - gap - blocker_len
                if min_c0 > max_c0:
                    continue
                driver_c0 = random.randint(min_c0, max_c0)
                blocker_c0 = driver_c0 + 3 + gap

            driver_box = (r0, driver_c0, r0 + driver_len - 1, driver_c0 + 2)
            blocker_box = (blocker_r0, blocker_c0, blocker_r0 + 2, blocker_c0 + blocker_len - 1)

        if not _box_inside(shape, driver_box) or not _box_inside(shape, blocker_box):
            continue
        if not _boxes_clear(driver_box, blocker_box):
            continue
        return [{"bbox": driver_box}, {"bbox": blocker_box}]

    return None


def _sample_extra_rectangles(
    shape: tuple[int, int], existing: list[dict], need: int
) -> list[dict] | None:
    extras: list[dict] = []
    boxes = [tuple(rect["bbox"]) for rect in existing]
    h, w = shape

    for _ in range(GEN_EXTRA_ATTEMPTS):
        if len(extras) == need:
            return extras

        horizontal = bool(random.getrandbits(1))
        if horizontal:
            lengths = _length_choices(w - 2 * GEN_MARGIN)
            if not lengths:
                continue
            long_side = random.choice(lengths)
            r0 = random.randint(GEN_MARGIN, h - GEN_MARGIN - 3)
            c0 = random.randint(GEN_MARGIN, w - GEN_MARGIN - long_side)
            bbox = (r0, c0, r0 + 2, c0 + long_side - 1)
        else:
            lengths = _length_choices(h - 2 * GEN_MARGIN)
            if not lengths:
                continue
            long_side = random.choice(lengths)
            r0 = random.randint(GEN_MARGIN, h - GEN_MARGIN - long_side)
            c0 = random.randint(GEN_MARGIN, w - GEN_MARGIN - 3)
            bbox = (r0, c0, r0 + long_side - 1, c0 + 2)

        if not _box_inside(shape, bbox):
            continue
        if all(_boxes_clear(bbox, other) for other in boxes):
            extras.append({"bbox": bbox})
            boxes.append(bbox)

    return None


def _assign_colors(rectangles: list[dict]) -> list[dict]:
    colors = random.sample(PALETTE, len(rectangles))
    colored: list[dict] = []
    for rect, color in zip(_sorted_rectangles(rectangles), colors):
        colored.append({"bbox": tuple(rect["bbox"]), "color": color})
    return colored


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GEN_GRID_MIN, GEN_GRID_MAX),
        random.randint(GEN_GRID_MIN, GEN_GRID_MAX),
    )
    target_rectangles = random.choice((2, 2, 3, 4))
    core = _sample_core_pair(shape)
    if core is None:
        return None

    extras_needed = target_rectangles - len(core)
    extras = _sample_extra_rectangles(shape, core, extras_needed)
    if extras is None:
        return None

    rectangles = _assign_colors(core + extras)
    input_grid = _draw_rectangles(shape, rectangles)
    expected, stats = _simulate_from_rectangles(shape, rectangles)

    orientations = {
        "horizontal" if rect["bbox"][3] - rect["bbox"][1] > rect["bbox"][2] - rect["bbox"][0] else "vertical"
        for rect in rectangles
    }
    if orientations != {"horizontal", "vertical"}:
        return None
    if int(stats["rectangle_blocks"]) < 1:
        return None
    if int(stats["boundary_blocks"]) < 1:
        return None
    if int(stats["ray_blocks"]) > 2:
        return None
    if min(stats["ray_lengths"]) <= 0:
        return None
    if max(stats["ray_lengths"]) < 2:
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 5")
