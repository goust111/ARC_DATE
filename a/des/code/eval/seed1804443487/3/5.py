from __future__ import annotations

import random
from collections import deque

import numpy as np


Cell = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Cell]:
    h, w = shape
    out: list[Cell] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q: deque[Cell] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Cell] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, grid.shape):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = {rr for rr, _ in cells}
            cols = {cc for _, cc in cells}
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "rows": rows,
                    "cols": cols,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                }
            )
    return comps


def _min_object_distance(marker: Cell, cells: list[Cell]) -> int:
    mr, mc = marker
    return min(abs(mr - r) + abs(mc - c) for r, c in cells)


def _marker_side(marker: Cell, bbox: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = bbox
    gaps: dict[str, int] = {}
    if c < c0:
        gaps["left"] = c0 - c
    if c > c1:
        gaps["right"] = c - c1
    if r < r0:
        gaps["up"] = r0 - r
    if r > r1:
        gaps["down"] = r - r1
    assert gaps, "marker must lie outside its object's bbox"

    best_gap = max(gaps.values())
    best = [side for side, gap in gaps.items() if gap == best_gap]
    for side in ("left", "right", "up", "down"):
        if side in best:
            return side
    raise AssertionError("unreachable")


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    bg = _most_common_color(grid)
    comps = _components(grid, bg)
    objects = [comp for comp in comps if 5 <= int(comp["area"]) <= 8]
    markers = [comp for comp in comps if int(comp["area"]) == 1]
    assert objects, "expected at least one medium object"
    assert len(markers) == 2 * len(objects), "expected exactly two singleton markers per object"
    assert len(objects) + len(markers) == len(comps), "unexpected non-marker component sizes"

    assigned: dict[int, list[dict[str, object]]] = {idx: [] for idx in range(len(objects))}
    for marker in markers:
        cell = marker["cells"][0]
        dists = sorted(
            (_min_object_distance(cell, obj["cells"]), idx) for idx, obj in enumerate(objects)
        )
        assert len(dists) == 1 or dists[0][0] < dists[1][0], "marker must have a unique nearest object"
        assigned[dists[0][1]].append(marker)

    for idx in range(len(objects)):
        assert len(assigned[idx]) == 2, "each object must own exactly two markers"

    return bg, objects, assigned


def _overlap_scores(obj_idx: int, objects: list[dict[str, object]]) -> tuple[int, int]:
    rows = objects[obj_idx]["rows"]
    cols = objects[obj_idx]["cols"]
    row_score = 0
    col_score = 0
    for other_idx, other in enumerate(objects):
        if other_idx == obj_idx:
            continue
        row_score += len(rows & other["rows"])
        col_score += len(cols & other["cols"])
    return row_score, col_score


def _orientation(
    obj_idx: int,
    objects: list[dict[str, object]],
    markers_by_object: dict[int, list[dict[str, object]]],
) -> str:
    row_score, col_score = _overlap_scores(obj_idx, objects)
    if row_score > col_score:
        return "vertical"
    if col_score > row_score:
        return "horizontal"

    bbox = objects[obj_idx]["bbox"]
    sides = {_marker_side(marker["cells"][0], bbox) for marker in markers_by_object[obj_idx]}
    if sides == {"left", "right"}:
        return "vertical"
    if sides == {"up", "down"}:
        return "horizontal"
    raise AssertionError("ambiguous cut direction")


def _side_colors(
    bbox: tuple[int, int, int, int],
    markers: list[dict[str, object]],
    orientation: str,
) -> tuple[int, int]:
    by_side = {_marker_side(marker["cells"][0], bbox): int(marker["color"]) for marker in markers}

    if orientation == "vertical":
        if "left" in by_side and "right" in by_side:
            return by_side["left"], by_side["right"]
        left_marker, right_marker = sorted(markers, key=lambda marker: marker["cells"][0][1])
        return int(left_marker["color"]), int(right_marker["color"])

    if "up" in by_side and "down" in by_side:
        return by_side["up"], by_side["down"]
    up_marker, down_marker = sorted(markers, key=lambda marker: marker["cells"][0][0])
    return int(up_marker["color"]), int(down_marker["color"])


def _paint_split(
    out: np.ndarray,
    cells: list[Cell],
    bbox: tuple[int, int, int, int],
    orientation: str,
    first_color: int,
    second_color: int,
) -> None:
    if orientation == "vertical":
        _r0, c0, _r1, c1 = bbox
        mid = (c0 + c1) // 2
        first = [(r, c) for r, c in cells if c < mid]
        second = [(r, c) for r, c in cells if c > mid]
        on_line = [(r, c) for r, c in cells if c == mid]
    else:
        r0, _c0, r1, _c1 = bbox
        mid = (r0 + r1) // 2
        first = [(r, c) for r, c in cells if r < mid]
        second = [(r, c) for r, c in cells if r > mid]
        on_line = [(r, c) for r, c in cells if r == mid]

    if len(first) <= len(second):
        first.extend(on_line)
    else:
        second.extend(on_line)

    for r, c in first:
        out[r, c] = first_color
    for r, c in second:
        out[r, c] = second_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each medium object according to projection-overlap direction, recolor the two halves using the nearby markers, and erase the markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, objects, markers_by_object = _parse_scene(g)
    out = g.copy()

    for markers in markers_by_object.values():
        for marker in markers:
            r, c = marker["cells"][0]
            out[r, c] = bg

    for obj_idx, obj in enumerate(objects):
        orientation = _orientation(obj_idx, objects, markers_by_object)
        first_color, second_color = _side_colors(obj["bbox"], markers_by_object[obj_idx], orientation)
        _paint_split(out, obj["cells"], obj["bbox"], orientation, first_color, second_color)

    return out


# --------------- generator ---------------

GRID_MIN = 16
GRID_MAX = 28
MAX_GEN_ATTEMPTS = 400
MARKER_OFFSET = 2
PACK_GAP = 5
PALETTE = tuple(range(10))


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    if total == 0:
        return [0] * parts
    bars = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for bar in bars + [total + parts - 1]:
        out.append(bar - prev - 1)
        prev = bar
    return out


def _shape_connected(cells: set[Cell], h: int, w: int) -> bool:
    start = next(iter(cells))
    seen = {start}
    q: deque[Cell] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, (h, w)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _random_shape() -> np.ndarray:
    for _ in range(300):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        lo = max(5, box_h + box_w - 1)
        hi = min(8, box_h * box_w - 1)
        if lo > hi:
            continue
        target = random.randint(lo, hi)

        cells = {(random.randrange(box_h), random.randrange(box_w))}
        frontier = {
            nbr
            for nbr in _neighbors4(*next(iter(cells)), (box_h, box_w))
            if nbr not in cells
        }
        while len(cells) < target and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            for nbr in _neighbors4(*pick, (box_h, box_w)):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target:
            continue
        if not _shape_connected(cells, box_h, box_w):
            continue
        if any(all((r, c) not in cells for c in range(box_w)) for r in range(box_h)):
            continue
        if any(all((r, c) not in cells for r in range(box_h)) for c in range(box_w)):
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to build a shape")


def _shape_cells(mask: np.ndarray) -> list[Cell]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _build_spec(orientation: str, object_color: int, first_color: int, second_color: int) -> dict[str, object]:
    mask = _random_shape()
    h, w = mask.shape
    cells = _shape_cells(mask)

    if orientation == "vertical":
        rows = sorted({r for r, _ in cells})
        left_r = random.choice(rows)
        right_r = random.choice(rows)
        markers = [
            {"rel": (left_r, -MARKER_OFFSET), "side": "left", "color": first_color},
            {"rel": (right_r, w - 1 + MARKER_OFFSET), "side": "right", "color": second_color},
        ]
    else:
        cols = sorted({c for _, c in cells})
        up_c = random.choice(cols)
        down_c = random.choice(cols)
        markers = [
            {"rel": (-MARKER_OFFSET, up_c), "side": "up", "color": first_color},
            {"rel": (h - 1 + MARKER_OFFSET, down_c), "side": "down", "color": second_color},
        ]

    return {
        "mask": mask,
        "cells": cells,
        "height": h,
        "width": w,
        "orientation": orientation,
        "object_color": object_color,
        "markers": markers,
    }


def _paint_scene(shape: tuple[int, int], specs: list[dict[str, object]], placements: list[Cell], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    expected = np.full(shape, bg, dtype=int)

    for spec, (r0, c0) in zip(specs, placements):
        abs_cells = [(r0 + r, c0 + c) for r, c in spec["cells"]]
        for r, c in abs_cells:
            inp[r, c] = int(spec["object_color"])

        bbox = (r0, c0, r0 + int(spec["height"]) - 1, c0 + int(spec["width"]) - 1)
        side_colors: dict[str, int] = {}
        for marker in spec["markers"]:
            mr = r0 + int(marker["rel"][0])
            mc = c0 + int(marker["rel"][1])
            inp[mr, mc] = int(marker["color"])
            side_colors[str(marker["side"])] = int(marker["color"])

        if spec["orientation"] == "vertical":
            _paint_split(expected, abs_cells, bbox, "vertical", side_colors["left"], side_colors["right"])
        else:
            _paint_split(expected, abs_cells, bbox, "horizontal", side_colors["up"], side_colors["down"])

    return inp, expected


def _abs_cells(spec: dict[str, object], origin: Cell) -> set[Cell]:
    r0, c0 = origin
    cells = {(r0 + r, c0 + c) for r, c in spec["cells"]}
    for marker in spec["markers"]:
        mr = r0 + int(marker["rel"][0])
        mc = c0 + int(marker["rel"][1])
        cells.add((mr, mc))
    return cells


def _fits(shape: tuple[int, int], cells: set[Cell]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _expanded(cells: set[Cell]) -> set[Cell]:
    out: set[Cell] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _choose_vertical_positions(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
    c_min: int,
    c_max: int,
    r_min: int,
    r_max: int,
) -> list[Cell] | None:
    if not specs:
        return []
    max_h = max(int(spec["height"]) for spec in specs)
    if r_max - r_min + 1 < max_h:
        return None
    band_r = random.randint(r_min, r_max - max_h + 1)

    widths = [int(spec["width"]) for spec in specs]
    required = sum(widths) + PACK_GAP * (len(specs) - 1)
    usable = c_max - c_min + 1
    if usable < required:
        return None
    extras = _random_partition(usable - required, len(specs) + 1)

    placements: list[Cell] = []
    cur = c_min + extras[0]
    for idx, spec in enumerate(specs):
        placements.append((band_r, cur))
        cur += int(spec["width"])
        if idx + 1 < len(specs):
            cur += PACK_GAP + extras[idx + 1]
    return placements


def _choose_horizontal_positions(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
    r_min: int,
    r_max: int,
    c_min: int,
    c_max: int,
) -> list[Cell] | None:
    if not specs:
        return []
    max_w = max(int(spec["width"]) for spec in specs)
    if c_max - c_min + 1 < max_w:
        return None
    band_c = random.randint(c_min, c_max - max_w + 1)

    heights = [int(spec["height"]) for spec in specs]
    required = sum(heights) + PACK_GAP * (len(specs) - 1)
    usable = r_max - r_min + 1
    if usable < required:
        return None
    extras = _random_partition(usable - required, len(specs) + 1)

    placements: list[Cell] = []
    cur = r_min + extras[0]
    for idx, spec in enumerate(specs):
        placements.append((cur, band_c))
        cur += int(spec["height"])
        if idx + 1 < len(specs):
            cur += PACK_GAP + extras[idx + 1]
    return placements


def _valid_positions(shape: tuple[int, int], specs: list[dict[str, object]], placements: list[Cell]) -> bool:
    blocked: set[Cell] = set()
    for spec, origin in zip(specs, placements):
        cells = _abs_cells(spec, origin)
        if not _fits(shape, cells):
            return False
        if cells & blocked:
            return False
        blocked |= _expanded(cells)
    return True


def _make_specs(mode: str) -> list[dict[str, object]]:
    color_pool = list(PALETTE)
    bg = random.choice(color_pool)
    color_pool.remove(bg)

    if mode == "mixed":
        pattern = ["vertical", "vertical", "horizontal", "horizontal"]
    elif mode == "vertical":
        pattern = ["vertical"] * random.randint(2, 3)
    else:
        pattern = ["horizontal"] * random.randint(2, 3)

    random.shuffle(color_pool)
    object_colors = color_pool[: len(pattern)]
    remaining = [color for color in PALETTE if color != bg]

    specs: list[dict[str, object]] = []
    for orientation, object_color in zip(pattern, object_colors):
        marker_colors = random.sample([color for color in remaining if color != object_color], 2)
        first_color, second_color = marker_colors
        specs.append(_build_spec(orientation, object_color, first_color, second_color))
    return [{"bg": bg}] + specs


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("vertical", "horizontal", "mixed"))
    if mode == "mixed":
        h = random.randint(20, GRID_MAX)
        w = random.randint(20, GRID_MAX)
    else:
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)

    specs = _make_specs(mode)
    bg = int(specs[0]["bg"])
    objects = specs[1:]

    if mode == "mixed":
        vertical_specs = [spec for spec in objects if spec["orientation"] == "vertical"]
        horizontal_specs = [spec for spec in objects if spec["orientation"] == "horizontal"]

        v_positions = _choose_vertical_positions((h, w), vertical_specs, 2, w // 2 - 3, 1, h // 3)
        h_positions = _choose_horizontal_positions((h, w), horizontal_specs, h // 2, h - 3, w // 2 + 2, w - 3)
        if v_positions is None or h_positions is None:
            return None
        specs_in_order = vertical_specs + horizontal_specs
        placements = v_positions + h_positions
    elif mode == "vertical":
        placements = _choose_vertical_positions((h, w), objects, 2, w - 3, 1, h - 4)
        if placements is None:
            return None
        specs_in_order = objects
    else:
        placements = _choose_horizontal_positions((h, w), objects, 1, h - 4, 2, w - 3)
        if placements is None:
            return None
        specs_in_order = objects

    if not _valid_positions((h, w), specs_in_order, placements):
        return None

    inp, expected = _paint_scene((h, w), specs_in_order, placements, bg)
    try:
        parsed_bg, parsed_objects, markers_by_object = _parse_scene(inp)
    except AssertionError:
        return None
    if parsed_bg != bg or len(parsed_objects) != len(specs_in_order):
        return None

    branches = {spec["orientation"] for spec in specs_in_order}
    seen_branches = {
        _orientation(obj_idx, parsed_objects, markers_by_object) for obj_idx in range(len(parsed_objects))
    }
    if seen_branches != branches and not (mode != "mixed" and len(seen_branches) == 1 and next(iter(seen_branches)) == mode):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
