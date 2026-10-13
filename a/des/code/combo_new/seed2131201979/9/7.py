from __future__ import annotations

import numpy as np


ARROW_COLOR = 5
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRECTION_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_ARROW_FRONT = {
    "up": frozenset({(0, 0), (0, 1), (0, 2)}),
    "down": frozenset({(2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (2, 0)}),
    "right": frozenset({(0, 2), (1, 2), (2, 2)}),
}
_STANDARD_MASKS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
_ARROW_MASKS = {
    "standard": _STANDARD_MASKS,
    "thick": {
        direction: frozenset(mask | ({(1, 0), (1, 2)} if direction in {"up", "down"} else {(0, 1), (2, 1)}))
        for direction, mask in _STANDARD_MASKS.items()
    },
    "hollow": {
        direction: frozenset(mask - {(1, 1)})
        for direction, mask in _STANDARD_MASKS.items()
    },
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _local_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        local[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return (r0, c0, r1, c1), local


def _spare_color(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    height, width = grid.shape
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    found = [(r, c, int(grid[r, c])) for r, c in corners if int(grid[r, c]) != bg]
    assert len(found) == 1, "expected exactly one non-background corner pixel"
    r, c, color = found[0]
    assert color != ARROW_COLOR, "spare color cannot be grey"
    assert np.count_nonzero(grid == color) == 1, "spare color must appear exactly once"
    return color, (r, c)


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == ARROW_COLOR)}
    assert grey_cells, "expected at least one grey arrow"
    height, width = grid.shape
    found: dict[frozenset[tuple[int, int]], dict] = {}

    for top in range(height - 2):
        for left in range(width - 2):
            local = {
                (r - top, c - left)
                for r, c in grey_cells
                if top <= r < top + 3 and left <= c < left + 3
            }
            if not local:
                continue
            local_frozen = frozenset(local)
            for kind, by_dir in _ARROW_MASKS.items():
                for direction, mask in by_dir.items():
                    if local_frozen != mask:
                        continue
                    cells = frozenset((top + dr, left + dc) for dr, dc in mask)
                    found[cells] = {
                        "kind": kind,
                        "direction": direction,
                        "top": top,
                        "left": left,
                        "cells": cells,
                    }

    arrows = sorted(found.values(), key=lambda item: (item["top"], item["left"]))
    coverage: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        for cell in arrow["cells"]:
            coverage[cell] = coverage.get(cell, 0) + 1
    assert set(coverage) == grey_cells, "every grey pixel must belong to a detected arrow"
    assert all(count == 1 for count in coverage.values()), "arrow patterns must not overlap"
    return arrows


def _match_arrow_to_object(
    grid: np.ndarray,
    arrow: dict,
    labels: np.ndarray,
    spare: int,
    bg: int,
) -> int:
    dr, dc = _DIRECTION_STEP[arrow["direction"]]
    matches: set[int] = set()
    top = int(arrow["top"])
    left = int(arrow["left"])

    for fr, fc in _ARROW_FRONT[arrow["direction"]]:
        r = top + fr + dr
        c = left + fc + dc
        while _in_bounds(grid.shape, r, c):
            label = int(labels[r, c])
            if label >= 0:
                matches.add(label)
                break
            cell = int(grid[r, c])
            if cell == bg:
                r += dr
                c += dc
                continue
            assert cell not in {ARROW_COLOR, spare}, "arrow line of sight must reach an object cleanly"
            break

    assert len(matches) == 1, "each arrow must point to exactly one object"
    return next(iter(matches))


def _paint_local(out: np.ndarray, top: int, left: int, local: np.ndarray, bg: int) -> None:
    for r, c in np.argwhere(local != bg):
        rr = top + int(r)
        cc = left + int(c)
        assert _in_bounds(out.shape, rr, cc), "transformed object must stay inside the grid"
        assert int(out[rr, cc]) == bg, "transformed objects must not overlap"
        out[rr, cc] = int(local[int(r), int(c)])


def _transform_local(local: np.ndarray, kind: str, bg: int, spare: int) -> np.ndarray:
    if kind == "standard":
        return np.fliplr(local)
    if kind == "thick":
        return np.repeat(np.repeat(local, 2, axis=0), 2, axis=1)
    assert kind == "hollow"
    out = local.copy()
    out[out != bg] = spare
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the arrow-coded transformation to each pointed object, then erase the arrows.

    Input structure:
    - The background is the dominant color, and grey pixels form arrow markers.
    - Every target object is a 4-connected non-grey foreground component.
    - Each arrow occupies a 3x3 box and points up, down, left, or right toward exactly one object.
    - One corner contains a single spare-color pixel whose color is unused by every object and arrow.

    Rule:
    - Detect each grey arrow and classify its variant by shape: standard, thick, or hollow.
    - Follow the arrow direction to the unique object it points at.
    - Standard arrows mirror the object left-right inside its own bounding box.
    - Thick arrows replace the object by its 2x enlargement, aligned to the original bounding-box top-left.
    - Hollow arrows recolor every pixel of the object to the spare color.
    - After all object transforms, erase every grey arrow back to background.

    Key insight:
    - The arrow does two jobs at once: its direction selects the target object, while its internal shaft pattern chooses which transformation to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    spare, _spare_pos = _spare_color(g, bg)
    arrows = _detect_arrows(g)

    object_mask = (g != bg) & (g != ARROW_COLOR) & (g != spare)
    components = _components(object_mask)
    assert components, "expected at least one target object"

    labels = np.full(g.shape, -1, dtype=int)
    objects: list[dict] = []
    for idx, coords in enumerate(components):
        bbox, local = _local_object(g, coords, bg)
        for r, c in coords:
            labels[int(r), int(c)] = idx
        objects.append({"coords": coords, "bbox": bbox, "local": local})

    matched = [_match_arrow_to_object(g, arrow, labels, spare, bg) for arrow in arrows]
    assert len(set(matched)) == len(matched), "different arrows must target different objects"
    assert set(matched) == set(range(len(objects))), "every object must be targeted by an arrow"

    out = g.copy()
    out[g == ARROW_COLOR] = bg
    for obj_idx in matched:
        coords = objects[obj_idx]["coords"]
        for r, c in coords:
            out[int(r), int(c)] = bg

    for arrow, obj_idx in zip(arrows, matched):
        obj = objects[obj_idx]
        r0, c0, _r1, _c1 = obj["bbox"]
        transformed = _transform_local(obj["local"], arrow["kind"], bg, spare)
        _paint_local(out, r0, c0, transformed, bg)

    return out


# --------------- generator ---------------

_PALETTE = tuple(range(10))
_GEN_ATTEMPTS = 300
_GRID_SIDE_RANGE = (22, 29)
_DIRECTIONS = tuple(_DIRECTION_STEP)


def _grow_shape(
    rng: np.random.Generator,
    raw_h: int,
    raw_w: int,
    area: int,
) -> set[tuple[int, int]] | None:
    start = (int(rng.integers(0, raw_h)), int(rng.integers(0, raw_w)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    return cells


def _sample_shape(
    rng: np.random.Generator,
    variant: str,
    color: int,
    bg: int,
) -> np.ndarray | None:
    max_dim = 3 if variant == "thick" else 5
    max_area = 6 if variant == "thick" else 8
    for _ in range(120):
        raw_h = int(rng.integers(2, max_dim + 1))
        raw_w = int(rng.integers(2, max_dim + 1))
        area_hi = min(max_area, raw_h * raw_w - 1)
        if area_hi < 3:
            continue
        area = int(rng.integers(3, area_hi + 1))
        cells = _grow_shape(rng, raw_h, raw_w, area)
        if cells is None:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        cropped = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in cropped) + 1
        w = max(c for _, c in cropped) + 1

        local = np.full((h, w), bg, dtype=int)
        for r, c in cropped:
            local[r, c] = color

        if variant == "standard" and np.array_equal(local != bg, np.fliplr(local != bg)):
            continue
        return local
    return None


def _zone_bounds(height: int, width: int) -> list[tuple[int, int, int, int]]:
    mid_r = height // 2
    mid_c = width // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, width - 2),
        (mid_r + 1, height - 2, 1, mid_c - 2),
        (mid_r + 1, height - 2, mid_c + 1, width - 2),
    ]


def _feasible_origin_interval(
    zone: tuple[int, int, int, int],
    local_shape: tuple[int, int],
    output_shape: tuple[int, int],
    direction: str,
    align: int,
    gap: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    zr0, zr1, zc0, zc1 = zone
    h, w = local_shape
    out_h, out_w = output_shape

    r_lo = zr0
    r_hi = zr1 - out_h + 1
    c_lo = zc0
    c_hi = zc1 - out_w + 1
    if r_lo > r_hi or c_lo > c_hi:
        return None

    if direction == "right":
        r_lo = max(r_lo, zr0 - align + 1)
        r_hi = min(r_hi, zr1 - align - 1)
        c_lo = max(c_lo, zc0 + gap + 3)
    elif direction == "left":
        r_lo = max(r_lo, zr0 - align + 1)
        r_hi = min(r_hi, zr1 - align - 1)
        c_hi = min(c_hi, zc1 - w - gap - 2)
    elif direction == "down":
        r_lo = max(r_lo, zr0 + gap + 3)
        c_lo = max(c_lo, zc0 - align + 1)
        c_hi = min(c_hi, zc1 - align - 1)
    else:
        assert direction == "up"
        r_hi = min(r_hi, zr1 - h - gap - 2)
        c_lo = max(c_lo, zc0 - align + 1)
        c_hi = min(c_hi, zc1 - align - 1)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _object_origin_to_arrow_top_left(
    r0: int,
    c0: int,
    local: np.ndarray,
    direction: str,
    align: int,
    gap: int,
) -> tuple[int, int]:
    h, w = local.shape
    if direction == "right":
        return r0 + align - 1, c0 - gap - 3
    if direction == "left":
        return r0 + align - 1, c0 + w + gap
    if direction == "down":
        return r0 - gap - 3, c0 + align - 1
    assert direction == "up"
    return r0 + h + gap, c0 + align - 1


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _local_cells(local: np.ndarray, top: int, left: int, bg: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(local != bg)
    }


def _place_object_in_zone(
    rng: np.random.Generator,
    zone: tuple[int, int, int, int],
    variant: str,
    color: int,
    bg: int,
    spare: int,
) -> dict | None:
    local = _sample_shape(rng, variant, color, bg)
    if local is None:
        return None

    transformed = _transform_local(local, variant, bg, spare)
    for _ in range(160):
        direction = str(rng.choice(_DIRECTIONS))
        gap = int(rng.integers(1, 3))
        occupied = np.argwhere(local != bg)
        if direction in {"left", "right"}:
            align = int(rng.choice(np.unique(occupied[:, 0])))
        else:
            align = int(rng.choice(np.unique(occupied[:, 1])))

        ranges = _feasible_origin_interval(
            zone,
            local.shape,
            transformed.shape,
            direction,
            align,
            gap,
        )
        if ranges is None:
            continue
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        arrow_top, arrow_left = _object_origin_to_arrow_top_left(r0, c0, local, direction, align, gap)

        object_cells = _local_cells(local, r0, c0, bg)
        arrow_cells = {(arrow_top + dr, arrow_left + dc) for dr, dc in _ARROW_MASKS[variant][direction]}
        output_cells = _local_cells(transformed, r0, c0, bg)
        return {
            "variant": variant,
            "direction": direction,
            "color": color,
            "object_top": r0,
            "object_left": c0,
            "arrow_top": arrow_top,
            "arrow_left": arrow_left,
            "local": local,
            "transformed": transformed,
            "object_cells": object_cells,
            "arrow_cells": arrow_cells,
            "output_cells": output_cells,
        }
    return None


def _render_example(
    height: int,
    width: int,
    bg: int,
    spare: int,
    spare_pos: tuple[int, int],
    placements: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)
    inp[spare_pos] = spare
    expected[spare_pos] = spare

    for item in placements:
        _paint_local(inp, item["object_top"], item["object_left"], item["local"], bg)
        _paint_cells(inp, item["arrow_cells"], ARROW_COLOR)
        _paint_local(expected, item["object_top"], item["object_left"], item["transformed"], bg)

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(_GRID_SIDE_RANGE[0], _GRID_SIDE_RANGE[1] + 1))
    width = int(rng.integers(_GRID_SIDE_RANGE[0], _GRID_SIDE_RANGE[1] + 1))
    zones = _zone_bounds(height, width)
    if min(zr1 - zr0 + 1 for zr0, zr1, _zc0, _zc1 in zones) < 9:
        return None
    if min(zc1 - zc0 + 1 for _zr0, _zr1, zc0, zc1 in zones) < 9:
        return None

    bg = int(rng.choice([color for color in _PALETTE if color != ARROW_COLOR]))
    spare = int(rng.choice([color for color in _PALETTE if color not in {bg, ARROW_COLOR}]))
    object_count = 3 + int(rng.integers(0, 2))

    variants = ["standard", "thick", "hollow"]
    if object_count == 4:
        variants.append(str(rng.choice(variants)))
    rng.shuffle(variants)

    object_colors = rng.choice(
        [color for color in _PALETTE if color not in {bg, ARROW_COLOR, spare}],
        size=object_count,
        replace=False,
    )
    zone_order = rng.permutation(len(zones))[:object_count]

    corners = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    ]
    spare_pos = corners[int(rng.integers(0, 4))]

    placements: list[dict] = []
    for zone_idx, variant, color in zip(zone_order, variants, object_colors):
        placed = _place_object_in_zone(rng, zones[int(zone_idx)], variant, int(color), bg, spare)
        if placed is None:
            return None
        placements.append(placed)

    # Keep the input visually clean and make the solver's assumptions explicit.
    occupied_input = {spare_pos}
    occupied_output = {spare_pos}
    for item in placements:
        all_input = item["object_cells"] | item["arrow_cells"]
        if any(cell in occupied_input for cell in all_input):
            return None
        if any(cell in occupied_output for cell in item["output_cells"]):
            return None
        if spare_pos in all_input or spare_pos in item["output_cells"]:
            return None
        occupied_input |= all_input
        occupied_output |= item["output_cells"]

    inp, expected = _render_example(height, width, bg, spare, spare_pos, placements)

    # No other corner may be occupied.
    for corner in corners:
        if corner != spare_pos and int(inp[corner]) != bg:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
