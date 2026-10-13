import random

import numpy as np


_BG = 0
_AXIS_COLOR = 5
_PALETTE = tuple(color for color in range(10) if color not in (_BG, _AXIS_COLOR))

_SIZE_OPTIONS = (13, 15, 17, 19, 21, 23, 25, 27, 29)
_DIMS_OPTIONS = ((3, 3), (3, 5), (5, 3), (5, 5))
_MARKER_SIDES = ("top", "bottom", "left", "right")
_OBJECT_COUNT_RANGE = (2, 5)
_OVERLAP_PROB = 0.35
_MAX_GENERATION_ATTEMPTS = 500


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_axis(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color) and int(np.count_nonzero(grid == color)) == w:
            candidates.append(("row", r, color))

    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color) and int(np.count_nonzero(grid == color)) == h:
            candidates.append(("col", c, color))

    assert len(candidates) == 1, "expected exactly one full straight mirror axis"
    return candidates[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            parts.append(np.array(coords, dtype=int))

    return parts


def _reflect(kind: str, axis_index: int, r: int, c: int) -> tuple[int, int]:
    if kind == "row":
        return 2 * axis_index - r, c
    return r, 2 * axis_index - c


def _parse_stamp(grid: np.ndarray, coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    cell_colors = grid[rows, cols]
    colors, counts = np.unique(cell_colors, return_counts=True)

    assert len(colors) == 2, "each stamp must contain one body color and one marker color"
    marker_slots = np.flatnonzero(counts == 1)
    assert marker_slots.size == 1, "each stamp must have exactly one marker pixel"

    marker_color = int(colors[marker_slots[0]])
    body_color = int(colors[1 - marker_slots[0]])
    marker_idx = np.flatnonzero(cell_colors == marker_color)
    assert marker_idx.size == 1
    marker_rc = (int(rows[marker_idx[0]]), int(cols[marker_idx[0]]))

    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    assert 3 <= height <= 5 and 3 <= width <= 5
    assert height % 2 == 1 and width % 2 == 1, "stamp bbox must have a unique edge midpoint"

    marker_rel = (marker_rc[0] - r0, marker_rc[1] - c0)
    valid_midpoints = {
        (0, width // 2),
        (height - 1, width // 2),
        (height // 2, 0),
        (height // 2, width - 1),
    }
    assert marker_rel in valid_midpoints, "marker must sit at the midpoint of one bbox edge"

    ordered_coords = sorted((int(r), int(c)) for r, c in coords)
    return {
        "coords": ordered_coords,
        "marker_coord": marker_rc,
        "body_color": body_color,
        "marker_color": marker_color,
        "bbox": (r0, c0, r1, c1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror every marked stamp across the central grey axis and swap its two stamp colors in the reflected copy.

    Input structure:
    - The background is black.
    - A single grey one-pixel-wide row or column spans the entire grid and acts as the mirror axis.
    - Every other 4-connected foreground object is a small stamp with exactly two colors: one body color and one unique marker pixel.
    - The marker pixel lies at the midpoint of one side of the stamp's bounding box.

    Rule:
    - Keep the input unchanged.
    - Reflect each stamp across the grey axis onto the opposite side of the grid.
    - Preserve the reflected stamp shape, but recolor every reflected body pixel with the original marker color.
    - Recolor the reflected counterpart of the unique marker pixel with the original body color.
    - When a reflected copy lands on existing content, the reflected pixels overwrite what was there.

    Key insight:
    - The axis controls only the reflected position, while the unique edge marker controls the reflected color swap: mirror the geometry, then exchange the stamp's body and marker colors in the copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    axis_kind, axis_index, _axis_color = _find_axis(g, bg)

    stamp_mask = g != bg
    if axis_kind == "row":
        stamp_mask[axis_index, :] = False
    else:
        stamp_mask[:, axis_index] = False

    components = _components(stamp_mask)
    assert components, "expected at least one stamp"

    stamps = [_parse_stamp(g, comp) for comp in components]
    stamps.sort(key=lambda item: item["bbox"])

    out = g.copy()
    h, w = g.shape
    for stamp in stamps:
        for r, c in stamp["coords"]:
            rr, cc = _reflect(axis_kind, axis_index, r, c)
            assert 0 <= rr < h and 0 <= cc < w
            color = stamp["body_color"] if (r, c) == stamp["marker_coord"] else stamp["marker_color"]
            out[rr, cc] = color

    return out


def _side_midpoint(height: int, width: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, width // 2
    if side == "bottom":
        return height - 1, width // 2
    if side == "left":
        return height // 2, 0
    return height // 2, width - 1


def _side_cells(height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(width)]
    if side == "bottom":
        return [(height - 1, c) for c in range(width)]
    if side == "left":
        return [(r, 0) for r in range(height)]
    return [(r, width - 1) for r in range(height)]


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    path = [(r, c)]
    row_first = random.choice((True, False))
    for first_rows in (row_first, not row_first):
        if first_rows:
            while r != rr:
                r += 1 if rr > r else -1
                path.append((r, c))
        else:
            while c != cc:
                c += 1 if cc > c else -1
                path.append((r, c))
    return path


def _shape_frontier(selected: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in selected:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in selected:
                frontier.add((nr, nc))
    return list(frontier)


def _generate_local_shape(height: int, width: int, marker_side: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    marker_rel = _side_midpoint(height, width, marker_side)
    selected = {marker_rel, (height // 2, width // 2)}

    other_sides = [side for side in _MARKER_SIDES if side != marker_side]
    random.shuffle(other_sides)
    for side in other_sides:
        target = random.choice(_side_cells(height, width, side))
        attach = random.choice(tuple(selected))
        for cell in _random_path(target, attach):
            selected.add(cell)

    min_area = max(len(selected), (height * width + 1) // 2)
    target_area = random.randint(min_area, height * width)
    while len(selected) < target_area:
        frontier = _shape_frontier(selected, height, width)
        if not frontier:
            break
        selected.add(random.choice(frontier))

    coords = sorted(selected)
    return coords, marker_rel


def _sample_template(used_body_colors: set[int]) -> dict:
    height, width = random.choice(_DIMS_OPTIONS)
    marker_side = random.choice(_MARKER_SIDES)
    local_cells, marker_rel = _generate_local_shape(height, width, marker_side)

    body_choices = [color for color in _PALETTE if color not in used_body_colors]
    body_color = random.choice(body_choices or list(_PALETTE))
    marker_color = random.choice([color for color in _PALETTE if color != body_color])

    return {
        "height": height,
        "width": width,
        "local_cells": local_cells,
        "marker_rel": marker_rel,
        "body_color": body_color,
        "marker_color": marker_color,
    }


def _placement_sides(axis_kind: str) -> tuple[str, str]:
    return ("top", "bottom") if axis_kind == "row" else ("left", "right")


def _sample_top_left(
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis_index: int,
    side: str,
    height: int,
    width: int,
) -> tuple[int, int] | None:
    h, w = grid_shape

    if axis_kind == "row":
        if side == "top":
            row_lo, row_hi = 0, axis_index - height
        else:
            row_lo, row_hi = axis_index + 1, h - height
        col_lo, col_hi = 0, w - width
    else:
        row_lo, row_hi = 0, h - height
        if side == "left":
            col_lo, col_hi = 0, axis_index - width
        else:
            col_lo, col_hi = axis_index + 1, w - width

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _source_cells(obj: dict) -> list[tuple[int, int]]:
    r0, c0 = obj["top_left"]
    return [(r0 + r, c0 + c) for r, c in obj["local_cells"]]


def _marker_abs(obj: dict) -> tuple[int, int]:
    r0, c0 = obj["top_left"]
    mr, mc = obj["marker_rel"]
    return r0 + mr, c0 + mc


def _mirror_cells(obj: dict, axis_kind: str, axis_index: int) -> list[tuple[int, int]]:
    return [_reflect(axis_kind, axis_index, r, c) for r, c in _source_cells(obj)]


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_object(
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis_index: int,
    side: str,
    template: dict,
    blocked_sources: set[tuple[int, int]],
    blocked_final: set[tuple[int, int]],
) -> dict | None:
    for _ in range(80):
        top_left = _sample_top_left(
            grid_shape,
            axis_kind,
            axis_index,
            side,
            template["height"],
            template["width"],
        )
        if top_left is None:
            return None

        obj = dict(template)
        obj["top_left"] = top_left
        source = set(_source_cells(obj))
        mirror = set(_mirror_cells(obj, axis_kind, axis_index))
        if source & blocked_sources:
            continue
        if (source | mirror) & blocked_final:
            continue
        return obj

    return None


def _place_overlap_pair(
    grid_shape: tuple[int, int],
    axis_kind: str,
    axis_index: int,
    used_body_colors: set[int],
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]]] | None:
    first_side, second_side = _placement_sides(axis_kind)
    if random.choice((True, False)):
        first_side, second_side = second_side, first_side

    template_a = _sample_template(used_body_colors)
    obj_a = _place_object(grid_shape, axis_kind, axis_index, first_side, template_a, set(), set())
    if obj_a is None:
        return None

    source_a = set(_source_cells(obj_a))
    mirror_a = set(_mirror_cells(obj_a, axis_kind, axis_index))
    blocked_sources = _expand(source_a, grid_shape)

    for _ in range(120):
        template_b = _sample_template(used_body_colors | {template_a["body_color"]})
        top_left = _sample_top_left(
            grid_shape,
            axis_kind,
            axis_index,
            second_side,
            template_b["height"],
            template_b["width"],
        )
        if top_left is None:
            return None

        obj_b = dict(template_b)
        obj_b["top_left"] = top_left
        source_b = set(_source_cells(obj_b))
        if source_b & blocked_sources:
            continue

        overlap = len(source_b & mirror_a)
        max_overlap = min(2, len(source_b) - 1, len(mirror_a) - 1)
        if overlap < 1 or overlap > max_overlap:
            continue

        mirror_b = set(_mirror_cells(obj_b, axis_kind, axis_index))
        final_cells = source_a | mirror_a | source_b | mirror_b
        return [obj_a, obj_b], _expand(source_a | source_b, grid_shape), _expand(final_cells, grid_shape)

    return None


def _paint_input(grid_shape: tuple[int, int], axis_kind: str, axis_index: int, objects: list[dict]) -> np.ndarray:
    grid = np.full(grid_shape, _BG, dtype=int)
    if axis_kind == "row":
        grid[axis_index, :] = _AXIS_COLOR
    else:
        grid[:, axis_index] = _AXIS_COLOR

    for obj in objects:
        marker = _marker_abs(obj)
        for r, c in _source_cells(obj):
            grid[r, c] = obj["marker_color"] if (r, c) == marker else obj["body_color"]

    return grid


def _expected_output(inp: np.ndarray, axis_kind: str, axis_index: int, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        marker = _marker_abs(obj)
        for r, c in _source_cells(obj):
            rr, cc = _reflect(axis_kind, axis_index, r, c)
            out[rr, cc] = obj["body_color"] if (r, c) == marker else obj["marker_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(_SIZE_OPTIONS)
    width = random.choice(_SIZE_OPTIONS)
    axis_kind = random.choice(("row", "col"))
    axis_index = height // 2 if axis_kind == "row" else width // 2
    overlap_mode = random.random() < _OVERLAP_PROB

    min_objects, max_objects = _OBJECT_COUNT_RANGE
    object_count = random.randint(max(2, min_objects), max_objects)
    objects: list[dict] = []
    blocked_sources: set[tuple[int, int]] = set()
    blocked_final: set[tuple[int, int]] = set()
    used_body_colors: set[int] = set()

    if overlap_mode:
        pair = _place_overlap_pair((height, width), axis_kind, axis_index, used_body_colors)
        if pair is None:
            return None
        pair_objects, blocked_sources, blocked_final = pair
        objects.extend(pair_objects)
        used_body_colors.update(obj["body_color"] for obj in pair_objects)

    while len(objects) < object_count:
        template = _sample_template(used_body_colors)
        side = random.choice(_placement_sides(axis_kind))
        obj = _place_object(
            (height, width),
            axis_kind,
            axis_index,
            side,
            template,
            blocked_sources,
            blocked_final,
        )
        if obj is None:
            return None

        source = set(_source_cells(obj))
        mirror = set(_mirror_cells(obj, axis_kind, axis_index))
        blocked_sources |= _expand(source, (height, width))
        blocked_final |= _expand(source | mirror, (height, width))
        objects.append(obj)
        used_body_colors.add(obj["body_color"])

    inp = _paint_input((height, width), axis_kind, axis_index, objects)
    expected = _expected_output(inp, axis_kind, axis_index, objects)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-stamp input")
