from __future__ import annotations

import itertools
import random

import numpy as np


GREY = 5
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min_r, min_c, max_r, max_c),
                    "area": len(coords),
                }
            )
    return components


def _color_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    components = _find_components(grid == int(color))
    for comp in components:
        comp["color"] = int(color)
    return components


def _template_data(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], int]:
    grey_components = _color_components(grid, GREY)
    assert len(grey_components) == 1, "expected exactly one grey template object"
    template = grey_components[0]
    r0, c0, _r1, _c1 = template["bbox"]
    rel = {
        (int(r) - int(r0), int(c) - int(c0))
        for r, c in template["coords"]
    }
    assert rel, "template must contain pixels"
    return rel, (int(r0), int(c0)), int(template["area"])


def _candidate_offsets(
    template_rel: set[tuple[int, int]], object_coords: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    offsets: set[tuple[int, int]] = set()
    for obj_r, obj_c in object_coords:
        for tpl_r, tpl_c in template_rel:
            offsets.add((obj_r - tpl_r, obj_c - tpl_c))
    return offsets


def _valid_alignment(
    grid: np.ndarray,
    bg: int,
    template_rel: set[tuple[int, int]],
    color: int,
    object_coords: set[tuple[int, int]],
    offset: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    off_r, off_c = offset
    translated = {(off_r + dr, off_c + dc) for dr, dc in template_rel}
    if not translated.issuperset(object_coords):
        return None
    if len(translated) == len(object_coords):
        return None
    for rr, cc in translated:
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        value = int(grid[rr, cc])
        if value not in {bg, color}:
            return None
    observed = {(rr, cc) for rr, cc in translated if int(grid[rr, cc]) == color}
    if observed != object_coords:
        return None
    return translated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete colored copy by translating the grey template shape onto it.

    Input structure:
    - The grid has one dominant background color.
    - Exactly one grey 4-connected object is a complete template shape.
    - The other foreground objects are monochrome copies of that same shape, but each copy is missing a few template pixels that currently show the background.

    Rule:
    - Extract the grey template's occupied coordinates as a relative shape.
    - For every non-grey foreground object, find the translation where its existing pixels match a subset of the template shape.
    - Fill every missing template position at that translation with the object's own color, and leave the grey template unchanged.

    Key insight:
    - Grey marks the only complete reference object; the colored objects are partial translated instances of the same shape, so the missing cells are found by template alignment rather than by local symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    template_rel, _template_origin, template_area = _template_data(g)

    out = g.copy()
    object_components: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color in {bg, GREY}:
            continue
        object_components.extend(_color_components(g, color))

    assert object_components, "expected at least one incomplete colored copy"

    for comp in object_components:
        color = int(comp["color"])
        object_coords = {tuple(map(int, coord)) for coord in comp["coords"]}
        assert 0 < len(object_coords) < template_area

        matches: list[set[tuple[int, int]]] = []
        for offset in _candidate_offsets(template_rel, object_coords):
            aligned = _valid_alignment(g, bg, template_rel, color, object_coords, offset)
            if aligned is not None:
                matches.append(aligned)

        unique_matches = {frozenset(match) for match in matches}
        assert len(unique_matches) == 1, "each colored object must align to the template uniquely"

        translated = next(iter(unique_matches))
        for rr, cc in translated:
            if out[rr, cc] == bg:
                out[rr, cc] = color

    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
COPY_COUNT_RANGE = (2, 5)
SHAPE_AREA_RANGE = (6, 12)
BOX_SIDE_RANGE = (3, 5)
SLOT_SPACING_RANGE = (2, 4)
OUTER_MARGIN_RANGE = (0, 2)
MISSING_COUNTS = (1, 2, 3)


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _grow_shape() -> set[tuple[int, int]] | None:
    box_h = random.randint(*BOX_SIDE_RANGE)
    box_w = random.randint(*BOX_SIDE_RANGE)
    max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w - 1)
    if max_area < SHAPE_AREA_RANGE[0]:
        return None

    target_area = random.randint(SHAPE_AREA_RANGE[0], max_area)
    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    cells = _normalize_shape(cells)
    bbox_h, bbox_w = _shape_bbox(cells)
    if bbox_h < 3 or bbox_w < 3:
        return None
    if len(cells) == bbox_h * bbox_w:
        return None
    return cells


def _removal_options(shape: set[tuple[int, int]]) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    ordered = sorted(shape)
    options = {count: [] for count in MISSING_COUNTS}
    for count in MISSING_COUNTS:
        for combo in itertools.combinations(ordered, count):
            remainder = shape.difference(combo)
            if remainder and _is_connected(remainder):
                options[count].append(combo)
    return options


def _random_shape_with_removals() -> tuple[set[tuple[int, int]], dict[int, list[tuple[tuple[int, int], ...]]]]:
    for _ in range(200):
        shape = _grow_shape()
        if shape is None:
            continue
        options = _removal_options(shape)
        if all(options[count] for count in MISSING_COUNTS):
            return shape, options
    raise RuntimeError("failed to generate a shape with all removal counts available")


def _layout_options(count: int, bbox_h: int, bbox_w: int) -> list[tuple[int, int, int, int, int, int]]:
    layouts: list[tuple[int, int, int, int, int, int]] = []
    for slot_rows in range(1, min(3, count) + 1):
        slot_cols = (count + slot_rows - 1) // slot_rows
        for cell_h in range(bbox_h + SLOT_SPACING_RANGE[0], bbox_h + SLOT_SPACING_RANGE[1] + 1):
            for cell_w in range(bbox_w + SLOT_SPACING_RANGE[0], bbox_w + SLOT_SPACING_RANGE[1] + 1):
                for top_margin in range(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1):
                    for left_margin in range(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1):
                        for bottom_margin in range(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1):
                            for right_margin in range(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1):
                                rows = top_margin + slot_rows * cell_h + bottom_margin
                                cols = left_margin + slot_cols * cell_w + right_margin
                                if max(rows, cols) <= 30:
                                    layouts.append(
                                        (
                                            slot_rows,
                                            slot_cols,
                                            cell_h,
                                            cell_w,
                                            rows,
                                            cols,
                                        )
                                    )
    return layouts


def _placement_positions(
    count: int, bbox_h: int, bbox_w: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    layouts = _layout_options(count, bbox_h, bbox_w)
    if not layouts:
        return None
    slot_rows, slot_cols, cell_h, cell_w, rows, cols = random.choice(layouts)

    top_margin = rows - slot_rows * cell_h
    left_margin = cols - slot_cols * cell_w
    top_pad = random.randint(0, top_margin)
    left_pad = random.randint(0, left_margin)

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen_slots = random.sample(slots, count)
    positions: list[tuple[int, int]] = []
    for sr, sc in chosen_slots:
        base_r = top_pad + sr * cell_h
        base_c = left_pad + sc * cell_w
        jitter_r = random.randint(0, cell_h - bbox_h - 1)
        jitter_c = random.randint(0, cell_w - bbox_w - 1)
        positions.append((base_r + jitter_r, base_c + jitter_c))
    random.shuffle(positions)
    return rows, cols, positions


def _paint_shape(grid: np.ndarray, shape: set[tuple[int, int]], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _build_pair() -> tuple[np.ndarray, np.ndarray]:
    copy_count = random.randint(*COPY_COUNT_RANGE)
    total_objects = copy_count + 1
    shape, options = _random_shape_with_removals()
    bbox_h, bbox_w = _shape_bbox(shape)

    placed = _placement_positions(total_objects, bbox_h, bbox_w)
    if placed is None:
        raise RuntimeError("no valid placement layout")
    rows, cols, positions = placed

    colors = random.sample(PALETTE, copy_count + 2)
    bg = next(color for color in colors if color != GREY)
    copy_palette = [color for color in colors if color not in {bg, GREY}]
    random.shuffle(copy_palette)

    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)

    template_origin = positions[0]
    _paint_shape(inp, shape, template_origin, GREY)
    _paint_shape(out, shape, template_origin, GREY)

    available_counts = [count for count in MISSING_COUNTS if options[count]]
    chosen_missing_counts = available_counts.copy()
    while len(chosen_missing_counts) < copy_count:
        chosen_missing_counts.append(random.choice(available_counts))
    random.shuffle(chosen_missing_counts)
    chosen_missing_counts = chosen_missing_counts[:copy_count]

    for origin, color, missing_count in zip(positions[1:], copy_palette, chosen_missing_counts):
        removed = set(random.choice(options[missing_count]))
        visible = shape.difference(removed)
        _paint_shape(inp, visible, origin, color)
        _paint_shape(out, shape, origin, color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    try:
        inp, expected = _build_pair()
    except RuntimeError:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
