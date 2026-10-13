from __future__ import annotations

import numpy as np

from arc_utils import bbox as _gen_bbox, bbox_h, bbox_w, crop_bbox as _gen_crop_bbox, find_components, get_bg, paste, paste as _gen_paste
import random


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _component_colors(g: np.ndarray, coords: np.ndarray, bg: int) -> set[int]:
    colors = set(map(int, np.unique(g[tuple(coords.T)])))
    colors.discard(int(bg))
    return colors


def _find_template_roles(
    g: np.ndarray, bg: int
) -> tuple[dict, set[tuple[int, int]], int, int]:
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    colored_components = [
        (comp, _component_colors(g, comp["coords"], bg)) for comp in components
    ]
    mixed = [(comp, colors) for comp, colors in colored_components if len(colors) > 1]
    assert len(mixed) == 1, "expected exactly one mixed-color template component"
    template, template_colors = mixed[0]

    outside_colors: set[int] = set()
    for comp, colors in colored_components:
        if comp is template:
            continue
        outside_colors.update(colors)

    marker_colors = template_colors & outside_colors
    source_colors = template_colors - marker_colors
    assert len(marker_colors) == 1, "expected one template marker color repeated outside"
    assert len(source_colors) == 1, "expected one source color confined to the template"

    return template, _coord_set(template["coords"]), source_colors.pop(), marker_colors.pop()


def _template_cells(
    g: np.ndarray,
    template: dict,
    template_coords: set[tuple[int, int]],
    source_color: int,
    marker_color: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, _r1, _c1 = map(int, template["bbox"])
    source_rel: list[tuple[int, int]] = []
    marker_rel: list[tuple[int, int]] = []

    for r, c in sorted(template_coords):
        rel = (r - r0, c - c0)
        color = int(g[r, c])
        if color == int(source_color):
            source_rel.append(rel)
        elif color == int(marker_color):
            marker_rel.append(rel)
        else:
            raise AssertionError("template contains an unexpected foreground color")

    assert source_rel, "template must contain source-color cells"
    assert marker_rel, "template must contain marker-color cells"
    return source_rel, marker_rel


def _target_marker_squares(
    g: np.ndarray,
    bg: int,
    marker_color: int,
    template_coords: set[tuple[int, int]],
) -> list[dict]:
    targets: list[dict] = []
    for comp in find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[marker_color]
    ):
        coords = _coord_set(comp["coords"])
        if coords <= template_coords:
            continue
        assert coords.isdisjoint(template_coords), "target marker overlaps the template"

        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        height = bbox_h(comp["bbox"])
        width = bbox_w(comp["bbox"])
        assert height == width, "target marker must be a square"
        assert int(comp["area"]) == height * width, "target marker square must be solid"

        targets.append({"id": len(targets), "scale": height, "r0": r0, "c0": c0})

    assert targets, "expected detached target marker squares"
    return targets


def _matched_instances(
    targets: list[dict], marker_rel: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], int]]:
    by_position = {
        (target["scale"], target["r0"], target["c0"]): target for target in targets
    }
    assert len(by_position) == len(targets), "target marker positions must be unique"

    instances: dict[frozenset[int], tuple[tuple[int, int], int]] = {}
    for anchor in targets:
        scale = int(anchor["scale"])
        for marker_r, marker_c in marker_rel:
            origin = (
                int(anchor["r0"]) - marker_r * scale,
                int(anchor["c0"]) - marker_c * scale,
            )
            members: list[int] = []
            for rel_r, rel_c in marker_rel:
                target = by_position.get(
                    (scale, origin[0] + rel_r * scale, origin[1] + rel_c * scale)
                )
                if target is None:
                    break
                members.append(int(target["id"]))
            else:
                member_ids = frozenset(members)
                previous = instances.get(member_ids)
                assert previous in (None, (origin, scale)), "ambiguous marker grouping"
                instances[member_ids] = (origin, scale)

    assert instances, "target markers must match the template marker layout"
    covered_ids = sorted(
        target_id for member_ids in instances for target_id in member_ids
    )
    assert covered_ids == list(range(len(targets))), "each target marker belongs to one instance"
    return sorted(instances.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete scaled copies of the blue source template at matching red marker squares.

    Input structure
    - A black background contains one 4-connected mixed-color template object and detached same-color target squares.
    - The template has source-color pixels and marker-color pixels; the detached marker objects are solid k x k squares.
    - The output shape is unchanged.

    Rule
    - Use the template bbox as the pattern coordinate system.
    - Group target marker squares into sets whose top-left corners match the template marker coordinates at one common scale k.
    - For each matched group, scale every source-color template cell to a k x k block and paint it at that instance's origin, clipping any part outside the grid.

    Key insight
    The detached marker squares are enlarged copies of the template's marker cells; their side length gives the scale for copying the source-color cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    template, template_coords, source_color, marker_color = _find_template_roles(g, bg)
    source_rel, marker_rel = _template_cells(
        g, template, template_coords, source_color, marker_color
    )
    targets = _target_marker_squares(g, bg, marker_color, template_coords)
    instances = _matched_instances(targets, marker_rel)

    out = g.copy()
    for (origin_r, origin_c), scale in instances:
        block = np.full((scale, scale), int(source_color), dtype=int)
        for rel_r, rel_c in source_rel:
            paste(
                out,
                block,
                origin_r + rel_r * scale,
                origin_c + rel_c * scale,
                transparent=bg,
                overwrite=False,
                clip=True,
            )

    return out

# --------------- generator (o5x) ---------------

Cell = tuple[int, int]
Box = tuple[int, int, int, int]
TemplateInfo = tuple[np.ndarray, list[Cell], list[Cell]]

# Canvas and palette bounds.
BACKGROUND_COLOR = 0
SOURCE_COLOR = 1
MARKER_COLOR = 2
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (12, 21)

# Template shape and content bounds.
TEMPLATE_WIDTH_RANGE = (3, 4)
TEMPLATE_HEIGHT_RANGE = (1, 4)
TEMPLATE_MIN_HEIGHT_FOR_WIDTH3 = 2
SOURCE_COUNT_RANGE = (3, 6)
MARKER_COUNT_RANGE = (1, 2)
TEMPLATE_AREA_RANGE = (4, 8)

# Target grouping and scale bounds.
TARGET_COUNT_RANGE = (1, 4)
TARGET_STRUCTURES_BY_COUNT = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3),),
    4: ((2, 2),),
}
SCALE_RANGE = (1, 3)

# Layout separation, template branch weights, and retries.
PLACEMENT_GAP = 2
ONE_MARKER_LINE_LEFT_PROB = 0.5
TWO_MARKER_HORIZONTAL_PROB = 0.55
GENERATION_RETRY_LIMIT = 300

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

def _gen_neighbors4(r: int, c: int, h: int, w: int) -> list[Cell]:
    out: list[Cell] = []
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _gen_grow_source(
    seed: set[Cell],
    blocked: set[Cell],
    h: int,
    w: int,
    target_n: int,
) -> set[Cell] | None:
    cells = set(seed)
    while len(cells) < target_n:
        frontier = sorted(
            {
                nbr
                for r, c in cells
                for nbr in _gen_neighbors4(r, c, h, w)
                if nbr not in cells and nbr not in blocked
            }
        )
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _gen_is_connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen == cells


def _gen_boxes_have_gap(
    a: Box, b: Box, gap: int = PLACEMENT_GAP
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _gen_crop_and_describe(canvas: np.ndarray) -> TemplateInfo | None:
    b = _gen_bbox(canvas != BACKGROUND_COLOR)
    if b is None:
        return None
    template = _gen_crop_bbox(canvas, b)
    source_rel = [tuple(map(int, rc)) for rc in np.argwhere(template == SOURCE_COLOR)]
    marker_rel = [tuple(map(int, rc)) for rc in np.argwhere(template == MARKER_COLOR)]
    if not (
        SOURCE_COUNT_RANGE[0] <= len(source_rel) <= SOURCE_COUNT_RANGE[1]
        and MARKER_COUNT_RANGE[0] <= len(marker_rel) <= MARKER_COUNT_RANGE[1]
    ):
        return None
    th, tw = template.shape
    if not (
        TEMPLATE_HEIGHT_RANGE[0] <= th <= TEMPLATE_HEIGHT_RANGE[1]
        and TEMPLATE_WIDTH_RANGE[0] <= tw <= TEMPLATE_WIDTH_RANGE[1]
        and TEMPLATE_AREA_RANGE[0] <= np.count_nonzero(template) <= TEMPLATE_AREA_RANGE[1]
    ):
        return None
    if not _gen_is_connected(set(source_rel)):
        return None
    src = set(source_rel)
    for mr, mc in marker_rel:
        if not any((mr + dr, mc + dc) in src for dr, dc in NEIGHBORS4):
            return None
    if len(marker_rel) == 2:
        boxes = [(r, c, r, c) for r, c in marker_rel]
        if not _gen_boxes_have_gap(boxes[0], boxes[1], PLACEMENT_GAP):
            return None
    return template, source_rel, marker_rel


def _gen_template_one_marker() -> TemplateInfo | None:
    tw = random.randint(*TEMPLATE_WIDTH_RANGE)
    min_height = (
        TEMPLATE_HEIGHT_RANGE[0]
        if tw == TEMPLATE_WIDTH_RANGE[1]
        else TEMPLATE_MIN_HEIGHT_FOR_WIDTH3
    )
    th = random.randint(min_height, TEMPLATE_HEIGHT_RANGE[1])
    max_source = min(SOURCE_COUNT_RANGE[1], th * tw - 1)
    if max_source < SOURCE_COUNT_RANGE[0]:
        return None
    source_n = random.randint(SOURCE_COUNT_RANGE[0], max_source)
    canvas = np.zeros((th, tw), dtype=int)

    if th == 1:
        source_n = SOURCE_COUNT_RANGE[0]
        if random.random() < ONE_MARKER_LINE_LEFT_PROB:
            marker = (0, 0)
            source = {(0, 1), (0, 2), (0, 3)}
        else:
            marker = (0, 3)
            source = {(0, 0), (0, 1), (0, 2)}
    else:
        row = random.randrange(th)
        if tw == 4:
            marker_col = random.choice((0, 3))
            marker = (row, marker_col)
            cols = range(1, 4) if marker_col == 0 else range(0, 3)
            seed = {(row, c) for c in cols}
        else:
            seed = {(row, 0), (row, 1), (row, 2)}
            candidates = [
                (rr, cc)
                for rr in (row - 1, row + 1)
                if 0 <= rr < th
                for cc in range(3)
            ]
            marker = random.choice(candidates)
        source = _gen_grow_source(seed, {marker}, th, tw, source_n)
        if source is None:
            return None

    for r, c in source:
        canvas[r, c] = SOURCE_COLOR
    canvas[marker] = MARKER_COLOR
    return _gen_crop_and_describe(canvas)


def _gen_template_two_markers() -> TemplateInfo | None:
    if random.random() < TWO_MARKER_HORIZONTAL_PROB:
        th, tw = random.randint(TEMPLATE_MIN_HEIGHT_FOR_WIDTH3, TEMPLATE_HEIGHT_RANGE[1]), 4
        row = random.randrange(th)
        markers = {(row, 0), (row, 3)}
        seed = {(row, 1), (row, 2)}
        min_source = SOURCE_COUNT_RANGE[0]
    else:
        th, tw = TEMPLATE_HEIGHT_RANGE[1], random.randint(*TEMPLATE_WIDTH_RANGE)
        col = random.randint(1, tw - 2)
        markers = {(0, col), (3, col)}
        seed = {(1, col), (2, col), (1, col - 1), (2, col + 1)}
        min_source = 4

    max_source = min(SOURCE_COUNT_RANGE[1], th * tw - MARKER_COUNT_RANGE[1])
    if max_source < min_source:
        return None
    source_n = random.randint(min_source, max_source)
    source = _gen_grow_source(seed, markers, th, tw, source_n)
    if source is None:
        return None

    canvas = np.zeros((th, tw), dtype=int)
    for r, c in source:
        canvas[r, c] = SOURCE_COLOR
    for marker in markers:
        canvas[marker] = MARKER_COLOR
    return _gen_crop_and_describe(canvas)


def _gen_choose_structure() -> tuple[int, int]:
    target_count = random.randint(*TARGET_COUNT_RANGE)
    choices = TARGET_STRUCTURES_BY_COUNT[target_count]
    if len(choices) == 1:
        return choices[0]
    return random.choice(choices)


def _gen_block_cells(r0: int, c0: int, k: int, h: int, w: int) -> set[Cell]:
    rr0, cc0 = max(0, r0), max(0, c0)
    rr1, cc1 = min(h, r0 + k), min(w, c0 + k)
    return {(r, c) for r in range(rr0, rr1) for c in range(cc0, cc1)}


def _gen_instance_at(
    origin: Cell,
    scale: int,
    source_rel: list[Cell],
    marker_rel: list[Cell],
    h: int,
    w: int,
) -> dict | None:
    origin_r, origin_c = origin
    marker_boxes: list[Box] = []
    marker_cells: set[Cell] = set()
    source_cells: set[Cell] = set()

    for rel_r, rel_c in marker_rel:
        r0 = origin_r + rel_r * scale
        c0 = origin_c + rel_c * scale
        if r0 < 0 or c0 < 0 or r0 + scale > h or c0 + scale > w:
            return None
        box = (r0, c0, r0 + scale - 1, c0 + scale - 1)
        if any(not _gen_boxes_have_gap(box, other, PLACEMENT_GAP) for other in marker_boxes):
            return None
        marker_boxes.append(box)
        marker_cells |= _gen_block_cells(r0, c0, scale, h, w)

    for rel_r, rel_c in source_rel:
        r0 = origin_r + rel_r * scale
        c0 = origin_c + rel_c * scale
        cells = _gen_block_cells(r0, c0, scale, h, w)
        if not cells:
            return None
        source_cells |= cells

    if marker_cells & source_cells:
        return None
    return {
        "origin": origin,
        "scale": scale,
        "marker_boxes": marker_boxes,
        "marker_cells": marker_cells,
        "source_cells": source_cells,
        "cells": marker_cells | source_cells,
    }


def _gen_origin_ranges(
    scale: int,
    source_rel: list[Cell],
    marker_rel: list[Cell],
    h: int,
    w: int,
) -> tuple[range, range] | None:
    min_r = max(-rel_r * scale for rel_r, _ in marker_rel)
    max_r = min(h - scale - rel_r * scale for rel_r, _ in marker_rel)
    min_c = max(-rel_c * scale for _, rel_c in marker_rel)
    max_c = min(w - scale - rel_c * scale for _, rel_c in marker_rel)

    min_r = max(min_r, max(-rel_r * scale - scale + 1 for rel_r, _ in source_rel))
    max_r = min(max_r, min(h - 1 - rel_r * scale for rel_r, _ in source_rel))
    min_c = max(min_c, max(-rel_c * scale - scale + 1 for _, rel_c in source_rel))
    max_c = min(max_c, min(w - 1 - rel_c * scale for _, rel_c in source_rel))
    if min_r > max_r or min_c > max_c:
        return None
    return range(min_r, max_r + 1), range(min_c, max_c + 1)


def _gen_place_instance(
    scale: int,
    source_rel: list[Cell],
    marker_rel: list[Cell],
    h: int,
    w: int,
    occupied: set[Cell],
    marker_boxes: list[Box],
) -> dict | None:
    ranges = _gen_origin_ranges(scale, source_rel, marker_rel, h, w)
    if ranges is None:
        return None
    rows, cols = ranges
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)
    for origin in candidates:
        inst = _gen_instance_at(origin, scale, source_rel, marker_rel, h, w)
        if inst is None or inst["cells"] & occupied:
            continue
        if any(
            not _gen_boxes_have_gap(box, old_box, PLACEMENT_GAP)
            for box in inst["marker_boxes"]
            for old_box in marker_boxes
        ):
            continue
        return inst
    return None


def _gen_grouping_unambiguous(instances: list[dict], marker_rel: list[Cell]) -> bool:
    if len(marker_rel) == 1:
        return True
    delta = (
        (marker_rel[1][0] - marker_rel[0][0]),
        (marker_rel[1][1] - marker_rel[0][1]),
    )
    records: list[tuple[int, int, int, int]] = []
    expected: set[frozenset[int]] = set()
    for gi, inst in enumerate(instances):
        group_ids: list[int] = []
        origin_r, origin_c = inst["origin"]
        scale = inst["scale"]
        for rel_r, rel_c in marker_rel:
            group_ids.append(len(records))
            records.append((scale, origin_r + rel_r * scale, origin_c + rel_c * scale, gi))
        expected.add(frozenset(group_ids))

    found: set[frozenset[int]] = set()
    for i, (ki, ri, ci, _gi) in enumerate(records):
        for j, (kj, rj, cj, _gj) in enumerate(records):
            if i >= j or ki != kj:
                continue
            dr, dc = rj - ri, cj - ci
            if (dr, dc) in (
                (delta[0] * ki, delta[1] * ki),
                (-delta[0] * ki, -delta[1] * ki),
            ):
                found.add(frozenset((i, j)))
    return found == expected


def _gen_place_template(
    template: np.ndarray,
    h: int,
    w: int,
    occupied: set[Cell],
    marker_boxes: list[Box],
) -> Cell | None:
    th, tw = template.shape
    rel_cells = [tuple(map(int, rc)) for rc in np.argwhere(template != BACKGROUND_COLOR)]
    candidates = [(r, c) for r in range(h - th + 1) for c in range(w - tw + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        template_box = (r0, c0, r0 + th - 1, c0 + tw - 1)
        if any(not _gen_boxes_have_gap(template_box, box, PLACEMENT_GAP) for box in marker_boxes):
            continue
        cells = {(r0 + r, c0 + c) for r, c in rel_cells}
        if cells & occupied:
            continue
        return r0, c0
    return None


def _gen_choose_template(marker_count: int) -> TemplateInfo | None:
    if marker_count == MARKER_COUNT_RANGE[0]:
        return _gen_template_one_marker()
    return _gen_template_two_markers()


def _gen_place_instances(
    group_count: int,
    source_rel: list[Cell],
    marker_rel: list[Cell],
    h: int,
    w: int,
) -> tuple[list[dict], set[Cell], list[Box]] | None:
    """Place all scaled instance footprints and return their occupancy records."""
    occupied: set[Cell] = set()
    marker_boxes: list[Box] = []
    instances: list[dict] = []
    scales = sorted([random.randint(*SCALE_RANGE) for _ in range(group_count)], reverse=True)
    for scale in scales:
        inst = _gen_place_instance(
            scale, source_rel, marker_rel, h, w, occupied, marker_boxes
        )
        if inst is None:
            return None
        instances.append(inst)
        occupied |= inst["cells"]
        marker_boxes.extend(inst["marker_boxes"])
    return instances, occupied, marker_boxes


def _gen_render_output(
    template: np.ndarray,
    template_pos: Cell,
    instances: list[dict],
    source_rel: list[Cell],
    marker_rel: list[Cell],
    h: int,
    w: int,
) -> np.ndarray:
    """Render the solved grid before source blocks are removed for the input."""
    out = np.zeros((h, w), dtype=int)
    tr, tc = template_pos
    _gen_paste(out, template, tr, tc, transparent=BACKGROUND_COLOR)
    for inst in instances:
        origin_r, origin_c = inst["origin"]
        scale = inst["scale"]
        marker_block = np.full((scale, scale), MARKER_COLOR, dtype=int)
        source_block = np.full((scale, scale), SOURCE_COLOR, dtype=int)
        for rel_r, rel_c in marker_rel:
            _gen_paste(
                out,
                marker_block,
                origin_r + rel_r * scale,
                origin_c + rel_c * scale,
            )
        for rel_r, rel_c in source_rel:
            _gen_paste(
                out,
                source_block,
                origin_r + rel_r * scale,
                origin_c + rel_c * scale,
                transparent=BACKGROUND_COLOR,
                overwrite=False,
                clip=True,
            )
    return out


def _gen_input_if_solver_matches(out: np.ndarray, instances: list[dict]) -> np.ndarray | None:
    """Remove source completions and keep only grids accepted by the solver oracle."""
    inp = out.copy()
    for inst in instances:
        for r, c in inst["source_cells"]:
            inp[r, c] = BACKGROUND_COLOR

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    marker_count, group_count = _gen_choose_structure()
    template_info = _gen_choose_template(marker_count)
    if template_info is None:
        return None
    template, source_rel, marker_rel = template_info

    placement = _gen_place_instances(group_count, source_rel, marker_rel, h, w)
    if placement is None:
        return None
    instances, occupied, marker_boxes = placement

    if not _gen_grouping_unambiguous(instances, marker_rel):
        return None

    template_pos = _gen_place_template(template, h, w, occupied, marker_boxes)
    if template_pos is None:
        return None

    out = _gen_render_output(template, template_pos, instances, source_rel, marker_rel, h, w)
    return _gen_input_if_solver_matches(out, instances)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 447fd412 input")
