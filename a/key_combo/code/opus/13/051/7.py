from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


BACKGROUND = int(Color.BLACK)
TEMPLATE_COLOR = int(Color.TEAL)
CANDIDATE_COLOR = int(Color.GREY)
INDICATOR_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN_SIZE = 18
GRID_MAX_SIZE = 26
AREA_RANGE = (3, 6)
MATCH_COUNT_RANGE = (2, 3)
CANDIDATE_COUNT_MAX = 6
PLACEMENT_ATTEMPTS = 400


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_cells(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = anchor
    return {(r0 + r, c0 + c) for r, c in shape}


def _dilate8(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _future_copy_steps(
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    direction: str,
    grid_shape: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    h, w = grid_shape
    dr, dc = DIR_VECTORS[direction]
    box_h, box_w = _shape_dims(shape)
    stride = box_h if dr else box_w

    steps: list[set[tuple[int, int]]] = []
    k = 1
    while True:
        moved = {
            (anchor[0] + r + dr * stride * k, anchor[1] + c + dc * stride * k)
            for r, c in shape
        }
        if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in moved):
            break
        steps.append(moved)
        k += 1
    return steps


def _indicator_options(
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0 = anchor
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    options: set[tuple[int, int]] = set()
    if direction == "right":
        for r, c in shape:
            if c == max_c:
                options.add((r0 + r, c0 + c + 1))
    elif direction == "left":
        for r, c in shape:
            if c == min_c:
                options.add((r0 + r, c0 + c - 1))
    elif direction == "down":
        for r, c in shape:
            if r == max_r:
                options.add((r0 + r + 1, c0 + c))
    else:
        for r, c in shape:
            if r == min_r:
                options.add((r0 + r - 1, c0 + c))
    ordered = list(options)
    random.shuffle(ordered)
    return ordered


def _touches_component(comp: dict, point: tuple[int, int]) -> bool:
    pr, pc = point
    coord_set = {tuple(map(int, rc)) for rc in comp["coords"]}
    return any(abs(r - pr) + abs(c - pc) == 1 for r, c in coord_set)


def _indicator_direction(indicator: dict, bbox: tuple[int, int, int, int]) -> str:
    ir, ic = map(int, indicator["coords"][0])
    r0, c0, r1, c1 = map(int, bbox)
    if ir < r0:
        return "up"
    if ir > r1:
        return "down"
    if ic < c0:
        return "left"
    if ic > c1:
        return "right"
    raise AssertionError("indicator must sit outside the candidate bbox")


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[tuple[dict, dict]]]:
    teal_objects = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[TEMPLATE_COLOR],
        return_format="dict",
    )
    candidates = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[CANDIDATE_COLOR],
        return_format="dict",
    )
    indicators = [
        comp
        for comp in find_components(
            grid,
            bg=BACKGROUND,
            connectivity=4,
            monochromatic=True,
            colors=list(INDICATOR_COLORS),
            return_format="dict",
        )
        if int(comp["area"]) == 1
    ]

    assert len(teal_objects) == 1, "expected one teal template object"
    assert 4 <= len(candidates) <= 6, "expected 4-6 grey candidates"
    assert len(indicators) == len(candidates), "every candidate must have one single-pixel indicator"

    remaining = set(range(len(indicators)))
    pairs: list[tuple[dict, dict]] = []
    for candidate in candidates:
        touching = [idx for idx in remaining if _touches_component(candidate, tuple(map(int, indicators[idx]["coords"][0])))]
        assert len(touching) == 1, "each candidate must have exactly one adjacent indicator"
        idx = touching[0]
        pairs.append((candidate, indicators[idx]))
        remaining.remove(idx)

    assert not remaining, "all indicators must be assigned"
    return teal_objects[0], pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile only the grey objects whose normalized shape matches the teal template, using each touching indicator to set direction and copy color.

    Input structure:
    - The grid has a black background, one teal 4-connected template object near a corner, and several grey 4-connected candidate objects.
    - Every grey candidate has exactly one adjacent single-pixel indicator in blue, red, green, or yellow.
    - Some grey candidates share the teal template's translation-normalized shape, while the others have different shapes.

    Rule:
    - Extract the teal template's normalized shape and compare it against each grey candidate after translation normalization.
    - For every grey candidate whose shape matches the template, keep the original grey object unchanged, read its unique adjacent indicator, and infer a cardinal direction from the side of the candidate bbox on which that indicator sits.
    - Repeat the matched object's shape along that direction in steps of one candidate-bbox width or height, recoloring every full translated copy with the indicator color, and stop before any copy would leave the grid.

    Key insight:
    - The teal object is only a gating template: shape matching decides which grey objects activate, while the touching indicator independently encodes both the expansion direction and the color of every tiled copy.
    """
    g = np.asarray(grid, dtype=int)
    template, candidate_pairs = _parse_scene(g)
    template_shape = _normalized_shape(template["coords"])

    out = g.copy()
    h, w = g.shape
    for candidate, indicator in candidate_pairs:
        candidate_shape = _normalized_shape(candidate["coords"])
        if candidate_shape != template_shape:
            continue

        direction = _indicator_direction(indicator, candidate["bbox"])
        dr, dc = DIR_VECTORS[direction]
        r0, c0, r1, c1 = map(int, candidate["bbox"])
        stride = (r1 - r0 + 1) if dr else (c1 - c0 + 1)
        color = int(indicator["color"])
        coords = np.asarray(candidate["coords"], dtype=int)

        step = 1
        while True:
            moved = coords + np.array([dr * stride * step, dc * stride * step], dtype=int)
            if (
                int(moved[:, 0].min()) < 0
                or int(moved[:, 0].max()) >= h
                or int(moved[:, 1].min()) < 0
                or int(moved[:, 1].max()) >= w
            ):
                break
            out[moved[:, 0], moved[:, 1]] = color
            step += 1

    return out


def _random_shape(
    *,
    exact_area: int | None = None,
    required_dims: tuple[int, int] | None = None,
    forbidden: set[tuple[tuple[int, int], ...]] | None = None,
) -> tuple[tuple[int, int], ...]:
    banned = forbidden or set()
    for _ in range(500):
        area = exact_area if exact_area is not None else random.randint(*AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
            frontier -= cells
            cells.add(random.choice(tuple(frontier)))

        coords = np.asarray(sorted(cells), dtype=int)
        shape = _normalized_shape(coords)
        box_h, box_w = _shape_dims(shape)
        if box_h == 1 or box_w == 1:
            continue
        if len(shape) == box_h * box_w:
            continue
        if required_dims is not None and (box_h, box_w) != required_dims:
            continue
        if shape in banned:
            continue
        return shape
    raise RuntimeError("failed to generate a valid irregular shape")


def _build_decoys(template_shape: tuple[tuple[int, int], ...], count: int) -> list[tuple[tuple[int, int], ...]]:
    used = {template_shape}
    decoys: list[tuple[tuple[int, int], ...]] = []
    template_dims = _shape_dims(template_shape)
    template_area = len(template_shape)

    if count >= 1:
        decoy = _random_shape(exact_area=template_area, forbidden=used)
        decoys.append(decoy)
        used.add(decoy)
    if count >= 2:
        decoy = _random_shape(required_dims=template_dims, forbidden=used)
        decoys.append(decoy)
        used.add(decoy)
    while len(decoys) < count:
        decoy = _random_shape(forbidden=used)
        decoys.append(decoy)
        used.add(decoy)
    return decoys


def _corner_anchors(h: int, w: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    margin = 3
    anchors = []
    ranges = [
        (range(0, min(margin, h - box_h) + 1), range(0, min(margin, w - box_w) + 1)),
        (range(0, min(margin, h - box_h) + 1), range(max(0, w - box_w - margin), w - box_w + 1)),
        (range(max(0, h - box_h - margin), h - box_h + 1), range(0, min(margin, w - box_w) + 1)),
        (range(max(0, h - box_h - margin), h - box_h + 1), range(max(0, w - box_w - margin), w - box_w + 1)),
    ]
    for rr, cc in ranges:
        choices = [(r, c) for r in rr for c in cc]
        random.shuffle(choices)
        anchors.extend(choices)
    random.shuffle(anchors)
    return anchors


def _all_anchors(h: int, w: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    anchors = [(r, c) for r in range(h - box_h + 1) for c in range(w - box_w + 1)]
    random.shuffle(anchors)
    return anchors


def _place_candidate(
    *,
    shape: tuple[tuple[int, int], ...],
    direction: str,
    min_copies: int,
    h: int,
    w: int,
    input_cells: set[tuple[int, int]],
    input_halo: set[tuple[int, int]],
    future_reserved: set[tuple[int, int]],
    future_halo: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[set[tuple[int, int]]]] | None:
    box_h, box_w = _shape_dims(shape)
    for anchor in _all_anchors(h, w, box_h, box_w):
        object_cells = _shape_cells(shape, anchor)
        if object_cells & input_halo:
            continue
        if object_cells & future_halo:
            continue

        for indicator in _indicator_options(shape, anchor, direction):
            ir, ic = indicator
            if not (0 <= ir < h and 0 <= ic < w):
                continue
            actual_cells = set(object_cells)
            actual_cells.add(indicator)
            if actual_cells & input_halo:
                continue
            if actual_cells & future_halo:
                continue

            steps = _future_copy_steps(shape, anchor, direction, (h, w))
            if len(steps) < min_copies:
                continue

            future_cells = set().union(*steps) if steps else set()
            if (future_cells - {indicator}) & input_halo:
                continue
            if (future_cells - {indicator}) & future_halo:
                continue
            if (future_cells - {indicator}) & future_reserved:
                continue
            if (future_cells - {indicator}) & input_cells:
                continue

            return anchor, indicator, steps
    return None


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _validate_generated_grid(grid: np.ndarray) -> None:
    template, candidate_pairs = _parse_scene(grid)
    template_shape = _normalized_shape(template["coords"])
    template_bbox = tuple(map(int, template["bbox"]))
    th, tw = _shape_dims(template_shape)
    assert 3 <= len(template_shape) <= 6
    assert len(template_shape) != th * tw

    h, w = grid.shape
    corner_dist = min(template_bbox[0], template_bbox[1], h - 1 - template_bbox[2], w - 1 - template_bbox[3])
    assert corner_dist <= 3

    matched: list[tuple[dict, dict]] = []
    unmatched: list[tuple[dict, dict]] = []
    for candidate, indicator in candidate_pairs:
        cand_shape = _normalized_shape(candidate["coords"])
        assert 3 <= len(cand_shape) <= 6
        if cand_shape == template_shape:
            matched.append((candidate, indicator))
        else:
            unmatched.append((candidate, indicator))

    assert 2 <= len(matched) <= 3
    assert len(unmatched) >= 2

    matched_dirs = {_indicator_direction(indicator, candidate["bbox"]) for candidate, indicator in matched}
    assert len(matched_dirs) >= 2

    template_area = len(template_shape)
    template_dims = _shape_dims(template_shape)
    unmatched_shapes = [_normalized_shape(candidate["coords"]) for candidate, _indicator in unmatched]
    assert any(len(shape) == template_area for shape in unmatched_shapes)
    assert any(_shape_dims(shape) == template_dims for shape in unmatched_shapes)

    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != BACKGROUND)}
    reserved: set[tuple[int, int]] = set()
    copy_counts: list[int] = []
    for candidate, indicator in matched:
        direction = _indicator_direction(indicator, candidate["bbox"])
        coords = np.asarray(candidate["coords"], dtype=int)
        anchor = (int(coords[:, 0].min()), int(coords[:, 1].min()))
        steps = _future_copy_steps(template_shape, anchor, direction, grid.shape)
        assert steps
        future_cells = set().union(*steps)
        indicator_cell = tuple(map(int, indicator["coords"][0]))
        assert not (future_cells - {indicator_cell}) & occupied
        assert not future_cells & reserved
        reserved |= future_cells
        copy_counts.append(len(steps))
    assert max(copy_counts) >= 2

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    for candidate, indicator in matched:
        direction = _indicator_direction(indicator, candidate["bbox"])
        coords = np.asarray(candidate["coords"], dtype=int)
        anchor = (int(coords[:, 0].min()), int(coords[:, 1].min()))
        steps = _future_copy_steps(template_shape, anchor, direction, grid.shape)
        color = int(indicator["color"])
        for cells in steps:
            for r, c in cells:
                assert int(out[r, c]) == color


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    w = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    template_shape = _random_shape()
    match_count = random.randint(*MATCH_COUNT_RANGE)
    candidate_count = random.randint(match_count + 2, CANDIDATE_COUNT_MAX)
    decoy_shapes = _build_decoys(template_shape, candidate_count - match_count)

    input_cells: set[tuple[int, int]] = set()
    input_halo: set[tuple[int, int]] = set()
    future_reserved: set[tuple[int, int]] = set()
    future_halo: set[tuple[int, int]] = set()

    box_h, box_w = _shape_dims(template_shape)
    template_anchor = None
    for anchor in _corner_anchors(h, w, box_h, box_w):
        cells = _shape_cells(template_shape, anchor)
        if cells & input_halo:
            continue
        template_anchor = anchor
        _paint_cells(grid, cells, TEMPLATE_COLOR)
        input_cells |= cells
        input_halo |= _dilate8(cells, h, w)
        break
    if template_anchor is None:
        return None

    matched_dirs = random.sample(list(DIR_VECTORS.keys()), match_count)
    matched_colors = [random.choice(INDICATOR_COLORS) for _ in range(match_count)]
    if match_count > 1 and len(set(matched_colors)) == 1:
        matched_colors[0], matched_colors[1] = random.sample(INDICATOR_COLORS, 2)
    min_copies = [2] + [1] * (match_count - 1)
    random.shuffle(min_copies)

    for direction, color, min_copy in zip(matched_dirs, matched_colors, min_copies):
        placed = _place_candidate(
            shape=template_shape,
            direction=direction,
            min_copies=min_copy,
            h=h,
            w=w,
            input_cells=input_cells,
            input_halo=input_halo,
            future_reserved=future_reserved,
            future_halo=future_halo,
        )
        if placed is None:
            return None

        anchor, indicator, steps = placed
        object_cells = _shape_cells(template_shape, anchor)
        actual_cells = set(object_cells)
        actual_cells.add(indicator)
        _paint_cells(grid, object_cells, CANDIDATE_COLOR)
        grid[indicator] = color

        input_cells |= actual_cells
        input_halo |= _dilate8(actual_cells, h, w)
        future_cells = set().union(*steps)
        future_reserved |= future_cells
        future_halo |= _dilate8(future_cells, h, w)

    for idx, shape in enumerate(decoy_shapes):
        direction = random.choice(matched_dirs if idx == 0 else list(DIR_VECTORS.keys()))
        color = random.choice(matched_colors if idx == 0 else INDICATOR_COLORS)
        placed = _place_candidate(
            shape=shape,
            direction=direction,
            min_copies=0,
            h=h,
            w=w,
            input_cells=input_cells,
            input_halo=input_halo,
            future_reserved=future_reserved,
            future_halo=future_halo,
        )
        if placed is None:
            return None
        anchor, indicator, _steps = placed
        object_cells = _shape_cells(shape, anchor)
        actual_cells = set(object_cells)
        actual_cells.add(indicator)
        _paint_cells(grid, object_cells, CANDIDATE_COLOR)
        grid[indicator] = color
        input_cells |= actual_cells
        input_halo |= _dilate8(actual_cells, h, w)

    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(PLACEMENT_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
