from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MARKER = 1
SPRITE = 2

GRID_HEIGHT_RANGE = (14, 19)
GRID_WIDTH_RANGE = (17, 22)
TOTAL_OBJECTS_RANGE = (4, 6)
SELECTED_COUNT_RANGE = (2, 4)
BOX_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4))
TEMPLATE_AREA_MAX = 8
MIN_PARTIAL_AREA = 4
PLACEMENT_GAP = 1
GEN_RETRIES = 400

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    mask[rr, cc] = True
    return mask


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, np.ndarray | BBox]]:
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict[str, np.ndarray | BBox]] = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            box = _bbox_from_coords(coords)
            comps.append({"coords": coords, "bbox": box, "mask": _mask_from_coords(coords, box)})
    return comps


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> Coord:
    height, width = shape
    on_top = r == 0
    on_bottom = r == height - 1
    on_left = c == 0
    on_right = c == width - 1
    assert on_top or on_bottom or on_left or on_right, "marker must lie on the boundary"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "corner markers are ambiguous"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _ray_target_index(shape: tuple[int, int], marker: Coord, bboxes: list[BBox]) -> int:
    r, c = marker
    dr, dc = _marker_direction(shape, r, c)
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        hits = [i for i, (r0, c0, r1, c1) in enumerate(bboxes) if r0 <= r <= r1 and c0 <= c <= c1]
        assert len(hits) <= 1, "ray cannot enter overlapping sprite boxes"
        if hits:
            return hits[0]
        r += dr
        c += dc
    raise AssertionError("each marker ray must hit a sprite box")


def _selected_indices(grid: np.ndarray, bboxes: list[BBox]) -> set[int]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert markers, "expected at least one blue boundary marker"
    return {_ray_target_index(grid.shape, marker, bboxes) for marker in markers}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the projected red sprite copies, merge their complementary masks, and erase the rest.

    Input structure:
    - A black grid contains several red 4-connected sprite copies, all with the same bounding-box size.
    - Blue single-pixel markers sit on the outer border and project inward orthogonally.
    - Some red copies are true partial instances of one template, while unhit copies are distractors.

    Rule:
    - For each blue border marker, follow its inward ray until it first enters a red sprite bounding box, and mark that sprite as selected.
    - Crop every selected sprite to its own box, align those local masks by their top-left corners, and take their pixelwise union to recover the full template.
    - Erase every red sprite, redraw the recovered template in red at each selected box, and keep the blue markers unchanged.

    Key insight:
    - The border projection does not paint onto the target; it secretly chooses which incomplete copies may contribute to the aligned union, while unselected copies must be discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _components_of_color(g, SPRITE)
    assert sprites, "expected red sprite copies"
    bboxes = [sprite["bbox"] for sprite in sprites]
    box_shapes = {(box[2] - box[0] + 1, box[3] - box[1] + 1) for box in bboxes}
    assert len(box_shapes) == 1, "all sprite boxes must share one size"
    box_h, box_w = next(iter(box_shapes))

    selected = _selected_indices(g, bboxes)
    assert selected, "expected at least one selected sprite"

    template = np.zeros((box_h, box_w), dtype=bool)
    for idx in selected:
        template |= np.asarray(sprites[idx]["mask"], dtype=bool)
    assert template.any(), "selected sprites must recover a nonempty template"

    out = g.copy()
    out[out == SPRITE] = BG
    for idx in selected:
        r0, c0, r1, c1 = bboxes[idx]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[template] = SPRITE
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _generate_template(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _touches_all_sides(mask):
            return mask
    return None


def _enumerate_partials(template: np.ndarray) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    area = len(coords)
    partials: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for bits in range(1, 1 << area):
        keep_count = bits.bit_count()
        if not (MIN_PARTIAL_AREA <= keep_count < area):
            continue
        mask = np.zeros_like(template, dtype=bool)
        for i, (r, c) in enumerate(coords):
            if (bits >> i) & 1:
                mask[r, c] = True
        if not _touches_all_sides(mask) or not _is_connected(mask):
            continue
        key = _mask_key(mask)
        if key not in seen:
            seen.add(key)
            partials.append(mask)
    return partials


def _choose_masks(partials: list[np.ndarray], selected_count: int, distractor_count: int, template: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    indices = list(range(len(partials)))
    combos = list(itertools.combinations(indices, selected_count))
    random.shuffle(combos)
    for combo in combos:
        selected = [partials[i] for i in combo]
        union = np.zeros_like(template, dtype=bool)
        for mask in selected:
            union |= mask
        if not np.array_equal(union, template):
            continue
        remaining = [partials[i] for i in indices if i not in combo]
        random.shuffle(remaining)
        if len(remaining) < distractor_count:
            continue
        distractors = remaining[:distractor_count]
        return selected, distractors
    return None


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: BBox, gap: int) -> BBox:
    return box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap


def _place_boxes(height: int, width: int, h: int, w: int, count: int) -> list[BBox] | None:
    candidates = [
        (r, c, r + h - 1, c + w - 1)
        for r in range(1, height - h)
        for c in range(1, width - w)
    ]
    for _ in range(160):
        random.shuffle(candidates)
        placed: list[BBox] = []
        for box in candidates:
            expanded = _expand_box(box, PLACEMENT_GAP)
            if any(_boxes_overlap(expanded, _expand_box(other, PLACEMENT_GAP)) for other in placed):
                continue
            placed.append(box)
            if len(placed) == count:
                return placed
    return None


def _projection_candidates(height: int, width: int, bboxes: list[BBox], idx: int) -> list[Coord]:
    r0, c0, r1, c1 = bboxes[idx]
    out: list[Coord] = []
    for c in range(c0, c1 + 1):
        if all(j == idx or not (oc0 <= c <= oc1) for j, (_or0, oc0, _or1, oc1) in enumerate(bboxes)):
            out.append((0, c))
            out.append((height - 1, c))
    for r in range(r0, r1 + 1):
        if all(j == idx or not (or0 <= r <= or1) for j, (or0, _oc0, or1, _oc1) in enumerate(bboxes)):
            out.append((r, 0))
            out.append((r, width - 1))
    random.shuffle(out)
    deduped: list[Coord] = []
    seen = set()
    for pos in out:
        if pos not in seen:
            seen.add(pos)
            deduped.append(pos)
    return deduped


def _render_input(height: int, width: int, bboxes: list[BBox], masks: list[np.ndarray], markers: list[Coord]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for box, mask in zip(bboxes, masks):
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        region[mask] = SPRITE
    for r, c in markers:
        grid[r, c] = MARKER
    return grid


def _render_expected(inp: np.ndarray, bboxes: list[BBox], selected_indices: set[int], template: np.ndarray) -> np.ndarray:
    out = inp.copy()
    out[out == SPRITE] = BG
    for idx in selected_indices:
        r0, c0, r1, c1 = bboxes[idx]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[template] = SPRITE
    return out


def _choose_counts() -> tuple[int, int]:
    total = random.randint(*TOTAL_OBJECTS_RANGE)
    options = []
    for selected in range(SELECTED_COUNT_RANGE[0], min(SELECTED_COUNT_RANGE[1], total - 1) + 1):
        distractors = total - selected
        if 1 <= distractors <= 2:
            options.append((selected, distractors))
    assert options
    return random.choice(options)


def _candidate_shapes(selected_count: int) -> list[tuple[int, int, int]]:
    min_area = max(6, selected_count + 4)
    options = []
    for h, w in BOX_SHAPES:
        for area in range(min_area, min(TEMPLATE_AREA_MAX, h * w) + 1):
            options.append((h, w, area))
    random.shuffle(options)
    return options


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    selected_count, distractor_count = _choose_counts()
    total_count = selected_count + distractor_count

    chosen: tuple[int, int, int] | None = None
    template: np.ndarray | None = None
    partials: list[np.ndarray] = []
    selected_masks: list[np.ndarray] = []
    distractor_masks: list[np.ndarray] = []

    for h, w, area in _candidate_shapes(selected_count):
        if (height - 2) * (width - 2) < total_count * h * w:
            continue
        template = _generate_template(h, w, area)
        if template is None:
            continue
        partials = _enumerate_partials(template)
        picked = _choose_masks(partials, selected_count, distractor_count, template)
        if picked is None:
            continue
        selected_masks, distractor_masks = picked
        chosen = (h, w, area)
        break
    if chosen is None or template is None:
        return None

    h, w, _area = chosen
    bboxes = _place_boxes(height, width, h, w, total_count)
    if bboxes is None:
        return None

    candidates = [_projection_candidates(height, width, bboxes, i) for i in range(total_count)]
    eligible = [i for i, cand in enumerate(candidates) if cand]
    if len(eligible) < selected_count:
        return None
    selected_indices = set(random.sample(eligible, selected_count))

    markers: list[Coord] = []
    used_markers: set[Coord] = set()
    for idx in selected_indices:
        options = [pos for pos in candidates[idx] if pos not in used_markers]
        if not options:
            return None
        pos = random.choice(options)
        markers.append(pos)
        used_markers.add(pos)

    extra_pool = [(idx, pos) for idx in selected_indices for pos in candidates[idx] if pos not in used_markers]
    random.shuffle(extra_pool)
    extra_count = random.randint(0, min(2, len(extra_pool)))
    for idx, pos in extra_pool[:extra_count]:
        markers.append(pos)
        used_markers.add(pos)

    selected_order = list(selected_indices)
    random.shuffle(selected_order)
    distractor_order = [i for i in range(total_count) if i not in selected_indices]
    random.shuffle(distractor_order)

    masks_by_index: list[np.ndarray | None] = [None] * total_count
    for idx, mask in zip(selected_order, selected_masks):
        masks_by_index[idx] = mask
    for idx, mask in zip(distractor_order, distractor_masks):
        masks_by_index[idx] = mask
    assert all(mask is not None for mask in masks_by_index)

    inp = _render_input(height, width, bboxes, [mask for mask in masks_by_index if mask is not None], markers)
    expected = _render_expected(inp, bboxes, selected_indices, template)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-selection ARC input")
