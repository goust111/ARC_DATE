from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6)

GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
HOLLOW_COUNT = 3
SOLID_COUNT_WEIGHTS = ((0, 2), (1, 3), (2, 2))
COMMON_INTERIOR_H_RANGE = (4, 6)
COMMON_INTERIOR_W_RANGE = (4, 6)
EXTRA_INTERIOR_MARGIN_MAX = 1
SOLID_H_RANGE = (3, 7)
SOLID_W_RANGE = (3, 7)
TEMPLATE_AREA_RANGE = (5, 10)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _has_enclosed_background_hole(region: np.ndarray) -> bool:
    for coords in _components(region == BACKGROUND):
        if np.all((coords[:, 0] > 0) & (coords[:, 0] < region.shape[0] - 1) & (coords[:, 1] > 0) & (coords[:, 1] < region.shape[1] - 1)):
            return True
    return False


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {BACKGROUND, GREY}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            h, w = region.shape
            if h < 3 or w < 3:
                continue

            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            border_area = int(border.sum())
            area = int(coords.shape[0])

            if not np.all(region[border] == color):
                continue

            kind = None
            inner = region[1:-1, 1:-1]
            if area == h * w and np.all(region == color):
                kind = "solid"
            elif area == border_area:
                if np.any(inner == color):
                    continue
                if not np.all(np.isin(inner, [BACKGROUND, GREY])):
                    continue
                if not np.any(inner == GREY):
                    continue
                if not _has_enclosed_background_hole(region):
                    continue
                kind = "hollow"

            if kind is None:
                continue

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "kind": kind,
                }
            )
    return objects


def _reconstruct_template_from_hollows(grid: np.ndarray, hollows: list[dict]) -> np.ndarray:
    template_h = max(obj["bbox"][2] - obj["bbox"][0] - 1 for obj in hollows)
    template_w = max(obj["bbox"][3] - obj["bbox"][1] - 1 for obj in hollows)
    template = np.zeros((template_h, template_w), dtype=bool)
    for obj in hollows:
        r0, c0, r1, c1 = obj["bbox"]
        inner_mask = grid[r0 + 1 : r1, c0 + 1 : c1] == GREY
        ih, iw = inner_mask.shape
        template[:ih, :iw] |= inner_mask
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the complete template from complementary fragments inside hollow frames, then stamp it back in each frame's own color.

    Input structure:
    - The grid has a black background plus 3-5 separated single-color rectangles with distinct colors.
    - Exactly three rectangles are one-cell-thick hollow frames whose interiors contain black cells and grey fragment pixels.
    - Any remaining rectangles are solid same-color distractors with no interior background hole.

    Rule:
    - Find all monochrome rectangular objects and select the ones whose complete border encloses an interior background region.
    - For each selected frame, read the grey fragment in the full interior coordinate system, using the interior's top-left corner as the origin.
    - Top-left align those full interior masks, take their union to recover the complete template, then clear each hollow interior to black.
    - Stamp the recovered template back into every hollow interior using that frame's border color, leaving solid rectangles and the black background unchanged.

    Key insight:
    - The grey pixels must be aligned by each frame's entire interior window, not by the fragment's own bounding box; only their union reveals the full shared shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _detect_rectangles(g)
    hollows = [obj for obj in objects if obj["kind"] == "hollow"]
    assert hollows, "expected hollow frames"

    template = _reconstruct_template_from_hollows(g, hollows)
    assert int(template.sum()) > 0, "expected a non-empty reconstructed template"

    out = g.copy()
    for obj in hollows:
        r0, c0, r1, c1 = obj["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = BACKGROUND
        ih, iw = inner.shape
        stamp = template[:ih, :iw]
        inner[stamp] = int(obj["color"])

    return out


def _generate_connected_shape(height: int, width: int, area: int) -> np.ndarray | None:
    if not (1 <= area <= height * width):
        return None

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < height and 0 <= nc < width
    }

    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if bbox_area == area:
        return None
    return mask


def _choose_solid_count() -> int:
    choices, weights = zip(*SOLID_COUNT_WEIGHTS)
    return random.choices(choices, weights=weights, k=1)[0]


def _generate_template_canvas() -> np.ndarray | None:
    common_h = random.randint(*COMMON_INTERIOR_H_RANGE)
    common_w = random.randint(*COMMON_INTERIOR_W_RANGE)

    for _ in range(80):
        box_h = random.randint(3, common_h)
        box_w = random.randint(3, common_w)
        if box_h * box_w <= TEMPLATE_AREA_RANGE[0]:
            continue

        margin_top_max = common_h - box_h
        margin_left_max = common_w - box_w

        require_offset = (margin_top_max > 0 or margin_left_max > 0) and random.random() < 0.75
        if require_offset:
            offset_options = []
            for top in range(margin_top_max + 1):
                for left in range(margin_left_max + 1):
                    if top > 0 or left > 0:
                        offset_options.append((top, left))
            if not offset_options:
                continue
            top, left = random.choice(offset_options)
        else:
            top = random.randint(0, margin_top_max)
            left = random.randint(0, margin_left_max)

        max_area = min(TEMPLATE_AREA_RANGE[1], box_h * box_w - 1)
        if max_area < TEMPLATE_AREA_RANGE[0]:
            continue
        area = random.randint(TEMPLATE_AREA_RANGE[0], max_area)
        local = _generate_connected_shape(box_h, box_w, area)
        if local is None:
            continue

        canvas = np.zeros((common_h, common_w), dtype=bool)
        canvas[top : top + box_h, left : left + box_w] = local
        if not _is_connected(canvas):
            continue
        return canvas
    return None


def _partition_template(template: np.ndarray) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    if len(coords) < HOLLOW_COUNT:
        return None

    global_box = _bbox(template)
    assert global_box is not None
    global_r0, global_c0, _, _ = global_box

    for _ in range(80):
        random.shuffle(coords)
        groups = [[coords[i]] for i in range(HOLLOW_COUNT)]
        for coord in coords[HOLLOW_COUNT:]:
            groups[random.randrange(HOLLOW_COUNT)].append(coord)

        parts = []
        shifted_part_found = False
        for group in groups:
            part = np.zeros_like(template, dtype=bool)
            for r, c in group:
                part[r, c] = True
            box = _bbox(part)
            if box is None:
                break
            if box[0] > global_r0 or box[1] > global_c0:
                shifted_part_found = True
            parts.append(part)
        else:
            if shifted_part_found or global_r0 > 0 or global_c0 > 0:
                return parts
    return None


def _make_hollow_patch(color: int, fragment: np.ndarray, extra_h: int, extra_w: int) -> np.ndarray:
    common_h, common_w = fragment.shape
    inner_h = common_h + extra_h
    inner_w = common_w + extra_w
    patch = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
    patch[1:-1, 1:-1] = BACKGROUND
    patch[1 : 1 + common_h, 1 : 1 + common_w][fragment] = GREY
    return patch


def _make_solid_patch(color: int) -> np.ndarray:
    height = random.randint(*SOLID_H_RANGE)
    width = random.randint(*SOLID_W_RANGE)
    return np.full((height, width), color, dtype=int)


def _fits_with_gap(occupied: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    return not occupied[r0:r1, c0:c1].any()


def _place_patches(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    patch_shapes = [patch.shape for patch in patches]
    total_area = sum(h * w for h, w in patch_shapes)
    max_h = max(h for h, _ in patch_shapes)
    max_w = max(w for _, w in patch_shapes)
    min_side = max(15, max_h, max_w, int(np.ceil(np.sqrt(total_area * 1.8))))

    for _ in range(120):
        grid_h = random.randint(min_side, max(min_side, GRID_H_RANGE[1]))
        grid_w = random.randint(min_side, max(min_side, GRID_W_RANGE[1]))
        if grid_h > GRID_H_RANGE[1] or grid_w > GRID_W_RANGE[1]:
            continue

        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        positions: list[tuple[int, int] | None] = [None] * len(patches)

        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            candidates = []
            for top in range(grid_h - ph + 1):
                for left in range(grid_w - pw + 1):
                    if _fits_with_gap(occupied, top, left, ph, pw, PLACEMENT_GAP):
                        candidates.append((top, left))
            if not candidates:
                break
            top, left = random.choice(candidates)
            positions[idx] = (top, left)
            occupied[top : top + ph, left : left + pw] = True
            mask = patch != BACKGROUND
            grid[top : top + ph, left : left + pw][mask] = patch[mask]
        else:
            return grid, [(int(r), int(c)) for r, c in positions if r is not None and c is not None]
    return None


def _expected_output_from_scene(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for obj in _detect_rectangles(grid):
        if obj["kind"] != "hollow":
            continue
        r0, c0, r1, c1 = obj["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = BACKGROUND
        ih, iw = inner.shape
        inner[template[:ih, :iw]] = int(obj["color"])
    return out


def _validate_generated_scene(grid: np.ndarray, template: np.ndarray) -> bool:
    objects = _detect_rectangles(grid)
    if len(objects) < 3 or len(objects) > 5:
        return False

    hollows = [obj for obj in objects if obj["kind"] == "hollow"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    if len(hollows) != HOLLOW_COUNT or len(solids) > 2:
        return False

    reconstructed = _reconstruct_template_from_hollows(grid, hollows)
    if not np.array_equal(reconstructed, template):
        return False
    if not _is_connected(reconstructed):
        return False

    template_box = _bbox(reconstructed)
    if template_box is None:
        return False
    area = int(reconstructed.sum())
    if not (TEMPLATE_AREA_RANGE[0] <= area <= TEMPLATE_AREA_RANGE[1]):
        return False
    if (template_box[2] - template_box[0] + 1) * (template_box[3] - template_box[1] + 1) == area:
        return False

    recovered_parts = []
    for obj in hollows:
        r0, c0, r1, c1 = obj["bbox"]
        part = grid[r0 + 1 : r1, c0 + 1 : c1] == GREY
        if int(part.sum()) == 0 or int(part.sum()) == area:
            return False
        recovered_parts.append(part)

    max_h = max(part.shape[0] for part in recovered_parts)
    max_w = max(part.shape[1] for part in recovered_parts)
    coverage = np.zeros((max_h, max_w), dtype=int)
    for part in recovered_parts:
        ph, pw = part.shape
        coverage[:ph, :pw] += part.astype(int)
    if not np.array_equal(coverage > 0, reconstructed):
        return False
    if np.any(coverage[reconstructed] != 1):
        return False

    expected = _expected_output_from_scene(grid, template)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    common_template = _generate_template_canvas()
    if common_template is None:
        return None

    parts = _partition_template(common_template)
    if parts is None:
        return None

    solid_count = _choose_solid_count()
    colors = random.sample(FRAME_COLORS, HOLLOW_COUNT + solid_count)
    hollow_colors = colors[:HOLLOW_COUNT]
    solid_colors = colors[HOLLOW_COUNT:]

    patches: list[np.ndarray] = []
    hollow_patches: list[np.ndarray] = []
    for color, part in zip(hollow_colors, parts):
        extra_h = random.randint(0, EXTRA_INTERIOR_MARGIN_MAX)
        extra_w = random.randint(0, EXTRA_INTERIOR_MARGIN_MAX)
        patch = _make_hollow_patch(color, part, extra_h, extra_w)
        patches.append(patch)
        hollow_patches.append(patch)

    padded_template = np.zeros(
        (
            max(patch.shape[0] - 2 for patch in hollow_patches),
            max(patch.shape[1] - 2 for patch in hollow_patches),
        ),
        dtype=bool,
    )
    padded_template[: common_template.shape[0], : common_template.shape[1]] = common_template

    for color in solid_colors:
        patches.append(_make_solid_patch(color))

    placed = _place_patches(patches)
    if placed is None:
        return None
    grid, _positions = placed

    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    if not _validate_generated_scene(grid, padded_template):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
