from __future__ import annotations

import random

import numpy as np


BLACK = 0
WORK_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 120
LAYOUT_MARGIN = 1
LAYOUT_GAP = 1
MAX_OUTER_MARGIN = 3


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    runs = []
    start = end = indices[0]
    for value in indices[1:]:
        if value == end + 1:
            end = value
        else:
            runs.append((start, end))
            start = end = value
    runs.append((start, end))
    return runs


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert bool(rows.any()) and bool(cols.any()), "expected a non-empty mask"
    return mask[rows][:, cols].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _trim_mask(mask.astype(bool))
    return min(_mask_key(_trim_mask(np.rot90(base, k))) for k in range(4))


def _components(
    region: np.ndarray,
    ignore_colors: set[int],
    offset: tuple[int, int] = (0, 0),
) -> list[dict[str, object]]:
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    off_r, off_c = offset
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if seen[r, c] or color in ignore_colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_local = np.asarray(coords, dtype=int)
            r0 = int(coords_local[:, 0].min())
            c0 = int(coords_local[:, 1].min())
            r1 = int(coords_local[:, 0].max())
            c1 = int(coords_local[:, 1].max())
            coords_global = coords_local + np.array([off_r, off_c], dtype=int)
            components.append(
                {
                    "color": color,
                    "coords_local": coords_local,
                    "coords_global": coords_global,
                    "bbox_local": (r0, c0, r1, c1),
                }
            )

    return components


def _component_mask(component: dict[str, object]) -> np.ndarray:
    coords = np.asarray(component["coords_local"], dtype=int)
    r0, c0, r1, c1 = map(int, component["bbox_local"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _region_info(region: np.ndarray, bg: int, offset: tuple[int, int]) -> dict[str, object]:
    comps = _components(region, ignore_colors={bg, BLACK}, offset=offset)
    colors = {int(comp["color"]) for comp in comps}
    return {"array": region, "components": comps, "colors": colors, "offset": offset}


def _is_dictionary_region(info: dict[str, object]) -> bool:
    comps = list(info["components"])
    colors = set(info["colors"])
    return len(comps) >= 2 and len(colors) == len(comps)


def _is_work_region(info: dict[str, object]) -> bool:
    comps = list(info["components"])
    colors = set(info["colors"])
    return len(comps) >= 1 and len(colors) == 1


def _find_partition(grid: np.ndarray, bg: int) -> dict[str, object]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []

    black_rows = [r for r in range(h) if bool(np.all(grid[r, :] == BLACK))]
    for start, end in _runs(black_rows):
        if start == 0 or end == h - 1:
            continue
        top = _region_info(grid[:start, :], bg, (0, 0))
        bottom = _region_info(grid[end + 1 :, :], bg, (end + 1, 0))
        if _is_dictionary_region(top) and _is_work_region(bottom):
            candidates.append({"orientation": "horizontal", "separator": (start, end), "dict": top, "work": bottom})
        if _is_dictionary_region(bottom) and _is_work_region(top):
            candidates.append({"orientation": "horizontal", "separator": (start, end), "dict": bottom, "work": top})

    black_cols = [c for c in range(w) if bool(np.all(grid[:, c] == BLACK))]
    for start, end in _runs(black_cols):
        if start == 0 or end == w - 1:
            continue
        left = _region_info(grid[:, :start], bg, (0, 0))
        right = _region_info(grid[:, end + 1 :], bg, (0, end + 1))
        if _is_dictionary_region(left) and _is_work_region(right):
            candidates.append({"orientation": "vertical", "separator": (start, end), "dict": left, "work": right})
        if _is_dictionary_region(right) and _is_work_region(left):
            candidates.append({"orientation": "vertical", "separator": (start, end), "dict": right, "work": left})

    assert len(candidates) == 1, "expected a unique separator split"
    return candidates[0]


def _downscale_mask(mask: np.ndarray, k: int) -> np.ndarray | None:
    h, w = mask.shape
    if k <= 0 or h % k or w % k:
        return None
    reduced_h = h // k
    reduced_w = w // k
    blocks = mask.reshape(reduced_h, k, reduced_w, k)
    coarse = blocks[:, 0, :, 0]
    if not bool(np.all(blocks == coarse[:, None, :, None])):
        return None
    return _trim_mask(coarse)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enlarged work-region object by matching its downscaled shape to the dictionary region.

    Input structure:
    - The most common color is the background.
    - A full black separator row or column splits the grid into a dictionary region and a work region.
    - The dictionary region contains several small monochrome connected objects, each with a unique color and a unique shape up to rotation.
    - The work region contains larger monochrome objects of one neutral color; each one is an integer-scaled copy of some dictionary shape, possibly rotated.

    Rule:
    - Identify which side of the separator is the dictionary by its multiple differently colored objects, and which side is the work area by its single shared foreground color.
    - Build a shape-to-color lookup from dictionary objects using tight boolean masks compared up to 0/90/180/270 degree rotation.
    - Infer the single scale factor k that lets every work object shrink cleanly from k-by-k monochrome blocks back to a dictionary-shape mask.
    - Recolor each work object with the color of its matching dictionary entry, while leaving the background, separator, and dictionary region unchanged.

    Key insight:
    - The work objects are not new shapes; they are rotated, uniformly enlarged copies of the dictionary silhouettes, so the task is solved by undoing one common scale factor and then matching only by rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    assert bg != BLACK, "background must differ from the black separator"

    partition = _find_partition(g, bg)
    dict_components = list(partition["dict"]["components"])
    work_components = list(partition["work"]["components"])
    assert dict_components and work_components

    dictionary: dict[tuple[str, ...], int] = {}
    for component in dict_components:
        mask = _component_mask(component)
        signature = _rotation_signature(mask)
        color = int(component["color"])
        assert signature not in dictionary, "dictionary shapes must be unique up to rotation"
        dictionary[signature] = color

    work_masks = [_component_mask(component) for component in work_components]
    max_k = max(max(mask.shape) for mask in work_masks)
    valid_matches: list[tuple[int, list[int]]] = []

    for k in range(1, max_k + 1):
        colors: list[int] = []
        ok = True
        for mask in work_masks:
            shrunk = _downscale_mask(mask, k)
            if shrunk is None:
                ok = False
                break
            signature = _rotation_signature(shrunk)
            if signature not in dictionary:
                ok = False
                break
            colors.append(dictionary[signature])
        if ok:
            valid_matches.append((k, colors))

    assert len(valid_matches) == 1, "scale factor must be uniquely determined"
    _k, target_colors = valid_matches[0]

    out = g.copy()
    for component, color in zip(work_components, target_colors):
        coords = np.asarray(component["coords_global"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _grow_random_shape(max_dim: int) -> np.ndarray | None:
    h = random.randint(2, max_dim)
    w = random.randint(2, max_dim)
    target = random.randint(3, h * w)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_mask(mask)
    if min(mask.shape) < 2:
        return None
    return mask


def _choose_dictionary_shapes(count: int, max_dim: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    signatures: set[tuple[str, ...]] = set()
    for _ in range(SHAPE_ATTEMPTS):
        if len(shapes) == count:
            return shapes
        candidate = _grow_random_shape(max_dim)
        if candidate is None:
            continue
        signature = _rotation_signature(candidate)
        if signature in signatures:
            continue
        signatures.add(signature)
        shapes.append(candidate)
    return None


def _scale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = canvas[top : top + h, left : left + w]
    view[mask] = int(color)


def _render_region(objects: list[dict[str, object]], bg: int) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    items = [dict(obj) for obj in objects]
    random.shuffle(items)
    n = len(items)
    candidates = []
    for rows in range(1, min(3, n) + 1):
        cols = (n + rows - 1) // rows
        if (rows - 1) * cols >= n:
            continue
        slots: list[list[dict[str, object]]] = [[] for _ in range(rows)]
        for idx, item in enumerate(items):
            slots[idx // cols].append(item)
        row_heights = [max(int(obj["mask"].shape[0]) for obj in row) + 1 for row in slots]
        col_widths = []
        for col in range(cols):
            col_items = [items[idx] for idx in range(n) if idx % cols == col]
            col_widths.append(max(int(obj["mask"].shape[1]) for obj in col_items) + 1)
        height = 2 * LAYOUT_MARGIN + sum(row_heights) + LAYOUT_GAP * (rows - 1)
        width = 2 * LAYOUT_MARGIN + sum(col_widths) + LAYOUT_GAP * (cols - 1)
        candidates.append((max(height, width), height * width, rows, cols, row_heights, col_widths, items.copy()))

    candidates.sort(key=lambda item: (item[0], item[1]))
    top_candidates = candidates[: min(2, len(candidates))]
    _, _, rows, cols, row_heights, col_widths, ordered = random.choice(top_candidates)
    height = 2 * LAYOUT_MARGIN + sum(row_heights) + LAYOUT_GAP * (rows - 1)
    width = 2 * LAYOUT_MARGIN + sum(col_widths) + LAYOUT_GAP * (cols - 1)

    canvas = np.full((height, width), bg, dtype=int)
    placements: list[dict[str, object]] = []
    row_tops = []
    current = LAYOUT_MARGIN
    for size in row_heights:
        row_tops.append(current)
        current += size + LAYOUT_GAP
    col_lefts = []
    current = LAYOUT_MARGIN
    for size in col_widths:
        col_lefts.append(current)
        current += size + LAYOUT_GAP

    for idx, item in enumerate(ordered):
        row = idx // cols
        col = idx % cols
        mask = np.asarray(item["mask"], dtype=bool)
        slot_h = row_heights[row]
        slot_w = col_widths[col]
        slack_h = slot_h - mask.shape[0]
        slack_w = slot_w - mask.shape[1]
        top = row_tops[row] + random.randint(0, slack_h)
        left = col_lefts[col] + random.randint(0, slack_w)
        _paint_mask(canvas, mask, top, left, int(item["color"]))
        placements.append({**item, "top": top, "left": left})

    return canvas, placements


def _combine_regions(
    dict_region: np.ndarray,
    work_region: np.ndarray,
    bg: int,
    orientation: str,
    dict_first: bool,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    dict_h, dict_w = dict_region.shape
    work_h, work_w = work_region.shape

    if orientation == "vertical":
        content_h = max(dict_h, work_h)
        content_w = dict_w + work_w + 1
        if content_h > MAX_GRID_SIDE or content_w > MAX_GRID_SIDE:
            return None
        extra_h = MAX_GRID_SIDE - content_h
        extra_w = MAX_GRID_SIDE - content_w
        top = random.randint(0, min(MAX_OUTER_MARGIN, extra_h))
        bottom = random.randint(0, min(MAX_OUTER_MARGIN, extra_h - top))
        left = random.randint(0, min(MAX_OUTER_MARGIN, extra_w))
        right = random.randint(0, min(MAX_OUTER_MARGIN, extra_w - left))
        height = content_h + top + bottom
        width = content_w + left + right
        canvas = np.full((height, width), bg, dtype=int)

        if dict_first:
            dict_left = left
            sep = dict_left + dict_w
            work_left = sep + 1
        else:
            work_left = left
            sep = work_left + work_w
            dict_left = sep + 1

        canvas[:, sep] = BLACK
        dict_top = top + random.randint(0, content_h - dict_h)
        work_top = top + random.randint(0, content_h - work_h)
        canvas[dict_top : dict_top + dict_h, dict_left : dict_left + dict_w] = dict_region
        canvas[work_top : work_top + work_h, work_left : work_left + work_w] = work_region
        return canvas, (dict_top, dict_left), (work_top, work_left)

    content_h = dict_h + work_h + 1
    content_w = max(dict_w, work_w)
    if content_h > MAX_GRID_SIDE or content_w > MAX_GRID_SIDE:
        return None
    extra_h = MAX_GRID_SIDE - content_h
    extra_w = MAX_GRID_SIDE - content_w
    top = random.randint(0, min(MAX_OUTER_MARGIN, extra_h))
    bottom = random.randint(0, min(MAX_OUTER_MARGIN, extra_h - top))
    left = random.randint(0, min(MAX_OUTER_MARGIN, extra_w))
    right = random.randint(0, min(MAX_OUTER_MARGIN, extra_w - left))
    height = content_h + top + bottom
    width = content_w + left + right
    canvas = np.full((height, width), bg, dtype=int)

    if dict_first:
        dict_top = top
        sep = dict_top + dict_h
        work_top = sep + 1
    else:
        work_top = top
        sep = work_top + work_h
        dict_top = sep + 1

    canvas[sep, :] = BLACK
    dict_left = left + random.randint(0, content_w - dict_w)
    work_left = left + random.randint(0, content_w - work_w)
    canvas[dict_top : dict_top + dict_h, dict_left : dict_left + dict_w] = dict_region
    canvas[work_top : work_top + work_h, work_left : work_left + work_w] = work_region
    return canvas, (dict_top, dict_left), (work_top, work_left)


def _build_expected_output(
    grid: np.ndarray,
    work_placements: list[dict[str, object]],
    work_origin: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    origin_r, origin_c = work_origin
    for placement in work_placements:
        mask = np.asarray(placement["mask"], dtype=bool)
        top = origin_r + int(placement["top"])
        left = origin_c + int(placement["left"])
        target_color = int(placement["target_color"])
        h, w = mask.shape
        view = out[top : top + h, left : left + w]
        view[mask] = target_color
    return out


def _sample_work_indices(dict_count: int, work_count: int) -> list[int]:
    base = list(range(dict_count))
    if work_count <= dict_count:
        indices = random.sample(base, work_count)
    else:
        indices = base + random.choices(base, k=work_count - dict_count)
    if work_count >= 2 and dict_count >= 2 and len(set(indices)) < 2:
        indices[0], indices[1] = 0, 1
    random.shuffle(indices)
    return indices


def _build_sample() -> np.ndarray | None:
    k = random.choice([2, 3])
    orientation = random.choice(["vertical", "horizontal"])
    dict_first = bool(random.getrandbits(1))

    if k == 3:
        dict_count = random.randint(2, 3)
        work_count = random.randint(2, 3)
        max_dim = 3
    else:
        dict_count = random.randint(2, 4)
        work_count = random.randint(2, 4)
        max_dim = 4

    bg = random.choice([color for color in PALETTE if color != WORK_COLOR])
    dict_colors = random.sample([color for color in PALETTE if color not in {bg, WORK_COLOR}], dict_count)
    shapes = _choose_dictionary_shapes(dict_count, max_dim)
    if shapes is None:
        return None

    dict_objects = [{"mask": mask, "color": color} for mask, color in zip(shapes, dict_colors)]
    dict_region_data = _render_region(dict_objects, bg)
    if dict_region_data is None:
        return None
    dict_region, _dict_placements = dict_region_data

    work_indices = _sample_work_indices(dict_count, work_count)
    work_objects: list[dict[str, object]] = []
    for index in work_indices:
        base_mask = shapes[index]
        rotated = np.rot90(base_mask, random.randint(0, 3))
        scaled = _scale_mask(rotated, k)
        work_objects.append({"mask": scaled, "color": WORK_COLOR, "target_color": dict_colors[index]})

    work_region_data = _render_region(work_objects, bg)
    if work_region_data is None:
        return None
    work_region, work_placements = work_region_data

    combined = _combine_regions(dict_region, work_region, bg, orientation, dict_first)
    if combined is None:
        return None
    grid, _dict_origin, work_origin = combined

    if grid.ndim != 2 or max(grid.shape) > MAX_GRID_SIDE:
        return None
    if _background_color(grid) != bg:
        return None

    expected = _build_expected_output(grid, work_placements, work_origin)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _build_sample()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input grid")
