import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    rows = np.any(patch != bg, axis=1)
    cols = np.any(patch != bg, axis=0)
    return np.asarray(patch[rows][:, cols], dtype=int)


def _crop_object_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    h, w = patch.shape
    window = dst[r0 : r0 + h, c0 : c0 + w]
    window[mask] = patch[mask]


def _is_straight_segment(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        return int(coords.shape[0]) == c1 - c0 + 1
    if c0 == c1:
        return int(coords.shape[0]) == r1 - r0 + 1
    return False


def _infer_separator_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    strict_candidates: list[tuple[int, int, int]] = []
    loose_candidates: list[tuple[int, int, int]] = []

    for color in colors:
        components = _components_from_mask(grid == color)
        if not components:
            continue
        line_flags = [_is_straight_segment(comp) for comp in components]
        total_cells = sum(int(comp.shape[0]) for comp in components)
        loose_candidates.append((sum(line_flags), total_cells, color))
        if all(line_flags):
            strict_candidates.append((len(components), total_cells, color))

    if strict_candidates:
        multi = [item for item in strict_candidates if item[0] >= 2]
        pool = multi if multi else strict_candidates
        pool.sort()
        return int(pool[-1][2])

    assert loose_candidates, "expected at least one non-background color"
    loose_candidates.sort()
    return int(loose_candidates[-1][2])


def _extract_objects(grid: np.ndarray, bg: int, separator: int) -> list[dict]:
    object_mask = (grid != bg) & (grid != separator)
    components = _components_from_mask(object_mask)
    objects: list[dict] = []
    for coords in components:
        patch, box = _crop_object_patch(grid, coords, bg)
        objects.append({"coords": coords, "bbox": box, "patch": patch})
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    return objects


def _count_contacts(objects: list[dict], segment_components: list[np.ndarray], shape: tuple[int, int]) -> None:
    h, w = shape
    segment_ids = np.full((h, w), -1, dtype=int)
    for seg_id, coords in enumerate(segment_components):
        segment_ids[coords[:, 0], coords[:, 1]] = seg_id

    for obj in objects:
        touched: set[int] = set()
        for r, c in obj["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    seg_id = int(segment_ids[nr, nc])
                    if seg_id >= 0:
                        touched.add(seg_id)
        obj["touches"] = len(touched)
        assert 0 <= obj["touches"] <= 2, "objects should touch 0, 1, or 2 separator segments"


def _pack_groups(objects: list[dict], bg: int) -> np.ndarray:
    groups = {0: [], 1: [], 2: []}
    for obj in sorted(objects, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        groups[int(obj["touches"])].append(obj["patch"])

    bands: list[np.ndarray] = []
    for touch_count in (0, 1, 2):
        patches = groups[touch_count]
        if not patches:
            bands.append(np.full((1, 1), bg, dtype=int))
            continue
        band_h = max(int(patch.shape[0]) for patch in patches)
        band_w = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
        band = np.full((band_h, band_w), bg, dtype=int)
        col = 0
        for patch in patches:
            _paste(band, patch, 0, col, bg)
            col += int(patch.shape[1]) + 1
        bands.append(band)

    out_h = sum(int(band.shape[0]) for band in bands)
    out_w = max(int(band.shape[1]) for band in bands)
    out = np.full((out_h, out_w), bg, dtype=int)
    row = 0
    for band in bands:
        bh, bw = band.shape
        _paste(out, band, row, 0, bg)
        row += bh
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    separator = _infer_separator_color(g, bg)
    segment_components = _components_from_mask(g == separator)
    assert segment_components, "expected separator segments"
    assert all(_is_straight_segment(comp) for comp in segment_components), "separator color must form straight segments"

    objects = _extract_objects(g, bg, separator)
    assert objects, "expected at least one foreground object"
    _count_contacts(objects, segment_components, g.shape)
    return _pack_groups(objects, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group foreground objects by how many separator segments they touch, then
    repack each group into its own horizontal band from top (0) to bottom (2).
    """
    return _solve(grid)


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))
GEN_RETRIES = 400

GROUP_COUNT_RANGE = (1, 2)
OBJECT_AREA_RANGE = (3, 7)
OBJECT_BOX_MIN = 2
OBJECT_BOX_MAX = 4

MOTIF_CANVAS = 6
SLOT_H = 9
SLOT_W = 9
OUTER_BORDER_RANGE = (0, 1)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_patch(cells: set[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    patch = np.full((h, w), bg, dtype=int)
    rows, cols = zip(*cells)
    patch[rows, cols] = color
    return patch


def _random_object_patch(color: int, bg: int) -> np.ndarray:
    for _ in range(200):
        box_h = random.randint(OBJECT_BOX_MIN, OBJECT_BOX_MAX)
        box_w = random.randint(OBJECT_BOX_MIN, OBJECT_BOX_MAX)
        max_area = min(OBJECT_AREA_RANGE[1], box_h * box_w)
        min_area = min(max_area, max(OBJECT_AREA_RANGE[0], (box_h * box_w + 1) // 2))
        area = random.randint(min_area, max_area)

        seed = (random.randrange(box_h), random.randrange(box_w))
        cells = {seed}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            weights = []
            for r, c in frontier:
                degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
                weights.append((degree + 1) ** 2)
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != area:
            continue

        patch = _cells_to_patch(cells, color, bg)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        patch = np.rot90(patch, random.randrange(4))
        patch = _trim_patch(np.asarray(patch, dtype=int), bg)
        if min(patch.shape) < 2:
            continue
        return patch

    raise RuntimeError("failed to generate a non-degenerate object patch")


def _choose_extensions(base_start: int, base_end: int, low: int, high: int) -> tuple[int, int]:
    max_left = base_start - low
    max_right = high - base_end
    options = [
        (left, right)
        for left in range(max_left + 1)
        for right in range(max_right + 1)
        if (base_end + right) - (base_start - left) + 1 >= 2
    ]
    assert options, "segment must have length at least 2"
    return random.choice(options)


def _finalize_motif(canvas: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(canvas != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    extra_top = random.randint(0, int(r0))
    extra_left = random.randint(0, int(c0))
    extra_bottom = random.randint(0, int(canvas.shape[0] - 1 - r1))
    extra_right = random.randint(0, int(canvas.shape[1] - 1 - c1))
    return np.asarray(
        canvas[r0 - extra_top : r1 + extra_bottom + 1, c0 - extra_left : c1 + extra_right + 1],
        dtype=int,
    )


def _build_motif(object_patch: np.ndarray, object_color: int, separator: int, touches: int, bg: int) -> dict:
    sides: tuple[str, ...]
    if touches == 0:
        sides = ()
    elif touches == 1:
        sides = (random.choice(("top", "bottom", "left", "right")),)
    else:
        sides = random.choice((("top", "bottom"), ("left", "right")))

    ph, pw = object_patch.shape
    top_need = int("top" in sides)
    bottom_need = int("bottom" in sides)
    left_need = int("left" in sides)
    right_need = int("right" in sides)

    for _ in range(100):
        r_min = top_need
        r_max = MOTIF_CANVAS - ph - bottom_need
        c_min = left_need
        c_max = MOTIF_CANVAS - pw - right_need
        if r_min > r_max or c_min > c_max:
            break

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        canvas = np.full((MOTIF_CANVAS, MOTIF_CANVAS), bg, dtype=int)
        _paste(canvas, object_patch, r0, c0, bg)

        obj_mask = object_patch == object_color
        top_cols = np.flatnonzero(obj_mask[0])
        bottom_cols = np.flatnonzero(obj_mask[-1])
        left_rows = np.flatnonzero(obj_mask[:, 0])
        right_rows = np.flatnonzero(obj_mask[:, -1])

        ok = True
        if "top" in sides:
            base_start = c0 + int(top_cols.min())
            base_end = c0 + int(top_cols.max())
            try:
                ext_left, ext_right = _choose_extensions(base_start, base_end, 0, MOTIF_CANVAS - 1)
            except AssertionError:
                ok = False
            else:
                canvas[r0 - 1, base_start - ext_left : base_end + ext_right + 1] = separator

        if ok and "bottom" in sides:
            base_start = c0 + int(bottom_cols.min())
            base_end = c0 + int(bottom_cols.max())
            try:
                ext_left, ext_right = _choose_extensions(base_start, base_end, 0, MOTIF_CANVAS - 1)
            except AssertionError:
                ok = False
            else:
                canvas[r0 + ph, base_start - ext_left : base_end + ext_right + 1] = separator

        if ok and "left" in sides:
            base_start = r0 + int(left_rows.min())
            base_end = r0 + int(left_rows.max())
            try:
                ext_top, ext_bottom = _choose_extensions(base_start, base_end, 0, MOTIF_CANVAS - 1)
            except AssertionError:
                ok = False
            else:
                canvas[base_start - ext_top : base_end + ext_bottom + 1, c0 - 1] = separator

        if ok and "right" in sides:
            base_start = r0 + int(right_rows.min())
            base_end = r0 + int(right_rows.max())
            try:
                ext_top, ext_bottom = _choose_extensions(base_start, base_end, 0, MOTIF_CANVAS - 1)
            except AssertionError:
                ok = False
            else:
                canvas[base_start - ext_top : base_end + ext_bottom + 1, c0 + pw] = separator

        if not ok:
            continue

        motif = _finalize_motif(canvas, bg)
        object_coords = np.argwhere(motif == object_color)
        object_box = _bbox(object_coords)
        return {
            "motif": motif,
            "object_patch": np.asarray(object_patch, dtype=int),
            "object_bbox": object_box,
            "touches": touches,
        }

    raise RuntimeError("failed to build a valid motif")


def _assemble_input(motifs: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    n = len(motifs)
    cols_options = [2, 3] if n >= 3 else [2]
    feasible_cols = []
    for cols in cols_options:
        rows = (n + cols - 1) // cols
        max_h = rows * SLOT_H + 2 * OUTER_BORDER_RANGE[1]
        max_w = cols * SLOT_W + 2 * OUTER_BORDER_RANGE[1]
        if max_h <= 30 and max_w <= 30:
            feasible_cols.append(cols)
    cols = random.choice(feasible_cols)
    rows = (n + cols - 1) // cols

    top = random.randint(*OUTER_BORDER_RANGE)
    bottom = random.randint(*OUTER_BORDER_RANGE)
    left = random.randint(*OUTER_BORDER_RANGE)
    right = random.randint(*OUTER_BORDER_RANGE)
    grid = np.full((top + rows * SLOT_H + bottom, left + cols * SLOT_W + right), bg, dtype=int)

    slot_ids = random.sample(range(rows * cols), n)
    objects: list[dict] = []
    for motif, slot_id in zip(motifs, slot_ids):
        mh, mw = motif["motif"].shape
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        off_r = random.randint(1, SLOT_H - mh - 1)
        off_c = random.randint(1, SLOT_W - mw - 1)
        r0 = top + slot_r * SLOT_H + off_r
        c0 = left + slot_c * SLOT_W + off_c
        _paste(grid, motif["motif"], r0, c0, bg)

        or0, oc0, or1, oc1 = motif["object_bbox"]
        objects.append(
            {
                "patch": motif["object_patch"],
                "bbox": (r0 + or0, c0 + oc0, r0 + or1, c0 + oc1),
                "touches": motif["touches"],
            }
        )

    return grid, objects


def _expected_output(objects: list[dict], bg: int) -> np.ndarray:
    return _pack_groups(objects, bg)


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    return int(np.count_nonzero(grid == bg)) > grid.size // 2


def _gen_attempt() -> np.ndarray | None:
    group_counts = {touches: random.randint(*GROUP_COUNT_RANGE) for touches in (0, 1, 2)}
    total_objects = sum(group_counts.values())
    if total_objects > len(COLOR_VALUES) - 2:
        return None

    bg, separator = random.sample(COLOR_VALUES, 2)
    object_colors = random.sample([c for c in COLOR_VALUES if c not in {bg, separator}], total_objects)

    motifs: list[dict] = []
    color_idx = 0
    for touches in (0, 1, 2):
        for _ in range(group_counts[touches]):
            color = object_colors[color_idx]
            color_idx += 1
            object_patch = _random_object_patch(color, bg)
            motifs.append(_build_motif(object_patch, color, separator, touches, bg))

    random.shuffle(motifs)
    grid, expected_objects = _assemble_input(motifs, bg)

    if not _background_is_dominant(grid, bg):
        return None
    if _infer_separator_color(grid, bg) != separator:
        return None

    objects = _extract_objects(grid, bg, separator)
    segments = _components_from_mask(grid == separator)
    if len(objects) != total_objects or not all(_is_straight_segment(comp) for comp in segments):
        return None

    _count_contacts(objects, segments, grid.shape)
    extracted_counts = sorted(int(obj["touches"]) for obj in objects)
    expected_counts = sorted(group_counts[0] * [0] + group_counts[1] * [1] + group_counts[2] * [2])
    if extracted_counts != expected_counts:
        return None

    expected = _expected_output(expected_objects, bg)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
