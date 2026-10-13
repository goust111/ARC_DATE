import numpy as np

from arc_utils import colors_present, components_mask, components_mask as _gen_components_mask, get_bg, get_bg as _gen_get_bg, paste as _gen_paste, random_free_location as _gen_random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the color of the largest contiguous non-background object and return it as a solid 3x3 block.

    Input structure:
    - The background is the dominant color.
    - Some foreground colors are scattered clutter split into many 4-connected components.
    - The meaningful foreground objects are colors whose pixels form exactly one 4-connected object.

    Rule:
    - For each non-background color, count its 4-connected monochrome components.
    - Among the colors with exactly one such object, choose the unique largest by pixel area.
    - Fill the fixed 3x3 output grid with that object's color.

    Key insight:
    - Clutter colors may have high total frequency, but they are fragmented; the answer is the biggest whole object color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    candidates: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        comps = components_mask(g == color, connectivity=4)
        if len(comps) == 1:
            candidates.append((int(comps[0].shape[0]), int(color)))

    assert candidates, "expected at least one single-component foreground color"
    max_area = max(area for area, _ in candidates)
    winners = [color for area, color in candidates if area == max_area]
    assert len(winners) == 1, "expected a unique largest single-component object color"

    return np.full((3, 3), winners[0], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas, palette, and validation shape.
GEN_GRID_SHAPE, GEN_OUTPUT_SHAPE = (20, 20), (3, 3)
GEN_GRID_CELLS, GEN_BACKGROUND, GEN_CONNECTIVITY = GEN_GRID_SHAPE[0] * GEN_GRID_SHAPE[1], 0, 4
GEN_PALETTE = tuple(range(1, 10))

# Object counts, areas, density, and rectangle rates.
GEN_OBJECT_COUNT_CHOICES = (3, 4)
GEN_ANSWER_AREA_MIN, GEN_ANSWER_AREA_MAX_EXCLUSIVE = 25, 40
GEN_ANSWER_TO_SECOND_MIN_RATIO, GEN_RECT_PATCH_PROB = 1.18, 0.35
GEN_FOREGROUND_TOTAL_MIN, GEN_FOREGROUND_TOTAL_MAX = 220, 256
GEN_RECT_MAX_SIDE, GEN_TINY_AREA_MAX, GEN_RECT_MIN_SIDE = 20, 4, 2
GEN_OBJECT_BOX_SIDE_MIN, GEN_OBJECT_BOX_SIDE_MAX_EXCLUSIVE, GEN_OBJECT_DENSITY_PERCENT_MIN = 2, 13, 54

# Clutter totals, fragmentation, and fragment shape bounds.
GEN_CLUTTER_COLOR_COUNT, GEN_CLUTTER_EXTRA_ROOM_MIN = 2, 9
GEN_CLUTTER_TOTAL_MIN, GEN_CLUTTER_TOTAL_MAX = 65, 95
GEN_CLUTTER_COMPONENT_MIN, GEN_CLUTTER_COMPONENT_MAX_EXCLUSIVE = 38, 55
GEN_CLUTTER_AVG_FRAGMENT_MIN, GEN_CLUTTER_AVG_FRAGMENT_MAX = 1.5, 2.1
GEN_FRAGMENT_MAX_SIZE_MIN, GEN_FRAGMENT_MAX_SIZE_MAX = 4, 10
GEN_FRAGMENT_SOFT_SIZE_MAX, GEN_FRAGMENT_SMALL_SIZE_LIMIT = 3, 8
GEN_FRAGMENT_SMALL_MAX_SIDE, GEN_FRAGMENT_LARGE_MAX_SIDE = 4, 5
GEN_FRAGMENT_BOX_AREA_SLACK = 6

# Retry and neighborhood policy.
GEN_FRAGMENT_PLACEMENT_RETRIES, GEN_ATTEMPTS = 8, 300
GEN_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_pick_clutter_profile() -> tuple[int, int]:
    total = random.randint(GEN_CLUTTER_TOTAL_MIN, GEN_CLUTTER_TOTAL_MAX)
    valid_counts = [
        k for k in range(GEN_CLUTTER_COMPONENT_MIN, GEN_CLUTTER_COMPONENT_MAX_EXCLUSIVE)
        if GEN_CLUTTER_AVG_FRAGMENT_MIN <= total / k <= GEN_CLUTTER_AVG_FRAGMENT_MAX
        and total - k >= GEN_CLUTTER_EXTRA_ROOM_MIN
    ]
    return total, random.choice(valid_counts)


def _gen_fragment_sizes(total: int, count: int) -> list[int]:
    sizes = [1] * count
    max_size = random.randint(GEN_FRAGMENT_MAX_SIZE_MIN, GEN_FRAGMENT_MAX_SIZE_MAX)
    big_idx = random.randrange(count)
    sizes[big_idx] = max_size
    extra = total - count - (max_size - 1)

    while extra > 0:
        choices = [i for i in range(count) if i != big_idx and sizes[i] < GEN_FRAGMENT_SOFT_SIZE_MAX]
        if not choices:
            choices = [i for i in range(count) if sizes[i] < max_size]
        i = random.choice(choices)
        sizes[i] += 1
        extra -= 1

    random.shuffle(sizes)
    return sorted(sizes, reverse=True)


def _gen_area_tuple(n: int, total: int, hi: int) -> list[int]:
    if n == 2:
        choices = [(a, total - a) for a in range(1, hi + 1) if 1 <= total - a <= hi]
    else:
        choices = [
            (a, b, total - a - b)
            for a in range(1, hi + 1)
            for b in range(1, hi + 1)
            if 1 <= total - a - b <= hi
        ]
    return list(random.choice(choices))


def _gen_pick_object_areas(clutter_totals: list[int]) -> tuple[int, list[int]]:
    clutter_sum = sum(clutter_totals)
    profiles = []
    for object_count in GEN_OBJECT_COUNT_CHOICES:
        for answer_area in range(GEN_ANSWER_AREA_MIN, GEN_ANSWER_AREA_MAX_EXCLUSIVE):
            max_other = int(answer_area / GEN_ANSWER_TO_SECOND_MIN_RATIO)
            min_total = answer_area + object_count - 1
            max_total = answer_area + (object_count - 1) * max_other
            bg_limited_total = GEN_GRID_CELLS - 1 - clutter_sum - max(clutter_totals)
            lo = max(min_total, GEN_FOREGROUND_TOTAL_MIN - clutter_sum)
            hi = min(max_total, GEN_FOREGROUND_TOTAL_MAX - clutter_sum, bg_limited_total)
            if lo <= hi:
                profiles.append((object_count, answer_area, max_other, lo, hi))

    if not profiles:
        return 0, []

    object_count, answer_area, max_other, lo, hi = random.choice(profiles)
    object_total = random.randint(lo, hi)
    other_areas = _gen_area_tuple(object_count - 1, object_total - answer_area, max_other)
    return answer_area, other_areas


def _gen_rect_patch(area: int, color: int) -> np.ndarray | None:
    pairs = []
    for h in range(1, min(GEN_RECT_MAX_SIDE, area) + 1):
        if area % h == 0:
            w = area // h
            if w <= GEN_RECT_MAX_SIDE and (area <= GEN_TINY_AREA_MAX or min(h, w) >= GEN_RECT_MIN_SIDE):
                pairs.append((h, w))
                if h != w:
                    pairs.append((w, h))
    if not pairs:
        return None
    h, w = random.choice(pairs)
    return np.full((h, w), color, dtype=int)


def _gen_grow_connected_coords(
    h: int,
    w: int,
    area: int,
    coords: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    while len(coords) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in coords
            for dr, dc in GEN_NEIGHBOR_OFFSETS
            if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in coords
        }
        coords.add(random.choice(tuple(frontier)))
    return coords


def _gen_object_patch(area: int, color: int) -> np.ndarray:
    if area <= GEN_TINY_AREA_MAX or random.random() < GEN_RECT_PATCH_PROB:
        rect = _gen_rect_patch(area, color)
        if rect is not None:
            return rect

    dims = []
    for h in range(GEN_OBJECT_BOX_SIDE_MIN, GEN_OBJECT_BOX_SIDE_MAX_EXCLUSIVE):
        for w in range(GEN_OBJECT_BOX_SIDE_MIN, GEN_OBJECT_BOX_SIDE_MAX_EXCLUSIVE):
            box_area = h * w
            if (
                box_area >= area
                and area * 100 >= GEN_OBJECT_DENSITY_PERCENT_MIN * box_area
                and h + w - 1 <= area
            ):
                dims.append((h, w))
    if not dims:
        rect = _gen_rect_patch(area, color)
        if rect is not None:
            return rect
        raise ValueError("no object patch dimensions")

    h, w = random.choice(dims)
    cr = random.randrange(h)
    cc = random.randrange(w)
    coords = {(r, cc) for r in range(h)}
    coords.update((cr, c) for c in range(w))
    coords = _gen_grow_connected_coords(h, w, area, coords)
    patch = np.zeros((h, w), dtype=int)
    arr = np.array(tuple(coords), dtype=int)
    patch[arr[:, 0], arr[:, 1]] = color
    return patch


def _gen_trim_patch(mask: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _gen_fragment_patch(size: int, color: int) -> np.ndarray:
    if size == 1:
        return np.array([[color]], dtype=int)

    max_side = GEN_FRAGMENT_SMALL_MAX_SIDE if size <= GEN_FRAGMENT_SMALL_SIZE_LIMIT else GEN_FRAGMENT_LARGE_MAX_SIDE
    dims = [
        (h, w)
        for h in range(1, max_side + 1)
        for w in range(1, max_side + 1)
        if h * w >= size and h * w <= size + GEN_FRAGMENT_BOX_AREA_SLACK and h + w > 2
    ]
    h, w = random.choice(dims)
    coords = {(random.randrange(h), random.randrange(w))}
    coords = _gen_grow_connected_coords(h, w, size, coords)

    mask = np.zeros((h, w), dtype=bool)
    arr = np.array(tuple(coords), dtype=int)
    mask[arr[:, 0], arr[:, 1]] = True
    return _gen_trim_patch(mask, color)


def _gen_valid_fragment_locations(canvas: np.ndarray, patch: np.ndarray, color: int) -> list[tuple[int, int]]:
    """Return placements where this fragment neither overlaps nor 4-touches same-color clutter."""
    coords = [tuple(map(int, rc)) for rc in np.argwhere(patch == color)]
    coord_set = set(coords)
    h, w = patch.shape
    rows, cols = canvas.shape
    locations = []

    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            ok = True
            for dr, dc in coords:
                r = r0 + dr
                c = c0 + dc
                if canvas[r, c] != GEN_BACKGROUND:
                    ok = False
                    break
                for dr, dc in GEN_NEIGHBOR_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if (nr - r0, nc - c0) in coord_set:
                        continue
                    if 0 <= nr < rows and 0 <= nc < cols and canvas[nr, nc] == color:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                locations.append((r0, c0))
    return locations


def _gen_place_clutter(canvas: np.ndarray, color: int, total: int, count: int) -> bool:
    for size in _gen_fragment_sizes(total, count):
        placed = False
        for _ in range(GEN_FRAGMENT_PLACEMENT_RETRIES):
            patch = _gen_fragment_patch(size, color)
            locations = _gen_valid_fragment_locations(canvas, patch, color)
            if locations:
                r, c = random.choice(locations)
                _gen_paste(canvas, patch, r, c, transparent=0)
                placed = True
                break
        if not placed:
            return False
    return True


def _gen_validate(
    grid: np.ndarray,
    answer_color: int,
    object_specs: list[tuple[int, int]],
    clutter_specs: list[tuple[int, int, int]],
) -> bool:
    counts = {int(c): int((grid == c).sum()) for c in np.unique(grid)}
    if _gen_get_bg(grid) != 0 or counts[0] <= max(v for k, v in counts.items() if k != 0):
        return False

    for color, area in object_specs:
        comps = _gen_components_mask(grid == color, connectivity=GEN_CONNECTIVITY)
        if len(comps) != 1 or int(comps[0].shape[0]) != area:
            return False

    for color, total, count in clutter_specs:
        comps = _gen_components_mask(grid == color, connectivity=GEN_CONNECTIVITY)
        sizes = [int(comp.shape[0]) for comp in comps]
        if len(comps) != count or sum(sizes) != total:
            return False
        if not (
            GEN_CLUTTER_AVG_FRAGMENT_MIN <= total / count <= GEN_CLUTTER_AVG_FRAGMENT_MAX
            and GEN_FRAGMENT_MAX_SIZE_MIN <= max(sizes) <= GEN_FRAGMENT_MAX_SIZE_MAX
        ):
            return False

    expected = np.full(GEN_OUTPUT_SHAPE, answer_color, dtype=int)
    return bool(np.array_equal(main(grid), expected))


def _gen_sample_scene_plan() -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], int] | None:
    clutter_profiles = [_gen_pick_clutter_profile(), _gen_pick_clutter_profile()]
    clutter_totals = [total for total, _ in clutter_profiles]
    answer_area, other_areas = _gen_pick_object_areas(clutter_totals)
    if answer_area == 0:
        return None

    object_count = len(other_areas) + 1
    colors = random.sample(GEN_PALETTE, object_count + GEN_CLUTTER_COLOR_COUNT)
    object_colors = colors[:object_count]
    clutter_colors = colors[object_count:]
    answer_color = object_colors[0]
    object_specs = [(answer_color, answer_area)]
    object_specs.extend(zip(object_colors[1:], other_areas))
    clutter_specs = [
        (color, total, count)
        for color, (total, count) in zip(clutter_colors, clutter_profiles)
    ]
    return object_specs, clutter_specs, answer_color


def _gen_place_objects(grid: np.ndarray, object_specs: list[tuple[int, int]]) -> bool:
    for color, area in sorted(object_specs, key=lambda item: item[1], reverse=True):
        patch = _gen_object_patch(area, color)
        try:
            r, c = _gen_random_free_location(grid, patch, transparent=GEN_BACKGROUND)
        except ValueError:
            return False
        _gen_paste(grid, patch, r, c, transparent=GEN_BACKGROUND)
    return True


def _gen_attempt() -> np.ndarray | None:
    plan = _gen_sample_scene_plan()
    if plan is None:
        return None

    object_specs, clutter_specs, answer_color = plan
    grid = np.zeros(GEN_GRID_SHAPE, dtype=int)
    if not _gen_place_objects(grid, object_specs):
        return None
    for color, total, count in clutter_specs:
        if not _gen_place_clutter(grid, color, total, count):
            return None

    if not _gen_validate(grid, answer_color, object_specs, clutter_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate valid 3194b014 input")
