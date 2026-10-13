import numpy as np

from arc_utils import Color, bbox_coords, components_mask, components_mask as _gen_components_mask, paste as _gen_paste, random_free_location as _gen_random_free_location
import itertools
import random


def _rectangle_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(integral[r1 + 1, c1 + 1] - integral[r0, c1 + 1] - integral[r1 + 1, c0] + integral[r0, c0])


def _max_solid_rectangles(component: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(component)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[component[:, 0] - r0, component[:, 1] - c0] = True

    integral = np.pad(local.astype(int).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    best_area = 0
    best: list[tuple[int, int, int, int]] = []
    h, w = local.shape

    for lr0 in range(h - 1):
        for lr1 in range(lr0 + 1, h):
            height = lr1 - lr0 + 1
            for lc0 in range(w - 1):
                for lc1 in range(lc0 + 1, w):
                    width = lc1 - lc0 + 1
                    area = height * width
                    if area < best_area:
                        continue
                    if _rectangle_sum(integral, lr0, lc0, lr1, lc1) != area:
                        continue
                    rect = (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1)
                    if area > best_area:
                        best_area = area
                        best = [rect]
                    else:
                        best.append(rect)

    assert all(0 <= rr0 <= rr1 < shape[0] and 0 <= cc0 <= cc1 < shape[1] for rr0, cc0, rr1, cc1 in best)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the largest rectangular black pocket(s) in each black component red.

    Input structure
    - The input is a grey field with black empty regions; no red pixels are
      present initially.
    - Black regions are interpreted as 4-connected components.
    - Some black components contain a solid rectangular subregion with both
      height and width at least 2.

    Rule
    - For each 4-connected black component, enumerate all solid all-black
      rectangles with height >= 2 and width >= 2.
    - If a component contains such rectangles, recolor every maximum-area
      rectangle in that component to red.
    - Leave grey pixels and black cells outside those selected rectangles
      unchanged.

    Key insight
    - The output is not the union of every black 2x2 square; within each black
      component, smaller overlapping rectangles are ignored unless they tie for
      the largest area.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert colors == {Color.BLACK, Color.GREY}, "expected only black empty regions on a grey field"

    out = g.copy()
    selected = np.zeros(g.shape, dtype=bool)
    for component in components_mask(g == Color.BLACK, connectivity=4):
        for r0, c0, r1, c1 in _max_solid_rectangles(component, g.shape):
            selected[r0 : r1 + 1, c0 : c1 + 1] = True

    assert selected.any(), "expected at least one rectangular black pocket"
    out[selected] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Palette.
_GEN_BLACK = 0
_GEN_RED = 2
_GEN_GREY = 5

# Canvas size and validation bounds.
_GEN_GRID_SHAPE = (18, 18)
_GEN_BLACK_COUNT_RANGE = (117, 145)
_GEN_RED_COUNT_RANGE = (16, 24)
_GEN_TOTAL_COMPONENT_RANGE = (32, 46)
_GEN_QUAL_COMPONENT_COUNT_RANGE = (3, 5)
_GEN_NOISE_COMPONENT_COUNT_RANGE = (28, 43)
_GEN_RETRY_LIMIT = 300

# Qualifying component counts and rectangle areas.
_GEN_QUAL_COMPONENT_COUNT_CHOICES = [3, 4, 5]
_GEN_QUAL_COMPONENT_COUNT_WEIGHTS = [1, 2, 2]
_GEN_RECT_AREA_CHOICES = (4, 6, 8)
_GEN_AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(4, 2)],
}

# Qualifying patch growth.
_GEN_QUAL_PATCH_SPAN = 5
_GEN_QUAL_COMPONENT_SIZE_RANGE = (6, 17)
_GEN_QUAL_EXTRA_CELL_MIN = 2
_GEN_QUAL_EXTRA_CELL_CAP = 12
_GEN_ARM_COUNT_RANGE = (2, 6)

# Noise component sizing.
_GEN_NOISE_COMPONENT_SIZE_RANGE = (1, 16)
_GEN_NOISE_TOTAL_DENSITY_CAP_PER_COMPONENT = 4
_GEN_NOISE_SINGLETON_FRACTION_RANGE = (0.40, 0.60)
_GEN_NOISE_CAP_CHOICES = [2, 3, 4, 5, 8, 12, 16]
_GEN_NOISE_CAP_WEIGHTS = [24, 28, 19, 13, 8, 5, 3]
_GEN_NOISE_CAP_GROW_STEP_RANGE = (1, 5)
_GEN_PATH_PATCH_SPAN = 8

# Placement and movement.
_GEN_PLACEMENT_PADDING = 1
_GEN_PLACEMENT_PADDING_CONNECTIVITY = 4
_GEN_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_EMPTY_RECT = frozenset()


def _gen_neighbors4(cell):
    r, c = cell
    for dr, dc in _GEN_DIRS:
        yield r + dr, c + dc


def _gen_between(value, bounds):
    return bounds[0] <= value <= bounds[1]


def _gen_bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _gen_adds_bad_2x2(cells, protected_rect, cand):
    trial = set(cells)
    trial.add(cand)
    r, c = cand
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            block = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
            if block <= trial and not block <= protected_rect:
                return True
    return False


def _gen_span_ok(cells, max_span):
    r0, c0, r1, c1 = _gen_bbox(cells)
    return (r1 - r0 + 1) <= max_span and (c1 - c0 + 1) <= max_span


def _gen_can_add_cell(cells, protected_rect, cand, *, bounds=None, max_span=None):
    if bounds is not None:
        max_r, max_c = bounds
        if not (0 <= cand[0] < max_r and 0 <= cand[1] < max_c):
            return False
    if cand in cells:
        return False
    if sum((nbr in cells) for nbr in _gen_neighbors4(cand)) != 1:
        return False
    if _gen_adds_bad_2x2(cells, protected_rect, cand):
        return False
    if max_span is not None and not _gen_span_ok(set(cells) | {cand}, max_span):
        return False
    return True


def _gen_rect_areas(count):
    combos = [
        combo
        for combo in itertools.product(_GEN_RECT_AREA_CHOICES, repeat=count)
        if _gen_between(sum(combo), _GEN_RED_COUNT_RANGE)
    ]
    return list(random.choice(combos))


def _gen_seed_candidates(rr, cc, h, w):
    candidates = []
    for r in range(rr, rr + h):
        candidates.append(((r, cc - 1), (0, -1)))
        candidates.append(((r, cc + w), (0, 1)))
    for c in range(cc, cc + w):
        candidates.append(((rr - 1, c), (-1, 0)))
        candidates.append(((rr + h, c), (1, 0)))
    return candidates


def _gen_direction_options(direction):
    dr, dc = direction
    dirs = [(dr, dc)]
    if dr == 0:
        dirs.extend([(-1, 0), (1, 0)])
    else:
        dirs.extend([(0, -1), (0, 1)])
    random.shuffle(dirs)
    return dirs


def _gen_initial_arm_endpoints(cells, rect_cells, seed_candidates, arm_count):
    """Seed one cell per arm around the rectangle; mutates cells."""
    random.shuffle(seed_candidates)
    endpoints = []
    bounds = (_GEN_QUAL_PATCH_SPAN, _GEN_QUAL_PATCH_SPAN)
    for cand, direction in seed_candidates:
        if len(endpoints) == arm_count:
            break
        if not _gen_can_add_cell(cells, rect_cells, cand, bounds=bounds):
            continue
        cells.add(cand)
        endpoints.append([cand, direction])
    return endpoints if len(endpoints) == arm_count else None


def _gen_grow_qualifying_arms(cells, rect_cells, endpoints, extra_steps):
    """Extend seeded arms inside the 5x5 patch; mutates cells and endpoints."""
    bounds = (_GEN_QUAL_PATCH_SPAN, _GEN_QUAL_PATCH_SPAN)
    for _ in range(extra_steps):
        random.shuffle(endpoints)
        placed = False
        for endpoint in endpoints:
            (r, c), direction = endpoint
            for ndr, ndc in _gen_direction_options(direction):
                cand = (r + ndr, c + ndc)
                if not _gen_can_add_cell(cells, rect_cells, cand, bounds=bounds):
                    continue
                cells.add(cand)
                endpoint[0] = cand
                endpoint[1] = (ndr, ndc)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _gen_render_patch(cells, red_cells=()):
    r0, c0, r1, c1 = _gen_bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), _GEN_GREY, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = _GEN_BLACK
    for r, c in red_cells:
        patch[r - r0, c - c0] = _GEN_RED
    return patch


def _gen_qualifying_patch(area):
    h, w = random.choice(_GEN_AREA_TO_DIMS[area])
    rr = random.randint(0, _GEN_QUAL_PATCH_SPAN - h)
    cc = random.randint(0, _GEN_QUAL_PATCH_SPAN - w)
    rect_cells = {(rr + r, cc + c) for r in range(h) for c in range(w)}
    cells = set(rect_cells)

    max_extra = min(_GEN_QUAL_COMPONENT_SIZE_RANGE[1] - area, _GEN_QUAL_EXTRA_CELL_CAP)
    extra_cells = random.randint(_GEN_QUAL_EXTRA_CELL_MIN, max_extra)
    arm_count = random.randint(_GEN_ARM_COUNT_RANGE[0], min(_GEN_ARM_COUNT_RANGE[1], extra_cells))

    seed_candidates = _gen_seed_candidates(rr, cc, h, w)
    endpoints = _gen_initial_arm_endpoints(cells, rect_cells, seed_candidates, arm_count)
    if endpoints is None:
        return None

    if not _gen_grow_qualifying_arms(cells, rect_cells, endpoints, extra_cells - arm_count):
        return None
    return _gen_render_patch(cells, rect_cells)


def _gen_path_patch(size):
    cells = {(0, 0)}
    while len(cells) < size:
        anchors = list(cells)
        random.shuffle(anchors)
        placed = False
        for anchor in anchors:
            dirs = list(_GEN_DIRS)
            random.shuffle(dirs)
            for dr, dc in dirs:
                cand = (anchor[0] + dr, anchor[1] + dc)
                if not _gen_can_add_cell(cells, _GEN_EMPTY_RECT, cand, max_span=_GEN_PATH_PATCH_SPAN):
                    continue
                cells.add(cand)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return _gen_render_patch(cells)


def _gen_noise_sizes(component_count, total_cells):
    if total_cells < component_count or total_cells > component_count * _GEN_NOISE_COMPONENT_SIZE_RANGE[1]:
        return None
    min_locked = int(component_count * _GEN_NOISE_SINGLETON_FRACTION_RANGE[0])
    max_locked = int(component_count * _GEN_NOISE_SINGLETON_FRACTION_RANGE[1])
    extra_capacity = _GEN_NOISE_COMPONENT_SIZE_RANGE[1] - 1
    needed_unlocked = max(1, (total_cells - component_count + extra_capacity - 1) // extra_capacity)
    max_locked = min(max_locked, component_count - needed_unlocked)
    if max_locked < min_locked:
        min_locked = max(0, max_locked)
    locked_singletons = random.randint(min_locked, max_locked) if max_locked >= 0 else 0

    order = list(range(component_count))
    random.shuffle(order)
    locked = set(order[:locked_singletons])
    caps = [1] * component_count
    for i in range(component_count):
        if i not in locked:
            caps[i] = random.choices(_GEN_NOISE_CAP_CHOICES, weights=_GEN_NOISE_CAP_WEIGHTS, k=1)[0]
    while sum(caps) < total_cells:
        growable = [i for i in range(component_count) if i not in locked and caps[i] < _GEN_NOISE_COMPONENT_SIZE_RANGE[1]]
        if not growable:
            return None
        i = random.choice(growable)
        caps[i] = min(
            _GEN_NOISE_COMPONENT_SIZE_RANGE[1],
            caps[i] + random.randint(*_GEN_NOISE_CAP_GROW_STEP_RANGE),
        )

    sizes = [1] * component_count
    remaining = total_cells - component_count
    while remaining > 0:
        avail = [i for i in range(component_count) if sizes[i] < caps[i]]
        if not avail:
            return None
        weights = [(caps[i] - sizes[i]) / (sizes[i] * sizes[i]) for i in avail]
        i = random.choices(avail, weights=weights, k=1)[0]
        sizes[i] += 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def _gen_has_2x2(mask):
    return bool((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).any())


def _gen_validate(inp, expected):
    if set(map(int, np.unique(inp))) != {_GEN_BLACK, _GEN_GREY}:
        return False
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    black_count = int((inp == _GEN_BLACK).sum())
    if not _gen_between(black_count, _GEN_BLACK_COUNT_RANGE):
        return False
    red_count = int((expected == _GEN_RED).sum())
    if not _gen_between(red_count, _GEN_RED_COUNT_RANGE):
        return False

    comps = _gen_components_mask(inp == _GEN_BLACK, connectivity=4)
    if not _gen_between(len(comps), _GEN_TOTAL_COMPONENT_RANGE):
        return False

    red_mask = expected == _GEN_RED
    q_count = 0
    nonq_count = 0
    for comp in comps:
        touches_red = bool(red_mask[comp[:, 0], comp[:, 1]].any())
        if touches_red:
            q_count += 1
            if not _gen_between(len(comp), _GEN_QUAL_COMPONENT_SIZE_RANGE):
                return False
            r0, c0, r1, c1 = bbox_coords(comp)
            if (r1 - r0 + 1) > _GEN_QUAL_PATCH_SPAN or (c1 - c0 + 1) > _GEN_QUAL_PATCH_SPAN:
                return False
        else:
            nonq_count += 1
            if not _gen_between(len(comp), _GEN_NOISE_COMPONENT_SIZE_RANGE):
                return False
            r0, c0, r1, c1 = bbox_coords(comp)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            if _gen_has_2x2(local):
                return False
    return _gen_between(q_count, _GEN_QUAL_COMPONENT_COUNT_RANGE) and _gen_between(
        nonq_count, _GEN_NOISE_COMPONENT_COUNT_RANGE
    )


def _gen_sample_qualifying_count():
    return random.choices(
        _GEN_QUAL_COMPONENT_COUNT_CHOICES,
        weights=_GEN_QUAL_COMPONENT_COUNT_WEIGHTS,
        k=1,
    )[0]


def _gen_place_patch(out, patch):
    try:
        r, c = _gen_random_free_location(
            out,
            patch,
            transparent=_GEN_GREY,
            padding=_GEN_PLACEMENT_PADDING,
            padding_connectivity=_GEN_PLACEMENT_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return False
    _gen_paste(out, patch, r, c, transparent=_GEN_GREY)
    return True


def _gen_place_qualifying_components(out, q_count):
    for area in _gen_rect_areas(q_count):
        patch = _gen_qualifying_patch(area)
        if patch is None:
            return False
        if not _gen_place_patch(out, patch):
            return False
    return True


def _gen_place_noise_components(out, q_count):
    q_cells = int((out != _GEN_GREY).sum())
    component_target = random.randint(*_GEN_TOTAL_COMPONENT_RANGE)
    noise_components = component_target - q_count
    if not _gen_between(noise_components, _GEN_NOISE_COMPONENT_COUNT_RANGE):
        return False

    min_noise = max(noise_components, _GEN_BLACK_COUNT_RANGE[0] - q_cells)
    max_noise = min(
        _GEN_BLACK_COUNT_RANGE[1] - q_cells,
        noise_components * _GEN_NOISE_TOTAL_DENSITY_CAP_PER_COMPONENT,
    )
    if min_noise > max_noise:
        return False
    noise_cells = random.randint(min_noise, max_noise)
    sizes = _gen_noise_sizes(noise_components, noise_cells)
    if sizes is None:
        return False

    for size in sorted(sizes, reverse=True):
        patch = _gen_path_patch(size)
        if patch is None:
            return False
        if not _gen_place_patch(out, patch):
            return False
    return True


def _gen_input_from_expected(expected):
    inp = expected.copy()
    inp[inp == _GEN_RED] = _GEN_BLACK
    return inp


def _gen_attempt() -> np.ndarray | None:
    out = np.full(_GEN_GRID_SHAPE, _GEN_GREY, dtype=int)
    q_count = _gen_sample_qualifying_count()
    if not _gen_place_qualifying_components(out, q_count):
        return None
    if not _gen_place_noise_components(out, q_count):
        return None
    inp = _gen_input_from_expected(out)
    if not _gen_validate(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRY_LIMIT} attempts")
