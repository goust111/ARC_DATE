import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each foreground color mask vertically inside its own bounding box.

    Input structure
    - A dominant background contains several monochrome foreground masks.
    - Each foreground color is a glyph-like mask; it may be split under
      4-connectivity but is treated as one mask by color.

    Rule
    - For each non-background color independently, find the mask's bounding box.
    - Keep every pixel's column and replace its row r by r0 + r1 - r, where
      r0 and r1 are the top and bottom rows of that color mask's bounding box.

    Key insight
    - The scene is not globally flipped; each colored mask is mirrored within its
      own vertical extent, so one-row and vertically symmetric masks stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    out = np.full_like(g, bg)

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        r0, _c0, r1, _c1 = bbox_coords(coords)

        rr = r0 + r1 - coords[:, 0]
        cc = coords[:, 1]
        assert np.all(out[rr, cc] == bg)
        out[rr, cc] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette choices.
GRID_SIZE_CHOICES = range(7, 16, 2)
COLOR_COUNT_CHOICES = range(3, 6)
COLOR_POOL = range(1, 10)

# Vertical band and mask dimensions.
BORDER_SIZE = 1
BAND_HEIGHT_RANGE = range(1, 5)
MASK_WIDTH_MIN = 3
MASK_WIDTH_MAX = 11
MASK_WIDTH_STEP = 2

# Mask density, symmetry, and connectivity constraints.
MIN_MASK_DENSITY = 0.39
MAX_MASK_DENSITY = 1.0
COMPONENT_COUNT_CHOICES = range(1, 6)
COMPONENT_CONNECTIVITY = 4
NONVERTICAL_MIN_HEIGHT = 2

# Retry limits.
MASK_ATTEMPTS = 20
GENERATE_ATTEMPTS = 300


def _gen_orbit_cells(r, c, w):
    cc = w - 1 - c
    if cc == c:
        return [(r, c)]
    return [(r, c), (r, cc)]


def _gen_orbit_blocked(cells, blocked):
    return any((r, c) in blocked for r, c in cells)


def _gen_has_neighbor(mask, cells):
    h, w = mask.shape
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                return True
    return False


def _gen_run_count(row):
    count = 0
    prev = False
    for val in row:
        if val and not prev:
            count += 1
        prev = bool(val)
    return count


def _gen_touches_bbox(mask):
    return mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()


def _gen_is_vertically_symmetric(mask):
    return np.array_equal(mask, np.flipud(mask))


def _gen_row_patterns(w, component_count):
    mid = w // 2
    patterns = []
    for bits in range(1 << (mid + 1)):
        left = [bool(bits & (1 << c)) for c in range(mid + 1)]
        row = np.array(left + left[:-1][::-1], dtype=bool)
        if not (row[0] and row[-1]):
            continue
        if _gen_run_count(row) != component_count:
            continue
        if row.mean() < MIN_MASK_DENSITY:
            continue
        patterns.append(row)
    return patterns


def _gen_valid_component_mask(mask, component_count):
    if not _gen_touches_bbox(mask):
        return False
    if mask.mean() < MIN_MASK_DENSITY:
        return False
    return len(components_mask(mask, connectivity=COMPONENT_CONNECTIVITY)) == component_count


def _gen_roughen_stripes(mask, component_count, want_nonvertical):
    h, w = mask.shape
    mid = w // 2
    steps = random.randint(0, max(1, h))
    if want_nonvertical:
        steps = max(1, steps)

    for _ in range(steps):
        candidates = []
        rows = list(range(h))
        random.shuffle(rows)
        cols = list(range(mid + 1))
        random.shuffle(cols)
        for r in rows:
            for c in cols:
                cells = _gen_orbit_cells(r, c, w)
                if not all(mask[x, y] for x, y in cells):
                    continue
                candidate = mask.copy()
                for x, y in cells:
                    candidate[x, y] = False
                if not _gen_valid_component_mask(candidate, component_count):
                    continue
                if want_nonvertical and _gen_is_vertically_symmetric(mask):
                    if _gen_is_vertically_symmetric(candidate):
                        continue
                candidates.append(cells)
        if not candidates:
            break
        for r, c in random.choice(candidates):
            mask[r, c] = False

    if want_nonvertical and _gen_is_vertically_symmetric(mask):
        return None
    return mask


def _gen_stripe_mask(h, w, component_count, want_nonvertical):
    patterns = _gen_row_patterns(w, component_count)
    random.shuffle(patterns)

    for row in patterns:
        mask = np.tile(row, (h, 1))
        mask = _gen_roughen_stripes(mask, component_count, want_nonvertical)
        if mask is not None and _gen_valid_component_mask(mask, component_count):
            return mask

    return None


def _gen_spine_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mid = w // 2
    mask[:, mid] = True
    if h == 1:
        mask[0, :] = True
    else:
        row = random.choice([0, h - 1])
        mask[row, :] = True
    return mask


def _gen_preserve_nonvertical(mask):
    h, w = mask.shape
    blocked = set()
    if h < NONVERTICAL_MIN_HEIGHT:
        return blocked

    mid = w // 2
    for r in range((h + 1) // 2):
        mr = h - 1 - r
        if r == mr:
            continue
        for c in range(mid + 1):
            cells = _gen_orbit_cells(r, c, w)
            mirror = _gen_orbit_cells(mr, c, w)
            if all(mask[x, y] for x, y in cells) and not any(mask[x, y] for x, y in mirror):
                blocked.update(mirror)
                return blocked

    for r in range((h + 1) // 2):
        mr = h - 1 - r
        if r == mr:
            continue
        cols = list(range(mid + 1))
        random.shuffle(cols)
        for c in cols:
            cells = _gen_orbit_cells(r, c, w)
            mirror = _gen_orbit_cells(mr, c, w)
            if not any(mask[x, y] for x, y in cells + mirror):
                for x, y in cells:
                    mask[x, y] = True
                blocked.update(mirror)
                return blocked

    return None


def _gen_grow_mask(mask, blocked):
    h, w = mask.shape
    total = h * w
    min_cells = int(np.ceil(MIN_MASK_DENSITY * total))
    max_cells = total - len(blocked)
    if max_cells < min_cells:
        return None

    start = int(mask.sum())
    target = random.randint(max(min_cells, start), max_cells)
    mid = w // 2

    while int(mask.sum()) < target:
        remaining = target - int(mask.sum())
        candidates = []
        overshoot = []
        for r in range(h):
            for c in range(mid + 1):
                cells = _gen_orbit_cells(r, c, w)
                if all(mask[x, y] for x, y in cells) or _gen_orbit_blocked(cells, blocked):
                    continue
                if not _gen_has_neighbor(mask, cells):
                    continue
                if len(cells) <= remaining:
                    candidates.append(cells)
                else:
                    overshoot.append(cells)

        if candidates:
            cells = random.choice(candidates)
        elif int(mask.sum()) < min_cells and overshoot:
            cells = random.choice(overshoot)
        else:
            break

        for r, c in cells:
            mask[r, c] = True

    if int(mask.sum()) < min_cells:
        return None
    return mask


def _gen_make_mask(h, w, want_nonvertical, target_components):
    if want_nonvertical and h < NONVERTICAL_MIN_HEIGHT:
        return None

    for _ in range(MASK_ATTEMPTS):
        if target_components > 1:
            mask = _gen_stripe_mask(h, w, target_components, want_nonvertical)
            if mask is None:
                continue
        else:
            mask = _gen_spine_mask(h, w)
            blocked = set()
            if want_nonvertical:
                blocked = _gen_preserve_nonvertical(mask)
                if blocked is None:
                    continue

            mask = _gen_grow_mask(mask, blocked)
            if mask is None:
                continue

        if want_nonvertical and _gen_is_vertically_symmetric(mask):
            continue
        if not np.array_equal(mask, np.fliplr(mask)):
            continue
        if not _gen_touches_bbox(mask):
            continue
        if not (MIN_MASK_DENSITY <= mask.mean() <= MAX_MASK_DENSITY):
            continue
        if len(components_mask(mask, connectivity=COMPONENT_CONNECTIVITY)) != target_components:
            continue
        return mask

    return None


def _gen_height_partitions(inner, k):
    parts = [[]]
    for _ in range(k):
        parts = [p + [h] for p in parts for h in BAND_HEIGHT_RANGE]
    needed = (k + 1) // 2
    return [
        p
        for p in parts
        if sum(p) == inner and sum(h >= NONVERTICAL_MIN_HEIGHT for h in p) >= needed
    ]


def _gen_sample_layout():
    valid_pairs = []
    partitions_by_pair = {}
    for n in GRID_SIZE_CHOICES:
        inner_height = n - 2 * BORDER_SIZE
        for k in COLOR_COUNT_CHOICES:
            parts = _gen_height_partitions(inner_height, k)
            if parts:
                valid_pairs.append((n, k))
                partitions_by_pair[(n, k)] = parts

    n, k = random.choice(valid_pairs)
    heights = list(random.choice(partitions_by_pair[(n, k)]))
    random.shuffle(heights)
    return n, k, heights


def _gen_nonvertical_indices(heights):
    eligible = [i for i, h in enumerate(heights) if h >= NONVERTICAL_MIN_HEIGHT]
    needed = (len(heights) + 1) // 2
    nonvertical_count = random.randint(needed, len(eligible))
    return set(random.sample(eligible, nonvertical_count))


def _gen_feasible_component_widths(widths):
    feasible = []
    for target_components in COMPONENT_COUNT_CHOICES:
        valid_widths = [
            width
            for width in widths
            if target_components == 1 or _gen_row_patterns(width, target_components)
        ]
        if valid_widths:
            feasible.append((target_components, valid_widths))
    return feasible


def _gen_render_bands(n, heights, colors, nonvertical):
    inp = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    max_width = min(MASK_WIDTH_MAX, n - 2 * BORDER_SIZE)
    widths = list(range(MASK_WIDTH_MIN, max_width + 1, MASK_WIDTH_STEP))
    feasible = _gen_feasible_component_widths(widths)

    r0 = BORDER_SIZE
    for i, (height, color) in enumerate(zip(heights, colors)):
        target_components, valid_widths = random.choice(feasible)
        width = random.choice(valid_widths)
        mask = _gen_make_mask(height, width, i in nonvertical, target_components)
        if mask is None:
            return None

        c0 = (n - width) // 2
        patch = mask.astype(int) * color
        inp[r0 : r0 + height, c0 : c0 + width] = patch
        expected[r0 : r0 + height, c0 : c0 + width] = np.flipud(patch)
        r0 += height

    if r0 != n - BORDER_SIZE:
        return None
    return inp, expected


def _gen_attempt():
    n, k, heights = _gen_sample_layout()
    colors = random.sample(COLOR_POOL, k)
    nonvertical = _gen_nonvertical_indices(heights)

    rendered = _gen_render_bands(n, heights, colors, nonvertical)
    if rendered is None:
        return None
    inp, expected = rendered

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
