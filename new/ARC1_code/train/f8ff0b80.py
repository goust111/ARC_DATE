import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg, paste, random_free_location
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Return the three foreground colors as a vertical list ordered by decreasing pixel count.

    Input structure
    A dominant background contains exactly three foreground color masks.

    Rule
    Count the pixels of each foreground color, sort the colors from largest mask
    to smallest mask, and emit those colors as a 3x1 column.

    Key insight
    The input positions and mask shapes are distractors; the output is the size
    ranking of the color masks.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 3

    counts = [(int(np.count_nonzero(g == color)), color) for color in colors]
    assert len({count for count, _ in counts}) == len(counts)

    ranked_colors = [color for count, color in sorted(counts, key=lambda item: item[0], reverse=True)]
    return np.array(ranked_colors, dtype=int).reshape(3, 1)

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry limits.
GRID_SIZE = 12
BACKGROUND_COLOR = 0
COLOR_POOL = range(1, 10)
COLOR_COUNT = 3
GENERATION_RETRIES = 300

# Pixel count constraints.
COUNT_MIN, COUNT_MAX_EXCLUSIVE = 3, 15
TOTAL_COUNT_MIN, TOTAL_COUNT_MAX = 15, 28

# Blob geometry constraints.
COMPONENT_MIN, COMPONENT_MAX = 1, 3
BBOX_SIZE_MIN, BBOX_SIZE_MAX_EXCLUSIVE = 2, 6
FILL_RATIO_MIN, FILL_RATIO_MAX = 0.4, 0.85
SHAPE_OPTION_LIMIT = 12

# Placement constraints.
PLACEMENT_BORDER, PLACEMENT_PADDING = 1, 1


def _gen_neighbors4(r, c, h, w):
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _gen_count_triplet():
    triples = [
        (a, b, c)
        for a, b, c in itertools.combinations(range(COUNT_MIN, COUNT_MAX_EXCLUSIVE), COLOR_COUNT)
        if TOTAL_COUNT_MIN <= a + b + c <= TOTAL_COUNT_MAX
    ]
    counts = list(random.choice(triples))
    random.shuffle(counts)
    return counts


def _gen_seed_sets(h, w, k):
    cells = [(r, c) for r in range(h) for c in range(w)]
    out = []
    for combo in itertools.combinations(cells, k):
        rows = [p[0] for p in combo]
        cols = [p[1] for p in combo]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.combinations(combo, 2)):
            continue
        out.append(combo)
    return out


def _gen_shape_options(count):
    options = []
    for k in range(COMPONENT_MIN, min(COMPONENT_MAX, count) + 1):
        for h in range(BBOX_SIZE_MIN, BBOX_SIZE_MAX_EXCLUSIVE):
            for w in range(BBOX_SIZE_MIN, BBOX_SIZE_MAX_EXCLUSIVE):
                fill = count / (h * w)
                if not (FILL_RATIO_MIN <= fill <= FILL_RATIO_MAX):
                    continue
                if k == 1:
                    if count >= h + w - 1:
                        options.append((k, h, w))
                elif _gen_seed_sets(h, w, k):
                    options.append((k, h, w))
    return options


def _gen_split_sizes(total, k):
    sizes = [1] * k
    for _ in range(total - k):
        sizes[random.randrange(k)] += 1
    random.shuffle(sizes)
    return sizes


def _gen_connected_full_bbox_mask(h, w, count):
    corners = [
        ((0, 0), (h - 1, w - 1)),
        ((0, w - 1), (h - 1, 0)),
        ((h - 1, 0), (0, w - 1)),
        ((h - 1, w - 1), (0, 0)),
    ]
    (r, c), (er, ec) = random.choice(corners)
    steps = []
    steps.extend([(1 if er > r else -1, 0)] * abs(er - r))
    steps.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    random.shuffle(steps)

    mask = np.zeros((h, w), dtype=bool)
    mask[r, c] = True
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True

    while int(mask.sum()) < count:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for nr, nc in _gen_neighbors4(int(rr), int(cc), h, w):
                if not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _gen_multi_component_mask(h, w, count, k):
    seed_sets = _gen_seed_sets(h, w, k)
    if not seed_sets:
        return None

    labels = np.full((h, w), -1, dtype=int)
    for i, (r, c) in enumerate(random.choice(seed_sets)):
        labels[r, c] = i

    targets = _gen_split_sizes(count, k)
    sizes = [1] * k

    while sizes != targets:
        progressed = False
        order = [i for i in range(k) if sizes[i] < targets[i]]
        random.shuffle(order)
        for i in order:
            frontier = []
            for rr, cc in np.argwhere(labels == i):
                for nr, nc in _gen_neighbors4(int(rr), int(cc), h, w):
                    if labels[nr, nc] != -1:
                        continue
                    touching_other = any(labels[ar, ac] not in (-1, i) for ar, ac in _gen_neighbors4(nr, nc, h, w))
                    if not touching_other:
                        frontier.append((nr, nc))
            if frontier:
                nr, nc = random.choice(frontier)
                labels[nr, nc] = i
                sizes[i] += 1
                progressed = True
        if not progressed:
            return None

    return labels >= 0


def _gen_mask(count):
    options = _gen_shape_options(count)
    random.shuffle(options)
    for k, h, w in options[:SHAPE_OPTION_LIMIT]:
        if k == 1:
            mask = _gen_connected_full_bbox_mask(h, w, count)
        else:
            mask = _gen_multi_component_mask(h, w, count, k)
        if mask is None:
            continue
        if int(mask.sum()) != count:
            continue
        if bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if len(components_mask(mask, connectivity=4)) != k:
            continue
        return mask
    return None


def _gen_objects(colors, counts):
    objects = []
    for color, count in zip(colors, counts):
        mask = _gen_mask(count)
        if mask is None:
            return None
        objects.append((int(mask.size), int(color), int(count), mask))
    return objects


def _place_objects(grid, bbox_canvas, objects):
    for _area, color, _count, mask in sorted(objects, reverse=True):
        patch = np.ones(mask.shape, dtype=int)
        try:
            r, c = random_free_location(
                bbox_canvas, patch, transparent=BACKGROUND_COLOR,
                border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return False
        paste(bbox_canvas, patch, r, c, transparent=BACKGROUND_COLOR)
        paste(grid, np.where(mask, color, BACKGROUND_COLOR), r, c, transparent=BACKGROUND_COLOR)
    return True


def _gen_attempt():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    bbox_canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    colors = random.sample(COLOR_POOL, COLOR_COUNT)
    counts = _gen_count_triplet()

    objects = _gen_objects(colors, counts)
    if objects is None:
        return None

    if not _place_objects(grid, bbox_canvas, objects):
        return None

    expected = np.array(
        [color for count, color in sorted(zip(counts, colors), reverse=True)],
        dtype=int,
    ).reshape(COLOR_COUNT, 1)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for f8ff0b80")
