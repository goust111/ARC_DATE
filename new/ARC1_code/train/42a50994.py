import numpy as np

from arc_utils import colors_present, components_mask, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete every isolated foreground pixel while preserving all foreground clusters.

    Input structure:
    - The grid has one dominant background color.
    - All foreground pixels share a single color.
    - Foreground pixels appear as isolated pixels and as 8-connected clusters.

    Rule:
    - Label the foreground pixels using 8-connectivity, so diagonal contact joins a cluster.
    - Replace each size-1 foreground component with the background color.
    - Keep every foreground component whose size is greater than one.

    Key insight:
    - The output is a noise removal: singleton foreground pixels disappear, but any pixel
      touching another foreground pixel in the 8-neighborhood remains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    fg = foreground_colors[0]

    out = np.full(g.shape, bg, dtype=int)
    components = components_mask(g != bg, connectivity=8)
    assert components, "expected at least one foreground component"
    for coords in components:
        if len(coords) > 1:
            out[coords[:, 0], coords[:, 1]] = fg

    return out

# --------------- generator (o5x) ---------------

# Canvas, color, and foreground budget.
_GEN_HEIGHT_RANGE, _GEN_WIDTH_RANGE, _GEN_COLOR_RANGE = (11, 17), (9, 19), (1, 9)
_GEN_BACKGROUND = 0
_GEN_FG_FRACTION_RANGE = (0.13, 0.24)

# Component counts and cluster sizes.
_GEN_SINGLETON_COUNT_RANGE, _GEN_CLUSTER_COUNT_RANGE = (8, 13), (3, 13)
_GEN_COMPONENT_COUNT_RANGE, _GEN_CLUSTER_SIZE_RANGE = (14, 25), (2, 12)
_GEN_COMMON_CLUSTER_MAX_SIZE = 6
_GEN_LARGE_CLUSTER_MIN_SIZE, _GEN_LARGE_CLUSTER_PROB = 7, 0.35

# Blob geometry and foreground allocation weights.
_GEN_BLOB_MIN_DENSITY, _GEN_BLOB_MAX_DIM = 0.30, 6
_GEN_BLOB_SPARSITY_WEIGHT = 0.6
_GEN_TOTAL_FG_BAND_WEIGHTS = (0.05, 0.25, 0.55, 0.15)

# Retry budget.
_GEN_MAX_ATTEMPTS = 300

_GEN_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _gen_mark_blocked(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        blocked[r0:r1, c0:c1] = True


def _gen_blob_coords(size):
    max_area = int(size / _GEN_BLOB_MIN_DENSITY)
    max_dim = min(_GEN_BLOB_MAX_DIM, size)
    dims = [
        (h, w)
        for h in range(1, max_dim + 1)
        for w in range(1, max_dim + 1)
        if h * w >= size and h * w <= max_area and h + w > 2
    ]
    weights = [1 + _GEN_BLOB_SPARSITY_WEIGHT * (h * w - size) for h, w in dims]
    h, w = random.choices(dims, weights=weights, k=1)[0]

    coords = {(random.randrange(h), random.randrange(w))}
    while len(coords) < size:
        frontier = []
        for r, c in coords:
            for dr, dc in _GEN_DIRS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))

    arr = np.array(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _gen_pick_total_foreground(n_singletons, n_clusters, lo, hi):
    bands = []
    low_hi = min(hi, n_singletons + 4 * n_clusters - 1)
    if lo <= low_hi:
        bands.append((lo, low_hi, _GEN_TOTAL_FG_BAND_WEIGHTS[0]))
    mid_low_lo = max(lo, n_singletons + 4 * n_clusters)
    mid_low_hi = min(hi, n_singletons + 5 * n_clusters - 1)
    if mid_low_lo <= mid_low_hi:
        bands.append((mid_low_lo, mid_low_hi, _GEN_TOTAL_FG_BAND_WEIGHTS[1]))
    mid_lo = max(lo, n_singletons + 5 * n_clusters)
    mid_hi = min(hi, n_singletons + 6 * n_clusters)
    if mid_lo <= mid_hi:
        bands.append((mid_lo, mid_hi, _GEN_TOTAL_FG_BAND_WEIGHTS[2]))
    high_lo = max(lo, n_singletons + 6 * n_clusters + 1)
    if high_lo <= hi:
        bands.append((high_lo, hi, _GEN_TOTAL_FG_BAND_WEIGHTS[3]))
    band = random.choices(bands, weights=[band[2] for band in bands], k=1)[0]
    return random.randint(band[0], band[1])


def _gen_choose_counts(h, w):
    area = h * w
    min_frac, max_frac = _GEN_FG_FRACTION_RANGE
    min_fg = int(np.ceil(min_frac * area))
    max_fg = int(np.floor(max_frac * area))
    min_components, max_components = _GEN_COMPONENT_COUNT_RANGE
    min_cluster_size, max_cluster_size = _GEN_CLUSTER_SIZE_RANGE
    choices = []
    for n_singletons in range(_GEN_SINGLETON_COUNT_RANGE[0], _GEN_SINGLETON_COUNT_RANGE[1] + 1):
        for n_clusters in range(_GEN_CLUSTER_COUNT_RANGE[0], _GEN_CLUSTER_COUNT_RANGE[1] + 1):
            if not min_components <= n_singletons + n_clusters <= max_components:
                continue
            lo = max(min_fg, n_singletons + min_cluster_size * n_clusters)
            hi = min(max_fg, n_singletons + max_cluster_size * n_clusters)
            if lo <= hi:
                choices.append((n_singletons, n_clusters, lo, hi))
    if not choices:
        return None

    singleton_options = sorted({choice[0] for choice in choices})
    n_singletons = random.choice(singleton_options)
    cluster_choices = [choice for choice in choices if choice[0] == n_singletons]
    cluster_options = sorted({choice[1] for choice in cluster_choices})
    n_clusters = random.choice(cluster_options)
    ranges = [choice for choice in cluster_choices if choice[1] == n_clusters]
    lo = min(choice[2] for choice in ranges)
    hi = max(choice[3] for choice in ranges)

    total_fg = _gen_pick_total_foreground(n_singletons, n_clusters, lo, hi)
    return n_singletons, n_clusters, total_fg


def _gen_cluster_sizes(n_clusters, cluster_sum):
    min_size, max_size = _GEN_CLUSTER_SIZE_RANGE
    sizes = [min_size] * n_clusters
    extra = cluster_sum - min_size * n_clusters

    if extra >= _GEN_LARGE_CLUSTER_MIN_SIZE - min_size and random.random() < _GEN_LARGE_CLUSTER_PROB:
        i = random.randrange(n_clusters)
        target = random.randint(_GEN_LARGE_CLUSTER_MIN_SIZE, min(max_size, sizes[i] + extra))
        extra -= target - sizes[i]
        sizes[i] = target

    while extra:
        under_six = [i for i, size in enumerate(sizes) if size < _GEN_COMMON_CLUSTER_MAX_SIZE]
        if under_six:
            low = min(sizes[i] for i in under_six)
            options = [i for i in under_six if sizes[i] == low]
        else:
            with_room = [i for i, size in enumerate(sizes) if size < max_size]
            low = min(sizes[i] for i in with_room)
            options = [i for i in with_room if sizes[i] == low]
        i = random.choice(options)
        sizes[i] += 1
        extra -= 1
    random.shuffle(sizes)
    return sizes


def _gen_place_coords(grid, blocked, coords, color):
    h, w = grid.shape
    patch_h = int(coords[:, 0].max()) + 1
    patch_w = int(coords[:, 1].max()) + 1
    anchors = []
    for r0 in range(h - patch_h + 1):
        rr = coords[:, 0] + r0
        for c0 in range(w - patch_w + 1):
            cc = coords[:, 1] + c0
            if not blocked[rr, cc].any():
                anchors.append((r0, c0))
    if not anchors:
        return None

    r0, c0 = random.choice(anchors)
    placed = coords + np.array([r0, c0], dtype=int)
    grid[placed[:, 0], placed[:, 1]] = color
    _gen_mark_blocked(blocked, placed)
    return placed


def _gen_place_singletons(grid, blocked, n_singletons, color):
    singleton_coords = []
    for _ in range(n_singletons):
        free = np.argwhere(~blocked)
        if len(free) == 0:
            return None
        r, c = free[random.randrange(len(free))]
        grid[r, c] = color
        singleton_coords.append((int(r), int(c)))
        _gen_mark_blocked(blocked, [(int(r), int(c))])
    return singleton_coords


def _gen_attempt():
    h = random.randint(*_GEN_HEIGHT_RANGE)
    w = random.randint(*_GEN_WIDTH_RANGE)
    color = random.randint(*_GEN_COLOR_RANGE)
    counts = _gen_choose_counts(h, w)
    if counts is None:
        return None
    n_singletons, n_clusters, total_fg = counts

    cluster_sum = total_fg - n_singletons
    sizes = _gen_cluster_sizes(n_clusters, cluster_sum)
    blobs = [_gen_blob_coords(size) for size in sizes]
    blobs.sort(key=lambda coords: (coords[:, 0].max() + 1) * (coords[:, 1].max() + 1), reverse=True)

    grid = np.full((h, w), _GEN_BACKGROUND, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for coords in blobs:
        if _gen_place_coords(grid, blocked, coords, color) is None:
            return None

    singleton_coords = _gen_place_singletons(grid, blocked, n_singletons, color)
    if singleton_coords is None:
        return None

    output = grid.copy()
    rows, cols = np.array(singleton_coords, dtype=int).T
    output[rows, cols] = _GEN_BACKGROUND

    min_frac, max_frac = _GEN_FG_FRACTION_RANGE
    if not (min_frac <= np.count_nonzero(grid) / grid.size <= max_frac):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 42a50994 input")
