import numpy as np

from arc_utils import Color, colors_present
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Paint every background cell on the same row/column parity lattice as the single colored marker.

    Input structure
    - A 2D grid containing black background pixels, grey fixed pixels, and exactly one non-grey/non-black colored marker pixel.

    Rule
    - The marker's row parity and column parity define the selected lattice.
    - Recolor each black background pixel with the marker color iff both its row parity and column parity match the marker.
    - Leave grey pixels and all background pixels on the other three parity lattices unchanged.

    Key insight
    - The marker is copied by coordinate parity, not by connectivity through the grey layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = set(colors_present(g))
    assert Color.BLACK in colors
    assert Color.GREY in colors
    marker_colors = colors - {Color.BLACK, Color.GREY}
    assert len(marker_colors) == 1
    marker_color = marker_colors.pop()

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape[0] == 1
    marker_r, marker_c = marker_coords[0]

    rr, cc = np.indices(g.shape)
    same_parity = (rr % 2 == marker_r % 2) & (cc % 2 == marker_c % 2)

    out = g.copy()
    out[(g == Color.BLACK) & same_parity] = marker_color
    return out

# --------------- generator (o5x) ---------------

from arc_utils import bbox_coords, components_mask, find_components

# Canvas and protected open core.
GRID_HEIGHT_CHOICES, GRID_WIDTH_CHOICES = (13, 13, 14, 15, 15), (13, 13, 14, 15, 15, 16, 17, 17)
CORE_HEIGHT_SMALL_CHOICES, CORE_HEIGHT_LARGE_CHOICES = (5, 5, 5), (5, 5, 7)
CORE_WIDTH_CHOICES, CORE_WIDE_WIDTH_THRESHOLD, CORE_WIDE_EXTRA_WIDTH = (5, 7), 15, 9
CORE_CENTER_JITTER_CHOICES = (-1, 0, 0, 1)
CORE_EDGE_MARGIN = 3

# Grey obstacle amount and blob growth.
GREY_DENSITY_SAMPLE_RANGE = (0.33, 0.45)
GREY_DENSITY_LIMITS = (0.30, 0.45)
BLOB_COUNT_RANGE = (2, 5)
SEED_EDGE_BAND, SEED_BASE_WEIGHT, SEED_CORNER_EXTRA_WEIGHT = 1, 2, 1
SEED_MIN_SEPARATIONS = (5, 4, 3, 2)
GROWTH_BAND_WIDTH_CHOICES = (3, 3, 4, 4, 5)
FRONTIER_BASE_WEIGHT, FRONTIER_SAME_NEIGHBOR_WEIGHT, FRONTIER_EDGE_WEIGHT = 1.0, 0.8, 0.5

# Acceptance checks and marker placement.
MIN_OPEN_COMPONENT_RATIO, OPEN_COMPONENT_FILL_RANGE = 0.50, (0.55, 0.67)
MARKER_MIN_GREY_DISTANCE, GEN_RETRY_LIMIT, MARKER_COLORS = 3, 300, (1, 2, 3, 4, 6, 7, 8, 9)

BACKGROUND_COLOR, OBSTACLE_COLOR = Color.BLACK, Color.GREY

def _gen_edge_dist(r, c, h, w):
    return min(r, c, h - 1 - r, w - 1 - c)


def _gen_pick_size():
    h = random.choice(GRID_HEIGHT_CHOICES)
    w = random.choice(GRID_WIDTH_CHOICES)
    return h, w


def _gen_core_bbox(h, w):
    core_h_choices = CORE_HEIGHT_LARGE_CHOICES if h >= 15 else CORE_HEIGHT_SMALL_CHOICES
    core_h = random.choice(core_h_choices)
    core_w_choices = list(CORE_WIDTH_CHOICES)
    if w >= CORE_WIDE_WIDTH_THRESHOLD:
        core_w_choices.append(CORE_WIDE_EXTRA_WIDTH)
    core_w = random.choice(core_w_choices)
    r_mid = h // 2 + random.choice(CORE_CENTER_JITTER_CHOICES)
    c_mid = w // 2 + random.choice(CORE_CENTER_JITTER_CHOICES)
    r0 = max(CORE_EDGE_MARGIN, min(h - core_h - CORE_EDGE_MARGIN, r_mid - core_h // 2))
    c0 = max(CORE_EDGE_MARGIN, min(w - core_w - CORE_EDGE_MARGIN, c_mid - core_w // 2))
    return r0, c0, r0 + core_h - 1, c0 + core_w - 1


def _gen_in_bbox(r, c, box):
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _gen_pick_seeds(h, w, count, core_box):
    cells = []
    for r in range(h):
        for c in range(w):
            if _gen_edge_dist(r, c, h, w) <= SEED_EDGE_BAND and not _gen_in_bbox(r, c, core_box):
                is_corner = (r in (0, h - 1)) and (c in (0, w - 1))
                weight = SEED_BASE_WEIGHT + int(is_corner) * SEED_CORNER_EXTRA_WEIGHT
                cells.extend([(r, c)] * weight)
    random.shuffle(cells)
    for min_sep in SEED_MIN_SEPARATIONS:
        seeds = []
        used = set()
        for r, c in cells:
            if (r, c) in used:
                continue
            used.add((r, c))
            if all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _gen_can_add(labels, blob_id, r, c):
    if labels[r, c] != 0:
        return False
    h, w = labels.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            other = labels[rr, cc]
            if other != 0 and other != blob_id:
                return False
    return True


def _gen_frontier(labels, blob_id, core_box, band_width):
    h, w = labels.shape
    coords = np.argwhere(labels == blob_id)
    seen = set()
    frontier = []
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(r + dr), int(c + dc)
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in seen or _gen_in_bbox(rr, cc, core_box):
                    continue
                seen.add((rr, cc))
                if _gen_edge_dist(rr, cc, h, w) > band_width:
                    continue
                if _gen_can_add(labels, blob_id, rr, cc):
                    same_neighbors = 0
                    for nr in range(max(0, rr - 1), min(h, rr + 2)):
                        for nc in range(max(0, cc - 1), min(w, cc + 2)):
                            same_neighbors += int(labels[nr, nc] == blob_id)
                    edge_weight = band_width + 1 - _gen_edge_dist(rr, cc, h, w)
                    weight = (
                        FRONTIER_BASE_WEIGHT
                        + FRONTIER_SAME_NEIGHBOR_WEIGHT * same_neighbors
                        + FRONTIER_EDGE_WEIGHT * edge_weight
                    )
                    frontier.append((blob_id, rr, cc, weight))
    return frontier


def _gen_grey_mask(h, w, target_total, blob_count, core_box):
    labels = np.zeros((h, w), dtype=int)
    seeds = _gen_pick_seeds(h, w, blob_count, core_box)
    if seeds is None:
        return None
    for i, (r, c) in enumerate(seeds, start=1):
        labels[r, c] = i

    band_width = random.choice(GROWTH_BAND_WIDTH_CHOICES)
    while int(np.count_nonzero(labels)) < target_total:
        options = []
        for blob_id in range(1, blob_count + 1):
            options.extend(_gen_frontier(labels, blob_id, core_box, band_width))
        if not options:
            return None
        weights = [item[3] for item in options]
        blob_id, r, c, _ = random.choices(options, weights=weights, k=1)[0]
        labels[r, c] = blob_id

    return labels > 0


def _gen_largest_component(mask):
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return None, None
    coords = max(comps, key=len)
    comp_mask = np.zeros(mask.shape, dtype=bool)
    comp_mask[coords[:, 0], coords[:, 1]] = True
    return coords, comp_mask


def _gen_component_fill(coords):
    r0, c0, r1, c1 = bbox_coords(coords)
    return len(coords) / ((r1 - r0 + 1) * (c1 - c0 + 1))


def _gen_marker_candidates(grid, comp_mask):
    grey = np.argwhere(grid == OBSTACLE_COLOR)
    h, w = grid.shape
    candidates = []
    for r, c in np.argwhere(comp_mask):
        r = int(r)
        c = int(c)
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        if grid[r, c] != BACKGROUND_COLOR:
            continue
        if not (comp_mask[r - 1, c] and comp_mask[r + 1, c] and comp_mask[r, c - 1] and comp_mask[r, c + 1]):
            continue
        if grey.size:
            d = np.abs(grey[:, 0] - r) + np.abs(grey[:, 1] - c)
            if int(d.min()) < MARKER_MIN_GREY_DISTANCE:
                continue
        candidates.append((r, c))
    return candidates


def _gen_expected_output(grid):
    marker_color = next(int(x) for x in np.unique(grid) if x not in (BACKGROUND_COLOR, OBSTACLE_COLOR))
    marker_r, marker_c = np.argwhere(grid == marker_color)[0]
    rr, cc = np.indices(grid.shape)
    out = grid.copy()
    out[(grid == BACKGROUND_COLOR) & (rr % 2 == marker_r % 2) & (cc % 2 == marker_c % 2)] = marker_color
    return out


def _gen_target_grey_count(total):
    density = random.uniform(*GREY_DENSITY_SAMPLE_RANGE)
    min_grey = max(1, round(GREY_DENSITY_LIMITS[0] * total))
    max_grey = round(GREY_DENSITY_LIMITS[1] * total)
    target_grey = random.randint(min_grey, max_grey)
    target_grey = max(target_grey, round(density * total))
    return min(target_grey, max_grey)


def _gen_grey_layout_valid(grey_mask, total):
    grey_count = int(np.count_nonzero(grey_mask))
    grey_ratio = grey_count / total
    if not (GREY_DENSITY_LIMITS[0] <= grey_ratio <= GREY_DENSITY_LIMITS[1]):
        return False
    if np.any(np.all(grey_mask, axis=0)) or np.any(np.all(grey_mask, axis=1)):
        return False

    grey_components = find_components(grey_mask.astype(int), bg=BACKGROUND_COLOR, connectivity=8, monochromatic=True, return_format="dict")
    return BLOB_COUNT_RANGE[0] <= len(grey_components) <= BLOB_COUNT_RANGE[1]


def _gen_valid_open_component(grid, total):
    open_coords, open_mask = _gen_largest_component(grid != OBSTACLE_COLOR)
    if open_coords is None or len(open_coords) < MIN_OPEN_COMPONENT_RATIO * total:
        return None
    fill = _gen_component_fill(open_coords)
    if not (OPEN_COMPONENT_FILL_RANGE[0] <= fill <= OPEN_COMPONENT_FILL_RANGE[1]):
        return None
    return open_mask


def _gen_place_marker(grid, open_mask):
    marker_options = _gen_marker_candidates(grid, open_mask)
    if not marker_options:
        return None
    r, c = random.choice(marker_options)
    grid[r, c] = random.choice(MARKER_COLORS)
    return grid


def _gen_attempt():
    h, w = _gen_pick_size()
    total = h * w
    target_grey = _gen_target_grey_count(total)
    blob_count = random.randint(*BLOB_COUNT_RANGE)
    core_box = _gen_core_bbox(h, w)

    grey_mask = _gen_grey_mask(h, w, target_grey, blob_count, core_box)
    if grey_mask is None or not _gen_grey_layout_valid(grey_mask, total):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[grey_mask] = OBSTACLE_COLOR
    open_mask = _gen_valid_open_component(grid, total)
    if open_mask is None:
        return None
    return _gen_place_marker(grid, open_mask)


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid d492a647 input")
