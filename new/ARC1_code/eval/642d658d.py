from collections import Counter

import numpy as np

from arc_utils import get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Return the color that appears as the arm color of the repeated yellow-centered plus marker.

    Input structure
    A noisy grid contains several plus-shaped markers: a yellow center pixel
    with four orthogonally adjacent pixels of one shared non-background color.

    Rule
    Count the arm color of each yellow-centered plus marker and output the
    unique color that occurs on more than one marker as a 1x1 grid.

    Key insight
    The surrounding noise is irrelevant; the duplicated marked plus motif
    identifies the answer color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    yellow = 4

    arm_counts: Counter[int] = Counter()
    for r, c in np.argwhere(g == yellow):
        r, c = int(r), int(c)
        assert 0 < r < h - 1 and 0 < c < w - 1
        arms = [
            int(g[r - 1, c]),
            int(g[r + 1, c]),
            int(g[r, c - 1]),
            int(g[r, c + 1]),
        ]
        assert len(set(arms)) == 1
        arm_color = arms[0]
        assert arm_color not in (0, yellow, bg)
        arm_counts[arm_color] += 1

    repeated = [color for color, count in arm_counts.items() if count > 1]
    assert len(repeated) == 1
    return np.array([[repeated[0]]], dtype=int)

# --------------- generator (o5x) ---------------

GEN_HEIGHT_RANGE = (20, 28)  # canvas/color bounds; randint upper bounds are exclusive.
GEN_WIDTH_RANGE = (22, 25)
GEN_BACKGROUND_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_BLACK, GEN_YELLOW, GEN_TRANSPARENT = 0, 4, -1

MARKER_COUNT_RANGE = (3, 7)  # marker count, spacing, and plus geometry.
MARKER_MIN_CHEB_DISTANCE = 5
MARKER_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_CELLS = 5

NOISE_GRID_FRAC_RANGE = (0.24, 0.34)  # total noise and non-background density bounds.
NON_BG_GRID_FRAC_RANGE = (0.27, 0.40)
CLUSTERED_NOISE_FRAC_RANGE = (0.70, 0.80)
MIN_CLUSTERED_NOISE_CELLS = 2
BLACK_NOISE_MIN_FRAC = 0.70

NOISE_COMPONENT_SIZE_RANGE = (3, 12)  # clustered-noise growth tuning.
NOISE_COMPONENT_TAIL_BUFFER = 2
NOISE_SEED_TRY_LIMIT = 80
NOISE_FRONTIER_RANDOM_WEIGHT = 1.5
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_MAX_ATTEMPTS = 300


def _gen_marker_patch(color):
    patch = np.full((3, 3), GEN_TRANSPARENT, dtype=int)
    patch[1, 1] = GEN_YELLOW
    patch[(0, 2, 1, 1), (1, 1, 0, 2)] = color
    return patch


def _gen_place_centers(h, w, n):
    positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    centers = []
    for _ in range(n):
        feasible = [
            (r, c)
            for r, c in positions
            if all(max(abs(r - rr), abs(c - cc)) >= MARKER_MIN_CHEB_DISTANCE for rr, cc in centers)
        ]
        if not feasible:
            return None
        centers.append(feasible[np.random.randint(len(feasible))])
    return centers


def _gen_noise_count(total, marker_cells):
    low = max(
        int(np.ceil(NOISE_GRID_FRAC_RANGE[0] * total)),
        int(np.ceil(NON_BG_GRID_FRAC_RANGE[0] * total - marker_cells)),
    )
    high = min(
        int(np.floor(NOISE_GRID_FRAC_RANGE[1] * total)),
        int(np.floor(NON_BG_GRID_FRAC_RANGE[1] * total - marker_cells)),
    )
    if low > high:
        return None
    return int(np.random.randint(low, high + 1))


def _gen_component_sizes(target):
    """Split clustered noise into shuffled connected-component sizes."""
    sizes = []
    remaining = int(target)
    while remaining > 0:
        if remaining <= NOISE_COMPONENT_SIZE_RANGE[1]:
            if remaining == 1:
                sizes[-1] += 1
            else:
                sizes.append(remaining)
            break
        size = int(np.random.randint(NOISE_COMPONENT_SIZE_RANGE[0], min(NOISE_COMPONENT_SIZE_RANGE[1], remaining - NOISE_COMPONENT_TAIL_BUFFER) + 1))
        sizes.append(size)
        remaining -= size
    np.random.shuffle(sizes)
    return sizes


def _gen_clear_of_old_noise(noise, r, c):
    h, w = noise.shape
    r0, r1 = max(0, r - 1), min(h, r + 2)
    c0, c1 = max(0, c - 1), min(w, c + 2)
    return not noise[r0:r1, c0:c1].any()


def _gen_add_frontier_neighbors(blocked, noise, comp, frontier, r, c):
    h, w = blocked.shape
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if (
            0 <= nr < h
            and 0 <= nc < w
            and not blocked[nr, nc]
            and not noise[nr, nc]
            and (nr, nc) not in comp
            and _gen_clear_of_old_noise(noise, nr, nc)
        ):
            frontier.add((nr, nc))


def _gen_place_component(blocked, noise, size):
    """Grow one isolated noise component without crossing blocked or prior noise."""
    candidates = np.argwhere(~blocked & ~noise)
    clear = [(int(r), int(c)) for r, c in candidates if _gen_clear_of_old_noise(noise, int(r), int(c))]
    seed_candidates = clear if len(clear) >= size else [(int(r), int(c)) for r, c in candidates]
    np.random.shuffle(seed_candidates)

    for sr, sc in seed_candidates[:NOISE_SEED_TRY_LIMIT]:
        comp = {(sr, sc)}
        frontier = set()
        _gen_add_frontier_neighbors(blocked, noise, comp, frontier, sr, sc)
        while len(comp) < size and frontier:
            choices = list(frontier)
            scores = []
            for r, c in choices:
                same = sum((r + dr, c + dc) in comp for dr, dc in NEIGHBORS_4)
                scores.append(-same + np.random.random() * NOISE_FRONTIER_RANDOM_WEIGHT)
            r, c = choices[int(np.argmin(scores))]
            frontier.remove((r, c))
            comp.add((r, c))
            _gen_add_frontier_neighbors(blocked, noise, comp, frontier, r, c)
        if len(comp) == size:
            return comp
    return None


def _gen_grow_cluster(blocked, target):
    noise = np.zeros(blocked.shape, dtype=bool)
    if int((~blocked).sum()) < target:
        return None

    for size in _gen_component_sizes(target):
        comp = _gen_place_component(blocked, noise, size)
        if comp is None:
            return None
        for r, c in comp:
            noise[r, c] = True

    return noise


def _gen_add_singletons(blocked, noise, target):
    candidates = np.argwhere(~blocked & ~noise)
    np.random.shuffle(candidates)
    remaining = int(target)
    for r, c in candidates:
        if remaining == 0:
            break
        r, c = int(r), int(c)
        if _gen_clear_of_old_noise(noise, r, c):
            noise[r, c] = True
            remaining -= 1
    return noise if remaining == 0 else None


def _gen_sample_setup():
    """Sample canvas, marker count, answer color, and shuffled marker assignments."""
    h = int(np.random.randint(*GEN_HEIGHT_RANGE))
    w = int(np.random.randint(*GEN_WIDTH_RANGE))
    bg = int(GEN_BACKGROUND_COLORS[np.random.randint(len(GEN_BACKGROUND_COLORS))])
    marker_n = int(np.random.randint(*MARKER_COUNT_RANGE))

    arm_palette = [c for c in GEN_BACKGROUND_COLORS if c != bg]
    chosen = np.random.choice(arm_palette, size=marker_n - 1, replace=False)
    answer = int(chosen[np.random.randint(len(chosen))])
    arm_assignments = [int(c) for c in chosen] + [answer]
    np.random.shuffle(arm_assignments)
    return h, w, bg, marker_n, answer, arm_assignments


def _gen_render_markers(grid, centers, arm_assignments):
    marker_mask = np.zeros(grid.shape, dtype=bool)
    for (r, c), color in zip(centers, arm_assignments):
        for dr, dc in MARKER_OFFSETS:
            marker_mask[r + dr, c + dc] = True
        paste(grid, _gen_marker_patch(color), r - 1, c - 1, transparent=GEN_TRANSPARENT)
    return marker_mask


def _gen_paint_noise(grid, noise_mask, noise_n, arm_assignments):
    noise_cells = np.argwhere(noise_mask)
    np.random.shuffle(noise_cells)
    black_n = int(np.random.randint(int(np.ceil(BLACK_NOISE_MIN_FRAC * noise_n)), noise_n + 1))
    for i, (r, c) in enumerate(noise_cells):
        r, c = int(r), int(c)
        if i < black_n:
            grid[r, c] = GEN_BLACK
        else:
            grid[r, c] = int(arm_assignments[np.random.randint(len(arm_assignments))])


def _gen_passes_oracle(grid, answer):
    expected = np.array([[answer]], dtype=int)
    try:
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _gen_attempt():
    h, w, bg, marker_n, answer, arm_assignments = _gen_sample_setup()
    total = h * w

    centers = _gen_place_centers(h, w, marker_n)
    if centers is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    marker_mask = _gen_render_markers(grid, centers, arm_assignments)

    noise_n = _gen_noise_count(total, marker_n * MARKER_CELLS)
    if noise_n is None:
        return None
    cluster_low = max(MIN_CLUSTERED_NOISE_CELLS, int(np.ceil(CLUSTERED_NOISE_FRAC_RANGE[0] * noise_n)))
    cluster_high = min(noise_n - 1, int(np.floor(CLUSTERED_NOISE_FRAC_RANGE[1] * noise_n)))
    if cluster_low > cluster_high:
        return None
    clustered_n = int(np.random.randint(cluster_low, cluster_high + 1))
    singleton_n = noise_n - clustered_n

    noise_mask = _gen_grow_cluster(marker_mask, clustered_n)
    if noise_mask is None:
        return None
    noise_mask = _gen_add_singletons(marker_mask, noise_mask, singleton_n)
    if noise_mask is None:
        return None

    _gen_paint_noise(grid, noise_mask, noise_n, arm_assignments)
    return grid if _gen_passes_oracle(grid, answer) else None


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 642d658d input")
