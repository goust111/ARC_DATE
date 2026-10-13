import numpy as np

from arc_utils import components_mask, get_bg


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != bg], return_counts=True)
    assert vals.size > 0, "expected foreground pixels"
    max_count = int(counts.max())
    major = vals[counts == max_count]
    assert major.size == 1, "expected a unique majority foreground color"
    return int(major[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid interior pixel of a seeded foreground component to that component's marker color.

    Input structure
    - The background is the dominant color.
    - A unique majority foreground color forms 4-connected networks of one-cell paths and thick solid blobs.
    - Each non-background 4-connected component contains exactly one singleton marker pixel in a non-majority color.

    Rule
    - For each foreground component, find pixels whose full 3x3 neighborhood lies inside that same component.
    - Recolor those 3x3-supported pixels to the component's marker color.
    - Leave background, one-cell paths, and blob borders unchanged.

    Key insight
    - Thick blob interiors are exactly the component pixels with all eight neighbors present; corridors and borders fail this local 3x3 test.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    base = _major_foreground_color(g, bg)
    out = g.copy()

    for coords in components_mask(g != bg, connectivity=4):
        comp_values = g[coords[:, 0], coords[:, 1]]
        marker_colors = np.setdiff1d(np.unique(comp_values), np.array([base], dtype=int))
        assert marker_colors.size == 1, "each component must contain exactly one marker color"
        marker = int(marker_colors[0])
        assert np.count_nonzero(comp_values == marker) == 1, "marker color must appear once in its component"

        comp_mask = np.zeros(g.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        for r, c in coords:
            r = int(r)
            c = int(c)
            if r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1:
                continue
            if comp_mask[r - 1 : r + 2, c - 1 : c + 2].all():
                out[r, c] = marker

    return out

# --------------- generator (o5x) ---------------

# Canvas and component layout bounds.
GEN_CANVAS_MIN = 18
GEN_CANVAS_MAX = 30
GEN_COMPONENT_COUNTS = (2, 3)
GEN_ZONE_GAP = 2
GEN_ZONE_MIN_SIZE = 7
GEN_THREE_STRIP_MIN_SIZE = 25

# Target area and density controls.
GEN_DENSITY_MIN = 0.31
GEN_DENSITY_MAX = 0.47
GEN_ZONE_MIN_TARGET = 20
GEN_ZONE_CAP_RATIO = 0.86

# Block and corridor construction bounds.
GEN_BLOCK_H_MIN = 3
GEN_BLOCK_H_MAX = 7
GEN_BLOCK_W_MIN = 3
GEN_BLOCK_W_MAX = 12
GEN_MIN_BLOCKS = 2
GEN_MAX_BLOCKS = 9
GEN_BLOCK_MIN_AREA = 9
GEN_INITIAL_BLOCK_WINDOW = 8
GEN_INITIAL_BLOCK_JITTER = 3
GEN_ADD_SIZE_WINDOW = 18
GEN_ADD_CHOICE_WINDOW = 24
GEN_CORRIDOR_LEN_MIN = 1
GEN_CORRIDOR_LEN_MAX = 6

# Component validation ratios.
GEN_BLOCK_INTERIOR_RATIO_CAP = 0.44
GEN_BLOCK_GOAL_MIN_AREA_DIVISOR = 78.0
GEN_BLOCK_GOAL_MAX_AREA_DIVISOR = 13.0
GEN_COMPONENT_AREA_MIN_RATIO = 0.72
GEN_COMPONENT_AREA_MAX_RATIO = 1.32
GEN_COMPONENT_GROW_RATIO = 0.86
GEN_INTERIOR_RATIO_MIN = 0.10
GEN_INTERIOR_RATIO_MAX = 0.42
GEN_MIN_CHEB_SEPARATION = 2

# Colors and retry limits.
GEN_BACKGROUND = 0
GEN_FOREGROUND = 1
GEN_MARKER_COLORS = (2, 3, 4, 6, 8)
GEN_RETRIES = 300


def _gen_randint(rng, lo, hi):
    return int(rng.integers(int(lo), int(hi) + 1))


def _gen_split_two(rng, n, gap=GEN_ZONE_GAP, min_size=GEN_ZONE_MIN_SIZE):
    avail = int(n) - int(gap)
    if avail < 2 * int(min_size):
        return None
    first = _gen_randint(rng, min_size, avail - min_size)
    return first, avail - first


def _gen_split_three(rng, n, gap=GEN_ZONE_GAP, min_size=GEN_ZONE_MIN_SIZE):
    avail = int(n) - 2 * int(gap)
    if avail < 3 * int(min_size):
        return None
    a = _gen_randint(rng, min_size, avail - 2 * min_size)
    b = _gen_randint(rng, min_size, avail - a - min_size)
    return a, b, avail - a - b


def _gen_zones(rng, n, k):
    gap = GEN_ZONE_GAP
    if k == 2:
        parts = _gen_split_two(rng, n, gap=gap)
        if parts is None:
            return None
        a, b = parts
        if bool(rng.integers(0, 2)):
            return [(0, 0, n - 1, a - 1), (0, a + gap, n - 1, n - 1)]
        return [(0, 0, a - 1, n - 1), (a + gap, 0, n - 1, n - 1)]

    if n >= GEN_THREE_STRIP_MIN_SIZE and bool(rng.integers(0, 2)):
        parts = _gen_split_three(rng, n, gap=gap)
        if parts is not None:
            a, b, _c = parts
            if bool(rng.integers(0, 2)):
                return [
                    (0, 0, n - 1, a - 1),
                    (0, a + gap, n - 1, a + gap + b - 1),
                    (0, a + gap + b + gap, n - 1, n - 1),
                ]
            return [
                (0, 0, a - 1, n - 1),
                (a + gap, 0, a + gap + b - 1, n - 1),
                (a + gap + b + gap, 0, n - 1, n - 1),
            ]

    row_parts = _gen_split_two(rng, n, gap=gap)
    col_parts = _gen_split_two(rng, n, gap=gap)
    if row_parts is None or col_parts is None:
        return None
    h0, _h1 = row_parts
    w0, _w1 = col_parts
    zones = [
        (0, 0, h0 - 1, w0 - 1),
        (0, w0 + gap, h0 - 1, n - 1),
        (h0 + gap, 0, n - 1, w0 - 1),
        (h0 + gap, w0 + gap, n - 1, n - 1),
    ]
    pick = rng.choice(4, size=3, replace=False)
    return [zones[int(i)] for i in pick]


def _gen_allocate_targets(rng, total, zones):
    mins = []
    caps = []
    for r0, c0, r1, c1 in zones:
        area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
        mins.append(GEN_ZONE_MIN_TARGET)
        caps.append(max(GEN_ZONE_MIN_TARGET, int(area * GEN_ZONE_CAP_RATIO)))
    if total < sum(mins) or total > sum(caps):
        return None

    extras = [0] * len(zones)
    rem = int(total) - sum(mins)
    order = [int(i) for i in rng.permutation(len(zones))]
    for pos, idx in enumerate(order):
        rest = order[pos + 1 :]
        rest_cap = sum(caps[j] - mins[j] for j in rest)
        lo = max(0, rem - rest_cap)
        hi = min(caps[idx] - mins[idx], rem)
        val = _gen_randint(rng, lo, hi)
        extras[idx] = val
        rem -= val
    return [mins[i] + extras[i] for i in range(len(zones))]


def _gen_block_candidates(max_h, max_w, desired_area):
    out = []
    for h in range(GEN_BLOCK_H_MIN, min(GEN_BLOCK_H_MAX, int(max_h)) + 1):
        for w in range(GEN_BLOCK_W_MIN, min(GEN_BLOCK_W_MAX, int(max_w)) + 1):
            ratio = ((h - 2) * (w - 2)) / float(h * w)
            if ratio <= GEN_BLOCK_INTERIOR_RATIO_CAP:
                out.append((h, w, abs(h * w - float(desired_area))))
    out.sort(key=lambda x: x[2])
    return [(h, w) for h, w, _ in out]


def _gen_place_initial_block(rng, mask, target_area, block_goal):
    hmax, wmax = mask.shape
    desired = max(GEN_BLOCK_MIN_AREA, target_area / max(1, block_goal))
    choices = _gen_block_candidates(hmax, wmax, desired)
    if not choices:
        return None
    h, w = choices[_gen_randint(rng, 0, min(len(choices), GEN_INITIAL_BLOCK_WINDOW) - 1)]
    base_r = (hmax - h) // 2
    base_c = (wmax - w) // 2
    jitter_r = max(0, min(GEN_INITIAL_BLOCK_JITTER, hmax - h))
    jitter_c = max(0, min(GEN_INITIAL_BLOCK_JITTER, wmax - w))
    r0 = min(max(0, base_r + _gen_randint(rng, -jitter_r, jitter_r)), hmax - h)
    c0 = min(max(0, base_c + _gen_randint(rng, -jitter_c, jitter_c)), wmax - w)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _gen_add_block(rng, mask, blocks, target_area, blocks_left):
    hmax, wmax = mask.shape
    current = int(mask.sum())
    desired = max(GEN_BLOCK_MIN_AREA, (target_area - current) / max(1, blocks_left))
    size_choices = _gen_block_candidates(hmax, wmax, desired)
    if not size_choices:
        return None
    size_choices = size_choices[: min(len(size_choices), GEN_ADD_SIZE_WINDOW)]

    candidates = []
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for bi in rng.permutation(len(blocks)):
        pr0, pc0, pr1, pc1 = blocks[int(bi)]
        for h, w in size_choices:
            for dr, dc in dirs:
                for length in rng.permutation(np.arange(GEN_CORRIDOR_LEN_MIN, GEN_CORRIDOR_LEN_MAX + 1)):
                    length = int(length)
                    if dc != 0:
                        rows = range(pr0 + 1, pr1)
                        for rr in rows:
                            for off in range(1, h - 1):
                                nr0 = rr - off
                                nr1 = nr0 + h - 1
                                if dc > 0:
                                    nc0 = pc1 + length + 1
                                    nc1 = nc0 + w - 1
                                    corr = (rr, pc1 + 1, rr, nc0 - 1)
                                else:
                                    nc1 = pc0 - length - 1
                                    nc0 = nc1 - w + 1
                                    corr = (rr, nc1 + 1, rr, pc0 - 1)
                                if _gen_valid_new_piece(mask, nr0, nc0, nr1, nc1, corr):
                                    add_area = h * w + length
                                    score = abs((current + add_area) - target_area)
                                    candidates.append((score, nr0, nc0, nr1, nc1, corr))
                    else:
                        cols = range(pc0 + 1, pc1)
                        for cc in cols:
                            for off in range(1, w - 1):
                                nc0 = cc - off
                                nc1 = nc0 + w - 1
                                if dr > 0:
                                    nr0 = pr1 + length + 1
                                    nr1 = nr0 + h - 1
                                    corr = (pr1 + 1, cc, nr0 - 1, cc)
                                else:
                                    nr1 = pr0 - length - 1
                                    nr0 = nr1 - h + 1
                                    corr = (nr1 + 1, cc, pr0 - 1, cc)
                                if _gen_valid_new_piece(mask, nr0, nc0, nr1, nc1, corr):
                                    add_area = h * w + length
                                    score = abs((current + add_area) - target_area)
                                    candidates.append((score, nr0, nc0, nr1, nc1, corr))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    choice = candidates[_gen_randint(rng, 0, min(len(candidates), GEN_ADD_CHOICE_WINDOW) - 1)]
    _, nr0, nc0, nr1, nc1, corr = choice
    mask[nr0 : nr1 + 1, nc0 : nc1 + 1] = True
    cr0, cc0, cr1, cc1 = corr
    mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = True
    return (nr0, nc0, nr1, nc1)


def _gen_valid_new_piece(mask, r0, c0, r1, c1, corr):
    hmax, wmax = mask.shape
    if r0 < 0 or c0 < 0 or r1 >= hmax or c1 >= wmax:
        return False
    if bool(mask[r0 : r1 + 1, c0 : c1 + 1].any()):
        return False
    cr0, cc0, cr1, cc1 = corr
    if cr0 < 0 or cc0 < 0 or cr1 >= hmax or cc1 >= wmax:
        return False
    return not bool(mask[cr0 : cr1 + 1, cc0 : cc1 + 1].any())


def _gen_interior_mask(mask):
    out = np.zeros(mask.shape, dtype=bool)
    out[1:-1, 1:-1] = (
        mask[:-2, :-2]
        & mask[:-2, 1:-1]
        & mask[:-2, 2:]
        & mask[1:-1, :-2]
        & mask[1:-1, 1:-1]
        & mask[1:-1, 2:]
        & mask[2:, :-2]
        & mask[2:, 1:-1]
        & mask[2:, 2:]
    )
    return out


def _gen_has_endpoint(mask):
    neighbor_count = np.zeros(mask.shape, dtype=np.uint8)
    neighbor_count[1:, :] += mask[:-1, :]
    neighbor_count[:-1, :] += mask[1:, :]
    neighbor_count[:, 1:] += mask[:, :-1]
    neighbor_count[:, :-1] += mask[:, 1:]
    return bool((mask & (neighbor_count < 2)).any())


def _gen_build_component(rng, zone_h, zone_w, target_area):
    mask = np.zeros((int(zone_h), int(zone_w)), dtype=bool)
    min_blocks = max(GEN_MIN_BLOCKS, int(np.ceil(target_area / GEN_BLOCK_GOAL_MIN_AREA_DIVISOR)))
    max_blocks = min(GEN_MAX_BLOCKS, max(min_blocks, int(np.ceil(target_area / GEN_BLOCK_GOAL_MAX_AREA_DIVISOR))))
    block_goal = _gen_randint(rng, min_blocks, max_blocks)

    first = _gen_place_initial_block(rng, mask, target_area, block_goal)
    if first is None:
        return None
    blocks = [first]
    while len(blocks) < block_goal:
        new_block = _gen_add_block(rng, mask, blocks, target_area, block_goal - len(blocks))
        if new_block is None:
            break
        blocks.append(new_block)

    while len(blocks) < GEN_MAX_BLOCKS and int(mask.sum()) < int(target_area * GEN_COMPONENT_GROW_RATIO):
        new_block = _gen_add_block(rng, mask, blocks, target_area, 1)
        if new_block is None:
            break
        blocks.append(new_block)

    if len(blocks) < GEN_MIN_BLOCKS:
        return None
    area = int(mask.sum())
    if area < int(target_area * GEN_COMPONENT_AREA_MIN_RATIO) or area > int(
        target_area * GEN_COMPONENT_AREA_MAX_RATIO
    ):
        return None
    interior = int(_gen_interior_mask(mask).sum())
    ratio = interior / float(area)
    if not (GEN_INTERIOR_RATIO_MIN <= ratio <= GEN_INTERIOR_RATIO_MAX):
        return None
    if _gen_has_endpoint(mask):
        return None
    return mask


def _gen_min_cheb(coords_a, coords_b):
    best = 10**9
    for r, c in coords_a:
        dist = np.maximum(np.abs(coords_b[:, 0] - int(r)), np.abs(coords_b[:, 1] - int(c)))
        val = int(dist.min())
        if val < best:
            best = val
            if best < GEN_MIN_CHEB_SEPARATION:
                return best
    return best


def _gen_component_mask(shape, coords):
    comp = np.zeros(shape, dtype=bool)
    comp[coords[:, 0], coords[:, 1]] = True
    return comp


def _gen_validate_grid(grid, expected_k):
    n = grid.shape[0]
    density = float(np.count_nonzero(grid)) / float(n * n)
    if not (GEN_DENSITY_MIN <= density <= GEN_DENSITY_MAX):
        return False
    comps = components_mask(grid != GEN_BACKGROUND, connectivity=4)
    if len(comps) != int(expected_k):
        return False

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _gen_min_cheb(comps[i], comps[j]) < GEN_MIN_CHEB_SEPARATION:
                return False

    for coords in comps:
        vals = grid[coords[:, 0], coords[:, 1]]
        markers = [int(v) for v in np.unique(vals) if int(v) not in (GEN_BACKGROUND, GEN_FOREGROUND)]
        if len(markers) != 1 or int(np.count_nonzero(vals == markers[0])) != 1:
            return False
        comp = _gen_component_mask(grid.shape, coords)
        interiors = _gen_interior_mask(comp)
        area = int(comp.sum())
        ratio = float(interiors.sum()) / float(area)
        if not (GEN_INTERIOR_RATIO_MIN <= ratio <= GEN_INTERIOR_RATIO_MAX):
            return False
        mr, mc = coords[np.flatnonzero(vals == markers[0])[0]]
        if not bool(interiors[int(mr), int(mc)]):
            return False
        if _gen_has_endpoint(comp):
            return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _gen_plan_layout(rng):
    """Choose the canvas, disjoint zones, and per-zone area targets."""
    n = _gen_randint(rng, GEN_CANVAS_MIN, GEN_CANVAS_MAX)
    k = GEN_COMPONENT_COUNTS[0] if bool(rng.integers(0, 2)) else GEN_COMPONENT_COUNTS[1]
    zones = _gen_zones(rng, n, k)
    if zones is None:
        return None

    lo = int(np.ceil(GEN_DENSITY_MIN * n * n))
    hi = int(np.floor(GEN_DENSITY_MAX * n * n))
    cap = sum(int((r1 - r0 + 1) * (c1 - c0 + 1) * GEN_ZONE_CAP_RATIO) for r0, c0, r1, c1 in zones)
    hi = min(hi, cap)
    if lo > hi:
        return None
    total_target = _gen_randint(rng, lo, hi)
    targets = _gen_allocate_targets(rng, total_target, zones)
    if targets is None:
        return None
    return n, k, zones, targets


def _gen_render_components(rng, n, zones, targets):
    """Fill each zone with one constructed foreground network."""
    grid = np.zeros((n, n), dtype=int)
    for zone, target in zip(zones, targets):
        r0, c0, r1, c1 = zone
        mask = _gen_build_component(rng, r1 - r0 + 1, c1 - c0 + 1, target)
        if mask is None:
            return None
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        view[mask] = GEN_FOREGROUND
    return grid


def _gen_place_markers(rng, grid, k):
    """Place one unique marker color on an interior cell of each component."""
    color_order = [GEN_MARKER_COLORS[int(i)] for i in rng.permutation(len(GEN_MARKER_COLORS))]
    comps = components_mask(grid != GEN_BACKGROUND, connectivity=4)
    if len(comps) != k:
        return False
    for coords, color in zip(comps, color_order):
        comp = _gen_component_mask(grid.shape, coords)
        interior_coords = np.argwhere(_gen_interior_mask(comp))
        if interior_coords.shape[0] == 0:
            return False
        rr, cc = interior_coords[_gen_randint(rng, 0, interior_coords.shape[0] - 1)]
        grid[int(rr), int(cc)] = int(color)
    return True


def _gen_attempt(rng):
    plan = _gen_plan_layout(rng)
    if plan is None:
        return None
    n, k, zones, targets = plan

    grid = _gen_render_components(rng, n, zones, targets)
    if grid is None:
        return None

    if not _gen_place_markers(rng, grid, k):
        return None
    if not _gen_validate_grid(grid, k):
        return None
    return grid


def generate_input():
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 09c534e7 input")
