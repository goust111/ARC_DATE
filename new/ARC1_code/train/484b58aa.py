import numpy as np

from arc_utils import Color, find_components, paste, random_free_location
import random


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int):
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _max_supported_translations(g: np.ndarray, missing_color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    best_support = 0
    best: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue

            a = g[r0:r1, c0:c1]
            b = g[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = (a != missing_color) & (b != missing_color)
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue

            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))

    assert best_support > 0
    assert len(best) == 2
    assert best[0] == (-best[1][0], -best[1][1])
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Fill black gaps by propagating the maximally supported repeating translation.

    Input structure
    A same-size colored periodic field contains black pixels that mark missing cells.
    All non-black pixels are already correct and provide many repeated-color pairs.

    Rule
    Find the nonzero translation vector whose overlapping visible cells have the
    largest support with no color disagreements.  Cells connected by this vector
    and its inverse are one periodic equivalence class; replace each black cell by
    the unique visible color in its class.

    Key insight
    The occlusions hide a repeated lattice, and the strongest self-translation of
    the visible pixels identifies exactly how to copy colors through the gaps.
    """
    g = np.asarray(grid, dtype=int)
    missing_color = Color.BLACK
    assert np.any(g == missing_color)

    h, w = g.shape
    shifts = _max_supported_translations(g, missing_color)

    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for dr, dc in shifts:
        r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
        for r in range(r0, r1):
            rr = r + dr
            for c in range(c0, c1):
                union(r * w + c, rr * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == missing_color:
                continue
            root = find(r * w + c)
            colors_by_root.setdefault(root, set()).add(color)

    assert all(len(colors) == 1 for colors in colors_by_root.values())

    out = g.copy()
    for r, c in np.argwhere(g == missing_color):
        colors = colors_by_root.get(find(int(r) * w + int(c)), set())
        assert len(colors) == 1
        out[int(r), int(c)] = next(iter(colors))

    return out

# --------------- generator (o5x) ---------------

# Canvas and color bounds.
GEN_SHAPE = (29, 29)
GEN_HEIGHT, GEN_WIDTH = GEN_SHAPE
GEN_BACKGROUND = Color.BLACK
GEN_NON_BLACK_COLORS = range(1, 10)
GEN_COLOR_COUNT_RANGE = (5, 7)

# Lattice vector search bounds.
GEN_V1_COORD_RANGE = range(-6, 7)
GEN_V2_COORD_RANGE = range(-9, 10)
GEN_V1_NORM_SQ_RANGE, GEN_V2_NORM_SQ_RANGE = (8, 44), (21, 81)
GEN_TILE_AREA_RANGE = (12, 54)
GEN_REDUCED_COEFF_RANGE = range(-4, 5)

# Black mask component and blob bounds.
GEN_COMPONENT_COUNT_RANGE, GEN_BLACK_TOTAL_RANGE, GEN_BLOB_AREA_RANGE = (2, 4), (59, 109), (6, 57)
GEN_RECT_DIM_RANGE = range(2, 14)
GEN_L_BLOB_DIM_RANGE = range(3, 13)
GEN_RECT_BLOB_PROB = 0.35
GEN_PLACEMENT_BORDER, GEN_PLACEMENT_PADDING, GEN_PLACEMENT_CONNECTIVITY = 0, 1, 8

# Retry limit for rejection sampling.
GEN_MAX_ATTEMPTS = 300

_GEN_VECTOR_PAIRS = None


def _gen_overlap_slices(shape, dr, dc):
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _gen_support_area(vec, h=GEN_HEIGHT, w=GEN_WIDTH):
    dr, dc = vec
    if abs(dr) >= h or abs(dc) >= w:
        return 0
    return (h - abs(dr)) * (w - abs(dc))


def _gen_norm_sq(vec):
    return vec[0] * vec[0] + vec[1] * vec[1]


def _gen_in_range(value, bounds):
    return bounds[0] <= value <= bounds[1]


def _gen_lattice_area(v1, v2):
    return abs(v1[0] * v2[1] - v1[1] * v2[0])


def _gen_candidate_vectors(coord_range, norm_range):
    for dr in coord_range:
        for dc in coord_range:
            if dr == 0 or dc == 0:
                continue
            vec = (dr, dc)
            norm_sq = _gen_norm_sq(vec)
            if _gen_in_range(norm_sq, norm_range):
                yield vec, norm_sq


def _gen_is_reduced_pair(v1, v2):
    n1 = _gen_norm_sq(v1)
    s1 = _gen_support_area(v1)
    for a in GEN_REDUCED_COEFF_RANGE:
        for b in GEN_REDUCED_COEFF_RANGE:
            if a == 0 and b == 0:
                continue
            vec = (a * v1[0] + b * v2[0], a * v1[1] + b * v2[1])
            if vec == v1 or vec == (-v1[0], -v1[1]):
                continue
            if vec == (0, 0):
                return False
            if _gen_norm_sq(vec) <= n1:
                return False
            if abs(vec[0]) < GEN_HEIGHT and abs(vec[1]) < GEN_WIDTH and _gen_support_area(vec) >= s1:
                return False
    return True


def _gen_has_shorter_sum_or_diff(v1, v2, n1):
    return (
        _gen_norm_sq((v1[0] + v2[0], v1[1] + v2[1])) <= n1
        or _gen_norm_sq((v1[0] - v2[0], v1[1] - v2[1])) <= n1
    )


def _gen_vector_pairs():
    global _GEN_VECTOR_PAIRS
    if _GEN_VECTOR_PAIRS is not None:
        return _GEN_VECTOR_PAIRS

    choices = []
    v1_candidates = _gen_candidate_vectors(GEN_V1_COORD_RANGE, GEN_V1_NORM_SQ_RANGE)
    v2_candidates = list(_gen_candidate_vectors(GEN_V2_COORD_RANGE, GEN_V2_NORM_SQ_RANGE))
    for v1, n1 in v1_candidates:
        for v2, n2 in v2_candidates:
            area = _gen_lattice_area(v1, v2)
            if not (n2 > n1 and _gen_in_range(area, GEN_TILE_AREA_RANGE)):
                continue
            if _gen_has_shorter_sum_or_diff(v1, v2, n1):
                continue
            if _gen_is_reduced_pair(v1, v2):
                choices.append((v1, v2))
    _GEN_VECTOR_PAIRS = choices
    return choices


def _gen_vector_pair():
    return random.choice(_gen_vector_pairs())


def _gen_lattice_key(r, c, v1, v2):
    a, b = v1
    p, q = v2
    det = a * q - b * p
    area = abs(det)
    return ((q * r - p * c) % area, (-b * r + a * c) % area)


def _gen_assign_lattice_colors(keys):
    n_colors = random.randint(*GEN_COLOR_COUNT_RANGE)
    palette = random.sample(GEN_NON_BLACK_COLORS, n_colors)
    key_list = list(keys)
    random.shuffle(key_list)
    for i, key in enumerate(key_list):
        keys[key] = palette[i] if i < n_colors else random.choice(palette)
    return n_colors


def _gen_periodic_field(v1, v2):
    keys = {}
    for r in range(GEN_HEIGHT):
        for c in range(GEN_WIDTH):
            keys.setdefault(_gen_lattice_key(r, c, v1, v2), None)

    if len(keys) != _gen_lattice_area(v1, v2):
        return None

    n_colors = _gen_assign_lattice_colors(keys)
    out = np.zeros(GEN_SHAPE, dtype=int)
    for r in range(GEN_HEIGHT):
        for c in range(GEN_WIDTH):
            out[r, c] = keys[_gen_lattice_key(r, c, v1, v2)]
    if len(set(map(int, out.ravel()))) != n_colors:
        return None
    return out


def _gen_split_total(total, k, lo=GEN_BLOB_AREA_RANGE[0], hi=GEN_BLOB_AREA_RANGE[1]):
    sizes = [lo] * k
    remaining = total - lo * k
    order = list(range(k))
    random.shuffle(order)
    for pos, idx in enumerate(order[:-1]):
        cap = min(hi - lo, remaining)
        max_add = min(cap, remaining)
        add = random.randint(0, max_add)
        sizes[idx] += add
        remaining -= add
    sizes[order[-1]] += remaining
    if any(s < lo or s > hi for s in sizes):
        return None
    random.shuffle(sizes)
    return sizes


def _gen_l_blob(area):
    candidates = [(h, w) for h in GEN_L_BLOB_DIM_RANGE for w in GEN_L_BLOB_DIM_RANGE if h * w >= area and area >= h + w - 1]
    if not candidates:
        return None
    h, w = random.choice(candidates)
    mask = np.ones((h, w), dtype=int)
    remove = h * w - area
    for r in range(h - 1, 0, -1):
        if remove <= 0:
            break
        take = min(remove, w - 1)
        mask[r, w - take : w] = 0
        remove -= take
    if remove != 0:
        return None
    return mask


def _gen_random_orientation(mask):
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = mask.T
    return np.ascontiguousarray(mask, dtype=int)


def _gen_blob(area):
    rects = [(h, w) for h in GEN_RECT_DIM_RANGE for w in GEN_RECT_DIM_RANGE if h * w == area]
    if rects and random.random() < GEN_RECT_BLOB_PROB:
        h, w = random.choice(rects)
        mask = np.ones((h, w), dtype=int)
    else:
        mask = _gen_l_blob(area)
        if mask is None:
            return None

    return _gen_random_orientation(mask)


def _gen_black_mask():
    k = random.randint(*GEN_COMPONENT_COUNT_RANGE)
    total = random.randint(*GEN_BLACK_TOTAL_RANGE)
    sizes = _gen_split_total(total, k)
    if sizes is None:
        return None

    canvas = np.zeros(GEN_SHAPE, dtype=int)
    for area in sizes:
        patch = _gen_blob(area)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=GEN_BACKGROUND,
                border=GEN_PLACEMENT_BORDER,
                padding=GEN_PLACEMENT_PADDING,
                padding_connectivity=GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=GEN_BACKGROUND)

    comps = find_components(canvas, bg=GEN_BACKGROUND, connectivity=4, monochromatic=True)
    if len(comps) != k:
        return None
    if sum(int(comp["area"]) for comp in comps) != total:
        return None
    return canvas.astype(bool)


def _gen_v1_classes_visible(mask, v1):
    h, w = mask.shape
    parent = list(range(h * w))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    dr, dc = v1
    r0, r1, c0, c1 = _gen_overlap_slices(mask.shape, dr, dc)
    for r in range(r0, r1):
        for c in range(c0, c1):
            union(r * w + c, (r + dr) * w + c + dc)

    all_roots = {find(i) for i in range(h * w)}
    visible_roots = {find(r * w + c) for r in range(h) for c in range(w) if not mask[r, c]}
    return all_roots == visible_roots


def _gen_best_translations(g):
    h, w = g.shape
    best_support = 0
    best = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _gen_overlap_slices(g.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            a = g[r0:r1, c0:c1]
            b = g[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = (a != GEN_BACKGROUND) & (b != GEN_BACKGROUND)
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue
            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))
    return best


def _gen_build_candidate():
    """Return (input, output, v1) after layout/mask checks, or None on rejection."""
    v1, v2 = _gen_vector_pair()
    out = _gen_periodic_field(v1, v2)
    if out is None:
        return None

    mask = _gen_black_mask()
    if mask is None or not _gen_v1_classes_visible(mask, v1):
        return None

    inp = out.copy()
    inp[mask] = 0
    return inp, out, v1


def _gen_translation_target(v1):
    return {v1, (-v1[0], -v1[1])}


def _gen_oracle_solves(inp, out):
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt():
    candidate = _gen_build_candidate()
    if candidate is None:
        return None
    inp, out, v1 = candidate
    if set(_gen_best_translations(inp)) != _gen_translation_target(v1):
        return None

    if not _gen_oracle_solves(inp, out):
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 484b58aa input")
