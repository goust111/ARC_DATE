import numpy as np
from functools import lru_cache
from arc_utils import components_mask as _gen_components_mask


MASK = 6


def _orbit(r: int, c: int, symmetry_sum: int, n: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (c, r),
        (symmetry_sum - r, c),
        (r, symmetry_sum - c),
        (symmetry_sum - r, symmetry_sum - c),
        (symmetry_sum - c, symmetry_sum - r),
        (c, symmetry_sum - r),
        (symmetry_sum - c, r),
    }
    return tuple(sorted((rr, cc) for rr, cc in pts if 0 <= rr < n and 0 <= cc < n))


def _symmetry_score(g: np.ndarray, symmetry_sum: int) -> int | None:
    n = g.shape[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    score = 0
    for r in range(n):
        for c in range(n):
            orbit = _orbit(r, c, symmetry_sum, n)
            if orbit in seen:
                continue
            seen.add(orbit)

            known = {int(g[rr, cc]) for rr, cc in orbit if g[rr, cc] != MASK}
            has_mask = any(g[rr, cc] == MASK for rr, cc in orbit)
            if len(known) > 1 or (has_mask and len(known) != 1):
                return None
            if len(orbit) > 1:
                score += len(orbit)
    return score


def _symmetry_sum(g: np.ndarray) -> int:
    n = g.shape[0]
    scored: list[tuple[int, int]] = []
    for symmetry_sum in range(2 * n + 3):
        score = _symmetry_score(g, symmetry_sum)
        if score is not None:
            scored.append((score, symmetry_sum))

    assert scored, "at least one square-symmetry frame must fit the known cells"
    best_score = max(score for score, _ in scored)
    best = [symmetry_sum for score, symmetry_sum in scored if score == best_score]
    assert len(best) == 1, f"the known cells must determine one strongest symmetry frame, got {best}"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each pink masked pixel with the color from its matching square-symmetry orbit.

    Input structure:
    - The grid is square and fully colored.
    - Pink pixels are masks covering missing cells; every other pixel is part of the completed pattern.
    - The completed pattern obeys main-diagonal symmetry and reflected square symmetry in a slightly larger hidden frame.

    Rule:
    - Infer the reflected square frame whose visible symmetry orbits make all non-pink cells agree.
    - For each pink pixel, collect all visible cells related by diagonal, horizontal, vertical, and rotated reflections in that frame.
    - Recolor the pink pixel to the unique non-pink color already present in its orbit.

    Key insight:
    Pink is not an object color; it hides entries of one globally D4-symmetric color table, so each missing value is copied from its symmetry orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"
    assert np.any(g == MASK), "input must contain pink masked pixels"

    n = g.shape[0]
    symmetry_sum = _symmetry_sum(g)
    out = g.copy()
    for r, c in zip(*np.where(g == MASK)):
        vals = {int(g[rr, cc]) for rr, cc in _orbit(int(r), int(c), symmetry_sum, n) if g[rr, cc] != MASK}
        assert len(vals) == 1, f"masked cell {(int(r), int(c))} must have one known orbit color"
        out[r, c] = vals.pop()

    assert not np.any(out == MASK), "all masked pixels must be filled"
    return out

# --------------- generator (o5x) ---------------

# Grid, symmetry frame, and palette.
_GEN_N = 30
_GEN_SYM_SUM = 31
_GEN_MASK = 6
_GEN_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)

# Orbit-color allocation bounds.
_GEN_MIN_COLOR_ORBITS, _GEN_MAX_COLOR_ORBITS = 5, 36

# Mask size and connected-component constraints.
_GEN_TARGET_MIN_CELLS = 39
_GEN_TARGET_MAX_CELLS_EXCLUSIVE = 143
_GEN_MIN_COMPONENTS, _GEN_MAX_COMPONENTS = 2, 4
_GEN_COMPONENT_MIN_CELLS, _GEN_COMPONENT_MAX_CELLS = 8, 87

# Rectangle primitives and placement spacing.
_GEN_RECT_MIN_SIDE, _GEN_RECT_MAX_SIDE = 2, 8
_GEN_RECT_MIN_AREA = 8
_GEN_RECT_MAX_AREA = _GEN_RECT_MAX_SIDE * _GEN_RECT_MAX_SIDE
_GEN_RECT_PADDING = 1
_GEN_RECT_DIMS = tuple(
    (h, w)
    for h in range(_GEN_RECT_MIN_SIDE, _GEN_RECT_MAX_SIDE + 1)
    for w in range(_GEN_RECT_MIN_SIDE, _GEN_RECT_MAX_SIDE + 1)
    if h * w >= _GEN_RECT_MIN_AREA
)

# Layout merge probabilities and retry limits.
_GEN_MERGED_PAIR_ATTEMPTS = 80
_GEN_MERGE_PROB_FOR_THREE_RECTS = 0.85
_GEN_FOUR_RECT_ZERO_MERGE_CUTOFF = 0.10
_GEN_FOUR_RECT_ONE_MERGE_CUTOFF = 0.35
_GEN_GENERATION_ATTEMPTS = 300

# Stateful bag smooths the sampled mask-cell targets across calls.
_GEN_TARGET_BAG = []


def _gen_all_orbits():
    seen = set()
    orbits = []
    for r in range(_GEN_N):
        for c in range(_GEN_N):
            orb = _orbit(r, c, _GEN_SYM_SUM, _GEN_N)
            if orb not in seen:
                seen.add(orb)
                orbits.append(orb)
    return orbits


_GEN_ORBITS = _gen_all_orbits()


def _gen_color_counts():
    counts = []
    remaining = len(_GEN_ORBITS)
    for i in range(len(_GEN_COLORS)):
        slots = len(_GEN_COLORS) - i - 1
        lo = max(_GEN_MIN_COLOR_ORBITS, remaining - _GEN_MAX_COLOR_ORBITS * slots)
        hi = min(_GEN_MAX_COLOR_ORBITS, remaining - _GEN_MIN_COLOR_ORBITS * slots)
        count = int(np.random.randint(lo, hi + 1))
        counts.append(count)
        remaining -= count
    np.random.shuffle(counts)
    return counts


def _gen_target_cell_count():
    if not _GEN_TARGET_BAG:
        targets = np.arange(_GEN_TARGET_MIN_CELLS, _GEN_TARGET_MAX_CELLS_EXCLUSIVE)
        _GEN_TARGET_BAG.extend(np.random.permutation(targets).astype(int).tolist())
    return int(_GEN_TARGET_BAG.pop())


def _gen_colored_output():
    colors = []
    for color, count in zip(np.random.permutation(_GEN_COLORS), _gen_color_counts()):
        colors.extend([int(color)] * int(count))
    np.random.shuffle(colors)

    out = np.zeros((_GEN_N, _GEN_N), dtype=int)
    for orbit_i, color in zip(np.random.permutation(len(_GEN_ORBITS)), colors):
        for r, c in _GEN_ORBITS[int(orbit_i)]:
            out[r, c] = color
    return out


@lru_cache(maxsize=None)
def _gen_can_sum_rect_areas(total, rect_count):
    if rect_count == 0:
        return total == 0
    return any(
        _gen_can_sum_rect_areas(total - h * w, rect_count - 1)
        for h, w in _GEN_RECT_DIMS
    )


def _gen_nearest_rect_area_sum(target, rect_count):
    lo = _GEN_RECT_MIN_AREA * rect_count
    hi = _GEN_RECT_MAX_AREA * rect_count
    possible = [v for v in range(lo, hi + 1) if _gen_can_sum_rect_areas(v, rect_count)]
    best = min(abs(v - target) for v in possible)
    choices = [v for v in possible if abs(v - target) == best]
    return int(choices[int(np.random.randint(len(choices)))])


def _gen_rect_sizes(rect_count, target):
    target = _gen_nearest_rect_area_sum(target, rect_count)
    remaining = target
    dims = []
    for i in range(rect_count):
        slots = rect_count - i - 1
        choices = [
            dim for dim in _GEN_RECT_DIMS
            if _gen_can_sum_rect_areas(remaining - dim[0] * dim[1], slots)
        ]
        h, w = choices[int(np.random.randint(len(choices)))]
        dims.append((h, w))
        remaining -= h * w
    np.random.shuffle(dims)
    return dims


def _gen_rect_location(mask, h, w, padding):
    locs = []
    for r in range(_GEN_N - h + 1):
        r0 = max(0, r - padding)
        r1 = min(_GEN_N, r + h + padding)
        for c in range(_GEN_N - w + 1):
            c0 = max(0, c - padding)
            c1 = min(_GEN_N, c + w + padding)
            if not mask[r0:r1, c0:c1].any():
                locs.append((r, c))
    if not locs:
        return None
    return locs[int(np.random.randint(len(locs)))]


def _gen_paint_rect(mask, rect):
    r, c, h, w = rect
    mask[r:r + h, c:c + w] = True


def _gen_can_place_mask(existing, new_mask, padding):
    coords = np.argwhere(new_mask)
    r0 = max(0, int(coords[:, 0].min()) - padding)
    r1 = min(_GEN_N, int(coords[:, 0].max()) + padding + 1)
    c0 = max(0, int(coords[:, 1].min()) - padding)
    c1 = min(_GEN_N, int(coords[:, 1].max()) + padding + 1)
    return not existing[r0:r1, c0:c1].any()


def _gen_merged_pair(existing, h1, w1, h2, w2):
    for _ in range(_GEN_MERGED_PAIR_ATTEMPTS):
        r1 = int(np.random.randint(0, _GEN_N - h1 + 1))
        c1 = int(np.random.randint(0, _GEN_N - w1 + 1))
        direction = int(np.random.randint(4))
        if direction < 2:
            c2 = c1 + w1 if direction == 0 else c1 - w2
            r2 = int(np.random.randint(r1 - h2 + 1, r1 + h1))
        else:
            r2 = r1 + h1 if direction == 2 else r1 - h2
            c2 = int(np.random.randint(c1 - w2 + 1, c1 + w1))

        if not (0 <= r2 <= _GEN_N - h2 and 0 <= c2 <= _GEN_N - w2):
            continue

        pair_mask = np.zeros((_GEN_N, _GEN_N), dtype=bool)
        _gen_paint_rect(pair_mask, (r1, c1, h1, w1))
        _gen_paint_rect(pair_mask, (r2, c2, h2, w2))
        area = int(pair_mask.sum())
        if (
            _GEN_COMPONENT_MIN_CELLS <= area <= _GEN_COMPONENT_MAX_CELLS
            and _gen_can_place_mask(existing, pair_mask, padding=_GEN_RECT_PADDING)
        ):
            return [(r1, c1, h1, w1), (r2, c2, h2, w2)]
    return None


def _gen_merge_count(rect_count):
    if rect_count == 3:
        return int(np.random.random() < _GEN_MERGE_PROB_FOR_THREE_RECTS)
    if rect_count == 4:
        pick = np.random.random()
        if pick < _GEN_FOUR_RECT_ZERO_MERGE_CUTOFF:
            return 0
        if pick < _GEN_FOUR_RECT_ONE_MERGE_CUTOFF:
            return 1
        return 2
    return 0


def _gen_groupings(dims, merge_count):
    dims = tuple(dims)
    if merge_count == 0:
        return [[(dim,) for dim in dims]]

    out = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            if dims[i][0] * dims[i][1] + dims[j][0] * dims[j][1] > _GEN_COMPONENT_MAX_CELLS:
                continue
            rest = tuple(dim for k, dim in enumerate(dims) if k not in (i, j))
            for tail in _gen_groupings(rest, merge_count - 1):
                out.append([(dims[i], dims[j])] + tail)
    return out


def _gen_rect_groups(dims):
    preferred = _gen_merge_count(len(dims))
    for merge_count in range(preferred, -1, -1):
        groupings = _gen_groupings(tuple(dims), merge_count)
        if groupings:
            groups = groupings[int(np.random.randint(len(groupings)))]
            np.random.shuffle(groups)
            return groups
    return [(dim,) for dim in dims]


def _gen_mask_from_rects(dims):
    mask = np.zeros((_GEN_N, _GEN_N), dtype=bool)
    for group in _gen_rect_groups(dims):
        if len(group) == 2:
            (h1, w1), (h2, w2) = group
            pair = _gen_merged_pair(mask, h1, w1, h2, w2)
            if pair is None:
                return None
            for rect in pair:
                _gen_paint_rect(mask, rect)
        else:
            h, w = group[0]
            loc = _gen_rect_location(mask, h, w, padding=_GEN_RECT_PADDING)
            if loc is None:
                return None
            _gen_paint_rect(mask, (loc[0], loc[1], h, w))
    return mask


def _gen_mask_is_valid(mask):
    masked = int(mask.sum())
    if not _GEN_TARGET_MIN_CELLS <= masked < _GEN_TARGET_MAX_CELLS_EXCLUSIVE:
        return False

    comps = _gen_components_mask(mask, connectivity=4)
    if not _GEN_MIN_COMPONENTS <= len(comps) <= _GEN_MAX_COMPONENTS:
        return False
    if any(len(comp) < _GEN_COMPONENT_MIN_CELLS or len(comp) > _GEN_COMPONENT_MAX_CELLS for comp in comps):
        return False

    for orb in _GEN_ORBITS:
        covered = sum(bool(mask[r, c]) for r, c in orb)
        if covered and covered == len(orb):
            return False
    return True


def _gen_choose_rect_count(target):
    """Sample a rectangle count from the target-compatible 2..4 range."""
    feasible_counts = [
        k for k in range(_GEN_MIN_COMPONENTS, _GEN_MAX_COMPONENTS + 1)
        if _GEN_RECT_MIN_AREA * k <= target <= _GEN_RECT_MAX_AREA * k
    ]
    weights = np.array([k * k for k in feasible_counts], dtype=float)
    weights /= weights.sum()
    return int(np.random.choice(feasible_counts, p=weights))


def _gen_mask_for_target(target):
    """Build one candidate mask whose rectangle areas match the sampled target."""
    rect_count = _gen_choose_rect_count(target)
    return _gen_mask_from_rects(_gen_rect_sizes(rect_count, target))


def _gen_oracle_matches(inp, expected):
    """Reject masks that do not leave the solver with a unique D4 completion."""
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt():
    out = _gen_colored_output()
    mask = _gen_mask_for_target(_gen_target_cell_count())
    if mask is None or not _gen_mask_is_valid(mask):
        return None

    inp = out.copy()
    inp[mask] = _GEN_MASK
    if not _gen_oracle_matches(inp, out):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 47996f11 input")
